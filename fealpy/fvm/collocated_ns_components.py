"""Shared algorithm components for collocated Navier-Stokes FVM solvers.

The file is ordered by the pressure-correction algorithm flow: collocated
discretization setup, momentum equation, face-flux algebra, pressure equation,
and velocity-pressure coupling.  Each class below represents one mathematical
component shared by SIMPLE and PISO; the solver files keep the actual algorithm
loops.
"""

from dataclasses import dataclass
from typing import Optional

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.sparse import CSRTensor, spdiags
from fealpy.decorator import cartesian, variantmethod

from .convection_integrator import ConvectionMatrixAssembler
from .scalar_cross_diffusion_integrator import CrossDiffusionRHSAssembler
from .scalar_diffusion_integrator import (
    ScalarDiffusionIntegrator,
    ScalarDiffusionMatrixAssembler,
)
from .scalar_source_integrator import ScalarSourceIntegrator
from .gradient_reconstruct import GradientReconstruct
from .face_gradient import reconstruct_face_gradient
from .dirichlet_bc import DirichletBC
from .fvm_geometry import (
    FVMGeometry,
    face_interpolation_owner_weight,
    interpolate_cell_to_face,
    selected_boundary_faces,
)
from .solver_diagnostics import (
    equation_residual_converged,
    normalized_equation_residual,
    normalized_relaxed_equation_residual,
)


@dataclass(frozen=True)
class ComponentMomentumSystems:
    """Scalar component systems and their two momentum response diagonals."""

    matrices: list
    rhs: TensorLike
    relaxed_diagonal: TensorLike
    spatial_diagonal: TensorLike


@dataclass(frozen=True)
class MomentumPredictorResult:
    """Steady momentum predictor with algebraically distinct responses."""

    correction_denominator: TensorLike
    spatial_diagonal: TensorLike
    velocity: TensorLike


class CollocatedDiscretizationSetup:
    """Build the common collocated finite-volume discretization state.

    This is the setup stage before any SIMPLE/PISO iteration: construct the
    cell-centred P0 space, tensor velocity space, mesh geometry, gradient
    reconstructors, boundary objects, and velocity dof layout.  Velocity fields
    use two shapes deliberately: physical operators take cell-vector values
    with shape ``(NC, GD)``, while assembled linear systems use component-major
    dof vectors with shape ``(GD*NC,)``.
    """

    # Setup and algebraic shape conversion.

    def init_collocated_discretization(
        self,
        degree: int,
        dirichlet_velocity,
        *,
        pressure_gradient_method: str = "layered_lsq",
        velocity_gradient_method: str = "layered_lsq",
        gradient_layer_weights=(1.0, 0.25),
        gradient_boundary_weight: float = 1.0,
        diffusion_method: str = "over_relaxed",
        diffusion_nonorthogonal_eps: float = 0.05,
        dirichlet_velocity_threshold=None,
        dirichlet_pressure=None,
        dirichlet_pressure_threshold=None,
        with_dirichlet_velocity_bc: bool = False,
        geometry=None,
    ) -> None:
        if degree != 0:
            raise ValueError("space_degree must be 0 for cell-centred FVM.")
        self.p = degree
        self.fvm_geometry = (
            geometry if geometry is not None else FVMGeometry(self.mesh)
        )
        self.GD = self.fvm_geometry.cell_center.shape[1]
        self.NC = self.fvm_geometry.NC
        self.NF = self.fvm_geometry.NF
        self.cm = self.fvm_geometry.cell_measure
        self.cell_center = self.fvm_geometry.cell_center
        self.face_center = self.fvm_geometry.face_center
        self.space = ScaledMonomialSpace(self.mesh, degree)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(self.GD, -1))
        self.diffusion_method = diffusion_method
        self.diffusion_nonorthogonal_eps = float(diffusion_nonorthogonal_eps)
        if diffusion_method not in self.pressure_diffusion_decomposition:
            raise ValueError(f"unknown diffusion method: {diffusion_method!r}")
        self.pressure_diffusion_decomposition.set(diffusion_method)

        self.pressure_gradient = GradientReconstruct(
            self.mesh,
            method=pressure_gradient_method,
            boundary_value=dirichlet_pressure,
            boundary_type=(
                "dirichlet" if dirichlet_pressure is not None else None
            ),
            boundary_threshold=dirichlet_pressure_threshold,
            layer_weights=gradient_layer_weights,
            boundary_weight=gradient_boundary_weight,
            geometry=self.fvm_geometry,
        )
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method=velocity_gradient_method,
            boundary_value=dirichlet_velocity,
            boundary_type="dirichlet",
            boundary_threshold=dirichlet_velocity_threshold,
            layer_weights=gradient_layer_weights,
            boundary_weight=gradient_boundary_weight,
            geometry=self.fvm_geometry,
        )
        self.dirichlet_velocity = dirichlet_velocity
        self.dirichlet_velocity_threshold = dirichlet_velocity_threshold
        if with_dirichlet_velocity_bc:
            self.dirichlet_velocity_bc = DirichletBC(
                self.mesh,
                dirichlet_velocity,
                threshold=dirichlet_velocity_threshold,
                geometry=self.fvm_geometry,
                diffusion_method=diffusion_method,
                nonorthogonal_eps=diffusion_nonorthogonal_eps,
            )

        self.face_to_cell = self.fvm_geometry.face_to_cell
        self.last_momentum_nonorthogonal_iterations = 0
        self.last_pressure_nonorthogonal_iterations = 0

    def cell_vector_to_dofs(self, cell_vector):
        """Return component-major algebraic dofs from ``(NC, GD)`` cell vectors."""
        return bm.swapaxes(cell_vector, 0, 1).flatten()

    def dofs_to_cell_vector(self, dofs):
        """Return ``(NC, GD)`` cell vectors from component-major algebraic dofs."""
        return bm.stack(
            [
                dofs[component * self.NC : (component + 1) * self.NC]
                for component in range(self.GD)
            ],
            axis=-1,
        )

    def component_cell_diagonal(self, diagonal):
        """Repeat a cell diagonal once per velocity component."""
        return bm.concatenate([diagonal for _ in range(self.GD)], axis=0)

    def component_response(self, a_p):
        """Return the cell response ``V/a_P`` as an ``(NC, GD)`` vector field."""
        return bm.stack(
            [
                self.cm
                / a_p[component * self.NC : (component + 1) * self.NC]
                for component in range(self.GD)
            ],
            axis=-1,
        )


class CollocatedLinearSystemComponents:
    """Route assembled linear systems to the configured solver backend.

    This class deliberately has no SIMPLE/PISO semantics.  Equation components
    choose the solver name for momentum, pressure, or reference solves; this
    boundary only applies that choice to ``A x = b``.
    """

    def solve_linear_system(self, matrix, rhs, *, solver: Optional[str] = None):
        """Solve a linear system through the configured solver boundary.

        Algorithm layers decide which concrete solver name applies to each
        equation and pass it here explicitly.  The lower solver boundary only
        knows how to solve ``A x = b`` with that named backend.
        """
        return self.linear_solver.solve(matrix, rhs, solver=solver)


class CollocatedMomentumEquation:
    """Momentum-predictor equation components shared by SIMPLE and PISO.

    The class builds and solves the finite-volume momentum equation around the
    common transport operator: diffusion, convection, time term when present,
    pressure-gradient source, explicit source, Dirichlet/natural boundary
    contributions, algebraic relaxation, and explicit non-orthogonal diffusion
    correction.
    """

    def solve_momentum_system(
        self,
        matrix,
        rhs,
        *,
        strategy: Optional[str] = None,
    ):
        """Solve the assembled component-major momentum linear system.

        ``vector`` keeps the historical ``(GD*NC) x (GD*NC)`` system.  For the
        current scalar diffusion/convection momentum operator, this matrix is
        block diagonal by velocity component, so ``component`` solves each
        ``NC x NC`` component block separately and concatenates the result in
        the same component-major ordering.
        """
        strategy = (
            getattr(self.controls, "momentum_solve_strategy", "vector")
            if strategy is None
            else strategy
        )
        if strategy == "vector":
            return self.solve_linear_system(
                matrix,
                rhs,
                solver=getattr(self, "momentum_linear_solver", None),
            )
        raise ValueError("vector momentum systems must use strategy='vector'.")

    # Momentum equation components.

    def scalar_momentum_diffusion_matrix(self, diffusion_coef):
        """Assemble and cache the scalar momentum diffusion matrix."""
        cache = getattr(self, "_scalar_momentum_diffusion_matrix_cache", None)
        if cache is None:
            cache = {}
            self._scalar_momentum_diffusion_matrix_cache = cache

        try:
            key = (
                "scalar",
                float(diffusion_coef),
                self.diffusion_method,
                self.diffusion_nonorthogonal_eps,
            )
        except TypeError:
            key = (
                "object",
                id(diffusion_coef),
                self.diffusion_method,
                self.diffusion_nonorthogonal_eps,
            )

        if key not in cache:
            cache[key] = ScalarDiffusionMatrixAssembler(
                self.space,
                geometry=self.fvm_geometry,
                method=self.diffusion_method,
                nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
            ).assembly(diffusion_coef)
        return cache[key]

    def scalar_momentum_convection_matrix(
        self,
        convection_face_velocity,
        interpolation: str,
    ):
        """Assemble the scalar momentum convection matrix."""
        cache = getattr(self, "_scalar_momentum_convection_matrix_assembler_cache", None)
        if cache is None:
            cache = {}
            self._scalar_momentum_convection_matrix_assembler_cache = cache
        if interpolation not in cache:
            cache[interpolation] = ConvectionMatrixAssembler(
                self.space,
                interpolation=interpolation,
                geometry=self.fvm_geometry,
            )
        return cache[interpolation].assembly(convection_face_velocity)

    def momentum_diffusion_matrix(self, diffusion_coef):
        """Assemble and cache the shared momentum diffusion matrix."""
        cache = getattr(self, "_momentum_diffusion_matrix_cache", None)
        if cache is None:
            cache = {}
            self._momentum_diffusion_matrix_cache = cache

        try:
            key = (
                "scalar",
                float(diffusion_coef),
                self.diffusion_method,
                self.diffusion_nonorthogonal_eps,
            )
        except TypeError:
            key = (
                "object",
                id(diffusion_coef),
                self.diffusion_method,
                self.diffusion_nonorthogonal_eps,
            )

        if key not in cache:
            cache[key] = BilinearForm(self.velocity_space).add_integrator(
                ScalarDiffusionIntegrator(
                    q=self.p + 2,
                    coef=diffusion_coef,
                    geometry=self.fvm_geometry,
                    method=self.diffusion_method,
                    nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
                )
            ).assembly()
        return cache[key]

    def momentum_convection_matrix(self, convection_face_velocity, interpolation: str):
        """Assemble the momentum convection matrix for the active face velocity."""
        cache = getattr(self, "_momentum_convection_matrix_assembler_cache", None)
        if cache is None:
            cache = {}
            self._momentum_convection_matrix_assembler_cache = cache
        if interpolation not in cache:
            cache[interpolation] = ConvectionMatrixAssembler(
                self.velocity_space,
                interpolation=interpolation,
                geometry=self.fvm_geometry,
            )
        return cache[interpolation].assembly(convection_face_velocity)

    def scalar_momentum_time_matrix(self, density, time_step):
        """Return the scalar backward-Euler momentum time matrix."""
        time_step = float(time_step)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        density = bm.array(density, dtype=self.cm.dtype)
        if density.shape == ():
            cell_diagonal = density * self.cm / time_step
        else:
            if density.shape != self.cm.shape:
                raise ValueError("density must be scalar or cell-wise.")
            cell_diagonal = density * self.cm / time_step

        return CSRTensor(
            crow=bm.arange(self.NC + 1),
            col=bm.arange(self.NC),
            values=cell_diagonal,
            spshape=(self.NC, self.NC),
        )

    def momentum_time_matrix(self, density, time_step):
        """Return the implicit backward-Euler momentum time matrix.

        For P0 finite volumes this is the diagonal contribution
        ``rho * V_C / dt`` repeated for each velocity component.
        """
        time_step = float(time_step)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        density = bm.array(density, dtype=self.cm.dtype)
        if density.shape == ():
            cell_diagonal = density * self.cm / time_step
        else:
            if density.shape != self.cm.shape:
                raise ValueError("density must be scalar or cell-wise.")
            cell_diagonal = density * self.cm / time_step

        total_dofs = self.GD * self.NC
        values = self.component_cell_diagonal(cell_diagonal)
        return CSRTensor(
            crow=bm.arange(total_dofs + 1),
            col=bm.arange(total_dofs),
            values=values,
            spshape=(total_dofs, total_dofs),
        )

    def component_momentum_linear_systems(
        self,
        matrix,
        rhs,
        previous_velocity,
        *,
        diffusion_coef,
        convection_face_velocity=None,
        relaxation: float = 1.0,
        matrix_policy: str = "shared",
    ):
        """Build scalar component momentum systems and return component-major RHS.

        ``matrix_policy="shared"`` is the default P0 finite-volume route for
        isotropic incompressible momentum equations: all velocity components
        share the same implicit scalar operator and only the RHS differs.
        ``"per_component"`` keeps the extension point for future component-wise
        implicit operators or component-dependent boundary types.

        ``relaxed_diagonal`` is the diagonal after algebraic momentum
        under-relaxation.  ``spatial_diagonal`` is the diagonal of the
        unrelaxed spatial operator.  Pressure/cell corrections use the former,
        while the base Rhie-Chow stabilization must use the latter so that the
        converged discrete continuity equation is independent of the
        relaxation factor.
        """
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("momentum equation relaxation alpha must be in (0, 1].")
        if matrix_policy not in {"shared", "per_component"}:
            raise ValueError("matrix_policy must be 'shared' or 'per_component'.")

        if previous_velocity.shape != (self.NC, self.GD):
            raise ValueError("previous velocity must have shape (NC, GD).")
        previous_dofs = self.cell_vector_to_dofs(previous_velocity)
        boundary_rhs = self.dirichlet_velocity_bc.apply_diffusion_rhs(
            rhs,
            coef=diffusion_coef,
            threshold=self.dirichlet_velocity_threshold,
        )
        if convection_face_velocity is not None:
            boundary_rhs = self.dirichlet_velocity_bc.apply_convection(
                boundary_rhs,
                convection_face_velocity,
                threshold=self.dirichlet_velocity_threshold,
            )

        shared_matrix = None
        shared_relaxed_diagonal = None
        shared_relax_delta = None
        if matrix_policy == "shared":
            shared_matrix = self.dirichlet_velocity_bc.apply_diffusion_matrix(
                matrix,
                coef=diffusion_coef,
                threshold=self.dirichlet_velocity_threshold,
            )
            shared_base_diagonal = shared_matrix.diags().values
            if relaxation < 1.0:
                shared_relax_delta = (
                    (1.0 / relaxation - 1.0) * shared_base_diagonal
                )
                shared_matrix = shared_matrix + spdiags(
                    shared_relax_delta,
                    0,
                    shared_matrix.shape[0],
                    shared_matrix.shape[1],
                    index_dtype=shared_matrix.itype,
                )
            shared_relaxed_diagonal = shared_matrix.diags().values

        matrices = [] if matrix_policy == "per_component" else [shared_matrix] * self.GD
        rhs_parts = []
        diagonal_parts = []
        spatial_diagonal_parts = []
        for component in range(self.GD):
            start = component * self.NC
            stop = start + self.NC
            component_rhs = boundary_rhs[start:stop]
            if matrix_policy == "per_component":
                component_matrix = self.dirichlet_velocity_bc.apply_diffusion_matrix(
                    matrix,
                    coef=diffusion_coef,
                    threshold=self.dirichlet_velocity_threshold,
                )
            else:
                component_matrix = shared_matrix

            if matrix_policy == "shared":
                diagonal = shared_relaxed_diagonal
                spatial_diagonal = shared_base_diagonal
                if shared_relax_delta is not None:
                    component_rhs = (
                        component_rhs
                        + shared_relax_delta * previous_dofs[start:stop]
                    )
            else:
                diagonal = component_matrix.diags().values
                spatial_diagonal = diagonal
                if relaxation < 1.0:
                    delta = (1.0 / relaxation - 1.0) * diagonal
                    component_matrix = component_matrix + spdiags(
                        delta,
                        0,
                        component_matrix.shape[0],
                        component_matrix.shape[1],
                        index_dtype=component_matrix.itype,
                    )
                    component_rhs = component_rhs + delta * previous_dofs[start:stop]
                    diagonal = component_matrix.diags().values

            if matrix_policy == "per_component":
                matrices.append(component_matrix)

            rhs_parts.append(component_rhs)
            diagonal_parts.append(diagonal)
            spatial_diagonal_parts.append(spatial_diagonal)

        return ComponentMomentumSystems(
            matrices=matrices,
            rhs=bm.concatenate(rhs_parts, axis=0),
            relaxed_diagonal=bm.concatenate(diagonal_parts, axis=0),
            spatial_diagonal=bm.concatenate(spatial_diagonal_parts, axis=0),
        )

    def solve_component_momentum_systems(self, matrices, rhs):
        """Solve scalar component momentum systems and return component-major dofs."""
        solutions = []
        for component, matrix in enumerate(matrices):
            start = component * self.NC
            stop = start + self.NC
            solutions.append(
                self.solve_linear_system(
                    matrix,
                    rhs[start:stop],
                    solver=getattr(self, "momentum_linear_solver", None),
                )
            )
        return bm.concatenate(solutions, axis=0)

    def momentum_matrix_action(self, matrix, dofs):
        """Apply a vector or component-list momentum matrix to velocity dofs."""
        if isinstance(matrix, (list, tuple)):
            parts = []
            for component, component_matrix in enumerate(matrix):
                start = component * self.NC
                stop = start + self.NC
                parts.append(component_matrix @ dofs[start:stop])
            return bm.concatenate(parts, axis=0)
        return matrix @ dofs

    def momentum_pressure_response_denominator(self, matrix, diagonal, *, scheme: str):
        """Return the cell denominator used by pressure-velocity correction.

        This function constructs the algebraic correction response.  SIMPLE
        uses the relaxed momentum diagonal ``a_P``.  SIMPLEC uses the row sum
        of the same relaxed momentum matrix, which corresponds to
        ``a_P - sum_N a_N`` under the finite-volume sign convention
        ``A_PP u_P - sum_N a_N u_N = rhs``.

        The unrelaxed spatial diagonal used by the base Rhie-Chow flux is kept
        separate by the steady momentum predictor.
        """
        if scheme == "simple":
            return diagonal
        if scheme != "simplec":
            raise ValueError("pressure response scheme must be 'simple' or 'simplec'.")

        denominator = self.momentum_matrix_action(matrix, bm.ones_like(diagonal))
        if bool(bm.to_numpy(bm.any(denominator <= 0.0))):
            raise ValueError("SIMPLEC pressure response denominator must be positive.")
        return denominator

    def solve_steady_momentum_predictor(
        self,
        pressure,
        face_velocity,
        previous_velocity,
        *,
        pressure_gradient=None,
        nonorthogonal_tol=None,
    ):
        """Return the named steady predictor and its two response diagonals."""
        if self.controls.momentum_solve_strategy == "component":
            return self.solve_component_steady_momentum_predictor(
                pressure,
                face_velocity,
                previous_velocity,
                pressure_gradient=pressure_gradient,
                nonorthogonal_tol=nonorthogonal_tol,
            )

        if nonorthogonal_tol is None:
            nonorthogonal_tol = self.controls.momentum_nonorthogonal_tol

        convection_face_velocity = self.convection_coef * face_velocity
        matrix = self.momentum_diffusion_matrix(self.diffusion_coef)
        if self.convection_coef != 0.0:
            matrix = matrix + self.momentum_convection_matrix(
                convection_face_velocity,
                self.controls.face_interpolation("momentum_face_interpolation"),
            )
            matrix = self.add_natural_velocity_convection_diagonal(
                matrix,
                self.convection_coef * face_velocity,
                self.natural_velocity_threshold,
            )
        rhs = self.steady_momentum_source_vector()
        rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
        matrix, rhs = self.dirichlet_velocity_bc.apply_diffusion(
            matrix,
            rhs,
            coef=self.diffusion_coef,
            threshold=self.dirichlet_velocity_threshold,
        )
        if self.convection_coef != 0.0:
            rhs = self.dirichlet_velocity_bc.apply_convection(
                rhs,
                convection_face_velocity,
                threshold=self.dirichlet_velocity_threshold,
            )
        rhs = rhs - self.pressure_gradient_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        matrix, rhs, diagonal, spatial_diagonal = self.relax_momentum_equation(
            matrix,
            rhs,
            self.cell_vector_to_dofs(previous_velocity),
            self.controls.momentum_equation_relaxation,
        )
        correction_denominator = self.momentum_pressure_response_denominator(
            matrix,
            diagonal,
            scheme=self.controls.pressure_response_scheme,
        )
        velocity = self.solve_momentum_system(matrix, rhs)
        velocity = self.correct_momentum_nonorthogonal_diffusion(
            matrix,
            rhs,
            velocity,
            relaxation_diagonal=diagonal - spatial_diagonal,
            previous_velocity=self.cell_vector_to_dofs(previous_velocity),
            max_iter=self.controls.momentum_nonorthogonal_max_iter,
            tol=nonorthogonal_tol,
            atol=self.controls.momentum_nonorthogonal_atol,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )
        return MomentumPredictorResult(
            correction_denominator=correction_denominator,
            spatial_diagonal=spatial_diagonal,
            velocity=self.dofs_to_cell_vector(velocity),
        )

    def solve_component_steady_momentum_predictor(
        self,
        pressure,
        face_velocity,
        previous_velocity,
        *,
        pressure_gradient=None,
        nonorthogonal_tol=None,
    ):
        """Return the named component predictor and its two responses."""
        if nonorthogonal_tol is None:
            nonorthogonal_tol = self.controls.momentum_nonorthogonal_tol
        convection_face_velocity = self.convection_coef * face_velocity
        matrix = self.scalar_momentum_diffusion_matrix(self.diffusion_coef)
        boundary_convection_velocity = None
        if self.convection_coef != 0.0:
            matrix = matrix + self.scalar_momentum_convection_matrix(
                convection_face_velocity,
                self.controls.face_interpolation("momentum_face_interpolation"),
            )
            matrix = self.add_natural_velocity_convection_diagonal(
                matrix,
                convection_face_velocity,
                self.natural_velocity_threshold,
            )
            boundary_convection_velocity = convection_face_velocity

        rhs = self.steady_momentum_source_vector()
        rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
        rhs = rhs - self.pressure_gradient_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        systems = self.component_momentum_linear_systems(
            matrix,
            rhs,
            previous_velocity,
            diffusion_coef=self.diffusion_coef,
            convection_face_velocity=boundary_convection_velocity,
            relaxation=self.controls.momentum_equation_relaxation,
            matrix_policy=self.controls.momentum_component_matrix_policy,
        )
        correction_denominator = self.momentum_pressure_response_denominator(
            systems.matrices,
            systems.relaxed_diagonal,
            scheme=self.controls.pressure_response_scheme,
        )
        velocity = self.solve_component_momentum_systems(
            systems.matrices,
            systems.rhs,
        )
        velocity = self.correct_component_momentum_nonorthogonal_diffusion(
            systems.matrices,
            systems.rhs,
            velocity,
            relaxation_diagonal=(
                systems.relaxed_diagonal - systems.spatial_diagonal
            ),
            previous_velocity=self.cell_vector_to_dofs(previous_velocity),
            max_iter=self.controls.momentum_nonorthogonal_max_iter,
            tol=nonorthogonal_tol,
            atol=self.controls.momentum_nonorthogonal_atol,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )
        return MomentumPredictorResult(
            correction_denominator=correction_denominator,
            spatial_diagonal=systems.spatial_diagonal,
            velocity=self.dofs_to_cell_vector(velocity),
        )

    def steady_momentum_balance(
        self,
        pressure,
        velocity,
        face_velocity,
        *,
        pressure_gradient=None,
    ):
        """Return the unrelaxed steady momentum ``lhs`` and effective ``rhs``.

        This diagnostic reassembles the selected spatial discretization at the
        supplied final fields.  Algebraic momentum relaxation is deliberately
        excluded, while enabled non-orthogonal diffusion remains part of the
        effective RHS.
        """
        if velocity.shape != (self.NC, self.GD):
            raise ValueError("cell velocity must have shape (NC, GD).")
        velocity_vector = velocity
        velocity_dofs = self.cell_vector_to_dofs(velocity_vector)
        convection_face_velocity = self.convection_coef * face_velocity
        pressure_gradient = (
            self.pressure_gradient.cell_gradient(pressure)
            if pressure_gradient is None
            else pressure_gradient
        )

        if self.controls.momentum_solve_strategy == "component":
            matrix = self.scalar_momentum_diffusion_matrix(self.diffusion_coef)
            boundary_convection_velocity = None
            if self.convection_coef != 0.0:
                matrix = matrix + self.scalar_momentum_convection_matrix(
                    convection_face_velocity,
                    self.controls.face_interpolation("momentum_face_interpolation"),
                )
                matrix = self.add_natural_velocity_convection_diagonal(
                    matrix,
                    convection_face_velocity,
                    self.natural_velocity_threshold,
                )
                boundary_convection_velocity = convection_face_velocity

            rhs = self.steady_momentum_source_vector()
            rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
            rhs = rhs - self.pressure_gradient_source(
                pressure,
                pressure_gradient=pressure_gradient,
            )
            systems = self.component_momentum_linear_systems(
                matrix,
                rhs,
                velocity_vector,
                diffusion_coef=self.diffusion_coef,
                convection_face_velocity=boundary_convection_velocity,
                relaxation=1.0,
                matrix_policy=self.controls.momentum_component_matrix_policy,
            )
            rhs = systems.rhs
            lhs = self.momentum_matrix_action(systems.matrices, velocity_dofs)
        else:
            matrix = self.momentum_diffusion_matrix(self.diffusion_coef)
            if self.convection_coef != 0.0:
                matrix = matrix + self.momentum_convection_matrix(
                    convection_face_velocity,
                    self.controls.face_interpolation("momentum_face_interpolation"),
                )
                matrix = self.add_natural_velocity_convection_diagonal(
                    matrix,
                    convection_face_velocity,
                    self.natural_velocity_threshold,
                )
            rhs = self.steady_momentum_source_vector()
            rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
            matrix, rhs = self.dirichlet_velocity_bc.apply_diffusion(
                matrix,
                rhs,
                coef=self.diffusion_coef,
                threshold=self.dirichlet_velocity_threshold,
            )
            if self.convection_coef != 0.0:
                rhs = self.dirichlet_velocity_bc.apply_convection(
                    rhs,
                    convection_face_velocity,
                    threshold=self.dirichlet_velocity_threshold,
                )
            rhs = rhs - self.pressure_gradient_source(
                pressure,
                pressure_gradient=pressure_gradient,
            )
            lhs = matrix @ velocity_dofs

        if self.controls.momentum_nonorthogonal_max_iter > 0:
            rhs = rhs + self.momentum_nonorthogonal_rhs(velocity_vector)
        return {"lhs": lhs, "rhs": rhs, "residual": lhs - rhs}

    def steady_momentum_terms(
        self,
        pressure,
        velocity,
        face_velocity,
        *,
        pressure_gradient=None,
    ):
        """Return named terms of the unrelaxed production momentum residual.

        The returned component-major vectors satisfy

        ``residual = diffusion + convection + pressure - source``.

        Diffusion includes its Dirichlet and Neumann boundary contributions
        and the enabled non-orthogonal correction.  Convection is recovered
        from the complete production residual after the other independently
        assembled terms are removed, so its boundary closure is exactly the
        one used by the production momentum equation.
        """
        if velocity.shape != (self.NC, self.GD):
            raise ValueError("cell velocity must have shape (NC, GD).")
        velocity_vector = velocity
        velocity_dofs = self.cell_vector_to_dofs(velocity_vector)
        pressure_gradient = (
            self.pressure_gradient.cell_gradient(pressure)
            if pressure_gradient is None
            else pressure_gradient
        )
        balance = self.steady_momentum_balance(
            pressure,
            velocity_vector,
            face_velocity,
            pressure_gradient=pressure_gradient,
        )

        zero_rhs = bm.zeros(self.GD * self.NC, dtype=velocity_dofs.dtype)
        neumann_source = self.neumann_velocity_diffusion_source(
            self.diffusion_coef
        )
        if self.controls.momentum_solve_strategy == "component":
            diffusion_matrix = self.scalar_momentum_diffusion_matrix(
                self.diffusion_coef
            )
            diffusion_systems = self.component_momentum_linear_systems(
                diffusion_matrix,
                zero_rhs,
                velocity_vector,
                diffusion_coef=self.diffusion_coef,
                relaxation=1.0,
                matrix_policy=self.controls.momentum_component_matrix_policy,
            )
            diffusion = (
                self.momentum_matrix_action(
                    diffusion_systems.matrices,
                    velocity_dofs,
                )
                - diffusion_systems.rhs
                - neumann_source
            )
        else:
            diffusion_matrix = self.momentum_diffusion_matrix(self.diffusion_coef)
            diffusion_matrix, diffusion_rhs = self.dirichlet_velocity_bc.apply_diffusion(
                diffusion_matrix,
                zero_rhs,
                coef=self.diffusion_coef,
                threshold=self.dirichlet_velocity_threshold,
            )
            diffusion = (
                diffusion_matrix @ velocity_dofs
                - diffusion_rhs
                - neumann_source
            )

        if self.controls.momentum_nonorthogonal_max_iter > 0:
            diffusion = diffusion - self.momentum_nonorthogonal_rhs(
                velocity_vector
            )

        pressure_term = self.pressure_gradient_source(
            pressure,
            pressure_gradient=pressure_gradient,
        )
        source_term = self.steady_momentum_source_vector()
        convection = (
            balance["residual"] - diffusion - pressure_term + source_term
        )
        return {
            "diffusion": diffusion,
            "convection": convection,
            "pressure": pressure_term,
            "source": source_term,
            "lhs": balance["lhs"],
            "rhs": balance["rhs"],
            "residual": balance["residual"],
        }

    def solve_transient_momentum_predictor(
        self,
        previous_velocity,
        previous_face_velocity,
        pressure,
        time,
    ):
        """Solve the transient PISO momentum predictor equation."""
        if self.controls.momentum_solve_strategy == "component":
            return self.solve_component_transient_momentum_predictor(
                previous_velocity,
                previous_face_velocity,
                pressure,
                time,
            )

        controls = self.controls
        matrix = self.momentum_diffusion_matrix(self.diffusion_coef)
        matrix = matrix + self.momentum_convection_matrix(
            self.convection_coef * previous_face_velocity,
            controls.face_interpolation_method,
        )
        matrix = self.add_natural_velocity_convection_diagonal(
            matrix,
            self.convection_coef * previous_face_velocity,
            self.natural_velocity_threshold,
        )

        @cartesian
        def src(points):
            return self.source(points, time)

        matrix = matrix + self.momentum_time_matrix(
            self.convection_coef,
            controls.tau,
        )
        rhs = self.momentum_source_vector(src)
        rhs = rhs + self.momentum_time_source(
            previous_velocity,
            self.convection_coef,
            controls.tau,
        )
        rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
        rhs = rhs - self.pressure_gradient_source(pressure)
        matrix, rhs = self.dirichlet_velocity_bc.apply_diffusion(
            matrix,
            rhs,
            coef=self.diffusion_coef,
            threshold=self.dirichlet_velocity_threshold,
        )
        rhs = self.dirichlet_velocity_bc.apply_convection(
            rhs,
            self.convection_coef * previous_face_velocity,
            threshold=self.dirichlet_velocity_threshold,
        )
        diagonal = matrix.diags().values
        solution = self.solve_momentum_system(matrix, rhs)
        solution = self.correct_momentum_nonorthogonal_diffusion(
            matrix,
            rhs,
            solution,
            max_iter=controls.momentum_nonorthogonal_max_iter,
            tol=controls.momentum_nonorthogonal_tol,
            atol=controls.momentum_nonorthogonal_atol,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )
        return self.dofs_to_cell_vector(solution), diagonal, matrix

    def solve_component_transient_momentum_predictor(
        self,
        previous_velocity,
        previous_face_velocity,
        pressure,
        time,
    ):
        """Solve the transient PISO momentum predictor as scalar component systems."""
        controls = self.controls
        matrix = self.scalar_momentum_diffusion_matrix(self.diffusion_coef)
        convection_face_velocity = self.convection_coef * previous_face_velocity
        matrix = matrix + self.scalar_momentum_convection_matrix(
            convection_face_velocity,
            controls.face_interpolation_method,
        )
        matrix = self.add_natural_velocity_convection_diagonal(
            matrix,
            convection_face_velocity,
            self.natural_velocity_threshold,
        )
        matrix = matrix + self.scalar_momentum_time_matrix(
            self.convection_coef,
            controls.tau,
        )

        @cartesian
        def src(points):
            return self.source(points, time)

        rhs = self.momentum_source_vector(src)
        rhs = rhs + self.momentum_time_source(
            previous_velocity,
            self.convection_coef,
            controls.tau,
        )
        rhs = rhs + self.neumann_velocity_diffusion_source(self.diffusion_coef)
        rhs = rhs - self.pressure_gradient_source(pressure)
        systems = self.component_momentum_linear_systems(
            matrix,
            rhs,
            previous_velocity,
            diffusion_coef=self.diffusion_coef,
            convection_face_velocity=convection_face_velocity,
            matrix_policy=controls.momentum_component_matrix_policy,
        )

        solution = self.solve_component_momentum_systems(
            systems.matrices,
            systems.rhs,
        )
        solution = self.correct_component_momentum_nonorthogonal_diffusion(
            systems.matrices,
            systems.rhs,
            solution,
            max_iter=controls.momentum_nonorthogonal_max_iter,
            tol=controls.momentum_nonorthogonal_tol,
            atol=controls.momentum_nonorthogonal_atol,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )
        return (
            self.dofs_to_cell_vector(solution),
            systems.relaxed_diagonal,
            systems.matrices,
        )

    def relax_momentum_equation(self, matrix, rhs, previous_velocity, alpha):
        """Apply matrix-level under-relaxation to a momentum equation.

        For an assembled system ``A U = b``, the relaxed system is
        ``(A + diag(delta)) U = b + diag(delta) U_old`` with
        ``delta = (1 / alpha - 1) diag(A)``.  Both the relaxed correction
        diagonal and the original spatial diagonal are returned because they
        have different roles in the SIMPLE/Rhie-Chow coupling.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("momentum equation relaxation alpha must be in (0, 1].")

        diagonal = matrix.diags().values
        if alpha == 1.0:
            return matrix, rhs, diagonal, diagonal

        delta = (1.0 / alpha - 1.0) * diagonal
        relaxed_matrix = matrix + spdiags(
            delta,
            0,
            matrix.shape[0],
            matrix.shape[1],
            index_dtype=matrix.itype,
        )
        relaxed_rhs = rhs + delta * previous_velocity
        return relaxed_matrix, relaxed_rhs, relaxed_matrix.diags().values, diagonal

    def correct_momentum_nonorthogonal_diffusion(
        self,
        matrix,
        rhs,
        velocity,
        *,
        relaxation_diagonal=None,
        previous_velocity=None,
        max_iter: int,
        tol: float,
        atol: float,
        iteration_attr: str,
    ):
        """Picard-correct a vector momentum system for non-orthogonal diffusion."""
        if max_iter == 0:
            setattr(self, iteration_attr, 0)
            return velocity

        corrected_velocity = velocity
        norm_weights = 1.0 / bm.tile(self.cm, (self.GD,))
        setattr(self, iteration_attr, 0)
        self.last_momentum_nonorthogonal_residual = None
        self.last_momentum_nonorthogonal_relative_update = 0.0
        for correction in range(max_iter + 1):
            cell_velocity = self.dofs_to_cell_vector(corrected_velocity)
            cross = self.momentum_nonorthogonal_rhs(cell_velocity)
            lhs = matrix @ corrected_velocity
            corrected_rhs = rhs + cross
            if relaxation_diagonal is None:
                metrics = normalized_equation_residual(
                    lhs,
                    corrected_rhs,
                    norm_weights=norm_weights,
                )
            else:
                metrics = normalized_relaxed_equation_residual(
                    lhs,
                    corrected_rhs,
                    corrected_velocity,
                    previous_velocity,
                    relaxation_diagonal,
                    norm_weights=norm_weights,
                )
            self.last_momentum_nonorthogonal_residual = metrics
            if equation_residual_converged(metrics, rtol=tol, atol=atol):
                return corrected_velocity
            if correction == max_iter:
                break

            next_velocity = self.solve_momentum_system(matrix, rhs + cross)
            update = float(
                bm.to_numpy(bm.linalg.norm(next_velocity - corrected_velocity))
            )
            scale = float(bm.to_numpy(bm.linalg.norm(next_velocity)))
            self.last_momentum_nonorthogonal_relative_update = (
                update / max(scale, 1.0e-30)
            )
            corrected_velocity = next_velocity
            setattr(self, iteration_attr, correction + 1)

        raise RuntimeError(
            "momentum non-orthogonal correction did not converge before "
            "momentum_nonorthogonal_max_iter"
        )

    def correct_component_momentum_nonorthogonal_diffusion(
        self,
        matrices,
        rhs,
        velocity,
        *,
        relaxation_diagonal=None,
        previous_velocity=None,
        max_iter: int,
        tol: float,
        atol: float,
        iteration_attr: str,
    ):
        """Picard-correct component momentum systems for non-orthogonal diffusion."""
        if max_iter == 0:
            setattr(self, iteration_attr, 0)
            return velocity

        corrected_velocity = velocity
        norm_weights = 1.0 / bm.tile(self.cm, (self.GD,))
        setattr(self, iteration_attr, 0)
        self.last_momentum_nonorthogonal_residual = None
        self.last_momentum_nonorthogonal_relative_update = 0.0
        for correction in range(max_iter + 1):
            cell_velocity = self.dofs_to_cell_vector(corrected_velocity)
            cross = self.momentum_nonorthogonal_rhs(cell_velocity)
            lhs_parts = []
            for component, matrix in enumerate(matrices):
                start = component * self.NC
                stop = start + self.NC
                lhs_parts.append(matrix @ corrected_velocity[start:stop])
            lhs = bm.concatenate(lhs_parts, axis=0)
            corrected_rhs = rhs + cross
            if relaxation_diagonal is None:
                metrics = normalized_equation_residual(
                    lhs,
                    corrected_rhs,
                    norm_weights=norm_weights,
                )
            else:
                metrics = normalized_relaxed_equation_residual(
                    lhs,
                    corrected_rhs,
                    corrected_velocity,
                    previous_velocity,
                    relaxation_diagonal,
                    norm_weights=norm_weights,
                )
            self.last_momentum_nonorthogonal_residual = metrics
            if equation_residual_converged(metrics, rtol=tol, atol=atol):
                return corrected_velocity
            if correction == max_iter:
                break

            next_velocity = self.solve_component_momentum_systems(
                matrices, rhs + cross
            )
            update = float(
                bm.to_numpy(bm.linalg.norm(next_velocity - corrected_velocity))
            )
            scale = float(bm.to_numpy(bm.linalg.norm(next_velocity)))
            self.last_momentum_nonorthogonal_relative_update = (
                update / max(scale, 1.0e-30)
            )
            corrected_velocity = next_velocity
            setattr(self, iteration_attr, correction + 1)

        raise RuntimeError(
            "momentum non-orthogonal correction did not converge before "
            "momentum_nonorthogonal_max_iter"
        )

    def momentum_time_source(self, previous_velocity, density, time_step):
        """Return the old-time RHS contribution ``rho * V_C / dt * U^n``."""
        time_step = float(time_step)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        density = bm.array(density, dtype=self.cm.dtype)
        if density.shape == ():
            cell_diagonal = density * self.cm / time_step
        else:
            if density.shape != self.cm.shape:
                raise ValueError("density must be scalar or cell-wise.")
            cell_diagonal = density * self.cm / time_step

        return self.cell_vector_to_dofs(previous_velocity * cell_diagonal[:, None])

    def add_natural_velocity_convection_diagonal(
        self,
        matrix,
        convection_face_velocity,
        threshold,
    ):
        """Add natural-boundary convection fluxes to the owner diagonal."""
        if threshold is None:
            return matrix

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return matrix

        selected = boundary_faces[flag]
        flux = bm.einsum(
            "ij,ij->i",
            convection_face_velocity[selected],
            self.fvm_geometry.S_f[selected],
        )
        owner = self.fvm_geometry.owner[selected]
        diagonal = bm.zeros(self.NC, dtype=flux.dtype)
        diagonal = bm.index_add(diagonal, owner, flux, axis=0)
        components = matrix.shape[0] // self.NC
        diagonal = bm.concatenate([diagonal for _ in range(components)], axis=0)
        return matrix + spdiags(
            diagonal,
            0,
            matrix.shape[0],
            matrix.shape[1],
            index_dtype=matrix.itype,
        )

    def momentum_source_vector(self, source):
        """Assemble the shared cell-integrated momentum source vector."""
        return LinearForm(self.velocity_space).add_integrator(
            ScalarSourceIntegrator(
                source,
                q=self.p + 2,
                geometry=self.fvm_geometry,
            )
        ).assembly()

    def neumann_velocity_boundary_data(self):
        """Return velocity Neumann boundary faces and normal derivatives."""
        threshold = getattr(self, "neumann_velocity_threshold", None)
        value = getattr(self, "neumann_velocity_data", None)
        if threshold is None or value is None:
            return None, None

        boundary_faces = selected_boundary_faces(self.fvm_geometry, threshold)
        if boundary_faces is None or boundary_faces.shape[0] == 0:
            return boundary_faces, bm.zeros((0, self.GD), dtype=self.cm.dtype)
        points = self.fvm_geometry.face_center[boundary_faces]
        return boundary_faces, bm.array(value(points), dtype=self.cm.dtype)

    def neumann_velocity_diffusion_source(self, diffusion_coef):
        r"""Return cell-integrated velocity Neumann diffusion RHS.

        ``neumann_velocity_data(points)`` is interpreted as the outward normal
        derivative density ``dU/dn`` on selected boundary faces.  The finite
        volume contribution is ``mu_f * dU/dn * |S_f|`` scattered to owner
        cells and returned in component-major momentum-vector layout.
        """
        boundary_faces, sn_grad = self.neumann_velocity_boundary_data()
        if boundary_faces is None or boundary_faces.shape[0] == 0:
            return bm.zeros((self.GD * self.NC,), dtype=self.cm.dtype)

        if isinstance(diffusion_coef, (int, float)):
            coef = diffusion_coef
        else:
            coef = bm.array(diffusion_coef)
            if coef.shape != ():
                coef = coef[boundary_faces]
        contribution = coef * self.fvm_geometry.mag_S_f[boundary_faces]
        cell_source = bm.zeros((self.NC, self.GD), dtype=sn_grad.dtype)
        cell_source = bm.index_add(
            cell_source,
            self.fvm_geometry.owner[boundary_faces],
            contribution[:, None] * sn_grad,
            axis=0,
        )
        return self.cell_vector_to_dofs(cell_source)

    def boundary_corrected_velocity_face_gradient(self, velocity, interpolation_method: str):
        """Return face velocity gradients with Dirichlet/natural/Neumann patches."""
        cell_gradient = self.velocity_gradient.cell_gradient(velocity)
        kwargs = {}

        dirichlet_faces, dirichlet_values = self.boundary_conditions.boundary_face_velocity(
            "velocity",
            mesh=self.mesh,
        )
        if dirichlet_faces is not None and dirichlet_faces.shape[0] > 0:
            kwargs["dirichlet_faces"] = dirichlet_faces
            kwargs["dirichlet_values"] = dirichlet_values

        neumann_faces = []
        neumann_values = []
        natural_faces = selected_boundary_faces(
            self.fvm_geometry,
            getattr(self, "natural_velocity_threshold", None),
        )
        if natural_faces is not None and natural_faces.shape[0] > 0:
            neumann_faces.append(natural_faces)
            neumann_values.append(
                bm.zeros((natural_faces.shape[0], velocity.shape[1]), dtype=velocity.dtype)
            )

        neumann_velocity_faces, neumann_velocity_values = self.neumann_velocity_boundary_data()
        if neumann_velocity_faces is not None and neumann_velocity_faces.shape[0] > 0:
            neumann_faces.append(neumann_velocity_faces)
            neumann_values.append(bm.array(neumann_velocity_values, dtype=velocity.dtype))

        if neumann_faces:
            kwargs["neumann_faces"] = bm.concatenate(neumann_faces, axis=0)
            kwargs["neumann_sn_grad"] = bm.concatenate(neumann_values, axis=0)

        return reconstruct_face_gradient(
            self.mesh,
            cell_gradient,
            geometry=self.fvm_geometry,
            cell_values=velocity,
            interpolation_method=interpolation_method,
            **kwargs,
        )

    def momentum_nonorthogonal_rhs(self, velocity: TensorLike) -> TensorLike:
        """Assemble the explicit momentum RHS from non-orthogonal diffusion.

        ``nonorthogonal`` is the algorithm-level correction.  The underlying
        operator is the cross-diffusion face flux assembled by
        ``CrossDiffusionRHSAssembler``.
        """
        grad_f = self.boundary_corrected_velocity_face_gradient(
            velocity,
            interpolation_method="average",
        )
        assembler = getattr(self, "_cross_diffusion_rhs_assembler", None)
        if assembler is None:
            assembler = CrossDiffusionRHSAssembler(
                self.velocity_space,
                geometry=self.fvm_geometry,
                method=self.diffusion_method,
                nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
            )
            self._cross_diffusion_rhs_assembler = assembler
        return assembler.assembly(
            grad_f=grad_f,
            coef=self.diffusion_coef,
            boundary_policy="all",
        )


class CollocatedFaceFluxAlgebra:
    """Face-value interpolation and conservative flux algebra.

    These methods implement the geometric operations used by pressure
    correction: interpolate cell fields to faces, compute
    ``phi_f = dot(u_f, S_f)``, and scatter signed face fluxes back to cell
    imbalance residuals.
    """

    def face_interpolate_cell_scalar(self, cell_values, method: str = "linear"):
        """Linearly interpolate a cell scalar to faces using face geometry."""
        return interpolate_cell_to_face(
            cell_values,
            geometry=self.fvm_geometry,
            method=method,
        )

    def face_interpolate_cell_vector(self, cell_vectors, method: str = "linear"):
        """Linearly interpolate a cell vector to faces using face geometry."""
        return interpolate_cell_to_face(
            cell_vectors,
            geometry=self.fvm_geometry,
            method=method,
        )

    def spatial_face_velocity(
        self,
        cell_velocity,
        *,
        boundary_faces=None,
        boundary_face_average=None,
    ):
        """Construct the configured spatial face velocity before pressure coupling."""
        if cell_velocity.shape != (self.NC, self.GD):
            raise ValueError("cell velocity must have shape (NC, GD).")
        if self.controls.rhie_chow_velocity_scheme == "interpolated":
            face_velocity = self.face_interpolate_cell_vector(
                cell_velocity,
                method=self.controls.face_interpolation_method,
            )
        else:
            face_velocity = self.second_order_reconstructed_face_velocity(
                cell_velocity
            )

        if self.controls.face_flux_correction_scheme == "none":
            return face_velocity

        flux_defect = self.face_flux_reconstruct.correction(
            cell_velocity,
            face_velocity,
            boundary_face_average=boundary_face_average,
            boundary_faces=boundary_faces,
        )
        return self.enforce_face_flux(
            face_velocity,
            self.compute_face_flux(face_velocity) + flux_defect,
        )

    def second_order_reconstructed_face_velocity(self, cell_velocity):
        """Blend owner/neighbour linear reconstructions at face centres."""
        if cell_velocity.shape != (self.NC, self.GD):
            raise ValueError("cell velocity must have shape (NC, GD).")
        gradient = self.velocity_gradient.cell_gradient(cell_velocity)
        owner = self.fvm_geometry.owner
        neighbour = self.fvm_geometry.neighbour
        owner_delta = (
            self.fvm_geometry.face_center - self.fvm_geometry.cell_center[owner]
        )
        neighbour_delta = (
            self.fvm_geometry.face_center
            - self.fvm_geometry.cell_center[neighbour]
        )
        owner_value = cell_velocity[owner] + bm.einsum(
            "ncd,nd->nc", gradient[owner], owner_delta
        )
        neighbour_value = cell_velocity[neighbour] + bm.einsum(
            "ncd,nd->nc", gradient[neighbour], neighbour_delta
        )
        owner_weight = face_interpolation_owner_weight(
            self.mesh,
            method="linear",
            geometry=self.fvm_geometry,
        )
        return (
            owner_weight[:, None] * owner_value
            + (1.0 - owner_weight[:, None]) * neighbour_value
        )

    def compute_face_flux(self, face_velocity):
        """Return the signed surface flux ``phi_f = u_f dot S_f``."""
        return bm.einsum("ij,ij->i", face_velocity, self.fvm_geometry.S_f)

    def divergence_from_flux(self, face_flux):
        """Scatter signed face fluxes to the cell flux imbalance."""
        return self.fvm_geometry.scatter_face_flux_to_cells(face_flux)

    def enforce_face_flux(self, face_velocity, target_flux):
        """Adjust only the normal component of a vector face velocity."""
        Sf = self.fvm_geometry.S_f
        current_flux = self.compute_face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        return face_velocity + ((target_flux - current_flux) / Sf_dot_Sf)[:, None] * Sf


class PressureGaugeMatrixAssembler:
    r"""Assemble the pure-Neumann pressure matrix with a gauge constraint.

    Pressure-correction methods need the scalar pressure Laplacian built from
    the face response coefficient ``r_f``.  With only Neumann-type pressure
    boundaries, this matrix has the constant-pressure nullspace.  This
    assembler appends a volume-weighted Lagrange-multiplier row and column to
    solve the augmented system ``[[A_p, V], [V^T, 0]] [p, c]^T = [b, 0]^T``
    and select a unique pressure representative.  It is the gauge fallback and
    reference route; the nullspace-aware route uses the unaugmented
    ``pressure_diffusion_matrix`` and handles the constant mode in the solver.
    The sparsity pattern is fixed by the mesh, so each call only updates the
    values induced by the current ``r_f``.
    """

    def __init__(
        self,
        space,
        *,
        geometry: Optional[FVMGeometry] = None,
        cell_measure: Optional[TensorLike] = None,
        diffusion_method: str = "over_relaxed",
        diffusion_nonorthogonal_eps: float = 0.05,
    ) -> None:
        self.space = space
        self.mesh = getattr(space, "mesh", None)
        self.geometry = geometry if geometry is not None else FVMGeometry(self.mesh)
        self.NC = self.geometry.NC
        self.sparse_shape = (self.NC + 1, self.NC + 1)
        self.cell_measure = (
            self.geometry.cell_measure if cell_measure is None else cell_measure
        )
        self.pressure_diffusion = ScalarDiffusionMatrixAssembler(
            space,
            geometry=self.geometry,
            method=diffusion_method,
            nonorthogonal_eps=diffusion_nonorthogonal_eps,
        )

        base_counts = (
            self.pressure_diffusion.crow[1:] - self.pressure_diffusion.crow[:-1]
        )
        base_rows = bm.repeat(
            bm.arange(
                self.NC,
                dtype=self.pressure_diffusion.col.dtype,
                device=bm.get_device(self.pressure_diffusion.col),
            ),
            base_counts,
        )
        cell = bm.arange(self.NC, dtype=base_rows.dtype, device=bm.get_device(base_rows))
        gauge_col = bm.full(
            (self.NC,),
            self.NC,
            dtype=base_rows.dtype,
            device=bm.get_device(base_rows),
        )
        rows = bm.concatenate([base_rows, cell, gauge_col])
        cols = bm.concatenate([self.pressure_diffusion.col, gauge_col, cell])
        self.gauge_values = bm.concatenate([self.cell_measure, self.cell_measure])

        nrow, ncol = self.sparse_shape
        flat = bm.astype(rows, bm.int64) * ncol + bm.astype(cols, bm.int64)
        order = bm.argsort(flat)
        flat_sorted = flat[order]
        group_start = bm.ones(
            (flat.shape[0],),
            dtype=bm.bool,
            device=bm.get_device(flat),
        )
        group_start = bm.set_at(
            group_start,
            slice(1, None),
            flat_sorted[1:] != flat_sorted[:-1],
        )
        unique_flat = flat_sorted[group_start]
        group_id_sorted = bm.cumsum(group_start, axis=0) - 1
        self.entry_to_value = group_id_sorted[bm.argsort(order)]

        row = unique_flat // ncol
        col = unique_flat % ncol
        counts = bm.bincount(row, minlength=nrow)
        counts = bm.astype(counts, base_rows.dtype)
        self.crow = bm.concatenate(
            [
                bm.zeros((1,), dtype=base_rows.dtype, device=bm.get_device(base_rows)),
                bm.cumsum(counts, axis=0),
            ],
            axis=0,
        )
        self.col = bm.astype(col, base_rows.dtype)

    def assembly(self, coef: TensorLike) -> CSRTensor:
        """Return the augmented pressure matrix for the current response field."""
        pressure_matrix = self.pressure_diffusion.assembly(coef)
        local_values = bm.concatenate([pressure_matrix.values, self.gauge_values])
        values = bm.zeros(
            (self.col.shape[0],),
            dtype=local_values.dtype,
            device=bm.get_device(local_values),
        )
        values = bm.index_add(values, self.entry_to_value, local_values, axis=0)
        return CSRTensor(self.crow, self.col, values, spshape=self.sparse_shape)


class CollocatedPressureEquation:
    """Pressure-equation components used by correction algorithms.

    The class provides the finite-volume pressure terms that SIMPLE and PISO
    call from their own algorithm loops: pressure-gradient source for the
    momentum equation, face response ``r_f``, orthogonal pressure flux,
    explicit non-orthogonal cross flux, pressure Dirichlet flux replacement,
    pressure Laplacian assembly, pure-Neumann compatibility projection, and
    pressure gauge or zero-mean representative selection.
    """

    def pressure_gradient_source(self, pressure, pressure_gradient=None):
        """Return the cell-integrated pressure-gradient source vector."""
        grad_p = (
            self.pressure_gradient.cell_gradient(pressure)
            if pressure_gradient is None
            else pressure_gradient
        )
        return bm.concatenate(
            [
                bm.einsum("i,i->i", grad_p[:, component], self.cm)
                for component in range(grad_p.shape[1])
            ]
        )

    def pressure_response_face_coefficient(self, a_p, interpolation_method: str):
        """Interpolate cell pressure response ``V/a_P`` to faces."""
        response = self.cm / a_p[: self.NC]
        return self.face_interpolate_cell_scalar(response, method=interpolation_method)

    @variantmethod("over_relaxed")
    def pressure_diffusion_decomposition(self):
        """Return the active complete pressure-Laplacian decomposition."""
        return self.fvm_geometry.diffusion_face_decomposition("over_relaxed")

    @pressure_diffusion_decomposition.register("bounded_over_relaxed")
    def pressure_diffusion_decomposition(self):
        return self.fvm_geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.diffusion_nonorthogonal_eps
        )

    @pressure_diffusion_decomposition.register("uncorrected")
    def pressure_diffusion_decomposition(self):
        return self.fvm_geometry.diffusion_face_decomposition("uncorrected")

    def pressure_orthogonal_flux(self, pressure, response_coef):
        """Return the implicit orthogonal pressure-Laplacian flux."""
        decomposition = self.pressure_diffusion_decomposition()
        coefficient = response_coef * decomposition.orthogonal_factor
        jump = pressure[self.face_to_cell[:, 0]] - pressure[self.face_to_cell[:, 1]]
        return coefficient * jump

    def add_dirichlet_pressure_flux(
        self,
        flux,
        pressure,
        response_coef,
        boundary_value,
        threshold,
    ):
        """Set pressure-induced fluxes on pressure Dirichlet boundary faces."""
        if threshold is None or boundary_value is None:
            return flux

        selected = selected_boundary_faces(self.fvm_geometry, threshold)
        if selected is None or selected.shape[0] == 0:
            return flux

        face_centers = self.fvm_geometry.face_center[selected]
        decomposition = self.pressure_diffusion_decomposition()
        coefficient = (
            response_coef[selected]
            * decomposition.orthogonal_factor[selected]
        )
        owner = self.fvm_geometry.owner[selected]
        boundary_pressure = boundary_value(face_centers)
        boundary_flux = coefficient * (pressure[owner] - boundary_pressure)
        return bm.set_at(flux, selected, flux[selected] + boundary_flux)

    def pressure_nonorthogonal_cross_flux(
        self,
        pressure: TensorLike,
        response_coef: TensorLike,
        *,
        interpolation_method: str,
        pressure_gradient=None,
        boundary_threshold=None,
    ) -> TensorLike:
        """Return the explicit non-orthogonal flux induced by a pressure field.

        The face-gradient interpolation follows the active face-interpolation
        setting used by the collocated pressure-correction route.  Pressure
        Boundary cross flux is retained only on selected pressure Dirichlet
        faces.  The supplied or reconstructed gradient must satisfy the same
        Dirichlet data as the pressure field being corrected.
        """
        if self.diffusion_method == "uncorrected":
            return bm.zeros_like(response_coef)
        T_f = self.pressure_diffusion_decomposition().T_f
        if float(bm.to_numpy(bm.max(bm.abs(T_f)))) == 0.0:
            return bm.zeros_like(response_coef)
        grad_p = (
            self.pressure_gradient.cell_gradient(pressure)
            if pressure_gradient is None
            else pressure_gradient
        )
        grad_f = reconstruct_face_gradient(
            self.mesh,
            grad_p,
            geometry=self.fvm_geometry,
            interpolation_method=interpolation_method,
        )
        cross_flux = response_coef * bm.einsum("ij,ij->i", T_f, grad_f)
        active_faces = bm.copy(self.fvm_geometry.is_internal)
        selected = selected_boundary_faces(
            self.fvm_geometry,
            boundary_threshold,
        )
        if selected is not None and selected.shape[0] > 0:
            active_faces = bm.set_at(active_faces, selected, True)
        return bm.where(active_faces, cross_flux, 0.0)

    def pressure_gauge_matrix(self, coef):
        """Assemble the pressure Laplacian with a volume-weighted gauge row."""
        assembler = getattr(self, "_pressure_gauge_matrix_assembler", None)
        if assembler is None:
            assembler = PressureGaugeMatrixAssembler(
                self.space,
                geometry=self.fvm_geometry,
                cell_measure=self.cm,
                diffusion_method=self.diffusion_method,
                diffusion_nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
            )
            self._pressure_gauge_matrix_assembler = assembler
        return assembler.assembly(coef)

    def pressure_diffusion_matrix(self, coef):
        """Assemble the scalar pressure Laplacian without boundary constraints."""
        assembler = getattr(self, "_scalar_diffusion_matrix_assembler", None)
        if assembler is None:
            assembler = ScalarDiffusionMatrixAssembler(
                self.space,
                geometry=self.fvm_geometry,
                method=self.diffusion_method,
                nonorthogonal_eps=self.diffusion_nonorthogonal_eps,
            )
            self._scalar_diffusion_matrix_assembler = assembler
        return assembler.assembly(coef)

    def pressure_rhs_compatibility(self, rhs):
        """Return the pure-Neumann pressure RHS compatibility residual."""
        return bm.sum(rhs)

    def project_pressure_rhs_to_range(self, rhs):
        """Project a cell-integrated pressure RHS to the Neumann operator range."""
        return rhs - self.pressure_rhs_compatibility(rhs) * self.cm / bm.sum(self.cm)

    def zero_mean_pressure(self, pressure):
        """Return the representative with zero volume-weighted pressure mean."""
        return pressure - bm.sum(self.cm * pressure) / bm.sum(self.cm)


class CollocatedVelocityPressureCoupling:
    """Velocity-pressure coupling after a pressure equation solve.

    SIMPLE passes a pressure correction ``p'`` to update velocity, while PISO
    passes a pressure state when constructing pressure-free velocity.  Both use
    the same cell response ``V/a_P`` and reconstructed pressure gradient.
    """

    def velocity_pressure_correction(
        self,
        cell_velocity,
        pressure_field,
        a_p,
        pressure_gradient=None,
    ):
        """Apply ``U <- U - rAU grad(p)`` for the supplied pressure argument.

        In SIMPLE callers this argument is a pressure correction ``p'``.  In
        the current PISO route it is the corrected pressure state, not an
        increment.
        """
        grad_p = (
            self.pressure_gradient.cell_gradient(pressure_field)
            if pressure_gradient is None
            else pressure_gradient
        )
        return cell_velocity - self.component_response(a_p) * grad_p

    def pressure_free_velocity(self, cell_velocity, pressure, a_p, pressure_gradient=None):
        """Remove the current pressure-gradient contribution from velocity."""
        grad_p = (
            self.pressure_gradient.cell_gradient(pressure)
            if pressure_gradient is None
            else pressure_gradient
        )
        return cell_velocity + self.component_response(a_p) * grad_p


class CollocatedNSFVMComponents(
    CollocatedDiscretizationSetup,
    CollocatedLinearSystemComponents,
    CollocatedMomentumEquation,
    CollocatedFaceFluxAlgebra,
    CollocatedPressureEquation,
    CollocatedVelocityPressureCoupling,
):
    """Facade exposing shared collocated NS components to SIMPLE and PISO.

    The facade deliberately contains no outer iteration loop.  Concrete solver
    classes inherit these mathematical components and define the SIMPLE or PISO
    sequence explicitly in their own files.
    """

    pass


__all__ = [
    "PressureGaugeMatrixAssembler",
    "CollocatedDiscretizationSetup",
    "CollocatedLinearSystemComponents",
    "CollocatedMomentumEquation",
    "CollocatedFaceFluxAlgebra",
    "CollocatedPressureEquation",
    "CollocatedVelocityPressureCoupling",
    "CollocatedNSFVMComponents",
]
