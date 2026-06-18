"""Shared internal operators for collocated Navier-Stokes FVM solvers."""

from typing import Optional

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm
from fealpy.sparse import CSRTensor, spdiags

from .convection_integrator import ConvectionMatrixAssembler
from .scalar_cross_diffusion_integrator import CrossDiffusionRHSAssembler
from .scalar_diffusion_integrator import (
    ScalarDiffusionIntegrator,
    ScalarDiffusionMatrixAssembler,
)
from .scalar_source_integrator import ScalarSourceIntegrator
from .gradient_reconstruct import GradientReconstruct
from .face_gradient import reconstruct_face_gradient
from .div_reconstruct import DivergenceReconstruct
from .dirichlet_bc import DirichletBC
from .fvm_geometry import FVMGeometry, selected_boundary_faces


class PressureGaugeMatrixAssembler:
    r"""Assemble the pressure Laplacian plus a volume-weighted gauge row.

    The matrix corresponds to ``ScalarDiffusionIntegrator`` on the pressure
    space plus one Lagrange-multiplier row and column for the pressure gauge.
    Its sparsity pattern is fixed by the mesh; each call updates only the
    values induced by the current face response coefficient.
    """

    def __init__(
        self,
        space,
        *,
        geometry: Optional[FVMGeometry] = None,
        cell_measure: Optional[TensorLike] = None,
    ) -> None:
        self.space = space
        self.mesh = getattr(space, "mesh", None)
        self.geometry = geometry if geometry is not None else FVMGeometry(self.mesh)
        self.NC = self.mesh.number_of_cells()
        self.sparse_shape = (self.NC + 1, self.NC + 1)
        self.cell_measure = (
            self.mesh.entity_measure("cell") if cell_measure is None else cell_measure
        )
        self.pressure_diffusion = ScalarDiffusionMatrixAssembler(
            space,
            geometry=self.geometry,
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
        """Return the pressure-gauge matrix for the current face coefficient."""
        pressure_matrix = self.pressure_diffusion.assembly(coef)
        local_values = bm.concatenate([pressure_matrix.values, self.gauge_values])
        values = bm.zeros(
            (self.col.shape[0],),
            dtype=local_values.dtype,
            device=bm.get_device(local_values),
        )
        values = bm.index_add(values, self.entry_to_value, local_values, axis=0)
        return CSRTensor(self.crow, self.col, values, spshape=self.sparse_shape)


class CollocatedNSFVMOperators:
    """Internal mixin for common collocated Navier-Stokes FVM algebra.

    Velocity fields use two shapes deliberately: physical operators take
    cell-vector values with shape ``(NC, GD)``, while assembled linear systems
    use component-major dof vectors with shape ``(GD*NC,)``.
    """

    # Setup and algebraic shape conversion.

    def init_collocated_discretization(
        self,
        degree: int,
        velocity_dirichlet,
        *,
        pressure_gradient_method: str = "layered_lsq",
        velocity_gradient_method: str = "layered_lsq",
        velocity_dirichlet_threshold=None,
        pressure_dirichlet=None,
        pressure_dirichlet_threshold=None,
        with_divergence: bool = False,
        with_velocity_dirichlet_bc: bool = False,
    ) -> None:
        self.p = degree
        self.GD = self.mesh.geo_dimension()
        self.space = ScaledMonomialSpace(self.mesh, degree)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(self.GD, -1))
        self.cell_center = self.mesh.entity_barycenter("cell")
        self.face_center = self.mesh.entity_barycenter("face")
        self.fvm_geometry = FVMGeometry(self.mesh)

        self.pressure_gradient = GradientReconstruct(
            self.mesh,
            method=pressure_gradient_method,
            gd=pressure_dirichlet,
            bc_type="dirichlet" if pressure_dirichlet is not None else None,
            threshold=pressure_dirichlet_threshold,
            geometry=self.fvm_geometry,
        )
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method=velocity_gradient_method,
            gd=velocity_dirichlet,
            bc_type="dirichlet",
            threshold=velocity_dirichlet_threshold,
            geometry=self.fvm_geometry,
        )
        self.velocity_dirichlet = velocity_dirichlet
        self.velocity_dirichlet_threshold = velocity_dirichlet_threshold
        if with_divergence:
            self.divergence = DivergenceReconstruct(self.mesh)
        if with_velocity_dirichlet_bc:
            self.velocity_dirichlet_bc = DirichletBC(
                self.mesh,
                velocity_dirichlet,
                threshold=velocity_dirichlet_threshold,
                geometry=self.fvm_geometry,
            )

        self.face_to_cell = self.fvm_geometry.face_to_cell
        self.last_nonorthogonal_iterations = 0
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

    # Momentum equation components.

    def momentum_diffusion_matrix(self, diffusion_coef):
        """Assemble and cache the shared momentum diffusion matrix."""
        cache = getattr(self, "_momentum_diffusion_matrix_cache", None)
        if cache is None:
            cache = {}
            self._momentum_diffusion_matrix_cache = cache

        try:
            key = ("scalar", float(diffusion_coef))
        except TypeError:
            key = ("object", id(diffusion_coef))

        if key not in cache:
            cache[key] = BilinearForm(self.velocity_space).add_integrator(
                ScalarDiffusionIntegrator(
                    q=self.p + 2,
                    coef=diffusion_coef,
                    geometry=self.fvm_geometry,
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

    def add_velocity_natural_convection_diagonal(
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
        return matrix + spdiags(diagonal, 0, matrix.shape[0], matrix.shape[1])

    def momentum_source_vector(self, source):
        """Assemble the shared cell-integrated momentum source vector."""
        return LinearForm(self.velocity_space).add_integrator(
            ScalarSourceIntegrator(source, q=self.p + 2)
        ).assembly()

    def velocity_neumann_boundary_data(self):
        """Return velocity Neumann boundary faces and normal derivatives."""
        threshold = getattr(self, "velocity_neumann_threshold", None)
        value = getattr(self, "velocity_neumann_data", None)
        if threshold is None or value is None:
            return None, None

        boundary_faces = selected_boundary_faces(self.fvm_geometry, threshold)
        if boundary_faces is None or boundary_faces.shape[0] == 0:
            return boundary_faces, bm.zeros((0, self.GD), dtype=self.cm.dtype)
        points = self.fvm_geometry.face_center[boundary_faces]
        return boundary_faces, bm.array(value(points), dtype=self.cm.dtype)

    def velocity_neumann_diffusion_source(self, diffusion_coef):
        r"""Return cell-integrated velocity Neumann diffusion RHS.

        ``velocity_neumann_data(points)`` is interpreted as the outward normal
        derivative density ``dU/dn`` on selected boundary faces.  The finite
        volume contribution is ``mu_f * dU/dn * |S_f|`` scattered to owner
        cells and returned in component-major momentum-vector layout.
        """
        boundary_faces, sn_grad = self.velocity_neumann_boundary_data()
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
            getattr(self, "velocity_natural_threshold", None),
        )
        if natural_faces is not None and natural_faces.shape[0] > 0:
            neumann_faces.append(natural_faces)
            neumann_values.append(
                bm.zeros((natural_faces.shape[0], velocity.shape[1]), dtype=velocity.dtype)
            )

        velocity_neumann_faces, velocity_neumann_values = self.velocity_neumann_boundary_data()
        if velocity_neumann_faces is not None and velocity_neumann_faces.shape[0] > 0:
            neumann_faces.append(velocity_neumann_faces)
            neumann_values.append(bm.array(velocity_neumann_values, dtype=velocity.dtype))

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
            )
            self._cross_diffusion_rhs_assembler = assembler
        return assembler.assembly(
            grad_f=grad_f,
            coef=getattr(self, "diffusion_coef", getattr(self, "mu", 1.0)),
            boundary_policy="all",
        )

    # Face interpolation and flux algebra.

    def face_interpolate_cell_scalar(self, cell_values, method: str = "linear"):
        """Linearly interpolate a cell scalar to faces using face geometry."""
        face_to_cell = self.face_to_cell[:, :2]
        if method == "linear":
            owner_weight = self.fvm_geometry.linear_owner_weight()
        elif method == "average":
            owner_weight = 0.5 * bm.ones_like(self.fvm_geometry.mag_S_f)
            owner_weight = bm.where(self.fvm_geometry.is_internal, owner_weight, 1.0)
        else:
            raise ValueError("method must be 'average' or 'linear'.")
        return (
            owner_weight * cell_values[face_to_cell[:, 0]]
            + (1.0 - owner_weight) * cell_values[face_to_cell[:, 1]]
        )

    def face_interpolate_cell_vector(self, cell_vectors, method: str = "linear"):
        """Linearly interpolate a cell vector to faces using face geometry."""
        face_to_cell = self.face_to_cell[:, :2]
        if method == "linear":
            owner_weight = self.fvm_geometry.linear_owner_weight()
        elif method == "average":
            owner_weight = 0.5 * bm.ones_like(self.fvm_geometry.mag_S_f)
            owner_weight = bm.where(self.fvm_geometry.is_internal, owner_weight, 1.0)
        else:
            raise ValueError("method must be 'average' or 'linear'.")
        return (
            owner_weight[:, None] * cell_vectors[face_to_cell[:, 0]]
            + (1.0 - owner_weight)[:, None] * cell_vectors[face_to_cell[:, 1]]
        )

    def face_flux(self, face_velocity):
        """Return the signed surface flux ``phi_f = u_f dot S_f``."""
        return bm.einsum("ij,ij->i", face_velocity, self.fvm_geometry.S_f)

    def divergence_from_flux(self, face_flux):
        """Scatter signed face fluxes to the cell flux imbalance."""
        return self.fvm_geometry.scatter_face_flux_to_cells(face_flux)

    def enforce_face_flux(self, face_velocity, target_flux):
        """Adjust only the normal component of a vector face velocity."""
        Sf = self.fvm_geometry.S_f
        current_flux = self.face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        return face_velocity + ((target_flux - current_flux) / Sf_dot_Sf)[:, None] * Sf

    # Pressure equation components.

    def pressure_gradient_source(self, pressure):
        """Return the cell-integrated pressure-gradient source vector."""
        grad_p = self.pressure_gradient.cell_gradient(pressure)
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

    def pressure_orthogonal_flux(self, pressure, response_coef):
        """Return the implicit orthogonal pressure-Laplacian flux."""
        _, mag_E_f, _ = self.fvm_geometry.over_relaxed_decomposition()
        coefficient = response_coef * mag_E_f / self.fvm_geometry.mag_d_f
        jump = pressure[self.face_to_cell[:, 0]] - pressure[self.face_to_cell[:, 1]]
        return coefficient * jump

    def add_pressure_dirichlet_flux(
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

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return flux

        selected = boundary_faces[flag]
        _, mag_E_f, _ = self.fvm_geometry.over_relaxed_decomposition()
        coefficient = (
            response_coef[selected]
            * mag_E_f[selected]
            / self.fvm_geometry.mag_d_f[selected]
        )
        owner = self.fvm_geometry.owner[selected]
        boundary_pressure = boundary_value(face_centers[flag])
        boundary_flux = coefficient * (pressure[owner] - boundary_pressure)
        return bm.set_at(flux, selected, boundary_flux)

    def pressure_nonorthogonal_cross_flux(
        self,
        pressure: TensorLike,
        response_coef: TensorLike,
        *,
        interpolation_method: str,
    ) -> TensorLike:
        """Return the explicit non-orthogonal flux induced by a pressure field.

        The face-gradient interpolation follows the active face-interpolation
        setting used by the collocated pressure-correction route.  Pressure
        boundary cross flux remains zero on non-coupled boundary faces,
        matching the pressure Laplacian correction route.
        """
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        grad_f = reconstruct_face_gradient(
            self.mesh,
            grad_p,
            geometry=self.fvm_geometry,
            interpolation_method=interpolation_method,
        )
        T_f = self.fvm_geometry.bounded_over_relaxed_decomposition()[2]
        cross_flux = response_coef * bm.einsum("ij,ij->i", T_f, grad_f)
        return bm.where(self.fvm_geometry.is_boundary, 0.0, cross_flux)

    def pressure_gauge_matrix(self, coef):
        """Assemble the pressure Laplacian with a volume-weighted gauge row."""
        assembler = getattr(self, "_pressure_gauge_matrix_assembler", None)
        if assembler is None:
            assembler = PressureGaugeMatrixAssembler(
                self.space,
                geometry=self.fvm_geometry,
                cell_measure=self.cm,
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
            )
            self._scalar_diffusion_matrix_assembler = assembler
        return assembler.assembly(coef)

    # Velocity-pressure correction components.

    def velocity_pressure_correction(self, cell_velocity, pressure_field, a_p):
        """Apply ``U <- U - rAU grad(p)`` for the supplied pressure argument.

        In SIMPLE callers this argument is a pressure correction ``p'``.  In
        the current PISO route it is the corrected pressure state, not an
        increment.
        """
        grad_p = self.pressure_gradient.cell_gradient(pressure_field)
        return cell_velocity - self.component_response(a_p) * grad_p

    def pressure_free_velocity(self, cell_velocity, pressure, a_p):
        """Remove the current pressure-gradient contribution from velocity."""
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        return cell_velocity + self.component_response(a_p) * grad_p
