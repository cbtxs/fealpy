"""Boundary-condition data adapters for collocated FVM solvers.

The solver kernels consume strict PDE boundary data: velocity Dirichlet,
velocity Neumann, velocity natural outlet, and pressure Dirichlet/reference
constraints.  Engineering patch names are resolved here before entering SIMPLE
or PISO, so solver code does not need to understand inlet/outlet/wall labels.

When pressure has no Dirichlet/reference patch, the pressure equation is closed
by the solver's homogeneous Neumann plus gauge route.  That implicit pressure
closure is different from an explicit ``pressure/neumann(value)`` engineering
condition, which is intentionally rejected for now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry, boundary_face_flag, selected_boundary_faces


def apply_face_velocity_constraint(face_velocity, boundary_faces, boundary_velocity):
    """Apply prescribed face velocity on selected boundary faces."""
    if boundary_velocity is None:
        return face_velocity
    if boundary_faces is None:
        raise ValueError("boundary_faces must be supplied with boundary_velocity.")
    return bm.set_at(bm.array(face_velocity), boundary_faces, bm.array(boundary_velocity))


def apply_boundary_flux_constraint(flux, boundary_faces, boundary_velocity, face_normal):
    """Apply prescribed owner-oriented normal flux on selected boundary faces."""
    if boundary_velocity is None:
        return flux
    if boundary_faces is None:
        raise ValueError("boundary_faces must be supplied with boundary_velocity.")
    target_flux = bm.einsum("ij,ij->i", bm.array(boundary_velocity), face_normal[boundary_faces])
    return bm.set_at(bm.array(flux), boundary_faces, target_flux)


@dataclass(frozen=True)
class BoundaryPatch:
    """Named boundary-face patch selected by physical face centers."""

    name: str
    selector: Callable


@dataclass(frozen=True)
class BoundaryCondition:
    """Engineering boundary condition assigned to one variable and patch.

    Current supported combinations are ``velocity/dirichlet``,
    ``velocity/neumann``, ``velocity/natural``, ``pressure/dirichlet``, and
    ``pressure/reference``.  Explicit ``pressure/neumann(value)`` is rejected
    because pressure-correction solvers need a separate flux-consistency
    design for that case.
    """

    variable: str
    patch: str
    kind: str
    value: object = None


class PDEBoundaryConditions:
    """Strict PDE boundary data consumed by collocated FVM solvers.

    This object has no engineering patch semantics.  It only exposes
    variable-based thresholds and value callables matching the protocol used by
    SIMPLE and PISO solver kernels:

    - velocity Dirichlet value;
    - velocity Neumann normal derivative;
    - velocity natural outlet selector;
    - pressure Dirichlet value, including engineering pressure reference.

    If pressure Dirichlet data are absent, the pressure equation uses the
    solver-side homogeneous Neumann plus gauge route.  That route is not
    represented as a ``pressure/neumann(value)`` boundary condition here.
    """

    def __init__(
        self,
        mesh,
        *,
        velocity_dirichlet=None,
        velocity_dirichlet_threshold=None,
        velocity_neumann=None,
        velocity_neumann_threshold=None,
        velocity_natural_threshold=None,
        pressure_dirichlet=None,
        pressure_dirichlet_threshold=None,
        geometry_class=FVMGeometry,
    ) -> None:
        self.mesh = mesh
        self.geometry = geometry_class(mesh)
        self.velocity_dirichlet = velocity_dirichlet
        self.velocity_dirichlet_threshold = velocity_dirichlet_threshold
        self.velocity_neumann = velocity_neumann
        self.velocity_neumann_threshold = velocity_neumann_threshold
        self.velocity_natural_threshold = velocity_natural_threshold
        self.pressure_dirichlet = pressure_dirichlet
        self.pressure_dirichlet_threshold_value = pressure_dirichlet_threshold

    def conditions_for(self, variable: str, kind: str | None = None):
        """Return non-empty markers matching the available PDE conditions."""
        if variable == "velocity" and kind is None:
            has_velocity_condition = (
                self.velocity_dirichlet is not None
                or self.velocity_neumann is not None
                or self.velocity_natural_threshold is not None
            )
            return (self,) if has_velocity_condition else ()
        if variable == "velocity" and kind == "dirichlet":
            return (self,) if self.velocity_dirichlet is not None else ()
        if variable == "velocity" and kind == "neumann":
            return (self,) if self.velocity_neumann is not None else ()
        if variable == "velocity" and kind == "natural":
            return (self,) if self.velocity_natural_threshold is not None else ()
        if variable == "pressure" and kind in (None, "dirichlet"):
            return (self,) if self.has_pressure_dirichlet() else ()
        return ()

    def dirichlet_value(self, variable: str):
        """Return the Dirichlet value callable for one variable."""
        if variable == "velocity" and self.velocity_dirichlet is not None:
            return self.velocity_dirichlet
        if variable == "pressure" and self.has_pressure_dirichlet():
            return self.pressure_dirichlet_value()
        raise ValueError(f"{variable!r} has no Dirichlet boundary condition.")

    def dirichlet_threshold(self, variable: str):
        """Return the Dirichlet threshold for one variable."""
        if variable == "velocity":
            return self.velocity_dirichlet_threshold
        if variable == "pressure":
            return self.pressure_dirichlet_threshold()
        raise ValueError(f"unsupported boundary variable: {variable!r}.")

    def natural_threshold(self, variable: str):
        """Return the natural threshold for one variable."""
        if variable != "velocity":
            raise ValueError("only velocity natural boundary is supported.")
        return self.velocity_natural_threshold

    def neumann_value(self, variable: str):
        """Return the Neumann value callable for one variable."""
        if variable == "velocity" and self.velocity_neumann is not None:
            return self.velocity_neumann
        raise ValueError(f"{variable!r} has no Neumann boundary condition.")

    def neumann_threshold(self, variable: str):
        """Return the Neumann threshold for one variable."""
        if variable != "velocity":
            raise ValueError("only velocity Neumann boundary is supported.")
        return self.velocity_neumann_threshold

    def boundary_face_velocity(self, variable: str = "velocity", *, mesh=None):
        """Return selected boundary faces and prescribed velocities."""
        if variable != "velocity":
            raise ValueError("boundary_face_velocity only supports 'velocity'.")
        if mesh is not None and mesh is not self.mesh:
            raise ValueError("mesh-bound boundary conditions require their original mesh.")
        geometry = self.geometry
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[boundary_faces]
        if self.velocity_dirichlet is None:
            return boundary_faces[:0], bm.zeros((0, points.shape[1]), dtype=points.dtype)
        if self.velocity_dirichlet_threshold is None:
            flag = bm.ones(boundary_faces.shape[0], dtype=bm.bool)
        else:
            flag = boundary_face_flag(points, self.velocity_dirichlet_threshold)
        return boundary_faces[flag], self.velocity_dirichlet(points)[flag]

    def has_pressure_dirichlet(self) -> bool:
        """Return whether pressure Dirichlet data are available."""
        return self.pressure_dirichlet is not None and self.pressure_dirichlet_threshold_value is not None

    def pressure_dirichlet_threshold(self):
        """Return pressure Dirichlet threshold."""
        return self.pressure_dirichlet_threshold_value

    def pressure_dirichlet_value(self):
        """Return pressure Dirichlet values as a callable."""
        value = self.pressure_dirichlet
        if callable(value):
            return value

        def constant(points):
            return bm.broadcast_to(bm.array(value, dtype=points.dtype), (points.shape[0],))

        return constant


@dataclass(frozen=True)
class BoundaryConditionData:
    """Mesh-independent PDE boundary data for solver adapters.

    This class preserves the old manufactured-case API where value callables
    and thresholds are known before the mesh object is passed into the solver.
    When a mesh is available, ``to_pde_boundary(mesh)`` returns the strict
    mesh-bound ``PDEBoundaryConditions`` object used by the solver protocol.
    """

    velocity_dirichlet: object
    velocity_dirichlet_threshold: object = None
    velocity_neumann: object = None
    velocity_neumann_threshold: object = None
    velocity_natural_threshold: object = None
    pressure_dirichlet: object = None
    pressure_dirichlet_threshold_value: object = None

    def to_pde_boundary(self, mesh):
        """Return strict PDE boundary data bound to a concrete mesh."""
        return PDEBoundaryConditions(
            mesh,
            velocity_dirichlet=self.velocity_dirichlet,
            velocity_dirichlet_threshold=self.velocity_dirichlet_threshold,
            velocity_neumann=self.velocity_neumann,
            velocity_neumann_threshold=self.velocity_neumann_threshold,
            velocity_natural_threshold=self.velocity_natural_threshold,
            pressure_dirichlet=self.pressure_dirichlet,
            pressure_dirichlet_threshold=self.pressure_dirichlet_threshold_value,
        )


class EngineeringBoundaryConditions:
    """Map engineering patches to strict PDE boundary-condition data.

    The adapter owns no SIMPLE/PISO algebra.  It converts named patches and
    engineering boundary data into value callables, thresholds, and selected
    boundary-face arrays that the existing boundary operators and solver face
    constraints consume.
    """

    _SUPPORTED_VARIABLES = {"velocity", "pressure"}
    _SUPPORTED_KINDS = {"dirichlet", "neumann", "natural", "reference"}

    def __init__(
        self,
        mesh,
        patches: Iterable[BoundaryPatch],
        conditions: Iterable[BoundaryCondition],
    ) -> None:
        self.mesh = mesh
        self.geometry = FVMGeometry(mesh)
        self.patches = tuple(patches)
        self.conditions = tuple(conditions)
        patch_by_name = {}
        for patch in self.patches:
            if patch.name in patch_by_name:
                raise ValueError(f"duplicate boundary patch: {patch.name!r}.")
            if not callable(patch.selector):
                raise ValueError(f"selector for patch {patch.name!r} must be callable.")
            patch_by_name[patch.name] = patch
        self._patch_by_name = patch_by_name
        self._validate_conditions()

    def _validate_conditions(self) -> None:
        seen = set()
        for condition in self.conditions:
            if condition.variable not in self._SUPPORTED_VARIABLES:
                raise ValueError(f"unsupported boundary variable: {condition.variable!r}.")
            if condition.kind not in self._SUPPORTED_KINDS:
                raise ValueError(f"unsupported boundary kind: {condition.kind!r}.")
            if condition.patch not in self._patch_by_name:
                raise ValueError(f"unknown boundary patch: {condition.patch!r}.")
            # Combination rules keep the engineering layer aligned with the
            # strict PDE boundary protocol consumed by SIMPLE/PISO.
            if condition.kind == "reference" and condition.variable != "pressure":
                raise ValueError("reference boundary is only supported for pressure.")
            if condition.kind == "natural" and condition.variable != "velocity":
                raise ValueError("natural boundary is only supported for velocity.")
            if condition.variable == "pressure" and condition.kind == "neumann":
                raise ValueError(
                    "explicit pressure/neumann(value) is not implemented for "
                    "pressure-correction solvers."
                )
            key = (condition.variable, condition.patch)
            if key in seen:
                raise ValueError(
                    f"duplicate condition for {condition.variable!r} on "
                    f"{condition.patch!r}."
                )
            seen.add(key)
            if condition.kind in ("dirichlet", "neumann", "reference") and condition.value is None:
                raise ValueError(
                    f"{condition.kind.capitalize()} condition on "
                    f"{condition.patch!r} needs value data."
                )

    def conditions_for(self, variable: str, kind: str | None = None):
        """Return conditions matching a variable and optionally a kind."""
        return tuple(
            condition
            for condition in self.conditions
            if condition.variable == variable
            and (kind is None or condition.kind == kind)
        )

    def to_pde_boundary(self):
        """Return strict PDE boundary data for solver kernels."""
        return PDEBoundaryConditions(
            self.mesh,
            velocity_dirichlet=(
                self.dirichlet_value("velocity")
                if self.has_dirichlet("velocity")
                else None
            ),
            velocity_dirichlet_threshold=(
                self.dirichlet_threshold("velocity")
                if self.has_dirichlet("velocity")
                else None
            ),
            velocity_neumann=(
                self.neumann_value("velocity")
                if self.conditions_for("velocity", "neumann")
                else None
            ),
            velocity_neumann_threshold=(
                self.neumann_threshold("velocity")
                if self.conditions_for("velocity", "neumann")
                else None
            ),
            velocity_natural_threshold=(
                self.natural_threshold("velocity")
                if self.conditions_for("velocity", "natural")
                else None
            ),
            pressure_dirichlet=(
                self.dirichlet_value("pressure")
                if self.has_dirichlet("pressure")
                else None
            ),
            pressure_dirichlet_threshold=(
                self.dirichlet_threshold("pressure")
                if self.has_dirichlet("pressure")
                else None
            ),
        )

    def has_dirichlet(self, variable: str) -> bool:
        """Return whether ``variable`` has at least one Dirichlet patch.

        Pressure reference patches are treated as pressure Dirichlet patches
        before the data reach the solver.
        """
        if variable == "pressure":
            return bool(
                self.conditions_for("pressure", "dirichlet")
                or self.conditions_for("pressure", "reference")
            )
        return bool(self.conditions_for(variable, "dirichlet"))

    def dirichlet_threshold(self, variable: str):
        """Return a face-center selector for all Dirichlet patches of variable."""
        kinds = ("dirichlet", "reference") if variable == "pressure" else ("dirichlet",)
        return self.condition_threshold(variable, kinds)

    def natural_threshold(self, variable: str):
        """Return a face-center selector for all natural patches of variable."""
        return self.condition_threshold(variable, ("natural",))

    def neumann_threshold(self, variable: str):
        """Return a face-center selector for all Neumann patches of variable."""
        return self.condition_threshold(variable, ("neumann",))

    def condition_threshold(self, variable: str, kinds):
        """Return a face-center selector for all patches of selected kinds."""
        conditions = tuple(
            condition
            for kind in kinds
            for condition in self.conditions_for(variable, kind)
        )

        def threshold(points):
            flag = bm.zeros(points.shape[0], dtype=bm.bool)
            for condition in conditions:
                flag = flag | self._patch_flag(condition.patch, points)
            return flag

        return threshold

    def dirichlet_value(self, variable: str):
        """Return a value callable assembled from Dirichlet patch data."""
        kinds = ("dirichlet", "reference") if variable == "pressure" else ("dirichlet",)
        return self.condition_value(variable, kinds)

    def neumann_value(self, variable: str):
        """Return a value callable assembled from Neumann patch data."""
        return self.condition_value(variable, ("neumann",))

    def condition_value(self, variable: str, kinds):
        """Return a value callable assembled from patch data for selected kinds."""
        conditions = tuple(
            condition
            for kind in kinds
            for condition in self.conditions_for(variable, kind)
        )
        if not conditions:
            raise ValueError(f"{variable!r} has no {kinds!r} boundary conditions.")

        def value(points):
            result = self._empty_value(points, variable)
            for condition in conditions:
                flag = self._patch_flag(condition.patch, points)
                if bool(bm.to_numpy(bm.any(flag))):
                    condition_value = self._condition_value(condition, points, variable)
                    result = bm.set_at(result, flag, condition_value[flag])
            return result

        return value

    def patch_face_mask(self, patch_name: str):
        """Return a boundary-face mask for one named patch."""
        boundary_faces = bm.nonzero(self.geometry.is_boundary)[0]
        points = self.geometry.face_center[boundary_faces]
        return self._patch_flag(patch_name, points)

    def patch_face_index(self, patch_name: str):
        """Return global face indices belonging to one named patch."""
        boundary_faces = bm.nonzero(self.geometry.is_boundary)[0]
        return boundary_faces[self.patch_face_mask(patch_name)]

    def boundary_face_velocity(self, variable: str = "velocity", *, mesh=None):
        """Return selected boundary faces and Dirichlet velocities on them."""
        if variable != "velocity":
            raise ValueError("boundary_face_velocity only supports 'velocity'.")

        if mesh is not None and mesh is not self.mesh:
            raise ValueError("mesh-bound boundary conditions require their original mesh.")
        geometry = self.geometry
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[boundary_faces]
        if not self.has_dirichlet(variable):
            return boundary_faces[:0], bm.zeros((0, points.shape[1]), dtype=points.dtype)

        flag = boundary_face_flag(points, self.dirichlet_threshold(variable))
        return boundary_faces[flag], self.dirichlet_value(variable)(points)[flag]

    def has_pressure_dirichlet(self) -> bool:
        """Return whether pressure has a strict Dirichlet patch."""
        return self.has_dirichlet("pressure")

    def pressure_dirichlet_threshold(self):
        """Return pressure Dirichlet patch selector."""
        return self.dirichlet_threshold("pressure")

    def pressure_dirichlet_value(self):
        """Return pressure Dirichlet value callable."""
        return self.dirichlet_value("pressure")

    def _patch_flag(self, patch_name: str, points):
        patch = self._patch_by_name[patch_name]
        return boundary_face_flag(points, patch.selector)

    def _empty_value(self, points, variable: str):
        if variable == "velocity":
            return bm.zeros(points.shape, dtype=points.dtype)
        return bm.zeros(points.shape[0], dtype=points.dtype)

    def _condition_value(self, condition: BoundaryCondition, points, variable: str):
        raw = condition.value(points) if callable(condition.value) else condition.value
        value = bm.array(raw, dtype=points.dtype)
        return self._broadcast_value(value, points, variable)

    def _broadcast_value(self, value, points, variable: str):
        n_point = points.shape[0]
        dimension = points.shape[1]
        if variable == "velocity":
            if value.shape == points.shape:
                return value
            if value.shape == (dimension,):
                return bm.broadcast_to(value, points.shape)
            if value.shape == ():
                return bm.broadcast_to(value, points.shape)
            raise ValueError(
                "velocity boundary value must be scalar, vector-sized, or "
                "point-wise vector data."
            )

        if value.shape == (n_point,):
            return value
        if value.shape == (n_point, 1):
            return value[:, 0]
        if value.shape == ():
            return bm.broadcast_to(value, (n_point,))
        raise ValueError(
            "pressure boundary value must be scalar or point-wise scalar data."
        )
