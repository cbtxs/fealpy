"""Engineering boundary-condition adapter for collocated FVM solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class BoundaryPatch:
    """Named boundary-face patch selected by physical face centers."""

    name: str
    selector: Callable


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition assigned to one variable on one named patch."""

    variable: str
    patch: str
    kind: str
    value: object = None


class EngineeringBoundaryConditions:
    """Map engineering patches to existing FVM boundary-condition helpers.

    The adapter owns no SIMPLE/PISO algebra.  It only converts named patches
    and physical boundary data into value callables, thresholds, and selected
    boundary-face arrays that ``DirichletBC`` and solver face constraints can
    consume.
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
        self._patch_by_name = self._build_patch_map(self.patches)
        self._validate_conditions()

    @staticmethod
    def _build_patch_map(patches):
        patch_by_name = {}
        for patch in patches:
            if patch.name in patch_by_name:
                raise ValueError(f"duplicate boundary patch: {patch.name!r}.")
            if not callable(patch.selector):
                raise ValueError(f"selector for patch {patch.name!r} must be callable.")
            patch_by_name[patch.name] = patch
        return patch_by_name

    def _validate_conditions(self) -> None:
        seen = set()
        for condition in self.conditions:
            if condition.variable not in self._SUPPORTED_VARIABLES:
                raise ValueError(
                    f"unsupported boundary variable: {condition.variable!r}."
                )
            if condition.kind not in self._SUPPORTED_KINDS:
                raise ValueError(f"unsupported boundary kind: {condition.kind!r}.")
            if condition.patch not in self._patch_by_name:
                raise ValueError(f"unknown boundary patch: {condition.patch!r}.")
            key = (condition.variable, condition.patch)
            if key in seen:
                raise ValueError(
                    f"duplicate condition for {condition.variable!r} on "
                    f"{condition.patch!r}."
                )
            seen.add(key)
            if condition.kind == "dirichlet" and condition.value is None:
                raise ValueError(
                    f"Dirichlet condition on {condition.patch!r} needs value data."
                )

    def conditions_for(self, variable: str, kind: str | None = None):
        """Return conditions matching a variable and optionally a kind."""
        return tuple(
            condition
            for condition in self.conditions
            if condition.variable == variable
            and (kind is None or condition.kind == kind)
        )

    def has_dirichlet(self, variable: str) -> bool:
        """Return whether ``variable`` has at least one Dirichlet patch."""
        return bool(self.conditions_for(variable, "dirichlet"))

    def dirichlet_threshold(self, variable: str):
        """Return a face-center selector for all Dirichlet patches of variable."""
        return self.kind_threshold(variable, "dirichlet")

    def natural_threshold(self, variable: str):
        """Return a face-center selector for all natural patches of variable."""
        return self.kind_threshold(variable, "natural")

    def kind_threshold(self, variable: str, kind: str):
        """Return a face-center selector for all patches of one condition kind."""
        conditions = self.conditions_for(variable, kind)

        def threshold(points):
            flag = bm.zeros(points.shape[0], dtype=bm.bool)
            for condition in conditions:
                flag = flag | self._patch_flag(condition.patch, points)
            return flag

        return threshold

    def dirichlet_value(self, variable: str):
        """Return a value callable assembled from Dirichlet patch data."""
        conditions = self.conditions_for(variable, "dirichlet")
        if not conditions:
            raise ValueError(f"{variable!r} has no Dirichlet boundary conditions.")

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

    def boundary_face_velocity(self, variable: str = "velocity"):
        """Return selected boundary faces and Dirichlet velocities on them."""
        if variable != "velocity":
            raise ValueError("boundary_face_velocity only supports 'velocity'.")

        boundary_faces = bm.nonzero(self.geometry.is_boundary)[0]
        points = self.geometry.face_center[boundary_faces]
        if not self.has_dirichlet(variable):
            return boundary_faces[:0], bm.zeros((0, points.shape[1]), dtype=points.dtype)

        flag = self.dirichlet_threshold(variable)(points)
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
        flag = bm.array(patch.selector(points), dtype=bm.bool)
        if flag.shape != (points.shape[0],):
            raise ValueError(
                f"selector for patch {patch_name!r} must return one boolean "
                "per point."
            )
        return flag

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
