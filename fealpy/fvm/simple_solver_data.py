"""Input data containers for the collocated SIMPLE solver."""

from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class SimpleSolverControls:
    """Discretization controls for ``CollocatedSimpleSolver``."""

    space_degree: int = 0
    pressure_gradient_method: str = "extended_lsq"
    velocity_gradient_method: str = "extended_lsq"
    rhie_chow_pressure_gradient_method: str = "extended_lsq"
    face_interpolation_method: str = "average"
    momentum_face_interpolation: str | None = None
    pressure_response_interpolation: str | None = None
    rhie_chow_velocity_interpolation: str | None = None
    momentum_nonorthogonal_max_iter: int = 10
    momentum_nonorthogonal_tol: float = 1.0e-4
    pressure_nonorthogonal_max_iter: int = 10
    pressure_nonorthogonal_tol: float = 1.0e-5

    @classmethod
    def from_options(cls, options):
        """Build controls from the historical model option dictionary."""
        return cls(
            space_degree=options.get("space_degree", 0),
            pressure_gradient_method=options.get(
                "pressure_gradient_method", "extended_lsq"
            ),
            velocity_gradient_method=options.get(
                "velocity_gradient_method", "extended_lsq"
            ),
            rhie_chow_pressure_gradient_method=options.get(
                "rhie_chow_pressure_gradient_method", "extended_lsq"
            ),
            face_interpolation_method=options.get(
                "face_interpolation_method", "average"
            ),
            momentum_face_interpolation=options.get("momentum_face_interpolation"),
            pressure_response_interpolation=options.get(
                "pressure_response_interpolation"
            ),
            rhie_chow_velocity_interpolation=options.get(
                "rhie_chow_velocity_interpolation"
            ),
            momentum_nonorthogonal_max_iter=options.get(
                "momentum_nonorthogonal_max_iter", 10
            ),
            momentum_nonorthogonal_tol=options.get(
                "momentum_nonorthogonal_tol", 1.0e-4
            ),
            pressure_nonorthogonal_max_iter=options.get(
                "pressure_nonorthogonal_max_iter", 10
            ),
            pressure_nonorthogonal_tol=options.get(
                "pressure_nonorthogonal_tol", 1.0e-5
            ),
        )

    def face_interpolation(self, key: str) -> str:
        """Return a face interpolation choice with shared fallback."""
        value = getattr(self, key)
        return self.face_interpolation_method if value is None else value


@dataclass(frozen=True)
class SimpleBoundaryConditions:
    """Minimal boundary data adapter for all-boundary or thresholded velocity BC."""

    velocity_dirichlet: object
    velocity_dirichlet_threshold: object = None
    velocity_natural_threshold: object = None
    pressure_dirichlet: object = None
    pressure_dirichlet_threshold_value: object = None

    def conditions_for(self, variable: str, kind: str | None = None):
        """Return non-empty markers matching the available simple conditions."""
        if variable == "velocity" and kind in (None, "dirichlet"):
            return (self,) if self.velocity_dirichlet is not None else ()
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

    def boundary_face_velocity(self, variable: str = "velocity", *, mesh):
        """Return selected boundary faces and prescribed velocities."""
        if variable != "velocity":
            raise ValueError("boundary_face_velocity only supports 'velocity'.")
        geometry = FVMGeometry(mesh)
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[boundary_faces]
        threshold = self.velocity_dirichlet_threshold
        if threshold is None:
            flag = bm.ones(boundary_faces.shape[0], dtype=bm.bool)
        else:
            flag = threshold(points)
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
