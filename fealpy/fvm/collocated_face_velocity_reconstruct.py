"""Pressure-stabilized face velocity reconstruction for collocated FVM solvers."""

from fealpy.backend import backend_manager as bm

from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import (
    FVMGeometry,
    boundary_face_flag,
    interpolate_cell_to_face,
)


class RhieChowInterpolation:
    """Build pressure-stabilized face velocities on collocated grids.

    The interpolation starts from cell-centred velocity interpolation and adds
    the standard Rhie-Chow pressure-gradient difference.  The optional
    ``face_response_coefficient`` lets a SIMPLE/PISO model pass the same face
    pressure response used by its pressure-correction equation, keeping the
    pressure equation and face-velocity update algebraically consistent.

    This class is a pressure-velocity coupling operator; it does not assemble
    the momentum equation or choose pressure relaxation parameters.
    """

    def __init__(
        self,
        mesh,
        *,
        pressure_gradient_method="layered_lsq",
        gradient_layer_weights=(1.0, 0.25),
        gradient_boundary_weight=1.0,
        velocity_interpolation="average",
        dirichlet_pressure=None,
        dirichlet_pressure_threshold=None,
        geometry=None,
    ):
        from .gradient_reconstruct import GradientReconstruct

        self.mesh = mesh
        self.fvm_geometry = geometry if geometry is not None else FVMGeometry(mesh)
        self.cm = self.fvm_geometry.cell_measure
        self.NC = self.fvm_geometry.NC
        self.GD = self.fvm_geometry.cell_center.shape[1]
        self.face_to_cell = self.fvm_geometry.face_to_cell
        self.velocity_interpolation = self._validate_velocity_interpolation(
            velocity_interpolation
        )
        self.dirichlet_pressure = dirichlet_pressure
        self.dirichlet_pressure_threshold = dirichlet_pressure_threshold
        self.gradient_reconstruct = GradientReconstruct(
            mesh,
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
        self.d_f = self.fvm_geometry.d_f
        self.mag_d_f = self.fvm_geometry.mag_d_f

    @staticmethod
    def _validate_velocity_interpolation(velocity_interpolation):
        if velocity_interpolation not in {"average", "linear"}:
            raise ValueError("velocity_interpolation must be 'average' or 'linear'.")
        return velocity_interpolation

    def interpolate_cell_value(self, value):
        return interpolate_cell_to_face(
            value,
            geometry=self.fvm_geometry,
            method=self.velocity_interpolation,
        )

    def cell_velocity_to_face(self, u, *, face_response_coefficient):
        """Interpolate cell velocity and attach an explicit face response."""
        if u.shape != (self.NC, self.GD):
            raise ValueError(
                "cell velocity must have shape (NC, GD); flattening is only "
                "allowed at the linear-system boundary."
            )
        face_velocity = self.interpolate_cell_value(u)
        face_response = face_response_coefficient[:, None]
        return face_velocity, face_response

    def pressure_gradient_difference(self, p, pressure_gradient=None):
        """Return the Rhie-Chow pressure-gradient difference.

        This is the difference between the cell-jump pressure gradient along
        the owner-neighbour line and the interpolated reconstructed gradient.
        """
        d_f, mag_d_f = self.d_f, self.mag_d_f
        partial_p = (p[self.face_to_cell[:, 1]] - p[self.face_to_cell[:, 0]]) / mag_d_f
        partial_p = self.apply_dirichlet_pressure_boundary_partial(p, partial_p)
        e_cf = d_f / mag_d_f[:, None]
        grad_p = (
            self.gradient_reconstruct.cell_gradient(p)
            if pressure_gradient is None
            else pressure_gradient
        )
        overline_grad_p_f = reconstruct_face_gradient(
            self.mesh,
            grad_p,
            geometry=self.fvm_geometry,
        )
        interpolated_normal_gradient = bm.einsum("ij,ij->i", overline_grad_p_f, e_cf)
        gradient_difference = (partial_p - interpolated_normal_gradient)[:, None] * e_cf
        return gradient_difference

    def apply_dirichlet_pressure_boundary_partial(self, p, partial_p):
        """Use pressure Dirichlet data in boundary compact pressure jumps."""
        if self.dirichlet_pressure is None:
            return partial_p

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        if self.dirichlet_pressure_threshold is not None:
            flag = boundary_face_flag(face_centers, self.dirichlet_pressure_threshold)
            boundary_faces = boundary_faces[flag]
            face_centers = face_centers[flag]

        if boundary_faces.shape[0] == 0:
            return partial_p

        owner = self.fvm_geometry.owner[boundary_faces]
        boundary_value = self.dirichlet_pressure(face_centers)
        boundary_partial = (boundary_value - p[owner]) / self.mag_d_f[boundary_faces]
        return bm.set_at(partial_p, boundary_faces, boundary_partial)

    def reconstruct(
        self,
        u,
        ap,
        p,
        face_response_coefficient=None,
        pressure_gradient=None,
    ):
        """Return pressure-stabilized vector face velocity."""
        if face_response_coefficient is None:
            cell_response = self.cm / ap[:self.NC]
            face_response_coefficient = self.interpolate_cell_value(cell_response)
        face_velocity, face_response = self.cell_velocity_to_face(
            u,
            face_response_coefficient=face_response_coefficient,
        )
        grad_diff = self.pressure_gradient_difference(
            p,
            pressure_gradient=pressure_gradient,
        )
        return face_velocity - face_response * grad_diff
