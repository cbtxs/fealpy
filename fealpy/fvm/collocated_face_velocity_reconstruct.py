"""Pressure-stabilized face velocity reconstruction for collocated FVM solvers."""

from fealpy.backend import backend_manager as bm

from .face_gradient import reconstruct_face_gradient
from .fvm_geometry import FVMGeometry, boundary_face_flag


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
        velocity_interpolation="average",
        pressure_dirichlet=None,
        pressure_dirichlet_threshold=None,
        geometry=None,
    ):
        from .gradient_reconstruct import GradientReconstruct

        self.mesh = mesh
        self.cm = self.mesh.entity_measure('cell')
        self.NC = mesh.number_of_cells()
        self.GD = mesh.geo_dimension()
        self.fvm_geometry = geometry if geometry is not None else FVMGeometry(mesh)
        self.face_to_cell = self.fvm_geometry.face_to_cell
        self.velocity_interpolation = self._validate_velocity_interpolation(
            velocity_interpolation
        )
        if self.velocity_interpolation == "linear":
            self.owner_weight = self.fvm_geometry.linear_owner_weight()
        else:
            weight = 0.5 * bm.ones_like(self.fvm_geometry.mag_S_f)
            self.owner_weight = bm.where(self.fvm_geometry.is_internal, weight, 1.0)
        self.pressure_dirichlet = pressure_dirichlet
        self.pressure_dirichlet_threshold = pressure_dirichlet_threshold
        self.gradient_reconstruct = GradientReconstruct(
            mesh,
            method=pressure_gradient_method,
            gd=pressure_dirichlet,
            bc_type="dirichlet" if pressure_dirichlet is not None else None,
            threshold=pressure_dirichlet_threshold,
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
        weight = self.owner_weight
        weight_shape = (weight.shape[0],) + (1,) * (value.ndim - 1)
        weight = weight.reshape(weight_shape)
        owner = self.face_to_cell[:, 0]
        neighbour = self.face_to_cell[:, 1]
        return weight * value[owner] + (1.0 - weight) * value[neighbour]

    def cell_velocity_to_face(self, u, ap, face_response_coefficient=None):
        """Interpolate cell velocity and pressure response to faces."""
        if u.ndim == 1:
            u = bm.stack(
                [
                    u[component * self.NC : (component + 1) * self.NC]
                    for component in range(self.GD)
                ],
                axis=-1,
            )
        face_velocity = self.interpolate_cell_value(u)
        if face_response_coefficient is None:
            ap = ap[:self.NC]
            dp = self.cm / ap
            face_response = self.interpolate_cell_value(dp)[:, None]
        else:
            face_response = face_response_coefficient[:, None]
        return face_velocity, face_response

    def GradientDifference(self, p, pressure_gradient=None):
        """Return the Rhie-Chow pressure-gradient difference.

        This is the difference between the cell-jump pressure gradient along
        the owner-neighbour line and the interpolated reconstructed gradient.
        """
        d_f, mag_d_f = self.d_f, self.mag_d_f
        partial_p = (p[self.face_to_cell[:, 1]] - p[self.face_to_cell[:, 0]]) / mag_d_f
        partial_p = self.apply_pressure_dirichlet_boundary_partial(p, partial_p)
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

    def apply_pressure_dirichlet_boundary_partial(self, p, partial_p):
        """Use pressure Dirichlet data in boundary compact pressure jumps."""
        if self.pressure_dirichlet is None:
            return partial_p

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        if self.pressure_dirichlet_threshold is not None:
            flag = boundary_face_flag(face_centers, self.pressure_dirichlet_threshold)
            boundary_faces = boundary_faces[flag]
            face_centers = face_centers[flag]

        if boundary_faces.shape[0] == 0:
            return partial_p

        owner = self.fvm_geometry.owner[boundary_faces]
        boundary_value = self.pressure_dirichlet(face_centers)
        boundary_partial = (boundary_value - p[owner]) / self.mag_d_f[boundary_faces]
        return bm.set_at(partial_p, boundary_faces, boundary_partial)

    def Interpolation(
        self,
        u,
        ap,
        p,
        face_response_coefficient=None,
        pressure_gradient=None,
    ):
        """Return pressure-stabilized vector face velocity."""
        face_velocity, face_response = self.cell_velocity_to_face(
            u, ap, face_response_coefficient
        )
        grad_diff = self.GradientDifference(p, pressure_gradient=pressure_gradient)
        return face_velocity - face_response * grad_diff
