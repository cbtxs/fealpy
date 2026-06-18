"""Finite-volume Neumann boundary-condition application helpers."""

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry, boundary_face_flag


class NeumannBC:
    """Apply prescribed normal flux data to FVM algebraic systems.

    The reusable path is ``DiffusionApply(f)``, which adds the integrated
    boundary flux contribution to owner cells.
    """

    def __init__(self, mesh, gd=None, threshold=None, geometry=None):
        """Store normal-flux data for later boundary-face integration."""
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold
        self.geometry = geometry

    def DiffusionApply(self, f):
        """Add integrated Neumann fluxes to owner-cell RHS entries.

        ``gd(points)`` is interpreted as the outward normal derivative or flux
        density on boundary face centers.  The finite-volume contribution is
        the face integral ``gd * |f|`` scattered to the boundary owner cells.
        """
        if self.gd is None:
            raise ValueError("NeumannBC.DiffusionApply requires flux data gd.")
        geometry = self.geometry if self.geometry is not None else FVMGeometry(self.mesh)
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[boundary_faces]
        if self.threshold is not None:
            flag = boundary_face_flag(points, self.threshold)
            boundary_faces = boundary_faces[flag]
            points = points[flag]
        neumann = self.gd(points)
        boundary_integrator = neumann * geometry.mag_S_f[boundary_faces]
        boundary_integrator = bm.array(boundary_integrator, dtype=f.dtype)
        f = bm.index_add(
            f,
            geometry.owner[boundary_faces],
            boundary_integrator,
            axis=0,
        )
        return f
