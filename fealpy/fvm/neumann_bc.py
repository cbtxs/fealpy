"""Finite-volume Neumann boundary-condition application helpers."""

from fealpy.sparse import spdiags
from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


class NeumannBC:
    """Apply prescribed normal flux data to FVM algebraic systems.

    The main reusable path is ``DiffusionApply(f)``, which adds the integrated
    boundary flux contribution to owner cells.  The threshold matrix path is a
    legacy value-pinning helper kept for compatibility with existing solver
    experiments; it should not be interpreted as the mathematical Neumann flux
    operator.
    """

    def __init__(self, mesh, gd=None, threshold=None):
        """Store normal-flux data for later boundary-face integration."""
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def ThresholdApply(self, A, f, uh=None):
        """Pin selected boundary-cell values by a threshold.

        This is a compatibility helper with the same algebraic structure as a
        value Dirichlet application.  Use ``DiffusionApply`` for the finite-
        volume Neumann flux contribution.

        Args:
            A (sparse matrix): System matrix to be modified.
            f (ndarray): Right-hand side vector to be modified.
            uh (ndarray, optional): Solution vector to store boundary values. If None, initialized as zeros.

        Returns:
            tuple: (A, f)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - f (ndarray): Modified right-hand side vector with boundary contributions.

        Raises:
            ValueError: If threshold is not a callable function.
        """
        total_bd_idx = self.mesh.boundary_cell_index()
        points = self.mesh.entity_barycenter('cell')
        NC = self.mesh.number_of_cells()
        bd_node = points[total_bd_idx]
        if callable(self.threshold):
            try:
                # Try applying condition to x-coordinate only
                x = bd_node[:, 0]
                bd_idx = self.threshold(x)
                bd_idx = bm.array(bd_idx, dtype=bm.bool)
                if not bm.any(bd_idx):  # Check if bd_idx is all False
                    y = bd_node[:, 1]
                    bd_idx = self.threshold(y)
                    bd_idx = bm.array(bd_idx, dtype=bm.bool)
            except Exception:
                # Fall back to applying condition to full node coordinates
                bd_idx = self.threshold(bd_node)
                bd_idx = bm.array(bd_idx, dtype=bm.bool)
        else:
            raise ValueError("self.threshold must be a callable (e.g., lambda x: (x==0.5)|(x==2.5) or a function).")
        index = total_bd_idx[bd_idx]
        bdFlag_u = bm.zeros(NC, dtype=getattr(f, "dtype", None))
        bdFlag_u = bm.set_at(bdFlag_u, index, 1)
        D0 = spdiags(1 - bdFlag_u, 0, A.shape[0], A.shape[0])  # Keeps interior equations
        D1 = spdiags(bdFlag_u, 0, A.shape[0], A.shape[0])      # Identity on boundary nodes
        # Apply boundary conditions to the matrix
        if uh is None:
            # Initialize uh as a zero vector if not provided
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        uh = bm.set_at(uh, index, self.gd(points[index]))
        f = f - A @ uh
        f = bm.set_at(f, index, uh[index])
        A = D0.matmul(A.matmul(D0)) + D1
        return A, f
    
    def DiffusionApply(self, f):
        """Add integrated Neumann fluxes to owner-cell RHS entries.

        ``gd(points)`` is interpreted as the outward normal derivative or flux
        density on boundary face centers.  The finite-volume contribution is
        the face integral ``gd * |f|`` scattered to the boundary owner cells.
        """
        if self.gd is None:
            raise ValueError("NeumannBC.DiffusionApply requires flux data gd.")
        geometry = FVMGeometry(self.mesh)
        bdedge = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[bdedge]
        neumann = self.gd(points)
        bd_integrator = neumann * geometry.mag_S_f[bdedge]
        bd_integrator = bm.array(bd_integrator, dtype=f.dtype)
        f = bm.index_add(f, geometry.owner[bdedge], bd_integrator, axis=0)
        return f

    def ConvectionApplyX(self, A, b):
        """Legacy x-component boundary matrix helper for RC experiments.

        This is not a general Neumann flux operator.  It only adds the x-normal
        boundary contribution to a pressure-velocity coupling matrix used by the
        older coupled RC model path.
        """

        NC = self.mesh.number_of_cells()
        geometry = FVMGeometry(self.mesh)
        Sf = geometry.S_f
        bdIdx = bm.zeros(NC, dtype=Sf.dtype)
        bdedge = bm.nonzero(geometry.is_boundary)[0]
        bde2c = geometry.owner[bdedge]
        bdIdx = bm.index_add(bdIdx, bde2c, Sf[bdedge, 0], axis=0)
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0

        return A

    def ConvectionApplyY(self, A, b):
        """Legacy y-component boundary matrix helper for RC experiments.

        This is not a general Neumann flux operator.  It only adds the y-normal
        boundary contribution to a pressure-velocity coupling matrix used by the
        older coupled RC model path.
        """

        NC = self.mesh.number_of_cells()
        geometry = FVMGeometry(self.mesh)
        Sf = geometry.S_f
        bdIdx = bm.zeros(NC, dtype=Sf.dtype)
        bdedge = bm.nonzero(geometry.is_boundary)[0]
        bde2c = geometry.owner[bdedge]
        bdIdx = bm.index_add(bdIdx, bde2c, Sf[bdedge, 1], axis=0)
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0

        return A
