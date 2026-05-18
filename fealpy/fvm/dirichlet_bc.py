from inspect import signature

from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags

from .vector_decomposition import VectorDecomposition

class DirichletBC:
    """
    A class to handle Dirichlet boundary conditions for PDEs on a mesh.

    This class provides methods to apply Dirichlet boundary conditions to different terms 
    in a PDE system, including diffusion, threshold-based boundary selection, and 
    divergence terms. It modifies the system matrix and right-hand side vector to 
    incorporate boundary conditions accurately in finite element or finite volume methods.

    Attributes:
        mesh (object): The computational mesh (e.g., QuadrangleMesh) used for discretization.
        gd (callable): Function providing Dirichlet boundary values at given points.
        threshold (callable, optional): Function to select specific boundary cells based on 
            their coordinates or other criteria.
    """

    def __init__(self, mesh, gd, threshold=None):
        """
        Initialize the DirichletBC class with mesh and boundary condition data.

        Args:
            mesh (object): The computational mesh for the PDE domain.
            gd (callable): Function that returns Dirichlet boundary values at given points.
            threshold (callable, optional): Function to identify specific boundary cells.
        """
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def ThresholdApply(self, A, f, uh=None):
        """
        Apply Dirichlet boundary conditions to selected boundary cells based on a threshold.

        This method modifies the system matrix `A` and right-hand side vector `f` by applying 
        Dirichlet boundary conditions to cells selected by the threshold function. It supports 
        selective boundary condition application based on coordinate criteria.

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
        bdFlag_u = bm.zeros(NC)
        bdFlag_u[index] = 1
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

    def DiffusionApply(self, A, b, coef=1.0, threshold=None):
        """
        Apply Dirichlet boundary conditions to the diffusion term.

        This method modifies the system matrix `A` and right-hand side vector `b` to 
        incorporate Dirichlet boundary conditions for the diffusion term, using boundary 
        edge contributions and vector/scalar field handling.

        Args:
            A (sparse matrix): System matrix to be modified.
            b (ndarray): Right-hand side vector to be modified.

        Returns:
            tuple: (A, b)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - b (ndarray): Modified right-hand side vector with boundary contributions.
        """
        bd_edge = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        NC = self.mesh.number_of_cells()
        vector_decomposition = VectorDecomposition(self.mesh)
        _, d = vector_decomposition.centroid_vector_calculation()
        Ef_abs = vector_decomposition.Sor()
        bd_integrator = Ef_abs[bd_edge] / d[bd_edge]
        edge_middle_point = self.mesh.entity_barycenter('face')
        bdedgepoint = edge_middle_point[bd_edge]
        if threshold is not None:
            bd_flag = self._boundary_face_flag(bdedgepoint, threshold)
            bd_edge = bd_edge[bd_flag]
            bd_integrator = bd_integrator[bd_flag]
            bdedgepoint = bdedgepoint[bd_flag]
            coef = self._select_boundary_coef(coef, bd_flag)
        coef = self._normalize_boundary_coef(coef, bd_integrator.shape[0])
        bd_integrator = coef * bd_integrator
        bde2c = e2c[bd_edge, 0]
        # Scalar field: bd_u shape (NE,), 2D vector field: (NE, 2), 3D vector field: (NE, 3)
        bd_u = self.gd(bdedgepoint)[..., None]
        bdIdx = bm.zeros(NC)
        bm.add_at(bdIdx, bde2c, bd_integrator)
        # Determine field dimension (scalar, 2D, or 3D) based on bd_u's second axis
        D = bd_u.shape[1]
        bdIdx = bm.tile(bdIdx, D)
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        if D == 1:
            bd_correct = (bd_integrator[:, None] * bd_u).reshape(-1)
            bm.add_at(b, bde2c, bd_correct)
        else:
            # Remove the extra axis from bd_u for computation
            bd_u = bm.squeeze(bd_u, axis=-1)
            bd_correct = bd_integrator[:, None] * bd_u
            bd_correct = bm.transpose(bd_correct).flatten()
            new_arr = bde2c + NC
            bde2c = bm.concat([bde2c, new_arr])
            bm.add_at(b, bde2c, bd_correct)
        return A, b

    def _boundary_face_flag(self, points, threshold):
        """
        Evaluate a face-based threshold on boundary face centers.

        The threshold convention follows FEALPy's coordinate-selector style:
        ``lambda x: ...`` receives the x-coordinate of boundary face centers,
        ``lambda y: ...`` receives the y-coordinate, and a non-axis name such
        as ``lambda p: ...`` receives the full point array.
        """
        if not callable(threshold):
            raise ValueError("threshold must be a callable boundary face selector.")

        axis = self._threshold_coordinate_axis(threshold)
        if axis is not None:
            if axis >= points.shape[1]:
                raise ValueError(
                    f"threshold requests coordinate axis {axis}, "
                    f"but boundary face centers have dimension {points.shape[1]}."
                )
            return self._validate_boundary_face_flag(
                threshold(points[:, axis]), points.shape[0]
            )

        return self._validate_boundary_face_flag(threshold(points), points.shape[0])

    def _threshold_coordinate_axis(self, threshold):
        """
        Infer whether a threshold wants a single coordinate component.

        This keeps ``DiffusionApply`` compatible with calls such as
        ``threshold=lambda x: ...`` without adding a separate axis argument.
        Unknown parameter names are treated as full-point thresholds.
        """
        try:
            params = list(signature(threshold).parameters.values())
        except (TypeError, ValueError):
            return None

        positional = [
            p for p in params
            if p.kind in (
                p.POSITIONAL_ONLY,
                p.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional) != 1:
            return None

        return {"x": 0, "y": 1, "z": 2}.get(positional[0].name)

    def _validate_boundary_face_flag(self, flag, n_boundary_face):
        flag = bm.array(flag, dtype=bm.bool)
        if flag.shape == (n_boundary_face,):
            return flag
        raise ValueError(
            "threshold must return a boolean array with one entry per boundary face."
        )

    def _select_boundary_coef(self, coef, bd_flag):
        if isinstance(coef, (int, float)):
            return coef

        coef = bm.array(coef)
        if coef.shape == ():
            return coef
        n_boundary_face = bd_flag.shape[0]
        n_selected_face = int(bm.to_numpy(bm.sum(bd_flag)))
        if coef.shape[0] == n_boundary_face:
            return coef[bd_flag]
        if coef.shape[0] == n_selected_face:
            return coef

        raise ValueError(
            "coef must be scalar, selected-boundary-face-wise, or boundary-face-wise."
        )

    def _normalize_boundary_coef(self, coef, n_boundary_face):
        if isinstance(coef, (int, float)):
            return coef

        coef = bm.array(coef)
        if coef.shape == ():
            return coef
        if coef.shape[0] == n_boundary_face:
            return coef

        raise ValueError(
            "coef must be scalar or have one entry per selected boundary face."
        )

    def DivApply(self, b):
        """
        Apply Dirichlet boundary conditions to the divergence term.

        This method modifies the right-hand side vector `b` to account for Dirichlet boundary 
        conditions in the divergence term, incorporating boundary face contributions and 
        vector field normals.

        Args:
            b (ndarray): Right-hand side vector to be modified.

        Returns:
            ndarray: Modified right-hand side vector with boundary contributions.
        """
        NC = self.mesh.number_of_cells()
        n = self.mesh.face_unit_normal()
        facemeasure = self.mesh.entity_measure('face')
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        e2c = self.mesh.edge_to_cell()
        bdedgepoint = edge_middle_point[bd_edge]
        bdSf = (facemeasure[:, None] * n)[bd_edge]  # (bdNE, 2)
        bde2c = e2c[bd_edge, 0]
        # 2D vector field: bd_u shape (bdNE, 2), 3D vector field: (bdNE, 3)
        bd_u = self.gd(bdedgepoint)
        bd_correct = bd_u * bdSf
        bd_correct = bm.transpose(bd_correct).flatten()
        new_arr = bde2c + NC
        bde2c = bm.concat([bde2c, new_arr])
        bm.add_at(b, bde2c, -bd_correct)
        return b

    def ConvectionApply(self, b, coef, threshold=None):
        """
        Apply Dirichlet boundary values to a finite-volume convection RHS.

        The interior convection operator only assembles owner-neighbour face
        contributions. On boundary faces the prescribed value contributes the
        known flux ``-(coef_f · S_f) g_D`` to the owner cell RHS.
        """
        if coef is None:
            return b

        bd_edge = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        NC = self.mesh.number_of_cells()
        Sf = self.mesh.edge_normal()[bd_edge]
        bdedgepoint = self.mesh.entity_barycenter("face")[bd_edge, :]
        flux = self._boundary_convection_flux(coef, bd_edge, Sf)

        if threshold is not None:
            bd_flag = self._boundary_face_flag(bdedgepoint, threshold)
            bd_edge = bd_edge[bd_flag]
            bdedgepoint = bdedgepoint[bd_flag]
            flux = flux[bd_flag]

        bde2c = e2c[bd_edge, 0]
        bd_value = self.gd(bdedgepoint)

        if len(bd_value.shape) == 1:
            bm.add_at(b, bde2c, -flux * bd_value)
            return b

        bd_correct = -flux[:, None] * bd_value
        indices = bm.concat(
            [bde2c + component * NC for component in range(bd_value.shape[1])]
        )
        bm.add_at(b, indices, bm.transpose(bd_correct).flatten())
        return b

    def _boundary_convection_flux(self, coef, bd_edge, Sf):
        coef = bm.array(coef)
        NF = self.mesh.number_of_faces()
        NBD = bd_edge.shape[0]

        if len(coef.shape) == 1:
            if coef.shape[0] == NF:
                return coef[bd_edge]
            if coef.shape[0] == NBD:
                return coef
            raise ValueError(
                "coef must be face-wise or boundary-face-wise when it is scalar."
            )

        if len(coef.shape) == 2:
            if coef.shape[0] == NF:
                boundary_coef = coef[bd_edge]
            elif coef.shape[0] == NBD:
                boundary_coef = coef
            else:
                raise ValueError(
                    "coef must be face-wise or boundary-face-wise when it is vector-valued."
                )
            return bm.einsum("ij,ij->i", boundary_coef, Sf)

        raise ValueError("coef must be a scalar face flux or a vector face field.")
