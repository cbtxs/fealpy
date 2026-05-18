from fealpy.backend import backend_manager as bm
from fealpy.sparse import COOTensor


class RhieChowInterpolation:
    """
    Rhie-Chow interpolation to prevent pressure-velocity decoupling
    in collocated grids for incompressible flow simulations.
    """

    def __init__(self, mesh):
        from .gradient_reconstruct import GradientReconstruct
        from .vector_decomposition import VectorDecomposition

        self.mesh = mesh
        self.cm = self.mesh.entity_measure('cell')
        self.NC = mesh.number_of_cells()
        self.NF = mesh.number_of_faces()
        self.cell_centers = mesh.entity_barycenter("cell")
        self.cell_measures = mesh.entity_measure("cell")
        self.edge_to_cell = mesh.edge_to_cell()
        self.cell_to_edge = mesh.cell_to_edge()
        self.gradient_reconstruct = GradientReconstruct(mesh)
        self.e, self.d = VectorDecomposition(mesh).centroid_vector_calculation()

    def Ucell2edge(self,u,ap):
        ap = ap[:self.NC][:,None]
        dp = self.cm[:,None]/ap
        u = bm.stack([u[:self.NC],u[self.NC:]],axis=-1)
        e2c = self.edge_to_cell
        # x = ap[e2c[:,0]]*u[e2c[:,0]]+ap[e2c[:,1]]*u[e2c[:,1]]
        # y = ap[e2c[:,0]]+ap[e2c[:,1]]
        # uf = x/y
        uf = (u[e2c[:,0]]+u[e2c[:,1]])/2
        df = (dp[e2c[:,1]]+dp[e2c[:,0]])/2
        return uf,df

    def GradientDifference(self, p):
        """
        Gradient difference calculation
        """
        e, d = self.e, self.d
        partial_p = (p[self.edge_to_cell[:,1]] - p[self.edge_to_cell[:,0]])/d
        e_cf = e / d[:, None]
        grad_p = self.gradient_reconstruct.LSQ(p)
        overline_grad_p_f = self.gradient_reconstruct.reconstruct(grad_p)
        GradientDifference = (partial_p - bm.einsum('ij,ij->i', overline_grad_p_f, e_cf))[:, None]*e_cf
        return GradientDifference

    def Interpolation(self,u,ap,p):
        """
        Perform Rhie-Chow interpolation
        """
        uf, df = self.Ucell2edge(u,ap)
        grad_diff = self.GradientDifference(p)
        return uf - df*grad_diff


class RhieChowCoupledOperator:
    """
    Rhie-Chow pressure operator for collocated coupled solvers.

    This class assembles the compact pressure-pressure block produced by
    substituting Rhie-Chow face interpolation into the continuity equation.
    It is intended for coupled systems of the form ``[[A, G], [B, LRC]]``.
    """

    def __init__(self, mesh, rho=1.0):
        self.mesh = mesh
        self.rho = rho

    def pressure_stabilization_matrix(self, ap):
        """
        Assemble the compact Rhie-Chow pressure block LRC.

        Parameters
        ----------
        ap
            Momentum diagonal coefficients. For a 2D coupled velocity block,
            the expected layout is ``[a_u, a_v]`` with length ``2*NC``.
            A scalar-cell array of length ``NC`` is also accepted and used for
            both velocity components.
        """
        NC = self.mesh.number_of_cells()
        owner, neighbour, _, _, _, beta = self._internal_pressure_flux_geometry(ap)
        alpha = self.rho * beta
        row = bm.concat([owner, owner, neighbour, neighbour])
        col = bm.concat([owner, neighbour, neighbour, owner])
        values = bm.concat([alpha, -alpha, alpha, -alpha])
        indices = bm.stack([row, col], axis=0)
        return COOTensor(indices, values, spshape=(NC, NC)).tocsr()

    def explicit_pressure_rhs(self, ap, p_old=None):
        """
        Assemble the explicit wide-stencil pressure-gradient correction.

        Passing ``p_old=None`` returns a zero vector.  Otherwise ``p_old`` is
        used to reconstruct the pressure-gradient contribution from the same
        non-orthogonal face split as :meth:`pressure_stabilization_matrix`.
        """
        NC = self.mesh.number_of_cells()
        if p_old is None:
            return bm.zeros(NC)

        mesh = self.mesh
        from fealpy.fvm import GradientReconstruct

        owner, neighbour, is_internal, d_pf, _, beta = (
            self._internal_pressure_flux_geometry(ap)
        )
        grad_p = GradientReconstruct(mesh).LSQ(p_old)
        grad_f = GradientReconstruct(mesh).reconstruct(grad_p)[is_internal]
        d_dot_grad = bm.einsum("ij,ij->i", d_pf, grad_f)
        face_rhs = -self.rho * beta * d_dot_grad

        rhs = bm.zeros(NC)
        bm.add_at(rhs, owner, face_rhs)
        bm.add_at(rhs, neighbour, -face_rhs)
        return rhs

    def explicit_pressure_matrix(self, ap):
        """
        Assemble the matrix form of ``explicit_pressure_rhs(ap, p)``.

        The wide-stencil Rhie-Chow pressure-gradient contribution is linear in
        pressure.  For steady Stokes coupled solves it should be placed on the
        left-hand side instead of being updated by a Picard loop.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        owner, neighbour, _, d_pf, _, beta = self._internal_pressure_flux_geometry(ap)
        n_internal = owner.shape[0]

        coeff_x = -self.rho * beta * d_pf[:, 0]
        coeff_y = -self.rho * beta * d_pf[:, 1]

        face = bm.arange(n_internal, dtype=owner.dtype)
        face_rows = bm.concat([face, face, face, face])
        face_cols = bm.concat([
            2 * owner,
            2 * neighbour,
            2 * owner + 1,
            2 * neighbour + 1,
        ])
        face_values = bm.concat([
            0.5 * coeff_x,
            0.5 * coeff_x,
            0.5 * coeff_y,
            0.5 * coeff_y,
        ])
        face_gradient = COOTensor(
            bm.stack([face_rows, face_cols], axis=0),
            face_values,
            spshape=(n_internal, 2 * NC),
        ).coalesce().tocsr()

        scatter_rows = bm.concat([owner, neighbour])
        scatter_cols = bm.concat([face, face])
        one = bm.ones(n_internal, dtype=face_values.dtype)
        scatter_values = bm.concat([one, -one])
        scatter = COOTensor(
            bm.stack([scatter_rows, scatter_cols], axis=0),
            scatter_values,
            spshape=(NC, n_internal),
        ).coalesce().tocsr()

        return scatter @ face_gradient @ self._lsq_gradient_matrix()

    def assemble_pressure_block(self, ap, p_old=None):
        """Return ``(LRC, bp)`` for the coupled continuity equation."""
        return (
            self.pressure_stabilization_matrix(ap),
            self.explicit_pressure_rhs(ap, p_old),
        )

    def _lsq_gradient_matrix(self):
        """Return the sparse matrix for the cell LSQ gradient operator."""
        mesh = self.mesh
        NC = mesh.number_of_cells()
        c2c = mesh.cell_to_cell()
        N = bm.concatenate((c2c[c2c].reshape(NC, -1), c2c), axis=1)
        N_sorted = bm.sort(N, axis=1)
        dup_mask = bm.zeros_like(N_sorted, dtype=bool)
        dup_mask[:, 1:] = N_sorted[:, 1:] == N_sorted[:, :-1]
        row_broadcast = bm.broadcast_to(
            bm.arange(N.shape[0])[:, None], N_sorted.shape
        )
        N_unique = N_sorted.copy()
        N_unique[dup_mask] = row_broadcast[dup_mask]
        N = bm.sort(N_unique, axis=1)

        cell_centers = mesh.entity_barycenter("cell")
        d = cell_centers[N] - cell_centers[:, None, :]
        A = bm.sum(bm.einsum("hij,hik->hijk", d, d), axis=1)
        det = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        wx = (A[:, 1, 1, None] * d[..., 0] - A[:, 0, 1, None] * d[..., 1]) / det[:, None]
        wy = (-A[:, 1, 0, None] * d[..., 0] + A[:, 0, 0, None] * d[..., 1]) / det[:, None]

        cell = bm.arange(NC, dtype=N.dtype)
        base_rows = bm.repeat(cell, N.shape[1])
        neighbour_cols = N.reshape(-1)
        rows = bm.concat([
            2 * base_rows,
            2 * base_rows + 1,
            2 * cell,
            2 * cell + 1,
        ])
        cols = bm.concat([
            neighbour_cols,
            neighbour_cols,
            cell,
            cell,
        ])
        values = bm.concat([
            wx.reshape(-1),
            wy.reshape(-1),
            -bm.sum(wx, axis=1),
            -bm.sum(wy, axis=1),
        ])
        return COOTensor(
            bm.stack([rows, cols], axis=0),
            values,
            spshape=(2 * NC, NC),
        ).coalesce().tocsr()

    def face_velocity(self, velocity, ap, pressure):
        """
        Compute Rhie-Chow corrected face velocity from cell fields.

        Parameters
        ----------
        velocity
            Cell-centered velocity with shape ``(NC, 2)`` or flattened layout
            ``[u, v]`` with length ``2*NC``.
        ap
            Momentum diagonal coefficients.
        pressure
            Cell-centered pressure.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        edge_to_cell = mesh.edge_to_cell()[:, :2]
        velocity = bm.array(velocity)
        if len(velocity.shape) == 1:
            velocity = bm.stack([velocity[:NC], velocity[NC:2 * NC]], axis=1)

        vf = 0.5 * (velocity[edge_to_cell[:, 0]] + velocity[edge_to_cell[:, 1]])
        is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
        if not bm.any(is_internal):
            return vf

        owner, neighbour, is_internal, d_pf, Sf, beta = (
            self._internal_pressure_flux_geometry(ap)
        )

        from fealpy.fvm import GradientReconstruct

        grad_p = GradientReconstruct(mesh).LSQ(pressure)
        interp_grad = GradientReconstruct(mesh).reconstruct(grad_p)[is_internal]
        jump = pressure[neighbour] - pressure[owner]
        d_dot_grad = bm.einsum("ij,ij->i", d_pf, interp_grad)
        correction_flux = -beta * (jump - d_dot_grad)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        normal_correction = (correction_flux / Sf_dot_Sf)[:, None] * Sf
        vf[is_internal] = vf[is_internal] + normal_correction
        return vf

    def boundary_velocity_rhs(self, boundary_velocity):
        """
        Assemble RHS contribution from known boundary face velocity.

        The continuity equation keeps internal face fluxes on the left-hand
        side. Known boundary flux ``u_b · S_b`` is moved to the right-hand side
        with a minus sign.
        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        bd_face = mesh.boundary_face_index()
        owner = mesh.edge_to_cell()[bd_face, 0]
        Sf = mesh.edge_normal()[bd_face]
        flux = bm.einsum("ij,ij->i", boundary_velocity, Sf)
        rhs = bm.zeros(NC)
        bm.add_at(rhs, owner, -self.rho * flux)
        return rhs

    def _internal_pressure_flux_geometry(self, ap):
        """Return geometry for the internal pressure-response face flux.

        The pressure response vector is ``Q_f = D_f S_f``.  On non-orthogonal
        faces it is split as ``Q_f = beta_f d_f + C_f`` with
        ``beta_f = (Q_f dot S_f)/(d_f dot S_f)``.  The compact RC matrix uses
        ``beta_f (p_N - p_P)`` and the explicit part uses
        ``beta_f d_f dot grad(p)_f`` so all RC paths share one face-flux
        definition.
        """
        mesh = self.mesh
        edge_to_cell = mesh.edge_to_cell()[:, :2]
        is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
        internal_edge_to_cell = edge_to_cell[is_internal]
        owner = internal_edge_to_cell[:, 0]
        neighbour = internal_edge_to_cell[:, 1]
        D = self._cell_momentum_response(ap)
        cell_center = mesh.entity_barycenter("cell")
        Sf = mesh.edge_normal()[is_internal]
        d_pf = cell_center[neighbour] - cell_center[owner]
        Df = 0.5 * (D[owner] + D[neighbour])
        response = Df * Sf
        numerator = bm.einsum("ij,ij->i", response, Sf)
        denominator = bm.einsum("ij,ij->i", Sf, d_pf)
        beta = numerator / denominator
        return owner, neighbour, is_internal, d_pf, Sf, beta

    def _cell_momentum_response(self, ap):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        cell_measure = mesh.entity_measure("cell")
        ap = bm.array(ap)
        if ap.shape[0] == 2 * NC:
            ap_u = ap[:NC]
            ap_v = ap[NC:2 * NC]
        elif ap.shape[0] == NC:
            ap_u = ap
            ap_v = ap
        else:
            raise ValueError("ap must have length NC or 2*NC.")
        return bm.stack([cell_measure / ap_u, cell_measure / ap_v], axis=1)
