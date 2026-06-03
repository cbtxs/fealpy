from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod

from .backend_utils import as_backend_array
from .face_interpolation import face_interpolation_owner_weight
from .fvm_geometry import FVMGeometry


class GradientReconstruct:
    """Variant-based finite-volume gradient reconstruction.

    The class separates two operations:

    - ``cell_gradient(U)`` reconstructs a cell-centered gradient from
      cell-centered values.
    Face-centered gradients are constructed by ``reconstruct_face_gradient`` in
    ``face_gradient.py``.  This class only reconstructs cell-centered gradients.

    Pressure-velocity coupling and Rhie-Chow corrections are deliberately kept
    outside this class.

    Notes for future 3D extension
    -----------------------------
    The core ``extended_lsq`` construction is dimension-independent in its
    geometry:

    - the stencil values ``N`` are cell indices;
    - ``d = x_j - x_i`` may have two or three coordinates;
    - ``A = sum(w d d^T)`` naturally becomes ``(NC, GD, GD)``;
    - the RHS accumulation uses ``weighted_d.shape[-1]`` and can also work
      for ``GD = 3``.

    The current implementation is nevertheless a 2D production path.  Before
    using it in 3D, generalize the following pieces:

    - ``_invert_lsq_matrix`` currently uses an explicit 2-by-2 inverse formula.
      Keep this fast path for 2D, but add a 3-by-3 or generic solve branch.
    - constrained Neumann LSQ builds a 2D KKT system and allows at most two
      independent boundary constraints per cell.  In 3D this must use ``GD``
      and allow up to ``GD`` independent constraints.
    - ``face_lsq`` and ``green_gauss`` allocate ``(NC, 2)`` or ``(NC, 2, 2)``
      arrays and must be changed to use the geometric dimension.
    - FEALPy 2D meshes expose control-volume faces through ``edge_*`` mesh
      APIs internally.  This class uses ``FVMGeometry`` as the face-geometry
      adapter, so the 3D path should extend that adapter first.
    """

    def __init__(
        self,
        mesh,
        *,
        method="extended_lsq",
        gd=None,
        bc_type=None,
        threshold=None,
        layer_weights=(1.0, 1.0),
        boundary_weight=1.0,
    ):
        self.mesh = mesh
        self.fvm_geometry = FVMGeometry(mesh)
        self.S_f = self.fvm_geometry.S_f
        self.gd = gd
        self.bc_type = bc_type
        self.threshold = threshold
        self.layer_weights = layer_weights
        self.boundary_weight = boundary_weight
        self._extended_lsq_cache_key = None
        self._extended_lsq_cache = None
        self._set_variant(self.cell_gradient, method, "cell_gradient")

    def _set_variant(self, handler, key, name):
        if key is None:
            return
        if key not in handler:
            raise ValueError(f"Unknown {name} variant: {key!r}.")
        handler.set(key)

    @variantmethod("extended_lsq")
    def cell_gradient(self, U):
        return self._extended_lsq(U)

    @cell_gradient.register("face_lsq")
    def cell_gradient(self, U):
        return self._face_lsq(U)

    @cell_gradient.register("weighted_lsq")
    def cell_gradient(self, U):
        return self._weighted_lsq(U)

    @cell_gradient.register("green_gauss")
    def cell_gradient(self, U):
        return self._green_gauss(U)

    def clear_cache(self):
        """Drop geometry caches after changing mesh coordinates or topology."""
        self._extended_lsq_cache_key = None
        self._extended_lsq_cache = None

    def _normalize_layer_weights(self):
        layer_weights = self.layer_weights
        try:
            if isinstance(layer_weights, (int, float)):
                weights = (float(layer_weights), float(layer_weights))
            else:
                weights = tuple(float(weight) for weight in layer_weights)
        except TypeError as exc:
            raise ValueError("layer_weights must be a scalar or a pair.") from exc

        if len(weights) != 2:
            raise ValueError("layer_weights must contain two values.")
        if weights[0] < 0.0 or weights[1] < 0.0:
            raise ValueError("layer_weights must be non-negative.")
        if weights[0] == 0.0 and weights[1] == 0.0:
            raise ValueError("at least one layer weight must be positive.")
        return weights

    def _normalize_boundary_weight(self):
        try:
            weight = float(self.boundary_weight)
        except TypeError as exc:
            raise ValueError("boundary_weight must be a scalar.") from exc

        if weight < 0.0:
            raise ValueError("boundary_weight must be non-negative.")
        return weight

    def _boundary_bc_type(self):
        return "dirichlet" if self.bc_type is None else self.bc_type

    def _extended_lsq(self, U):
        U = as_backend_array(U)
        N, weighted_d, A, cell_centers = self._extended_lsq_geometry()
        b = self._build_extended_lsq_rhs(U, N, weighted_d)
        bc_type = None
        if self.gd is not None:
            bc_type = self._boundary_bc_type()
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(f"Unknown LSQ boundary bc_type: {self.bc_type!r}.")

        # Dirichlet data is an extra sample value and modifies the LSQ system.
        if bc_type == "dirichlet":
            boundary_weight = self._normalize_boundary_weight()
            if boundary_weight != 0.0:
                A = bm.copy(A)
                A, b = self._add_extended_dirichlet_lsq(
                    A, b, U, cell_centers, boundary_weight
                )

        grad = self._solve_lsq_system(A, b, "extended_lsq")

        # Neumann data constrains the solved gradient's normal component.
        if bc_type == "neumann":
            grad = self._apply_extended_neumann_constraints(grad, A, b)
        return grad

    def _build_extended_lsq_rhs(self, U, N, weighted_d):
        cells = bm.arange(U.shape[0], dtype=N.dtype)
        if U.ndim == 1:
            b = bm.zeros((U.shape[0], weighted_d.shape[-1]), dtype=U.dtype)
            for k in range(N.shape[1]):
                b = self._add_lsq_rhs_samples(
                    b,
                    cells,
                    weighted_d[:, k, :],
                    U[N[:, k]] - U,
                )
            return b

        b = bm.zeros((U.shape[0], U.shape[1], weighted_d.shape[-1]), dtype=U.dtype)
        for k in range(N.shape[1]):
            b = self._add_lsq_rhs_samples(
                b,
                cells,
                weighted_d[:, k, :],
                U[N[:, k]] - U,
            )
        return b

    def _extended_lsq_geometry(self):
        weights = self._normalize_layer_weights()
        if self._extended_lsq_cache_key != weights:
            self._extended_lsq_cache_key = weights
            self._extended_lsq_cache = self._build_extended_lsq_geometry(weights)
        return self._extended_lsq_cache

    def _build_extended_lsq_geometry(self, layer_weights):
        first_weight, second_weight = layer_weights
        NC = self.mesh.number_of_cells()
        c2c = self.mesh.cell_to_cell()
        N = self._extended_lsq_stencil(c2c, NC)
        cell_centers = self.fvm_geometry.cell_center
        d = cell_centers[N] - cell_centers[:, None, :]
        direct_neighbor = bm.any(N[:, :, None] == c2c[:, None, :], axis=2)
        self_neighbor = N == bm.arange(N.shape[0])[:, None]
        weights = bm.where(direct_neighbor, first_weight, second_weight)
        weights = bm.where(self_neighbor, 0.0, weights)
        weighted_d = weights[:, :, None] * d
        A = bm.zeros((NC, d.shape[-1], d.shape[-1]), dtype=cell_centers.dtype)
        cells = bm.arange(NC, dtype=N.dtype)
        for k in range(N.shape[1]):
            A = self._add_lsq_matrix_samples(A, cells, d[:, k, :], weights[:, k])
        return N, weighted_d, A, cell_centers

    def _extended_lsq_stencil(self, c2c, NC):
        N = bm.concatenate((c2c[c2c].reshape(NC, -1), c2c), axis=1)
        N_sorted = bm.sort(N, axis=1)
        dup_mask = bm.zeros_like(N_sorted, dtype=bool)
        dup_mask = bm.set_at(
            dup_mask,
            (slice(None), slice(1, None)),
            N_sorted[:, 1:] == N_sorted[:, :-1],
        )
        row_broadcast = bm.broadcast_to(
            bm.arange(N.shape[0], dtype=N_sorted.dtype)[:, None], N_sorted.shape
        )
        N_unique = bm.copy(N_sorted)
        N_unique = bm.set_at(N_unique, dup_mask, row_broadcast[dup_mask])
        return bm.sort(N_unique, axis=1)

    def _add_lsq_matrix_samples(self, A, cells, d, weight=1.0):
        r"""Scatter LSQ normal-equation matrix samples.

        For cell ``K`` and sample displacement ``d``, least squares minimizes
        ``sum w (grad_K · d - delta_u)^2``.  The normal equation is

            A_K grad_K = b_K,
            A_K = sum w d d^T,
            b_K = sum w delta_u d.

        This helper contributes the ``A_K`` part and is shared by the LSQ
        variants below.
        """
        outer = bm.einsum("ni,nj->nij", d, d)
        if not isinstance(weight, (int, float)):
            weight = bm.array(weight, dtype=d.dtype)
        if getattr(weight, "shape", ()) != ():
            outer = weight[:, None, None] * outer
        else:
            outer = weight * outer
        return bm.index_add(A, cells, outer, axis=0)

    def _add_lsq_rhs_samples(self, b, cells, d, delta_u, weight=1.0):
        r"""Scatter LSQ normal-equation RHS samples.

        This adds the matching right-hand side

            b_K = sum w delta_u d

        for scalar fields or vector fields whose components share the same
        geometric LSQ matrix.
        """
        if delta_u.ndim == 1:
            rhs = delta_u[:, None] * d
        else:
            rhs = delta_u[:, :, None] * d[:, None, :]

        if not isinstance(weight, (int, float)):
            weight = bm.array(weight, dtype=d.dtype)
        if getattr(weight, "shape", ()) != ():
            if delta_u.ndim == 1:
                rhs = weight[:, None] * rhs
            else:
                rhs = weight[:, None, None] * rhs
        else:
            rhs = weight * rhs
        return bm.index_add(b, cells, rhs, axis=0)

    def _solve_lsq_system(self, A, b, method):
        """Solve the per-cell LSQ normal equations ``A_K grad_K = b_K``."""
        inv_A = self._invert_lsq_matrix(A, method)
        if b.ndim == 2:
            return bm.einsum("nij,nj->ni", inv_A, b)
        return bm.einsum("nij,nkj->nki", inv_A, b)

    def _add_extended_dirichlet_lsq(self, A, b, U, cell_centers, boundary_weight):
        bdedge = self._selected_boundary_faces()
        owner = self.fvm_geometry.owner[bdedge]
        face_centers = self.fvm_geometry.face_center
        bd_d = face_centers[bdedge] - cell_centers[owner]
        bd_value = self.gd(face_centers[bdedge])
        bd_delta_u = bd_value - U[owner]
        A = self._add_lsq_matrix_samples(A, owner, bd_d, boundary_weight)
        b = self._add_lsq_rhs_samples(b, owner, bd_d, bd_delta_u, boundary_weight)
        return A, b

    def _extended_neumann_boundary_data(self):
        bdedge = self._selected_boundary_faces()
        owner = self.fvm_geometry.owner[bdedge]
        face_centers = self.fvm_geometry.face_center
        unit_normal = self.fvm_geometry.n_f[bdedge]
        bd_value = self.gd(face_centers[bdedge])
        return owner, unit_normal, bd_value

    def _apply_extended_neumann_constraints(self, grad, A, b):
        owner, unit_normal, bd_value = self._extended_neumann_boundary_data()
        if owner.shape[0] == 0:
            return grad

        # 2D cells can impose at most two independent normal-gradient
        # constraints.  A future 3D version should replace this limit by the
        # geometric dimension and build the KKT system with size GD + n_bc.
        constrained_grad = bm.copy(grad)
        owner_list = bm.to_numpy(owner).tolist()
        unique_owner = sorted(set(owner_list))
        for cell in unique_owner:
            local_indices = [
                index for index, owner_value in enumerate(owner_list)
                if owner_value == cell
            ]
            if len(local_indices) > 2:
                raise ValueError(
                    "constrained Neumann LSQ supports at most two independent "
                    "boundary constraints per cell in 2D."
                )
            constrained_grad = self._solve_cell_neumann_constraint(
                constrained_grad,
                A,
                b,
                bd_value,
                unit_normal,
                cell,
                local_indices,
            )
        return constrained_grad

    def _solve_cell_neumann_constraint(
        self, grad, A, b, bd_value, unit_normal, cell, local_indices
    ):
        # Current KKT block is intentionally 2D.  Generalizing to 3D requires
        # using GD = A.shape[-1] in all slice bounds and returning GD entries.
        local_indices = bm.array(local_indices, dtype=bm.int64)
        normal = unit_normal[local_indices]
        n_constraint = len(local_indices)
        kkt = bm.zeros((2 + n_constraint, 2 + n_constraint), dtype=A.dtype)
        kkt = bm.set_at(kkt, (slice(None, 2), slice(None, 2)), A[cell])
        kkt = bm.set_at(
            kkt,
            (slice(None, 2), slice(2, None)),
            bm.swapaxes(normal, 0, 1),
        )
        kkt = bm.set_at(kkt, (slice(2, None), slice(None, 2)), normal)

        if b.ndim == 2:
            rhs = bm.concatenate([
                b[cell],
                bd_value[local_indices],
            ])
            solution = bm.linalg.solve(kkt, rhs[:, None]).squeeze(-1)
            return bm.set_at(grad, cell, solution[:2])

        components = []
        for component in range(b.shape[1]):
            rhs = bm.concatenate([
                b[cell, component],
                bd_value[local_indices, component],
            ])
            solution = bm.linalg.solve(kkt, rhs[:, None]).squeeze(-1)
            components.append(solution[:2])
        return bm.set_at(grad, cell, bm.stack(components, axis=0))

    def _face_lsq(self, U):
        # This one-layer face-neighbour LSQ is still specialized to 2D below.
        # Use GD = cell_centers.shape[1] before enabling it for 3D meshes.
        NC = self.mesh.number_of_cells()
        cell_centers = self.fvm_geometry.cell_center
        is_internal = self.fvm_geometry.is_internal
        owner = self.fvm_geometry.owner[is_internal]
        neighbour = self.fvm_geometry.neighbour[is_internal]

        A = bm.zeros((NC, 2, 2), dtype=cell_centers.dtype)
        if U.ndim == 1:
            b = bm.zeros((NC, 2), dtype=U.dtype)
        else:
            b = bm.zeros((NC, U.shape[1], 2), dtype=U.dtype)

        d = cell_centers[neighbour] - cell_centers[owner]
        delta_u = U[neighbour] - U[owner]
        A = self._add_lsq_matrix_samples(A, owner, d)
        A = self._add_lsq_matrix_samples(A, neighbour, d)
        b = self._add_lsq_rhs_samples(b, owner, d, delta_u)
        b = self._add_lsq_rhs_samples(b, neighbour, d, delta_u)

        if self.gd is not None:
            if self.bc_type not in (None, "dirichlet"):
                raise ValueError("face_lsq only accepts Dirichlet boundary values.")
            bdedge = self._selected_boundary_faces()
            bd_owner = self.fvm_geometry.owner[bdedge]
            face_centers = self.fvm_geometry.face_center
            bd_d = face_centers[bdedge] - cell_centers[bd_owner]
            bd_value = self.gd(face_centers[bdedge])
            bd_delta_u = bd_value - U[bd_owner]
            A = self._add_lsq_matrix_samples(A, bd_owner, bd_d)
            b = self._add_lsq_rhs_samples(b, bd_owner, bd_d, bd_delta_u)

        return self._solve_lsq_system(A, b, "face_lsq")

    def _weighted_lsq(self, U):
        r"""Weighted face-based LSQ gradient.

        This is the same least-squares normal equation used by the other LSQ
        variants, but each face sample is weighted by a face-measure/distance
        scale and by the linear face-interpolation weight.
        """
        U = as_backend_array(U)
        NC = self.mesh.number_of_cells()
        cell_centers = self.fvm_geometry.cell_center
        face_centers = self.fvm_geometry.face_center
        owner = self.fvm_geometry.owner
        neighbour = self.fvm_geometry.neighbour
        is_internal = self.fvm_geometry.is_internal
        internal_owner = owner[is_internal]
        internal_neighbour = neighbour[is_internal]
        internal_face = bm.nonzero(is_internal)[0]
        face_measure = self.fvm_geometry.mag_S_f

        A = bm.zeros((NC, 2, 2), dtype=cell_centers.dtype)
        if U.ndim == 1:
            b = bm.zeros((NC, 2), dtype=U.dtype)
        else:
            b = bm.zeros((NC, U.shape[1], 2), dtype=U.dtype)

        d = cell_centers[internal_neighbour] - cell_centers[internal_owner]
        scale = face_measure[internal_face] / bm.einsum("ij,ij->i", d, d)
        owner_weight = face_interpolation_owner_weight(
            self.mesh,
            method="linear",
        )[internal_face]
        A = self._add_lsq_matrix_samples(
            A,
            internal_owner,
            d,
            (1.0 - owner_weight) * scale,
        )
        A = self._add_lsq_matrix_samples(
            A,
            internal_neighbour,
            d,
            owner_weight * scale,
        )
        delta = U[internal_neighbour] - U[internal_owner]
        b = self._add_lsq_rhs_samples(
            b,
            internal_owner,
            d,
            delta,
            (1.0 - owner_weight) * scale,
        )
        b = self._add_lsq_rhs_samples(
            b,
            internal_neighbour,
            d,
            delta,
            owner_weight * scale,
        )

        bdedge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        bd_owner = owner[bdedge]
        unit_normal = self.fvm_geometry.n_f[bdedge]
        center_to_face = face_centers[bdedge] - cell_centers[bd_owner]
        projected = bm.einsum("ij,ij->i", unit_normal, center_to_face)
        bd_d = unit_normal * projected[:, None]
        bd_scale = face_measure[bdedge] / bm.einsum("ij,ij->i", bd_d, bd_d)
        A = self._add_lsq_matrix_samples(A, bd_owner, bd_d, bd_scale)

        if self.gd is not None:
            bc_type = self._boundary_bc_type()
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(
                    "weighted_lsq accepts only Dirichlet or Neumann boundary data."
                )

            selected = self._selected_boundary_faces()
            selected_owner = owner[selected]
            points = face_centers[selected]
            selected_normal = self.fvm_geometry.n_f[selected]
            selected_d = selected_normal * bm.einsum(
                "ij,ij->i",
                selected_normal,
                points - cell_centers[selected_owner],
            )[:, None]
            selected_scale = (
                face_measure[selected] / bm.einsum("ij,ij->i", selected_d, selected_d)
            )
            bd_value = self.gd(points)
            if bc_type == "neumann":
                normal_distance = bm.abs(
                    bm.einsum(
                        "ij,ij->i",
                        points - cell_centers[selected_owner],
                        selected_normal,
                    )
                )
                if U.ndim == 1:
                    bd_value = U[selected_owner] + bd_value * normal_distance
                else:
                    bd_value = U[selected_owner] + bd_value * normal_distance[:, None]
            b = self._add_lsq_rhs_samples(
                b,
                selected_owner,
                selected_d,
                bd_value - U[selected_owner],
                selected_scale,
            )

        return self._solve_lsq_system(A, b, "weighted_lsq")

    def _invert_lsq_matrix(self, A, method):
        a00 = A[:, 0, 0]
        a01 = A[:, 0, 1]
        a10 = A[:, 1, 0]
        a11 = A[:, 1, 1]
        det = a00 * a11 - a01 * a10
        trace_scale = bm.abs(a00) + bm.abs(a11)
        scale = bm.maximum(trace_scale, bm.ones_like(trace_scale))
        if bm.any(bm.abs(det) <= 1.0e-14 * scale * scale):
            raise ValueError(f"{method} stencil is rank deficient.")
        inv_A = bm.zeros_like(A)
        inv_A = bm.set_at(inv_A, (slice(None), 0, 0), a11 / det)
        inv_A = bm.set_at(inv_A, (slice(None), 0, 1), -a01 / det)
        inv_A = bm.set_at(inv_A, (slice(None), 1, 0), -a10 / det)
        inv_A = bm.set_at(inv_A, (slice(None), 1, 1), a00 / det)
        return inv_A

    def _green_gauss(self, U):
        # Green-Gauss is mathematically dimension-independent, but this
        # implementation allocates 2D gradient arrays and depends on 2D
        # ``edge_*`` face geometry.  Generalize the allocations and geometry
        # adapter before using this variant in 3D.
        if self.gd is not None and self.bc_type is None:
            raise ValueError("bc_type must be set when gd is given.")
        if self.bc_type is not None and self.gd is None:
            raise ValueError("gd must be provided when bc_type is set.")
        if self.bc_type not in (None, "dirichlet", "neumann"):
            raise ValueError(f"Unknown Green-Gauss bc_type: {self.bc_type!r}.")

        U = as_backend_array(U)
        cell_measure = self.mesh.entity_measure("cell")
        scalar_field = U.ndim == 1
        NC = self.mesh.number_of_cells()
        if scalar_field:
            grad_U = bm.zeros((NC, 2), dtype=U.dtype)
        else:
            grad_U = bm.zeros((NC, U.shape[1], 2), dtype=U.dtype)

        grad_U = self._add_internal_green_gauss(grad_U, U, scalar_field)
        if self.gd is not None:
            grad_U = self._add_boundary_green_gauss(
                grad_U, U, scalar_field, cell_measure
            )

        if scalar_field:
            return grad_U / cell_measure[:, None]
        return grad_U / cell_measure[:, None, None]

    def _add_internal_green_gauss(self, grad_U, U, scalar_field):
        is_internal = self.fvm_geometry.is_internal
        owner = self.fvm_geometry.owner[is_internal]
        neighbour = self.fvm_geometry.neighbour[is_internal]
        Sf = self.S_f[is_internal]
        if scalar_field:
            face_value = 0.5 * (U[owner] + U[neighbour])
            flux_grad = face_value[:, None] * Sf
        else:
            face_value = 0.5 * (U[owner] + U[neighbour])
            flux_grad = face_value[:, :, None] * Sf[:, None, :]
        grad_U = bm.index_add(grad_U, owner, flux_grad, axis=0)
        grad_U = bm.index_add(grad_U, neighbour, flux_grad, axis=0, alpha=-1)
        return grad_U

    def _add_boundary_green_gauss(self, grad_U, U, scalar_field, cell_measure):
        bdedge = self._selected_boundary_faces()
        owner = self.fvm_geometry.owner[bdedge]
        face_centers = self.fvm_geometry.face_center
        points = face_centers[bdedge]
        if self.bc_type == "dirichlet":
            bd_value = self.gd(points)
        else:
            unit_normal = self.fvm_geometry.n_f[bdedge]
            cell_centers = self.fvm_geometry.cell_center
            center_to_face = points - cell_centers[owner]
            normal_distance = bm.abs(
                bm.einsum("ij,ij->i", center_to_face, unit_normal)
            )
            normal_derivative = self.gd(points)
            if scalar_field:
                bd_value = U[owner] + normal_derivative * normal_distance
            else:
                bd_value = U[owner] + normal_derivative * normal_distance[:, None]

        if scalar_field:
            flux_grad = bd_value[:, None] * self.S_f[bdedge]
        else:
            flux_grad = bd_value[:, :, None] * self.S_f[bdedge, None, :]
        grad_U = bm.index_add(grad_U, owner, flux_grad, axis=0)
        return grad_U

    def _selected_boundary_faces(self):
        bdedge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        if self.threshold is None:
            return bdedge
        face_centers = self.fvm_geometry.face_center[bdedge]
        flag = self._boundary_face_flag(face_centers, self.threshold)
        return bdedge[flag]

    def _boundary_face_flag(self, points, threshold):
        if not callable(threshold):
            raise ValueError("threshold must be a callable boundary face selector.")
        flag = as_backend_array(threshold(points), dtype=bm.bool)
        if flag.shape == (points.shape[0],):
            return flag
        raise ValueError(
            "threshold must return a boolean array with one entry per boundary face."
        )
