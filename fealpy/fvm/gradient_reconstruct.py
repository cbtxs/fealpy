from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod

from .fvm_geometry import FVMGeometry, face_interpolation_owner_weight


class LSQGradientReconstruct:
    """Least-squares cell-gradient reconstruction and geometry caches."""

    def __init__(self, owner):
        self.owner = owner
        self.mesh = owner.mesh
        self.GD = owner.GD
        self.fvm_geometry = owner.fvm_geometry
        self.clear_cache()

    def clear_cache(self):
        self._extended_lsq_cache_key = None
        self._extended_lsq_cache = None
        self._extended_lsq_dirichlet_cache_key = None
        self._extended_lsq_dirichlet_cache = None
        self._face_lsq_cache_key = None
        self._face_lsq_cache = None
        self._weighted_lsq_cache = None

    def extended_lsq(self, U):
        weights = self.owner.layer_weights
        if self._extended_lsq_cache_key != weights:
            first_weight, second_weight = weights
            NC = self.mesh.number_of_cells()
            c2c = self.mesh.cell_to_cell()
            N = self._extended_lsq_stencil(c2c, NC)
            cell_centers = self.fvm_geometry.cell_center
            d = cell_centers[N] - cell_centers[:, None, :]
            direct_neighbor = bm.any(N[:, :, None] == c2c[:, None, :], axis=2)
            self_neighbor = N == bm.arange(N.shape[0])[:, None]
            sample_weight = bm.where(direct_neighbor, first_weight, second_weight)
            sample_weight = bm.where(self_neighbor, 0.0, sample_weight)
            weighted_d = sample_weight[:, :, None] * d
            A = bm.zeros((NC, self.GD, self.GD), dtype=cell_centers.dtype)
            cells = bm.arange(NC, dtype=N.dtype)
            for k in range(N.shape[1]):
                A = self._add_lsq_matrix_samples(A, cells, d[:, k, :], sample_weight[:, k])
            inv_A = self._invert_lsq_matrix(A, "extended_lsq")
            self._extended_lsq_cache_key = weights
            self._extended_lsq_cache = (N, weighted_d, A, inv_A, cell_centers)
            self._extended_lsq_dirichlet_cache_key = None
            self._extended_lsq_dirichlet_cache = None

        N, weighted_d, A, inv_A, cell_centers = self._extended_lsq_cache
        cells = bm.arange(U.shape[0], dtype=N.dtype)
        if U.ndim == 1:
            b = bm.zeros((U.shape[0], self.GD), dtype=U.dtype)
            for k in range(N.shape[1]):
                b = self._add_lsq_rhs_samples(b, cells, weighted_d[:, k, :], U[N[:, k]] - U)
        else:
            b = bm.zeros((U.shape[0], U.shape[1], self.GD), dtype=U.dtype)
            for k in range(N.shape[1]):
                b = self._add_lsq_rhs_samples(b, cells, weighted_d[:, k, :], U[N[:, k]] - U)

        bc_type = None
        if self.owner.gd is not None:
            bc_type = "dirichlet" if self.owner.bc_type is None else self.owner.bc_type
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(f"Unknown LSQ boundary bc_type: {self.owner.bc_type!r}.")

        if bc_type == "dirichlet":
            boundary_weight = self.owner.boundary_weight
            if boundary_weight != 0.0:
                cache_key = (weights, boundary_weight)
                if self._extended_lsq_dirichlet_cache_key != cache_key:
                    bdedge = self.owner._selected_boundary_faces()
                    bd_owner = self.fvm_geometry.owner[bdedge]
                    face_centers = self.fvm_geometry.face_center
                    bd_d = face_centers[bdedge] - cell_centers[bd_owner]
                    A_dirichlet = self._add_lsq_matrix_samples(
                        bm.copy(A), bd_owner, bd_d, boundary_weight
                    )
                    inv_A_dirichlet = self._invert_lsq_matrix(A_dirichlet, "extended_lsq")
                    self._extended_lsq_dirichlet_cache_key = cache_key
                    self._extended_lsq_dirichlet_cache = (bdedge, bd_owner, bd_d, inv_A_dirichlet)
                bdedge, bd_owner, bd_d, inv_A = self._extended_lsq_dirichlet_cache
                face_centers = self.fvm_geometry.face_center
                bd_value = self.owner.gd(face_centers[bdedge])
                b = self._add_lsq_rhs_samples(
                    b, bd_owner, bd_d, bd_value - U[bd_owner], boundary_weight
                )

        grad = self._solve_lsq_system(A, b, "extended_lsq", inv_A=inv_A)

        if bc_type == "neumann":
            bdedge = self.owner._selected_boundary_faces()
            owner = self.fvm_geometry.owner[bdedge]
            unit_normal = self.fvm_geometry.n_f[bdedge]
            face_centers = self.fvm_geometry.face_center
            bd_value = self.owner.gd(face_centers[bdedge])
            if owner.shape[0] == 0:
                return grad

            constrained_grad = bm.copy(grad)
            owner_list = bm.to_numpy(owner).tolist()
            unique_owner = sorted(set(owner_list))
            for cell in unique_owner:
                local_indices = [
                    index for index, owner_value in enumerate(owner_list)
                    if owner_value == cell
                ]
                if len(local_indices) > self.GD:
                    raise ValueError(
                        "constrained Neumann LSQ supports at most GD independent "
                        "boundary constraints per cell."
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
        return grad

    def face_lsq(self, U):
        if self.GD != 2:
            raise ValueError("face_lsq is currently only implemented for 2D meshes.")
        cache_key = self.owner.gd is not None
        if self._face_lsq_cache_key != cache_key:
            NC = self.mesh.number_of_cells()
            cell_centers = self.fvm_geometry.cell_center
            is_internal = self.fvm_geometry.is_internal
            owner = self.fvm_geometry.owner[is_internal]
            neighbour = self.fvm_geometry.neighbour[is_internal]
            A = bm.zeros((NC, 2, 2), dtype=cell_centers.dtype)
            d = cell_centers[neighbour] - cell_centers[owner]
            A = self._add_lsq_matrix_samples(A, owner, d)
            A = self._add_lsq_matrix_samples(A, neighbour, d)
            bdedge = None
            bd_owner = None
            bd_d = None
            if self.owner.gd is not None:
                if self.owner.bc_type not in (None, "dirichlet"):
                    raise ValueError("face_lsq only accepts Dirichlet boundary values.")
                bdedge = self.owner._selected_boundary_faces()
                bd_owner = self.fvm_geometry.owner[bdedge]
                face_centers = self.fvm_geometry.face_center
                bd_d = face_centers[bdedge] - cell_centers[bd_owner]
                A = self._add_lsq_matrix_samples(A, bd_owner, bd_d)
            inv_A = self._invert_lsq_matrix(A, "face_lsq")
            self._face_lsq_cache_key = cache_key
            self._face_lsq_cache = (owner, neighbour, d, bdedge, bd_owner, bd_d, A, inv_A)

        owner, neighbour, d, bdedge, bd_owner, bd_d, A, inv_A = self._face_lsq_cache
        NC = self.mesh.number_of_cells()
        if U.ndim == 1:
            b = bm.zeros((NC, 2), dtype=U.dtype)
        else:
            b = bm.zeros((NC, U.shape[1], 2), dtype=U.dtype)
        delta_u = U[neighbour] - U[owner]
        b = self._add_lsq_rhs_samples(b, owner, d, delta_u)
        b = self._add_lsq_rhs_samples(b, neighbour, d, delta_u)

        if self.owner.gd is not None:
            face_centers = self.fvm_geometry.face_center
            bd_value = self.owner.gd(face_centers[bdedge])
            b = self._add_lsq_rhs_samples(b, bd_owner, bd_d, bd_value - U[bd_owner])

        return self._solve_lsq_system(A, b, "face_lsq", inv_A=inv_A)

    def weighted_lsq(self, U):
        if self._weighted_lsq_cache is None:
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
            A = bm.zeros((NC, self.GD, self.GD), dtype=cell_centers.dtype)
            d = cell_centers[internal_neighbour] - cell_centers[internal_owner]
            scale = face_measure[internal_face] / bm.einsum("ij,ij->i", d, d)
            owner_weight = face_interpolation_owner_weight(
                self.mesh, method="linear"
            )[internal_face]
            owner_scale = (1.0 - owner_weight) * scale
            neighbour_scale = owner_weight * scale
            A = self._add_lsq_matrix_samples(A, internal_owner, d, owner_scale)
            A = self._add_lsq_matrix_samples(A, internal_neighbour, d, neighbour_scale)

            bdedge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
            bd_owner = owner[bdedge]
            unit_normal = self.fvm_geometry.n_f[bdedge]
            center_to_face = face_centers[bdedge] - cell_centers[bd_owner]
            projected = bm.einsum("ij,ij->i", unit_normal, center_to_face)
            bd_d = unit_normal * projected[:, None]
            bd_scale = face_measure[bdedge] / bm.einsum("ij,ij->i", bd_d, bd_d)
            A = self._add_lsq_matrix_samples(A, bd_owner, bd_d, bd_scale)
            inv_A = self._invert_lsq_matrix(A, "weighted_lsq")
            self._weighted_lsq_cache = (
                internal_owner,
                internal_neighbour,
                d,
                owner_scale,
                neighbour_scale,
                bdedge,
                bd_owner,
                bd_d,
                bd_scale,
                A,
                inv_A,
            )

        (
            internal_owner,
            internal_neighbour,
            d,
            owner_scale,
            neighbour_scale,
            bdedge,
            bd_owner,
            bd_d,
            bd_scale,
            A,
            inv_A,
        ) = self._weighted_lsq_cache
        NC = self.mesh.number_of_cells()
        if U.ndim == 1:
            b = bm.zeros((NC, self.GD), dtype=U.dtype)
        else:
            b = bm.zeros((NC, U.shape[1], self.GD), dtype=U.dtype)

        delta = U[internal_neighbour] - U[internal_owner]
        b = self._add_lsq_rhs_samples(b, internal_owner, d, delta, owner_scale)
        b = self._add_lsq_rhs_samples(b, internal_neighbour, d, delta, neighbour_scale)

        if self.owner.gd is not None:
            bc_type = "dirichlet" if self.owner.bc_type is None else self.owner.bc_type
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(
                    "weighted_lsq accepts only Dirichlet or Neumann boundary data."
                )
            selected = self.owner._selected_boundary_faces()
            selected_owner = self.fvm_geometry.owner[selected]
            points = self.fvm_geometry.face_center[selected]
            selected_normal = self.fvm_geometry.n_f[selected]
            cell_centers = self.fvm_geometry.cell_center
            selected_d = selected_normal * bm.einsum(
                "ij,ij->i",
                selected_normal,
                points - cell_centers[selected_owner],
            )[:, None]
            face_measure = self.fvm_geometry.mag_S_f
            selected_scale = face_measure[selected] / bm.einsum("ij,ij->i", selected_d, selected_d)
            bd_value = self.owner.gd(points)
            if bc_type == "neumann":
                normal_distance = bm.abs(
                    bm.einsum("ij,ij->i", points - cell_centers[selected_owner], selected_normal)
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

        return self._solve_lsq_system(A, b, "weighted_lsq", inv_A=inv_A)

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

        This adds ``b_K = sum w delta_u d`` for scalar or vector fields whose
        components share the same geometric LSQ matrix.
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

    def _solve_lsq_system(self, A, b, method, *, inv_A=None):
        """Solve the per-cell LSQ normal equations ``A_K grad_K = b_K``."""
        if inv_A is None:
            inv_A = self._invert_lsq_matrix(A, method)
        if b.ndim == 2:
            return bm.einsum("nij,nj->ni", inv_A, b)
        return bm.einsum("nij,nkj->nki", inv_A, b)

    def _solve_cell_neumann_constraint(
        self, grad, A, b, bd_value, unit_normal, cell, local_indices
    ):
        local_indices = bm.array(local_indices, dtype=bm.int64)
        normal = unit_normal[local_indices]
        n_constraint = len(local_indices)
        kkt = bm.zeros((self.GD + n_constraint, self.GD + n_constraint), dtype=A.dtype)
        kkt = bm.set_at(kkt, (slice(None, self.GD), slice(None, self.GD)), A[cell])
        kkt = bm.set_at(
            kkt,
            (slice(None, self.GD), slice(self.GD, None)),
            bm.swapaxes(normal, 0, 1),
        )
        kkt = bm.set_at(kkt, (slice(self.GD, None), slice(None, self.GD)), normal)

        if b.ndim == 2:
            rhs = bm.concatenate([
                b[cell],
                bd_value[local_indices],
            ])
            solution = bm.linalg.solve(kkt, rhs[:, None]).squeeze(-1)
            return bm.set_at(grad, cell, solution[:self.GD])

        components = []
        for component in range(b.shape[1]):
            rhs = bm.concatenate([
                b[cell, component],
                bd_value[local_indices, component],
            ])
            solution = bm.linalg.solve(kkt, rhs[:, None]).squeeze(-1)
            components.append(solution[:self.GD])
        return bm.set_at(grad, cell, bm.stack(components, axis=0))

    def _invert_lsq_matrix(self, A, method):
        if self.GD != 2:
            det = bm.linalg.det(A)
            scale = bm.maximum(bm.linalg.norm(A, axis=(1, 2)), bm.ones_like(det))
            if bm.any(bm.abs(det) <= 1.0e-14 * scale**self.GD):
                raise ValueError(f"{method} stencil is rank deficient.")
            return bm.linalg.inv(A)

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


class GreenGaussGradientReconstruct:
    """Green-Gauss cell-gradient reconstruction."""

    def __init__(self, owner):
        self.owner = owner
        self.mesh = owner.mesh
        self.GD = owner.GD
        self.fvm_geometry = owner.fvm_geometry

    def green_gauss(self, U):
        # Green-Gauss is dimension-independent once owner-oriented face
        # geometry is supplied by FVMGeometry.
        if self.owner.gd is not None and self.owner.bc_type is None:
            raise ValueError("bc_type must be set when gd is given.")
        if self.owner.bc_type is not None and self.owner.gd is None:
            raise ValueError("gd must be provided when bc_type is set.")
        if self.owner.bc_type not in (None, "dirichlet", "neumann"):
            raise ValueError(f"Unknown Green-Gauss bc_type: {self.owner.bc_type!r}.")

        cell_measure = self.mesh.entity_measure("cell")
        scalar_field = U.ndim == 1
        NC = self.mesh.number_of_cells()
        if scalar_field:
            grad_U = bm.zeros((NC, self.GD), dtype=U.dtype)
        else:
            grad_U = bm.zeros((NC, U.shape[1], self.GD), dtype=U.dtype)

        is_internal = self.fvm_geometry.is_internal
        owner = self.fvm_geometry.owner[is_internal]
        neighbour = self.fvm_geometry.neighbour[is_internal]
        Sf = self.owner.S_f[is_internal]
        face_value = 0.5 * (U[owner] + U[neighbour])
        if scalar_field:
            flux_grad = face_value[:, None] * Sf
        else:
            flux_grad = face_value[:, :, None] * Sf[:, None, :]
        grad_U = bm.index_add(grad_U, owner, flux_grad, axis=0)
        grad_U = bm.index_add(grad_U, neighbour, flux_grad, axis=0, alpha=-1)

        if self.owner.gd is not None:
            bdedge = self.owner._selected_boundary_faces()
            bd_owner = self.fvm_geometry.owner[bdedge]
            points = self.fvm_geometry.face_center[bdedge]
            if self.owner.bc_type == "dirichlet":
                bd_value = self.owner.gd(points)
            else:
                unit_normal = self.fvm_geometry.n_f[bdedge]
                center_to_face = points - self.fvm_geometry.cell_center[bd_owner]
                normal_distance = bm.abs(bm.einsum("ij,ij->i", center_to_face, unit_normal))
                normal_derivative = self.owner.gd(points)
                if scalar_field:
                    bd_value = U[bd_owner] + normal_derivative * normal_distance
                else:
                    bd_value = U[bd_owner] + normal_derivative * normal_distance[:, None]

            if scalar_field:
                flux_grad = bd_value[:, None] * self.owner.S_f[bdedge]
            else:
                flux_grad = bd_value[:, :, None] * self.owner.S_f[bdedge, None, :]
            grad_U = bm.index_add(grad_U, bd_owner, flux_grad, axis=0)

        if scalar_field:
            return grad_U / cell_measure[:, None]
        return grad_U / cell_measure[:, None, None]


class GradientReconstruct:
    """Variant-based finite-volume gradient reconstruction.

    The class separates two operations:

    - ``cell_gradient(U)`` reconstructs a cell-centered gradient from
      cell-centered values.
    Face-centered gradients are constructed by ``reconstruct_face_gradient`` in
    ``face_gradient.py``.  This class only reconstructs cell-centered gradients.

    Pressure-velocity coupling and Rhie-Chow corrections are deliberately kept
    outside this class.

    Dimension status
    ----------------
    ``extended_lsq``, ``weighted_lsq``, and ``green_gauss`` use the cached
    ``GD = mesh.geo_dimension()`` for gradient dimensions.  The 2D LSQ inverse
    keeps the explicit fast path; other dimensions use the backend batched
    inverse.  ``face_lsq`` is still a 2D historical variant and should be
    generalized only if it becomes part of the production 3D route.

    Future cleanup directions
    -------------------------
    - Performance: keep the current LSQ geometry/inverse caches, then consider
      further reducing ``weighted_lsq`` boundary-geometry recomputation, RHS
      assembly work, and temporary arrays.
    - 3D extension: validate boundary Neumann constraints and ``face_lsq`` on
      three-dimensional control volumes before treating those paths as stable.
    - Boundary semantics: ``gd``/``bc_type``/``threshold`` are inherited from
      the historical manufactured-solution interface.  Revisit them after the
      engineering boundary-condition layer becomes stable.
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
        self.GD = mesh.geo_dimension()
        self.fvm_geometry = FVMGeometry(mesh)
        self.S_f = self.fvm_geometry.S_f
        self.gd = gd
        self.bc_type = bc_type
        self.threshold = threshold

        try:
            if isinstance(layer_weights, (int, float)):
                layer_weights = (float(layer_weights), float(layer_weights))
            else:
                layer_weights = tuple(float(weight) for weight in layer_weights)
        except TypeError as exc:
            raise ValueError("layer_weights must be a scalar or a pair.") from exc
        if len(layer_weights) != 2:
            raise ValueError("layer_weights must contain two values.")
        if layer_weights[0] < 0.0 or layer_weights[1] < 0.0:
            raise ValueError("layer_weights must be non-negative.")
        if layer_weights[0] == 0.0 and layer_weights[1] == 0.0:
            raise ValueError("at least one layer weight must be positive.")
        self.layer_weights = layer_weights

        try:
            boundary_weight = float(boundary_weight)
        except TypeError as exc:
            raise ValueError("boundary_weight must be a scalar.") from exc
        if boundary_weight < 0.0:
            raise ValueError("boundary_weight must be non-negative.")
        self.boundary_weight = boundary_weight

        self.lsq_reconstruct = LSQGradientReconstruct(self)
        self.green_gauss_reconstruct = GreenGaussGradientReconstruct(self)
        self._set_variant(self.cell_gradient, method, "cell_gradient")

    def _set_variant(self, handler, key, name):
        if key is None:
            return
        if key not in handler:
            raise ValueError(f"Unknown {name} variant: {key!r}.")
        handler.set(key)

    @variantmethod("extended_lsq")
    def cell_gradient(self, U):
        return self.lsq_reconstruct.extended_lsq(U)

    @cell_gradient.register("face_lsq")
    def cell_gradient(self, U):
        return self.lsq_reconstruct.face_lsq(U)

    @cell_gradient.register("weighted_lsq")
    def cell_gradient(self, U):
        return self.lsq_reconstruct.weighted_lsq(U)

    @cell_gradient.register("green_gauss")
    def cell_gradient(self, U):
        return self.green_gauss_reconstruct.green_gauss(U)

    def clear_cache(self):
        """Drop geometry caches after changing mesh coordinates or topology."""
        self.lsq_reconstruct.clear_cache()

    @property
    def _extended_lsq_cache(self):
        return self.lsq_reconstruct._extended_lsq_cache

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
        flag = bm.array(threshold(points), dtype=bm.bool)
        if flag.shape == (points.shape[0],):
            return flag
        raise ValueError(
            "threshold must return a boolean array with one entry per boundary face."
        )
