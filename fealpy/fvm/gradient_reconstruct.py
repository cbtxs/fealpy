from fealpy.backend import backend_manager as bm
from fealpy.decorator.variantmethod import variantmethod

from .fvm_geometry import (
    FVMGeometry,
    face_interpolation_owner_weight,
    selected_boundary_faces,
)


def least_squares_rhs(U, stencil, weighted_d):
    r"""Return the LSQ normal-equation RHS from a fixed cell stencil.

    ``weighted_d[i, k]`` stores ``w_{ik} d_{ik}``, so the returned RHS is

    .. math::
        b_i = \sum_k (u_{N_{ik}} - u_i) w_{ik} d_{ik}.

    The same geometric stencil is applied component-wise for vector fields.
    """
    if U.ndim == 1:
        return bm.einsum("ns,nsg->ng", U[stencil] - U[:, None], weighted_d)
    return bm.einsum("nsc,nsg->ncg", U[stencil] - U[:, None, :], weighted_d)


class LSQGradientReconstruct:
    """Least-squares cell-gradient reconstruction and geometry caches."""

    def __init__(self, owner):
        self.owner = owner
        self.mesh = owner.mesh
        self.GD = owner.GD
        self.fvm_geometry = owner.fvm_geometry
        self.clear_cache()

    def clear_cache(self):
        self._layered_lsq_cache_key = None
        self._layered_lsq_cache = None
        self._layered_lsq_dirichlet_cache_key = None
        self._layered_lsq_dirichlet_cache = None
        self._face_weighted_lsq_cache = None
        self._face_weighted_lsq_boundary_cache_key = None
        self._face_weighted_lsq_boundary_cache = None

    def layered_lsq(self, U):
        weights = self.owner.layer_weights
        if self._layered_lsq_cache_key != weights:
            first_weight, second_weight = weights
            NC = self.mesh.number_of_cells()
            c2c = self.padded_cell_neighbors(NC)
            N = self.layered_lsq_stencil(c2c, NC)
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
                A = self.add_lsq_matrix_samples(A, cells, d[:, k, :], sample_weight[:, k])
            bc_type = None if self.owner.boundary_value is None else (
                "dirichlet" if self.owner.boundary_type is None else self.owner.boundary_type
            )
            use_dirichlet_samples = (
                bc_type == "dirichlet" and self.owner.boundary_weight != 0.0
            )
            inv_A = (
                None
                if use_dirichlet_samples
                else self.invert_lsq_matrix(A, "layered_lsq")
            )
            self._layered_lsq_cache_key = weights
            self._layered_lsq_cache = (N, weighted_d, A, inv_A, cell_centers)
            self._layered_lsq_dirichlet_cache_key = None
            self._layered_lsq_dirichlet_cache = None

        N, weighted_d, A, inv_A, cell_centers = self._layered_lsq_cache
        b = least_squares_rhs(U, N, weighted_d)

        bc_type = None
        if self.owner.boundary_value is not None:
            bc_type = "dirichlet" if self.owner.boundary_type is None else self.owner.boundary_type
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(
                    f"Unknown LSQ boundary_type: {self.owner.boundary_type!r}."
                )

        if bc_type == "dirichlet":
            boundary_weight = self.owner.boundary_weight
            if boundary_weight != 0.0:
                cache_key = (weights, boundary_weight)
                if self._layered_lsq_dirichlet_cache_key != cache_key:
                    boundary_faces = selected_boundary_faces(
                        self.fvm_geometry,
                        self.owner.boundary_threshold,
                        default_all=True,
                    )
                    bd_owner = self.fvm_geometry.owner[boundary_faces]
                    face_centers = self.fvm_geometry.face_center
                    bd_d = face_centers[boundary_faces] - cell_centers[bd_owner]
                    A_dirichlet = self.add_lsq_matrix_samples(
                        bm.copy(A), bd_owner, bd_d, boundary_weight
                    )
                    inv_A_dirichlet = self.invert_lsq_matrix(A_dirichlet, "layered_lsq")
                    self._layered_lsq_dirichlet_cache_key = cache_key
                    self._layered_lsq_dirichlet_cache = (
                        boundary_faces,
                        bd_owner,
                        bd_d,
                        inv_A_dirichlet,
                    )
                boundary_faces, bd_owner, bd_d, inv_A = self._layered_lsq_dirichlet_cache
                face_centers = self.fvm_geometry.face_center
                bd_value = self.owner.boundary_value(face_centers[boundary_faces])
                b = self.add_lsq_rhs_samples(
                    b, bd_owner, bd_d, bd_value - U[bd_owner], boundary_weight
                )

        grad = self.solve_lsq_system(A, b, "layered_lsq", inv_A=inv_A)

        if bc_type == "neumann":
            boundary_faces = selected_boundary_faces(
                self.fvm_geometry,
                self.owner.boundary_threshold,
                default_all=True,
            )
            owner = self.fvm_geometry.owner[boundary_faces]
            unit_normal = self.fvm_geometry.n_f[boundary_faces]
            face_centers = self.fvm_geometry.face_center
            bd_value = self.owner.boundary_value(face_centers[boundary_faces])
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
                constrained_grad = self.solve_cell_neumann_constraint(
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

    def padded_cell_neighbors(self, NC):
        """Return a dense neighbour stencil for fixed or variable face counts."""
        faces_per_cell = self.mesh.number_of_faces_of_cells()
        if getattr(faces_per_cell, "ndim", 0) == 0:
            return self.mesh.cell_to_cell()

        max_faces = int(bm.to_numpy(bm.max(faces_per_cell)))
        cells = bm.arange(NC, dtype=self.mesh.itype)
        cell_to_cell = bm.broadcast_to(cells[:, None], (NC, max_faces))
        cell_to_cell = bm.copy(cell_to_cell)
        face_to_cell = self.fvm_geometry.face_to_cell
        owner_local_face = self.fvm_geometry.owner_local_face
        neighbour_local_face = self.fvm_geometry.neighbour_local_face
        if owner_local_face is None or neighbour_local_face is None:
            raise RuntimeError(
                "variable-face cell stencils require local face indices in "
                "FVMGeometry."
            )
        cell_to_cell = bm.set_at(
            cell_to_cell,
            (face_to_cell[:, 0], owner_local_face),
            face_to_cell[:, 1],
        )
        cell_to_cell = bm.set_at(
            cell_to_cell,
            (face_to_cell[:, 1], neighbour_local_face),
            face_to_cell[:, 0],
        )
        return cell_to_cell

    def face_weighted_lsq(self, U):
        if self._face_weighted_lsq_cache is None:
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
                self.mesh, method="linear", geometry=self.fvm_geometry
            )[internal_face]
            owner_scale = (1.0 - owner_weight) * scale
            neighbour_scale = owner_weight * scale
            A = self.add_lsq_matrix_samples(A, internal_owner, d, owner_scale)
            A = self.add_lsq_matrix_samples(A, internal_neighbour, d, neighbour_scale)
            rhs_cells = bm.concatenate((internal_owner, internal_neighbour))
            rhs_d = bm.concatenate((d, d), axis=0)
            rhs_weight = bm.concatenate((owner_scale, neighbour_scale))
            rhs_sample = bm.concatenate((
                bm.arange(internal_owner.shape[0], dtype=internal_owner.dtype),
                bm.arange(internal_owner.shape[0], dtype=internal_owner.dtype),
            ))

            boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
            bd_owner = owner[boundary_faces]
            unit_normal = self.fvm_geometry.n_f[boundary_faces]
            center_to_face = face_centers[boundary_faces] - cell_centers[bd_owner]
            projected = bm.einsum("ij,ij->i", unit_normal, center_to_face)
            bd_d = unit_normal * projected[:, None]
            bd_scale = face_measure[boundary_faces] / bm.einsum("ij,ij->i", bd_d, bd_d)
            A = self.add_lsq_matrix_samples(A, bd_owner, bd_d, bd_scale)
            inv_A = self.invert_lsq_matrix(A, "face_weighted_lsq")
            self._face_weighted_lsq_cache = (
                internal_owner,
                internal_neighbour,
                d,
                rhs_cells,
                rhs_d,
                rhs_weight,
                rhs_sample,
                boundary_faces,
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
            rhs_cells,
            rhs_d,
            rhs_weight,
            rhs_sample,
            boundary_faces,
            bd_owner,
            bd_d,
            bd_scale,
            A,
            inv_A,
        ) = self._face_weighted_lsq_cache
        NC = self.mesh.number_of_cells()
        if U.ndim == 1:
            b = bm.zeros((NC, self.GD), dtype=U.dtype)
        else:
            b = bm.zeros((NC, U.shape[1], self.GD), dtype=U.dtype)

        delta = U[internal_neighbour] - U[internal_owner]
        b = self.add_lsq_rhs_samples(
            b,
            rhs_cells,
            rhs_d,
            delta[rhs_sample],
            rhs_weight,
        )

        if self.owner.boundary_value is not None:
            bc_type = "dirichlet" if self.owner.boundary_type is None else self.owner.boundary_type
            if bc_type not in ("dirichlet", "neumann"):
                raise ValueError(
                    "face_weighted_lsq accepts only Dirichlet or Neumann boundary data."
                )
            cache_key = (bc_type, self.owner.boundary_threshold)
            if self._face_weighted_lsq_boundary_cache_key != cache_key:
                selected = selected_boundary_faces(
                    self.fvm_geometry,
                    self.owner.boundary_threshold,
                    default_all=True,
                )
                selected_owner = self.fvm_geometry.owner[selected]
                points = self.fvm_geometry.face_center[selected]
                selected_normal = self.fvm_geometry.n_f[selected]
                cell_centers = self.fvm_geometry.cell_center
                cell_to_face = points - cell_centers[selected_owner]
                normal_distance = bm.abs(
                    bm.einsum("ij,ij->i", cell_to_face, selected_normal)
                )
                selected_d = selected_normal * bm.einsum(
                    "ij,ij->i",
                    selected_normal,
                    cell_to_face,
                )[:, None]
                face_measure = self.fvm_geometry.mag_S_f
                selected_scale = face_measure[selected] / bm.einsum(
                    "ij,ij->i", selected_d, selected_d
                )
                self._face_weighted_lsq_boundary_cache_key = cache_key
                self._face_weighted_lsq_boundary_cache = (
                    selected_owner,
                    points,
                    selected_d,
                    selected_scale,
                    normal_distance,
                )
            (
                selected_owner,
                points,
                selected_d,
                selected_scale,
                normal_distance,
            ) = self._face_weighted_lsq_boundary_cache
            bd_value = self.owner.boundary_value(points)
            if bc_type == "neumann":
                if U.ndim == 1:
                    bd_value = U[selected_owner] + bd_value * normal_distance
                else:
                    bd_value = U[selected_owner] + bd_value * normal_distance[:, None]
            b = self.add_lsq_rhs_samples(
                b,
                selected_owner,
                selected_d,
                bd_value - U[selected_owner],
                selected_scale,
            )

        return self.solve_lsq_system(A, b, "face_weighted_lsq", inv_A=inv_A)

    def layered_lsq_stencil(self, c2c, NC):
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

    def add_lsq_matrix_samples(self, A, cells, d, weight=1.0):
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

    def add_lsq_rhs_samples(self, b, cells, d, delta_u, weight=1.0):
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

    def solve_lsq_system(self, A, b, method, *, inv_A=None):
        """Solve the per-cell LSQ normal equations ``A_K grad_K = b_K``."""
        if inv_A is None:
            inv_A = self.invert_lsq_matrix(A, method)
        if b.ndim == 2:
            return bm.einsum("nij,nj->ni", inv_A, b)
        return bm.einsum("nij,nkj->nki", inv_A, b)

    def solve_cell_neumann_constraint(
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

    def invert_lsq_matrix(self, A, method):
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


class QuadraticLSQGradientReconstruct:
    r"""Quadratic k-exact gradients for cell-average finite-volume fields.

    Around cell centroid ``x_P`` the reconstructed polynomial is written as

    ``u_h = u_P + g_P r + 1/2 H_P : (r r - M_P)``,

    where ``M_P`` is the cell-average second central moment.  Neighbour-cell
    equations therefore use ``d d + M_N - M_P`` in their quadratic columns.
    This distinction is required because FVM unknowns are cell averages, not
    point samples at cell centroids.
    """

    def __init__(self, owner):
        self.owner = owner
        self.mesh = owner.mesh
        self.GD = owner.GD
        self.fvm_geometry = owner.fvm_geometry
        self.clear_cache()

    def clear_cache(self):
        self._cache = None

    def _cell_second_moment(self):
        center = self.fvm_geometry.cell_center

        def integrand(points, index=None):
            local_center = center if index is None else center[index]
            delta = points - local_center[:, None, :]
            return bm.einsum("cqi,cqj->cqij", delta, delta)

        measure = self.mesh.entity_measure("cell")
        return self.mesh.integral(integrand, q=3, celltype=True) / measure[:, None, None]

    def _features(self, displacement, moment_difference, scale):
        scaled_d = displacement / scale[..., None]
        scaled_moment = moment_difference / scale[..., None, None] ** 2
        columns = [scaled_d[..., component] for component in range(self.GD)]
        for first in range(self.GD):
            for second in range(first, self.GD):
                value = (
                    scaled_d[..., first] * scaled_d[..., second]
                    + scaled_moment[..., first, second]
                )
                if first == second:
                    value = 0.5 * value
                columns.append(value)
        return bm.stack(columns, axis=-1)

    def _unique_stencil(self, stencil, NC):
        sorted_stencil = bm.sort(stencil, axis=1)
        duplicate = bm.zeros_like(sorted_stencil, dtype=bm.bool)
        duplicate = bm.set_at(
            duplicate,
            (slice(None), slice(1, None)),
            sorted_stencil[:, 1:] == sorted_stencil[:, :-1],
        )
        cells = bm.broadcast_to(
            bm.arange(NC, dtype=sorted_stencil.dtype)[:, None],
            sorted_stencil.shape,
        )
        unique = bm.set_at(bm.copy(sorted_stencil), duplicate, cells[duplicate])
        return bm.sort(unique, axis=1)

    def _build_cache(self):
        NC = self.mesh.number_of_cells()
        c2c = self.owner.lsq_reconstruct.padded_cell_neighbors(NC)
        second = c2c[c2c].reshape(NC, -1)
        third = c2c[second].reshape(NC, -1)
        stencil = self._unique_stencil(
            bm.concatenate((c2c, second, third), axis=1), NC
        )

        center = self.fvm_geometry.cell_center
        moment = self._cell_second_moment()
        displacement = center[stencil] - center[:, None, :]
        distance = bm.linalg.norm(displacement, axis=-1)
        active = stencil != bm.arange(NC, dtype=stencil.dtype)[:, None]
        characteristic = bm.max(distance, axis=1)
        if bm.any(characteristic <= 0.0):
            raise ValueError("quadratic_lsq stencil has zero diameter.")

        moment_difference = moment[stencil] - moment[:, None, :, :]
        feature = self._features(
            displacement,
            moment_difference,
            characteristic[:, None],
        )
        scaled_distance = distance / characteristic[:, None]
        sample_weight = bm.where(
            active,
            1.0 / bm.maximum(scaled_distance, 1.0e-14) ** 2,
            0.0,
        )
        normal_matrix = bm.einsum(
            "ns,nsi,nsj->nij", sample_weight, feature, feature
        )

        boundary_data = None
        bc_type = None if self.owner.boundary_value is None else (
            "dirichlet" if self.owner.boundary_type is None else self.owner.boundary_type
        )
        if bc_type not in (None, "dirichlet"):
            raise ValueError("quadratic_lsq currently supports Dirichlet data only.")
        if bc_type == "dirichlet" and self.owner.boundary_weight != 0.0:
            boundary_faces = selected_boundary_faces(
                self.fvm_geometry,
                self.owner.boundary_threshold,
                default_all=True,
            )
            boundary_owner = self.fvm_geometry.owner[boundary_faces]
            boundary_points = self.fvm_geometry.face_center[boundary_faces]
            boundary_displacement = boundary_points - center[boundary_owner]
            boundary_moment_difference = -moment[boundary_owner]
            boundary_scale = characteristic[boundary_owner]
            boundary_feature = self._features(
                boundary_displacement,
                boundary_moment_difference,
                boundary_scale,
            )
            boundary_distance = (
                bm.linalg.norm(boundary_displacement, axis=-1) / boundary_scale
            )
            boundary_weight = self.owner.boundary_weight / bm.maximum(
                boundary_distance, 1.0e-14
            ) ** 2
            boundary_outer = bm.einsum(
                "n,ni,nj->nij",
                boundary_weight,
                boundary_feature,
                boundary_feature,
            )
            normal_matrix = bm.index_add(
                normal_matrix, boundary_owner, boundary_outer, axis=0
            )
            boundary_data = (
                boundary_faces,
                boundary_owner,
                boundary_points,
                boundary_feature,
                boundary_weight,
            )

        determinant = bm.linalg.det(normal_matrix)
        matrix_scale = bm.maximum(
            bm.linalg.norm(normal_matrix, axis=(1, 2)),
            bm.ones_like(determinant),
        )
        ncoeff = normal_matrix.shape[-1]
        if bm.any(bm.abs(determinant) <= 1.0e-14 * matrix_scale**ncoeff):
            raise ValueError("quadratic_lsq stencil is rank deficient.")

        self._cache = (
            stencil,
            feature,
            sample_weight,
            characteristic,
            bm.linalg.inv(normal_matrix),
            boundary_data,
        )

    def quadratic_lsq(self, U):
        if self._cache is None:
            self._build_cache()
        (
            stencil,
            feature,
            sample_weight,
            characteristic,
            inverse,
            boundary_data,
        ) = self._cache

        delta = U[stencil] - U[:, None]
        if U.ndim == 1:
            rhs = bm.einsum("ns,nsi,ns->ni", sample_weight, feature, delta)
        else:
            rhs = bm.einsum("ns,nsi,nsc->nci", sample_weight, feature, delta)

        if boundary_data is not None:
            (
                _,
                boundary_owner,
                boundary_points,
                boundary_feature,
                boundary_weight,
            ) = boundary_data
            boundary_value = self.owner.boundary_value(boundary_points)
            boundary_delta = boundary_value - U[boundary_owner]
            if U.ndim == 1:
                boundary_rhs = bm.einsum(
                    "n,ni,n->ni",
                    boundary_weight,
                    boundary_feature,
                    boundary_delta,
                )
            else:
                boundary_rhs = bm.einsum(
                    "n,ni,nc->nci",
                    boundary_weight,
                    boundary_feature,
                    boundary_delta,
                )
            rhs = bm.index_add(rhs, boundary_owner, boundary_rhs, axis=0)

        if U.ndim == 1:
            coefficients = bm.einsum("nij,nj->ni", inverse, rhs)
            return coefficients[:, : self.GD] / characteristic[:, None]
        coefficients = bm.einsum("nij,ncj->nci", inverse, rhs)
        return coefficients[:, :, : self.GD] / characteristic[:, None, None]


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
        if self.owner.boundary_value is not None and self.owner.boundary_type is None:
            raise ValueError("boundary_type must be set when boundary_value is given.")
        if self.owner.boundary_type is not None and self.owner.boundary_value is None:
            raise ValueError("boundary_value is required when boundary_type is set.")
        if self.owner.boundary_type not in (None, "dirichlet", "neumann"):
            raise ValueError(
                f"Unknown Green-Gauss boundary_type: {self.owner.boundary_type!r}."
            )

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
        grad_U = bm.index_add(
            grad_U,
            bm.concatenate((owner, neighbour)),
            bm.concatenate((flux_grad, -flux_grad), axis=0),
            axis=0,
        )

        if self.owner.boundary_value is not None:
            boundary_faces = selected_boundary_faces(
                self.fvm_geometry,
                self.owner.boundary_threshold,
                default_all=True,
            )
            bd_owner = self.fvm_geometry.owner[boundary_faces]
            points = self.fvm_geometry.face_center[boundary_faces]
            if self.owner.boundary_type == "dirichlet":
                bd_value = self.owner.boundary_value(points)
            else:
                unit_normal = self.fvm_geometry.n_f[boundary_faces]
                center_to_face = points - self.fvm_geometry.cell_center[bd_owner]
                normal_distance = bm.abs(bm.einsum("ij,ij->i", center_to_face, unit_normal))
                normal_derivative = self.owner.boundary_value(points)
                if scalar_field:
                    bd_value = U[bd_owner] + normal_derivative * normal_distance
                else:
                    bd_value = U[bd_owner] + normal_derivative * normal_distance[:, None]

            if scalar_field:
                flux_grad = bd_value[:, None] * self.owner.S_f[boundary_faces]
            else:
                flux_grad = (
                    bd_value[:, :, None] * self.owner.S_f[boundary_faces, None, :]
                )
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
    ``layered_lsq``, ``face_weighted_lsq``, ``quadratic_lsq``, and
    ``green_gauss`` use the cached ``GD = mesh.geo_dimension()`` for gradient
    dimensions.  The linear 2D LSQ inverse keeps the explicit fast path;
    quadratic and other-dimensional systems use the backend batched inverse.

    Implementation notes
    --------------------
    - Performance: keep the current LSQ geometry/inverse caches, then consider
      further reducing ``face_weighted_lsq`` boundary-geometry recomputation, RHS
      assembly work, and temporary arrays.
    - 3D extension: validate boundary Neumann constraints on three-dimensional
      control volumes before treating those paths as stable.
    - Boundary samples use the explicit ``boundary_value``, ``boundary_type``,
      and ``boundary_threshold`` protocol for Dirichlet or Neumann data.
    """

    def __init__(
        self,
        mesh,
        *,
        method="layered_lsq",
        boundary_value=None,
        boundary_type=None,
        boundary_threshold=None,
        layer_weights=(1.0, 0.25),
        boundary_weight=1.0,
        geometry=None,
    ):
        self.mesh = mesh
        self.GD = mesh.geo_dimension()
        self.fvm_geometry = geometry if geometry is not None else FVMGeometry(mesh)
        self.S_f = self.fvm_geometry.S_f
        self.boundary_value = boundary_value
        self.boundary_type = boundary_type
        self.boundary_threshold = boundary_threshold

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
        self.quadratic_lsq_reconstruct = QuadraticLSQGradientReconstruct(self)
        self.green_gauss_reconstruct = GreenGaussGradientReconstruct(self)
        if method is not None:
            if method not in self.cell_gradient:
                raise ValueError(f"Unknown cell_gradient variant: {method!r}.")
            self.cell_gradient.set(method)

    @variantmethod("layered_lsq")
    def cell_gradient(self, U):
        return self.lsq_reconstruct.layered_lsq(U)

    @cell_gradient.register("face_weighted_lsq")
    def cell_gradient(self, U):
        return self.lsq_reconstruct.face_weighted_lsq(U)

    @cell_gradient.register("quadratic_lsq")
    def cell_gradient(self, U):
        return self.quadratic_lsq_reconstruct.quadratic_lsq(U)

    @cell_gradient.register("green_gauss")
    def cell_gradient(self, U):
        return self.green_gauss_reconstruct.green_gauss(U)

    def clear_cache(self):
        """Drop geometry caches after changing mesh coordinates or topology."""
        self.lsq_reconstruct.clear_cache()
        self.quadratic_lsq_reconstruct.clear_cache()
