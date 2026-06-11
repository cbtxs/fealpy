
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

from ...backend import bm
from ...backend import Tensor, dtype

if TYPE_CHECKING:
    from ..view import Mesh


class InterpolationPoints:
    @classmethod
    def multi_index_matrix(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index matrix for interpolation points of
        degree p with n vertices. The multi-index matrix is of shape
        (C(p+n-1, n-1), n) and each row corresponds to the multi-index of
        an interpolation point.

        Parameters:
            p (int): Degree of interpolation.
            n (int): Number of vertices in the Simplex.
            dtype (dtype, optional): Data type of the output tensor. If None, it will
                default to int32.

        Returns:
            Tensor: A tensor of shape (C(p+n-1, n-1), n) containing the multi-indices.
        """
        if dtype is None:
            dtype = bm.int32

        sep = bm.flip(bm.asarray(
            tuple(combinations_with_replacement(range(p+1), n-1)),
            dtype=dtype
        ), axis=0)
        raw = bm.zeros((sep.shape[0], n+1), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @classmethod
    def multi_index_inner(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index corresponding to the inner interpolation
        points of degree p with n vertices.

        See also: `multi_index_matrix`."""
        if p < n:
            if dtype is None:
                dtype = bm.int32
            return bm.zeros((0, n), dtype=dtype)
        return cls.multi_index_matrix(p - n, n, dtype=dtype) + 1


def to_ipoint(mesh: "Mesh", entity: str, order: int) -> Tensor: # [num_entities, num_ip]
    """Get the interpolation point indices for the given entity and order,
    in unstructured meshes.
    The interpolation point indices are ordered from lower-dimensional
    sub-entities to higher-dimensional entities, and the interpolation points
    of each sub-entity are ordered according to the vertex orientation.

    Parameters:
        mesh (Mesh): The mesh object.
        entity (str): The name of the entity (e.g., "cell", "face", "edge").
        order (int): The degree of interpolation.

    Returns:
        Tensor: A tensor of shape (num_entities, num_ip) containing the
            interpolation point indices.
    """
    collected = []
    dim_cursor = 0
    ip_cursor = 0
    tgt_entity = mesh.sector(entity)
    shutdown = False

    while True:
        for subentity in mesh.entity_views(dim_cursor):
            ### (1) Get ip mapping from sub-entity to the global
            num_sub_entity = subentity.size()
            num_internal_ip = subentity.num_multi_index(order, internal=True)
            if num_internal_ip == 0:
                dim_cursor += 1
                continue
            sub_map = bm.arange(
                ip_cursor,
                ip_cursor + num_sub_entity * num_internal_ip,
                dtype=bm.int64,
            )
            sub_map = bm.reshape(sub_map, (num_sub_entity, num_internal_ip))

            if dim_cursor == tgt_entity.schema.top_dim:
                full_map = bm.reshape(sub_map, (tgt_entity.size(), -1))
                collected.append(full_map)
                shutdown = True
                break

            ### (2) Permute the ip according to the vertex orientation
            tgt_to_sub = tgt_entity.to(subentity).tgt_indices # [num_entities, num_local_subs]
            full_map = bm.reshape(sub_map[tgt_to_sub], (-1, num_internal_ip))
            # [num_entities * num_local_subs, num_internal_ip]
            global_vo = tgt_entity.global_permutations(subentity.schema.name) # [num_entities, num_local_subs, num_subs_vertex]
            global_vo = bm.reshape(global_vo, (-1, global_vo.shape[-1]))
            # [num_entities * num_local_subs, num_subs_vertex]

            for vo, do in subentity.schema.vo_to_do((order,)).items():
                vo = bm.asarray(vo, dtype=bm.uint8, device=full_map.device)
                vo_mask = bm.all(global_vo == vo[None, :], axis=-1) # [num_entities * num_local_subs]
                full_map = bm.where(vo_mask[:, None], full_map[:, do], full_map)

            full_map = bm.reshape(full_map, (tgt_entity.size(), -1))
            # [num_entities, num_local_subs * num_internal_ip]

            ### (3) Collect the ip mapping
            collected.append(full_map)
            ip_cursor += num_sub_entity * num_internal_ip
            dim_cursor += 1
        else:
            if dim_cursor > tgt_entity.schema.top_dim:
                break

        if shutdown:
            break

    return bm.concat(collected, axis=1)


def ipoints(mesh: "Mesh", entity: str, order: tuple[int, ...]) -> Tensor:
    """Get the interpolation points for the given entity and order.

    Parameters:
        mesh (Mesh): The mesh object.
        entity (str): The name of the entity (e.g., "cell", "face", "edge").
        order (tuple[int, ...]): The degree of interpolation.

    Returns:
        Tensor: A tensor of shape (num_ip, GD) containing the interpolation points.
    """
    collected = []
    dim_cursor = 0

    while True:
        for subentity in mesh.entity_views(dim_cursor):
            mi = subentity.schema.multi_index(order, internal=True)
