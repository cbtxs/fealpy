
from collections.abc import Iterable, Iterator
from typing import NamedTuple, TYPE_CHECKING

from ...backend import bm
from ...backend import Tensor
from ..storage import MeshBlock, EntitySector, Relation

if TYPE_CHECKING:
    from ..schema.entity_schema import EntitySchema


def _unique_unordered_rows_across(*arrays: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Unique rows across 2D tensors after row-wise canonicalization.

    Rows are treated as unordered sets by sorting each row first.
    """
    if not arrays:
        raise ValueError("at least one array is required")

    for arr in arrays:
        if len(arr.shape) != 2:
            raise ValueError("only 2D tensors are supported")

    total = bm.concat(arrays, axis=0)  # (total_rows, ncols)
    canonical_total = bm.sort(total, axis=1)

    indices = bm.lexsort(tuple(reversed(canonical_total.T)), axis=0)  # sorted <-> original
    sorted_canonical = canonical_total[indices]

    diff_flag = bm.any(sorted_canonical[1:] != sorted_canonical[:-1], axis=1)
    true = bm.ones((1,), dtype=bm.bool, device=diff_flag.device)
    diff_flag = bm.concat([true, diff_flag])

    # choose representative rows in their original ordering
    unique = total[indices[diff_flag]]
    sorted_to_unique = bm.cumulative_sum(diff_flag, axis=0) - 1  # sorted -> unique

    original_to_sorted = bm.empty_like(indices)
    original_to_sorted[indices] = bm.arange(
        len(indices),
        dtype=original_to_sorted.dtype,
        device=original_to_sorted.device,
    )
    total_to_unique = sorted_to_unique[original_to_sorted]  # original -> unique

    array_indptr = [0]
    for arr in arrays:
        array_indptr.append(array_indptr[-1] + len(arr))

    arr_to_unique = tuple(
        total_to_unique[array_indptr[i]:array_indptr[i + 1]]
        for i in range(len(arrays))
    )

    return unique, arr_to_unique


def get_total_face(cell: Tensor, local_face: list[list[int]]) -> Tensor:
    total_face = cell[:, local_face]
    NFC = len(local_face[0])
    return bm.reshape(total_face, (-1, NFC))


class ConstructResult(NamedTuple):
    face_type: str
    face: Tensor
    cell_to_face: tuple[Tensor, ...]


class TopologyBuilder:
    @classmethod
    def construct_lower_dims(
        cls,
        cells: Iterable[Tensor],
        local_face_dicts: Iterable[dict[str, list[list[int]]]],
    ) -> Iterator[ConstructResult]:
        """
        Construct lower-dimensional elements.

        Parameters:
            cells (Iterable[Tensor]):
                A sequence of cells, containing tensors in the shape of (NC, NVF).
            local_face_dicts (Iterable[dict[str, list[list[int]]]]):
                A sequence of local face dictionaries. Keys are used to tag the
                faces, and values are local face indices.

        Returns:
            Iterator[ConstructResult]:
                An iterator of ConstructResult, which contains the face type name,
                the unique face array, and the cell-to-face mapping for each input
                cell.
        """
        # NOTE: {face_kind: ([total_face,], [NFC,])}
        face_table: dict[str, tuple[list[Tensor], list[int]]] = {}

        for cell, local_face_dict in zip(cells, local_face_dicts):
            for face_kind, local_face in local_face_dict.items():
                if face_kind not in face_table:
                    face_table[face_kind] = ([], [])

                face_table[face_kind][0].append(get_total_face(cell, local_face))
                face_table[face_kind][1].append(len(local_face))

        for face_kind, (total_face_list, nfc_list) in face_table.items():
            face, js = _unique_unordered_rows_across(*total_face_list)
            cell2faces = tuple(
                bm.reshape(j, (-1, NFC))
                for NFC, j in zip(nfc_list, js)
            )

            yield ConstructResult(face_kind, face, cell2faces)

    @classmethod
    def construct(cls, storage: MeshBlock, src_name: str | None = None) -> None:
        """Construct lower-dimensional blocks and relations.

        If ``src_name`` is given, the construction starts from that block only;
        otherwise all root blocks are used as sources.
        """
        if src_name is None:
            current_blocks = [storage.get_sector(name) for name in storage.root_entity_names]
        else:
            current_blocks = [storage.get_sector(src_name)]

        while current_blocks:
            faces: list[EntitySector] = []

            for const_result in cls.construct_lower_dims(
                [block.indices for block in current_blocks],
                [block.schema.local_faces for block in current_blocks],
            ):
                face_type_name, face_array, cell2face_from_each_cell = const_result

                if face_type_name not in storage.sectors:
                    storage.add_sector(EntitySector(face_type_name, face_array), root=False)
                else:
                    storage.sectors[face_type_name].indices = face_array

                face_block = storage.get_sector(face_type_name)
                faces.append(face_block)

                for cell2face, block in zip(cell2face_from_each_cell, current_blocks):
                    storage.relations[(block.schema_name, face_type_name)] = Relation(
                        src_name=block.schema_name,
                        tgt_name=face_type_name,
                        tgt_indices=cell2face,
                    )

            current_blocks = faces


class TopologyInferer:
    _pattern_select_cache: dict[tuple[str, str, int], Tensor] = {}

    @staticmethod
    def _dim(storage: MeshBlock, name: str) -> int:
        return storage.get_sector(name).schema.top_dim

    @classmethod
    def _select_first_occurrence_positions(cls, row: Tensor) -> Tensor:
        if len(row) == 0:
            return bm.asarray([], dtype=bm.int32)

        _, first_indices, _, _ = bm.unique_all(row)
        order = bm.argsort(first_indices, axis=0)
        return first_indices[order]

    @classmethod
    def _deduplicate_homogeneous_by_pattern(
        cls,
        tgt_indices: Tensor,
        pattern_key: tuple[str, str, int],
    ) -> Tensor:
        if len(tgt_indices) == 0:
            if len(tgt_indices.shape) >= 2:
                return bm.reshape(tgt_indices, (0, tgt_indices.shape[1]))
            return bm.reshape(tgt_indices, (0, 0))

        if len(tgt_indices.shape) != 2:
            raise ValueError("inferred relation is not homogeneous")

        row_width = tgt_indices.shape[1]
        select_pos = cls._pattern_select_cache.get(pattern_key)

        if select_pos is None:
            select_pos = cls._select_first_occurrence_positions(tgt_indices[0])
            cls._pattern_select_cache[pattern_key] = select_pos
        elif len(select_pos) > 0 and bool(bm.any(select_pos >= row_width)):
            select_pos = cls._select_first_occurrence_positions(tgt_indices[0])
            cls._pattern_select_cache[pattern_key] = select_pos

        return tgt_indices[:, select_pos]

    @classmethod
    def _compose_homogeneous(cls, src_to_mid: Relation, mid_to_tgt: Relation) -> Tensor:
        if src_to_mid.src_indices is not None or mid_to_tgt.src_indices is not None:
            raise ValueError("only homogeneous relations are supported for inference")

        composed = mid_to_tgt.tgt_indices[src_to_mid.tgt_indices]
        return bm.reshape(composed, (len(src_to_mid.tgt_indices), -1))

    @classmethod
    def _merge_candidates(
        cls,
        candidates: list[Tensor],
        src_name: str,
        tgt_name: str,
        mid_dim: int
    ) -> Tensor:
        if not candidates:
            raise ValueError("no candidates to merge")

        if len(candidates) == 1:
            merged = candidates[0]
        else:
            merged = bm.concat(candidates, axis=1)
        return cls._deduplicate_homogeneous_by_pattern(
            tgt_indices=merged,
            pattern_key=(src_name, tgt_name, mid_dim)
        )

    @classmethod
    def _iter_adjacent_children(
        cls,
        storage: MeshBlock,
        mid_name: str,
    ) -> Iterator[tuple[str, Relation]]:
        parent_dim = cls._dim(storage, mid_name)
        for (src, tgt), relation in storage.relations.items():
            if src != mid_name:
                continue
            if cls._dim(storage, tgt) != parent_dim - 1:
                continue
            yield tgt, relation

    @classmethod
    def _infer_from(cls, storage: MeshBlock, src_name: str, dst_name: str) -> None:
        src_dim = cls._dim(storage, src_name)
        dst_dim = cls._dim(storage, dst_name)
        if src_dim <= dst_dim:
            return

        if src_dim - 1 < dst_dim:
            raise ValueError(f"no adjacent lower-dimensional relation found for {src_name!r}")

        for mid_dim in range(src_dim - 1, dst_dim, -1): # in [src_dim-1, dst_dim+1]
            # 1) Get all `src -> parent` relations
            src_to_mids: list[tuple[str, Relation]] = []

            if mid_dim == src_dim - 1: # get the highest-dimensional's children by its schema name
                src_to_mids = list(cls._iter_adjacent_children(storage, src_name))
            else: # get the rest by their dimension
                for (rel_src, rel_tgt), relation in storage.relations.items():
                    if rel_src != src_name:
                        continue
                    if cls._dim(storage, rel_tgt) != mid_dim:
                        continue
                    src_to_mids.append((rel_tgt, relation))

            if not src_to_mids:
                raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

            # 2) Get all `mid -> dst` relations and compose them with
            #   `src -> mid` to get `src -> dst` candidates.
            candidates_of_dst: dict[str, list[Tensor]] = {}
            for mid_name, src_to_mid in src_to_mids:
                for dst_name, mid_to_dst in cls._iter_adjacent_children(storage, mid_name):
                    if (src_name, dst_name) in storage.relations:
                        composed = storage.relations[(src_name, dst_name)].tgt_indices
                    else:
                        composed = cls._compose_homogeneous(src_to_mid, mid_to_dst)
                    candidates_of_dst.setdefault(dst_name, []).append(composed)

            for dst_name, candidates in candidates_of_dst.items():
                merged = cls._merge_candidates(candidates, src_name, dst_name, mid_dim)
                storage.relations[(src_name, dst_name)] = Relation(
                    src_name=src_name,
                    tgt_name=dst_name,
                    tgt_indices=merged,
                )

        if (src_name, dst_name) not in storage.relations:
            raise ValueError(f"cannot infer relation from {src_name!r} to {dst_name!r}")

    @classmethod
    def infer(cls, storage: MeshBlock, src_name: str, dst_name: str) -> None:
        src_dim = cls._dim(storage, src_name)
        dst_dim = cls._dim(storage, dst_name)

        if src_dim <= dst_dim:
            raise ValueError(f"expect src dimension > dst dimension, got {src_name!r} -> {dst_name!r}")

        cls._infer_from(storage, src_name, dst_name)


class LocalIndicesInferer:
    """推断实体的跨多个维度的 local indices 关系。

    原理：通过递归利用现有的 schema 中定义的 local_faces，推断一个实体
    对更低维实体的局部编号关系。采用"先到先得"原则：子实体第一次出现时
    即被分配 local index。

    这种推断方式与 TopologyInferer 的全局推断原理一致，区别在于：
    - TopologyInferer 处理的是全局级别的去重和编号
    - LocalIndicesInferer 处理的是单个 schema 内部的局部关系

    数据结构选择：使用 Python dict/list 而非张量，理由：
    1. local_faces 数据量极小（一个 pyramid 最多几十条边）
    2. Python dict 天然保持插入顺序，直接对应"先到先得"原则
    3. 代码逻辑清晰易维护
    4. 张量的初始化和操作开销对小数据集不值得
    """

    _cache: dict[tuple[str, str], list[list[int]]] = {}
    """缓存推断结果 (src_name, tgt_name) -> local_indices"""

    @classmethod
    def _dim(cls, schema_name: str) -> int:
        """获取 schema 的拓扑维数。"""
        from ..schema.registry import SCHEMA_REGISTRY
        return SCHEMA_REGISTRY[schema_name].top_dim

    @classmethod
    def _canonicalize_row(cls, row: tuple[int, ...]) -> tuple[int, ...]:
        """规范化一行（实体），使其作为无序集合的规范表示。

        对于边、面等实体，其拓扑等价性不受节点顺序影响（仅受方向差异）。
        规范形式为：按节点值升序排列的元组。
        """
        return tuple(sorted(row))

    @classmethod
    def infer(cls, schema: type["EntitySchema"], tgt_name: str) -> list[list[int]]:
        """推断 schema 对 tgt_name 的 local indices。

        Parameters:
            schema (EntitySchema): 源实体的 schema 类（如 PyramidSchema）
            tgt_name (str): 目标实体名称（如 "edge"）

        Returns:
            list[list[int]]: 目标实体的 local indices。每个列表表示一个实体的
                           节点在源实体中的索引。结果按"先到先得"原则排序。

        Raises:
            ValueError: 如果无法推断所请求的关系
        """
        from ..schema.registry import SCHEMA_REGISTRY

        cache_key = (schema.name, tgt_name)
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        src_dim = schema.top_dim
        tgt_dim = SCHEMA_REGISTRY[tgt_name].top_dim

        if src_dim <= tgt_dim:
            raise ValueError(
                f"source dimension ({src_dim}) must be > target dimension ({tgt_dim}) "
                f"for {schema.name!r} -> {tgt_name!r}"
            )

        # 基本情况：tgt_name 在 local_faces 中直接存在
        if tgt_name in schema.local_faces:
            result = schema.local_faces[tgt_name]
            cls._cache[cache_key] = result
            return result

        # 递推情况：从 local_faces 中的中间层推断
        seen_entities = {}  # canonical form -> original representation

        for mid_name, mid_local_indices in schema.local_faces.items():
            mid_schema = SCHEMA_REGISTRY[mid_name]

            # 递归获取中间层到目标的 local indices
            mid_to_tgt = cls.infer(mid_schema, tgt_name)

            # 将推断结果映射回源实体的节点编号
            for mid_idx, mid_local in enumerate(mid_local_indices):
                # mid_local 是源实体中该中间层实体的节点索引
                # mid_to_tgt[...] 是中间层对目标实体的节点索引
                for tgt_local in mid_to_tgt:
                    # tgt_local 中的每个元素是中间层的节点编号（0-based）
                    # 需要映射到源实体的节点编号
                    src_local = tuple(mid_local[node_idx] for node_idx in tgt_local)

                    # 规范化作为 key（便于无序去重）
                    canonical = cls._canonicalize_row(src_local)

                    # "先到先得"：只在第一次见到该实体时添加
                    if canonical not in seen_entities:
                        seen_entities[canonical] = list(src_local)

        if not seen_entities:
            raise ValueError(
                f"cannot infer relation from {schema.name!r} to {tgt_name!r}: "
                f"no intermediate dimensions found in local_faces"
            )

        # 转换为 list[list[int]] 格式，保持插入顺序
        result = list(seen_entities.values())
        cls._cache[cache_key] = result
        return result