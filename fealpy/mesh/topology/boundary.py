from typing import NamedTuple

from ...backend import bm
from ...backend import Tensor
from ..storage import MeshBlock

__all__ = ["BoundaryInfo", "BoundaryInferencer"]


class BoundaryInfo(NamedTuple):
	index: Tensor
	mask: Tensor
	count: Tensor | None = None


class BoundaryInferencer:
	@staticmethod
	def _top_dimension(storage: MeshBlock) -> int:
		if not storage.root_entity_names:
			return -1
		return max(storage.get_sector(name).schema.top_dim for name in storage.root_entity_names)

	@staticmethod
	def _entity_names_by_dim(storage: MeshBlock, top_dim: int) -> list[str]:
		return [
			block.schema_name
			for block in storage.sectors.values()
			if block.schema.top_dim == top_dim
		]

	@staticmethod
	def _mask_to_index(mask: Tensor) -> Tensor:
		return bm.nonzero(mask)[0]

	@staticmethod
	def _accumulate_target_counts(num_target: int, relation_targets: list[Tensor]) -> Tensor:
		count = bm.zeros((num_target,), dtype=bm.int32)

		for tgt_indices in relation_targets:
			flat = bm.reshape(tgt_indices, (-1,))
			if len(flat) == 0:
				continue

			unique_idx, unique_count = bm.unique_counts(flat)
			count[unique_idx] += unique_count

		return count

	@classmethod
	def infer_codim1(cls, storage: MeshBlock) -> dict[str, BoundaryInfo]:
		"""Infer boundary info for all codimension-1 entity blocks.

		Boundary rule:
			Count adjacent top-dimensional entities. A codim-1 entity is on
			boundary iff adjacency count equals 1.
		"""
		top_dim = cls._top_dimension(storage)
		if top_dim < 0:
			return {}

		codim1_dim = top_dim - 1
		codim1_names = cls._entity_names_by_dim(storage, codim1_dim)
		if not codim1_names:
			return {}

		out: dict[str, BoundaryInfo] = {}

		for codim1_name in codim1_names:
			num_target = len(storage.get_sector(codim1_name).indices)
			relation_targets: list[Tensor] = []

			for (src_name, tgt_name), relation in storage.relations.items():
				if tgt_name != codim1_name:
					continue

				src_dim = storage.get_sector(src_name).schema.top_dim
				if src_dim != top_dim:
					continue

				relation_targets.append(relation.tgt_indices)

			count = cls._accumulate_target_counts(num_target, relation_targets)
			mask = bm.equal(count, 1)
			index = cls._mask_to_index(mask)
			out[codim1_name] = BoundaryInfo(index=index, mask=mask, count=count)

		return out

	@classmethod
	def infer_top_dim(
		cls,
		storage: MeshBlock,
		codim1_boundary: dict[str, BoundaryInfo] | None = None,
	) -> dict[str, BoundaryInfo]:
		"""Infer boundary info for top-dimensional entity blocks.

		Boundary rule:
			A top-dimensional entity is on boundary iff it references at least
			one boundary codim-1 entity.
		"""
		top_dim = cls._top_dimension(storage)
		if top_dim < 0:
			return {}

		if codim1_boundary is None:
			codim1_boundary = cls.infer_codim1(storage)

		top_names = cls._entity_names_by_dim(storage, top_dim)
		out: dict[str, BoundaryInfo] = {}

		for top_name in top_names:
			num_cell = len(storage.get_sector(top_name).indices)
			cell_mask = bm.zeros((num_cell,), dtype=bm.bool)

			for (src_name, tgt_name), relation in storage.relations.items():
				if src_name != top_name:
					continue
				if tgt_name not in codim1_boundary:
					continue

				boundary_mask = codim1_boundary[tgt_name].mask
				hit = boundary_mask[relation.tgt_indices]

				if len(hit.shape) == 1:
					local_mask = hit
				else:
					local_mask = bm.any(hit, axis=1)

				cell_mask = bm.logical_or(cell_mask, local_mask)

			cell_index = cls._mask_to_index(cell_mask)
			out[top_name] = BoundaryInfo(index=cell_index, mask=cell_mask, count=None)

		return out

	@classmethod
	def infer_all(cls, storage: MeshBlock) -> dict[str, BoundaryInfo]:
		"""Infer boundary info for all entity blocks."""
		top_dim = cls._top_dimension(storage)
		if top_dim < 0:
			return {}

		codim1_boundary = cls.infer_codim1(storage)
		top_boundary = cls.infer_top_dim(storage, codim1_boundary=codim1_boundary)

		boundary_masks: dict[str, Tensor] = {}
		out: dict[str, BoundaryInfo] = {}

		for name, info in codim1_boundary.items():
			boundary_masks[name] = info.mask
			out[name] = info

		for name, info in top_boundary.items():
			boundary_masks[name] = info.mask
			out[name] = info

		for dim in range(top_dim - 2, -1, -1):
			dim_names = cls._entity_names_by_dim(storage, dim)
			parent_dim = dim + 1

			for name in dim_names:
				num_entity = len(storage.get_sector(name).indices)
				mask = bm.zeros((num_entity,), dtype=bm.bool)

				for (src_name, tgt_name), relation in storage.relations.items():
					if tgt_name != name:
						continue
					if storage.get_sector(src_name).schema.top_dim != parent_dim:
						continue
					if src_name not in boundary_masks:
						continue

					parent_mask = boundary_masks[src_name]
					parent_index = cls._mask_to_index(parent_mask)
					if len(parent_index) == 0:
						continue

					child_index = bm.reshape(relation.tgt_indices[parent_index], (-1,))
					if len(child_index) == 0:
						continue

					mask[child_index] = True

				index = cls._mask_to_index(mask)
				info = BoundaryInfo(index=index, mask=mask, count=None)
				boundary_masks[name] = mask
				out[name] = info

		return out

	@classmethod
	def infer_entity(
		cls,
		storage: MeshBlock,
		entity_name: str,
		*,
		precomputed: dict[str, BoundaryInfo] | None = None,
	) -> BoundaryInfo:
		"""Infer boundary info for one entity block by name."""
		if precomputed is None:
			precomputed = cls.infer_all(storage)

		try:
			return precomputed[entity_name]
		except KeyError as e:
			raise ValueError(f"No entity block found for {entity_name!r}") from e
