
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from ...backend import Tensor
from ..storage import MeshBlock
from .entity_view import EntityView

__all__ = ["Mesh"]


class AdjointRelation(NamedTuple):
    src_indices: Tensor | None
    tgt_indices: Tensor


@dataclass
class Mesh:
    block: MeshBlock

    def _ensure_top_dim(self, etype: str | int) -> int:
        if isinstance(etype, int):
            if etype < 0:
                etype += self.top_dimension() + 1
            if etype < 0 or etype > self.top_dimension():
                raise ValueError(f"Invalid top dimension: {etype}")
            return etype

        if etype in {"cell", "CELL", "Cell"}:
            return self.top_dimension()
        if etype in {"face", "FACE", "Face"}:
            return self.top_dimension() - 1
        if etype in {"edge", "EDGE", "Edge"}:
            return 1
        if etype in {"vertex", "VERTEX", "Vertex"}:
            return 0
        if etype in {"node", "NODE", "Node"}:
            return 0
        raise ValueError(f"Unknown entity name: {etype}")

    ## Entity getters

    def entity_count(self, etype: str | int, /) -> int:
        """Number of entity types with the given dimension."""
        top_dim = self._ensure_top_dim(etype)

        count = 0
        for block in self.block.sectors.values():
            if block.schema.top_dim == top_dim:
                count += block.indices.shape[0]
        return count

    def entities(self, etype: str | int, /) -> Iterable[Tensor]:
        """Provides an iterable of indices/positions for entities with the given dimension."""
        top_dim = self._ensure_top_dim(etype)

        if top_dim == 0:
            yield self.block.positions
            return

        for block in self.entity_views(etype):
            yield block.indices

    def entity_views(self, etype: str | int, /) -> Iterable[EntityView]:
        """Provides an iterable of views for entities with the given dimension."""
        top_dim = self._ensure_top_dim(etype)

        for block in self.block.sectors.values():
            if block.schema.top_dim == top_dim:
                yield EntityView(self.block, block)

    ## Relation getters

    def relation(self, src_etype: int | str, tgt_etype: int | str, /):
        """Provide topological relationship between entities with given types."""
        for src in self.entity_views(src_etype):
            for tgt in self.entity_views(tgt_etype):
                yield src.to(tgt)

    ## Checkers

    def is_simplex_mesh(self) -> bool:
        """Check if the mesh is a simplex mesh."""
        for name in self.block.sectors.keys():
            if name not in {"node", "edge", "tri", "tet"}:
                return False
        return True

    def is_tensor_mesh(self) -> bool:
        """Check if the mesh is a tensor mesh."""
        for name in self.block.sectors.keys():
            if name not in {"node", "edge", "quad", "hex"}:
                return False
        return True

    def is_elemental(self, entity_name: str | None = None, /) -> bool:
        """Check if the mesh has only one root entity type."""
        is_one_root = len(self.block.root_entity_names) == 1
        if entity_name is None:
            return is_one_root
        return is_one_root and self.block.root_entity_names[0] == entity_name

    ## Other getters

    def geo_dimension(self) -> int:
        return int(self.block.positions.shape[1])

    def fealpy_api(self):
        """Provides a view of the mesh compatible with FEALPy's API."""
        from .fealpy_api import FEALPyMesh
        return FEALPyMesh(self.block)

    def sector(self, name: str, /) -> EntityView:
        """Provides a view of the entity with the given name."""
        return EntityView(self.block, self.block.get_sector(name))

    def top_dimension(self) -> int:
        if not self.block.root_entity_names:
            return -1
        return max(self.block.sectors[name].schema.top_dim
                   for name in self.block.root_entity_names)

    # Setters (in-place modification)

    def uniform_refine(self, times: int = 1) -> None:
        """Uniformly refine the mesh a given number of times.

        Parameters:
            times: Number of times to refine the mesh. Default is 1.
        """
        raise NotImplementedError("Uniform refinement is not implemented yet.")
