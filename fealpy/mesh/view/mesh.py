
from collections.abc import Iterable

from ..storage import MeshBlock
from .entity_view import EntityView

__all__ = ["Mesh"]


class Mesh:
    def __init__(self, block: MeshBlock):
        self.block = block

    def geo_dimension(self) -> int:
        return int(self.block.positions.shape[1])

    def legacy(self):
        """Provides a view of the mesh compatible with FEALPy's API."""
        from .fealpy_legacy import FEALPyMesh
        return FEALPyMesh(self.block)

    def sector(self, name: str, /) -> EntityView:
        """Provides a view of the entity with the given name."""
        return EntityView(self.block, self.block.get_sector(name))

    def sectors(self, top_dim: int | None = None, /) -> Iterable[EntityView]:
        """Provides an iterable of views for entities with the given top dimension."""
        if isinstance(top_dim, int) and top_dim < 0:
            top_dim += self.top_dimension() + 1

        for block in self.block.sectors.values():
            if top_dim is None or block.schema.top_dim == top_dim:
                yield EntityView(self.block, block)

    def top_dimension(self) -> int:
        if not self.block.root_entity_names:
            return -1
        return max(self.block.sectors[name].schema.top_dim
                   for name in self.block.root_entity_names)
