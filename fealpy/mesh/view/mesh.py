
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
        if etype in {"node", "NODE", "Node"}:
            return 0
        raise ValueError(f"Unknown entity name: {etype}")

    ## Entity getters

    def entity_view_by_topdim(self, top_dim: int | str, /) -> list[EntityView]:
        return [
            EntityView(self.block, block) for block in self.block.sectors.values()
            if block.schema.top_dim == self._ensure_top_dim(top_dim)
        ]

    def entity_view_by_name(self, name: str, /) -> EntityView:
        try:
            return EntityView(self.block, self.block.get_sector(name))
        except KeyError:
            raise ValueError(f"No entity found with name: {name}")

    @property
    def Point(self) -> EntityView:
        return self.entity_view_by_name("point")

    @property
    def Segment(self) -> EntityView:
        return self.entity_view_by_name("segment")

    @property
    def Tri(self) -> EntityView:
        return self.entity_view_by_name("tri")

    @property
    def Quad(self) -> EntityView:
        return self.entity_view_by_name("quad")

    @property
    def Tet(self) -> EntityView:
        return self.entity_view_by_name("tet")

    @property
    def Prism(self) -> EntityView:
        return self.entity_view_by_name("prism")

    @property
    def Pyramid(self) -> EntityView:
        return self.entity_view_by_name("pyramid")

    @property
    def Hex(self) -> EntityView:
        return self.entity_view_by_name("hex")

    @property
    def Cells(self) -> list[EntityView]:
        return self.entity_view_by_topdim(-1)

    @property
    def Faces(self) -> list[EntityView]:
        return self.entity_view_by_topdim(-2)

    @property
    def Edges(self) -> list[EntityView]:
        return self.entity_view_by_topdim(1)

    @property
    def Nodes(self) -> list[EntityView]:
        return self.entity_view_by_topdim(0)

    ## Checkers

    def is_simplex_mesh(self) -> bool:
        """Check if the mesh is a simplex mesh."""
        for name in self.block.sectors.keys():
            if name not in {"point", "segment", "tri", "tet"}:
                return False
        return True

    def is_tensor_mesh(self) -> bool:
        """Check if the mesh is a tensor mesh."""
        for name in self.block.sectors.keys():
            if name not in {"point", "segment", "quad", "hex"}:
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

    # Plot

    @property
    def add_plot(self):
        """Provides a plotting interface for the mesh."""
        from ..plotting.classic import MeshPloter
        return MeshPloter(self)
