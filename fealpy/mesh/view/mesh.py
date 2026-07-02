
from dataclasses import dataclass
from typing import NamedTuple, overload, Self

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

    def _ensure_positive_top_dim(self, top_dim: int | str) -> int:
        if isinstance(top_dim, str):
            return self._get_top_dim(top_dim)

        if top_dim < 0:
            top_dim += self.top_dimension() + 1
        if top_dim < 0 or top_dim > self.top_dimension():
            raise ValueError(f"Invalid top dimension: {top_dim}")
        return top_dim

    def _get_top_dim(self, etype: str) -> int:
        if etype in {"cell", "CELL", "Cell"}:
            return self.top_dimension()
        if etype in {"face", "FACE", "Face"}:
            return self.top_dimension() - 1
        if etype in {"edge", "EDGE", "Edge"}:
            return 1
        if etype in {"node", "NODE", "Node"}:
            return 0
        raise ValueError(f"Unknown entity name: {etype}")

    def _get_etype_and_idx(self, etype_string: str) -> tuple[int, int]:
        etype_idx = etype_string.split(":")
        if len(etype_idx) == 1:
            etype = etype_idx[0].strip()
            idx = 0
        else:
            etype, idx = etype_idx
            idx = int(idx.strip())

        if etype in {"cell", "CELL", "Cell"}:
            return self.top_dimension(), idx
        if etype in {"face", "FACE", "Face"}:
            return self.top_dimension() - 1, idx
        if etype in {"edge", "EDGE", "Edge"}:
            return 1, idx
        if etype in {"node", "NODE", "Node"}:
            return 0, idx
        raise ValueError(f"Unknown etype name: {etype}, "
                         "available options are: cell, face, edge, node.")

    ## Entity getters

    @overload
    def Entity(self, name_or_topdim: str | int, /) -> EntityView: ...
    @overload
    def Entity(self, name_or_topdim: str | int, index: int, /) -> EntityView: ...
    def Entity(self, name_or_topdim: str | int, index: int = 0, /) -> EntityView:
        """Get an entity view by its name or top dimension.

        Parameters:
            name_or_topdim (str | int): The name or top dimension of the entity.
                It can be:
                - the name of the entity (e.g., "point", "segment", "tri", "quad", "tet", "hex"),
                - the entity type with an optional index (e.g., "cell:0", "face:1", "edge:2", "node:3"),
                - just the entity type (e.g., "cell", "face", "edge", "node"),
                - the top dimension as an integer (e.g., 0 for nodes, 1 for edges).
                Where the negative top dimension counts from the end (e.g., -1 for cells, -2 for faces).
            index (int, optional): The index to select a type of entity of the specified top dimension.
                The negative index counts from the end. Default is 0, which selects the first entity of that dimension.
                Ignored if `name_or_topdim` is the name or a string that specifies the index.

        Returns:
            EntityView: The requested entity view.

        Example:
            To get the QUAD from a Prism mesh, these are equivalent:
            >>> mesh.Entity("quad")
            >>> mesh.Entity("face:1") # while "face:0" would return the TRI
            >>> mesh.Entity("face", 1)
            >>> mesh.Entity("face", -1)
            >>> mesh.Entity(2, 1)
            >>> mesh.Entity(-2, 1)
            >>> mesh.Quad
            >>> mesh.Faces[1]
        """
        if isinstance(name_or_topdim, int):
            view_list = self.Entity_by_topdim(name_or_topdim)
        else:
            if not isinstance(name_or_topdim, str):
                raise TypeError(f"Expected str or int for name_or_topdim, got {type(name_or_topdim)}")
            if ":" in name_or_topdim:
                return self.Entity_by_etype(name_or_topdim)
            else:
                try:
                    topdim = self._get_top_dim(name_or_topdim)
                    view_list = self.Entity_by_topdim(topdim)
                except ValueError:
                    return self.Entity_by_name(name_or_topdim)

        if index < 0:
            index += len(view_list)
        if index < 0 or index >= len(view_list):
            raise ValueError(f"index {index} is out of range {len(view_list)} "
                             f"for top dimension {name_or_topdim}")

        return view_list[index]

    def Entity_by_topdim(self, top_dim: int | str, /) -> list[EntityView]:
        return [
            EntityView(self.block, block) for block in self.block.sectors.values()
            if block.schema.top_dim == self._ensure_positive_top_dim(top_dim)
        ]

    def Entity_by_etype(self, etype_string: str, /) -> EntityView:
        etype, idx = self._get_etype_and_idx(etype_string)
        return self.Entity_by_topdim(etype)[idx]

    def Entity_by_name(self, name: str, /) -> EntityView:
        try:
            return EntityView(self.block, self.block.get_sector(name))
        except KeyError:
            raise ValueError(f"No entity found with name: {name}")

    @property
    def Point(self) -> EntityView:
        return self.Entity_by_name("point")

    @property
    def Segment(self) -> EntityView:
        return self.Entity_by_name("segment")

    @property
    def Tri(self) -> EntityView:
        return self.Entity_by_name("tri")

    @property
    def Quad(self) -> EntityView:
        return self.Entity_by_name("quad")

    @property
    def Tet(self) -> EntityView:
        return self.Entity_by_name("tet")

    @property
    def Prism(self) -> EntityView:
        return self.Entity_by_name("prism")

    @property
    def Pyramid(self) -> EntityView:
        return self.Entity_by_name("pyramid")

    @property
    def Hex(self) -> EntityView:
        return self.Entity_by_name("hex")

    @property
    def Cells(self) -> list[EntityView]:
        return self.Entity_by_topdim(-1)

    @property
    def Faces(self) -> list[EntityView]:
        return self.Entity_by_topdim(-2)

    @property
    def Edges(self) -> list[EntityView]:
        return self.Entity_by_topdim(1)

    @property
    def Nodes(self) -> list[EntityView]:
        return self.Entity_by_topdim(0)

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
        """Get the geometric dimension of the mesh."""
        return int(self.block.positions.shape[1])

    def fealpy_api(self):
        """Provides a view of the mesh compatible with FEALPy's API."""
        from .fealpy_api import FEALPyMesh
        return FEALPyMesh(self.block)

    def top_dimension(self) -> int:
        """Get the topological dimension of the mesh."""
        if not self.block.root_entity_names:
            return -1
        return max(self.block.sectors[name].schema.top_dim
                   for name in self.block.root_entity_names)

    # Setters (in-place modification)

    def construct(self, exclude: list[str] | None = None) -> Self:
        """Construct the mesh topology, optionally excluding certain entity types."""
        from ..topology.builder import TopologyBuilder
        TopologyBuilder.construct(self.block, exclude=exclude)
        return self

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
