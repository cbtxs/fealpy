
from ..backend import Tensor

from .storage import MeshBlock, EntitySector
from .topology.builder import TopologyBuilder
from .view import Mesh, FEALPyMesh

__all__ = [
    'MeshFactory',
    'TriangleMesh',
]


class MeshFactory:
    pass


class TriangleMesh(MeshFactory):
    def __new__(self, node: Tensor, cell: Tensor) -> FEALPyMesh:
        block = MeshBlock(positions=node)
        block.add_sector(EntitySector("tri", cell), root=True)
        TopologyBuilder.construct(block)
        mesh = Mesh(block)

        return mesh.fealpy_api()

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1],
        nx=10,
        ny=10,
        *,
        threshold=None,
        device=None
    ):
        """Create a triangle mesh of the box."""
        from ..mesher.box import Box2d
        box = Box2d(box, nx, ny, device=device)
        return box.triangulate().fealpy_api()
