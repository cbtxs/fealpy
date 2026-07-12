
from ..backend import Tensor

from .storage import MeshBlock, EntitySector
from .topology.builder import TopologyBuilder
from .view import Mesh, FEALPyMesh

__all__ = [
    'MeshFactory',
    'IntervalMesh',
    'EdgeMesh',
    'TriangleMesh',
    'QuadrangleMesh',
    'TetrahedronMesh',
    'PrismMesh',
    'PyramidMesh',
    'HexahedronMesh',
    'PolygonMesh',
    'LagrangeTriangleMesh',
    'LagrangeQuadrangleMesh',
]


class MeshFactory(type):
    schema: str
    def __instancecheck__(cls, instance) -> bool:
        if not isinstance(instance, Mesh):
            return False
        return instance.is_elemental(cls.schema)


class _MeshFactoryNewMixin(metaclass=MeshFactory):
    def __new__(cls, node: Tensor, cell: Tensor) -> FEALPyMesh:
        block = MeshBlock(positions=node)
        block.add_sector(EntitySector(cls.schema, cell), root=True)
        TopologyBuilder.construct(block)
        mesh = Mesh(block)

        return mesh.fealpy_api()


class IntervalMesh(_MeshFactoryNewMixin):
    schema = "segment"


class EdgeMesh(_MeshFactoryNewMixin):
    schema = "segment"


class TriangleMesh(_MeshFactoryNewMixin):
    schema = "tri"

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


class QuadrangleMesh(_MeshFactoryNewMixin):
    schema = "quad"

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
        """Create a quadrangle mesh of the box."""
        from ..mesher.box import Box2d
        box = Box2d(box, nx, ny, device=device)
        return box.quadrangulate().fealpy_api()


class TetrahedronMesh(_MeshFactoryNewMixin):
    schema = "tet"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        threshold=None,
        device=None
    ):
        """Create a tetrahedron mesh of the box."""
        from ..mesher.box import Box3d
        box = Box3d(box, nx, ny, nz, device=device)
        return box.tetrahedralize().fealpy_api()


class PrismMesh(_MeshFactoryNewMixin):
    schema = "prism"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        threshold=None,
        device=None
    ):
        """Create a prism mesh of the box."""
        from ..mesher.box import Box3d
        box = Box3d(box, nx, ny, nz, device=device)
        return box.prismatize().fealpy_api()


class PyramidMesh(_MeshFactoryNewMixin):
    schema = "pyramid"


class HexahedronMesh(_MeshFactoryNewMixin):
    schema = "hex"

    @classmethod
    def from_box(
        cls,
        box=[0, 1, 0, 1, 0, 1],
        nx=10,
        ny=10,
        nz=10,
        *,
        threshold=None,
        device=None
    ):
        """Create a hexahedron mesh of the box."""
        from ..mesher.box import Box3d
        box = Box3d(box, nx, ny, nz, device=device)
        return box.hexahedralize().fealpy_api()


class PolygonMesh(metaclass=MeshFactory):
    schema = "poly"


class LagrangeTriangleMesh(metaclass=MeshFactory):
    schema = "lagrange_tri"


class LagrangeQuadrangleMesh(metaclass=MeshFactory):
    schema = "lagrange_quad"
