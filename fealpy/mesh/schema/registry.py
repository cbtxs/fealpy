
from .entity_schema import EntitySchema
from .node import NodeSchema
from .edge import EdgeSchema
from .triangle import TriangleSchema
from .quadrilateral import QuadrilateralSchema
from .tetrahedron import TetrahedronSchema
from .prism import PrismSchema
from .pyramid import PyramidSchema
from .hexahedron import HexahedronSchema

__all__ = ["SCHEMA_REGISTRY"]


SCHEMA_REGISTRY: dict[str, type[EntitySchema]] = {
	"node": NodeSchema,
	"edge": EdgeSchema,
	"tri": TriangleSchema,
	"quad": QuadrilateralSchema,
	"tet": TetrahedronSchema,
	"prism": PrismSchema,
	"pyramid": PyramidSchema,
	"hex": HexahedronSchema,
}