
from .entity_schema import EntitySchema
from .classic import *

__all__ = ["SCHEMA_REGISTRY"]


SCHEMA_REGISTRY: dict[str, type[EntitySchema]] = {
	"point": PointSchema,
	"segment": SegmentSchema,
	"tri": TriangleSchema,
	"quad": QuadrilateralSchema,
	"tet": TetrahedronSchema,
	"prism": PrismSchema,
	"pyramid": PyramidSchema,
	"hex": HexahedronSchema,
}