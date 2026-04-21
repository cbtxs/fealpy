from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["NodeSchema"]


class NodeSchema(ShapedEntitySchema):
    name = "node"
    top_dim = 0
    local_faces = {}

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        if index is None:
            return ctx.block.positions
        return ctx.block.positions[index]
