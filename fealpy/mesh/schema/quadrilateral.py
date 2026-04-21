from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["QuadrilateralSchema"]


class QuadrilateralSchema(ShapedEntitySchema):
    name = "quad"
    top_dim = 2
    local_faces = {
        'edge': [[0, 1], [2, 3], [0, 2], [1, 3]]
    }