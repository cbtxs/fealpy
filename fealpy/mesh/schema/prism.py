from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["PrismSchema"]


class PrismSchema(ShapedEntitySchema):
    name = "prism"
    top_dim = 3
    local_faces = {
        'tri': [[0, 1, 2], [3, 4, 5]],
        'quad': [[0, 1, 3, 4], [0, 2, 3, 5], [1, 2, 4, 5]]
    }