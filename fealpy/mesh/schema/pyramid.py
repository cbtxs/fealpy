from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["PyramidSchema"]


class PyramidSchema(ShapedEntitySchema):
    name = "pyramid"
    top_dim = 3
    local_faces = {
        'quad': [[0, 1, 2, 3]],
        'tri': [[0, 1, 4], [2, 3, 4], [0, 2, 4], [1, 3, 4]]
    }