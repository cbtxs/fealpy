from .view import Mesh

__all__ = [
    "Mesh",
    "HomogeneousMesh",
    "SimplexMesh",
    "TensorMesh",
    "StructuredMesh",
]

# deprecated, will be removed in future versions
class HomogeneousMesh(Mesh):
    """Homogeneous mesh."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, Mesh)


class SimplexMesh(HomogeneousMesh):
    """Simplex mesh."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, Mesh) and instance.is_simplex_mesh()


class TensorMesh(HomogeneousMesh):
    """Tensor mesh."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, Mesh) and instance.is_tensor_mesh()


class StructuredMesh(HomogeneousMesh):
    """Structured mesh."""
    def __instancecheck__(cls, instance):
        return isinstance(instance, Mesh)
    # TODO: change after we have structured mesh implementation
