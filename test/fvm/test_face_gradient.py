import numpy as np

import fealpy.fvm as fvm
from fealpy.fvm import GradientReconstruct
from fealpy.mesh import TriangleMesh


def _skew_two_cell_mesh():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 1.0],
            [1.5, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TriangleMesh(nodes, cells)


def test_face_gradient_public_api_uses_neutral_name():
    assert hasattr(fvm, "reconstruct_face_gradient")
    assert not hasattr(fvm, "openfoam_face_gradient")


def test_gradient_reconstruct_is_cell_gradient_only():
    mesh = _skew_two_cell_mesh()
    reconstruct = GradientReconstruct(mesh)

    assert hasattr(reconstruct, "cell_gradient")
    assert not hasattr(reconstruct, "face_gradient")
