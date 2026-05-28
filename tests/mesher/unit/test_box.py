from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import bm
from fealpy.mesher.box import Box1d, Box2d, Box3d


def _np(value):
    return bm.to_numpy(value)


def test_box1d_initialize_cache_and_connectivity():
    box = Box1d(box=[-1.0, 1.0], nx=4)
    node, cell = box.initialize()

    expected_node = np.array(
        [[-1.0], [-0.5], [0.0], [0.5], [1.0]],
        dtype=np.float64,
    )
    expected_cell = np.array(
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        dtype=np.int32,
    )

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    node_cached, cell_cached = box.initialize()
    assert node is node_cached
    assert cell is cell_cached

    box.clear()
    node_new, cell_new = box.initialize()
    assert node_new is not node
    assert cell_new is not cell


def test_box1d_segmentize_mesh():
    mesh = Box1d(box=[0.0, 1.0], nx=2).segmentize()

    expected_positions = np.array([[0.0], [0.5], [1.0]], dtype=np.float64)
    np.testing.assert_allclose(_np(mesh.block.positions), expected_positions)

    expected_edge = np.array([[0, 1], [1, 2]], dtype=np.int32)
    np.testing.assert_array_equal(_np(mesh.sector("edge").indices), expected_edge)

    assert mesh.entity_count("edge") == 2
    assert mesh.entity_count("node") == 3

    node_indices = _np(mesh.sector("node").indices).reshape(-1)
    np.testing.assert_array_equal(node_indices, np.array([0, 1, 2], dtype=np.int32))


def test_box2d_initialize_and_quadrangulate():
    box = Box2d(box=[0.0, 1.0, 0.0, 1.0], nx=1, ny=1)
    node, cell = box.initialize()

    expected_node = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    expected_cell = np.array([[0, 2, 3, 1]], dtype=np.int32)

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    mesh = box.quadrangulate()
    np.testing.assert_array_equal(_np(mesh.sector("quad").indices), expected_cell)
    assert mesh.entity_count("cell") == 1
    assert mesh.entity_count("edge") == 4
    assert mesh.entity_count("node") == 4


def test_box2d_triangulate():
    mesh = Box2d(nx=1, ny=1).triangulate()
    tri = _np(mesh.sector("tri").indices)

    expected_tri = np.array([[0, 2, 1], [2, 3, 1]], dtype=np.int32)
    np.testing.assert_array_equal(tri, expected_tri)

    assert mesh.entity_count("cell") == 2
    assert mesh.entity_count("edge") == 5


def test_box3d_hexahedralize():
    box = Box3d(nx=1, ny=1, nz=1)
    node, cell = box.initialize()

    expected_node = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected_cell = np.array([[0, 4, 6, 2, 1, 5, 7, 3]], dtype=np.int32)

    np.testing.assert_allclose(_np(node), expected_node)
    np.testing.assert_array_equal(_np(cell), expected_cell)

    mesh = box.hexahedralize()
    np.testing.assert_array_equal(_np(mesh.sector("hex").indices), expected_cell)

    assert mesh.entity_count("cell") == 1
    assert mesh.entity_count("face") == 6
    assert mesh.entity_count("edge") == 12
    assert mesh.entity_count("node") == 8


@pytest.mark.parametrize(
    "method_name, sector_name, expected_cells",
    [
        (
            "tetrahedralize",
            "tet",
            np.array(
                [
                    [0, 4, 6, 7],
                    [0, 5, 4, 7],
                    [0, 1, 5, 7],
                    [0, 3, 1, 7],
                    [0, 2, 3, 7],
                    [0, 6, 2, 7],
                ],
                dtype=np.int32,
            ),
        ),
        (
            "prismatize",
            "prism",
            np.array(
                [
                    [0, 4, 6, 1, 5, 7],
                    [0, 6, 2, 1, 7, 3],
                ],
                dtype=np.int32,
            ),
        ),
        (
            "pyramidalize",
            "pyramid",
            np.array(
                [
                    [0, 4, 6, 2, 3],
                    [0, 4, 1, 5, 3],
                    [0, 6, 1, 7, 3],
                ],
                dtype=np.int32,
            ),
        ),
    ],
    ids=["tetrahedralize", "prismatize", "pyramidalize"],
)
def test_box3d_decompositions(method_name, sector_name, expected_cells):
    box = Box3d(nx=1, ny=1, nz=1)
    mesh = getattr(box, method_name)()

    actual_cells = _np(mesh.sector(sector_name).indices)
    np.testing.assert_array_equal(actual_cells, expected_cells)
    assert mesh.entity_count("cell") == expected_cells.shape[0]
