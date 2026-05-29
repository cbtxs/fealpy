from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import EdgeSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


def _as_list(value):
    return bm.to_numpy(value).tolist()


def _assert_allclose(actual, expected, message):
    assert bm.allclose(actual, expected), (
        f"{message}\n"
        f"actual: {_as_list(actual)}\n"
        f"expected: {_as_list(expected)}"
    )


def _assert_equal(actual, expected, message):
    actual_list = _as_list(actual)
    expected_list = _as_list(expected)
    assert actual_list == expected_list, (
        f"{message}\n"
        f"actual: {actual_list}\n"
        f"expected: {expected_list}"
    )


def _assert_shape(actual, expected_shape, message):
    assert actual.shape == expected_shape, (
        f"{message}\n"
        f"actual shape: {actual.shape}\n"
        f"expected shape: {expected_shape}"
    )


def _build_edge_view():
    positions = bm.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray(
        [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 3],
        ],
        dtype=bm.int64,
    )
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("edge", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("edge")




def _build_3d_edge_view():
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [2.0, 4.0, 5.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1], [2, 3], [4, 5]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("edge", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("edge")


def _build_degenerate_edge_view():
    positions = bm.asarray(
        [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("edge", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("edge")

def _build_two_edge_view():
    positions = bm.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 2.0],
            [2.0, 5.0],
        ],
        dtype=bm.float64,
    )
    edge = bm.asarray([[0, 1], [2, 3]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("edge", edge), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("edge")


class TestEdgeSchema:
    def test_schema_dispatch_and_metadata(self):
        mesh, edge_view = _build_edge_view()

        assert edge_view.schema is EdgeSchema
        assert edge_view.size() == 4
        assert edge_view.top_dimension() == 1
        assert edge_view.geo_dimension() == mesh.geo_dimension() == 2
        assert EdgeSchema.local_entity("node") == [[0], [1]]
        assert EdgeSchema.ccw == {"node": [[0], [1]]}

    def test_multi_index_via_schema_behind_user_view(self):
        _, edge_view = _build_edge_view()

        _assert_equal(
            edge_view.schema.multi_index((0,)),
            bm.asarray([[0, 0]], dtype=bm.int32),
            "Degree 0 edge multi-index has one all-zero row",
        )
        _assert_equal(
            edge_view.schema.multi_index((2,)),
            bm.asarray([[2, 0], [1, 1], [0, 2]], dtype=bm.int32),
            "Degree 2 edge multi-index should enumerate the two-vertex simplex indices",
        )
        assert edge_view.schema.num_multi_index((2,)) == 3

    def test_multi_index_rejects_invalid_order_argument(self):
        _, edge_view = _build_edge_view()

        with pytest.raises(TypeError):
            edge_view.schema.multi_index(2)

        with pytest.raises(ValueError):
            edge_view.schema.multi_index((-1,))

    def test_multi_index_sort_returns_lexicographic_order(self):
        _, edge_view = _build_edge_view()

        multi_index = bm.asarray(
            [
                [0, 2],
                [2, 0],
                [1, 1],
            ],
            dtype=bm.int32,
        )
        _assert_equal(
            edge_view.schema.multi_index_sort(multi_index),
            bm.asarray([0, 2, 1], dtype=bm.int64),
            "Edge multi-index rows should be sorted lexicographically",
        )

    def test_barycenter_measure_grad_lambda_through_user_view(self):
        _, edge_view = _build_edge_view()

        _assert_allclose(
            edge_view.barycenter(),
            bm.asarray([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1.0, 0.5]], dtype=bm.float64),
            "Edge barycenter should be the average of its two endpoints",
        )
        _assert_allclose(
            edge_view.measure(),
            bm.asarray([1.0, 1.0, 2.0**0.5, 1.0], dtype=bm.float64),
            "Edge measure should be endpoint distance",
        )
        _assert_allclose(
            edge_view.grad_lambda(),
            bm.asarray(
                [
                    [[-1.0, -0.0], [1.0, 0.0]],
                    [[-0.0, -1.0], [0.0, 1.0]],
                    [[0.5, -0.5], [-0.5, 0.5]],
                    [[-0.0, -1.0], [0.0, 1.0]],
                ],
                dtype=bm.float64,
            ),
            "Edge barycentric gradients should match the affine edge formula",
        )

    def test_bc_to_point_via_schema_behind_user_view(self):
        _, edge_view = _build_two_edge_view()
        ctx = edge_view.context()
        bcs = (
            bm.asarray(
                [
                    [1.0, 0.0],
                    [0.25, 0.75],
                ],
                dtype=bm.float64,
            ),
        )

        points = edge_view.schema.bc_to_point(ctx, bcs, None)

        _assert_shape(points, (2, 2, 2), "bc_to_point maps two barycentric samples for each edge entity")
        _assert_allclose(
            points,
            bm.asarray(
                [
                    [[0.0, 0.0], [0.75, 0.0]],
                    [[2.0, 2.0], [2.0, 4.25]],
                ],
                dtype=bm.float64,
            ),
            "Edge barycentric coordinates should map to physical points",
        )

    def test_quadrature_formula_via_schema_behind_user_view(self):
        _, edge_view = _build_edge_view()

        qf = edge_view.schema.quadrature_formula(2, qtype="legendre")
        bcs, weights = qf.get_quadrature_points_and_weights()

        _assert_shape(bcs, (2, 2), "Edge order-2 quadrature has two barycentric points")
        _assert_shape(weights, (2,), "Edge order-2 quadrature has two weights")

    def test_normal_and_tangent_through_user_view(self):
        _, edge_view = _build_edge_view()

        tangent = edge_view.tangent()
        normal = edge_view.normal()

        _assert_shape(tangent, (edge_view.size(), 1, edge_view.geo_dimension()), "Handoff rule: tangent shape is [entity_count, T, G]")
        _assert_allclose(
            tangent,
            bm.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 1.0]], [[0.0, 1.0]]], dtype=bm.float64),
            "Edge tangent is the non-unit endpoint difference x1 - x0",
        )
        _assert_shape(normal, (edge_view.size(), 1, edge_view.geo_dimension()), "2D edge normal has one direction")
        _assert_allclose(
            normal,
            bm.asarray([[[0.0, -1.0]], [[1.0, -0.0]], [[1.0, 1.0]], [[1.0, -0.0]]], dtype=bm.float64),
            "Current 2D edge normal follows [dy, -dx]",
        )

    def test_scalar_index_keeps_single_entity_axis(self):
        _, edge_view = _build_two_edge_view()

        _assert_allclose(
            edge_view.barycenter(index=1),
            bm.asarray([[2.0, 3.5]], dtype=bm.float64),
            "Scalar index should preserve one edge entity axis in barycenter",
        )
        _assert_allclose(
            edge_view.measure(index=1),
            bm.asarray([3.0], dtype=bm.float64),
            "Scalar index should preserve one edge entity axis in measure",
        )
        _assert_shape(edge_view.grad_lambda(index=1), (1, 2, 2), "Scalar index should preserve grad_lambda entity axis")
        _assert_shape(edge_view.tangent(index=1), (1, 1, 2), "Scalar index should preserve tangent entity axis")

    def test_3d_normal_returns_two_orthogonal_directions(self):
        _, edge_view = _build_3d_edge_view()

        tangent = edge_view.tangent()
        normal = edge_view.normal()

        _assert_shape(normal, (edge_view.size(), 2, 3), "3D edge normal should have G - T = 2 directions")
        dot_n0_t = bm.sum(normal[:, 0, :] * tangent[:, 0, :], axis=1)
        dot_n1_t = bm.sum(normal[:, 1, :] * tangent[:, 0, :], axis=1)
        dot_n0_n1 = bm.sum(normal[:, 0, :] * normal[:, 1, :], axis=1)
        _assert_allclose(
            dot_n0_t,
            bm.zeros((edge_view.size(),), dtype=bm.float64),
            "First 3D edge normal direction must be orthogonal to tangent",
        )
        _assert_allclose(
            dot_n1_t,
            bm.zeros((edge_view.size(),), dtype=bm.float64),
            "Second 3D edge normal direction must be orthogonal to tangent",
        )
        _assert_allclose(
            dot_n0_n1,
            bm.zeros((edge_view.size(),), dtype=bm.float64),
            "The two 3D edge normal directions must be mutually orthogonal",
        )
        assert bm.all(bm.sum(normal[:, 0, :] * normal[:, 0, :], axis=1) > 0)
        assert bm.all(bm.sum(normal[:, 1, :] * normal[:, 1, :], axis=1) > 0)

    def test_degenerate_edge_normal_raises_value_error(self):
        _, edge_view = _build_degenerate_edge_view()

        with pytest.raises(ValueError):
            edge_view.normal()

