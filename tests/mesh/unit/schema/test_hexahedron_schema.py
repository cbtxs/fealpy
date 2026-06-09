from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import HexahedronSchema
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


def _build_single_hex_view():
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=bm.float64,
    )
    hex_cell = bm.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("hex", hex_cell), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("hex")


def _build_rectangular_hex_view():
    positions = bm.asarray(
        [
            [10.0, -1.0, 2.0],
            [12.0, -1.0, 2.0],
            [10.0, 2.0, 2.0],
            [12.0, 2.0, 2.0],
            [10.0, -1.0, 7.0],
            [12.0, -1.0, 7.0],
            [10.0, 2.0, 7.0],
            [12.0, 2.0, 7.0],
        ],
        dtype=bm.float64,
    )
    hex_cell = bm.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int64)
    block = MeshBlock(positions=positions)
    block.add_sector(EntitySector("hex", hex_cell), root=True)
    mesh = Mesh(block)
    return mesh, mesh.sector("hex")


class TestHexahedronSchema:
    def test_schema_dispatch_and_metadata(self):
        mesh, hex_view = _build_single_hex_view()

        assert hex_view.schema is HexahedronSchema
        assert hex_view.size() == 1
        assert hex_view.top_dimension() == 3
        assert hex_view.geo_dimension() == mesh.geo_dimension() == 3

        assert len(HexahedronSchema.local_entity("quad")) == 6
        assert len(HexahedronSchema.ccw["quad"]) == 6

    def test_schema_only_defines_handoff_methods(self):
        allowed_methods = {
            "barycenter",
            "bc_to_point",
            "geo_dimension",
            "multi_index",
            "grad_lambda",
            "measure",
            "normal",
            "quadrature_formula",
            "tangent",
        }
        defined_methods = {
            name
            for name, value in HexahedronSchema.__dict__.items()
            if isinstance(value, classmethod)
        }

        assert defined_methods == allowed_methods

    def test_multi_index_requires_tuple_order(self):
        _, hex_view = _build_single_hex_view()

        mi = hex_view.schema.multi_index((2,))
        _assert_shape(mi, (27, 3), "Degree 2 hexahedron multi-index has 27 triples")
        _assert_allclose(
            bm.min(mi, axis=0),
            bm.asarray([0, 0, 0], dtype=mi.dtype),
            "Each degree 2 tensor-product direction starts at 0",
        )
        _assert_allclose(
            bm.max(mi, axis=0),
            bm.asarray([2, 2, 2], dtype=mi.dtype),
            "Each degree 2 tensor-product direction reaches order 2",
        )
        _assert_allclose(
            mi[:5],
            bm.asarray(
                [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1]],
                dtype=mi.dtype,
            ),
            "Hexahedron multi-index should enumerate z fastest, then y, then x",
        )

        anisotropic = hex_view.schema.multi_index((1, 2, 3))
        _assert_shape(anisotropic, (24, 3), "Anisotropic hexahedron order has product size")
        _assert_allclose(
            bm.min(anisotropic, axis=0),
            bm.asarray([0, 0, 0], dtype=anisotropic.dtype),
            "Anisotropic hexahedron multi-index starts each direction at 0",
        )
        _assert_allclose(
            bm.max(anisotropic, axis=0),
            bm.asarray([1, 2, 3], dtype=anisotropic.dtype),
            "Anisotropic hexahedron multi-index respects each direction order",
        )

        with pytest.raises(TypeError):
            hex_view.schema.multi_index(2)

        with pytest.raises(ValueError):
            hex_view.schema.multi_index((-1,))

    def test_barycenter_measure_and_bc_to_point(self):
        _, hex_view = _build_single_hex_view()
        ctx = hex_view.context()

        _assert_allclose(
            hex_view.barycenter(),
            bm.asarray([[0.5, 0.5, 0.5]], dtype=bm.float64),
            "Unit hexahedron barycenter should be the average of its eight vertices",
        )
        _assert_allclose(
            hex_view.measure(),
            bm.asarray([1.0], dtype=bm.float64),
            "Unit hexahedron measure should be one",
        )

        bcs = (
            bm.asarray([[0.2, 0.8], [0.65, 0.35]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3], [0.1, 0.9]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6], [0.75, 0.25]], dtype=bm.float64),
        )
        points = hex_view.schema.bc_to_point(ctx, bcs, None)

        _assert_shape(
            points,
            (1, 2, 2, 2, 3),
            "Tensor-product barycentric coordinates map to a tensor grid of points",
        )
        _assert_allclose(
            points[0, 0, 0, 0],
            bm.asarray([0.8, 0.3, 0.6], dtype=bm.float64),
            "Non-symmetric tensor-product coordinates should map as (u1, v1, w1)",
        )
        _assert_allclose(
            points[0, 1, 1, 1],
            bm.asarray([0.35, 0.9, 0.25], dtype=bm.float64),
            "Second tensor-product sample should preserve each coordinate direction",
        )

    def test_rectangular_hex_geometry_uses_physical_scales(self):
        _, hex_view = _build_rectangular_hex_view()
        ctx = hex_view.context()

        _assert_allclose(
            hex_view.barycenter(),
            bm.asarray([[11.0, 0.5, 4.5]], dtype=bm.float64),
            "Rectangular hexahedron barycenter should average the eight physical vertices",
        )
        _assert_allclose(
            hex_view.measure(),
            bm.asarray([30.0], dtype=bm.float64),
            "Rectangular hexahedron volume should be dx * dy * dz",
        )

        bcs = (
            bm.asarray([[0.2, 0.8]], dtype=bm.float64),
            bm.asarray([[0.7, 0.3]], dtype=bm.float64),
            bm.asarray([[0.4, 0.6]], dtype=bm.float64),
        )
        points = hex_view.schema.bc_to_point(ctx, bcs, None)
        _assert_allclose(
            points[0, 0, 0, 0],
            bm.asarray([11.6, -0.1, 5.0], dtype=bm.float64),
            "Tensor-product coordinates should scale and translate in physical space",
        )

    def test_normal_tangent_and_quadrature(self):
        _, hex_view = _build_single_hex_view()

        _assert_shape(
            hex_view.normal(),
            (1, 0, 3),
            "For T=G=3 hexahedra, normal space has zero directions",
        )
        _assert_allclose(
            hex_view.tangent(),
            bm.asarray([[[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]]], dtype=bm.float64),
            "Unit hexahedron tangent basis should follow local x, y, z edges",
        )

        qf = hex_view.schema.quadrature_formula(2)
        bcs, weights = qf.get_quadrature_points_and_weights()

        assert isinstance(bcs, tuple)
        assert len(bcs) == 3
        _assert_shape(weights, (8,), "Order 2 hexahedron tensor quadrature has eight weights")
        _assert_allclose(
            bm.sum(weights),
            bm.asarray(1.0, dtype=weights.dtype),
            "Reference hexahedron tensor quadrature weights should sum to one",
        )

    def test_rectangular_hex_tangent_uses_non_unit_edges(self):
        _, hex_view = _build_rectangular_hex_view()

        _assert_allclose(
            hex_view.tangent(),
            bm.asarray([[[2.0, 0.0, 0.0],
                         [0.0, 3.0, 0.0],
                         [0.0, 0.0, 5.0]]], dtype=bm.float64),
            "Tangent directions should preserve non-unit physical edge vectors",
        )
