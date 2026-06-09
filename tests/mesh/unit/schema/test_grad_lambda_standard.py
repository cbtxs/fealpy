import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import (
    EdgeSchema,
    TriangleSchema,
    QuadrilateralSchema,
    TetrahedronSchema,
    HexahedronSchema,
    PrismSchema,
    PyramidSchema,
)
from fealpy.mesh.schema.entity_schema import EntityContext
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


def to_numpy(value):
    return bm.to_numpy(value)


def make_ctx(schema_name, positions, cells):
    block = MeshBlock(positions=bm.asarray(positions, dtype=bm.float64))
    sector = EntitySector(schema_name=schema_name, indices=bm.asarray(cells, dtype=bm.int32))
    block.add_sector(sector, root=True)
    return EntityContext(block=block, sector=sector)


def test_entity_view_grad_lambda_forwards_ref_and_bcs():
    ctx = make_ctx("edge", [[0.0, 0.0], [2.0, 0.0]], [[0, 1]])
    mesh = Mesh(ctx.block)
    view = mesh.sector("edge")

    phys = view.grad_lambda(ref=False)
    ref = view.grad_lambda(ref=True)

    assert phys.shape == (1, 2, 2)
    assert ref.shape == (1, 2, 2)
    np.testing.assert_allclose(to_numpy(phys[0]), [[-0.5, 0.0], [0.5, 0.0]])
    np.testing.assert_allclose(to_numpy(ref[0]), np.eye(2))


def test_simplex_grad_lambda_ref_returns_barycentric_identity():
    edge = make_ctx("edge", [[0.0], [2.0]], [[0, 1]])
    tri = make_ctx("tri", [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], [[0, 1, 2]])
    tet = make_ctx(
        "tet",
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        [[0, 1, 2, 3]],
    )

    np.testing.assert_allclose(to_numpy(EdgeSchema.grad_lambda(edge, None, ref=True)[0]), np.eye(2))
    np.testing.assert_allclose(to_numpy(TriangleSchema.grad_lambda(tri, None, ref=True)[0]), np.eye(3))
    np.testing.assert_allclose(to_numpy(TetrahedronSchema.grad_lambda(tet, None, ref=True)[0]), np.eye(4))

    assert EdgeSchema.grad_lambda(edge, None, ref=False).shape == (1, 2, 1)
    assert TriangleSchema.grad_lambda(tri, None, ref=False).shape == (1, 3, 2)
    assert TetrahedronSchema.grad_lambda(tet, None, ref=False).shape == (1, 4, 3)


def test_quadrilateral_grad_lambda_supports_ref_and_physical_with_bcs_dimension():
    ctx = make_ctx(
        "quad",
        [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]],
        [[0, 1, 2, 3]],
    )
    bcs = (
        bm.asarray([[0.25, 0.75], [0.6, 0.4]], dtype=bm.float64),
        bm.asarray([[0.8, 0.2]], dtype=bm.float64),
    )

    ref = QuadrilateralSchema.grad_lambda(ctx, None, bcs=bcs, ref=True)
    phys = QuadrilateralSchema.grad_lambda(ctx, None, bcs=bcs, ref=False)
    center_ref = QuadrilateralSchema.grad_lambda(ctx, None, ref=True)
    center_phys = QuadrilateralSchema.grad_lambda(ctx, None, ref=False)

    assert ref.shape == (1, 2, 4, 4)
    assert phys.shape == (1, 2, 4, 2)
    assert center_ref.shape == (1, 4, 4)
    assert center_phys.shape == (1, 4, 2)

    expected_ref0 = np.array([
        [0.8, 0.0, 0.25, 0.0],
        [0.0, 0.8, 0.75, 0.0],
        [0.2, 0.0, 0.0, 0.25],
        [0.0, 0.2, 0.0, 0.75],
    ])
    np.testing.assert_allclose(to_numpy(ref[0, 0]), expected_ref0)
    np.testing.assert_allclose(
        to_numpy(center_phys[0]),
        [[-0.25, -1.0 / 6.0], [0.25, -1.0 / 6.0], [0.25, 1.0 / 6.0], [-0.25, 1.0 / 6.0]],
    )


def test_hexahedron_grad_lambda_supports_tensor_product_reference_and_physical():
    ctx = make_ctx(
        "hex",
        [
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0], [2.0, 0.0, 4.0], [0.0, 3.0, 4.0], [2.0, 3.0, 4.0],
        ],
        [[0, 1, 2, 3, 4, 5, 6, 7]],
    )

    ref = HexahedronSchema.grad_lambda(ctx, None, ref=True)
    phys = HexahedronSchema.grad_lambda(ctx, None, ref=False)

    assert ref.shape == (1, 8, 6)
    assert phys.shape == (1, 8, 3)
    np.testing.assert_allclose(
        to_numpy(phys[0, 0]),
        [-0.125, -1.0 / 12.0, -1.0 / 16.0],
    )
    np.testing.assert_allclose(
        to_numpy(phys[0, 7]),
        [0.125, 1.0 / 12.0, 1.0 / 16.0],
    )


def test_prism_and_pyramid_grad_lambda_are_available_with_default_and_bcs():
    prism = make_ctx(
        "prism",
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]],
        [[0, 1, 2, 3, 4, 5]],
    )
    pyramid = make_ctx(
        "pyramid",
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [1.0, 1.0, 2.0]],
        [[0, 1, 2, 3, 4]],
    )

    prism_ref = PrismSchema.grad_lambda(prism, None, ref=True)
    prism_phys = PrismSchema.grad_lambda(prism, None, ref=False)
    pyramid_ref = PyramidSchema.grad_lambda(pyramid, None, ref=True)
    pyramid_phys = PyramidSchema.grad_lambda(pyramid, None, ref=False)

    assert prism_ref.shape == (1, 6, 5)
    assert prism_phys.shape == (1, 6, 3)
    assert pyramid_ref.shape == (1, 5, 6)
    assert pyramid_phys.shape == (1, 5, 3)

    pyramid_bcs = (
        bm.asarray([[0.5, 0.5], [0.25, 0.75]], dtype=bm.float64),
        bm.asarray([[0.5, 0.5]], dtype=bm.float64),
        bm.asarray([[0.5, 0.5]], dtype=bm.float64),
    )
    assert PyramidSchema.grad_lambda(pyramid, None, bcs=pyramid_bcs, ref=True).shape == (1, 2, 5, 6)
    assert PyramidSchema.grad_lambda(pyramid, None, bcs=pyramid_bcs, ref=False).shape == (1, 2, 5, 3)
