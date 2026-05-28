from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import NodeSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh


def _as_list(value):
    return bm.to_numpy(value).tolist()


def _pass(name, basis, actual):
    print(f"[PASS] {name}")
    print(f"       basis: {basis}")
    print(f"       actual: {actual}")


def _assert_allclose(name, actual, expected, basis):
    if not bm.allclose(actual, expected):
        raise AssertionError(
            f"{name} failed\n"
            f"basis: {basis}\n"
            f"actual: {_as_list(actual)}\n"
            f"expected: {_as_list(expected)}"
        )
    _pass(name, basis, _as_list(actual))


def _assert_equal(name, actual, expected, basis):
    actual_list = _as_list(actual)
    expected_list = _as_list(expected)
    if actual_list != expected_list:
        raise AssertionError(
            f"{name} failed\n"
            f"basis: {basis}\n"
            f"actual: {actual_list}\n"
            f"expected: {expected_list}"
        )
    _pass(name, basis, actual_list)


def _assert_shape(name, actual, expected_shape, basis):
    if actual.shape != expected_shape:
        raise AssertionError(
            f"{name} failed\n"
            f"basis: {basis}\n"
            f"actual shape: {actual.shape}\n"
            f"expected shape: {expected_shape}"
        )
    _pass(name, basis, actual.shape)


def _build_user_mesh():
    positions = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=bm.float64,
    )
    node_sector = EntitySector(
        schema_name="node",
        indices=bm.asarray([0, 2], dtype=bm.int64),
    )
    block = MeshBlock(positions=positions)
    block.add_sector(node_sector, root=True)
    return Mesh(block)


def check_node_schema_with_user_view():
    mesh = _build_user_mesh()
    node_view = mesh.sector("node")
    ctx = node_view.context()
    nnode = node_view.size()
    gd = mesh.geo_dimension()
    top_dim = node_view.top_dimension()

    print("NodeSchema validation through Mesh.sector('node')")
    print(f"positions = {_as_list(mesh.block.positions)}")
    print(f"node_view.indices = {_as_list(node_view.indices)}")
    print(f"N = {nnode}, GD = {gd}, T = {top_dim}")
    print()

    if node_view.schema is not NodeSchema:
        raise AssertionError("mesh.sector('node') did not resolve to NodeSchema")
    _pass(
        "schema dispatch",
        "user obtains node algorithms through Mesh.sector('node'), whose schema must be NodeSchema",
        node_view.schema.__name__,
    )

    if NodeSchema.ccw != {}:
        raise AssertionError(f"ccw failed: actual={NodeSchema.ccw}, expected={{}}")
    _pass(
        "ccw metadata",
        "handoff requires every schema to expose ccw; node has no sub-entities, so it is empty",
        NodeSchema.ccw,
    )

    if node_view.geo_dimension() != gd:
        raise AssertionError(f"geo_dimension failed: actual={node_view.geo_dimension()}, expected={gd}")
    _pass(
        "geo_dimension",
        "EntityView.geo_dimension() delegates to schema; expected positions.shape[1]",
        node_view.geo_dimension(),
    )

    expected_barycenter = bm.asarray(
        [
            [0.0, 0.0, 0.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=bm.float64,
    )
    _assert_allclose(
        "barycenter",
        node_view.barycenter(),
        expected_barycenter,
        "a 0D node entity is its own barycenter; user view should return positions[node_view.indices]",
    )

    _assert_allclose(
        "measure",
        node_view.measure(),
        bm.ones((nnode,), dtype=bm.float64),
        "0D entity measure is 1 for each node under new schema semantics",
    )

    grad = node_view.grad_lambda()
    _assert_shape(
        "grad_lambda shape",
        grad,
        (nnode, 1, gd),
        "node has one barycentric coordinate and GD cartesian directions",
    )
    _assert_allclose(
        "grad_lambda value",
        grad,
        bm.zeros((nnode, 1, gd), dtype=bm.float64),
        "the only node barycentric coordinate is constant 1, so its gradient is 0",
    )

    normal = node_view.normal()
    _assert_shape(
        "normal shape",
        normal,
        (nnode, gd - top_dim, gd),
        "handoff rule: normal return shape is [entity_count, G - T, G]",
    )
    _assert_allclose(
        "normal value",
        normal,
        bm.broadcast_to(bm.eye(gd, dtype=bm.float64), (nnode, gd, gd)),
        "for a node T=0, the normal space is the full ambient space represented by standard basis vectors",
    )

    tangent = node_view.tangent()
    _assert_shape(
        "tangent shape",
        tangent,
        (nnode, top_dim, gd),
        "handoff rule: tangent return shape is [entity_count, T, G]; node has T=0",
    )

    bcs = (bm.asarray([[1.0], [1.0]], dtype=bm.float64),)
    point = node_view.schema.bc_to_point(ctx, bcs, None)
    _assert_shape(
        "bc_to_point shape",
        point,
        (nnode, 2, gd),
        "bc_to_point is not currently wrapped by EntityView, so this verifies the schema method behind node_view",
    )
    expected_point = bm.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[4.0, 5.0, 6.0], [4.0, 5.0, 6.0]],
        ],
        dtype=bm.float64,
    )
    _assert_allclose(
        "bc_to_point value",
        point,
        expected_point,
        "node barycentric coordinate is [1], so physical points equal the node coordinates",
    )

    _assert_equal(
        "multi_index",
        node_view.schema.multi_index((3,)),
        bm.asarray([[3]], dtype=bm.int32),
        "handoff requires p to be a tuple; node is the one-vertex simplex degeneration, so degree p has one index [p]",
    )

    try:
        node_view.schema.multi_index(3)
    except TypeError:
        _pass(
            "multi_index rejects scalar p",
            "handoff says p must be a tuple of one or more integers",
            "TypeError",
        )
    else:
        raise AssertionError("multi_index should reject scalar p; handoff requires tuple input")

    print()
    print("All NodeSchema user-view checks passed.")


def main():
    check_node_schema_with_user_view()


if __name__ == "__main__":
    main()
