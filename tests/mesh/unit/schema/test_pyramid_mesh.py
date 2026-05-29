import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema.pyramid import PyramidSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view.mesh import Mesh


PYRAMID_DATA = [
    {
        "node": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 1.0],
            ],
            dtype=np.float64,
        ),
        "cell": np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
        "measure": np.array([1.0 / 3.0], dtype=np.float64),
        "barycenter": np.array([[0.5, 0.5, 0.2]], dtype=np.float64),
        "normal_shape": (1, 0, 3),
        "tangent_shape": (1, 3, 3),
        "local_edges": [
            [0, 1],
            [2, 3],
            [0, 2],
            [1, 3],
            [0, 4],
            [1, 4],
            [2, 4],
            [3, 4],
        ],
    }
]


def _build_pyramid_mesh(data) -> Mesh:
    block = MeshBlock(positions=bm.from_numpy(data["node"]))
    block.add_sector(EntitySector("pyramid", bm.from_numpy(data["cell"])), root=True)
    return Mesh(block)


def _interval_bc(t: float):
    return bm.from_numpy(np.array([[1.0 - t, t]], dtype=np.float64))


def _pyramid_bcs(u: float, v: float, w: float):
    return (_interval_bc(u), _interval_bc(v), _interval_bc(w))


class TestPyramidMesh:
    """
    Unit tests for pyramid entity schema rules and first-stage geometric methods.
    """

    def test_pyramid_schema_local_faces_and_ccw(self) -> None:
        """
        Verify topology-oriented local faces and orientation-oriented ccw faces.
        """
        assert PyramidSchema.local_faces == {
            "quad": [[0, 1, 2, 3]],
            "tri": [[0, 1, 4], [2, 3, 4], [0, 2, 4], [1, 3, 4]],
        }
        assert PyramidSchema.ccw == {
            "quad": [[0, 2, 3, 1]],
            "tri": [[0, 1, 4], [2, 4, 3], [0, 4, 2], [1, 3, 4]],
        }

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_schema_geometry_on_standard_cell(self, data) -> None:
        """
        Verify volume, vertex-average barycenter, normal shape, and tangent shape.
        """
        bm.set_backend("numpy")
        pyramid = _build_pyramid_mesh(data).sector("pyramid")

        np.testing.assert_allclose(
            bm.to_numpy(pyramid.measure()),
            data["measure"],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            bm.to_numpy(pyramid.barycenter()),
            data["barycenter"],
            atol=1e-14,
        )

        normal = pyramid.normal()
        tangent = pyramid.tangent()
        assert normal.shape == data["normal_shape"]
        assert tangent.shape == data["tangent_shape"]
        assert np.linalg.matrix_rank(bm.to_numpy(tangent)[0]) == 3

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_schema_infers_local_edges(self, data) -> None:
        """
        Verify that pyramid local edges are inferred from quad/tri local faces.
        """
        assert PyramidSchema.local_entity("edge") == data["local_edges"]

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_geometry_shape_function_and_bc_to_point(self, data) -> None:
        """
        Verify collapsed-coordinate geometry shape functions and physical mapping.
        """
        bm.set_backend("numpy")
        pyramid = _build_pyramid_mesh(data).sector("pyramid")
        bcs = _pyramid_bcs(0.25, 0.75, 0.5)

        phi = PyramidSchema.geometry_shape_function(bcs)
        np.testing.assert_allclose(
            bm.to_numpy(phi),
            np.array([[0.09375, 0.03125, 0.28125, 0.09375, 0.5]]),
            atol=1e-14,
        )
        np.testing.assert_allclose(
            bm.to_numpy(phi).sum(axis=1),
            np.ones(1),
            atol=1e-14,
        )

        point = PyramidSchema.bc_to_point(pyramid.context(), bcs, None)
        np.testing.assert_allclose(
            bm.to_numpy(point),
            np.array([[[0.375, 0.625, 0.5]]]),
            atol=1e-14,
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_reference_vertices_map_to_physical_vertices(self, data) -> None:
        """
        Verify vertex interpolation and collapsed apex layer.
        """
        bm.set_backend("numpy")
        pyramid = _build_pyramid_mesh(data).sector("pyramid")

        reference_vertices = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        for i, ref_vertex in enumerate(reference_vertices):
            point = PyramidSchema.bc_to_point(
                pyramid.context(), _pyramid_bcs(*ref_vertex), None
            )
            np.testing.assert_allclose(
                bm.to_numpy(point)[0, 0],
                data["node"][i],
                atol=1e-14,
            )

        apex_layer_point = PyramidSchema.bc_to_point(
            pyramid.context(), _pyramid_bcs(0.37, 0.61, 1.0), None
        )
        np.testing.assert_allclose(
            bm.to_numpy(apex_layer_point)[0, 0],
            data["node"][4],
            atol=1e-14,
        )

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_jacobi_matrix_and_grad_transform(self, data) -> None:
        """
        Verify reference derivatives, Jacobian, and chain-rule gradient transform.
        """
        bm.set_backend("numpy")
        pyramid = _build_pyramid_mesh(data).sector("pyramid")
        bcs = _pyramid_bcs(0.5, 0.5, 0.5)

        gphi = PyramidSchema.geometry_grad_shape_function(bcs)
        np.testing.assert_allclose(
            bm.to_numpy(gphi).sum(axis=1),
            np.zeros((1, 3)),
            atol=1e-14,
        )

        J = PyramidSchema.jacobi_matrix(pyramid.context(), bcs, None)
        np.testing.assert_allclose(
            bm.to_numpy(J),
            np.array([[[[0.5, 0.0, 0.0],
                        [0.0, 0.5, 0.0],
                        [0.0, 0.0, 1.0]]]]),
            atol=1e-14,
        )

        ref_grad = bm.from_numpy(
            np.array([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]], dtype=np.float64)
        )
        grad = PyramidSchema.transform_grad(pyramid.context(), bcs, ref_grad, None)
        np.testing.assert_allclose(
            bm.to_numpy(grad),
            np.array([[[[2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 1.0]]]]),
            atol=1e-14,
        )

    def test_pyramid_geometry_gradient_matches_finite_difference(self) -> None:
        """
        Verify reference derivatives against finite differences of shape functions.
        """
        bm.set_backend("numpy")
        u, v, w = 0.23, 0.41, 0.35
        eps = 1.0e-6

        def phi_at(a, b, c):
            return bm.to_numpy(PyramidSchema.geometry_shape_function(
                _pyramid_bcs(a, b, c)
            ))[0]

        finite_diff = np.stack(
            [
                (phi_at(u + eps, v, w) - phi_at(u - eps, v, w)) / (2.0 * eps),
                (phi_at(u, v + eps, w) - phi_at(u, v - eps, w)) / (2.0 * eps),
                (phi_at(u, v, w + eps) - phi_at(u, v, w - eps)) / (2.0 * eps),
            ],
            axis=-1,
        )
        gphi = bm.to_numpy(PyramidSchema.geometry_grad_shape_function(
            _pyramid_bcs(u, v, w)
        ))[0]
        np.testing.assert_allclose(gphi, finite_diff, atol=1e-10)

    @pytest.mark.parametrize(
        "data",
        PYRAMID_DATA,
        ids=["standard-tensor-product-pyramid"],
    )
    def test_pyramid_transform_grad_matches_inverse_jacobian(self, data) -> None:
        """
        Verify transform_grad against the explicit inverse-Jacobian formula.
        """
        bm.set_backend("numpy")
        pyramid = _build_pyramid_mesh(data).sector("pyramid")
        bcs = _pyramid_bcs(0.2, 0.7, 0.4)
        ref_grad_np = np.array(
            [[[1.0, 2.0, -0.5],
              [-0.25, 0.75, 1.5]]],
            dtype=np.float64,
        )
        ref_grad = bm.from_numpy(ref_grad_np)

        J = bm.to_numpy(PyramidSchema.jacobi_matrix(pyramid.context(), bcs, None))[0, 0]
        grad = bm.to_numpy(PyramidSchema.transform_grad(
            pyramid.context(), bcs, ref_grad, None
        ))[0, 0]
        expected = ref_grad_np[0] @ np.linalg.inv(J)
        np.testing.assert_allclose(grad, expected, atol=1e-14)
