import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.storage import MeshBlock, EntitySector
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.schema.entity_schema import EntityContext


BACKENDS = ["numpy"]
BACKEND_IDS = ["backend-numpy"]


class TestPrismSchema:
    @staticmethod
    def make_ctx():
        node = bm.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ], dtype=bm.float64)

        cell = bm.array([
            [0, 1, 2, 3, 4, 5],
            [3, 4, 5, 6, 7, 8],
        ], dtype=bm.int32)

        storage = MeshBlock(positions=node)
        storage.add_sector(EntitySector("prism", cell), root=True)
        TopologyBuilder.construct(storage)
        return EntityContext(block=storage, sector=storage.get_sector("prism"))

    @staticmethod
    def bcs():
        return (
            bm.array([[1.0 / 2.0, 1.0 / 4.0, 1.0 / 4.0]], dtype=bm.float64),
            bm.array([[1.0 / 3.0, 2.0 / 3.0]], dtype=bm.float64),
        )

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_multi_index(self, backend):
        bm.set_backend(backend)
        schema = self.make_ctx().sector.schema

        mi = bm.to_numpy(schema.multi_index((2, 2)))

        expected = np.array([
            [2, 0, 0, 2, 0],
            [2, 0, 0, 1, 1],
            [2, 0, 0, 0, 2],
            [1, 1, 0, 2, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 0, 2],
            [1, 0, 1, 2, 0],
            [1, 0, 1, 1, 1],
            [1, 0, 1, 0, 2],
            [0, 2, 0, 2, 0],
            [0, 2, 0, 1, 1],
            [0, 2, 0, 0, 2],
            [0, 1, 1, 2, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 1, 0, 2],
            [0, 0, 2, 2, 0],
            [0, 0, 2, 1, 1],
            [0, 0, 2, 0, 2],
        ], dtype=np.int32)

        np.testing.assert_array_equal(mi, expected)

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_shape_function(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema

        phi = schema.shape_function(self.bcs(), p=1)
        gphi = schema.grad_shape_function(ctx, self.bcs(), p=1)

        expected_phi = np.array([[
            1.0 / 6.0,
            1.0 / 3.0,
            1.0 / 12.0,
            1.0 / 6.0,
            1.0 / 12.0,
            1.0 / 6.0,
        ]])

        expected_gphi = np.array([[
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 2.0],
            [-2.0 / 3.0, -2.0 / 3.0,  1.0 / 2.0],
            [ 1.0 / 3.0,  0.0,       -1.0 / 4.0],
            [ 2.0 / 3.0,  0.0,        1.0 / 4.0],
            [ 0.0,        1.0 / 3.0, -1.0 / 4.0],
            [ 0.0,        2.0 / 3.0,  1.0 / 4.0],
        ]])

        np.testing.assert_allclose(bm.to_numpy(phi), expected_phi, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(gphi), expected_gphi, atol=1.0e-12)

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_geometry(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema

        point = schema.bc_to_point(ctx, self.bcs())
        barycenter = schema.barycenter(ctx)
        measure = schema.measure(ctx)

        expected_point = np.array([
            [[1.0 / 4.0, 1.0 / 4.0, 2.0 / 3.0]],
            [[1.0 / 4.0, 1.0 / 4.0, 5.0 / 3.0]],
        ])

        expected_barycenter = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0, 1.5],
        ])

        expected_measure = np.array([0.5, 0.5])

        np.testing.assert_allclose(bm.to_numpy(point), expected_point, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(barycenter), expected_barycenter, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(measure), expected_measure, atol=1.0e-12)

    @pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
    def test_jacobi(self, backend):
        bm.set_backend(backend)
        ctx = self.make_ctx()
        schema = ctx.sector.schema

        J, gphi = schema.jacobi_matrix(ctx, self.bcs(), return_grad=True)
        G = schema.first_fundamental_form(ctx, self.bcs())

        expected_J = np.tile(np.eye(3), (2, 1, 1, 1))
        expected_G = np.tile(np.eye(3), (2, 1, 1, 1))

        np.testing.assert_allclose(bm.to_numpy(J), expected_J, atol=1.0e-12)
        np.testing.assert_allclose(bm.to_numpy(G), expected_G, atol=1.0e-12)


if __name__ == "__main__":
    # pytest.main(["./test_prism_schema.py", "-k", "test_multi_index"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_shape_function"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_geometry"])
    # pytest.main(["./test_prism_schema.py", "-k", "test_jacobi"])
    pytest.main(["./test_prism_schema.py"])