import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry
from fealpy.model import PDEModelManager


def _mean_nonorthogonal_ratio(mesh):
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    geometry = FVMGeometry(mesh)
    correction = np.asarray(geometry.bounded_over_relaxed_decomposition()[2])
    area = np.asarray(geometry.mag_S_f)
    ratio = np.linalg.norm(correction, axis=1) / area
    return float(np.mean(ratio[is_internal]))


def test_poisson_manager_loads_box_with_circular_hole_example():
    pytest.importorskip("gmsh")
    pde = PDEModelManager("poisson").get_example(13)

    mesh = pde.init_mesh["uniform_tri"](nx=8, ny=8)

    assert pde.geo_dimension() == 2
    assert mesh.number_of_cells() > 0
    assert hasattr(pde, "solution")
    assert hasattr(pde, "source")
    assert hasattr(pde, "dirichlet")


def test_exp0013_complex_tri_mesh_is_more_nonorthogonal_than_regular_box():
    pytest.importorskip("gmsh")
    bm.set_backend("numpy")
    complex_pde = PDEModelManager("poisson").get_example(13)

    regular_mesh = complex_pde.init_mesh["uniform_tri"](nx=12, ny=12)
    complex_mesh = complex_pde.init_mesh["complex_tri"]()

    regular_ratio = _mean_nonorthogonal_ratio(regular_mesh)
    complex_ratio = _mean_nonorthogonal_ratio(complex_mesh)

    assert complex_mesh.number_of_cells() > 100
    assert complex_ratio > 1.3 * regular_ratio
    assert complex_ratio > 0.13


def test_exp0013_bad_tri_mesh_triggers_bounded_over_relaxed_stabilization():
    bm.set_backend("numpy")
    pde = PDEModelManager("poisson").get_example(13)
    mesh = pde.init_mesh["bad_tri"]()
    geometry = FVMGeometry(mesh)
    eps = 0.05

    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    delta = np.asarray(geometry.d_f)
    normal = np.asarray(geometry.n_f)
    face_area = np.asarray(geometry.mag_S_f)
    ratio = np.einsum("ij,ij->i", normal, delta) / np.linalg.norm(delta, axis=1)
    stabilized_norm = np.linalg.norm(
        np.asarray(geometry.bounded_over_relaxed_decomposition(eps=eps)[2]),
        axis=1,
    )
    stabilized_ratio = stabilized_norm / face_area

    assert mesh.number_of_cells() == 2
    assert np.any(ratio[is_internal] <= eps)
    assert np.max(stabilized_ratio[is_internal]) <= 1.0 + 1.0 / eps
