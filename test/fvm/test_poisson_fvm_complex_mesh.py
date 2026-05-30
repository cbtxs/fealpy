import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import NonOrthogonalGeometry
from fealpy.model import PDEModelManager


def _mean_nonorthogonal_ratio(mesh):
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    geometry = NonOrthogonalGeometry(mesh)
    correction = np.asarray(geometry.openfoam_correction_vector())
    area = np.asarray(geometry.face_area_norm())
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


def test_exp0013_bad_tri_mesh_triggers_openfoam_stabilization():
    bm.set_backend("numpy")
    pde = PDEModelManager("poisson").get_example(13)
    mesh = pde.init_mesh["bad_tri"]()
    geometry = NonOrthogonalGeometry(mesh, eps=0.05)

    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    delta = np.asarray(geometry.cell_center_vector())
    normal = np.asarray(geometry.unit_normal())
    face_area = np.asarray(geometry.face_area_norm())
    ratio = np.einsum("ij,ij->i", normal, delta) / np.linalg.norm(delta, axis=1)
    stabilized_norm = np.linalg.norm(np.asarray(geometry.openfoam_correction_vector()), axis=1)
    stabilized_ratio = stabilized_norm / face_area

    assert mesh.number_of_cells() == 2
    assert np.any(ratio[is_internal] <= geometry.eps)
    assert np.max(stabilized_ratio[is_internal]) <= 1.0 + 1.0 / geometry.eps
