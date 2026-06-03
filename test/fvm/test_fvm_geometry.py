import numpy as np
import pytest

import fealpy.fvm as fvm
from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    FVMGeometry,
    face_interpolation_owner_weight,
)
from fealpy.mesh import QuadrangleMesh, TriangleMesh


def _meshes():
    bm.set_backend("numpy")
    return [
        QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2),
        TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2),
    ]


@pytest.mark.parametrize("mesh", _meshes())
def test_geometry_vectors_are_owner_oriented_on_quad_and_tri(mesh):
    geometry = FVMGeometry(mesh)
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    owner = edge_to_cell[:, 0]
    neighbour = edge_to_cell[:, 1]
    is_internal = owner != neighbour

    assert geometry.S_f.shape == (mesh.number_of_faces(), mesh.geo_dimension())
    assert geometry.d_f.shape == geometry.S_f.shape
    np.testing.assert_array_equal(np.asarray(geometry.owner), owner)
    np.testing.assert_array_equal(np.asarray(geometry.neighbour), neighbour)
    np.testing.assert_array_equal(np.asarray(geometry.is_internal), is_internal)
    np.testing.assert_array_equal(np.asarray(geometry.is_boundary), ~is_internal)

    np.testing.assert_allclose(
        np.asarray(geometry.mag_S_f),
        np.linalg.norm(np.asarray(geometry.S_f), axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(geometry.n_f),
        np.asarray(geometry.S_f) / np.asarray(geometry.mag_S_f)[:, None],
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(geometry.mag_d_f),
        np.linalg.norm(np.asarray(geometry.d_f), axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

    projection = np.einsum(
        "ij,ij->i",
        np.asarray(geometry.S_f),
        np.asarray(geometry.d_f),
    )
    assert np.all(projection > 0.0)


@pytest.mark.parametrize("mesh", _meshes())
def test_boundary_owner_to_face_vector_and_normal_distance(mesh):
    geometry = FVMGeometry(mesh)
    boundary = np.asarray(geometry.is_boundary)
    owner = np.asarray(geometry.owner)[boundary]
    face_center = np.asarray(mesh.entity_barycenter("face"))[boundary]
    cell_center = np.asarray(mesh.entity_barycenter("cell"))[owner]

    expected_vector = face_center - cell_center
    expected_distance = np.einsum(
        "ij,ij->i",
        expected_vector,
        np.asarray(geometry.n_f)[boundary],
    )

    np.testing.assert_allclose(
        np.asarray(geometry.boundary_owner_to_face_vector),
        expected_vector,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(geometry.boundary_normal_distance),
        expected_distance,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    assert np.all(expected_distance > 0.0)


@pytest.mark.parametrize("mesh", _meshes())
def test_normal_distance_returns_selected_face_projection(mesh):
    geometry = FVMGeometry(mesh)
    faces = np.asarray(mesh.boundary_face_index(), dtype=np.int64)[:2]

    actual = np.asarray(geometry.normal_distance(faces))
    expected = np.einsum(
        "ij,ij->i",
        np.asarray(geometry.d_f)[faces],
        np.asarray(geometry.n_f)[faces],
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("mesh", _meshes())
def test_over_relaxed_decomposition_returns_Ef_magEf_and_Tf(mesh):
    geometry = FVMGeometry(mesh)
    S_f = np.asarray(geometry.S_f)
    d_f = np.asarray(geometry.d_f)
    E_f, mag_E_f, T_f = geometry.over_relaxed_decomposition()

    expected_E_f = (
        np.einsum("ij,ij->i", S_f, S_f)
        / np.einsum("ij,ij->i", d_f, S_f)
    )[:, None] * d_f

    np.testing.assert_allclose(
        np.asarray(E_f),
        expected_E_f,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(mag_E_f),
        np.linalg.norm(expected_E_f, axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(T_f),
        S_f - expected_E_f,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.einsum("ij,ij->i", np.asarray(T_f), S_f),
        0.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("mesh", _meshes())
def test_bounded_over_relaxed_decomposition_returns_stabilized_Ef_and_Tf(mesh):
    geometry = FVMGeometry(mesh)
    eps = 0.05
    E_f, mag_E_f, T_f = geometry.bounded_over_relaxed_decomposition(eps=eps)

    denominator = np.maximum(
        np.einsum("ij,ij->i", np.asarray(geometry.n_f), np.asarray(geometry.d_f)),
        eps * np.asarray(geometry.mag_d_f),
    )
    expected_E_f = (
        np.asarray(geometry.mag_S_f) / denominator
    )[:, None] * np.asarray(geometry.d_f)

    np.testing.assert_allclose(
        np.asarray(E_f),
        expected_E_f,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(mag_E_f),
        np.linalg.norm(expected_E_f, axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(T_f),
        np.asarray(geometry.S_f) - expected_E_f,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_geometry_layer_does_not_export_legacy_vector_decomposition():
    assert not hasattr(fvm, "VectorDecomposition")


def test_decomposition_quantities_are_not_initialized_as_basic_geometry():
    geometry = FVMGeometry(_meshes()[0])

    assert not hasattr(geometry, "E_f")
    assert not hasattr(geometry, "mag_E_f")
    assert not hasattr(geometry, "T_f")


@pytest.mark.parametrize("mesh", _meshes())
def test_linear_owner_weight_matches_existing_face_interpolation(mesh):
    geometry = FVMGeometry(mesh)

    np.testing.assert_allclose(
        np.asarray(geometry.linear_owner_weight()),
        np.asarray(face_interpolation_owner_weight(mesh, method="linear")),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("mesh", _meshes())
def test_linear_owner_weight_is_one_on_boundary_faces(mesh):
    geometry = FVMGeometry(mesh)
    weight = np.asarray(geometry.linear_owner_weight())

    np.testing.assert_allclose(
        weight[np.asarray(geometry.is_boundary)],
        1.0,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("mesh", _meshes())
def test_scatter_face_flux_to_cells_uses_owner_oriented_signs(mesh):
    geometry = FVMGeometry(mesh)
    internal_face = np.flatnonzero(np.asarray(geometry.is_internal))[0]
    owner = int(np.asarray(geometry.owner)[internal_face])
    neighbour = int(np.asarray(geometry.neighbour)[internal_face])
    face_flux = np.zeros(mesh.number_of_faces())
    face_flux[internal_face] = 3.0

    scattered = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    expected = np.zeros(mesh.number_of_cells())
    expected[owner] += 3.0
    expected[neighbour] -= 3.0
    np.testing.assert_allclose(scattered, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("mesh", _meshes())
def test_scatter_face_flux_to_cells_adds_boundary_flux_to_owner_only(mesh):
    geometry = FVMGeometry(mesh)
    boundary_face = np.flatnonzero(np.asarray(geometry.is_boundary))[0]
    owner = int(np.asarray(geometry.owner)[boundary_face])
    face_flux = np.zeros(mesh.number_of_faces())
    face_flux[boundary_face] = -2.5

    scattered = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    expected = np.zeros(mesh.number_of_cells())
    expected[owner] -= 2.5
    np.testing.assert_allclose(scattered, expected, rtol=1.0e-13, atol=1.0e-13)


@pytest.mark.parametrize("mesh", _meshes())
def test_scatter_face_flux_to_cells_supports_vector_fluxes(mesh):
    geometry = FVMGeometry(mesh)
    face_flux = np.stack(
        [
            np.linspace(0.2, 1.2, mesh.number_of_faces()),
            np.linspace(-0.5, 0.5, mesh.number_of_faces()),
        ],
        axis=1,
    )
    expected = np.zeros((mesh.number_of_cells(), 2))
    owner = np.asarray(geometry.owner)
    neighbour = np.asarray(geometry.neighbour)
    internal = np.asarray(geometry.is_internal)
    np.add.at(expected, owner, face_flux)
    np.add.at(expected, neighbour[internal], -face_flux[internal])

    scattered = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    np.testing.assert_allclose(scattered, expected, rtol=1.0e-13, atol=1.0e-13)


def test_geometry_interface_does_not_expose_deprecated_nonorthogonal_names():
    geometry = FVMGeometry(_meshes()[0])

    assert not hasattr(geometry, "orthogonal_flux_coeff")
    assert not hasattr(geometry, "delta_coeff")
    assert not hasattr(geometry, "correction_vector")
    assert not hasattr(geometry, "average_owner_weight")
    assert not hasattr(geometry, "distance_owner_weight")
