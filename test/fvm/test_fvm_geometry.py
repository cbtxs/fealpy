import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    FVMGeometry,
    face_interpolation_owner_weight,
    interpolate_cell_to_face,
)
from fealpy.fvm.fvm_geometry import (
    DiffusionFaceDecomposition,
    face_interpolation_owner_weight as geometry_face_interpolation_owner_weight,
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
    decomposition = geometry.diffusion_face_decomposition("over_relaxed")
    E_f = decomposition.E_f
    mag_E_f = decomposition.mag_E_f
    T_f = decomposition.T_f

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
    decomposition = geometry.diffusion_face_decomposition(
        "bounded_over_relaxed", eps=eps
    )
    E_f = decomposition.E_f
    mag_E_f = decomposition.mag_E_f
    T_f = decomposition.T_f

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


@pytest.mark.parametrize("mesh", _meshes())
@pytest.mark.parametrize(
    "method",
    ["over_relaxed", "bounded_over_relaxed", "uncorrected"],
)
def test_diffusion_face_decomposition_is_named_complete_and_cached(mesh, method):
    geometry = FVMGeometry(mesh)

    first = geometry.diffusion_face_decomposition(method, eps=0.05)
    second = geometry.diffusion_face_decomposition(method, eps=0.05)

    assert isinstance(first, DiffusionFaceDecomposition)
    assert first is second
    np.testing.assert_allclose(
        np.asarray(first.E_f + first.T_f),
        np.asarray(geometry.S_f),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(first.mag_E_f),
        np.linalg.norm(np.asarray(first.E_f), axis=1),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        np.asarray(first.orthogonal_factor),
        np.asarray(first.mag_E_f) / np.asarray(geometry.mag_d_f),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    if method == "uncorrected":
        corrected = geometry.diffusion_face_decomposition("over_relaxed")
        np.testing.assert_allclose(
            np.asarray(first.T_f),
            np.asarray(corrected.T_f),
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_diffusion_face_decomposition_cache_key_includes_eps():
    geometry = FVMGeometry(_meshes()[1])

    first = geometry.diffusion_face_decomposition(
        "bounded_over_relaxed", eps=0.05
    )
    second = geometry.diffusion_face_decomposition(
        "bounded_over_relaxed", eps=0.10
    )

    assert first is not second


def test_diffusion_face_decomposition_rejects_unknown_method():
    geometry = FVMGeometry(_meshes()[0])

    with pytest.raises(ValueError, match="unknown diffusion method"):
        geometry.diffusion_face_decomposition("misspelled")


@pytest.mark.parametrize("mesh", _meshes())
def test_linear_owner_weight_matches_exported_face_interpolation(mesh):
    geometry = FVMGeometry(mesh)

    np.testing.assert_allclose(
        np.asarray(geometry.linear_owner_weight()),
        np.asarray(face_interpolation_owner_weight(mesh, method="linear")),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("mesh", _meshes())
def test_interpolate_cell_to_face_matches_geometry_owner_weights(mesh):
    geometry = FVMGeometry(mesh)
    values = bm.arange(mesh.number_of_cells(), dtype=bm.float64)
    weights = geometry.linear_owner_weight()
    expected = (
        weights * values[geometry.owner]
        + (1.0 - weights) * values[geometry.neighbour]
    )

    actual = interpolate_cell_to_face(
        values,
        geometry=geometry,
        method="linear",
    )

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


@pytest.mark.parametrize("mesh", _meshes())
def test_face_interpolation_owner_weight_is_exported_from_geometry(mesh):
    np.testing.assert_allclose(
        np.asarray(geometry_face_interpolation_owner_weight(mesh, method="linear")),
        np.asarray(FVMGeometry(mesh).linear_owner_weight()),
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
