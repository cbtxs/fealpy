import inspect
import numpy as np
import pytest

from fealpy.model.poisson.exp0002 import Exp0002
from fealpy.mesh import QuadrangleMesh, TetrahedronMesh
from fealpy.fvm import FVMGeometry, GradientReconstruct
from fealpy.fvm.gradient_reconstruct import (
    GreenGaussGradientReconstruct,
    LSQGradientReconstruct,
    least_squares_rhs,
)


def test_lsq_recovers_linear_gradient_on_quad_and_tri_meshes():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(mesh).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_gradient_reconstruct_reuses_supplied_fvm_geometry(monkeypatch):
    import fealpy.fvm.gradient_reconstruct as gradient_module

    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1]

    def forbidden_geometry(*args, **kwargs):
        raise AssertionError("GradientReconstruct should reuse supplied FVMGeometry")

    monkeypatch.setattr(gradient_module, "FVMGeometry", forbidden_geometry)

    reconstruct = GradientReconstruct(mesh, geometry=geometry)
    grad = np.asarray(reconstruct.cell_gradient(field))

    assert reconstruct.fvm_geometry is geometry
    assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_least_squares_rhs_matches_stencil_accumulation_for_scalar_and_vector_fields():
    N = np.array([
        [0, 1, 2, 1],
        [1, 0, 2, 2],
        [2, 0, 1, 0],
    ])
    weighted_d = np.array([
        [[0.0, 0.0], [1.0, 0.2], [0.4, 0.8], [-0.3, 0.5]],
        [[0.0, 0.0], [-0.6, 0.1], [0.7, -0.2], [0.2, 0.9]],
        [[0.0, 0.0], [-0.4, -0.8], [0.3, -0.5], [0.5, 0.4]],
    ])

    scalar = np.array([1.0, 2.5, -0.5])
    expected_scalar = np.zeros((3, 2))
    for k in range(N.shape[1]):
        expected_scalar += (scalar[N[:, k]] - scalar)[:, None] * weighted_d[:, k, :]

    vector = np.stack([scalar, -2.0 * scalar + 0.5], axis=-1)
    expected_vector = np.zeros((3, 2, 2))
    for k in range(N.shape[1]):
        delta = vector[N[:, k]] - vector
        expected_vector += delta[:, :, None] * weighted_d[:, k, None, :]

    np.testing.assert_allclose(least_squares_rhs(scalar, N, weighted_d), expected_scalar)
    np.testing.assert_allclose(least_squares_rhs(vector, N, weighted_d), expected_vector)


def test_layered_lsq_recovers_3d_linear_gradient_on_tetra_mesh():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(GradientReconstruct(mesh).cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())
    assert np.linalg.norm(grad - np.array([1.0, 2.0, 3.0])) < 1.0e-10


def test_face_weighted_lsq_recovers_3d_linear_gradient_on_tetra_interior_cells():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    def gd(points):
        return points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(
        GradientReconstruct(mesh, method="face_weighted_lsq", gd=gd).cell_gradient(field)
    )
    geometry = FVMGeometry(mesh)
    interior = np.ones(mesh.number_of_cells(), dtype=bool)
    interior[np.asarray(geometry.owner[geometry.is_boundary])] = False

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())
    assert np.linalg.norm(grad[interior] - np.array([1.0, 2.0, 3.0])) < 1.0e-10


def test_green_gauss_allocates_3d_gradient_shape_on_tetra_mesh():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    field = np.ones(mesh.number_of_cells())

    grad = np.asarray(GradientReconstruct(mesh, method="green_gauss").cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())


def test_layered_lsq_layer_weights_keep_linear_consistency():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(
            mesh, layer_weights=(1.0, 0.25)
        ).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_layered_lsq_layer_weights_change_nonlinear_reconstruction():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2
    equal = np.asarray(
        GradientReconstruct(mesh, layer_weights=(1.0, 1.0)).cell_gradient(field)
    )
    first_layer_heavy = np.asarray(
        GradientReconstruct(mesh, layer_weights=(1.0, 0.05)).cell_gradient(field)
    )

    assert np.linalg.norm(equal - first_layer_heavy) > 1.0e-4


def test_layered_lsq_reuses_geometry_between_fields(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1] + 3.0
    second_field = -0.5 * points[:, 0] + points[:, 1] - 1.0
    reconstruct = GradientReconstruct(mesh)

    call_count = 0
    original_cell_to_cell = mesh.cell_to_cell

    def counted_cell_to_cell(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_cell_to_cell(*args, **kwargs)

    monkeypatch.setattr(mesh, "cell_to_cell", counted_cell_to_cell)

    first_grad = np.asarray(reconstruct.cell_gradient(first_field))
    second_grad = np.asarray(reconstruct.cell_gradient(second_field))

    assert call_count == 1
    assert np.linalg.norm(first_grad - np.array([1.0, 2.0])) < 1.0e-10
    assert np.linalg.norm(second_grad - np.array([-0.5, 1.0])) < 1.0e-10


def test_layered_lsq_reuses_cached_matrix_without_dirichlet_boundary():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0
    reconstruct = GradientReconstruct(mesh)

    reconstruct.cell_gradient(field)
    cached_matrix = reconstruct.lsq_reconstruct._layered_lsq_cache[2]
    reconstruct.cell_gradient(field)
    matrix = reconstruct.lsq_reconstruct._layered_lsq_cache[2]

    assert matrix is cached_matrix


def test_face_weighted_lsq_reuses_cached_inverse_between_fields(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1]
    second_field = points[:, 0] ** 2 - points[:, 1]
    reconstruct = GradientReconstruct(mesh, method="face_weighted_lsq")

    call_count = 0
    original = LSQGradientReconstruct.invert_lsq_matrix

    def counting_invert(self, A, method):
        nonlocal call_count
        call_count += 1
        return original(self, A, method)

    monkeypatch.setattr(LSQGradientReconstruct, "invert_lsq_matrix", counting_invert)

    reconstruct.cell_gradient(first_field)
    reconstruct.cell_gradient(second_field)

    assert call_count == 1


def test_face_weighted_lsq_accumulates_internal_rhs_in_one_scatter(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1]
    reconstruct = GradientReconstruct(mesh, method="face_weighted_lsq")

    call_count = 0
    original = LSQGradientReconstruct.add_lsq_rhs_samples

    def counting_add_rhs(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LSQGradientReconstruct, "add_lsq_rhs_samples", counting_add_rhs)

    reconstruct.cell_gradient(field)

    assert call_count == 1


def test_face_weighted_lsq_reuses_internal_scatter_geometry_between_fields(monkeypatch):
    import fealpy.fvm.gradient_reconstruct as gradient_module

    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1]
    second_field = points[:, 0] ** 2 - points[:, 1]
    reconstruct = GradientReconstruct(mesh, method="face_weighted_lsq")

    reconstruct.cell_gradient(first_field)
    call_count = 0
    original = gradient_module.bm.concatenate

    def counting_concatenate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gradient_module.bm, "concatenate", counting_concatenate)

    reconstruct.cell_gradient(second_field)

    assert call_count == 0


def test_face_weighted_lsq_reuses_boundary_geometry_between_fields(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1]
    second_field = -points[:, 0] + 0.5 * points[:, 1]

    def gd(points):
        return points[:, 0] + 2.0 * points[:, 1]

    reconstruct = GradientReconstruct(mesh, method="face_weighted_lsq", gd=gd)
    call_count = 0
    import fealpy.fvm.gradient_reconstruct as gradient_module

    original = gradient_module.selected_boundary_faces

    def counting_selected_boundary_faces(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        gradient_module,
        "selected_boundary_faces",
        counting_selected_boundary_faces,
    )

    reconstruct.cell_gradient(first_field)
    reconstruct.cell_gradient(second_field)

    assert call_count == 1


def test_green_gauss_accumulates_internal_flux_in_one_scatter(monkeypatch):
    import fealpy.fvm.gradient_reconstruct as gradient_module

    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.ones(mesh.number_of_cells())
    reconstruct = GradientReconstruct(mesh, method="green_gauss")

    call_count = 0
    original = gradient_module.bm.index_add

    def counting_index_add(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gradient_module.bm, "index_add", counting_index_add)

    reconstruct.cell_gradient(field)

    assert call_count == 1


def test_green_gauss_variant_delegates_to_green_gauss_reconstructor(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.ones(mesh.number_of_cells())
    reconstruct = GradientReconstruct(mesh, method="green_gauss")

    call_count = 0

    def counting_green_gauss(self, U):
        nonlocal call_count
        call_count += 1
        return np.zeros((mesh.number_of_cells(), 2))

    monkeypatch.setattr(
        GreenGaussGradientReconstruct,
        "green_gauss",
        counting_green_gauss,
    )

    reconstruct.cell_gradient(field)

    assert call_count == 1


def test_layered_lsq_uses_dirichlet_boundary_data_when_given():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def one(points):
        return np.ones(points.shape[0])

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=one,
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    boundary_faces = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[boundary_faces]
    owner = mesh.edge_to_cell()[boundary_faces, 0]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=1) > 1.0e-12)[0].tolist())

    assert nonzero_owner == left_owner


def test_layered_lsq_boundary_weight_zero_keeps_default_result():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1] ** 2

    def shifted_value(points):
        return 10.0 + points[:, 0]

    default = np.asarray(GradientReconstruct(mesh).cell_gradient(field))
    with_zero_weight = np.asarray(GradientReconstruct(
        mesh,
        gd=shifted_value,
        boundary_weight=0.0,
    ).cell_gradient(field))

    assert np.linalg.norm(with_zero_weight - default) < 1.0e-12


def test_layered_lsq_neumann_boundary_data_enforces_normal_derivative():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def normal_derivative(points):
        return np.ones(points.shape[0])

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=normal_derivative,
        bc_type="neumann",
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    boundary_faces = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[boundary_faces]
    selected = np.abs(face_center[:, 0]) < 1.0e-12
    selected_faces = boundary_faces[selected]
    owner = mesh.edge_to_cell()[selected_faces, 0]
    unit_normal = (
        mesh.edge_normal()[selected_faces]
        / mesh.entity_measure("face")[selected_faces, None]
    )
    normal_component = np.einsum("ij,ij->i", grad[owner], unit_normal)

    assert np.linalg.norm(normal_component - 1.0) < 1.0e-12


def test_layered_lsq_neumann_boundary_data_supports_vector_fields():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros((mesh.number_of_cells(), 2))

    def vector_normal_derivative(points):
        return np.stack([
            np.ones(points.shape[0]),
            2.0 * np.ones(points.shape[0]),
        ], axis=-1)

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=vector_normal_derivative,
        bc_type="neumann",
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    boundary_faces = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[boundary_faces]
    selected = np.abs(face_center[:, 0]) < 1.0e-12
    selected_faces = boundary_faces[selected]
    owner = mesh.edge_to_cell()[selected_faces, 0]
    unit_normal = (
        mesh.edge_normal()[selected_faces]
        / mesh.entity_measure("face")[selected_faces, None]
    )
    normal_component = np.einsum("icd,id->ic", grad[owner], unit_normal)

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert np.linalg.norm(normal_component - np.array([1.0, 2.0])) < 1.0e-12


def test_layered_lsq_dirichlet_boundary_data_supports_vector_fields():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros((mesh.number_of_cells(), 2))

    def vector_value(points):
        return np.stack([
            np.ones(points.shape[0]),
            2.0 * np.ones(points.shape[0]),
        ], axis=-1)

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=vector_value,
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
    ).cell_gradient(field))

    boundary_faces = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[boundary_faces]
    owner = mesh.edge_to_cell()[boundary_faces, 0]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=(1, 2)) > 1.0e-12)[0].tolist())

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert nonzero_owner == left_owner


def test_layered_lsq_rejects_invalid_layer_weights():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = mesh.entity_barycenter("cell")[:, 0]
    reconstruct = GradientReconstruct(mesh)

    with pytest.raises(ValueError, match="layer_weights"):
        GradientReconstruct(mesh, layer_weights=(1.0,)).cell_gradient(field)

    with pytest.raises(ValueError, match="non-negative"):
        GradientReconstruct(mesh, layer_weights=(1.0, -1.0)).cell_gradient(field)


def test_face_weighted_lsq_matches_reference():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    def gd(points):
        return points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    grad = np.asarray(GradientReconstruct(
        mesh,
        method="face_weighted_lsq",
        gd=gd,
    ).cell_gradient(field))
    expected = _face_weighted_lsq_reference(mesh, field, gd=gd)

    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_lsq_common_helpers_document_normal_equations():
    source = inspect.getsource(LSQGradientReconstruct)

    assert hasattr(LSQGradientReconstruct, "add_lsq_matrix_samples")
    assert hasattr(LSQGradientReconstruct, "add_lsq_rhs_samples")
    assert "A_K = sum" in source
    assert "b_K = sum" in source


def test_face_weighted_lsq_recovers_linear_gradient_with_boundary_values():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        def gd(points):
            return points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(
            mesh,
            method="face_weighted_lsq",
            gd=gd,
        ).cell_gradient(field))

        expected = _face_weighted_lsq_reference(mesh, field, gd=gd)
        interior = _interior_cell_mask(mesh)

        assert np.linalg.norm(grad - expected) < 1.0e-12
        assert np.linalg.norm(grad[interior] - np.array([1.0, 2.0])) < 1.0e-10
        if meshtype == "uniform_quad":
            assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_face_weighted_lsq_supports_vector_fields():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = np.stack([
        points[:, 0] + 2.0 * points[:, 1],
        -0.5 * points[:, 0] + points[:, 1],
    ], axis=-1)

    def gd(points):
        return np.stack([
            points[:, 0] + 2.0 * points[:, 1],
            -0.5 * points[:, 0] + points[:, 1],
        ], axis=-1)

    grad = np.asarray(GradientReconstruct(
        mesh,
        method="face_weighted_lsq",
        gd=gd,
    ).cell_gradient(field))

    expected = _face_weighted_lsq_reference(mesh, field, gd=gd)
    interior = _interior_cell_mask(mesh)

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert np.linalg.norm(grad - expected) < 1.0e-12
    assert np.linalg.norm(grad[interior, 0, :] - np.array([1.0, 2.0])) < 1.0e-10
    assert np.linalg.norm(grad[interior, 1, :] - np.array([-0.5, 1.0])) < 1.0e-10


def test_face_weighted_lsq_uses_patch_normal_delta_on_skewed_boundary():
    node = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.4, 1.0],
        [0.0, 1.0],
    ])
    cell = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadrangleMesh(node, cell)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 + 0.5 * points[:, 1]

    def gd(points):
        return points[:, 0] ** 2 + 0.5 * points[:, 1]

    grad = np.asarray(GradientReconstruct(
        mesh,
        method="face_weighted_lsq",
        gd=gd,
    ).cell_gradient(field))
    expected = _face_weighted_lsq_reference(
        mesh,
        field,
        gd=gd,
        boundary_delta="normal",
    )
    full_delta = _face_weighted_lsq_reference(
        mesh,
        field,
        gd=gd,
        boundary_delta="full",
    )

    assert np.linalg.norm(full_delta - expected) > 1.0e-3
    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_green_gauss_neumann_uses_true_normal_distance_on_skewed_quad():
    node = np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.4, 1.0],
        [0.0, 1.0],
    ])
    cell = np.array([[0, 1, 2, 3]], dtype=np.int32)
    mesh = QuadrangleMesh(node, cell)
    field = np.zeros(mesh.number_of_cells())

    def normal_derivative(points):
        return np.ones(points.shape[0])

    grad = np.asarray(GradientReconstruct(
        mesh,
        method="green_gauss",
        gd=normal_derivative,
        bc_type="neumann",
        threshold=lambda points: np.abs(points[:, 1]) < 1.0e-12,
    ).cell_gradient(field))

    boundary_faces = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[boundary_faces]
    selected = np.abs(face_center[:, 1]) < 1.0e-12
    selected_faces = boundary_faces[selected]
    owner = mesh.edge_to_cell()[selected_faces, 0]
    cell_center = mesh.entity_barycenter("cell")
    face_measure = mesh.entity_measure("face")
    face_normal = mesh.edge_normal()
    unit_normal = face_normal[selected_faces] / face_measure[selected_faces, None]
    normal_distance = np.abs(np.einsum(
        "ij,ij->i",
        face_center[selected] - cell_center[owner],
        unit_normal,
    ))

    expected = np.zeros_like(grad)
    np.add.at(
        expected,
        owner,
        normal_distance[:, None] * face_normal[selected_faces],
    )
    expected /= mesh.entity_measure("cell")[:, None]

    assert np.linalg.norm(grad - expected) < 1.0e-12


def _face_weighted_lsq_reference(
    mesh,
    U,
    gd=None,
    bc_type="dirichlet",
    boundary_delta="normal",
):
    U = np.asarray(U)
    NC = mesh.number_of_cells()
    cell_centers = np.asarray(mesh.entity_barycenter("cell"))
    face_centers = np.asarray(mesh.entity_barycenter("face"))
    face_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    owner = face_to_cell[:, 0]
    neighbour = face_to_cell[:, 1]
    is_internal = owner != neighbour
    Sf = np.asarray(mesh.edge_normal())
    magSf = np.linalg.norm(Sf, axis=1)
    owner_weight = _linear_owner_weight_reference(mesh)

    A = np.zeros((NC, 2, 2), dtype=float)
    if U.ndim == 1:
        b = np.zeros((NC, 2), dtype=float)
    else:
        b = np.zeros((NC, U.shape[1], 2), dtype=float)

    face = np.nonzero(is_internal)[0]
    own = owner[face]
    nei = neighbour[face]
    d = cell_centers[nei] - cell_centers[own]
    scale = magSf[face] / np.einsum("ij,ij->i", d, d)
    outer = scale[:, None, None] * np.einsum("ni,nj->nij", d, d)
    w = owner_weight[face]
    np.add.at(A, own, (1.0 - w)[:, None, None] * outer)
    np.add.at(A, nei, w[:, None, None] * outer)
    delta = U[nei] - U[own]
    if U.ndim == 1:
        rhs = scale[:, None] * delta[:, None] * d
    else:
        rhs = scale[:, None, None] * delta[:, :, None] * d[:, None, :]
    np.add.at(b, own, (1.0 - w)[:, None] * rhs if U.ndim == 1 else (1.0 - w)[:, None, None] * rhs)
    np.add.at(b, nei, w[:, None] * rhs if U.ndim == 1 else w[:, None, None] * rhs)

    boundary_faces = mesh.boundary_face_index()
    boundary_owner = owner[boundary_faces]
    boundary_d = face_centers[boundary_faces] - cell_centers[boundary_owner]
    if boundary_delta == "normal":
        unit_normal = Sf[boundary_faces] / magSf[boundary_faces, None]
        boundary_d = unit_normal * np.einsum(
            "ij,ij->i", unit_normal, boundary_d
        )[:, None]
    elif boundary_delta != "full":
        raise ValueError("boundary_delta must be 'normal' or 'full'.")
    boundary_scale = magSf[boundary_faces] / np.einsum(
        "ij,ij->i", boundary_d, boundary_d
    )
    boundary_outer = boundary_scale[:, None, None] * np.einsum(
        "ni,nj->nij", boundary_d, boundary_d
    )
    np.add.at(A, boundary_owner, boundary_outer)
    if gd is not None:
        boundary_value = gd(face_centers[boundary_faces])
        if bc_type == "neumann":
            unit_normal = Sf[boundary_faces] / magSf[boundary_faces, None]
            normal_distance = np.abs(
                np.einsum("ij,ij->i", boundary_d, unit_normal)
            )
            boundary_value = U[boundary_owner] + boundary_value * normal_distance
        boundary_delta_u = boundary_value - U[boundary_owner]
        if U.ndim == 1:
            boundary_rhs = (
                boundary_scale[:, None] * boundary_delta_u[:, None] * boundary_d
            )
        else:
            boundary_rhs = (
                boundary_scale[:, None, None]
                * boundary_delta_u[:, :, None]
                * boundary_d[:, None, :]
            )
        np.add.at(b, boundary_owner, boundary_rhs)

    if U.ndim == 1:
        return np.stack([np.linalg.solve(A[cell], b[cell]) for cell in range(NC)])
    return np.stack([
        np.stack([
            np.linalg.solve(A[cell], b[cell, component])
            for cell in range(NC)
        ])
        for component in range(U.shape[1])
    ], axis=1)


def _linear_owner_weight_reference(mesh):
    face_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    owner = face_to_cell[:, 0]
    neighbour = face_to_cell[:, 1]
    face_centers = np.asarray(mesh.entity_barycenter("face"))
    cell_centers = np.asarray(mesh.entity_barycenter("cell"))
    Sf = np.asarray(mesh.edge_normal())
    owner_dist = np.abs(np.einsum("ij,ij->i", Sf, face_centers - cell_centers[owner]))
    neighbour_dist = np.abs(
        np.einsum("ij,ij->i", Sf, cell_centers[neighbour] - face_centers)
    )
    total_dist = owner_dist + neighbour_dist
    weight = np.where(total_dist > 0.0, neighbour_dist / total_dist, 0.5)
    return np.where(owner != neighbour, weight, 1.0)


def _interior_cell_mask(mesh):
    mask = np.ones(mesh.number_of_cells(), dtype=bool)
    face_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    boundary_faces = np.asarray(mesh.boundary_face_index())
    mask[face_to_cell[boundary_faces, 0]] = False
    return mask
