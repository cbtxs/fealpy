import numpy as np
import pytest

from fealpy.model.poisson.exp0002 import Exp0002
from fealpy.mesh import QuadrangleMesh
from fealpy.fvm import GradientReconstruct


def test_lsq_recovers_linear_gradient_on_quad_and_tri_meshes():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(mesh).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_extended_lsq_layer_weights_keep_linear_consistency():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(
            mesh, layer_weights=(1.0, 0.25)
        ).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_extended_lsq_layer_weights_change_nonlinear_reconstruction():
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


def test_extended_lsq_reuses_geometry_between_fields(monkeypatch):
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


def test_extended_lsq_reuses_cached_matrix_without_dirichlet_boundary():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0
    reconstruct = GradientReconstruct(mesh)

    reconstruct.cell_gradient(field)
    cached_matrix = reconstruct._extended_lsq_cache[2]
    matrix, _, _ = reconstruct._build_extended_lsq_system(field)

    assert matrix is cached_matrix


def test_extended_lsq_uses_dirichlet_boundary_data_when_given():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def one(points):
        return np.ones(points.shape[0])

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=one,
        threshold=lambda x: np.abs(x) < 1.0e-12,
    ).cell_gradient(field))

    bd_edge = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[bd_edge]
    owner = mesh.edge_to_cell()[bd_edge, 0]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=1) > 1.0e-12)[0].tolist())

    assert nonzero_owner == left_owner


def test_extended_lsq_boundary_weight_zero_keeps_default_result():
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


def test_extended_lsq_neumann_boundary_data_enforces_normal_derivative():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = np.zeros(mesh.number_of_cells())

    def normal_derivative(points):
        return np.ones(points.shape[0])

    grad = np.asarray(GradientReconstruct(
        mesh,
        gd=normal_derivative,
        bc_type="neumann",
        threshold=lambda x: np.abs(x) < 1.0e-12,
    ).cell_gradient(field))

    bd_edge = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[bd_edge]
    selected = np.abs(face_center[:, 0]) < 1.0e-12
    selected_edge = bd_edge[selected]
    owner = mesh.edge_to_cell()[selected_edge, 0]
    unit_normal = (
        mesh.edge_normal()[selected_edge]
        / mesh.entity_measure("face")[selected_edge, None]
    )
    normal_component = np.einsum("ij,ij->i", grad[owner], unit_normal)

    assert np.linalg.norm(normal_component - 1.0) < 1.0e-12


def test_extended_lsq_neumann_boundary_data_supports_vector_fields():
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
        threshold=lambda x: np.abs(x) < 1.0e-12,
    ).cell_gradient(field))

    bd_edge = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[bd_edge]
    selected = np.abs(face_center[:, 0]) < 1.0e-12
    selected_edge = bd_edge[selected]
    owner = mesh.edge_to_cell()[selected_edge, 0]
    unit_normal = (
        mesh.edge_normal()[selected_edge]
        / mesh.entity_measure("face")[selected_edge, None]
    )
    normal_component = np.einsum("icd,id->ic", grad[owner], unit_normal)

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert np.linalg.norm(normal_component - np.array([1.0, 2.0])) < 1.0e-12


def test_extended_lsq_dirichlet_boundary_data_supports_vector_fields():
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
        threshold=lambda x: np.abs(x) < 1.0e-12,
    ).cell_gradient(field))

    bd_edge = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[bd_edge]
    owner = mesh.edge_to_cell()[bd_edge, 0]
    left_owner = set(np.asarray(owner[np.abs(face_center[:, 0]) < 1.0e-12]).tolist())
    nonzero_owner = set(np.where(np.linalg.norm(grad, axis=(1, 2)) > 1.0e-12)[0].tolist())

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert nonzero_owner == left_owner


def test_extended_lsq_rejects_invalid_layer_weights():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    field = mesh.entity_barycenter("cell")[:, 0]
    reconstruct = GradientReconstruct(mesh)

    with pytest.raises(ValueError, match="layer_weights"):
        GradientReconstruct(mesh, layer_weights=(1.0,)).cell_gradient(field)

    with pytest.raises(ValueError, match="non-negative"):
        GradientReconstruct(mesh, layer_weights=(1.0, -1.0)).cell_gradient(field)


def test_face_lsq_recovers_linear_gradient_on_quad_mesh():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0

    grad = np.asarray(GradientReconstruct(
        mesh, method="face_lsq"
    ).cell_gradient(field))

    assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_face_lsq_recovers_linear_gradient_on_tri_mesh_with_dirichlet_value():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0

    def gd(points):
        return points[:, 0] + 2.0 * points[:, 1] + 3.0

    grad = np.asarray(GradientReconstruct(
        mesh, method="face_lsq", gd=gd
    ).cell_gradient(field))

    assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_face_lsq_reports_rank_deficient_stencil_without_boundary_value():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0

    with pytest.raises(ValueError, match="rank deficient"):
        GradientReconstruct(mesh, method="face_lsq").cell_gradient(field)


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

    bd_edge = mesh.boundary_face_index()
    face_center = mesh.entity_barycenter("face")[bd_edge]
    selected = np.abs(face_center[:, 1]) < 1.0e-12
    selected_edge = bd_edge[selected]
    owner = mesh.edge_to_cell()[selected_edge, 0]
    cell_center = mesh.entity_barycenter("cell")
    face_measure = mesh.entity_measure("face")
    face_normal = mesh.edge_normal()
    unit_normal = face_normal[selected_edge] / face_measure[selected_edge, None]
    normal_distance = np.abs(np.einsum(
        "ij,ij->i",
        face_center[selected] - cell_center[owner],
        unit_normal,
    ))

    expected = np.zeros_like(grad)
    np.add.at(
        expected,
        owner,
        normal_distance[:, None] * face_normal[selected_edge],
    )
    expected /= mesh.entity_measure("cell")[:, None]

    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_gradient_reconstruct_keeps_only_variant_api():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    reconstruct = GradientReconstruct(mesh)

    assert hasattr(reconstruct, "cell_gradient")
    assert hasattr(reconstruct, "face_gradient")
    assert not hasattr(reconstruct, "LSQ")
    assert not hasattr(reconstruct, "AverageGradientreDirichlet")
    assert not hasattr(reconstruct, "AverageGradientreNeumann")
    assert not hasattr(reconstruct, "reconstruct")
    assert not hasattr(reconstruct, "QuadraticLSQ")
    assert not hasattr(reconstruct, "OpenFOAMLSQ")
