import inspect
import numpy as np
import pytest

from fealpy.model.poisson.exp0002 import Exp0002
from fealpy.mesh import QuadrangleMesh, TetrahedronMesh
from fealpy.fvm import FVMGeometry, GradientReconstruct
from fealpy.fvm.gradient_reconstruct import (
    GreenGaussGradientReconstruct,
    LSQGradientReconstruct,
)


def test_lsq_recovers_linear_gradient_on_quad_and_tri_meshes():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(mesh).cell_gradient(field))

        assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_extended_lsq_recovers_3d_linear_gradient_on_tetra_mesh():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(GradientReconstruct(mesh).cell_gradient(field))

    assert grad.shape == (mesh.number_of_cells(), mesh.geo_dimension())
    assert np.linalg.norm(grad - np.array([1.0, 2.0, 3.0])) < 1.0e-10


def test_weighted_lsq_recovers_3d_linear_gradient_on_tetra_interior_cells():
    mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    def gd(points):
        return points[:, 0] + 2.0 * points[:, 1] + 3.0 * points[:, 2]

    grad = np.asarray(
        GradientReconstruct(mesh, method="weighted_lsq", gd=gd).cell_gradient(field)
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
    reconstruct.cell_gradient(field)
    matrix = reconstruct._extended_lsq_cache[2]

    assert matrix is cached_matrix


def test_weighted_lsq_reuses_cached_inverse_between_fields(monkeypatch):
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    first_field = points[:, 0] + 2.0 * points[:, 1]
    second_field = points[:, 0] ** 2 - points[:, 1]
    reconstruct = GradientReconstruct(mesh, method="weighted_lsq")

    call_count = 0
    original = LSQGradientReconstruct._invert_lsq_matrix

    def counting_invert(self, A, method):
        nonlocal call_count
        call_count += 1
        return original(self, A, method)

    monkeypatch.setattr(LSQGradientReconstruct, "_invert_lsq_matrix", counting_invert)

    reconstruct.cell_gradient(first_field)
    reconstruct.cell_gradient(second_field)

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


def test_extended_lsq_uses_dirichlet_boundary_data_when_given():
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
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
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
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
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
        threshold=lambda points: np.abs(points[:, 0]) < 1.0e-12,
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


def test_weighted_lsq_matches_reference():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    def gd(points):
        return points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    grad = np.asarray(GradientReconstruct(
        mesh,
        method="weighted_lsq",
        gd=gd,
    ).cell_gradient(field))
    expected = _weighted_lsq_reference(mesh, field, gd=gd)

    assert np.linalg.norm(grad - expected) < 1.0e-12


def test_legacy_weighted_lsq_alias_name_is_removed():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_tri"](nx=4, ny=4)
    points = mesh.entity_barycenter("cell")
    field = points[:, 0] ** 2 - 0.5 * points[:, 1] ** 2 + points[:, 0] * points[:, 1]

    with pytest.raises(ValueError, match="Unknown cell_gradient variant"):
        GradientReconstruct(mesh, method="openfoam" + "_least_squares").cell_gradient(
            field
        )


def test_gradient_reconstruct_uses_generic_internal_names():
    pde = Exp0002()
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    reconstruct = GradientReconstruct(mesh, method="weighted_lsq")

    internal_names = {
        name for name in dir(reconstruct)
        if name.startswith("_openfoam") or name == "_openfoam_lsq_cache"
    }

    assert internal_names == set()


def test_lsq_common_helpers_document_normal_equations():
    source = inspect.getsource(LSQGradientReconstruct)

    assert hasattr(LSQGradientReconstruct, "_add_lsq_matrix_samples")
    assert hasattr(LSQGradientReconstruct, "_add_lsq_rhs_samples")
    assert not hasattr(GradientReconstruct, "_add_lsq_matrix_samples")
    assert not hasattr(GradientReconstruct, "_add_lsq_rhs_samples")
    assert not hasattr(GradientReconstruct, "_weighted_lsq_coefficients")
    assert not hasattr(GradientReconstruct, "_weighted_lsq_geometry")
    assert not hasattr(GradientReconstruct, "_threshold_coordinate_axis")
    assert not hasattr(GradientReconstruct, "_validate_boundary_face_flag")
    assert "A_K = sum" in source
    assert "b_K = sum" in source


def test_weighted_lsq_recovers_linear_gradient_with_boundary_values():
    pde = Exp0002()

    for meshtype in ["uniform_quad", "uniform_tri"]:
        mesh = pde.init_mesh[meshtype](nx=4, ny=4)
        points = mesh.entity_barycenter("cell")
        field = points[:, 0] + 2.0 * points[:, 1] + 3.0

        def gd(points):
            return points[:, 0] + 2.0 * points[:, 1] + 3.0

        grad = np.asarray(GradientReconstruct(
            mesh,
            method="weighted_lsq",
            gd=gd,
        ).cell_gradient(field))

        expected = _weighted_lsq_reference(mesh, field, gd=gd)
        interior = _interior_cell_mask(mesh)

        assert np.linalg.norm(grad - expected) < 1.0e-12
        assert np.linalg.norm(grad[interior] - np.array([1.0, 2.0])) < 1.0e-10
        if meshtype == "uniform_quad":
            assert np.linalg.norm(grad - np.array([1.0, 2.0])) < 1.0e-10


def test_weighted_lsq_supports_vector_fields():
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
        method="weighted_lsq",
        gd=gd,
    ).cell_gradient(field))

    expected = _weighted_lsq_reference(mesh, field, gd=gd)
    interior = _interior_cell_mask(mesh)

    assert grad.shape == (mesh.number_of_cells(), 2, 2)
    assert np.linalg.norm(grad - expected) < 1.0e-12
    assert np.linalg.norm(grad[interior, 0, :] - np.array([1.0, 2.0])) < 1.0e-10
    assert np.linalg.norm(grad[interior, 1, :] - np.array([-0.5, 1.0])) < 1.0e-10


def test_weighted_lsq_uses_patch_normal_delta_on_skewed_boundary():
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
        method="weighted_lsq",
        gd=gd,
    ).cell_gradient(field))
    expected = _weighted_lsq_reference(
        mesh,
        field,
        gd=gd,
        boundary_delta="normal",
    )
    full_delta = _weighted_lsq_reference(
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
    assert not hasattr(reconstruct, "face_gradient")
    assert not hasattr(reconstruct, "LSQ")
    assert not hasattr(reconstruct, "AverageGradientreDirichlet")
    assert not hasattr(reconstruct, "AverageGradientreNeumann")
    assert not hasattr(reconstruct, "reconstruct")
    assert not hasattr(reconstruct, "QuadraticLSQ")
    assert not hasattr(reconstruct, "OpenFOAMLSQ")


def _weighted_lsq_reference(
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
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    owner = e2c[:, 0]
    neighbour = e2c[:, 1]
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

    bdedge = mesh.boundary_face_index()
    bd_owner = owner[bdedge]
    bd_d = face_centers[bdedge] - cell_centers[bd_owner]
    if boundary_delta == "normal":
        unit_normal = Sf[bdedge] / magSf[bdedge, None]
        bd_d = unit_normal * np.einsum("ij,ij->i", unit_normal, bd_d)[:, None]
    elif boundary_delta != "full":
        raise ValueError("boundary_delta must be 'normal' or 'full'.")
    bd_scale = magSf[bdedge] / np.einsum("ij,ij->i", bd_d, bd_d)
    bd_outer = bd_scale[:, None, None] * np.einsum("ni,nj->nij", bd_d, bd_d)
    np.add.at(A, bd_owner, bd_outer)
    if gd is not None:
        bd_value = gd(face_centers[bdedge])
        if bc_type == "neumann":
            unit_normal = Sf[bdedge] / magSf[bdedge, None]
            normal_distance = np.abs(np.einsum("ij,ij->i", bd_d, unit_normal))
            bd_value = U[bd_owner] + bd_value * normal_distance
        bd_delta = bd_value - U[bd_owner]
        if U.ndim == 1:
            bd_rhs = bd_scale[:, None] * bd_delta[:, None] * bd_d
        else:
            bd_rhs = bd_scale[:, None, None] * bd_delta[:, :, None] * bd_d[:, None, :]
        np.add.at(b, bd_owner, bd_rhs)

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
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    owner = e2c[:, 0]
    neighbour = e2c[:, 1]
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
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    bdedge = np.asarray(mesh.boundary_face_index())
    mask[e2c[bdedge, 0]] = False
    return mask
