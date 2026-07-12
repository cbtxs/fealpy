import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fvm import (
    CrossDiffusionRHSAssembler,
    FVMGeometry,
    ScalarCrossDiffusionIntegrator,
)
from fealpy.mesh import TriangleMesh


def _box_space(nx=1, ny=1):
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=ny)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def _bad_two_triangle_space():
    bm.set_backend("numpy")
    a = 0.01
    y = 1.0
    node = bm.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [-a, -y],
            [a, y],
        ],
        dtype=bm.float64,
    )
    cell = bm.array([[0, 1, 2], [0, 3, 1]], dtype=bm.int32)
    mesh = TriangleMesh(node, cell)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def _expected_scalar_scatter(mesh, face_flux):
    face_to_cell = np.asarray(mesh.edge_to_cell()[:, :2], dtype=np.int64)
    expected = np.zeros(mesh.number_of_cells(), dtype=float)
    internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    np.add.at(expected, face_to_cell[:, 0], face_flux)
    np.add.at(expected, face_to_cell[internal, 1], -face_flux[internal])
    return expected


def _expected_vector_scatter(mesh, face_flux):
    face_to_cell = np.asarray(mesh.edge_to_cell()[:, :2], dtype=np.int64)
    expected = np.zeros((mesh.number_of_cells(), face_flux.shape[1]), dtype=float)
    internal = face_to_cell[:, 0] != face_to_cell[:, 1]
    np.add.at(expected, face_to_cell[:, 0], face_flux)
    np.add.at(expected, face_to_cell[internal, 1], -face_flux[internal])
    return expected


def test_face_flux_correction_scatter_scalar():
    mesh, space = _box_space(nx=1, ny=1)
    face_flux = np.linspace(0.2, 1.2, mesh.number_of_edges())

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_face_flux_correction_scatter_reuses_fvm_geometry(monkeypatch):
    mesh, space = _box_space(nx=2, ny=1)
    face_flux = np.linspace(0.2, 1.2, mesh.number_of_edges())
    calls = []
    original_scatter = FVMGeometry.scatter_face_flux_to_cells

    def counted_scatter(self, selected_face_flux):
        calls.append(np.asarray(selected_face_flux).copy())
        return original_scatter(self, selected_face_flux)

    monkeypatch.setattr(FVMGeometry, "scatter_face_flux_to_cells", counted_scatter)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], face_flux, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_face_flux_correction_scatter_vector():
    mesh, scalar_space = _box_space(nx=1, ny=1)
    space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    face_flux = np.stack(
        [
            np.linspace(0.2, 1.2, mesh.number_of_edges()),
            np.linspace(-0.4, 0.6, mesh.number_of_edges()),
        ],
        axis=1,
    )

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs).reshape(2, mesh.number_of_cells()).T,
        _expected_vector_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_integrator_assembly_delegates_face_flux_construction(monkeypatch):
    import fealpy.fvm.scalar_cross_diffusion_integrator as cross_module

    mesh, space = _box_space(nx=1, ny=1)
    face_flux = np.linspace(0.2, 1.2, mesh.number_of_edges())
    calls = []

    def counted_face_flux(space_arg, geometry, face_to_cell, **kwargs):
        calls.append((space_arg, geometry, face_to_cell, kwargs))
        return face_flux

    monkeypatch.setattr(
        cross_module,
        "scalar_cross_diffusion_face_flux",
        counted_face_flux,
    )

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            np.ones((mesh.number_of_edges(), mesh.geo_dimension())),
        )
    ).assembly()

    assert len(calls) == 1
    assert calls[0][0] is space
    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_default_cross_diffusion_matches_bounded_over_relaxed_scatter():
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    correction = np.asarray(FVMGeometry(mesh).bounded_over_relaxed_decomposition()[2])
    face_flux = coef * np.einsum("ij,ij->i", correction, grad_f)
    face_flux[~is_internal] = 0.0

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(np.zeros(mesh.number_of_cells()), grad_f, coef=coef)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_cross_diffusion_rhs_assembler_matches_linear_form_scalar():
    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            coef=coef,
            geometry=geometry,
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(space, geometry=geometry).assembly(
        uh=np.zeros(mesh.number_of_cells()),
        grad_f=grad_f,
        coef=coef,
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_cross_diffusion_rhs_assembler_matches_linear_form_vector_boundary_all():
    mesh, scalar_space = _box_space(nx=2, ny=1)
    space = TensorFunctionSpace(scalar_space, shape=(2, -1))
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.stack([
                np.linspace(-0.3, 0.5, mesh.number_of_edges()),
                np.linspace(0.1, 0.7, mesh.number_of_edges()),
            ], axis=1),
            np.stack([
                np.linspace(0.4, 0.9, mesh.number_of_edges()),
                np.linspace(-0.2, 0.3, mesh.number_of_edges()),
            ], axis=1),
        ],
        axis=1,
    )
    coef = np.linspace(0.8, 1.4, mesh.number_of_edges())

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros((mesh.number_of_cells(), 2)),
            grad_f,
            coef=coef,
            geometry=geometry,
            boundary_policy="all",
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(space, geometry=geometry).assembly(
        uh=np.zeros((mesh.number_of_cells(), 2)),
        grad_f=grad_f,
        coef=coef,
        boundary_policy="all",
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_cross_diffusion_rhs_assembler_matches_linear_form_limited():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    uh = np.array([0.0, 1.0])
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 100.0

    reference = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            geometry=geometry,
            correction_method="limited",
            limit_coeff=0.5,
        )
    ).assembly()
    fast = CrossDiffusionRHSAssembler(space, geometry=geometry).assembly(
        uh=uh,
        grad_f=grad_f,
        correction_method="limited",
        limit_coeff=0.5,
    )

    np.testing.assert_allclose(np.asarray(fast), np.asarray(reference), rtol=1.0e-13, atol=1.0e-13)


def test_default_cross_diffusion_reuses_fvm_geometry_bounded_decomposition(monkeypatch):
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.5, mesh.number_of_edges()),
            np.linspace(0.1, 0.7, mesh.number_of_edges()),
        ],
        axis=1,
    )
    calls = []
    original_decomposition = FVMGeometry.bounded_over_relaxed_decomposition

    def counted_decomposition(self, *, eps=0.05):
        calls.append(eps)
        return original_decomposition(self, eps=eps)

    monkeypatch.setattr(
        FVMGeometry,
        "bounded_over_relaxed_decomposition",
        counted_decomposition,
    )

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(np.zeros(mesh.number_of_cells()), grad_f)
    ).assembly()

    _, _, T_f = original_decomposition(FVMGeometry(mesh), eps=0.05)
    face_flux = np.einsum("ij,ij->i", np.asarray(T_f), grad_f)
    face_flux[np.asarray(FVMGeometry(mesh).is_boundary)] = 0.0

    assert calls == [0.05]
    np.testing.assert_allclose(
        np.asarray(rhs),
        _expected_scalar_scatter(mesh, face_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_orthogonal_correction_vector_is_zero_like_face_area_vector():
    mesh, _ = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    zero = np.asarray(np.zeros_like(np.asarray(geometry.S_f)))

    assert zero.shape == (mesh.number_of_edges(), mesh.geo_dimension())
    np.testing.assert_allclose(zero, 0.0, atol=0.0)


def test_bounded_over_relaxed_orthogonal_coefficient_is_stabilized():
    mesh, _ = _box_space(nx=2, ny=1)
    eps = 0.05
    geometry = FVMGeometry(mesh)
    _, mag_E_f, _ = geometry.bounded_over_relaxed_decomposition(eps=eps)
    coefficient = np.asarray(mag_E_f) / np.asarray(geometry.mag_d_f)
    denominator = np.asarray(geometry.mag_S_f) / coefficient
    projected = np.einsum(
        "ij,ij->i",
        np.asarray(geometry.n_f),
        np.asarray(geometry.d_f),
    )
    lower_bound = eps * np.asarray(geometry.mag_d_f)

    np.testing.assert_allclose(
        denominator,
        np.maximum(projected, lower_bound),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_bounded_over_relaxed_Tf_is_stabilized_on_bad_internal_face():
    a = 0.01
    y = 1.0
    node = bm.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [-a, -y],
            [a, y],
        ],
        dtype=bm.float64,
    )
    cell = bm.array([[0, 1, 2], [0, 3, 1]], dtype=bm.int32)
    mesh = TriangleMesh(node, cell)
    geometry = FVMGeometry(mesh)
    eps = 0.05

    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    delta = np.asarray(geometry.d_f)
    normal = np.asarray(geometry.n_f)
    face_area = np.asarray(geometry.mag_S_f)
    ratio = np.einsum("ij,ij->i", normal, delta) / np.linalg.norm(delta, axis=1)
    T_f = np.asarray(geometry.bounded_over_relaxed_decomposition(eps=eps)[2])
    stabilized_ratio = np.linalg.norm(T_f, axis=1) / face_area

    assert np.any(ratio[is_internal] <= eps)
    assert np.max(stabilized_ratio[is_internal]) <= 1.0 + 1.0 / eps


def test_orthogonal_correction_method_returns_zero_rhs():
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.ones((mesh.number_of_edges(), mesh.geo_dimension()))

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            correction_method="orthogonal",
        )
    ).assembly()

    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=0.0)


def test_limited_correction_with_zero_limit_coeff_returns_zero_internal_rhs():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 1.0

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.array([0.0, 1.0]),
            grad_f,
            geometry=geometry,
            correction_method="limited",
            limit_coeff=0.0,
        )
    ).assembly()

    np.testing.assert_allclose(np.asarray(rhs), 0.0, atol=1.0e-13)


def test_limited_correction_matches_manual_limiter_on_bad_internal_face():
    mesh, space = _bad_two_triangle_space()
    geometry = FVMGeometry(mesh)
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    uh = np.array([0.0, 1.0])
    grad_f = np.zeros((mesh.number_of_edges(), mesh.geo_dimension()))
    grad_f[is_internal, 0] = 100.0
    _, mag_E_f, correction_vector = geometry.bounded_over_relaxed_decomposition(
        eps=0.05
    )
    correction_vector = np.asarray(correction_vector)
    full_flux = np.einsum("ij,ij->i", correction_vector, grad_f)
    orthogonal_coeff = np.asarray(mag_E_f) / np.asarray(geometry.mag_d_f)
    owner = edge_to_cell[:, 0]
    neighbour = edge_to_cell[:, 1]
    orthogonal_flux = np.zeros(mesh.number_of_edges())
    orthogonal_flux[is_internal] = (
        orthogonal_coeff[is_internal]
        * np.abs(uh[neighbour[is_internal]] - uh[owner[is_internal]])
    )
    limit_coeff = 0.5
    limiter = np.ones(mesh.number_of_edges())
    limiter[is_internal] = np.minimum(
        limit_coeff * orthogonal_flux[is_internal]
        / ((1.0 - limit_coeff) * np.abs(full_flux[is_internal]) + 1.0e-30),
        1.0,
    )
    expected = _expected_scalar_scatter(mesh, full_flux * limiter)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            geometry=geometry,
            correction_method="limited",
            limit_coeff=limit_coeff,
        )
    ).assembly()

    assert np.max(np.abs(full_flux[is_internal] * limiter[is_internal])) < np.max(
        np.abs(full_flux[is_internal])
    )
    np.testing.assert_allclose(np.asarray(rhs), expected, rtol=1.0e-13, atol=1.0e-13)


def test_correction_vector_matches_equivalent_face_flux_correction():
    mesh, space = _box_space(nx=2, ny=1)
    correction_vector = np.stack(
        [
            np.linspace(0.1, 0.5, mesh.number_of_edges()),
            np.linspace(-0.2, 0.4, mesh.number_of_edges()),
        ],
        axis=1,
    )
    grad_f = np.stack(
        [
            np.linspace(-0.3, 0.7, mesh.number_of_edges()),
            np.linspace(0.4, 0.9, mesh.number_of_edges()),
        ],
        axis=1,
    )
    face_flux = np.einsum("ij,ij->i", correction_vector, grad_f)

    from_vector = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            correction_vector=correction_vector,
        )
    ).assembly()
    from_flux = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(face_flux_correction=face_flux)
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(from_vector),
        np.asarray(from_flux),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_default_correction_method_matches_explicit_bounded_over_relaxed():
    mesh, space = _box_space(nx=2, ny=1)
    grad_f = np.stack(
        [
            np.linspace(-0.1, 0.3, mesh.number_of_edges()),
            np.linspace(0.2, 0.6, mesh.number_of_edges()),
        ],
        axis=1,
    )

    default_rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(np.zeros(mesh.number_of_cells()), grad_f)
    ).assembly()
    geometry_rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            geometry=FVMGeometry(mesh),
            correction_method="bounded_over_relaxed",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(geometry_rhs),
        np.asarray(default_rhs),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_bounded_over_relaxed_zeros_boundary_correction_by_default():
    mesh, space = _box_space(nx=1, ny=1)
    geometry = FVMGeometry(mesh)
    edge_to_cell = np.asarray(mesh.edge_to_cell()[:, :2])
    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    grad_f = np.stack(
        [
            np.linspace(0.2, 1.0, mesh.number_of_edges()),
            np.linspace(-0.4, 0.8, mesh.number_of_edges()),
        ],
        axis=1,
    )
    correction_vector = np.asarray(geometry.bounded_over_relaxed_decomposition()[2])
    full_flux = np.einsum("ij,ij->i", correction_vector, grad_f)
    expected_flux = full_flux.copy()
    expected_flux[~is_internal] = 0.0
    expected = _expected_scalar_scatter(mesh, expected_flux)

    assert np.max(np.abs(full_flux[~is_internal])) > 0.0

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            geometry=geometry,
            correction_method="bounded_over_relaxed",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_bounded_over_relaxed_boundary_policy_all_keeps_raw_boundary_flux():
    mesh, space = _box_space(nx=1, ny=1)
    geometry = FVMGeometry(mesh)
    grad_f = np.stack(
        [
            np.linspace(0.2, 1.0, mesh.number_of_edges()),
            np.linspace(-0.4, 0.8, mesh.number_of_edges()),
        ],
        axis=1,
    )
    face_flux = np.einsum(
        "ij,ij->i",
        np.asarray(geometry.bounded_over_relaxed_decomposition()[2]),
        grad_f,
    )
    expected = _expected_scalar_scatter(mesh, face_flux)

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            np.zeros(mesh.number_of_cells()),
            grad_f,
            geometry=geometry,
            correction_method="bounded_over_relaxed",
            boundary_policy="all",
        )
    ).assembly()

    np.testing.assert_allclose(
        np.asarray(rhs),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_legacy_correction_method_is_not_supported():
    mesh, space = _box_space(nx=1, ny=1)
    grad_f = np.ones((mesh.number_of_edges(), mesh.geo_dimension()))

    with pytest.raises(ValueError, match="Unsupported correction_method"):
        LinearForm(space).add_integrator(
            ScalarCrossDiffusionIntegrator(
                np.zeros(mesh.number_of_cells()),
                grad_f,
                correction_method="legacy",
            )
        ).assembly()


def _assert_finite_geometry_quantities(mesh):
    geometry = FVMGeometry(mesh)
    _, mag_E_f, T_f = geometry.bounded_over_relaxed_decomposition()
    assert np.asarray(T_f).shape == (
        mesh.number_of_edges(),
        mesh.geo_dimension(),
    )
    assert np.all(np.isfinite(np.asarray(mag_E_f) / np.asarray(geometry.mag_d_f)))
    assert np.all(np.isfinite(np.asarray(T_f)))


def test_circle_mesher_bounded_over_relaxed_geometry_smoke():
    pytest.importorskip("gmsh")
    from fealpy.mesher.circle_mesher import CircleMesher

    mesh = CircleMesher(h=0.4).init_mesh()

    _assert_finite_geometry_quantities(mesh)


def test_box_with_circular_hole_mesher_bounded_over_relaxed_geometry_smoke():
    pytest.importorskip("gmsh")
    from fealpy.mesher import BoxWithCircularHoleMesher2D

    mesh = BoxWithCircularHoleMesher2D(
        {
            "box": (-3.0, 3.0, -2.0, 2.0),
            "center": (0.0, 0.0),
            "radius": 0.5,
            "h": 0.5,
        }
    ).init_mesh()

    _assert_finite_geometry_quantities(mesh)
