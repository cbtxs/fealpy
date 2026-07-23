import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry, NSFVMPISOModel


def _model_options(nx=2, ny=2, nt=1):
    return {
        "pde": 3,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "duration": (0, 1),
        "space_degree": 0,
        "pbar_log": False,
        "log_level": "WARNING",
    }


def test_divergence_from_flux_matches_owner_oriented_geometry_scatter():
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    phi = bm.linspace(0.2, 1.4, model.mesh.number_of_faces())
    expected = FVMGeometry(model.mesh).scatter_face_flux_to_cells(phi)

    divergence = model.divergence_from_flux(phi)

    np.testing.assert_allclose(
        np.asarray(divergence),
        np.asarray(expected),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_pressure_nonorthogonal_cross_flux_uses_explicit_interpolation(monkeypatch):
    import fealpy.fvm.collocated_ns_components as ns_components
    from fealpy.fvm.fvm_geometry import DiffusionFaceDecomposition

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    pressure = bm.zeros(model.NC, dtype=model.cm.dtype)
    response = bm.ones(model.mesh.number_of_faces(), dtype=model.cm.dtype)
    seen = []

    def record_face_gradient(mesh, cell_gradient, *, geometry, interpolation_method, **kwargs):
        seen.append(interpolation_method)
        return bm.zeros((mesh.number_of_faces(), mesh.geo_dimension()), dtype=model.cm.dtype)

    monkeypatch.setattr(ns_components, "reconstruct_face_gradient", record_face_gradient)
    monkeypatch.setattr(
        model.fvm_geometry,
        "diffusion_face_decomposition",
        lambda method="over_relaxed", eps=0.05: DiffusionFaceDecomposition(
            E_f=model.fvm_geometry.S_f,
            mag_E_f=model.fvm_geometry.mag_S_f,
            T_f=bm.ones(
                (model.mesh.number_of_faces(), model.GD), dtype=model.cm.dtype
            ),
            orthogonal_factor=(
                model.fvm_geometry.mag_S_f / model.fvm_geometry.mag_d_f
            ),
        ),
    )

    model.pressure_nonorthogonal_cross_flux(
        pressure,
        response,
        interpolation_method="linear",
    )

    assert seen == ["linear"]


def test_ns_operators_share_one_configured_diffusion_decomposition(monkeypatch):
    bm.set_backend("numpy")
    options = _model_options()
    options["diffusion_method"] = "bounded_over_relaxed"
    options["diffusion_nonorthogonal_eps"] = 0.1
    model = NSFVMPISOModel(options)

    assert model.controls.diffusion_method == "bounded_over_relaxed"
    assert model.dirichlet_velocity_bc.nonorthogonal_eps == 0.1

    calls = []
    original_decomposition = model.fvm_geometry.diffusion_face_decomposition

    def record_decomposition(method="over_relaxed", *, eps=0.05):
        calls.append((method, eps))
        if method != "bounded_over_relaxed":
            raise AssertionError(
                "configured bounded diffusion must not use another decomposition"
            )
        return original_decomposition(method, eps=eps)

    monkeypatch.setattr(
        model.fvm_geometry,
        "diffusion_face_decomposition",
        record_decomposition,
    )

    model.scalar_momentum_diffusion_matrix(1.0)
    velocity = bm.zeros((model.NC, model.GD), dtype=model.cm.dtype)
    model.momentum_nonorthogonal_rhs(velocity)
    model.boundary_corrected_momentum_explicit_source(velocity)
    response = bm.ones(model.mesh.number_of_faces(), dtype=model.cm.dtype)
    pressure = bm.zeros(model.NC, dtype=model.cm.dtype)
    model.pressure_diffusion_matrix(response)
    model.pressure_orthogonal_flux(pressure, response)
    model.pressure_nonorthogonal_cross_flux(
        pressure,
        response,
        interpolation_method="average",
    )
    model.dirichlet_velocity_bc.diffusion_boundary_data(coef=1.0)

    assert calls
    assert set(calls) == {("bounded_over_relaxed", 0.1)}


def test_dirichlet_pressure_full_flux_reproduces_affine_pressure():
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    geometry = model.fvm_geometry
    gradient = bm.array([1.7, -0.8], dtype=model.cm.dtype)
    pressure = 0.3 + bm.einsum("kd,d->k", geometry.cell_center, gradient)
    response = bm.linspace(
        0.7,
        1.3,
        model.mesh.number_of_faces(),
        dtype=model.cm.dtype,
    )
    boundary_value = lambda p: 0.3 + bm.einsum("kd,d->k", p, gradient)
    boundary_threshold = lambda p: bm.ones(p.shape[0], dtype=bm.bool)

    orthogonal_flux = model.pressure_orthogonal_flux(pressure, response)
    cross_flux = model.pressure_nonorthogonal_cross_flux(
        pressure,
        response,
        interpolation_method="average",
        pressure_gradient=bm.broadcast_to(
            gradient[None, :], (model.NC, model.GD)
        ),
        boundary_threshold=boundary_threshold,
    )
    numerical_flux = model.add_dirichlet_pressure_flux(
        orthogonal_flux - cross_flux,
        pressure,
        response,
        boundary_value,
        boundary_threshold,
    )
    exact_flux = -response * bm.einsum("fd,d->f", geometry.S_f, gradient)

    np.testing.assert_allclose(
        np.asarray(numerical_flux),
        np.asarray(exact_flux),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_component_nonorthogonal_correction_accepts_converged_base_solution(
    monkeypatch,
):
    from fealpy.sparse import spdiags

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    identity = spdiags(
        bm.ones(model.NC), 0, model.NC, model.NC
    )
    rhs = bm.zeros(model.GD * model.NC)
    velocity = bm.zeros_like(rhs)
    monkeypatch.setattr(
        model,
        "momentum_nonorthogonal_rhs",
        lambda value: bm.zeros_like(rhs),
    )
    monkeypatch.setattr(
        model,
        "solve_component_momentum_systems",
        lambda matrices, current_rhs: (_ for _ in ()).throw(
            AssertionError("a converged base solution needs no correction solve")
        ),
    )

    actual = model.correct_component_momentum_nonorthogonal_diffusion(
        [identity] * model.GD,
        rhs,
        velocity,
        max_iter=2,
        tol=1.0e-12,
        atol=0.0,
        iteration_attr="last_momentum_nonorthogonal_iterations",
    )

    np.testing.assert_allclose(np.asarray(actual), 0.0, atol=0.0)
    assert model.last_momentum_nonorthogonal_iterations == 0
    assert model.last_momentum_nonorthogonal_residual["relative"] == 0.0


def test_component_nonorthogonal_correction_raises_at_safety_limit(monkeypatch):
    from fealpy.sparse import spdiags

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    identity = spdiags(
        bm.ones(model.NC), 0, model.NC, model.NC
    )
    rhs = bm.zeros(model.GD * model.NC)
    velocity = bm.zeros_like(rhs)
    monkeypatch.setattr(
        model,
        "momentum_nonorthogonal_rhs",
        lambda value: bm.ones_like(rhs),
    )
    monkeypatch.setattr(
        model,
        "solve_component_momentum_systems",
        lambda matrices, current_rhs: bm.zeros_like(rhs),
    )

    with pytest.raises(RuntimeError, match="momentum non-orthogonal correction"):
        model.correct_component_momentum_nonorthogonal_diffusion(
            [identity] * model.GD,
            rhs,
            velocity,
            max_iter=1,
            tol=1.0e-12,
            atol=0.0,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )


def test_pressure_gradient_source_accepts_precomputed_gradient(monkeypatch):
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    pressure = bm.zeros(model.NC, dtype=model.cm.dtype)
    gradient = bm.ones((model.NC, model.GD), dtype=model.cm.dtype)

    class ForbiddenGradient:
        def cell_gradient(self, _):
            raise AssertionError("pressure_gradient_source should reuse supplied gradient")

    model.pressure_gradient = ForbiddenGradient()

    source = model.pressure_gradient_source(pressure, pressure_gradient=gradient)

    expected = bm.concatenate([model.cm for _ in range(model.GD)], axis=0)
    np.testing.assert_allclose(np.asarray(source), np.asarray(expected))


def test_component_momentum_dirichlet_rhs_applies_vector_boundary_once():
    bm.set_backend("numpy")
    from fealpy.fvm import DirichletBC

    model = NSFVMPISOModel(_model_options())
    calls = []

    def dirichlet_velocity(points):
        calls.append(points.shape[0])
        return bm.zeros((points.shape[0], model.GD), dtype=model.cm.dtype)

    model.dirichlet_velocity = dirichlet_velocity
    model.dirichlet_velocity_bc = DirichletBC(
        model.mesh,
        dirichlet_velocity,
        threshold=model.dirichlet_velocity_threshold,
        geometry=model.fvm_geometry,
    )

    matrix = model.scalar_momentum_diffusion_matrix(1.0)
    rhs = bm.zeros(model.GD * model.NC, dtype=model.cm.dtype)
    previous_velocity = bm.zeros((model.NC, model.GD), dtype=model.cm.dtype)
    face_velocity = bm.zeros((model.mesh.number_of_faces(), model.GD), dtype=model.cm.dtype)

    model.component_momentum_linear_systems(
        matrix,
        rhs,
        previous_velocity,
        diffusion_coef=1.0,
        convection_face_velocity=face_velocity,
        matrix_policy="shared",
    )

    assert len(calls) == 2


def test_component_momentum_system_exposes_spatial_and_relaxed_diagonals():
    bm.set_backend("numpy")
    from fealpy.fvm.collocated_ns_components import ComponentMomentumSystems

    model = NSFVMPISOModel(_model_options())
    matrix = model.scalar_momentum_diffusion_matrix(1.0)
    rhs = bm.zeros(model.GD * model.NC, dtype=model.cm.dtype)
    previous_velocity = bm.zeros((model.NC, model.GD), dtype=model.cm.dtype)

    systems_05 = model.component_momentum_linear_systems(
        matrix,
        rhs,
        previous_velocity,
        diffusion_coef=1.0,
        relaxation=0.5,
    )
    systems_09 = model.component_momentum_linear_systems(
        matrix,
        rhs,
        previous_velocity,
        diffusion_coef=1.0,
        relaxation=0.9,
    )

    assert isinstance(systems_05, ComponentMomentumSystems)
    np.testing.assert_allclose(
        np.asarray(systems_05.spatial_diagonal),
        np.asarray(systems_09.spatial_diagonal),
    )
    np.testing.assert_allclose(
        np.asarray(systems_05.relaxed_diagonal),
        np.asarray(systems_05.spatial_diagonal) / 0.5,
    )
    np.testing.assert_allclose(
        np.asarray(systems_09.relaxed_diagonal),
        np.asarray(systems_09.spatial_diagonal) / 0.9,
    )


def test_collocated_discretization_reuses_one_fvm_geometry_for_gradients():
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())

    assert model.pressure_gradient.fvm_geometry is model.fvm_geometry
    assert model.velocity_gradient.fvm_geometry is model.fvm_geometry


def test_solver_setup_and_coefficient_validation_live_outside_operator_mixin():
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, init_fvm_linear_solver
    from fealpy.fvm.solver_controls import nonnegative_scalar, positive_scalar

    solver = init_fvm_linear_solver(linear_solver="scipy", reference=bm.zeros(1))

    assert isinstance(solver, FVMLinearSolver)
    assert solver.config.solver == "scipy"
    assert positive_scalar(bm.array(2.0), "mu") == 2.0
    assert nonnegative_scalar(bm.array(0.0), "rho") == 0.0


def test_simplec_momentum_response_denominator_uses_matrix_row_sum():
    from fealpy.sparse import CSRTensor

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    total = model.GD * model.NC
    rows = bm.arange(total, dtype=bm.int64)
    crow = bm.arange(0, 2 * total + 1, 2, dtype=bm.int64)
    col = bm.stack([rows, (rows + 1) % total], axis=1).reshape(-1)
    values = bm.stack(
        [
            4.0 * bm.ones(total, dtype=model.cm.dtype),
            -1.0 * bm.ones(total, dtype=model.cm.dtype),
        ],
        axis=1,
    ).reshape(-1)
    matrix = CSRTensor(crow, col, values, spshape=(total, total))
    diagonal = 4.0 * bm.ones(total, dtype=model.cm.dtype)

    simple = model.momentum_pressure_response_denominator(
        matrix,
        diagonal,
        scheme="simple",
    )
    simplec = model.momentum_pressure_response_denominator(
        matrix,
        diagonal,
        scheme="simplec",
    )

    np.testing.assert_allclose(np.asarray(simple), 4.0)
    np.testing.assert_allclose(np.asarray(simplec), 3.0)


def test_spatial_face_velocity_dispatches_second_order_scheme(monkeypatch):
    from fealpy.fvm import NSFVMSimpleModel

    bm.set_backend("numpy")
    model = NSFVMSimpleModel({
        "pde": 1,
        "nx": 2,
        "ny": 2,
        "rhie_chow_velocity_scheme": "second_order_reconstructed",
        "pressure_constraint": "gauge",
        "log_level": "ERROR",
    })
    marker = bm.ones((model.NF, model.GD), dtype=model.cm.dtype) * 7.0
    monkeypatch.setattr(
        model,
        "second_order_reconstructed_face_velocity",
        lambda velocity: marker,
    )

    actual = model.spatial_face_velocity(
        bm.zeros((model.NC, model.GD), dtype=model.cm.dtype)
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(marker))


def test_spatial_face_velocity_applies_only_requested_flux_defect(monkeypatch):
    from types import SimpleNamespace

    from fealpy.fvm import NSFVMSimpleModel

    bm.set_backend("numpy")
    model = NSFVMSimpleModel({
        "pde": 1,
        "nx": 2,
        "ny": 2,
        "rhie_chow_velocity_scheme": "interpolated",
        "face_flux_correction_scheme": "cell_anchored_quadratic",
        "pressure_constraint": "gauge",
        "log_level": "ERROR",
    })
    base = bm.zeros((model.NF, model.GD), dtype=model.cm.dtype)
    defect = bm.arange(model.NF, dtype=model.cm.dtype) + 1.0
    monkeypatch.setattr(
        model,
        "face_interpolate_cell_vector",
        lambda velocity, method: base,
    )
    monkeypatch.setattr(
        model,
        "face_flux_reconstruct",
        SimpleNamespace(correction=lambda *args, **kwargs: defect),
    )

    actual = model.spatial_face_velocity(
        bm.zeros((model.NC, model.GD), dtype=model.cm.dtype)
    )

    np.testing.assert_allclose(
        np.asarray(model.compute_face_flux(actual)),
        np.asarray(defect),
    )
