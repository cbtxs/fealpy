import inspect

import pytest


def _cavity_solver(
    convection_coef=None,
    *,
    controls=None,
    linear_solver=None,
    resolution=2,
):
    from fealpy.fvm import (
        CollocatedSimpleSolver,
        FVMLinearSolverConfig,
        LidDrivenCavityCase,
        PDEBoundaryConditions,
        SimpleSolverControls,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=resolution, ny=resolution)
    solver_kwargs = {}
    if linear_solver is None:
        solver_kwargs["linear_solver_config"] = FVMLinearSolverConfig(solver="scipy")
    else:
        solver_kwargs["linear_solver"] = linear_solver

    return CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=case.mu,
        convection_coef=case.rho if convection_coef is None else convection_coef,
        source=case.source,
        boundary_conditions=PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=case.dirichlet_velocity,
        ),
        controls=controls or SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
        ),
        log_level="ERROR",
        **solver_kwargs,
    )


def test_collocated_simple_solver_default_pressure_relaxation_is_conservative():
    from fealpy.fvm import CollocatedSimpleSolver

    solve_parameters = inspect.signature(CollocatedSimpleSolver.solve).parameters
    assert solve_parameters["relax"].default == 0.3


def test_collocated_simple_solver_reuses_boundary_geometry():
    solver = _cavity_solver()

    assert solver.fvm_geometry is solver.boundary_conditions.geometry


def test_cell_vector_dof_conversion_is_component_major_on_torch_backend():
    pytest.importorskip("torch")
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver()
    try:
        bm.set_backend("pytorch")
        cell_vector = bm.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        dofs = solver.cell_vector_to_dofs(cell_vector)

        assert bm.to_numpy(dofs).tolist() == [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
        assert bm.to_numpy(solver.dofs_to_cell_vector(dofs)).tolist() == [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    finally:
        bm.set_backend("numpy")


def test_collocated_simple_solver_runs_without_model_adapter():
    solver = _cavity_solver()

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-3)

    assert velocity.shape == (solver.NC, solver.GD)
    assert pressure.shape == (solver.NC,)
    assert solver.face_velocity.shape == (
        solver.mesh.number_of_faces(),
        solver.GD,
    )
    assert solver.face_flux.shape == (solver.mesh.number_of_faces(),)
    assert len(solver.residuals) == 1
    assert "pressure_relax" not in solver.residuals[0]


def test_rhie_chow_requires_physical_cell_velocity_layout():
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver(convection_coef=0.0)
    algebraic_velocity = bm.zeros(solver.GD * solver.NC, dtype=solver.cm.dtype)
    face_response = bm.ones(
        solver.mesh.number_of_faces(),
        dtype=solver.cm.dtype,
    )

    with pytest.raises(ValueError, match=r"cell velocity must have shape \(NC, GD\)"):
        solver.rhie_chow.cell_velocity_to_face(
            algebraic_velocity,
            face_response_coefficient=face_response,
        )


def test_simple_reconstructs_stopping_flux_from_updated_state(monkeypatch):
    solver = _cavity_solver(convection_coef=0.0)
    calls = []
    original = solver.rhie_chow_face_velocity

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(solver, "rhie_chow_face_velocity", record)

    solver.solve(max_iter=1, tol=1.0e-30)

    assert len(calls) == 2
    assert solver.face_velocity is calls[-1]


def test_collocated_simple_diagnostics_are_disabled_by_default():
    from fealpy.fvm import SimpleSolverControls

    assert SimpleSolverControls().diagnostics_enabled is False


def test_simple_always_records_true_momentum_residual_and_termination_state():
    solver = _cavity_solver(convection_coef=0.0)

    solver.solve(max_iter=1, tol=1.0e-30)

    residual = solver.residuals[-1]
    assert "momentum_residual_absolute" in residual
    assert "momentum_residual_relative" in residual
    assert solver.converged is False
    assert solver.termination_reason == "max_iter"
    assert solver.outer_iterations == 1


def test_simple_records_fixed_point_residual_termination():
    solver = _cavity_solver(convection_coef=0.0)

    solver.solve(max_iter=1, tol=1.0e6)

    assert solver.converged is True
    assert solver.termination_reason == "fixed_point_residuals"
    assert solver.outer_iterations == 1


def test_collocated_simple_solver_shares_gradient_reconstruction_weights():
    from fealpy.fvm import SimpleSolverControls

    controls = SimpleSolverControls(
        pressure_constraint="gauge",
        momentum_solve_strategy="vector",
        gradient_layer_weights=(1.0, 0.05),
        gradient_boundary_weight=0.75,
    )
    solver = _cavity_solver(controls=controls)

    reconstructors = (
        solver.pressure_gradient,
        solver.velocity_gradient,
        solver.rhie_chow.gradient_reconstruct,
    )
    for reconstruct in reconstructors:
        assert reconstruct.layer_weights == (1.0, 0.05)
        assert reconstruct.boundary_weight == 0.75


def test_collocated_simple_solver_records_true_momentum_diagnostics():
    from fealpy.fvm import SimpleSolverControls

    solver = _cavity_solver(
        convection_coef=0.0,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
            diagnostics_enabled=True,
        ),
    )

    solver.solve(max_iter=1, tol=1.0e-3)

    residual = solver.residuals[-1]
    assert residual["momentum_residual_absolute"] >= 0.0
    assert residual["momentum_residual_relative"] >= 0.0
    assert solver.face_velocity.shape == (solver.mesh.number_of_faces(), solver.GD)


def test_steady_momentum_terms_sum_to_production_balance():
    import numpy as np
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver()
    cell_center = solver.fvm_geometry.cell_center
    velocity = bm.stack(
        (cell_center[:, 0] + cell_center[:, 1], cell_center[:, 0] - cell_center[:, 1]),
        axis=-1,
    )
    pressure = cell_center[:, 0] - 0.25 * cell_center[:, 1]
    face_velocity = solver.face_interpolate_cell_vector(velocity, method="linear")

    terms = solver.steady_momentum_terms(pressure, velocity, face_velocity)
    expected_residual = (
        terms["diffusion"]
        + terms["convection"]
        + terms["pressure"]
        - terms["source"]
    )
    balance = solver.steady_momentum_balance(pressure, velocity, face_velocity)

    np.testing.assert_allclose(
        bm.to_numpy(terms["residual"]),
        bm.to_numpy(expected_residual),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(balance["residual"]),
        bm.to_numpy(terms["residual"]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_collocated_simple_solver_runs_with_simplec_pressure_response():
    from fealpy.fvm import SimpleSolverControls

    solver = _cavity_solver(
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            pressure_response_scheme="simplec",
            momentum_solve_strategy="vector",
        ),
    )

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-3)

    assert solver.controls.pressure_response_scheme == "simplec"
    assert velocity.shape == (solver.NC, solver.GD)
    assert pressure.shape == (solver.NC,)


def test_second_order_face_velocity_reconstruction_recovers_linear_vector_field(
    monkeypatch,
):
    import numpy as np
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver(convection_coef=0.0)
    cell_center = solver.fvm_geometry.cell_center
    face_center = solver.fvm_geometry.face_center
    velocity = bm.stack(
        (
            cell_center[:, 0] + 2.0 * cell_center[:, 1],
            -cell_center[:, 0] + 3.0 * cell_center[:, 1],
        ),
        axis=-1,
    )
    exact_gradient = bm.array([
        [[1.0, 2.0], [-1.0, 3.0]]
        for _ in range(solver.NC)
    ])
    monkeypatch.setattr(
        solver.velocity_gradient.lsq_reconstruct,
        "layered_lsq",
        lambda value: exact_gradient,
    )

    face_velocity = solver.second_order_reconstructed_face_velocity(velocity)
    expected = bm.stack(
        (
            face_center[:, 0] + 2.0 * face_center[:, 1],
            -face_center[:, 0] + 3.0 * face_center[:, 1],
        ),
        axis=-1,
    )

    np.testing.assert_allclose(bm.to_numpy(face_velocity), bm.to_numpy(expected))


def test_collocated_simple_solver_runs_with_second_order_face_velocity_scheme():
    from fealpy.fvm import SimpleSolverControls

    solver = _cavity_solver(
        convection_coef=0.0,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            rhie_chow_velocity_scheme="second_order_reconstructed",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
        ),
    )

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-3)

    assert solver.controls.rhie_chow_velocity_scheme == "second_order_reconstructed"
    assert velocity.shape == (solver.NC, solver.GD)
    assert pressure.shape == (solver.NC,)


def test_collocated_simple_solver_runs_with_cell_anchored_face_flux_correction(
    monkeypatch,
):
    from fealpy.fvm import SimpleSolverControls

    solver = _cavity_solver(
        convection_coef=0.0,
        resolution=4,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            face_flux_correction_scheme="cell_anchored_quadratic",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
        ),
    )
    calls = {"count": 0}
    original = solver.boundary_conditions.boundary_face_velocity_average

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        solver.boundary_conditions,
        "boundary_face_velocity_average",
        counted,
    )

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-3)

    assert solver.face_flux_reconstruct.method == "cell_anchored_quadratic"
    assert solver.face_flux_reconstruct.diagnostics()["minimum_rank"] == 5
    assert calls["count"] == 1
    assert velocity.shape == (solver.NC, solver.GD)
    assert pressure.shape == (solver.NC,)


def test_ns_fvm_simple_model_forwards_simplec_pressure_response_option():
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMSimpleModel

    model = NSFVMSimpleModel(
        {
            "pde": 1,
            "mesh_type": "uniform_quad",
            "nx": 2,
            "ny": 2,
            "pressure_constraint": "gauge",
            "pressure_response_scheme": "simplec",
            "rhie_chow_velocity_scheme": "second_order_reconstructed",
            "momentum_solve_strategy": "vector",
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )

    assert model.controls.pressure_response_scheme == "simplec"
    assert model.controls.rhie_chow_velocity_scheme == "second_order_reconstructed"


def test_ns_fvm_simple_model_forwards_face_flux_correction_options():
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMSimpleModel

    model = NSFVMSimpleModel(
        {
            "pde": 1,
            "mesh_type": "uniform_quad",
            "nx": 4,
            "ny": 4,
            "pressure_constraint": "gauge",
            "face_flux_correction_scheme": "cell_anchored_quadratic",
            "face_flux_quadrature_order": 4,
            "face_flux_max_stencil_layers": 5,
            "momentum_solve_strategy": "vector",
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )

    assert model.controls.face_flux_correction_scheme == "cell_anchored_quadratic"
    assert model.controls.face_flux_quadrature_order == 4
    assert model.controls.face_flux_max_stencil_layers == 5


def test_collocated_simple_solver_accepts_zero_convection_for_stokes_limit():
    solver = _cavity_solver(convection_coef=0.0)

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-3)

    assert velocity.shape == (solver.NC, solver.GD)
    assert pressure.shape == (solver.NC,)
    assert len(solver.residuals) == 1


def test_collocated_simple_solver_accepts_string_linear_solver_choice():
    solver = _cavity_solver(convection_coef=0.0, linear_solver="scipy")

    assert solver.linear_solver.config.solver == "scipy"


def test_collocated_simple_solver_uses_recommended_default_linear_policy():
    from fealpy.fvm import (
        CollocatedSimpleSolver,
        LidDrivenCavityCase,
        PDEBoundaryConditions,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=2, ny=2)
    solver = CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=case.mu,
        convection_coef=case.rho,
        source=case.source,
        boundary_conditions=PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=case.dirichlet_velocity,
        ),
        log_level="ERROR",
    )

    assert solver.momentum_linear_solver == "scipy_bicgstab"
    assert solver.pressure_nullspace_linear_solver == "petsc_gmres_hypre"


def test_simple_temporary_velocity_reuses_steady_source_rhs(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import SimpleSolverControls

    bm.set_backend("numpy")

    class ZeroLinearSolver:
        def solve(self, matrix, rhs, *, solver=None):
            return bm.zeros(rhs.shape[0], dtype=rhs.dtype)

    solver = _cavity_solver(
        convection_coef=0.0,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
        ),
        linear_solver=ZeroLinearSolver(),
    )
    call_count = 0

    def counted_source_vector(source):
        nonlocal call_count
        call_count += 1
        return bm.ones(solver.GD * solver.NC, dtype=solver.cm.dtype)

    monkeypatch.setattr(solver, "momentum_source_vector", counted_source_vector)

    p = bm.zeros(solver.NC, dtype=solver.cm.dtype)
    uf = bm.zeros((solver.mesh.number_of_faces(), solver.GD), dtype=solver.cm.dtype)
    u0 = bm.zeros((solver.NC, solver.GD), dtype=solver.cm.dtype)

    solver.temporary_velocity(p, uf, u0)
    solver.temporary_velocity(p, uf, u0)

    assert call_count == 1


def test_simple_temporary_velocity_returns_named_dual_response():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import SimpleSolverControls
    from fealpy.fvm.collocated_ns_components import MomentumPredictorResult

    bm.set_backend("numpy")
    solver = _cavity_solver(
        convection_coef=0.0,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
        ),
    )
    p = bm.zeros(solver.NC, dtype=solver.cm.dtype)
    uf = bm.zeros((solver.mesh.number_of_faces(), solver.GD), dtype=solver.cm.dtype)
    u0 = bm.zeros((solver.NC, solver.GD), dtype=solver.cm.dtype)

    result = solver.temporary_velocity(p, uf, u0)

    assert isinstance(result, MomentumPredictorResult)
    assert result.correction_denominator.shape == (solver.GD * solver.NC,)
    assert result.spatial_diagonal.shape == (solver.GD * solver.NC,)
    assert result.velocity.shape == (solver.NC, solver.GD)


def test_simple_pressure_gauge_matrix_does_not_use_bilinear_assembly(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.ones(solver.mesh.number_of_faces(), dtype=solver.cm.dtype)

    def fail_assembly(self):
        raise AssertionError("pressure gauge matrix should use cached FVM structure")

    monkeypatch.setattr(BilinearForm, "assembly", fail_assembly)

    matrix = solver.pressure_gauge_matrix(coef)

    assert matrix.shape == (solver.NC + 1, solver.NC + 1)


def test_simple_pressure_gauge_matrix_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm, BlockForm
    from fealpy.sparse import COOTensor
    from fealpy.fvm import ScalarDiffusionIntegrator

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.mesh.number_of_faces())
    matrix = solver.pressure_gauge_matrix(coef)

    reference = BilinearForm(solver.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.fvm_geometry)
    ).assembly()
    gauge_index = bm.stack(
        [
            bm.zeros(solver.NC, dtype=bm.int32),
            bm.arange(solver.NC, dtype=bm.int32),
        ],
        axis=0,
    )
    gauge = COOTensor(gauge_index, solver.cm, spshape=(1, solver.NC))
    reference = BlockForm([[reference, gauge.T], [gauge, None]])
    reference = reference.assembly_sparse_matrix(format="csr")

    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_scalar_diffusion_matrix_assembler_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm
    from fealpy.fvm.scalar_diffusion_integrator import (
        ScalarDiffusionIntegrator,
        ScalarDiffusionMatrixAssembler,
    )

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.mesh.number_of_faces())
    matrix = ScalarDiffusionMatrixAssembler(
        solver.space,
        geometry=solver.fvm_geometry,
    ).assembly(coef)

    reference = BilinearForm(solver.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.fvm_geometry)
    ).assembly()
    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_stokes_simple_model_accepts_momentum_relaxation_option():
    from fealpy.fvm import StokesFVMSimpleModel

    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "momentum_equation_relaxation": 0.6,
            "rhie_chow_velocity_scheme": "second_order_reconstructed",
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.controls.momentum_equation_relaxation == 0.6
    assert model.controls.rhie_chow_velocity_scheme == "second_order_reconstructed"
