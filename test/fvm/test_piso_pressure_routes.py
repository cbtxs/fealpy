import numpy as np
import pytest


def _model_options(nx=4, ny=4, nt=1, *, recommended_defaults=False):
    options = {
        "pde": 3,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "duration": (0, 1),
        "space_degree": 0,
        "pbar_log": False,
        "log_level": "WARNING",
    }
    if not recommended_defaults:
        options.update(
            {
                "pressure_constraint": "gauge",
                "momentum_solve_strategy": "vector",
            }
        )
    return options


def test_piso_model_uses_recommended_default_linear_policy():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(
        _model_options(nx=2, ny=2, nt=1, recommended_defaults=True)
    )

    assert model.momentum_linear_solver == "scipy_bicgstab"
    assert model.pressure_nullspace_linear_solver == "petsc_gmres_hypre"


def test_piso_solver_reuses_boundary_geometry():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))

    assert model.fvm_geometry is model.boundary_conditions.geometry


def test_piso_pressure_state_boundary_inherits_diffusion_configuration():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "diffusion_method": "bounded_over_relaxed",
            "diffusion_nonorthogonal_eps": 0.1,
        }
    )
    model = NSFVMPISOModel(options)
    model.dirichlet_pressure_value = lambda p: p[:, 0]
    model.dirichlet_pressure_threshold = lambda p: bm.ones(
        p.shape[0], dtype=bm.bool
    )
    model._init_discretization(model.controls.space_degree)

    assert model.pressure_state_bc.diffusion_method == "bounded_over_relaxed"
    assert model.pressure_state_bc.nonorthogonal_eps == 0.1


def test_collocated_momentum_diffusion_matrix_is_cached():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))

    assert model.momentum_diffusion_matrix(model.mu) is model.momentum_diffusion_matrix(model.mu)


def test_collocated_momentum_time_components_match_cell_diagonal():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64).reshape(model.NC, 2)
    density = 3.0
    time_step = 0.25

    matrix = model.momentum_time_matrix(density, time_step)
    source = model.momentum_time_source(velocity, density, time_step)
    expected_diagonal = density * model.cm / time_step
    expected_source = (velocity * expected_diagonal[:, None]).flatten(order="F")

    assert np.allclose(
        np.asarray(bm.to_numpy(matrix.diags().values)),
        np.asarray(bm.to_numpy(bm.concatenate([expected_diagonal, expected_diagonal]))),
    )
    assert np.allclose(np.asarray(bm.to_numpy(source)), np.asarray(bm.to_numpy(expected_source)))


def test_piso_pressure_free_flux_matches_velocity_route():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64).reshape(model.NC, 2)
    pressure = bm.arange(model.NC, dtype=bm.float64)
    a_p = bm.ones(2 * model.NC)

    pressure_free_velocity, flux = model.pressure_free_flux(velocity, pressure, a_p)
    expected_velocity = model.pressure_free_velocity(velocity, pressure, a_p)
    expected_flux = model.compute_face_flux(
        model.face_interpolate_cell_vector(expected_velocity, method=model.controls.face_interpolation_method)
    )

    assert np.allclose(np.asarray(bm.to_numpy(pressure_free_velocity)), np.asarray(bm.to_numpy(expected_velocity)))
    assert np.allclose(np.asarray(bm.to_numpy(flux)), np.asarray(bm.to_numpy(expected_flux)))


def test_piso_pressure_correction_step_uses_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_module

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    nf = model.mesh.number_of_faces()
    nc = model.NC
    pressure_free_velocity = bm.ones((nc, 2), dtype=model.cm.dtype)
    pressure_free_flux = bm.ones(nf, dtype=model.cm.dtype) * 0.25
    seen = []

    def record_pressure_free_flux(intermediate_velocity, pressure, a_p):
        seen.append((intermediate_velocity, pressure, a_p))
        return pressure_free_velocity, pressure_free_flux

    monkeypatch.setattr(model, "pressure_free_flux", record_pressure_free_flux)
    monkeypatch.setattr(model, "transient_face_flux_correction", lambda *args, **kwargs: bm.zeros(nf))
    monkeypatch.setattr(
        piso_module,
        "apply_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity, face_normal, **kwargs: flux,
    )
    monkeypatch.setattr(
        model,
        "solve_pressure_state_equation",
        lambda rhs, a_p, **kwargs: (
            bm.zeros(nc),
            bm.zeros(nf),
            {
                "orthogonal_flux": bm.zeros(nf),
                "cross_flux": bm.zeros(nf),
                "boundary_pressure_flux": bm.zeros(nf),
            },
        ),
    )
    monkeypatch.setattr(model, "velocity_pressure_correction", lambda u_free, pressure_state, a_p: u_free)

    next_velocity, _, phi = model.pressure_correction_step(
        bm.zeros((nc, 2)),
        bm.zeros(nc),
        bm.ones(2 * nc),
        None,
        None,
    )

    assert len(seen) == 1
    assert np.allclose(np.asarray(bm.to_numpy(next_velocity)), np.asarray(bm.to_numpy(pressure_free_velocity)))
    assert np.allclose(np.asarray(bm.to_numpy(phi)), np.asarray(bm.to_numpy(pressure_free_flux)))


def test_rhie_chow_face_velocity_uses_explicit_boundary_velocity():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    u = np.ones((model.NC, 2))
    ap = np.ones(2 * model.NC)
    pressure = np.zeros(model.NC)
    boundary_faces = model.mesh.boundary_face_index()
    boundary_velocity = np.zeros((boundary_faces.shape[0], 2))

    face_velocity = model.rhie_chow_face_velocity(
        u,
        ap,
        pressure,
        boundary_faces=boundary_faces,
        boundary_velocity=boundary_velocity,
    )

    assert np.allclose(np.asarray(face_velocity[boundary_faces]), 0.0)


def test_piso_transient_flux_correction_uses_limited_ddtcorr(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    nf = model.mesh.number_of_faces()
    old_face_flux = bm.array(np.linspace(1.0, 2.0, nf))
    old_cell_flux = old_face_flux - 0.25
    response = bm.ones(nf) * 2.0
    face_flux_returns = [old_face_flux, old_cell_flux]

    monkeypatch.setattr(
        model,
        "compute_face_flux",
        lambda face_velocity: face_flux_returns.pop(0),
    )
    monkeypatch.setattr(
        model,
        "face_interpolate_cell_vector",
        lambda cell_velocity, method: bm.zeros((nf, 2), dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "pressure_response_face_coefficient",
        lambda a_p, interpolation_method: response,
    )

    correction = model.transient_face_flux_correction(
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
        bm.ones(2 * model.NC),
    )

    boundary = np.asarray(
        bm.to_numpy(model.face_to_cell[:, 0] == model.face_to_cell[:, 1])
    )
    flux_correction = np.asarray(bm.to_numpy(old_face_flux - old_cell_flux))
    coeff = 1.0 - np.minimum(
        np.abs(flux_correction) / np.asarray(bm.to_numpy(np.abs(old_face_flux))),
        1.0,
    )
    coeff[boundary] = 0.0
    expected = (
        np.asarray(bm.to_numpy(response))
        * coeff
        * flux_correction
        / model.controls.tau
    )

    assert face_flux_returns == []
    assert np.allclose(np.asarray(bm.to_numpy(correction)), expected)
    assert np.allclose(np.asarray(bm.to_numpy(correction))[boundary], 0.0)


def test_piso_rejects_unknown_options():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["momentum_explicit_correction"] = "openfoam"

    with pytest.raises(
        ValueError,
        match="unsupported NSFVMPISOModel options: momentum_explicit_correction",
    ):
        NSFVMPISOModel(options)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rhie_chow_velocity_scheme", "second_order_reconstructed"),
        ("face_flux_correction_scheme", "cell_anchored_quadratic"),
        ("face_flux_quadrature_order", 4),
        ("face_flux_max_stencil_layers", 5),
        ("face_flux_max_condition", 75.0),
        ("pressure_flux_response_scheme", "face_operator_consistent"),
    ],
)
def test_piso_rejects_retired_high_accuracy_options(name, value):
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options[name] = value

    with pytest.raises(
        ValueError,
        match=rf"unsupported NSFVMPISOModel options: {name}",
    ):
        NSFVMPISOModel(options)


def test_piso_snapshot_callback_respects_interval_and_start_step():
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=4)
    options.update(
        {
            "snapshot_interval": 2,
            "snapshot_start_step": 2,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    called_steps = []

    model.solve(snapshot_callback=lambda *, step, **kwargs: called_steps.append(step))

    assert called_steps == [2, 4]


def test_piso_face_interpolation_option_reaches_momentum_convection(monkeypatch):
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel
    import fealpy.fvm.collocated_ns_components as collocated_components

    seen = []
    original = collocated_components.ConvectionMatrixAssembler

    class RecordingConvectionMatrixAssembler(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("interpolation"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        collocated_components,
        "ConvectionMatrixAssembler",
        RecordingConvectionMatrixAssembler,
    )
    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "face_interpolation_method": "linear",
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()

    model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert seen == ["linear"]
    assert model.rhie_chow.velocity_interpolation == "linear"


def test_piso_face_interpolation_option_reaches_pressure_response(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["face_interpolation_method"] = "linear"
    model = NSFVMPISOModel(options)
    seen = []
    nf = model.mesh.number_of_faces()

    def record_scalar(cell_values, method=None):
        seen.append(method)
        return bm.zeros(nf, dtype=cell_values.dtype)

    monkeypatch.setattr(model, "face_interpolate_cell_scalar", record_scalar)

    model.pressure_response_face_coefficient(
        bm.ones(2 * model.NC),
        model.controls.face_interpolation_method,
    )

    assert seen == ["linear"]


def test_piso_face_interpolation_option_reaches_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_module

    options = _model_options(nx=2, ny=2, nt=1)
    options["face_interpolation_method"] = "linear"
    model = NSFVMPISOModel(options)
    seen = []
    nf = model.mesh.number_of_faces()

    def record_vector(cell_velocity, method=None):
        seen.append(method)
        return bm.zeros((nf, 2), dtype=model.cm.dtype)

    monkeypatch.setattr(model, "face_interpolate_cell_vector", record_vector)
    monkeypatch.setattr(model, "pressure_free_velocity", lambda intermediate_velocity, pressure, a_p: intermediate_velocity)
    monkeypatch.setattr(
        model,
        "compute_face_flux",
        lambda face_velocity: bm.zeros(nf),
    )
    monkeypatch.setattr(model, "transient_face_flux_correction", lambda *args, **kwargs: bm.zeros(nf))
    monkeypatch.setattr(
        piso_module,
        "apply_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity, face_normal, **kwargs: flux,
    )
    monkeypatch.setattr(
        model,
        "solve_pressure_state_equation",
        lambda rhs, a_p, **kwargs: (
            bm.zeros(model.NC),
            bm.zeros(nf),
            {
                "orthogonal_flux": bm.zeros(nf),
                "cross_flux": bm.zeros(nf),
                "boundary_pressure_flux": bm.zeros(nf),
            },
        ),
    )
    monkeypatch.setattr(model, "velocity_pressure_correction", lambda pressure_free_velocity, pressure_state, a_p: pressure_free_velocity)

    model.pressure_correction_step(bm.zeros((model.NC, 2)), bm.zeros(model.NC), bm.ones(2 * model.NC), None, None)

    assert seen == ["linear"]


def test_piso_momentum_nonorthogonal_correction_uses_picard_loop(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "momentum_nonorthogonal_max_iter": 2,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()
    solves = []
    original_solve = model.solve_momentum_system

    def solve_momentum(matrix, rhs):
        solves.append(1)
        return original_solve(matrix, rhs)

    monkeypatch.setattr(model, "solve_momentum_system", solve_momentum)

    U, _, _ = model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert U.shape == U0.shape
    assert len(solves) == 1 + model.last_momentum_nonorthogonal_iterations
    assert model.last_momentum_nonorthogonal_iterations <= 2
    assert (
        model.last_momentum_nonorthogonal_residual["relative"]
        <= model.controls.momentum_nonorthogonal_tol
    )


def test_piso_zero_momentum_nonorthogonal_still_solves_base_equation(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "momentum_nonorthogonal_max_iter": 0,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()
    solves = []

    def forbidden_cross_source(velocity):
        raise AssertionError("zero nonorthogonal corrections should not build cross RHS")

    def solve_momentum(matrix, rhs, *, solver=None):
        solves.append(1)
        return bm.ones(2 * model.NC, dtype=U0.dtype)

    monkeypatch.setattr(model, "boundary_corrected_momentum_explicit_source", forbidden_cross_source, raising=False)
    monkeypatch.setattr(model.linear_solver, "solve", solve_momentum)

    U, _, _ = model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert len(solves) == 1
    assert model.last_momentum_nonorthogonal_iterations == 0
    assert np.allclose(np.asarray(bm.to_numpy(U)), 1.0)


def test_piso_pressure_nonorthogonal_stops_on_complete_residual():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 2
    options["linear_solver_config"] = FVMLinearSolverConfig(solver="scipy")
    model = NSFVMPISOModel(options)
    nc = model.NC

    pressure, _, diagnostics = model.solve_pressure_state_equation(
        bm.zeros(nc, dtype=model.cm.dtype),
        bm.ones(2 * nc, dtype=model.cm.dtype),
        initial_pressure_state=bm.zeros(nc, dtype=model.cm.dtype),
    )

    assert np.allclose(np.asarray(bm.to_numpy(pressure)), 0.0)
    assert model.last_pressure_nonorthogonal_iterations == 0
    assert model.last_pressure_nonorthogonal_residual["relative"] == 0.0
    assert diagnostics["cross_flux"].shape == (model.mesh.number_of_faces(),)


def test_piso_pressure_nonorthogonal_first_rhs_uses_entering_pressure(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_module

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 2
    model = NSFVMPISOModel(options)
    nf = model.mesh.number_of_faces()
    nc = model.NC
    captured_cross_rhs = []
    solve_values = []

    class FakeSolver:
        def solve(self, matrix, rhs, *, solver=None):
            pressure_value = float(len(solve_values) + 2)
            solve_values.append(pressure_value)
            pressure = bm.ones(nc, dtype=model.cm.dtype) * pressure_value
            return bm.concatenate([pressure, bm.zeros(1, dtype=model.cm.dtype)])

    model.linear_solver = FakeSolver()
    monkeypatch.setattr(model, "divergence_from_flux", lambda flux: bm.ones(nc, dtype=model.cm.dtype) * flux[0])
    monkeypatch.setattr(
        model,
        "pressure_nonorthogonal_cross_flux",
            lambda pressure, coef, *, interpolation_method, boundary_threshold=None: (
            bm.ones(nf, dtype=model.cm.dtype) * pressure[0]
        ),
    )
    monkeypatch.setattr(model, "pressure_orthogonal_flux", lambda pressure, coef: bm.zeros(nf, dtype=model.cm.dtype))
    monkeypatch.setattr(model, "add_dirichlet_pressure_flux", lambda flux, pressure, coef, dirichlet_value, threshold: flux)
    original_assemble = model.assemble_pressure_state_system

    def record_assembled_system(rhs, coef, cross_rhs):
        captured_cross_rhs.append(np.asarray(bm.to_numpy(cross_rhs)).copy())
        return original_assemble(rhs, coef, cross_rhs)

    monkeypatch.setattr(model, "assemble_pressure_state_system", record_assembled_system)
    monkeypatch.setattr(
        piso_module,
        "normalized_equation_residual",
        lambda lhs, rhs: {"absolute": 0.0, "relative": 0.0, "scale": 1.0},
    )

    entering_pressure = bm.ones(nc, dtype=model.cm.dtype) * 7.0
    model.solve_pressure_state_equation(
        bm.zeros(nc, dtype=model.cm.dtype),
        bm.ones(2 * nc, dtype=model.cm.dtype),
        initial_pressure_state=entering_pressure,
    )

    assert solve_values == [2.0]
    assert np.allclose(captured_cross_rhs[0], 7.0)
    assert np.allclose(captured_cross_rhs[1], 2.0)


def test_piso_zero_pressure_nonorthogonal_solves_base_system_once(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 0
    model = NSFVMPISOModel(options)
    nc = model.NC
    nf = model.mesh.number_of_faces()
    assembled_cross_rhs = []
    solves = []

    class FakeSolver:
        def solve(self, matrix, rhs, *, solver=None):
            solves.append(1)
            return bm.concatenate([
                bm.ones(nc, dtype=model.cm.dtype) * 3.0,
                bm.zeros(1, dtype=model.cm.dtype),
            ])

    model.linear_solver = FakeSolver()
    monkeypatch.setattr(
        model,
        "pressure_nonorthogonal_cross_flux",
        lambda pressure, coef, *, interpolation_method, boundary_threshold=None: (_ for _ in ()).throw(
            AssertionError("zero nonorthogonal corrections should not build cross flux")
        ),
    )
    monkeypatch.setattr(model, "pressure_orthogonal_flux", lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype))
    monkeypatch.setattr(model, "add_dirichlet_pressure_flux", lambda flux, pressure, coef, dirichlet_value, threshold: flux)

    def record_system(rhs, coef, cross_rhs):
        assembled_cross_rhs.append(np.asarray(bm.to_numpy(cross_rhs)).copy())
        return object(), bm.zeros(nc + 1, dtype=model.cm.dtype)

    monkeypatch.setattr(model, "assemble_pressure_state_system", record_system)

    pressure, pressure_flux, _ = model.solve_pressure_state_equation(
        bm.zeros(nc, dtype=model.cm.dtype),
        bm.ones(2 * nc, dtype=model.cm.dtype),
        initial_pressure_state=bm.ones(nc, dtype=model.cm.dtype),
    )

    assert len(solves) == 1
    assert model.last_pressure_nonorthogonal_iterations == 0
    assert np.allclose(assembled_cross_rhs[0], 0.0)
    assert np.allclose(np.asarray(bm.to_numpy(pressure)), 3.0)
    assert np.allclose(np.asarray(bm.to_numpy(pressure_flux)), 1.0)


def test_piso_corrector_diagnostics_can_be_enabled_without_callback():
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "diagnostics_enabled": True,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)

    model.solve()

    assert len(model.corrector_diagnostics) == model.controls.n_correctors
    first = model.corrector_diagnostics[0]
    assert first["step"] == 1
    assert first["corrector"] == 1
    assert "pressure_free_divergence_linf" in first
    assert "rhie_chow_flux_error_linf" in first


def test_piso_corrector_callback_still_enables_diagnostics():
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update({"linear_solver_config": FVMLinearSolverConfig(solver="scipy"), "log_level": "ERROR"})
    model = NSFVMPISOModel(options)
    rows = []

    model.solve(corrector_callback=lambda **row: rows.append(row))

    assert len(rows) == model.controls.n_correctors
    assert model.corrector_diagnostics == rows
    assert rows[0]["step"] == 1
    assert rows[0]["corrector"] == 1
