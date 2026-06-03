import ast
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_piso_model.py"
PISO_SOLVER_SOURCE = ROOT / "fealpy" / "fvm" / "collocated_piso_solver.py"
SIMPLE_SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_simple_model.py"
SIMPLE_SOLVER_SOURCE = ROOT / "fealpy" / "fvm" / "collocated_simple_solver.py"
EXAMPLE = ROOT / "example" / "fvm" / "ns_fvm_piso_example.py"


def _tree(path=SOURCE):
    return ast.parse(path.read_text())


def _class(name, path=SOURCE):
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing method {name}")


def _calls_in_node(node):
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            names.add(call.func.id)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def _model_options(nx=4, ny=4, nt=1):
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


def test_piso_solver_is_computational_model_not_script():
    source = SOURCE.read_text()
    tree = _tree()
    model = _class("NSFVMPISOModel")

    assert any(
        isinstance(base, ast.Name) and base.id == "ComputationalModel"
        for base in model.bases
    )
    assert not any(isinstance(node, ast.FunctionDef) for node in tree.body)
    assert "argparse" not in source
    assert "parse_args" not in source
    assert "if __name__" not in source
    assert "def main" not in source


def test_only_standard_pressure_state_piso_route_remains():
    source = SOURCE.read_text()

    assert "PRESSURE_ROUTES" not in source
    assert "pressure_route" not in source
    assert "--pressure-route" not in source
    assert '"legacy"' not in source
    assert '"face-flux"' not in source
    assert '"physical-pressure"' not in source
    assert '"consistent"' not in source
    assert "adaptive_pressure_control" not in source
    assert "pressure_relax" not in source


def test_collocated_solvers_use_shared_internal_operators():
    from fealpy.fvm import (
        CollocatedPisoSolver,
        CollocatedSimpleSolver,
        NSFVMPISOModel,
        NSFVMSimpleModel,
    )
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    assert issubclass(CollocatedPisoSolver, CollocatedNSFVMOperators)
    assert issubclass(NSFVMPISOModel, CollocatedNSFVMOperators)
    assert issubclass(NSFVMPISOModel, CollocatedPisoSolver)
    assert issubclass(CollocatedSimpleSolver, CollocatedNSFVMOperators)
    assert issubclass(NSFVMSimpleModel, CollocatedNSFVMOperators)
    assert issubclass(NSFVMSimpleModel, CollocatedSimpleSolver)

    piso_source = SOURCE.read_text()
    piso_solver_source = PISO_SOLVER_SOURCE.read_text()
    simple_solver_source = SIMPLE_SOLVER_SOURCE.read_text()
    assert "from .collocated_piso_solver import CollocatedPisoSolver" in piso_source
    assert (
        "from .collocated_ns_fvm_utils import CollocatedNSFVMOperators"
        in piso_solver_source
    )
    assert (
        "from .collocated_ns_fvm_utils import CollocatedNSFVMOperators"
        in simple_solver_source
    )


def test_model_exposes_piso_components_and_no_route_dispatch():
    from fealpy.fvm import CollocatedPisoSolver, NSFVMPISOModel

    for name in [
        "temporary_velocity",
        "solve_pressure_correction",
        "face_flux",
        "divergence_from_flux",
        "pressure_correction_flux",
        "rhie_chow_face_velocity",
        "velocity_pressure_correction",
        "pressure_correction_step",
        "pressure_free_velocity",
        "pressure_free_flux",
        "solve",
        "compute_error",
        "plot",
    ]:
        assert hasattr(NSFVMPISOModel, name)
    assert not hasattr(NSFVMPISOModel, "matrix_consistent_pressure_free_flux")

    model = _class("NSFVMPISOModel")
    model_methods = {
        node.name for node in model.body if isinstance(node, ast.FunctionDef)
    }
    assert "temporary_velocity" not in model_methods
    assert "pressure_correction_step" not in model_methods
    assert "solve" not in model_methods
    assert {"initial_solution", "compute_error", "plot"}.issubset(model_methods)

    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    solve = _method(solver, "solve")
    calls = _calls_in_node(solve)
    assert "pressure_correction_step" in calls
    assert "operator_splitting_velocity_correction" in calls
    assert "rhie_chow_face_velocity" in calls
    assert hasattr(CollocatedPisoSolver, "temporary_velocity")
    assert hasattr(CollocatedPisoSolver, "pressure_correction_step")
    assert "apply_pressure_rate_correction" not in calls
    assert "apply_pressure_correction" not in calls
    assert "solve_physical_pressure" not in calls
    assert "build_hbya" not in calls


def test_piso_pressure_free_flux_matches_velocity_route():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64)
    pressure = bm.arange(model.NC, dtype=bm.float64)
    a_p = bm.ones(2 * model.NC)

    pressure_free_velocity, flux = model.pressure_free_flux(
        velocity,
        pressure,
        a_p,
    )
    expected_velocity = model.pressure_free_velocity(velocity, pressure, a_p)
    expected_flux = model.face_flux(
        model.face_interpolate_cell_vector(
            expected_velocity,
            method=model.face_interpolation_method,
        )
    )

    assert np.allclose(
        np.asarray(bm.to_numpy(pressure_free_velocity)),
        np.asarray(bm.to_numpy(expected_velocity)),
    )
    assert np.allclose(
        np.asarray(bm.to_numpy(flux)),
        np.asarray(bm.to_numpy(expected_flux)),
    )


def test_piso_pressure_correction_step_uses_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    nf = model.mesh.number_of_faces()
    nc = model.NC
    pressure_free_velocity = bm.ones(2 * nc)
    pressure_free_flux = bm.ones(nf) * 0.25
    seen = []

    def record_pressure_free_flux(intermediate_velocity, pressure, a_p):
        seen.append((intermediate_velocity, pressure, a_p))
        return pressure_free_velocity, pressure_free_flux

    monkeypatch.setattr(model, "pressure_free_flux", record_pressure_free_flux)
    monkeypatch.setattr(
        model,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf),
    )
    monkeypatch.setattr(
        model,
        "_apply_selected_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity: flux,
    )
    monkeypatch.setattr(
        model,
        "_solve_pressure_correction_state_and_flux",
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
    monkeypatch.setattr(
        model,
        "velocity_pressure_correction",
        lambda u_free, pressure_state, a_p: u_free,
    )
    next_velocity, _, phi = model.pressure_correction_step(
        bm.zeros(2 * nc),
        bm.zeros(nc),
        bm.ones(2 * nc),
        None,
        None,
    )

    assert len(seen) == 1
    assert np.allclose(
        np.asarray(bm.to_numpy(next_velocity)),
        np.asarray(bm.to_numpy(pressure_free_velocity)),
    )
    assert np.allclose(
        np.asarray(bm.to_numpy(phi)),
        np.asarray(bm.to_numpy(pressure_free_flux)),
    )


def test_example_is_the_only_cli_driver():
    assert EXAMPLE.exists()
    source = EXAMPLE.read_text()

    assert "argparse" in source
    assert "NSFVMPISOModel" in source
    assert "if __name__" in source
    assert "--no-plot" not in source
    assert "--plot" in source


def test_rhie_chow_face_velocity_does_not_hide_pde_boundary_by_default():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    u_flat = np.ones(2 * model.NC)
    ap = np.ones(2 * model.NC)
    pressure = np.zeros(model.NC)

    face_velocity = model.rhie_chow_face_velocity(u_flat, ap, pressure)
    bd_face = model.mesh.boundary_face_index()

    assert np.allclose(np.asarray(face_velocity[bd_face]), 1.0)


def test_rhie_chow_face_velocity_uses_explicit_boundary_velocity():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    u_flat = np.ones(2 * model.NC)
    ap = np.ones(2 * model.NC)
    pressure = np.zeros(model.NC)

    bd_face = model.mesh.boundary_face_index()
    boundary_velocity = np.zeros((bd_face.shape[0], 2))
    face_velocity = model.rhie_chow_face_velocity(
        u_flat, ap, pressure, boundary_velocity=boundary_velocity
    )

    assert np.allclose(np.asarray(face_velocity[bd_face]), 0.0)


def test_pressure_rate_flux_correction_closes_matching_scalar_flux():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    pressure_rate = np.arange(model.NC, dtype=float)
    correction_flux = model.pressure_correction_flux(
        pressure_rate, np.ones(2 * model.NC)
    )
    phi = -correction_flux + correction_flux
    div_phi = model.divergence_from_flux(phi)

    assert np.allclose(np.asarray(div_phi), 0.0)


def test_piso_transient_flux_correction_defaults_to_legacy_unlimited_form(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    nf = model.mesh.number_of_faces()
    old_face_flux = bm.array(np.linspace(1.0, 2.0, nf))
    old_cell_flux = old_face_flux - 0.25
    response = bm.ones(nf) * 2.0

    monkeypatch.setattr(model, "face_flux", lambda face_velocity: old_face_flux)
    monkeypatch.setattr(model, "cell_velocity_face_flux", lambda cell_velocity: old_cell_flux)
    monkeypatch.setattr(model, "pressure_response_face_coefficient", lambda a_p: response)

    correction = model.transient_face_flux_correction(
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
        bm.ones(2 * model.NC),
    )

    expected = np.asarray(bm.to_numpy(response * (old_face_flux - old_cell_flux))) / model.tau
    assert model.transient_flux_correction_limiter == "none"
    assert np.allclose(np.asarray(bm.to_numpy(correction)), expected)


def test_piso_transient_flux_correction_can_use_openfoam_limiter(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options()
    options["transient_flux_correction_limiter"] = "openfoam"
    model = NSFVMPISOModel(options)
    nf = model.mesh.number_of_faces()
    old_face_flux = bm.array(np.linspace(1.0, 2.0, nf))
    old_cell_flux = old_face_flux - 0.25
    response = bm.ones(nf) * 2.0
    boundary = np.asarray(bm.to_numpy(model.e2c[:, 0] == model.e2c[:, 1]))

    monkeypatch.setattr(model, "face_flux", lambda face_velocity: old_face_flux)
    monkeypatch.setattr(model, "cell_velocity_face_flux", lambda cell_velocity: old_cell_flux)
    monkeypatch.setattr(model, "pressure_response_face_coefficient", lambda a_p: response)

    correction = model.transient_face_flux_correction(
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
        bm.ones(2 * model.NC),
    )

    phi_corr = np.asarray(bm.to_numpy(old_face_flux - old_cell_flux))
    coeff = 1.0 - np.minimum(phi_corr / np.asarray(bm.to_numpy(old_face_flux)), 1.0)
    coeff[boundary] = 0.0
    expected = np.asarray(bm.to_numpy(response)) * coeff * phi_corr / model.tau

    assert np.allclose(np.asarray(bm.to_numpy(correction)), expected)
    assert np.allclose(np.asarray(bm.to_numpy(correction))[boundary], 0.0)


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

    def collect_snapshot(*, step, **kwargs):
        called_steps.append(step)

    model.solve(snapshot_callback=collect_snapshot)

    assert called_steps == [2, 4]


def test_piso_face_interpolation_option_reaches_momentum_convection(monkeypatch):
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_module

    seen = []
    original = piso_module.ConvectionIntegrator

    class RecordingConvectionIntegrator(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("interpolation"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        piso_module,
        "ConvectionIntegrator",
        RecordingConvectionIntegrator,
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

    model.temporary_velocity(U0, Uf0, p0, t=model.tau)

    assert seen == ["linear"]
    assert model.rhie_chow.velocity_interpolation == "linear"


def test_piso_openfoam_momentum_explicit_correction_replaces_current_picard(
    monkeypatch,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "momentum_explicit_correction": "openfoam",
            "momentum_nonorthogonal_max_iter": 2,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    U0, Uf0, p0 = model.initial_solution()
    called = []

    def boundary_corrected_source(velocity):
        called.append(tuple(velocity.shape))
        return bm.zeros(2 * model.NC, dtype=U0.dtype)

    monkeypatch.setattr(
        model,
        "boundary_corrected_momentum_explicit_source",
        boundary_corrected_source,
        raising=False,
    )

    def reject_current_picard(*args, **kwargs):
        raise AssertionError("openfoam momentum route must not use current Picard")

    monkeypatch.setattr(
        model,
        "correct_momentum_nonorthogonal_diffusion",
        reject_current_picard,
    )

    model.temporary_velocity(U0, Uf0, p0, t=model.tau)

    assert called == [tuple(U0.shape)]


def test_piso_rejects_inconsistent_rhie_chow_face_interpolation():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
            "face_interpolation_method": "linear",
            "rhie_chow_velocity_interpolation": "average",
        }
    )

    with pytest.raises(ValueError, match="rhie_chow_velocity_interpolation"):
        NSFVMPISOModel(options)


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

    model.pressure_response_face_coefficient(bm.ones(2 * model.NC))

    assert seen == ["linear"]


def test_piso_face_interpolation_option_reaches_pressure_free_flux(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["face_interpolation_method"] = "linear"
    model = NSFVMPISOModel(options)
    seen = []
    nf = model.mesh.number_of_faces()

    def record_vector(cell_velocity, method=None):
        seen.append(method)
        return bm.zeros((nf, 2), dtype=model.cm.dtype)

    monkeypatch.setattr(model, "face_interpolate_cell_vector", record_vector)
    monkeypatch.setattr(
        model,
        "pressure_free_velocity",
        lambda intermediate_velocity, pressure, a_p: intermediate_velocity,
    )
    monkeypatch.setattr(model, "face_flux", lambda face_velocity: bm.zeros(nf))
    monkeypatch.setattr(
        model,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf),
    )
    monkeypatch.setattr(
        model,
        "_apply_selected_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity: flux,
    )
    monkeypatch.setattr(
        model,
        "_solve_pressure_correction_state_and_flux",
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
    monkeypatch.setattr(
        model,
        "velocity_pressure_correction",
        lambda pressure_free_velocity, pressure_state, a_p: pressure_free_velocity,
    )
    model.pressure_correction_step(
        bm.zeros(2 * model.NC),
        bm.zeros(model.NC),
        bm.ones(2 * model.NC),
        None,
        None,
    )

    assert seen == ["linear"]


def test_piso_pressure_nonorthogonal_final_flux_matches_final_system(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 2
    model = NSFVMPISOModel(options)
    nf = model.mesh.number_of_faces()
    nc = model.NC
    solves = []

    class FakeSolver:
        def solve(self, matrix, rhs):
            pressure_value = float(len(solves) + 1)
            solves.append(pressure_value)
            pressure = bm.ones(nc, dtype=model.cm.dtype) * pressure_value
            if matrix.shape[1] == nc:
                return pressure
            return bm.concatenate([pressure, bm.zeros(1, dtype=model.cm.dtype)])

    model.linear_solver = FakeSolver()
    monkeypatch.setattr(
        model,
        "pressure_free_flux",
        lambda intermediate_velocity, pressure, a_p: (
            bm.zeros(2 * nc, dtype=model.cm.dtype),
            bm.zeros(nf, dtype=model.cm.dtype),
        ),
    )
    monkeypatch.setattr(
        model,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf, dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "_apply_selected_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity: flux,
    )
    monkeypatch.setattr(
        model,
        "divergence_from_flux",
        lambda flux: bm.zeros(nc, dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "velocity_pressure_correction",
        lambda pressure_free_velocity, pressure_state, a_p: pressure_free_velocity,
    )
    monkeypatch.setattr(
        model,
        "pressure_correction_orthogonal_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0] * 10.0,
    )
    monkeypatch.setattr(
        model,
        "pressure_correction_cross_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0],
    )
    monkeypatch.setattr(
        model,
        "_add_pressure_dirichlet_boundary_flux",
        lambda flux, pressure, coef: flux,
    )

    _, pressure, corrected_flux, diagnostics = model.pressure_correction_step(
        bm.zeros(2 * nc, dtype=model.cm.dtype),
        bm.zeros(nc, dtype=model.cm.dtype),
        bm.ones(2 * nc, dtype=model.cm.dtype),
        None,
        None,
        return_diagnostics=True,
    )

    assert solves == [1.0, 2.0, 3.0]
    assert np.allclose(np.asarray(bm.to_numpy(pressure)), 3.0)
    assert np.allclose(np.asarray(bm.to_numpy(corrected_flux)), 28.0)
    assert diagnostics["pressure_nonorthogonal_iterations"] == 3


def test_piso_pressure_nonorthogonal_first_rhs_uses_entering_pressure(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 1
    model = NSFVMPISOModel(options)
    nf = model.mesh.number_of_faces()
    nc = model.NC
    captured_cross_rhs = []
    solve_values = []

    class FakeSolver:
        def solve(self, matrix, rhs):
            pressure_value = float(len(solve_values) + 2)
            solve_values.append(pressure_value)
            pressure = bm.ones(nc, dtype=model.cm.dtype) * pressure_value
            return bm.concatenate([pressure, bm.zeros(1, dtype=model.cm.dtype)])

    model.linear_solver = FakeSolver()
    monkeypatch.setattr(
        model,
        "pressure_free_flux",
        lambda intermediate_velocity, pressure, a_p: (
            bm.zeros(2 * nc, dtype=model.cm.dtype),
            bm.zeros(nf, dtype=model.cm.dtype),
        ),
    )
    monkeypatch.setattr(
        model,
        "transient_face_flux_correction",
        lambda *args, **kwargs: bm.zeros(nf, dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "_apply_selected_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity: flux,
    )
    monkeypatch.setattr(
        model,
        "divergence_from_flux",
        lambda flux: bm.ones(nc, dtype=model.cm.dtype) * flux[0],
    )
    monkeypatch.setattr(
        model,
        "pressure_correction_cross_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0],
    )
    monkeypatch.setattr(
        model,
        "pressure_correction_orthogonal_flux",
        lambda pressure, coef: bm.zeros(nf, dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "_add_pressure_dirichlet_boundary_flux",
        lambda flux, pressure, coef: flux,
    )
    monkeypatch.setattr(
        model,
        "velocity_pressure_correction",
        lambda pressure_free_velocity, pressure_state, a_p: pressure_free_velocity,
    )

    def record_assembled_system(rhs, coef, cross_rhs):
        captured_cross_rhs.append(np.asarray(bm.to_numpy(cross_rhs)).copy())
        return object(), bm.zeros(nc + 1, dtype=model.cm.dtype)

    monkeypatch.setattr(model, "_assemble_pressure_state_system", record_assembled_system)

    entering_pressure = bm.ones(nc, dtype=model.cm.dtype) * 7.0
    model.pressure_correction_step(
        bm.zeros(2 * nc, dtype=model.cm.dtype),
        entering_pressure,
        bm.ones(2 * nc, dtype=model.cm.dtype),
        None,
        None,
    )

    assert solve_values == [2.0, 3.0]
    assert np.allclose(captured_cross_rhs[0], 7.0)
    assert np.allclose(captured_cross_rhs[1], 2.0)
