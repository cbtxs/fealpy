import ast
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_piso_model.py"
PISO_SOLVER_SOURCE = ROOT / "fealpy" / "fvm" / "collocated_piso_solver.py"
SIMPLE_SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_simple_model.py"
SIMPLE_SOLVER_SOURCE = ROOT / "fealpy" / "fvm" / "collocated_simple_solver.py"
COLLOCATED_NS_UTILS = ROOT / "fealpy" / "fvm" / "collocated_ns_fvm_utils.py"
FVM_INIT = ROOT / "fealpy" / "fvm" / "__init__.py"
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


def test_collocated_shared_operators_do_not_depend_on_model_adapter_semantics():
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    source = COLLOCATED_NS_UTILS.read_text()

    assert "PDEModelManager" not in source
    assert "fealpy.model" not in source
    for name in [
        "_resolve_navier_stokes_pde",
        "_normalized_mesh_type",
        "_option_int",
        "_init_momentum_coefficients",
        "_init_navier_stokes_mesh",
    ]:
        assert not hasattr(CollocatedNSFVMOperators, name)


def test_collocated_shared_operators_provide_momentum_discrete_components():
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    assert not hasattr(CollocatedNSFVMOperators, "momentum_transport_matrix")
    assert hasattr(CollocatedNSFVMOperators, "momentum_diffusion_matrix")
    assert hasattr(CollocatedNSFVMOperators, "momentum_convection_matrix")
    assert hasattr(CollocatedNSFVMOperators, "momentum_time_matrix")
    assert hasattr(CollocatedNSFVMOperators, "momentum_time_source")
    assert hasattr(CollocatedNSFVMOperators, "add_velocity_natural_convection_diagonal")
    assert hasattr(CollocatedNSFVMOperators, "momentum_source_vector")
    assert hasattr(CollocatedNSFVMOperators, "pressure_gradient_source")

    simple_solver = _class("CollocatedSimpleSolver", SIMPLE_SOLVER_SOURCE)
    simple_momentum = _method(simple_solver, "temporary_velocity")
    simple_calls = _calls_in_node(simple_momentum)
    assert "momentum_diffusion_matrix" in simple_calls
    assert "momentum_convection_matrix" in simple_calls
    assert "momentum_source_vector" in simple_calls
    assert "pressure_gradient_source" in simple_calls
    simple_natural = _method(simple_solver, "_apply_velocity_natural_convection")
    simple_natural_calls = _calls_in_node(simple_natural)
    assert "add_velocity_natural_convection_diagonal" in simple_natural_calls

    piso_solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    piso_momentum = _method(piso_solver, "temporary_velocity")
    piso_calls = _calls_in_node(piso_momentum)
    assert "momentum_diffusion_matrix" in piso_calls
    assert "momentum_convection_matrix" in piso_calls
    assert "momentum_time_matrix" in piso_calls
    assert "momentum_time_source" in piso_calls
    assert "momentum_source_vector" in piso_calls
    assert "pressure_gradient_source" in piso_calls
    piso_natural = _method(piso_solver, "add_momentum_natural_convection_boundary")
    piso_natural_calls = _calls_in_node(piso_natural)
    assert "add_velocity_natural_convection_diagonal" in piso_natural_calls


def test_piso_physical_velocity_methods_use_cell_velocity_shape():
    piso_solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    shared_operators = _class("CollocatedNSFVMOperators", COLLOCATED_NS_UTILS)

    for owner, name in [
        (piso_solver, "temporary_velocity"),
        (piso_solver, "boundary_corrected_momentum_explicit_source"),
        (piso_solver, "boundary_corrected_momentum_face_gradient"),
        (piso_solver, "piso_neighbour_velocity_correction"),
        (shared_operators, "pressure_free_velocity"),
        (shared_operators, "velocity_pressure_correction"),
    ]:
        calls = _calls_in_node(_method(owner, name))
        assert "_cell_velocity" not in calls
        assert "_flatten_velocity" not in calls

    temporary_velocity = _method(piso_solver, "temporary_velocity")
    temporary_source = ast.get_source_segment(
        PISO_SOLVER_SOURCE.read_text(),
        temporary_velocity,
    )
    assert "bm.stack" in temporary_source


def test_piso_velocity_natural_boundary_is_normalized_at_initialization():
    from fealpy.fvm import CollocatedPisoSolver

    assert not hasattr(CollocatedPisoSolver, "_velocity_natural_threshold")
    assert not hasattr(CollocatedPisoSolver, "_velocity_natural_faces")

    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    constructor = _method(solver, "__init__")
    init_calls = _calls_in_node(constructor)
    assert "natural_threshold" in init_calls

    face_gradient = _method(solver, "boundary_corrected_momentum_face_gradient")
    face_gradient_source = ast.get_source_segment(
        PISO_SOLVER_SOURCE.read_text(),
        face_gradient,
    )
    assert "self.velocity_natural_threshold" in face_gradient_source
    assert "_velocity_natural" not in face_gradient_source

    assert not hasattr(CollocatedPisoSolver, "_apply_velocity_natural_convection")
    assert hasattr(CollocatedPisoSolver, "add_momentum_natural_convection_boundary")

    natural_convection = _method(solver, "add_momentum_natural_convection_boundary")
    natural_convection_source = ast.get_source_segment(
        PISO_SOLVER_SOURCE.read_text(),
        natural_convection,
    )
    assert "self.velocity_natural_threshold" in natural_convection_source
    assert "_velocity_natural_threshold" not in natural_convection_source
    assert "_velocity_natural_faces" not in natural_convection_source


def test_collocated_momentum_diffusion_matrix_is_cached():
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))

    first = model.momentum_diffusion_matrix(model.mu)
    second = model.momentum_diffusion_matrix(model.mu)

    assert first is second


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

    assert np.allclose(
        np.asarray(bm.to_numpy(matrix.diags().values)),
        np.asarray(bm.to_numpy(bm.concatenate([expected_diagonal, expected_diagonal]))),
    )
    assert np.allclose(
        np.asarray(bm.to_numpy(source)),
        np.asarray(
            bm.to_numpy((velocity * expected_diagonal[:, None]).flatten(order="F"))
        ),
    )


def test_collocated_shared_operators_provide_pressure_flux_components():
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    assert hasattr(CollocatedNSFVMOperators, "pressure_response_face_coefficient")
    assert hasattr(CollocatedNSFVMOperators, "pressure_orthogonal_flux")
    assert hasattr(CollocatedNSFVMOperators, "add_pressure_dirichlet_flux")

    simple_solver = _class("CollocatedSimpleSolver", SIMPLE_SOLVER_SOURCE)
    simple_flux = _method(simple_solver, "pressure_correction_flux")
    simple_flux_calls = _calls_in_node(simple_flux)
    assert "pressure_orthogonal_flux" in simple_flux_calls
    simple_boundary_flux = _method(simple_solver, "_add_pressure_dirichlet_boundary_flux")
    simple_boundary_calls = _calls_in_node(simple_boundary_flux)
    assert "add_pressure_dirichlet_flux" in simple_boundary_calls

    simple_solve = _method(simple_solver, "solve")
    simple_solve_calls = _calls_in_node(simple_solve)
    assert "pressure_response_face_coefficient" in simple_solve_calls

    piso_solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    piso_pressure_solve = _method(piso_solver, "solve_pressure_state_equation")
    piso_pressure_solve_calls = _calls_in_node(piso_pressure_solve)
    assert "pressure_response_face_coefficient" in piso_pressure_solve_calls
    assert "pressure_orthogonal_flux" in piso_pressure_solve_calls
    assert "_pressure_nonorthogonal_cross_flux" in piso_pressure_solve_calls
    assert "add_pressure_dirichlet_flux" in piso_pressure_solve_calls


def test_piso_solver_keeps_controls_in_control_container():
    from fealpy.fvm import CollocatedPisoSolver

    for name in [
        "validate_piso_controls",
        "validate_snapshot_controls",
        "validate_face_interpolation_method",
        "validate_nonorthogonal_controls",
    ]:
        assert name not in CollocatedPisoSolver.__dict__

    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    init_solver = _method(solver, "__init__")
    assigned = {
        target.attr
        for node in ast.walk(init_solver)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }
    assert not assigned.intersection(
        {
            "duration",
            "nt",
            "tau",
            "n_correctors",
            "snapshot_interval",
            "snapshot_start_step",
            "use_transient_flux_correction",
            "face_interpolation_method",
            "momentum_nonorthogonal_max_iter",
            "momentum_nonorthogonal_tol",
            "pressure_nonorthogonal_max_iter",
            "pressure_nonorthogonal_tol",
            "diagnostics_enabled",
        }
    )


def test_piso_model_passes_computational_model_logger_to_solver_constructor():
    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    constructor = _method(solver, "__init__")
    init_parameters = {parameter.arg for parameter in constructor.args.kwonlyargs}

    assert "logger" in init_parameters
    assert "log_level" in init_parameters
    assert "pbar_log" in init_parameters

    model = _class("NSFVMPISOModel")
    model_init = _method(model, "__init__")
    solver_init_call = next(
        call
        for call in ast.walk(model_init)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "__init__"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "CollocatedPisoSolver"
    )
    init_solver_keywords = {keyword.arg for keyword in solver_init_call.keywords}
    solver_source = PISO_SOLVER_SOURCE.read_text()
    assert "logger" in init_solver_keywords
    assert "pbar_log" not in init_solver_keywords
    assert "log_level" not in init_solver_keywords
    assert "self.logger = logger or self._build_logger" in solver_source


def test_model_exposes_piso_components_and_no_route_dispatch():
    from fealpy.fvm import CollocatedPisoSolver, NSFVMPISOModel
    from fealpy.fvm import solver_diagnostics
    import fealpy.fvm.engineering_boundary_conditions as boundary_module

    for name in [
        "temporary_velocity",
        "face_flux",
        "divergence_from_flux",
        "solve_pressure_state_equation",
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
    solver_methods = {
        node.name for node in solver.body if isinstance(node, ast.FunctionDef)
    }
    assert "__getattr__" not in solver_methods
    assert "_init_piso_solver" not in solver_methods
    assert "_boundary_face_velocity" not in solver_methods
    assert "_velocity_dirichlet_data" not in solver_methods
    assert "_velocity_dirichlet_threshold" not in solver_methods
    assert "_pressure_dirichlet_data" not in solver_methods
    assert "_pressure_dirichlet_threshold" not in solver_methods
    assert "_has_pressure_dirichlet" not in solver_methods
    solve = _method(solver, "solve")
    calls = _calls_in_node(solve)
    assert "advance_time_step" in calls

    advance = _method(solver, "advance_time_step")
    advance_calls = _calls_in_node(advance)
    assert "temporary_velocity" in advance_calls
    assert "piso_pressure_corrector_loop" in advance_calls
    assert "rhie_chow_face_velocity" in advance_calls

    correctors = _method(solver, "piso_pressure_corrector_loop")
    corrector_calls = _calls_in_node(correctors)
    assert "pressure_correction_step" in corrector_calls
    assert "piso_neighbour_velocity_correction" in corrector_calls
    assert "record_piso_corrector_diagnostics" in corrector_calls
    assert "_advance_time_step" not in solver_methods
    assert "_run_pressure_correctors" not in solver_methods
    assert "_record_corrector_diagnostics" not in solver_methods

    assert not hasattr(CollocatedPisoSolver, "_pressure_correction_diagnostics")
    assert hasattr(solver_diagnostics, "record_piso_corrector_diagnostics")
    assert not hasattr(CollocatedPisoSolver, "_apply_selected_face_velocity_dirichlet")
    assert not hasattr(CollocatedPisoSolver, "_apply_selected_boundary_flux_constraint")
    assert hasattr(CollocatedPisoSolver, "_assemble_pressure_state_system")
    assert not hasattr(CollocatedPisoSolver, "_add_pressure_dirichlet_boundary_flux")
    assert not (ROOT / "fealpy" / "fvm" / "piso_pressure_equation.py").exists()
    assert not (ROOT / "fealpy" / "fvm" / "solver_boundary.py").exists()
    assert "piso_pressure_equation" not in PISO_SOLVER_SOURCE.read_text()
    assert "solver_boundary" not in PISO_SOLVER_SOURCE.read_text()
    assert "solver_boundary" not in FVM_INIT.read_text()
    assert hasattr(boundary_module, "apply_face_velocity_constraint")
    assert hasattr(boundary_module, "apply_boundary_flux_constraint")
    assert hasattr(boundary_module, "selected_boundary_faces")
    assert hasattr(solver_diagnostics, "pressure_correction_diagnostics")
    assert hasattr(CollocatedPisoSolver, "temporary_velocity")
    assert hasattr(CollocatedPisoSolver, "pressure_correction_step")
    assert not hasattr(CollocatedPisoSolver, "solve_pressure_state")
    assert not hasattr(CollocatedPisoSolver, "pressure_state_flux")
    assert not hasattr(CollocatedPisoSolver, "pressure_state_flux_from_coefficient")
    assert not hasattr(CollocatedPisoSolver, "pressure_nonorthogonal_cross_flux")
    assert not hasattr(CollocatedPisoSolver, "cell_velocity_face_flux")
    assert not hasattr(CollocatedPisoSolver, "solve_pressure_correction")
    assert not hasattr(CollocatedPisoSolver, "pressure_correction_flux")
    assert not hasattr(CollocatedPisoSolver, "pressure_correction_flux_from_coefficient")
    assert not hasattr(CollocatedPisoSolver, "pressure_correction_orthogonal_flux")
    assert not hasattr(CollocatedPisoSolver, "pressure_correction_cross_flux")
    assert not hasattr(CollocatedPisoSolver, "operator_splitting_velocity_correction")
    assert "apply_pressure_rate_correction" not in calls
    assert "apply_pressure_correction" not in calls
    assert "solve_physical_pressure" not in calls
    assert "build_hbya" not in calls


def test_piso_pressure_free_flux_matches_velocity_route():
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options(nx=2, ny=2, nt=1))
    velocity = bm.arange(2 * model.NC, dtype=bm.float64).reshape(model.NC, 2)
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
            method=model.controls.face_interpolation_method,
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
    import fealpy.fvm.collocated_piso_solver as piso_module

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


def test_piso_transient_flux_correction_uses_limited_ddtcorr(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel

    model = NSFVMPISOModel(_model_options())
    nf = model.mesh.number_of_faces()
    old_face_flux = bm.array(np.linspace(1.0, 2.0, nf))
    old_cell_flux = old_face_flux - 0.25
    response = bm.ones(nf) * 2.0

    face_flux_returns = [old_face_flux, old_cell_flux]
    monkeypatch.setattr(model, "face_flux", lambda face_velocity: face_flux_returns.pop(0))
    monkeypatch.setattr(
        model,
        "face_interpolate_cell_vector",
        lambda cell_velocity, method: bm.zeros((nf, 2), dtype=model.cm.dtype),
    )
    monkeypatch.setattr(model, "pressure_response_face_coefficient", lambda a_p: response)

    correction = model.transient_face_flux_correction(
        bm.zeros((model.NC, 2)),
        bm.zeros((nf, 2)),
        bm.ones(2 * model.NC),
    )

    boundary = np.asarray(bm.to_numpy(model.e2c[:, 0] == model.e2c[:, 1]))
    phi_corr = np.asarray(bm.to_numpy(old_face_flux - old_cell_flux))
    coeff = 1.0 - np.minimum(phi_corr / np.asarray(bm.to_numpy(old_face_flux)), 1.0)
    coeff[boundary] = 0.0
    expected = np.asarray(bm.to_numpy(response)) * coeff * phi_corr / model.controls.tau

    assert face_flux_returns == []
    assert np.allclose(np.asarray(bm.to_numpy(correction)), expected)
    assert np.allclose(np.asarray(bm.to_numpy(correction))[boundary], 0.0)


@pytest.mark.parametrize("legacy_limiter", ["none", "openfoam"])
def test_piso_rejects_legacy_transient_flux_limiter_option(legacy_limiter):
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options()
    options["transient_flux_correction_limiter"] = legacy_limiter

    with pytest.raises(ValueError, match="transient_flux_correction_limiter.*removed"):
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

    def collect_snapshot(*, step, **kwargs):
        called_steps.append(step)

    model.solve(snapshot_callback=collect_snapshot)

    assert called_steps == [2, 4]


def test_piso_face_interpolation_option_reaches_momentum_convection(monkeypatch):
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel
    import fealpy.fvm.collocated_ns_fvm_utils as collocated_utils

    seen = []
    original = collocated_utils.ConvectionIntegrator

    class RecordingConvectionIntegrator(original):
        def __init__(self, *args, **kwargs):
            seen.append(kwargs.get("interpolation"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        collocated_utils,
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

    _, _, _ = model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert seen == ["linear"]
    assert model.rhie_chow.velocity_interpolation == "linear"


def test_piso_momentum_predictor_always_uses_explicit_correction(
    monkeypatch,
):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import FVMLinearSolverConfig, NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options.update(
        {
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
        raise AssertionError("PISO momentum predictor must not use current Picard")

    monkeypatch.setattr(
        model,
        "correct_momentum_nonorthogonal_diffusion",
        reject_current_picard,
        raising=False,
    )

    U, a_p, matrix = model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert called == [tuple(U0.shape)]
    assert U.shape == U0.shape
    assert a_p.shape == (2 * model.NC,)
    assert matrix.shape == (2 * model.NC, 2 * model.NC)


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
    source_inputs = []
    solves = []

    def boundary_corrected_source(velocity):
        source_inputs.append(tuple(velocity.shape))
        return bm.zeros(2 * model.NC, dtype=U0.dtype)

    def solve_momentum(matrix, rhs):
        solves.append(1)
        return bm.ones(2 * model.NC, dtype=U0.dtype) * len(solves)

    monkeypatch.setattr(
        model,
        "boundary_corrected_momentum_explicit_source",
        boundary_corrected_source,
        raising=False,
    )
    monkeypatch.setattr(model.linear_solver, "solve", solve_momentum)

    U, _, _ = model.temporary_velocity(U0, Uf0, p0, t=model.controls.tau)

    assert source_inputs == [tuple(U0.shape), tuple(U0.shape)]
    assert len(solves) == 2
    assert model.last_momentum_nonorthogonal_iterations == 2
    assert np.allclose(np.asarray(bm.to_numpy(U)), 2.0)


def test_piso_nonorthogonal_loops_use_correction_state_names():
    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    momentum = _method(solver, "temporary_velocity")
    pressure = _method(solver, "solve_pressure_state_equation")
    momentum_source = ast.get_source_segment(PISO_SOLVER_SOURCE.read_text(), momentum)
    pressure_source = ast.get_source_segment(PISO_SOLVER_SOURCE.read_text(), pressure)

    assert "correction_velocity" in momentum_source
    assert "old_U" in momentum_source
    assert "correction_pressure" in pressure_source


def test_piso_pressure_state_equation_uses_solver_nonorthogonal_controls():
    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    method = _method(solver, "solve_pressure_state_equation")
    parameters = [parameter.arg for parameter in method.args.args]
    source = ast.get_source_segment(PISO_SOLVER_SOURCE.read_text(), method)

    assert "nonorthogonal_max_iter" not in parameters
    assert "nonorthogonal_tol" not in parameters
    assert "max_iter = int(self.controls.pressure_nonorthogonal_max_iter)" in source


def test_piso_temporary_velocity_always_returns_momentum_matrix():
    solver = _class("CollocatedPisoSolver", PISO_SOLVER_SOURCE)
    method = _method(solver, "temporary_velocity")
    parameters = [parameter.arg for parameter in method.args.args]

    assert "return_matrix" not in parameters


def test_piso_rejects_legacy_current_momentum_explicit_correction():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["momentum_explicit_correction"] = "current"

    with pytest.raises(ValueError, match="current.*removed"):
        NSFVMPISOModel(options)


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


def test_piso_pressure_nonorthogonal_iterations_count_total_solves(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import NSFVMPISOModel
    import fealpy.fvm.collocated_piso_solver as piso_module

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
        piso_module,
        "apply_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity, face_normal, **kwargs: flux,
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
        "pressure_orthogonal_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0] * 10.0,
    )
    monkeypatch.setattr(
        model,
        "_pressure_nonorthogonal_cross_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0],
    )
    monkeypatch.setattr(
        model,
        "add_pressure_dirichlet_flux",
        lambda flux, pressure, coef, dirichlet_value, threshold: flux,
    )

    _, pressure, corrected_flux, diagnostics = model.pressure_correction_step(
        bm.zeros(2 * nc, dtype=model.cm.dtype),
        bm.zeros(nc, dtype=model.cm.dtype),
        bm.ones(2 * nc, dtype=model.cm.dtype),
        None,
        None,
        return_diagnostics=True,
    )

    assert solves == [1.0, 2.0]
    assert np.allclose(np.asarray(bm.to_numpy(pressure)), 2.0)
    assert np.allclose(np.asarray(bm.to_numpy(corrected_flux)), 19.0)
    assert diagnostics["pressure_nonorthogonal_iterations"] == 2


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
        piso_module,
        "apply_boundary_flux_constraint",
        lambda flux, boundary_faces, boundary_velocity, face_normal, **kwargs: flux,
    )
    monkeypatch.setattr(
        model,
        "divergence_from_flux",
        lambda flux: bm.ones(nc, dtype=model.cm.dtype) * flux[0],
    )
    monkeypatch.setattr(
        model,
        "_pressure_nonorthogonal_cross_flux",
        lambda pressure, coef: bm.ones(nf, dtype=model.cm.dtype) * pressure[0],
    )
    monkeypatch.setattr(
        model,
        "pressure_orthogonal_flux",
        lambda pressure, coef: bm.zeros(nf, dtype=model.cm.dtype),
    )
    monkeypatch.setattr(
        model,
        "add_pressure_dirichlet_flux",
        lambda flux, pressure, coef, dirichlet_value, threshold: flux,
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


def test_piso_pressure_nonorthogonal_max_iter_must_be_positive():
    from fealpy.fvm import NSFVMPISOModel

    options = _model_options(nx=2, ny=2, nt=1)
    options["pressure_nonorthogonal_max_iter"] = 0

    with pytest.raises(ValueError, match="pressure_nonorthogonal_max_iter"):
        NSFVMPISOModel(options)


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
    options.update(
        {
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
        }
    )
    model = NSFVMPISOModel(options)
    rows = []

    model.solve(corrector_callback=lambda **row: rows.append(row))

    assert len(rows) == model.controls.n_correctors
    assert model.corrector_diagnostics == rows
    assert rows[0]["step"] == 1
    assert rows[0]["corrector"] == 1
