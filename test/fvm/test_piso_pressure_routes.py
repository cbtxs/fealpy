import ast
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_piso_model.py"
EXAMPLE = ROOT / "example" / "fvm" / "ns_fvm_piso_example.py"


def _tree(path=SOURCE):
    return ast.parse(path.read_text())


def _class(name):
    for node in _tree().body:
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


def test_only_standard_incremental_piso_route_remains():
    source = SOURCE.read_text()

    assert "PRESSURE_ROUTES" not in source
    assert "pressure_route" not in source
    assert "--pressure-route" not in source
    assert '"legacy"' not in source
    assert '"face-flux"' not in source
    assert '"physical-pressure"' not in source
    assert '"consistent"' not in source


def test_model_exposes_piso_components_and_no_route_dispatch():
    model = _class("NSFVMPISOModel")

    for name in [
        "set_pde",
        "set_mesh",
        "set_space",
        "temporary_velocity",
        "solve_pressure_correction",
        "face_flux",
        "divergence_from_flux",
        "pressure_correction_flux",
        "rhie_chow_face_velocity",
        "velocity_pressure_correction",
        "apply_pressure_rate_correction",
        "solve",
        "compute_error",
        "plot",
    ]:
        _method(model, name)

    solve = _method(model, "solve")
    calls = _calls_in_node(solve)
    assert "solve_pressure_correction" in calls
    assert "pressure_correction_flux" in calls
    assert "apply_pressure_rate_correction" in calls
    assert "apply_pressure_correction" not in calls
    assert "solve_physical_pressure" not in calls
    assert "build_hbya" not in calls


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
