import pytest

from fealpy.fvm import PisoSolverControls, SimpleSolverControls


def test_simple_solver_controls_default_to_current_recommended_policy():
    controls = SimpleSolverControls()

    assert controls.pressure_constraint == "nullspace"
    assert controls.pressure_response_scheme == "simple"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"


def test_simple_solver_controls_accept_simplec_pressure_response_scheme():
    controls = SimpleSolverControls(pressure_response_scheme="simplec")

    assert controls.pressure_response_scheme == "simplec"


def test_simple_solver_controls_reject_unknown_pressure_response_scheme():
    with pytest.raises(ValueError, match="pressure_response_scheme"):
        SimpleSolverControls(pressure_response_scheme="unknown")


def test_piso_solver_controls_default_to_current_recommended_policy():
    controls = PisoSolverControls()

    assert controls.pressure_constraint == "nullspace"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"
