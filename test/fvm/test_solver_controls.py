from fealpy.fvm import PisoSolverControls, SimpleSolverControls


def test_simple_solver_controls_default_to_current_recommended_policy():
    controls = SimpleSolverControls()

    assert controls.pressure_constraint == "nullspace"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"


def test_piso_solver_controls_default_to_current_recommended_policy():
    controls = PisoSolverControls()

    assert controls.pressure_constraint == "nullspace"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"
