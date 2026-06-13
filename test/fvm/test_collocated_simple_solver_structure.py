import inspect

import pytest


def _cavity_solver(convection_coef=None, *, linear_solver=None):
    from fealpy.fvm import (
        BoundaryConditionData,
        CollocatedSimpleSolver,
        FVMLinearSolverConfig,
        LidDrivenCavityCase,
        SimpleSolverControls,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=2, ny=2)
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
        boundary_conditions=BoundaryConditionData(case.dirichlet_velocity).to_pde_boundary(mesh),
        controls=SimpleSolverControls(space_degree=0),
        log_level="ERROR",
        **solver_kwargs,
    )


def test_collocated_simple_solver_is_public_algorithm_core():
    from fealpy.fvm import CollocatedSimpleSolver, NSFVMSimpleModel
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    assert issubclass(CollocatedSimpleSolver, CollocatedNSFVMOperators)
    assert issubclass(NSFVMSimpleModel, CollocatedSimpleSolver)


def test_collocated_simple_solver_interface_uses_discrete_inputs():
    from fealpy.fvm import CollocatedSimpleSolver

    parameters = inspect.signature(CollocatedSimpleSolver).parameters
    for name in ("mesh", "diffusion_coef", "convection_coef", "source", "boundary_conditions"):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty

    assert "controls" in parameters
    assert "pde" not in parameters
    assert "options" not in parameters


def test_collocated_simple_solver_default_pressure_relaxation_is_conservative():
    from fealpy.fvm import CollocatedSimpleSolver

    solve_parameters = inspect.signature(CollocatedSimpleSolver.solve).parameters
    assert solve_parameters["relax"].default == 0.3


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

    uh, vh, ph = solver.solve(max_iter=1, tol=1.0e-3)

    assert not hasattr(solver, "pde")
    assert uh.shape == (solver.NC,)
    assert vh.shape == (solver.NC,)
    assert ph.shape == (solver.NC,)
    assert len(solver.residuals) == 1
    assert solver.residuals[0]["pressure_relax"] == 0.3


def test_collocated_simple_solver_accepts_zero_convection_for_stokes_limit():
    solver = _cavity_solver(convection_coef=0.0)

    uh, vh, ph = solver.solve(max_iter=1, tol=1.0e-3)

    assert uh.shape == (solver.NC,)
    assert vh.shape == (solver.NC,)
    assert ph.shape == (solver.NC,)
    assert len(solver.residuals) == 1


def test_collocated_simple_solver_accepts_string_linear_solver_choice():
    solver = _cavity_solver(convection_coef=0.0, linear_solver="scipy")

    assert solver.linear_solver.config.solver == "scipy"


def test_stokes_simple_model_uses_collocated_simple_algorithm_core():
    from fealpy.fvm import CollocatedSimpleSolver, StokesFVMSimpleModel

    assert issubclass(StokesFVMSimpleModel, CollocatedSimpleSolver)


def test_stokes_simple_model_accepts_momentum_relaxation_option():
    from fealpy.fvm import StokesFVMSimpleModel

    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "momentum_equation_relaxation": 0.6,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.controls.momentum_equation_relaxation == 0.6
