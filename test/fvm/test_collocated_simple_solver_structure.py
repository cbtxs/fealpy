import ast
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FVM_DIR = ROOT / "fealpy" / "fvm"
SIMPLE_MODEL_SOURCE = FVM_DIR / "ns_fvm_simple_model.py"
SIMPLE_SOLVER_SOURCE = FVM_DIR / "collocated_simple_solver.py"
SIMPLE_ITERATION_CONTROL_SOURCE = FVM_DIR / "simple_iteration_control.py"


def _class_methods(path, class_name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
            }
    raise AssertionError(f"missing class {class_name} in {path}")


def test_collocated_simple_solver_is_exported_algorithm_owner():
    from fealpy.fvm import CollocatedSimpleSolver, NSFVMSimpleModel
    from fealpy.fvm.collocated_ns_fvm_utils import CollocatedNSFVMOperators

    assert SIMPLE_SOLVER_SOURCE.exists()
    assert issubclass(CollocatedSimpleSolver, CollocatedNSFVMOperators)
    assert issubclass(NSFVMSimpleModel, CollocatedSimpleSolver)


def test_ns_fvm_simple_model_keeps_only_manufactured_case_adapter_methods():
    model_methods = _class_methods(SIMPLE_MODEL_SOURCE, "NSFVMSimpleModel")
    solver_methods = _class_methods(SIMPLE_SOLVER_SOURCE, "CollocatedSimpleSolver")

    algorithm_methods = {
        "temporary_velocity",
        "pressure_correct",
        "pressure_correction_flux",
        "correct_face_velocity_with_pressure_correction",
        "solve",
    }

    non_algorithm_methods = {"_init_mesh", "compute_error", "plot", "plot_residual"}

    assert algorithm_methods <= solver_methods
    assert solver_methods.isdisjoint(non_algorithm_methods)
    assert model_methods.isdisjoint(algorithm_methods)
    assert non_algorithm_methods <= model_methods


def test_collocated_simple_solver_interface_uses_discrete_inputs():
    from fealpy.fvm import CollocatedSimpleSolver

    signature = inspect.signature(CollocatedSimpleSolver)
    parameters = signature.parameters

    for name in (
        "mesh",
        "diffusion_coef",
        "convection_coef",
        "source",
        "boundary_conditions",
    ):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty

    assert "controls" in parameters
    assert "pde" not in parameters
    assert "options" not in parameters


def test_collocated_simple_solver_uses_conservative_default_pressure_relaxation():
    from fealpy.fvm import CollocatedSimpleSolver

    signature = inspect.signature(CollocatedSimpleSolver.solve)

    assert signature.parameters["relax"].default == 0.03


def test_fvm_top_level_exports_do_not_include_internal_simple_helpers():
    import fealpy.fvm as fvm

    assert not hasattr(fvm, "SimpleIterationControl")
    assert not hasattr(fvm, "write_fealpy_cylinder_openfoam_case")


def test_collocated_simple_solver_can_run_without_computational_model_adapter():
    from fealpy.fvm import (
        CollocatedSimpleSolver,
        FVMLinearSolverConfig,
        LidDrivenCavityCase,
        SimpleBoundaryConditions,
        SimpleSolverControls,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=2, ny=2)
    solver = CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=case.mu,
        convection_coef=case.rho,
        source=case.source,
        boundary_conditions=SimpleBoundaryConditions(case.dirichlet_velocity),
        controls=SimpleSolverControls(space_degree=0),
        linear_solver_config=FVMLinearSolverConfig(solver="scipy"),
        log_level="ERROR",
    )

    uh, vh, ph = solver.solve(max_iter=1, tol=1.0e-3)

    assert not hasattr(solver, "pde")
    assert uh.shape == (solver.NC,)
    assert vh.shape == (solver.NC,)
    assert ph.shape == (solver.NC,)
    assert len(solver.residuals) == 1
    assert solver.residuals[0]["pressure_relax"] == 0.03


def test_simple_iteration_control_is_separated_from_solver_core():
    from fealpy.fvm.simple_iteration_control import SimpleIterationControl

    solver_methods = _class_methods(SIMPLE_SOLVER_SOURCE, "CollocatedSimpleSolver")

    iteration_control_methods = {
        "_simple_residual",
        "_simple_iteration_log_message",
        "_log_simple_iteration",
        "_simple_tolerances",
        "_pressure_relaxation_controller",
        "_pressure_update_step",
    }

    assert SIMPLE_ITERATION_CONTROL_SOURCE.exists()
    assert inspect.isclass(SimpleIterationControl)
    assert solver_methods.isdisjoint(iteration_control_methods)


def test_collocated_simple_solver_avoids_low_value_glue_helpers():
    solver_methods = _class_methods(SIMPLE_SOLVER_SOURCE, "CollocatedSimpleSolver")

    low_value_helpers = {
        "_initial_simple_fields",
        "_store_solution",
        "_face_interpolation_option",
        "_pressure_correct_with_response",
    }

    assert solver_methods.isdisjoint(low_value_helpers)
