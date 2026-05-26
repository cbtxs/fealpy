import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = ROOT / "fealpy" / "fvm" / "ns_fvm_staggered_piso_model.py"
EXAMPLE_SOURCE = ROOT / "example" / "fvm" / "ns_fvm_staggered_piso_example.py"
INIT_SOURCE = ROOT / "fealpy" / "fvm" / "__init__.py"


def _source(path=MODEL_SOURCE):
    return path.read_text()


def _tree(path=MODEL_SOURCE):
    return ast.parse(_source(path))


def _class(name):
    for node in _tree().body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _method(cls, name):
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing method {name}")


def test_staggered_piso_solver_is_computational_model_not_script():
    source = _source()
    tree = _tree()
    cls = _class("NSFVMStaggeredPISOModel")

    base_names = {
        base.id for base in cls.bases if isinstance(base, ast.Name)
    }
    assert "ComputationalModel" in base_names
    assert "PDEModelManager(\"navier_stokes\").get_example(3)" not in source
    assert "nx = 40" not in source

    module_statements = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    ]
    assert module_statements == []


def test_staggered_piso_model_keeps_standard_substeps_as_methods():
    cls = _class("NSFVMStaggeredPISOModel")
    method_names = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}

    assert {
        "set_pde",
        "set_mesh",
        "initial_solution",
        "compute_temporary_velocity_u",
        "compute_temporary_velocity_v",
        "pressure_response_on_pedge",
        "correct_pressure_compute",
        "solve",
        "compute_error",
        "plot",
    }.issubset(method_names)

    solve_source = ast.get_source_segment(_source(), _method(cls, "solve"))
    pressure_source = ast.get_source_segment(
        _source(), _method(cls, "correct_pressure_compute")
    )
    assert "t + self.tau" in solve_source
    assert "self.pressure_response_on_pedge" in pressure_source
    assert "coef=response" in pressure_source


def test_staggered_piso_example_exports_driver_for_model():
    example = _source(EXAMPLE_SOURCE)
    init_source = _source(INIT_SOURCE)

    assert "from fealpy.fvm import NSFVMStaggeredPISOModel" in example
    assert "model = NSFVMStaggeredPISOModel(options)" in example
    assert "model.solve()" in example
    assert "from .ns_fvm_staggered_piso_model import NSFVMStaggeredPISOModel" in init_source
