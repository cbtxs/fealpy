import ast
from pathlib import Path


FVM_DIR = Path(__file__).resolve().parents[2] / "fealpy" / "fvm"
MODEL_FILES = [
    FVM_DIR / "ns_fvm_simple_model.py",
    FVM_DIR / "ns_fvm_staggered_simple_model.py",
]


def _method(path, name):
    tree = ast.parse(path.read_text())
    for class_node in [node for node in tree.body if isinstance(node, ast.ClassDef)]:
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"missing method {name} in {path.name}")


def _attribute_chain(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _attribute_chain(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def test_ns_simple_plot_residual_has_no_scalar_residual_fallback():
    for path in MODEL_FILES:
        method = _method(path, "plot_residual")
        source = ast.get_source_segment(path.read_text(), method)

        assert "isinstance(self.residuals[0], dict)" not in source

        for call in ast.walk(method):
            if not isinstance(call, ast.Call):
                continue
            func_name = _attribute_chain(call.func)
            if func_name != "plt.semilogy":
                continue
            assert call.args, f"empty semilogy call in {path.name}"
            assert _attribute_chain(call.args[0]) != "self.residuals"


def test_ns_simple_plot_residual_uses_fixed_residual_keys():
    for path in MODEL_FILES:
        method = _method(path, "plot_residual")
        constants = {
            node.value
            for node in ast.walk(method)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

        assert "mass" in constants
        assert "pressure_update" in constants


def test_ns_simple_solve_has_no_commented_adaptive_relaxation_branch():
    for path in MODEL_FILES:
        source = path.read_text()

        assert "# if err <" not in source
        assert "# elif err <" not in source
        assert "#     p += 0.3*p_corr" not in source
        assert "#     p += 0.05*p_corr" not in source


def test_ns_simple_models_have_no_debug_prints_or_dead_alternatives():
    dead_fragments = [
        "# grad_p =",
        "# grad_u =",
        "# print",
        "# p = self.pde.pressure",
        "# uerr =",
        "# verr =",
        "# perr =",
        "# uerror =",
        "# verror =",
        "# perror =",
        "p_edge2",
        "NeumannBC",
    ]

    for path in MODEL_FILES:
        source = path.read_text()
        tree = ast.parse(source)

        for call in ast.walk(tree):
            if isinstance(call, ast.Call):
                assert _attribute_chain(call.func) != "print"

        for fragment in dead_fragments:
            assert fragment not in source


def test_collocated_ns_simple_pressure_correct_has_no_unused_pressure_argument():
    method = _method(FVM_DIR / "ns_fvm_simple_model.py", "pressure_correct")

    assert [arg.arg for arg in method.args.args] == ["self", "ap", "uf"]
