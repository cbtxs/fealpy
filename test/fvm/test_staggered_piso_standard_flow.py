import ast
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / "fealpy" / "fvm" / "ns_fvm_staggered_piso_model.py"


def _source():
    return SOURCE.read_text()


def _tree():
    return ast.parse(_source())


def _method(name):
    for node in _tree().body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == name:
                return item
    raise AssertionError(f"missing function {name}")


def _calls_in_node(node):
    names = set()
    for call in ast.walk(node):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            names.add(call.func.id)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            names.add(call.func.attr)
    return names


def test_staggered_piso_uses_scaled_backward_euler_momentum_matrix():
    source = _source()

    assert "values=self.ucm," in source
    assert "values=self.vcm," in source
    assert "A = A * self.tau + mass_matrix" in source
    assert "A = self.tau * A + mass_matrix" in source
    assert "self.tau * (rhs - pressure_term)" in source


def test_staggered_piso_pressure_correction_is_increment_for_scaled_momentum():
    source = _source()
    solve = _method("solve")
    velocity_correction = _method("velocity_pressure_correction")
    velocity_correction_source = ast.get_source_segment(source, velocity_correction)

    assert "p1 = p0 + p_corr1" in source
    assert "p2 = p1 + p_corr2" in source
    assert "u_new = u - self.ucm / uap * ugrad_p[:, 0]" in velocity_correction_source
    assert "v_new = v - self.vcm / vap * vgrad_p[:, 1]" in velocity_correction_source
    assert "p1 = p0 + self.tau" not in source
    assert "p2 = p1 + self.tau" not in source
    assert "t + self.tau" in ast.get_source_segment(source, solve)


def test_staggered_piso_pressure_matrix_uses_momentum_response_not_edge_squared():
    source = _source()
    correct_pressure = _method("correct_pressure_compute")
    calls = _calls_in_node(correct_pressure)
    correct_source = ast.get_source_segment(source, correct_pressure)

    assert "pressure_response_on_pedge" in calls
    assert "coef=response" in correct_source
    assert "coef=self.tau*response" not in correct_source
    assert "p_edge2" not in source
    assert "p_edge2 / a_p_edge" not in source
