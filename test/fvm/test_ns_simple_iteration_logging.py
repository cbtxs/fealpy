from fealpy.fvm.simple_iteration_control import SimpleIterationControl


def test_simple_iteration_log_message_reports_required_iteration_data():
    message = SimpleIterationControl.iteration_log_message(
        simple_iteration=7,
        nonorthogonal_iterations=3,
        pressure_criterion=1.2e-4,
        pressure_relax=0.05,
        mass_residual=2.3e-3,
        pressure_correction=0.42,
    )

    assert "[SIMPLE 7]" in message
    assert "nonorthogonal iterations: 3" in message
    assert "pressure criterion: 1.20e-04" in message
    assert "pressure relax: 5.00e-02" in message


def test_simple_iteration_control_builds_default_tolerances():
    assert SimpleIterationControl.tolerances(1.0e-5, None, None) == (
        1.0e-5,
        1.0e-4,
    )
    assert SimpleIterationControl.tolerances(1.0e-5, 2.0e-6, 3.0e-6) == (
        2.0e-6,
        3.0e-6,
    )
