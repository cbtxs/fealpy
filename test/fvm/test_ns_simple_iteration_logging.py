from fealpy.fvm.ns_fvm_simple_model import NSFVMSimpleModel


def test_simple_iteration_log_message_reports_required_iteration_data():
    model = object.__new__(NSFVMSimpleModel)

    message = model._simple_iteration_log_message(
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
