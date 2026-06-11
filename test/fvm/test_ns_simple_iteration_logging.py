from fealpy.fvm.simple_residual import (
    simple_iteration_log_message,
    simple_pressure_update_step,
    simple_tolerances,
)


def test_simple_iteration_log_message_reports_required_iteration_data():
    message = simple_iteration_log_message(
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


def test_simple_residual_builds_default_tolerances():
    assert simple_tolerances(1.0e-5, None, None) == (
        1.0e-5,
        1.0e-4,
    )
    assert simple_tolerances(1.0e-5, 2.0e-6, 3.0e-6) == (
        2.0e-6,
        3.0e-6,
    )


def test_simple_iteration_mass_uses_pressure_corrected_face_velocity(monkeypatch):
    from fealpy.backend import backend_manager as bm
    import fealpy.fvm.simple_residual as simple_residual

    bm.set_backend("numpy")
    raw_uf = object()
    corrected_uf = object()

    def fake_mass_residual(mesh, uf):
        return 2.0 if uf is raw_uf else 1.0e-8

    monkeypatch.setattr(simple_residual, "collocated_mass_residual", fake_mass_residual)
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value: 0.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p: 0.0)

    _, residual = simple_pressure_update_step(
        [],
        None,
        raw_uf,
        bm.ones(1),
        bm.zeros(1),
        pressure_relax=0.3,
        nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=1,
        stopping_face_velocity=corrected_uf,
    )

    assert residual["mass"] == 1.0e-8
    assert residual["mass_before_pressure_correction"] == 2.0
    assert residual["pressure_relax_action"] == "fixed"
