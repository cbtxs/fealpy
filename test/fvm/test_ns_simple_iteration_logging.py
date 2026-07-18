from fealpy.fvm.simple_residual import (
    simple_iteration_residual,
    simple_tolerances,
)
from fealpy.fvm.solver_diagnostics import simple_iteration_log_message


def test_simple_iteration_log_message_reports_required_iteration_data():
    message = simple_iteration_log_message(
        simple_iteration=7,
        nonorthogonal_iterations=3,
        pressure_criterion=1.2e-4,
        momentum_residual=3.4e-5,
        mass_residual=2.3e-3,
        pressure_correction=0.42,
    )

    assert "[SIMPLE 7]" in message
    assert "nonorthogonal iterations: 3" in message
    assert "pressure criterion: 1.20e-04" in message
    assert "pressure relax" not in message
    assert "momentum residual: 3.40e-05" in message
    assert "mass residual: 2.30e-03" in message
    assert "pressure correction L2: 4.20e-01" in message


def test_simple_residual_builds_fixed_point_tolerances():
    assert simple_tolerances(1.0e-5, None, None) == (
        1.0e-5,
        1.0e-5,
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

    def fake_mass_metrics(mesh, uf, geometry=None):
        return {
            "relative_l1": 2.0 if uf is raw_uf else 1.0e-8,
            "relative_l2": 2.0 if uf is raw_uf else 1.0e-8,
            "divergence_l2": 4.0 if uf is raw_uf else 3.0e-8,
            "absolute_linf": 3.0 if uf is raw_uf else 2.0e-8,
        }

    monkeypatch.setattr(simple_residual, "collocated_mass_metrics", fake_mass_metrics)
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 0.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 0.0)

    residual = simple_iteration_residual(
        None,
        raw_uf,
        bm.ones(1),
        bm.zeros(1),
        nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=1,
        stopping_face_velocity=corrected_uf,
        record_mass_before_pressure_correction=True,
    )

    assert residual["mass"] == 1.0e-8
    assert residual["mass_imbalance_linf"] == 2.0e-8
    assert residual["mass_before_pressure_correction"] == 2.0
    assert "pressure_update" not in residual
    assert "pressure_relax" not in residual
    assert "pressure_relax_action" not in residual
    assert "pressure_relax_reduced" not in residual


def test_simple_iteration_mass_skips_raw_face_velocity_by_default(monkeypatch):
    from fealpy.backend import backend_manager as bm
    import fealpy.fvm.simple_residual as simple_residual

    bm.set_backend("numpy")
    calls = []
    raw_uf = object()
    corrected_uf = object()

    def fake_mass_metrics(mesh, uf, geometry=None):
        calls.append(uf)
        return {
            "relative_l1": 1.0e-8,
            "relative_l2": 1.0e-8,
            "divergence_l2": 3.0e-8,
            "absolute_linf": 2.0e-8,
        }

    monkeypatch.setattr(simple_residual, "collocated_mass_metrics", fake_mass_metrics)
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 0.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 0.0)

    residual = simple_iteration_residual(
        None,
        raw_uf,
        bm.ones(1),
        bm.zeros(1),
        nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=1,
        stopping_face_velocity=corrected_uf,
    )

    assert calls == [corrected_uf]
    assert residual["mass"] == 1.0e-8
    assert residual["mass_imbalance_linf"] == 2.0e-8
    assert "mass_before_pressure_correction" not in residual


def test_simple_iteration_pressure_criterion_uses_pressure_correction_l2(monkeypatch):
    from fealpy.backend import backend_manager as bm
    import fealpy.fvm.simple_residual as simple_residual

    bm.set_backend("numpy")
    monkeypatch.setattr(
        simple_residual,
        "collocated_mass_metrics",
            lambda mesh, uf, geometry=None: {
                "relative_l1": 0.0,
                "relative_l2": 0.0,
                "divergence_l2": 0.0,
                "absolute_linf": 0.0,
        },
    )
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 2.5)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 1.0e-8)

    residual = simple_iteration_residual(
        None,
        object(),
        bm.ones(1),
        bm.zeros(1),
        nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=1,
        stopping_face_velocity=object(),
    )

    assert residual["pressure_correction"] == 2.5
    assert residual["pressure_criterion"] == 2.5
    assert "pressure_correction_relative" not in residual


def test_simple_iteration_can_record_relative_pressure_correction(monkeypatch):
    from fealpy.backend import backend_manager as bm
    import fealpy.fvm.simple_residual as simple_residual

    bm.set_backend("numpy")
    monkeypatch.setattr(
        simple_residual,
        "collocated_mass_metrics",
            lambda mesh, uf, geometry=None: {
                "relative_l1": 0.0,
                "relative_l2": 0.0,
                "divergence_l2": 0.0,
                "absolute_linf": 0.0,
        },
    )
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 2.5)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 1.0e-8)

    residual = simple_iteration_residual(
        None,
        object(),
        bm.ones(1),
        bm.zeros(1),
        nonorthogonal_iterations=1,
        momentum_nonorthogonal_iterations=1,
        stopping_face_velocity=object(),
        record_pressure_correction_relative=True,
    )

    assert residual["pressure_correction_relative"] == 1.0e-8
