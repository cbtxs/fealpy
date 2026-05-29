import math

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve as direct_spsolve


def test_relative_inf_norm_handles_zero_reference():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 2, "nx": 2, "ny": 2, "log_level": "ERROR"})

    delta = bm.array([0.0, 2.0, -3.0])
    reference = bm.zeros(3)

    assert float(model._relative_inf_norm(delta, reference)) == 3.0


def test_pressure_delta_without_mean_removes_constant_shift():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 2, "nx": 2, "ny": 2, "log_level": "ERROR"})
    p_old = bm.array([1.0, 2.0, 3.0, 4.0])
    p_new = p_old + 7.0

    delta = model._pressure_delta_without_mean(p_new, p_old)

    assert float(bm.max(bm.abs(delta))) < 1.0e-14


def test_rc_rhs_is_repeatable_for_same_pressure(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model
    from fealpy.fvm.rhie_chow import RhieChowCoupledOperator

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    model = rc_model.NSFVMRCModel({"pde": 2, "nx": 4, "ny": 4, "log_level": "ERROR"})
    uf = bm.zeros((model.mesh.number_of_faces(), 2))
    _, _, _, _, ap = model.assembly_base_system(uf)
    rc_operator = RhieChowCoupledOperator(model.mesh)
    pressure = bm.linspace(0.0, 1.0, model.NC)

    rhs1 = model._rc_rhs(rc_operator, ap, pressure)
    rhs2 = model._rc_rhs(rc_operator, ap, pressure)

    assert float(bm.max(bm.abs(rhs1 - rhs2))) < 1.0e-14


def test_solve_rhie_chow_can_return_diagnostics(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    model = rc_model.NSFVMRCModel({"pde": 2, "nx": 8, "ny": 8, "log_level": "ERROR"})

    uh, vh, ph, diagnostics = model.solve_rhie_chow(
        max_iter=3,
        min_iter=2,
        return_diagnostics=True,
    )

    assert len(diagnostics) >= 2
    assert diagnostics[-1]["iteration"] == len(diagnostics)
    assert "res_face_velocity" in diagnostics[-1]
    assert "res_pressure" in diagnostics[-1]
    assert "res_rc_rhs" in diagnostics[-1]
    assert "omega" in diagnostics[-1]
    assert model.rhie_chow_diagnostics is diagnostics


def test_default_solve_stops_after_all_three_residuals_are_small(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    model = rc_model.NSFVMRCModel({"pde": 2, "nx": 20, "ny": 20, "log_level": "ERROR"})
    *_, diagnostics = model.solve_rhie_chow(return_diagnostics=True)

    final = diagnostics[-1]
    assert final["iteration"] >= 2
    assert final["res_face_velocity"] < 1.0e-7
    assert final["res_pressure"] < 1.0e-7
    assert final["res_rc_rhs"] < 1.0e-7


def test_relax_pressure_picard_returns_raw_pressure():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 2, "nx": 2, "ny": 2, "log_level": "ERROR"})
    p_old = bm.array([0.0, 1.0, 2.0, 3.0])
    p_raw = bm.array([1.0, 2.0, 3.0, 4.0])
    state = {"previous_delta": None, "omega": 1.0}

    p_used, new_state = model._relax_pressure(
        p_old,
        p_raw,
        state,
        relaxation="picard",
        omega=1.0,
        omega_min=0.2,
        omega_max=1.2,
    )

    assert float(bm.max(bm.abs(p_used - p_raw))) < 1.0e-14
    assert new_state["omega"] == 1.0


def test_relax_pressure_fixed_uses_fixed_omega():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 2, "nx": 2, "ny": 2, "log_level": "ERROR"})
    p_old = bm.array([0.0, 1.0, 2.0, 3.0])
    p_raw = bm.array([2.0, 3.0, 4.0, 5.0])
    state = {"previous_delta": None, "omega": 0.5}

    p_used, new_state = model._relax_pressure(
        p_old,
        p_raw,
        state,
        relaxation="fixed",
        omega=0.5,
        omega_min=0.2,
        omega_max=1.2,
    )

    expected = p_old + 0.5 * (p_raw - p_old)
    assert float(bm.max(bm.abs(p_used - expected))) < 1.0e-14
    assert new_state["omega"] == 0.5


def test_solve_accepts_fixed_relaxation(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    picard_model = rc_model.NSFVMRCModel(
        {"pde": 2, "nx": 8, "ny": 8, "log_level": "ERROR"}
    )
    fixed_model = rc_model.NSFVMRCModel(
        {"pde": 2, "nx": 8, "ny": 8, "log_level": "ERROR"}
    )

    _, _, ph_picard, _ = picard_model.solve_rhie_chow(
        max_iter=1,
        return_diagnostics=True,
    )
    _, _, ph_fixed, diagnostics = fixed_model.solve_rhie_chow(
        max_iter=1,
        relaxation="fixed",
        omega=0.5,
        return_diagnostics=True,
    )

    assert diagnostics
    assert all(abs(row["omega"] - 0.5) < 1.0e-14 for row in diagnostics)
    assert float(bm.max(bm.abs(ph_fixed - 0.5 * ph_picard))) < 1.0e-12


def test_relax_pressure_aitken_clamps_omega():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 2, "nx": 2, "ny": 2, "log_level": "ERROR"})
    p_old = bm.zeros(4)
    p_raw = bm.array([10.0, -10.0, 10.0, -10.0])
    state = {
        "previous_delta": bm.array([1.0, -1.0, 1.0, -1.0]),
        "omega": 1.0,
    }

    p_used, new_state = model._relax_pressure(
        p_old,
        p_raw,
        state,
        relaxation="aitken",
        omega=1.0,
        omega_min=0.2,
        omega_max=1.2,
    )

    assert 0.2 <= new_state["omega"] <= 1.2
    expected = p_old + new_state["omega"] * (p_raw - p_old)
    assert float(bm.max(bm.abs(p_used - expected))) < 1.0e-14
    assert new_state["omega"] != 1.0


def test_solve_accepts_aitken_relaxation(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    model = rc_model.NSFVMRCModel({"pde": 2, "nx": 8, "ny": 8, "log_level": "ERROR"})
    *_, diagnostics = model.solve_rhie_chow(
        max_iter=8,
        relaxation="aitken",
        return_diagnostics=True,
    )

    assert diagnostics
    assert all(0.2 <= row["omega"] <= 1.2 for row in diagnostics)
    assert all(math.isfinite(row["res_pressure"]) for row in diagnostics)


def test_ns_rhie_chow_default_iterations_converge_pressure(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_rc_model as rc_model

    monkeypatch.setattr(
        rc_model,
        "spsolve",
        lambda A, b, solver="mumps": direct_spsolve(A, b, "scipy"),
    )

    pressure_errors = []
    for nx in (20, 40):
        model = rc_model.NSFVMRCModel(
            {"pde": 2, "nx": nx, "ny": nx, "log_level": "ERROR"}
        )
        model.solve_rhie_chow()
        pressure_errors.append(float(model.compute_error()[2]))

    pressure_order = math.log(pressure_errors[0] / pressure_errors[1], 2)

    assert pressure_order > 1.8
