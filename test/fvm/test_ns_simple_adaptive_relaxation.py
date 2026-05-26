from fealpy.fvm.ns_fvm_simple_model import NSFVMSimpleModel


def _model():
    return object.__new__(NSFVMSimpleModel)


def _adapt(
    residuals,
    current_relax=0.2,
    deterioration_count=0,
    small_update_count=0,
    cooldown=0,
    **kwargs,
):
    options = {
        "relax_min": 1.0e-4,
        "relax_max": 0.3,
        "growth_factor": 1.05,
        "severe_growth_factor": 2.0,
        "patience": 3,
        "reduction_factor": 0.5,
        "increase_patience": 3,
        "increase_factor": 1.25,
        "small_update": 1.0e-3,
        "cooldown_steps": 3,
        "residual_key": "pressure_correction",
    }
    options.update(kwargs)
    return _model()._adapt_pressure_relaxation(
        residuals,
        current_relax=current_relax,
        deterioration_count=deterioration_count,
        small_update_count=small_update_count,
        cooldown=cooldown,
        **options,
    )


def test_pressure_relaxation_reduces_after_sustained_pressure_correction_growth():
    residuals = [
        {"pressure_correction": 1.0},
        {"pressure_correction": 1.06},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=0.2,
        deterioration_count=0,
    )
    assert relax == 0.2
    assert bad_count == 1
    assert small_count == 0
    assert cooldown == 0
    assert action == "keep"

    residuals.append({"pressure_correction": 1.13})
    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=relax,
        deterioration_count=bad_count,
    )
    assert relax == 0.2
    assert bad_count == 2
    assert small_count == 0
    assert cooldown == 0
    assert action == "keep"

    residuals.append({"pressure_correction": 1.20})
    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=relax,
        deterioration_count=bad_count,
    )
    assert relax == 0.1
    assert bad_count == 0
    assert small_count == 0
    assert cooldown == 3
    assert action == "reduce"


def test_pressure_relaxation_reduces_immediately_on_severe_pressure_correction_growth():
    residuals = [
        {"pressure_correction": 1.0},
        {"pressure_correction": 2.2},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=0.2,
    )

    assert relax == 0.1
    assert bad_count == 0
    assert small_count == 0
    assert cooldown == 3
    assert action == "reduce"


def test_pressure_relaxation_cooldown_prevents_repeated_adjustment():
    residuals = [
        {"pressure_correction": 1.0, "pressure_update": 5.0e-4, "mass": 1.0},
        {"pressure_correction": 1.0, "pressure_update": 5.0e-4, "mass": 0.9},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=1.0e-4,
        small_update_count=2,
        cooldown=2,
    )

    assert relax == 1.0e-4
    assert bad_count == 0
    assert small_count == 2
    assert cooldown == 1
    assert action == "cooldown"


def test_pressure_relaxation_increases_when_update_is_too_small():
    residuals = [
        {"mass": 2.4e-2, "pressure_correction": 36.7, "pressure_update": 6.3e-4},
        {"mass": 2.39e-2, "pressure_correction": 36.0, "pressure_update": 6.2e-4},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=1.0e-4,
        small_update_count=2,
    )

    assert relax == 1.25e-4
    assert bad_count == 0
    assert small_count == 0
    assert cooldown == 3
    assert action == "increase"


def test_pressure_relaxation_increase_respects_initial_relaxation_cap():
    residuals = [
        {"mass": 2.4e-2, "pressure_correction": 36.7, "pressure_update": 6.3e-4},
        {"mass": 2.39e-2, "pressure_correction": 36.0, "pressure_update": 6.2e-4},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=0.29,
        small_update_count=2,
    )

    assert relax == 0.3
    assert bad_count == 0
    assert small_count == 0
    assert cooldown == 3
    assert action == "increase"


def test_pressure_relaxation_does_not_increase_when_mass_gets_worse():
    residuals = [
        {"mass": 2.4e-2, "pressure_correction": 36.7, "pressure_update": 6.3e-4},
        {"mass": 2.7e-2, "pressure_correction": 36.0, "pressure_update": 6.2e-4},
    ]

    relax, bad_count, small_count, cooldown, action = _adapt(
        residuals,
        current_relax=1.0e-4,
        small_update_count=2,
    )

    assert relax == 1.0e-4
    assert bad_count == 0
    assert small_count == 0
    assert cooldown == 0
    assert action == "keep"
