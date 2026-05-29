from fealpy.fvm.pressure_correction_control import (
    PressureRelaxationConfig,
    PressureRelaxationController,
    pressure_correction_converged,
)


def test_pressure_relaxation_reduces_on_severe_growth():
    controller = PressureRelaxationController(
        initial=0.4,
        config=PressureRelaxationConfig(
            min_value=0.01,
            max_value=0.4,
            severe_growth_factor=2.0,
            cooldown_steps=2,
        ),
    )
    residuals = [
        {"pressure_correction": 1.0, "pressure_update": 0.4, "mass": 1.0},
        {"pressure_correction": 3.0, "pressure_update": 1.2, "mass": 1.0},
    ]

    value, action = controller.update(residuals)

    assert value == 0.2
    assert action == "reduce"
    assert controller.cooldown == 2


def test_pressure_correction_converged_uses_mass_and_pressure_update():
    residual = {"mass": 1.0e-7, "pressure_update": 2.0e-7}

    assert pressure_correction_converged(
        residual, tol_mass=1.0e-6, tol_pressure_update=1.0e-6
    )
    assert not pressure_correction_converged(
        residual, tol_mass=1.0e-8, tol_pressure_update=1.0e-6
    )
