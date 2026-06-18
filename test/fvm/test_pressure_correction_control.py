import pytest

from fealpy.fvm.simple_residual import pressure_correction_converged


def test_pressure_correction_converged_uses_mass_and_pressure_correction():
    residual = {
        "mass": 1.0e-7,
        "pressure_update": 2.0e-7,
        "pressure_correction": 2.0e-3,
    }

    assert not pressure_correction_converged(
        residual, tol_mass=1.0e-6, tol_pressure_correction=1.0e-6
    )
    residual["pressure_correction"] = 2.0e-7
    assert pressure_correction_converged(
        residual, tol_mass=1.0e-6, tol_pressure_correction=1.0e-6
    )
    assert not pressure_correction_converged(
        residual, tol_mass=1.0e-8, tol_pressure_correction=1.0e-6
    )


def test_pressure_correction_converged_can_use_absolute_pressure_correction():
    residual = {
        "mass": 1.0e-7,
        "pressure_update": 1.0e-8,
        "pressure_correction": 2.0e-3,
        "pressure_correction_relative": 1.0e-4,
    }

    assert not pressure_correction_converged(
        residual,
        tol_mass=1.0e-6,
        tol_pressure_correction=1.0e-3,
    )

    residual["pressure_correction"] = 5.0e-4
    assert pressure_correction_converged(
        residual,
        tol_mass=1.0e-6,
        tol_pressure_correction=1.0e-3,
    )


def test_pressure_correction_converged_rejects_old_pressure_update_name():
    residual = {
        "mass": 1.0e-7,
        "pressure_update": 1.0e-8,
        "pressure_correction": 1.0e-8,
    }

    with pytest.raises(TypeError):
        pressure_correction_converged(
            residual,
            tol_mass=1.0e-6,
            tol_pressure_update=1.0e-6,
        )
