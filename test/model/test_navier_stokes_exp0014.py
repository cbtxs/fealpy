import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager
from fealpy.model.navier_stokes.exp0014 import Exp0014


def test_exp0014_reproduces_quadratic_pipe_balance():
    bm.set_backend("numpy")
    pde = Exp0014()
    points = bm.array(
        [
            [0.0, 0.0, 1.5],
            [0.25, 0.0, 1.0],
        ],
        dtype=bm.float64,
    )

    np.testing.assert_allclose(
        bm.to_numpy(pde.velocity(points)),
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.75]],
    )
    np.testing.assert_allclose(bm.to_numpy(pde.pressure(points)), 0.0)
    np.testing.assert_allclose(
        bm.to_numpy(pde.grad_velocity(points)),
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
        ],
    )
    np.testing.assert_allclose(
        bm.to_numpy(pde.lap_velocity(points)),
        [[0.0, 0.0, -16.0], [0.0, 0.0, -16.0]],
    )
    np.testing.assert_allclose(
        bm.to_numpy(pde.source(points)),
        [[0.0, 0.0, 16.0], [0.0, 0.0, 16.0]],
    )
    np.testing.assert_allclose(bm.to_numpy(pde.div_velocity(points)), 0.0)
    np.testing.assert_allclose(bm.to_numpy(pde.convective(points)), 0.0)
    assert pde.exact_flow_rate() == pytest.approx(np.pi / 8.0)


def test_exp0014_quartic_sidecar_has_exact_source():
    bm.set_backend("numpy")
    pde = Exp0014({"profile_power": 2})
    points = bm.array(
        [
            [0.0, 0.0, 1.5],
            [0.25, 0.0, 1.0],
        ],
        dtype=bm.float64,
    )

    np.testing.assert_allclose(
        bm.to_numpy(pde.velocity(points))[:, 2],
        [1.0, 0.75**2],
    )
    np.testing.assert_allclose(
        bm.to_numpy(pde.source(points)),
        [[0.0, 0.0, 32.0], [0.0, 0.0, 16.0]],
    )
    assert pde.exact_flow_rate() == pytest.approx(np.pi / 12.0)


def test_exp0014_classifies_pipe_end_and_wall_points():
    bm.set_backend("numpy")
    pde = Exp0014({"lc": 0.1})
    points = bm.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.5, 0.0, 1.5],
            [0.0, 0.0, 1.5],
        ],
        dtype=bm.float64,
    )

    np.testing.assert_array_equal(
        bm.to_numpy(pde.is_inlet_boundary(points)),
        [True, False, False, False],
    )
    np.testing.assert_array_equal(
        bm.to_numpy(pde.is_outlet_boundary(points)),
        [False, True, False, False],
    )
    np.testing.assert_array_equal(
        bm.to_numpy(pde.is_wall_boundary(points)),
        [False, False, True, False],
    )


def test_exp0014_rejects_unsupported_profile_power():
    with pytest.raises(ValueError, match="profile_power must be 1 or 2"):
        Exp0014({"profile_power": 3})


def test_navier_stokes_example_14_is_registered():
    pde = PDEModelManager("navier_stokes").get_example(14)

    assert isinstance(pde, Exp0014)
