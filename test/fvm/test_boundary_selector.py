import numpy as np
import pytest

from fealpy.backend import backend_manager as bm


def test_incompressible_pde_protocols_define_canonical_boundary_data():
    from fealpy.model.navier_stokes import NavierStokesPDEDataProtocol
    from fealpy.model.stokes import StokesPDEDataProtocol

    for protocol in (StokesPDEDataProtocol, NavierStokesPDEDataProtocol):
        assert callable(protocol.dirichlet_velocity)
        assert callable(protocol.dirichlet_pressure)


def test_boundary_face_flag_always_passes_full_point_array():
    from fealpy.fvm.fvm_geometry import boundary_face_flag

    points = bm.array(
        [
            [0.0, 0.25],
            [0.5, 0.25],
            [1.0, 0.75],
        ],
        dtype=bm.float64,
    )

    seen = {}

    def selector(x):
        seen["ndim"] = x.ndim
        return x < 0.75 if x.ndim == 1 else x[:, 0] < 0.75

    flag = boundary_face_flag(points, selector)

    assert seen["ndim"] == 2
    np.testing.assert_array_equal(np.asarray(flag), [True, True, False])


def test_boundary_face_flag_rejects_wrong_shape():
    from fealpy.fvm.fvm_geometry import boundary_face_flag

    points = bm.zeros((3, 2), dtype=bm.float64)
    with pytest.raises(ValueError, match="one entry per boundary face"):
        boundary_face_flag(points, lambda p: bm.array([True, False]))


def test_rhie_chow_dirichlet_pressure_uses_public_boundary_selector():
    from fealpy.fvm.collocated_face_velocity_reconstruct import RhieChowInterpolation
    from fealpy.mesh import TriangleMesh

    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)

    def dirichlet_pressure(points):
        return points[:, 0] + points[:, 1]

    rhie_chow = RhieChowInterpolation(
        mesh,
        dirichlet_pressure=dirichlet_pressure,
        dirichlet_pressure_threshold=lambda p: bm.ones(p.shape[0], dtype=bm.bool),
    )
    pressure = bm.zeros(mesh.number_of_cells(), dtype=bm.float64)
    gradient_difference = rhie_chow.pressure_gradient_difference(pressure)

    assert gradient_difference.shape == (
        mesh.number_of_faces(),
        mesh.geo_dimension(),
    )
