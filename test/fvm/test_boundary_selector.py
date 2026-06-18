import numpy as np
import pytest

from fealpy.backend import backend_manager as bm


def test_boundary_face_flag_supports_axis_and_point_thresholds():
    from fealpy.fvm.fvm_geometry import boundary_face_flag

    points = bm.array(
        [
            [0.0, 0.25],
            [0.5, 0.25],
            [1.0, 0.75],
        ],
        dtype=bm.float64,
    )

    np.testing.assert_array_equal(
        np.asarray(boundary_face_flag(points, lambda x: x < 0.75)),
        np.array([True, True, False]),
    )
    np.testing.assert_array_equal(
        np.asarray(boundary_face_flag(points, lambda p: p[:, 1] > 0.5)),
        np.array([False, False, True]),
    )


def test_boundary_face_flag_rejects_wrong_shape():
    from fealpy.fvm.fvm_geometry import boundary_face_flag

    points = bm.zeros((3, 2), dtype=bm.float64)
    with pytest.raises(ValueError, match="one entry per boundary face"):
        boundary_face_flag(points, lambda p: bm.array([True, False]))


def test_rhie_chow_pressure_dirichlet_uses_public_boundary_selector():
    from fealpy.fvm.rhie_chow import RhieChowInterpolation
    from fealpy.mesh import TriangleMesh

    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)

    def pressure_dirichlet(points):
        return points[:, 0] + points[:, 1]

    rhie_chow = RhieChowInterpolation(
        mesh,
        pressure_dirichlet=pressure_dirichlet,
        pressure_dirichlet_threshold=lambda p: bm.ones(p.shape[0], dtype=bm.bool),
    )
    pressure = bm.zeros(mesh.number_of_cells(), dtype=bm.float64)
    gradient_difference = rhie_chow.GradientDifference(pressure)

    assert gradient_difference.shape == (
        mesh.number_of_faces(),
        mesh.geo_dimension(),
    )
