import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry, NSFVMPISOModel


def _model_options(nx=2, ny=2, nt=1):
    return {
        "pde": 3,
        "nx": nx,
        "ny": ny,
        "nt": nt,
        "duration": (0, 1),
        "space_degree": 0,
        "pbar_log": False,
        "log_level": "WARNING",
    }


def test_divergence_from_flux_reuses_fvm_geometry_scatter(monkeypatch):
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    phi = bm.linspace(0.2, 1.4, model.mesh.number_of_faces())
    expected = FVMGeometry(model.mesh).scatter_face_flux_to_cells(phi)

    calls = []
    original_scatter = FVMGeometry.scatter_face_flux_to_cells

    def counted_scatter(self, face_flux):
        calls.append(np.asarray(face_flux).copy())
        return original_scatter(self, face_flux)

    monkeypatch.setattr(FVMGeometry, "scatter_face_flux_to_cells", counted_scatter)

    divergence = model.divergence_from_flux(phi)

    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.asarray(phi), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(
        np.asarray(divergence),
        np.asarray(expected),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
