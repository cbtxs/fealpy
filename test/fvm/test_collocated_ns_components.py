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


def test_momentum_nonorthogonal_rhs_uses_fast_rhs_assembler(monkeypatch):
    import fealpy.fvm.collocated_ns_components as ns_components

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    velocity = bm.zeros((model.NC, model.GD), dtype=bm.float64)

    class ForbiddenLinearForm:
        def __init__(self, *args, **kwargs):
            raise AssertionError("momentum_nonorthogonal_rhs should use fast RHS assembler")

    monkeypatch.setattr(ns_components, "LinearForm", ForbiddenLinearForm)

    rhs = model.momentum_nonorthogonal_rhs(velocity)

    assert rhs.shape == (model.GD * model.NC,)


def test_pressure_nonorthogonal_cross_flux_uses_explicit_interpolation(monkeypatch):
    import fealpy.fvm.collocated_ns_components as ns_components

    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())
    pressure = bm.zeros(model.NC, dtype=model.cm.dtype)
    response = bm.ones(model.mesh.number_of_faces(), dtype=model.cm.dtype)
    seen = []

    def record_face_gradient(mesh, cell_gradient, *, geometry, interpolation_method, **kwargs):
        seen.append(interpolation_method)
        return bm.zeros((mesh.number_of_faces(), mesh.geo_dimension()), dtype=model.cm.dtype)

    monkeypatch.setattr(ns_components, "reconstruct_face_gradient", record_face_gradient)

    model.pressure_nonorthogonal_cross_flux(
        pressure,
        response,
        interpolation_method="linear",
    )

    assert seen == ["linear"]


def test_collocated_discretization_reuses_one_fvm_geometry_for_gradients():
    bm.set_backend("numpy")
    model = NSFVMPISOModel(_model_options())

    assert model.pressure_gradient.fvm_geometry is model.fvm_geometry
    assert model.velocity_gradient.fvm_geometry is model.fvm_geometry


def test_solver_setup_and_coefficient_validation_live_outside_operator_mixin():
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, init_fvm_linear_solver
    from fealpy.fvm.solver_controls import nonnegative_scalar, positive_scalar

    solver = init_fvm_linear_solver(linear_solver="scipy", reference=bm.zeros(1))

    assert isinstance(solver, FVMLinearSolver)
    assert solver.config.solver == "scipy"
    assert positive_scalar(bm.array(2.0), "mu") == 2.0
    assert nonnegative_scalar(bm.array(0.0), "rho") == 0.0
