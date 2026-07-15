import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.fvm import (
    CollocatedSimpleSolver,
    FVMLinearSolverConfig,
    PDEBoundaryConditions,
    SimpleSolverControls,
    StokesFVMSimpleModel,
)


def _solve_stokes_model(*, momentum_relaxation=0.7, pressure_relaxation=0.3):
    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "linear_solver": "scipy",
            "momentum_equation_relaxation": momentum_relaxation,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    velocity, pressure = model.solve(
        max_iter=400,
        tol=1.0e-8,
        relax=pressure_relaxation,
    )
    return model, velocity, pressure


def test_stokes_simple_reaches_momentum_and_mass_fixed_point():
    bm.set_backend("numpy")
    model, velocity, pressure = _solve_stokes_model()
    residual = model.residuals[-1]
    errors = model.compute_error()

    assert model.converged is True
    assert model.termination_reason == "fixed_point_residuals"
    assert residual["mass"] < 1.0e-7
    assert residual["momentum_residual_relative"] < 1.0e-7
    assert velocity.shape == (model.NC, model.GD)
    assert pressure.shape == (model.NC,)
    assert all(np.isfinite(float(error)) for error in errors)
    assert max(float(error) for error in errors) < 1.0


def test_stokes_fixed_point_is_independent_of_relaxation_parameters():
    bm.set_backend("numpy")
    first, velocity_first, pressure_first = _solve_stokes_model(
        momentum_relaxation=0.5,
        pressure_relaxation=0.2,
    )
    second, velocity_second, pressure_second = _solve_stokes_model(
        momentum_relaxation=0.8,
        pressure_relaxation=0.4,
    )
    pressure_first = pressure_first - bm.mean(pressure_first)
    pressure_second = pressure_second - bm.mean(pressure_second)

    assert first.converged is True
    assert second.converged is True
    np.testing.assert_allclose(
        bm.to_numpy(velocity_first),
        bm.to_numpy(velocity_second),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        bm.to_numpy(pressure_first),
        bm.to_numpy(pressure_second),
        rtol=0.0,
        atol=1.0e-5,
    )


def test_collocated_stokes_runs_on_three_dimensional_hexahedra():
    from fealpy.model.navier_stokes.exp0012 import Exp0012

    bm.set_backend("numpy")
    pde = Exp0012({"mu": 1.0})
    mesh = pde.init_mesh["uniform_hex"](nx=2, ny=2, nz=2)
    viscosity = float(bm.to_numpy(pde.viscosity()))

    @cartesian
    def source(points):
        return -viscosity * pde.lap_velocity(points) + pde.grad_pressure(points)

    solver = CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=viscosity,
        convection_coef=0.0,
        source=source,
        boundary_conditions=PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=pde.dirichlet_velocity,
        ),
        controls=SimpleSolverControls(
            pressure_constraint="nullspace",
            rhie_chow_velocity_scheme="second_order_reconstructed",
        ),
        linear_solver_config=FVMLinearSolverConfig(solver="scipy"),
        log_level="ERROR",
    )

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-6, relax=0.3)

    assert velocity.shape == (mesh.number_of_cells(), 3)
    assert solver.face_velocity.shape == (mesh.number_of_faces(), 3)
    assert solver.face_flux.shape == (mesh.number_of_faces(),)
    assert pressure.shape == (mesh.number_of_cells(),)
    assert bool(bm.all(bm.isfinite(velocity)))
    assert bool(bm.all(bm.isfinite(pressure)))
