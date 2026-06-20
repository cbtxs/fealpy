import inspect

import pytest


def _cavity_solver(convection_coef=None, *, controls=None, linear_solver=None):
    from fealpy.fvm import (
        BoundaryConditionData,
        CollocatedSimpleSolver,
        FVMLinearSolverConfig,
        LidDrivenCavityCase,
        SimpleSolverControls,
    )

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=2, ny=2)
    solver_kwargs = {}
    if linear_solver is None:
        solver_kwargs["linear_solver_config"] = FVMLinearSolverConfig(solver="scipy")
    else:
        solver_kwargs["linear_solver"] = linear_solver

    return CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=case.mu,
        convection_coef=case.rho if convection_coef is None else convection_coef,
        source=case.source,
        boundary_conditions=BoundaryConditionData(case.dirichlet_velocity).to_pde_boundary(mesh),
        controls=controls or SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
        ),
        log_level="ERROR",
        **solver_kwargs,
    )


def test_collocated_simple_solver_is_public_algorithm_core():
    from fealpy.fvm import CollocatedSimpleSolver, NSFVMSimpleModel
    from fealpy.fvm.collocated_ns_components import CollocatedNSFVMComponents

    assert issubclass(CollocatedSimpleSolver, CollocatedNSFVMComponents)
    assert issubclass(NSFVMSimpleModel, CollocatedSimpleSolver)


def test_collocated_simple_solver_interface_uses_discrete_inputs():
    from fealpy.fvm import CollocatedSimpleSolver

    parameters = inspect.signature(CollocatedSimpleSolver).parameters
    for name in ("mesh", "diffusion_coef", "convection_coef", "source", "boundary_conditions"):
        assert name in parameters
        assert parameters[name].default is inspect.Parameter.empty

    assert "controls" in parameters
    assert "pde" not in parameters
    assert "options" not in parameters


def test_collocated_simple_solver_default_pressure_relaxation_is_conservative():
    from fealpy.fvm import CollocatedSimpleSolver

    solve_parameters = inspect.signature(CollocatedSimpleSolver.solve).parameters
    assert solve_parameters["relax"].default == 0.3


def test_momentum_strategy_branch_is_hidden_from_algorithm_entrypoints():
    from fealpy.fvm import CollocatedPisoSolver, CollocatedSimpleSolver

    for method in (
        CollocatedSimpleSolver.temporary_velocity,
        CollocatedPisoSolver.temporary_velocity,
    ):
        source = inspect.getsource(method)
        assert "momentum_solve_strategy" not in source
        assert "component_temporary_velocity" not in source


def test_cell_vector_dof_conversion_is_component_major_on_torch_backend():
    pytest.importorskip("torch")
    from fealpy.backend import backend_manager as bm

    solver = _cavity_solver()
    try:
        bm.set_backend("pytorch")
        cell_vector = bm.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        dofs = solver.cell_vector_to_dofs(cell_vector)

        assert bm.to_numpy(dofs).tolist() == [1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]
        assert bm.to_numpy(solver.dofs_to_cell_vector(dofs)).tolist() == [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    finally:
        bm.set_backend("numpy")


def test_collocated_simple_solver_runs_without_model_adapter():
    solver = _cavity_solver()

    uh, vh, ph = solver.solve(max_iter=1, tol=1.0e-3)

    assert uh.shape == (solver.NC,)
    assert vh.shape == (solver.NC,)
    assert ph.shape == (solver.NC,)
    assert len(solver.residuals) == 1
    assert "pressure_relax" not in solver.residuals[0]


def test_collocated_simple_solver_accepts_zero_convection_for_stokes_limit():
    solver = _cavity_solver(convection_coef=0.0)

    uh, vh, ph = solver.solve(max_iter=1, tol=1.0e-3)

    assert uh.shape == (solver.NC,)
    assert vh.shape == (solver.NC,)
    assert ph.shape == (solver.NC,)
    assert len(solver.residuals) == 1


def test_collocated_simple_solver_accepts_string_linear_solver_choice():
    solver = _cavity_solver(convection_coef=0.0, linear_solver="scipy")

    assert solver.linear_solver.config.solver == "scipy"


def test_collocated_simple_solver_uses_recommended_default_linear_policy():
    from fealpy.fvm import BoundaryConditionData, CollocatedSimpleSolver, LidDrivenCavityCase

    case = LidDrivenCavityCase(re=10.0)
    mesh = case.init_mesh["uniform_quad"](nx=2, ny=2)
    solver = CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=case.mu,
        convection_coef=case.rho,
        source=case.source,
        boundary_conditions=BoundaryConditionData(case.dirichlet_velocity).to_pde_boundary(mesh),
        log_level="ERROR",
    )

    assert solver.momentum_linear_solver == "scipy_bicgstab"
    assert solver.pressure_nullspace_linear_solver == "petsc_gmres_hypre"
    assert not hasattr(solver.linear_solver.config, "momentum_solver")


def test_simple_temporary_velocity_reuses_steady_source_rhs(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fvm import SimpleSolverControls

    bm.set_backend("numpy")

    class ZeroLinearSolver:
        def solve(self, matrix, rhs, *, solver=None):
            return bm.zeros(rhs.shape[0], dtype=rhs.dtype)

    solver = _cavity_solver(
        convection_coef=0.0,
        controls=SimpleSolverControls(
            space_degree=0,
            pressure_constraint="gauge",
            momentum_solve_strategy="vector",
            momentum_nonorthogonal_max_iter=0,
            pressure_nonorthogonal_max_iter=0,
        ),
        linear_solver=ZeroLinearSolver(),
    )
    call_count = 0

    def counted_source_vector(source):
        nonlocal call_count
        call_count += 1
        return bm.ones(solver.GD * solver.NC, dtype=solver.cm.dtype)

    monkeypatch.setattr(solver, "momentum_source_vector", counted_source_vector)

    p = bm.zeros(solver.NC, dtype=solver.cm.dtype)
    uf = bm.zeros((solver.mesh.number_of_faces(), solver.GD), dtype=solver.cm.dtype)
    u0 = bm.zeros(solver.GD * solver.NC, dtype=solver.cm.dtype)

    solver.temporary_velocity(p, uf, u0)
    solver.temporary_velocity(p, uf, u0)

    assert call_count == 1


def test_simple_pressure_gauge_matrix_does_not_use_bilinear_assembly(monkeypatch):
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.ones(solver.mesh.number_of_faces(), dtype=solver.cm.dtype)

    def fail_assembly(self):
        raise AssertionError("pressure gauge matrix should use cached FVM structure")

    monkeypatch.setattr(BilinearForm, "assembly", fail_assembly)

    matrix = solver.pressure_gauge_matrix(coef)

    assert matrix.shape == (solver.NC + 1, solver.NC + 1)


def test_simple_pressure_gauge_matrix_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm, BlockForm
    from fealpy.sparse import COOTensor
    from fealpy.fvm import ScalarDiffusionIntegrator

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.mesh.number_of_faces())
    matrix = solver.pressure_gauge_matrix(coef)

    reference = BilinearForm(solver.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.fvm_geometry)
    ).assembly()
    gauge_index = bm.stack(
        [
            bm.zeros(solver.NC, dtype=bm.int32),
            bm.arange(solver.NC, dtype=bm.int32),
        ],
        axis=0,
    )
    gauge = COOTensor(gauge_index, solver.cm, spshape=(1, solver.NC))
    reference = BlockForm([[reference, gauge.T], [gauge, None]])
    reference = reference.assembly_sparse_matrix(format="csr")

    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_scalar_diffusion_matrix_assembler_matches_bilinear_reference():
    import numpy as np
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import BilinearForm
    from fealpy.fvm.scalar_diffusion_integrator import (
        ScalarDiffusionIntegrator,
        ScalarDiffusionMatrixAssembler,
    )

    solver = _cavity_solver(convection_coef=0.0)
    coef = bm.linspace(0.3, 1.4, solver.mesh.number_of_faces())
    matrix = ScalarDiffusionMatrixAssembler(
        solver.space,
        geometry=solver.fvm_geometry,
    ).assembly(coef)

    reference = BilinearForm(solver.space).add_integrator(
        ScalarDiffusionIntegrator(q=2, coef=coef, geometry=solver.fvm_geometry)
    ).assembly()
    diff = matrix.to_scipy() - reference.to_scipy()

    np.testing.assert_allclose(diff.data, 0.0, atol=1.0e-13)


def test_stokes_simple_model_uses_collocated_simple_algorithm_core():
    from fealpy.fvm import CollocatedSimpleSolver, StokesFVMSimpleModel

    assert issubclass(StokesFVMSimpleModel, CollocatedSimpleSolver)


def test_stokes_simple_model_accepts_momentum_relaxation_option():
    from fealpy.fvm import StokesFVMSimpleModel

    model = StokesFVMSimpleModel(
        {
            "pde": 1,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "momentum_equation_relaxation": 0.6,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    assert model.controls.momentum_equation_relaxation == 0.6
