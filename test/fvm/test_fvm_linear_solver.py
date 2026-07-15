from fealpy.backend import backend_manager as bm


def test_init_fvm_linear_solver_returns_existing_solver_object():
    from fealpy.fvm.fvm_linear_solver import init_fvm_linear_solver

    class ExistingSolver:
        def solve(self, matrix, rhs, *, solver=None):
            return rhs

    existing = ExistingSolver()

    assert init_fvm_linear_solver(linear_solver=existing) is existing


def test_collocated_operator_passes_explicit_solver_name():
    from fealpy.fvm.collocated_ns_components import CollocatedNSFVMComponents

    calls = []

    class RecordingSolver:
        def solve(self, matrix, rhs, *, solver=None):
            calls.append(solver)
            return rhs

    operator = object.__new__(CollocatedNSFVMComponents)
    operator.linear_solver = RecordingSolver()

    assert operator.solve_linear_system("A", "b", solver="scipy_bicgstab") == "b"
    assert calls == ["scipy_bicgstab"]


def test_init_fvm_linear_solver_accepts_dict_config():
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, init_fvm_linear_solver

    solver = init_fvm_linear_solver(
        linear_solver_config={"backend": "numpy", "device": "cpu", "solver": "scipy"}
    )

    assert isinstance(solver, FVMLinearSolver)
    assert solver.config.backend == "numpy"
    assert solver.config.device == "cpu"
    assert solver.config.solver == "scipy"


def test_fvm_linear_solver_selection_is_explicit():
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig

    cpu_solver = FVMLinearSolver(FVMLinearSolverConfig(backend="numpy", device="cpu"))
    cuda_solver = FVMLinearSolver(
        FVMLinearSolverConfig(backend="pytorch", device="cuda:0")
    )

    assert cpu_solver.select_solver(None) == "mumps"
    assert cpu_solver.select_solver("scipy") == "scipy"
    assert cuda_solver.select_solver(None) == "cupy"


def test_fvm_linear_solver_fealpy_cpu_route_uses_spsolve(monkeypatch):
    import fealpy.fvm.fvm_linear_solver as linear_solver_module
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver

    class Matrix:
        def __init__(self, name):
            self.name = name

        def copy(self):
            return Matrix("copied")

    calls = []

    def fake_spsolve(matrix, rhs, solver_name):
        calls.append((matrix.name, tuple(bm.to_numpy(rhs)), solver_name))
        return bm.array([7.0, 11.0])

    monkeypatch.setattr(linear_solver_module, "spsolve", fake_spsolve)

    solver = FVMLinearSolver()
    rhs = bm.array([1.0, 2.0])

    scipy_result = solver.solve_with_fealpy_solver(Matrix("original"), rhs, "scipy")
    mumps_result = solver.solve_with_fealpy_solver(Matrix("original"), rhs, "mumps")

    assert calls == [
        ("copied", (1.0, 2.0), "scipy"),
        ("original", (1.0, 2.0), "mumps"),
    ]
    assert bm.to_numpy(scipy_result).tolist() == [7.0, 11.0]
    assert bm.to_numpy(mumps_result).tolist() == [7.0, 11.0]


def test_fvm_linear_solver_solve_uses_named_routes(monkeypatch):
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig

    calls = []
    solver = FVMLinearSolver(FVMLinearSolverConfig(solver="scipy"))

    def fake_cpu_route(matrix, rhs, solver_name):
        calls.append(("cpu", solver_name))
        return rhs

    def fake_cupy_route(matrix, rhs, matrix_type):
        calls.append(("cupy", matrix_type))
        return rhs

    monkeypatch.setattr(solver, "solve_with_fealpy_solver", fake_cpu_route)
    monkeypatch.setattr(solver, "solve_with_cupy", fake_cupy_route)

    rhs = bm.array([1.0])
    assert solver.solve("A", rhs) is rhs
    assert solver.solve("A", rhs, solver="cupy", matrix_type="L") is rhs
    assert calls == [("cpu", "scipy"), ("cupy", "L")]


def test_constant_nullspace_solver_reuses_cached_petsc_solver(monkeypatch):
    import fealpy.fvm.fvm_linear_solver as linear_solver_module
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig

    calls = []

    class CachedSolver:
        def __init__(self, matrix, cell_measure, *, solver_name, rtol, atol, max_it):
            calls.append(("init", matrix, tuple(bm.to_numpy(cell_measure)), solver_name))

        def matches(self, matrix, *, solver_name, rtol, atol, max_it):
            calls.append(("matches", matrix, solver_name))
            return True

        def solve(self, matrix, rhs, cell_measure):
            calls.append(("solve", matrix, tuple(bm.to_numpy(rhs))))
            return rhs

        def destroy(self):
            calls.append(("destroy",))

    monkeypatch.setattr(
        linear_solver_module,
        "CachedPetscConstantNullspaceSolver",
        CachedSolver,
    )

    solver = FVMLinearSolver(
        FVMLinearSolverConfig(constant_nullspace_solver="petsc_cg_jacobi")
    )
    rhs = bm.array([1.0, -1.0])
    cell_measure = bm.array([0.5, 0.5])

    assert solver.solve_constant_nullspace("A", rhs, cell_measure) is rhs
    assert solver.solve_constant_nullspace("A", rhs, cell_measure) is rhs
    assert calls == [
        ("init", "A", (0.5, 0.5), "petsc_cg_jacobi"),
        ("solve", "A", (1.0, -1.0)),
        ("matches", "A", "petsc_cg_jacobi"),
        ("solve", "A", (1.0, -1.0)),
    ]


def test_constant_nullspace_solver_can_disable_cache(monkeypatch):
    from fealpy.fvm.fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig

    calls = []
    solver = FVMLinearSolver(
        FVMLinearSolverConfig(
            constant_nullspace_solver="petsc_cg_jacobi",
            constant_nullspace_cache=False,
        )
    )

    def one_shot(matrix, rhs, cell_measure, *, solver_name, rtol, atol, max_it):
        calls.append((matrix, tuple(bm.to_numpy(rhs)), solver_name))
        return rhs

    monkeypatch.setattr(solver, "solve_constant_nullspace_with_petsc", one_shot)

    rhs = bm.array([1.0, -1.0])
    cell_measure = bm.array([0.5, 0.5])

    assert solver.solve_constant_nullspace("A", rhs, cell_measure) is rhs
    assert calls == [("A", (1.0, -1.0), "petsc_cg_jacobi")]
