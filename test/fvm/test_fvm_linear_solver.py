from fealpy.backend import backend_manager as bm


def test_init_fvm_linear_solver_returns_existing_solver_object():
    from fealpy.fvm.fvm_linear_solver import init_fvm_linear_solver

    class ExistingSolver:
        def solve(self, matrix, rhs):
            return rhs

    existing = ExistingSolver()

    assert init_fvm_linear_solver(linear_solver=existing) is existing


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
