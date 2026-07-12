"""Linear-system solve boundary for finite-volume models."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.sparse import COOTensor, CSRTensor


@dataclass
class FVMLinearSolverConfig:
    """Linear solver policy for finite-volume models.

    The config describes the array backend used by the FVM code and the sparse
    linear solver used at solve time.  It intentionally does not encode any
    equation-specific mathematics.
    """

    backend: str = "numpy"
    device: str = "cpu"
    solver: str = "auto"
    default_matrix_type: str = "G"
    constant_nullspace_solver: Optional[str] = None
    constant_nullspace_rtol: float = 1.0e-10
    constant_nullspace_atol: float = 1.0e-14
    constant_nullspace_max_it: int = 1000
    constant_nullspace_cache: bool = True
    iterative_rtol: float = 1.0e-8
    iterative_atol: float = 0.0
    iterative_max_it: int = 1000
    iterative_true_residual_tol: float = 1.0e-7


class IterationCounter:
    """Count callback calls from third-party iterative solvers."""

    def __init__(self):
        self.niter = 0

    def __call__(self, value=None):
        self.niter += 1


class CachedPetscConstantNullspaceSolver:
    """Reusable PETSc KSP for a singular matrix with constant nullspace.

    The matrix pattern, constant nullspace, and KSP/PC objects are kept across
    RHS updates.  Matrix values may change as long as the sparse pattern is unchanged.
    """

    def __init__(
        self,
        matrix,
        cell_measure,
        *,
        solver_name: str,
        rtol: float,
        atol: float,
        max_it: int,
    ):
        from petsc4py import PETSc

        parts = solver_name.split("_", 2)
        if len(parts) != 3:
            raise ValueError(
                "PETSc constant-nullspace solver must use the form "
                "'petsc_<ksp>_<pc>', for example 'petsc_cg_gamg'."
            )
        _, ksp_type, pc_type = parts
        ksp_type = "bcgs" if ksp_type == "bicgstab" else ksp_type

        mat = self.matrix_to_scipy(matrix)
        self.PETSc = PETSc
        self.solver_name = solver_name
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_it = int(max_it)
        self.shape = mat.shape
        self.indptr = np.asarray(mat.indptr, dtype=PETSc.IntType)
        self.indices = np.asarray(mat.indices, dtype=PETSc.IntType)
        self.data = np.asarray(mat.data, dtype=PETSc.ScalarType).copy()
        self.measure = np.asarray(bm.to_numpy(cell_measure), dtype=PETSc.ScalarType)
        self.comm = PETSc.COMM_SELF
        self.petsc_mat = PETSc.Mat().createAIJ(
            size=mat.shape,
            csr=(
                self.indptr,
                self.indices,
                self.data,
            ),
            comm=self.comm,
        )
        self.petsc_mat.assemble()
        self.nullspace = PETSc.NullSpace().create(constant=True, comm=self.comm)
        self.petsc_mat.setNullSpace(self.nullspace)
        self.petsc_mat.setNearNullSpace(self.nullspace)
        self.ksp = PETSc.KSP().create(comm=self.comm)
        self.ksp.setOperators(self.petsc_mat)
        self.ksp.setType(ksp_type)
        self.ksp.getPC().setType(pc_type)
        self.ksp.setTolerances(rtol=self.rtol, atol=self.atol, max_it=self.max_it)
        self.ksp.setFromOptions()
        self.last_rhs_sum_before = None
        self.last_rhs_sum_after = None
        self.last_iterations = None
        self.last_reason = None
        self.last_relative_residual = None

    def matches(self, matrix, *, solver_name: str, rtol: float, atol: float, max_it: int):
        """Return whether this cached KSP can be reused for the next solve."""
        if (
            solver_name != self.solver_name
            or float(rtol) != self.rtol
            or float(atol) != self.atol
            or int(max_it) != self.max_it
        ):
            return False

        mat = self.matrix_to_scipy(matrix)
        return (
            mat.shape == self.shape
            and np.array_equal(mat.indptr, self.indptr)
            and np.array_equal(mat.indices, self.indices)
        )

    def solve(self, matrix, rhs, cell_measure):
        """Solve one compatible RHS and return the zero-mean solution."""
        PETSc = self.PETSc
        mat = self.matrix_to_scipy(matrix)
        if (
            mat.shape != self.shape
            or not np.array_equal(mat.indptr, self.indptr)
            or not np.array_equal(mat.indices, self.indices)
        ):
            raise ValueError(
                "cached PETSc constant-nullspace solver received a new matrix pattern."
            )

        measure = np.asarray(bm.to_numpy(cell_measure), dtype=PETSc.ScalarType)
        if not np.array_equal(measure, self.measure):
            raise ValueError(
                "cached PETSc constant-nullspace solver received changed cell measures."
            )

        data = np.asarray(mat.data, dtype=PETSc.ScalarType)
        if not np.array_equal(data, self.data):
            self.data = data.copy()
            self.petsc_mat.setValuesCSR(self.indptr, self.indices, self.data)
            self.petsc_mat.assemble()
            self.ksp.setOperators(self.petsc_mat)

        rhs_np = np.asarray(bm.to_numpy(rhs), dtype=PETSc.ScalarType)
        rhs_sum_before = np.sum(rhs_np)
        rhs_np = rhs_np - rhs_sum_before * self.measure / np.sum(self.measure)
        rhs_vec = PETSc.Vec().createWithArray(rhs_np, comm=self.comm)
        self.nullspace.remove(rhs_vec)
        projected_rhs = np.array(rhs_vec.getArray(readonly=True), copy=True)
        rhs_sum_after = np.sum(projected_rhs)
        solution_vec = rhs_vec.duplicate()
        solution_vec.set(0.0)

        self.ksp.solve(rhs_vec, solution_vec)
        reason = self.ksp.getConvergedReason()
        iterations = self.ksp.getIterationNumber()
        solution = np.array(solution_vec.getArray(readonly=True), copy=True)
        solution -= np.sum(self.measure * solution) / np.sum(self.measure)
        relative_residual = np.linalg.norm(mat @ solution - projected_rhs) / max(
            np.linalg.norm(projected_rhs),
            1.0,
        )

        solution_vec.destroy()
        rhs_vec.destroy()
        if reason <= 0:
            raise RuntimeError(
                f"PETSc constant-nullspace solve failed: reason={reason}."
            )
        if relative_residual > max(1.0e-7, 100.0 * self.rtol):
            raise RuntimeError(
                "PETSc constant-nullspace solve failed true residual check: "
                f"{relative_residual:.3e}."
            )
        self.last_rhs_sum_before = float(np.real(rhs_sum_before))
        self.last_rhs_sum_after = float(np.real(rhs_sum_after))
        self.last_iterations = int(iterations)
        self.last_reason = int(reason)
        self.last_relative_residual = float(relative_residual)
        return bm.array(solution, dtype=rhs.dtype)

    def destroy(self):
        self.ksp.destroy()
        self.nullspace.destroy()
        self.petsc_mat.destroy()

    @staticmethod
    def matrix_to_scipy(matrix):
        if isinstance(matrix, CSRTensor):
            return matrix.to_scipy().tocsr()
        if isinstance(matrix, COOTensor):
            return matrix.tocsr().to_scipy().tocsr()
        if hasattr(matrix, "tocsr"):
            return matrix.tocsr()
        raise TypeError(f"Unsupported sparse matrix type: {type(matrix)!r}")


class FVMLinearSolver:
    """Centralized sparse linear-system solver entry for FVM models.

    The purpose of this class is to isolate backend/device conversion and
    direct-solver dispatch from PDE models.  Models should hand it an assembled
    FEALPy sparse matrix and RHS.  Equation-specific policies and constraints
    remain in the calling algorithm layer.

    The current CUDA route is a compatibility path based on PyTorch CUDA
    tensors and CuPy sparse solvers.  It is not yet a cached or matrix-free GPU
    high-performance solver.
    """

    def __init__(self, config: Optional[FVMLinearSolverConfig] = None):
        self.config = config or FVMLinearSolverConfig()
        self._constant_nullspace_solver_cache = None
        self.last_constant_nullspace_rhs_sum_before = None
        self.last_constant_nullspace_rhs_sum_after = None
        self.last_constant_nullspace_iterations = None
        self.last_constant_nullspace_reason = None
        self.last_constant_nullspace_relative_residual = None
        self.last_iterative_solver = None
        self.last_iterative_iterations = None
        self.last_iterative_info = None
        self.last_iterative_relative_residual = None

    def __del__(self):
        self.clear_constant_nullspace_cache()

    def clear_constant_nullspace_cache(self):
        """Destroy cached PETSc constant-nullspace objects, if any."""
        cache = getattr(self, "_constant_nullspace_solver_cache", None)
        if cache is not None:
            cache.destroy()
            self._constant_nullspace_solver_cache = None

    def solve(
        self,
        A,
        b,
        matrix_type: Optional[str] = None,
        solver: Optional[str] = None,
    ):
        solver_name = self.select_solver(solver)
        matrix_type = matrix_type or self.config.default_matrix_type

        if solver_name == "cupy":
            return self.solve_with_cupy(A, b, matrix_type)
        if solver_name.startswith("scipy_"):
            return self.solve_with_scipy_iterative(A, b, solver_name)
        if solver_name.startswith("petsc_"):
            return self.solve_with_petsc_iterative(A, b, solver_name)
        return self.solve_with_fealpy_solver(A, b, solver_name)

    def select_solver(self, solver: Optional[str]) -> str:
        """Return the concrete sparse solver selected for this solve."""
        if solver is not None:
            return solver
        if self.config.solver != "auto":
            return self.config.solver
        if self.config.backend == "pytorch" and self.config.device.startswith("cuda"):
            return "cupy"
        return "mumps"

    def solve_with_fealpy_solver(self, A, b, solver_name: str):
        """Solve through FEALPy's sparse direct-solver boundary."""
        if solver_name == "scipy":
            return spsolve(A.copy(), b, solver_name)
        return spsolve(A, b, solver_name)

    def solve_with_scipy_iterative(self, A, b, solver_name: str):
        """Solve with SciPy Krylov methods selected by ``solver_name``."""
        from scipy.sparse.linalg import bicgstab, cg, gmres, lgmres, minres

        matrix = self.matrix_to_scipy(A).tocsr()
        rhs = np.asarray(bm.to_numpy(b), dtype=float)
        counter = IterationCounter()
        rtol = self.config.iterative_rtol
        atol = self.config.iterative_atol
        max_it = self.config.iterative_max_it
        if solver_name == "scipy_cg":
            solution, info = cg(matrix, rhs, rtol=rtol, atol=atol, maxiter=max_it, callback=counter)
        elif solver_name == "scipy_minres":
            solution, info = minres(matrix, rhs, rtol=rtol, maxiter=max_it, callback=counter)
        elif solver_name == "scipy_gmres":
            try:
                solution, info = gmres(
                    matrix,
                    rhs,
                    rtol=rtol,
                    atol=atol,
                    restart=min(50, matrix.shape[0]),
                    maxiter=max_it,
                    callback=counter,
                    callback_type="pr_norm",
                )
            except TypeError:
                solution, info = gmres(
                    matrix,
                    rhs,
                    rtol=rtol,
                    atol=atol,
                    restart=min(50, matrix.shape[0]),
                    maxiter=max_it,
                    callback=counter,
                )
        elif solver_name == "scipy_lgmres":
            solution, info = lgmres(
                matrix, rhs, rtol=rtol, atol=atol, maxiter=max_it, callback=counter
            )
        elif solver_name == "scipy_bicgstab":
            solution, info = bicgstab(
                matrix, rhs, rtol=rtol, atol=atol, maxiter=max_it, callback=counter
            )
        else:
            raise ValueError(f"unknown SciPy iterative solver {solver_name!r}.")
        self.check_iterative_solution(matrix, rhs, solution, solver_name, counter.niter, int(info))
        return bm.array(solution, dtype=b.dtype)

    def solve_with_petsc_iterative(self, A, b, solver_name: str):
        """Solve with PETSc KSP/PC selected by ``petsc_<ksp>_<pc>`` names."""
        from petsc4py import PETSc

        parts = solver_name.split("_", 2)
        if len(parts) != 3:
            raise ValueError(
                "PETSc solver names must use 'petsc_<ksp>_<pc>', "
                f"got {solver_name!r}."
            )
        _, ksp_type, pc_type = parts
        ksp_type = "bcgs" if ksp_type == "bicgstab" else ksp_type

        matrix = self.matrix_to_scipy(A).tocsr()
        rhs = np.asarray(bm.to_numpy(b), dtype=PETSc.ScalarType)
        indptr = np.asarray(matrix.indptr, dtype=PETSc.IntType)
        indices = np.asarray(matrix.indices, dtype=PETSc.IntType)
        data = np.asarray(matrix.data, dtype=PETSc.ScalarType)
        petsc_matrix = PETSc.Mat().createAIJ(
            size=matrix.shape,
            csr=(indptr, indices, data),
            comm=PETSc.COMM_SELF,
        )
        rhs_vec = PETSc.Vec().createWithArray(rhs, comm=PETSc.COMM_SELF)
        solution_vec = rhs_vec.duplicate()
        ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
        ksp.setOperators(petsc_matrix)
        ksp.setType(ksp_type)
        ksp.getPC().setType(pc_type)
        ksp.setTolerances(
            rtol=self.config.iterative_rtol,
            atol=self.config.iterative_atol,
            max_it=self.config.iterative_max_it,
        )
        ksp.setFromOptions()
        try:
            ksp.solve(rhs_vec, solution_vec)
            reason = int(ksp.getConvergedReason())
            iterations = int(ksp.getIterationNumber())
            solution = np.asarray(solution_vec.getArray(readonly=True), dtype=float).copy()
            info = 0 if reason > 0 else reason
            self.check_iterative_solution(matrix, rhs, solution, solver_name, iterations, info)
            return bm.array(solution, dtype=b.dtype)
        finally:
            ksp.destroy()
            solution_vec.destroy()
            rhs_vec.destroy()
            petsc_matrix.destroy()

    def check_iterative_solution(self, matrix, rhs, solution, solver_name, iterations, info):
        """Validate and store diagnostics for an iterative linear solve."""
        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0)
        relative_residual = float(np.linalg.norm(matrix @ solution - rhs) / rhs_norm)
        self.last_iterative_solver = solver_name
        self.last_iterative_iterations = int(iterations)
        self.last_iterative_info = int(info)
        self.last_iterative_relative_residual = relative_residual
        residual_tol = max(self.config.iterative_true_residual_tol, 100.0 * self.config.iterative_rtol)
        if info != 0:
            raise RuntimeError(
                f"{solver_name} failed to converge: info={info}, "
                f"iterations={iterations}, true residual={relative_residual:.3e}."
            )
        if relative_residual > residual_tol:
            raise RuntimeError(
                f"{solver_name} failed true residual check: "
                f"{relative_residual:.3e} > {residual_tol:.3e}."
            )

    def solve_with_cupy(self, A, b, matrix_type: str):
        """Solve on the CuPy sparse route after backend array conversion."""
        import cupyx.scipy.sparse.linalg as cpx_linalg

        A_gpu = self._matrix_to_cupy(A)
        b_gpu = self._array_to_cupy(b)

        if matrix_type == "U":
            x_gpu = cpx_linalg.spsolve_triangular(A_gpu, b_gpu, lower=False)
        elif matrix_type == "L":
            x_gpu = cpx_linalg.spsolve_triangular(A_gpu, b_gpu, lower=True)
        else:
            x_gpu = cpx_linalg.spsolve(A_gpu, b_gpu)

        return self._cupy_to_backend(x_gpu, b)

    def _matrix_to_cupy(self, A):
        import cupyx.scipy.sparse as cpx_sparse

        if isinstance(A, CSRTensor):
            crow = self._array_to_cupy(A.crow)
            col = self._array_to_cupy(A.col)
            values = self._array_to_cupy(A.values)
            return cpx_sparse.csr_matrix((values, col, crow), shape=A.sparse_shape)

        if isinstance(A, COOTensor):
            indices = self._array_to_cupy(A.indices)
            values = self._array_to_cupy(A.values)
            return cpx_sparse.coo_matrix(
                (values, (indices[0], indices[1])), shape=A.sparse_shape
            ).tocsr()

        raise TypeError(f"Unsupported sparse matrix type: {type(A)!r}")

    def _array_to_cupy(self, value):
        import cupy as cp

        if isinstance(value, cp.ndarray):
            return value
        if isinstance(value, np.ndarray):
            return cp.asarray(value)

        module = type(value).__module__
        if module.startswith("torch"):
            if value.device.type == "cuda":
                return cp.from_dlpack(value.contiguous())
            return cp.asarray(bm.to_numpy(value))

        return cp.asarray(bm.to_numpy(value))

    def _cupy_to_backend(self, value, reference):
        if self.config.backend == "pytorch":
            module = type(reference).__module__
            if module.startswith("torch") and reference.device.type == "cuda":
                import torch

                return torch.utils.dlpack.from_dlpack(value)

        if self.config.backend == "cupy":
            return value

        return bm.tensor(value.get())

    @staticmethod
    def matrix_to_scipy(matrix):
        """Return a SciPy CSR matrix from FEALPy sparse matrix inputs."""
        if isinstance(matrix, CSRTensor):
            return matrix.to_scipy().tocsr()
        if isinstance(matrix, COOTensor):
            return matrix.tocsr().to_scipy().tocsr()
        if hasattr(matrix, "tocsr"):
            return matrix.tocsr()
        raise TypeError(f"Unsupported sparse matrix type: {type(matrix)!r}")

    def solve_constant_nullspace(
        self,
        A,
        b,
        cell_measure,
        *,
        solver: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_it: Optional[int] = None,
    ):
        """Solve a compatible singular system with constant nullspace data."""
        solver_name = self.select_solver(
            solver if solver is not None else self.config.constant_nullspace_solver
        )
        if not solver_name.startswith("petsc_"):
            raise ValueError(
                "constant-nullspace solves require a PETSc KSP solver. "
                "Set FVMLinearSolverConfig(constant_nullspace_solver='petsc_gmres_hypre') "
                "or pass another PETSc solver such as 'petsc_cg_gamg'. "
                f"The selected solver was {solver_name!r}, which cannot carry "
                "constant-nullspace information."
            )
        rtol = self.config.constant_nullspace_rtol if rtol is None else float(rtol)
        atol = self.config.constant_nullspace_atol if atol is None else float(atol)
        max_it = (
            self.config.constant_nullspace_max_it if max_it is None else int(max_it)
        )
        if rtol <= 0.0:
            raise ValueError("constant_nullspace_rtol must be positive.")
        if atol <= 0.0:
            raise ValueError("constant_nullspace_atol must be positive.")
        if max_it <= 0:
            raise ValueError("constant_nullspace_max_it must be positive.")
        if self.config.constant_nullspace_cache:
            cache = self._constant_nullspace_solver_cache
            if cache is not None and not cache.matches(
                A,
                solver_name=solver_name,
                rtol=rtol,
                atol=atol,
                max_it=max_it,
            ):
                self.clear_constant_nullspace_cache()
                cache = None
            if cache is None:
                cache = CachedPetscConstantNullspaceSolver(
                    A,
                    cell_measure,
                    solver_name=solver_name,
                    rtol=rtol,
                    atol=atol,
                    max_it=max_it,
                )
                self._constant_nullspace_solver_cache = cache
            solution = cache.solve(A, b, cell_measure)
            self.store_constant_nullspace_diagnostics(cache)
            return solution
        return self.solve_constant_nullspace_with_petsc(
            A,
            b,
            cell_measure,
            solver_name=solver_name,
            rtol=rtol,
            atol=atol,
            max_it=max_it,
        )

    def solve_constant_nullspace_with_petsc(
        self,
        A,
        b,
        cell_measure,
        *,
        solver_name: str,
        rtol: float,
        atol: float,
        max_it: int,
    ):
        """Solve a compatible constant-nullspace system through PETSc."""
        cache = CachedPetscConstantNullspaceSolver(
            A,
            cell_measure,
            solver_name=solver_name,
            rtol=rtol,
            atol=atol,
            max_it=max_it,
        )
        try:
            solution = cache.solve(A, b, cell_measure)
            self.store_constant_nullspace_diagnostics(cache)
            return solution
        finally:
            cache.destroy()

    def store_constant_nullspace_diagnostics(self, cache):
        self.last_constant_nullspace_rhs_sum_before = getattr(
            cache, "last_rhs_sum_before", None
        )
        self.last_constant_nullspace_rhs_sum_after = getattr(
            cache, "last_rhs_sum_after", None
        )
        self.last_constant_nullspace_iterations = getattr(
            cache, "last_iterations", None
        )
        self.last_constant_nullspace_reason = getattr(cache, "last_reason", None)
        self.last_constant_nullspace_relative_residual = getattr(
            cache, "last_relative_residual", None
        )


def init_fvm_linear_solver(
    linear_solver=None,
    linear_solver_config=None,
    *,
    reference=None,
) -> FVMLinearSolver:
    """Return an FVM linear-solver object from explicit solver inputs."""
    if linear_solver is not None and hasattr(linear_solver, "solve"):
        return linear_solver

    config = linear_solver_config
    if isinstance(config, dict):
        config = FVMLinearSolverConfig(**config)
    if config is not None:
        return FVMLinearSolver(config)

    device = "cpu"
    if reference is not None:
        try:
            device = str(bm.get_device(reference))
        except Exception:
            device = "cpu"
    return FVMLinearSolver(
        FVMLinearSolverConfig(
            backend=bm.backend_name,
            device=device,
            solver=linear_solver or "auto",
        )
    )
