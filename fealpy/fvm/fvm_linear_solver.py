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
    Navier-Stokes, SIMPLE, PISO, or boundary-condition mathematics.
    """

    backend: str = "numpy"
    device: str = "cpu"
    solver: str = "auto"
    default_matrix_type: str = "G"


class FVMLinearSolver:
    """Centralized sparse linear-system solver entry for FVM models.

    The purpose of this class is to isolate backend/device conversion and
    direct-solver dispatch from PDE models.  Models should hand it an assembled
    FEALPy sparse matrix and RHS; SIMPLE, PISO, boundary-condition, and
    pressure-normalization mathematics remain in the model layer.

    The current CUDA route is a compatibility path based on PyTorch CUDA
    tensors and CuPy sparse solvers.  It is not yet a cached or matrix-free GPU
    high-performance solver.
    """

    def __init__(self, config: Optional[FVMLinearSolverConfig] = None):
        self.config = config or FVMLinearSolverConfig()

    def solve(self, A, b, matrix_type: Optional[str] = None, solver: Optional[str] = None):
        solver_name = self.select_solver(solver)
        matrix_type = matrix_type or self.config.default_matrix_type

        if solver_name == "cupy":
            return self.solve_with_cupy(A, b, matrix_type)
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
