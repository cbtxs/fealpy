"""Thin state-solver adapter for cashocs-style shape optimization."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Mapping

from fealpy.backend import backend_manager as bm
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

from .benchmark_common import _write_vtu_frame, get_value


@dataclass(slots=True)
class StateSolveResult:
    velocity: Any = None
    pressure: Any = None
    system_matrix: Any = None
    rhs: Any = None
    mesh: Any = None
    state_vector: Any = None
    velocity_space: Any = None
    pressure_space: Any = None
    state_system_is_linear: bool | None = None
    converged: bool | None = None
    nonlinear_iterations: int | None = None
    final_residual_u: float | None = None
    final_residual_p: float | None = None
    velocity_dirichlet_threshold: Any = None
    pressure_dirichlet_threshold: Any = None


def _as_state_result(result: Any, mesh: Any) -> StateSolveResult:
    if isinstance(result, StateSolveResult):
        return result
    if isinstance(result, tuple) and len(result) >= 2:
        return StateSolveResult(velocity=result[0], pressure=result[1], mesh=mesh)
    if isinstance(result, Mapping):
        return StateSolveResult(
            velocity=result.get("velocity"),
            pressure=result.get("pressure"),
            system_matrix=result.get("system_matrix"),
            rhs=result.get("rhs"),
            mesh=result.get("mesh", mesh),
            state_vector=result.get("state_vector"),
            velocity_space=result.get("velocity_space"),
            pressure_space=result.get("pressure_space"),
            state_system_is_linear=result.get("state_system_is_linear"),
            velocity_dirichlet_threshold=result.get("velocity_dirichlet_threshold"),
            pressure_dirichlet_threshold=result.get("pressure_dirichlet_threshold"),
        )
    return StateSolveResult(
        velocity=get_value(result, "velocity"),
        pressure=get_value(result, "pressure"),
        system_matrix=get_value(result, "system_matrix", "matrix"),
        rhs=get_value(result, "rhs"),
        mesh=get_value(result, "mesh", default=mesh),
        state_vector=get_value(result, "state_vector", "state", "solution"),
        velocity_space=get_value(result, "velocity_space", "uspace", "u_space"),
        pressure_space=get_value(result, "pressure_space", "pspace", "p_space"),
        state_system_is_linear=get_value(result, "state_system_is_linear", default=None),
        velocity_dirichlet_threshold=get_value(result, "velocity_dirichlet_threshold"),
        pressure_dirichlet_threshold=get_value(result, "pressure_dirichlet_threshold"),
    )


def _write_state_vtu_if_requested(mesh: Any, state_result: StateSolveResult, options: Any = None) -> None:
    vtu_path = get_value(options, "debug_vtu_path", "state_vtu_path")
    if vtu_path is not None:
        _write_vtu_frame(Path(vtu_path), mesh, state_result)


def _attach_solver_metadata(state_result: StateSolveResult, solver: Any, mesh: Any) -> StateSolveResult:
    if state_result.mesh is None:
        state_result.mesh = mesh
    fem = getattr(solver, "fem", None)
    if fem is not None:
        if state_result.velocity_space is None:
            state_result.velocity_space = getattr(fem, "uspace", getattr(fem, "velocity_space", None))
        if state_result.pressure_space is None:
            state_result.pressure_space = getattr(fem, "pspace", getattr(fem, "pressure_space", None))
    return state_result


def _build_fem_update_guess(fem: Any, initial_guess: Any = None) -> Any:
    guessed_velocity = get_value(initial_guess, "velocity", "state_velocity", "u", default=None)
    if guessed_velocity is not None:
        return guessed_velocity
    velocity_space = getattr(fem, "uspace", None)
    if velocity_space is None:
        return None
    return velocity_space.function()


def _pin_pressure_gauge(
    fem: Any,
    matrix: Any,
    rhs: Any,
    fluid_model: Any,
    gauge_dof: int = 0,
    gauge_value: Any = 0.0,
) -> tuple[Any, Any]:
    pressure_space = getattr(fem, "pspace", None)
    velocity_space = getattr(fem, "uspace", None)
    if pressure_space is None or velocity_space is None:
        return matrix, rhs
    pde = getattr(fluid_model, "pde", fluid_model)
    pgdof = pressure_space.number_of_global_dofs()
    if pgdof <= 0:
        return matrix, rhs
    gauge_index = int(gauge_dof)
    if gauge_index < 0:
        gauge_index += pgdof
    if gauge_index < 0 or gauge_index >= pgdof:
        raise IndexError(f"pressure gauge dof {gauge_dof} out of range for {pgdof} pressure dofs")
    pressure_mask = bm.zeros(pgdof, dtype=bool)
    pressure_mask[gauge_index] = True
    pressure_values = bm.zeros(pgdof, dtype=pressure_space.ftype, device=getattr(pressure_space, "device", None))
    pressure_values[gauge_index] = gauge_value
    bc = DirichletBC(
        (velocity_space, pressure_space),
        gd=(pde.velocity_dirichlet, pressure_values),
        threshold=(pde.is_velocity_boundary, pressure_mask),
        method="interp",
    )
    return bc.apply(matrix, rhs)


def solve_state_system(
    mesh: Any,
    fluid_model: Any,
    geometry_contract: Any,
    initial_guess: Any = None,
    options: Any = None,
) -> StateSolveResult:
    
    fem = get_value(fluid_model, "fem", default=None)
    solve = get_value(fluid_model, "solve", default=None)
    if fem is None or not callable(solve) or not callable(get_value(fluid_model, "linear_system", default=None)):
        raise TypeError("fluid_model must be a native FEALPy model exposing fem, solve, and linear_system")
    fluid_model.update_mesh(mesh)
    BForm, LForm = fluid_model.linear_system()
    update = getattr(fem, "update", None)
    if callable(update):
        try:
            update_parameters = signature(update).parameters
        except (TypeError, ValueError):
            update_parameters = None
        if update_parameters and len(update_parameters) > 0:
            update(_build_fem_update_guess(fem, initial_guess=initial_guess))
        else:
            update()
    matrix = BForm.assembly()
    rhs = LForm.assembly()
    apply_bc = getattr(fem, "apply_bc", None)
    if callable(apply_bc):
        matrix, rhs = apply_bc(matrix, rhs, getattr(fluid_model, "pde", fluid_model))
    pressure_gauge = get_value(
        options,
        "pressure_gauge",
        default=get_value(getattr(fluid_model, "options", None), "pressure_gauge", default="lagrange_multiplier"),
    )
    
    if pressure_gauge == "pin_dof":
        matrix, rhs = _pin_pressure_gauge(
            fem,
            matrix,
            rhs,
            fluid_model,
            gauge_dof=get_value(options, "pressure_gauge_dof", default=get_value(getattr(fluid_model, "options", None), "pressure_gauge_dof", default=0)),
            gauge_value=get_value(options, "pressure_gauge_value", default=get_value(getattr(fluid_model, "options", None), "pressure_gauge_value", default=0.0)),
        )
    elif hasattr(fem, "lagrange_multiplier"):
        matrix, rhs = fem.lagrange_multiplier(matrix, rhs)
    x = solve(matrix, rhs, solver=get_value(options, "linear_solver", default="mumps"))
    
    ugdof = fem.uspace.number_of_global_dofs()
    pgdof = fem.pspace.number_of_global_dofs()
    velocity = fem.uspace.function()
    pressure = fem.pspace.function()
    velocity[:] = x[:ugdof]
    pressure[:] = x[ugdof : ugdof + pgdof]
    state_result = StateSolveResult(
        velocity=velocity,
        pressure=pressure,
        system_matrix=matrix,
        rhs=rhs,
        mesh=mesh,
        state_vector=x,
        velocity_space=fem.uspace,
        pressure_space=fem.pspace,
        state_system_is_linear=bool(get_value(fluid_model, "state_system_is_linear", default=True)),
    )
    _write_state_vtu_if_requested(mesh, state_result, options)
    return _attach_solver_metadata(state_result, fluid_model, mesh)
