"""伴随方程求解器骨架。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from fealpy.backend import backend_manager as bm
from fealpy.fem import (
    BilinearForm,
    BlockForm,
    DirichletBC,
    LinearBlockForm,
    LinearForm,
    PressWorkIntegrator,
    ScalarDiffusionIntegrator,
    VectorSourceIntegrator,
)
from fealpy.solver import spsolve

try:
    from .geometry_regularization import (
        _polygon_vertex_normals,
        _polygon_vertex_weights,
    )
    from .objective import ObjectiveDerivativeSource
except ImportError:  # pragma: no cover
    from geometry_regularization import (
        _polygon_vertex_normals,
        _polygon_vertex_weights,
    )
    from objective import ObjectiveDerivativeSource


@dataclass(slots=True)
class AdjointSolveResult:
    """伴随方程求解结果。"""

    adjoint: Any = None
    system_matrix: Any = None
    rhs: Any = None
    adjoint_vector: Any = None
    adjoint_rhs: Any = None
    mesh: Any = None
    shape_derivative: Any = None
    shape_density: Any = None


@dataclass(slots=True)
class AdjointWeakForm:
    """FEALPy 伴随弱式打包结果。"""

    mesh: Any
    velocity_space: Any
    pressure_space: Any
    velocity_bilinear_form: Any
    pressure_velocity_form: Any
    velocity_pressure_form: Any
    load_form: Any
    block_form: Any
    linear_block_form: Any
    adjoint_rhs_source: Any = None
    boundary_conditions: tuple[Any, ...] = ()


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    """从映射或对象属性里取第一个可用字段。"""
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _velocity_gradient_at_nodes(mesh: Any, field: Any) -> Any | None:
    if field is None or not hasattr(field, "space") or not hasattr(field.space, "grad_value"):
        return None
    try:
        qf = mesh.quadrature_formula(q=4, etype="cell")
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad_u = field.space.grad_value(uh=field, bc=bcs)
        grad_u = bm.einsum("n, knij -> kij", ws, grad_u)
        cellmeasure = mesh.entity_measure("cell")
        n2c = mesh.node_to_cell()
        weights = bm.ones(n2c.shape)
        weights *= cellmeasure
        weights = n2c.mul(weights)
        weights = weights.toarray()
        weights_sum = bm.sum(weights, axis=1)
        weights = weights / weights_sum[:, None]
        grad_at_nodes = bm.einsum("lk, kij -> lij", weights, grad_u)
        return bm.asarray(grad_at_nodes, dtype=float)
    except Exception:
        return None


def _ns_boundary_shape_derivative(
    mesh: Any,
    state_result: Any,
    adjoint_result: Any,
    geometry_contract: Any,
    options: Any = None,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]] | None:
    """构造稳态 NS 耗散目标的严格边界形状导数代表。

    这里采用论文中的 Hadamard 结构：

    dJ(Ω)[V] = ∫_{Γ_D} G_NS (V · n) ds

    对于当前弯管耗散目标，优先对齐论文第 5.1 节的最简式：

    G_NS = μ |∂_n u|^2

    如果显式指定了扩展模式，则也可保留伴随修正项

    G_NS = μ (|∂_n u|^2 - ∂_n λ · ∂_n u)

    然后将其作为法向边界代表 `G_NS n` 返回，供后续几何梯度与 Riesz
    投影继续处理。
    """
    state_velocity = get_value(state_result, "velocity", "state_velocity", "u")
    if state_velocity is None:
        return None

    objective_parameters = get_value(options, "objective_parameters", default=None)
    is_hole_boundary = bool(get_value(objective_parameters, "is_hole_boundary", default=False))
    resolved_objective_parameters = objective_parameters if isinstance(objective_parameters, Mapping) else {}
    design_node_order = get_value(
        resolved_objective_parameters,
        "design_boundary_node_order",
        "design_boundary_node_ids",
        default=None,
    )
    if design_node_order is None:
        return None

    design_node_ids = bm.asarray(design_node_order, dtype=int).reshape(-1)
    if design_node_ids.size == 0:
        return None

    if isinstance(mesh, Mapping):
        nodes = mesh.get("node", mesh.get("nodes"))
    else:
        nodes = get_value(mesh, "node", "nodes")
    if nodes is None:
        return None
    nodes = bm.asarray(nodes, dtype=float)

    coords = nodes[design_node_ids]
    normals = _polygon_vertex_normals(coords, is_hole_boundary=is_hole_boundary)
    vertex_weights = _polygon_vertex_weights(coords)
    state_grad_at_nodes = _velocity_gradient_at_nodes(mesh, state_velocity)
    if state_grad_at_nodes is None:
        return None

    viscosity = float(
        resolved_objective_parameters.get(
            "viscosity",
            get_value(state_result, "viscosity", "mu", "nu", default=1.0),
        )
    )
    density_variant = str(
        resolved_objective_parameters.get(
            "shape_density_variant",
            resolved_objective_parameters.get("shape_density_formula", "paper"),
        )
    ).strip().lower()
    factor_volume = float(resolved_objective_parameters.get("factor_volume", 0.0) or 0.0)
    factor_barycenter = float(resolved_objective_parameters.get("factor_barycenter", 0.0) or 0.0)
    reference_volume = float(resolved_objective_parameters.get("volume_reference", 0.0) or 0.0)
    reference_barycenter = bm.asarray(
        resolved_objective_parameters.get("barycenter_reference", bm.zeros(2, dtype=float)),
        dtype=float,
    ).reshape(-1)
    gradient = bm.zeros((design_node_ids.size, 2), dtype=float)
    density_values = bm.zeros(design_node_ids.size, dtype=float)
    for index, node_id in enumerate(design_node_ids.tolist()):
        normal = bm.asarray(normals[index], dtype=float)
        state_normal = bm.zeros(2, dtype=float)
        if 0 <= int(node_id) < state_grad_at_nodes.shape[0]:
            grad_u = bm.asarray(state_grad_at_nodes[int(node_id)], dtype=float)
            state_normal = grad_u @ normal
        density = float(viscosity) * float(bm.dot(state_normal, state_normal))
        if density_variant not in {"paper", "paper_dissipation", "paper_dissipation_only"}:
            adjoint = get_value(adjoint_result, "adjoint", default=None)
            if isinstance(adjoint, Mapping):
                adjoint_velocity = get_value(adjoint, "velocity", "state_velocity", "u")
                adjoint_grad_at_nodes = _velocity_gradient_at_nodes(mesh, adjoint_velocity) if adjoint_velocity is not None else None
                if adjoint_grad_at_nodes is not None and 0 <= int(node_id) < adjoint_grad_at_nodes.shape[0]:
                    grad_v = bm.asarray(adjoint_grad_at_nodes[int(node_id)], dtype=float)
                    adjoint_normal = grad_v @ normal
                    density -= float(viscosity) * float(bm.dot(adjoint_normal, state_normal))
        density_values[index] = density
        gradient[index] = density * normal * float(vertex_weights[index])

    return {
        int(node_id): (float(vector[0]), float(vector[1]))
        for node_id, vector in zip(design_node_ids.tolist(), gradient, strict=True)
    }, {
        int(node_id): float(density)
        for node_id, density in zip(design_node_ids.tolist(), density_values, strict=True)
    }


def _normalize_adjoint_rhs(adjoint_rhs: Any) -> tuple[Any, Any]:
    """把目标导数源项规范成可装配对象和元数据。"""
    if isinstance(adjoint_rhs, ObjectiveDerivativeSource):
        return adjoint_rhs.source, adjoint_rhs
    if callable(adjoint_rhs):
        wrapped = ObjectiveDerivativeSource(source=adjoint_rhs)
        return wrapped.source, wrapped
    return adjoint_rhs, adjoint_rhs


def _as_adjoint_result(result: Any) -> AdjointSolveResult:
    """把外部求解结果规整成伴随结果。"""
    if isinstance(result, AdjointSolveResult):
        return result
    if isinstance(result, Mapping):
        return AdjointSolveResult(
            adjoint=result.get("adjoint"),
            system_matrix=result.get("system_matrix"),
            rhs=result.get("rhs"),
            adjoint_vector=result.get("adjoint_vector"),
            adjoint_rhs=result.get("adjoint_rhs"),
            mesh=result.get("mesh"),
            shape_derivative=result.get("shape_derivative"),
            shape_density=result.get("shape_density"),
        )
    return AdjointSolveResult(
        adjoint=get_value(result, "adjoint"),
        system_matrix=get_value(result, "system_matrix", "matrix"),
        rhs=get_value(result, "rhs"),
        adjoint_vector=get_value(result, "adjoint_vector", "solution"),
        adjoint_rhs=get_value(result, "adjoint_rhs"),
        mesh=get_value(result, "mesh"),
        shape_derivative=get_value(result, "shape_derivative"),
        shape_density=get_value(result, "shape_density"),
    )


def _build_adjoint_dirichlet_bc(
    state_result: Any,
    velocity_space: Any,
    pressure_space: Any,
    options: Any = None,
) -> DirichletBC:
    """构造伴随速度/压力边界条件。"""
    velocity_gd = get_value(
        state_result,
        "adjoint_velocity_dirichlet_data",
        "adjoint_inlet_velocity",
        "adjoint_velocity_bc",
        default=0.0,
    )
    velocity_threshold = get_value(
        state_result,
        "adjoint_velocity_dirichlet_threshold",
        "velocity_dirichlet_threshold",
        "adjoint_velocity_boundary_threshold",
        "adjoint_velocity_threshold",
        default=velocity_space.is_boundary_dof(),
    )
    pressure_gd = get_value(state_result, "adjoint_pressure_dirichlet_data", "adjoint_pressure_bc", default=0.0)
    if isinstance(pressure_gd, (int, float)):
        pressure_gd = bm.zeros(
            pressure_space.number_of_global_dofs(),
            dtype=float,
        ) + pressure_gd
    pressure_threshold = get_value(
        state_result,
        "adjoint_pressure_dirichlet_threshold",
        "pressure_dirichlet_threshold",
        "adjoint_pressure_boundary_threshold",
        default=bm.zeros(
            pressure_space.number_of_global_dofs(),
            dtype=bool,
        ),
    )

    return DirichletBC(
        (velocity_space, pressure_space),
        gd=(velocity_gd, pressure_gd),
        threshold=(velocity_threshold, pressure_threshold),
        method=get_value(options, "boundary_method", default="interp"),
    )


def assemble_adjoint_weak_form(
    mesh: Any,
    state_result: Any,
    adjoint_rhs: Any,
    geometry_contract: Any,
    options: Any = None,
) -> AdjointWeakForm:
    if hasattr(state_result, "state_spaces"):
        spaces = state_result.state_spaces
        if callable(spaces):
            spaces = spaces(mesh, geometry_contract)
    elif hasattr(state_result, "build_state_spaces") and callable(state_result.build_state_spaces):
        spaces = state_result.build_state_spaces(mesh, geometry_contract)
    else:
        spaces = (
            get_value(state_result, "velocity_space", "uspace", "u_space"),
            get_value(state_result, "pressure_space", "pspace", "p_space"),
        )
    if not (isinstance(spaces, (tuple, list)) and len(spaces) >= 2):
        raise ValueError("state_result must provide velocity and pressure spaces")

    velocity_space, pressure_space = spaces[0], spaces[1]
    source, source_metadata = _normalize_adjoint_rhs(adjoint_rhs)
    q = get_value(options, "q", default=None)
    viscosity = get_value(state_result, "viscosity", "mu", "nu", default=1.0)
    velocity_bilinear_form = BilinearForm(velocity_space)
    velocity_bilinear_form.add_integrator(ScalarDiffusionIntegrator(coef=viscosity, q=q))
    pressure_velocity_form = BilinearForm((pressure_space, velocity_space))
    pressure_velocity_form.add_integrator(PressWorkIntegrator(coef=-1.0, q=q))
    velocity_pressure_form = BilinearForm((pressure_space, velocity_space))
    velocity_pressure_form.add_integrator(PressWorkIntegrator(coef=-1.0, q=q))
    load_form = LinearForm(velocity_space)
    if source not in (None, 0, 0.0):
        load_form.add_integrator(VectorSourceIntegrator(source, q=q))
    return AdjointWeakForm(
        mesh=mesh,
        velocity_space=velocity_space,
        pressure_space=pressure_space,
        velocity_bilinear_form=velocity_bilinear_form,
        pressure_velocity_form=pressure_velocity_form,
        velocity_pressure_form=velocity_pressure_form,
        load_form=load_form,
        block_form=BlockForm(
            [
                [velocity_bilinear_form, pressure_velocity_form],
                [velocity_pressure_form.T, None],
            ]
        ),
        linear_block_form=LinearBlockForm([load_form, LinearForm(pressure_space)]),
        adjoint_rhs_source=source_metadata,
        boundary_conditions=(
            _build_adjoint_dirichlet_bc(state_result, velocity_space, pressure_space, options),
        ),
    )


def solve_adjoint_system(
    mesh: Any,
    state_result: Any,
    adjoint_rhs: Any,
    geometry_contract: Any,
    options: Any = None,
) -> AdjointSolveResult:
    """求解伴随方程。"""
    provider = get_value(
        state_result,
        "solve_adjoint_system",
        "solve_adjoint_equation",
        "solve",
    )
    if callable(provider):
        result = provider(
            mesh,
            adjoint_rhs,
            geometry_contract,
            options=options,
        )
        result = _as_adjoint_result(result)
        if result.shape_derivative is None:
            ns_shape = _ns_boundary_shape_derivative(mesh, state_result, result, geometry_contract, options)
            if isinstance(ns_shape, tuple) and len(ns_shape) == 2:
                result.shape_derivative, result.shape_density = ns_shape
            else:
                result.shape_derivative = ns_shape
        return result

    weak_form = assemble_adjoint_weak_form(mesh, state_result, adjoint_rhs, geometry_contract, options=options)

    if isinstance(weak_form, AdjointWeakForm):
        reuse_state_matrix = bool(get_value(options, "reuse_state_matrix", default=True))
        diagnostics = bool(get_value(options, "diagnostics", default=False))
        state_matrix = get_value(state_result, "system_matrix", "matrix", default=None)
        velocity_dofs = weak_form.velocity_space.number_of_global_dofs()
        pressure_dofs = weak_form.pressure_space.number_of_global_dofs()
        mixed_dofs = int(velocity_dofs + pressure_dofs)
        state_matrix_shape = getattr(state_matrix, "shape", None)
        if reuse_state_matrix and state_matrix_shape is not None:
            if tuple(int(value) for value in state_matrix_shape[:2]) != (mixed_dofs, mixed_dofs):
                reuse_state_matrix = False
                if diagnostics:
                    print(
                        "[adjoint] reuse_state_matrix=0, "
                        f"state matrix shape {state_matrix_shape} does not match mixed dofs {mixed_dofs}"
                    )
        if reuse_state_matrix and state_matrix is not None and hasattr(state_matrix, "T"):
            matrix = state_matrix.T
            if diagnostics:
                print("[adjoint] reuse_state_matrix=1, using transpose of state/Newton matrix")
        else:
            matrix = weak_form.block_form.assembly(format="csr")
            if diagnostics:
                print("[adjoint] reuse_state_matrix=0, assembling adjoint block matrix")
        rhs = weak_form.linear_block_form.assembly(format="dense")
        bc = _build_adjoint_dirichlet_bc(state_result, weak_form.velocity_space, weak_form.pressure_space, options)
        matrix, rhs = bc.apply(matrix, rhs)
        x = spsolve(matrix, rhs, solver=get_value(options, "linear_solver", default="mumps"))
        velocity = weak_form.velocity_space.function()
        pressure = weak_form.pressure_space.function()
        velocity[:] = x[:velocity_dofs]
        pressure[:] = x[velocity_dofs:velocity_dofs + pressure_dofs]
        result = AdjointSolveResult(
            adjoint={"velocity": velocity, "pressure": pressure},
            system_matrix=matrix,
            rhs=rhs,
            adjoint_vector=x,
            adjoint_rhs=adjoint_rhs,
            mesh=mesh,
        )
        if result.shape_derivative is None:
            ns_shape = _ns_boundary_shape_derivative(mesh, state_result, result, geometry_contract, options)
            if isinstance(ns_shape, tuple) and len(ns_shape) == 2:
                result.shape_derivative, result.shape_density = ns_shape
            else:
                result.shape_derivative = ns_shape
        return result
    result = AdjointSolveResult(
        adjoint=None,
        system_matrix=weak_form,
        rhs=adjoint_rhs,
        adjoint_vector=None,
        adjoint_rhs=adjoint_rhs,
        mesh=mesh,
    )
    if result.shape_derivative is None:
        ns_shape = _ns_boundary_shape_derivative(mesh, state_result, result, geometry_contract, options)
        if isinstance(ns_shape, tuple) and len(ns_shape) == 2:
            result.shape_derivative, result.shape_density = ns_shape
        else:
            result.shape_derivative = ns_shape
    return result
