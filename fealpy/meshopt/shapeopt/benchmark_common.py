"""Shared flat helpers for the cashocs-style benchmark entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fealpy.backend import backend_manager as bm
from fealpy.mesh import LagrangeTriangleMesh
from fealpy.mesh.vtk_extent import write_to_vtu


def get_value(source: Any, *names: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return next((source[name] for name in names if name in source), default)
    return next((getattr(source, name) for name in names if hasattr(source, name)), default)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().casefold()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off", "none", ""}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _add_cli_argument(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    dest: str | None = None,
    type: Callable[[Any], Any] | None = None,
    default: Any = None,
    aliases: Sequence[str] = (),
) -> None:
    option_strings = list(dict.fromkeys([f"--{name.replace('_', '-')}", f"--{name}"] + [f"--{alias}" for alias in aliases]))
    parser.add_argument(*option_strings, dest=dest or name, type=type, default=default)


def get_mesh_nodes(mesh: Any) -> bm.ndarray | None:
    if isinstance(mesh, Mapping):
        nodes = mesh.get("node", mesh.get("nodes"))
        if nodes is None:
            return None
        return bm.asarray(nodes, dtype=float)
    nodes = getattr(mesh, "node", None)
    if nodes is None:
        nodes = getattr(mesh, "nodes", None)
    if nodes is None:
        return None
    return bm.asarray(nodes, dtype=float)


def require_mesh_nodes(mesh: Any) -> bm.ndarray:
    nodes = get_mesh_nodes(mesh)
    if nodes is None:
        raise ValueError("mesh does not expose node coordinates")
    return bm.asarray(nodes, dtype=float)


def get_boundary_node_mask(mesh: Any) -> bm.ndarray:
    method = next((getattr(mesh, attribute, None) for attribute in ("is_boundary_node", "boundary_node_flag") if callable(getattr(mesh, attribute, None))), None)
    if method is not None:
        return bm.asarray(method(), dtype=bool).reshape(-1)
    raise ValueError("mesh does not expose a boundary-node flag")


def get_role_nodes(mesh: Any, *names: str) -> bm.ndarray:
    nodes = next(
        (
            bm.asarray(value, dtype=int).reshape(-1)
            for name in names
            for value in [getattr(mesh, name, None)]
            if value is not None and bm.asarray(value, dtype=int).reshape(-1).size > 0
        ),
        None,
    )
    if nodes is not None:
        return bm.asarray(bm.unique(nodes), dtype=int)
    return bm.asarray([], dtype=int)


def _build_visualization_mesh(mesh: Any) -> Any:
    if isinstance(mesh, LagrangeTriangleMesh):
        return mesh
    return LagrangeTriangleMesh.from_triangle_mesh(mesh, p=2)


def _polygon_area_and_centroid(points: Any, fallback_center: tuple[float, float] = (0.0, 0.0)) -> tuple[float, bm.ndarray]:
    coords = bm.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 3 or coords.shape[1] < 2:
        return 0.0, bm.asarray(fallback_center, dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    x_next = bm.roll(x, -1)
    y_next = bm.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(bm.sum(cross))
    if abs(area) <= 1.0e-14:
        return 0.0, coords.mean(axis=0)
    cx = float(bm.sum((x + x_next) * cross) / (6.0 * area))
    cy = float(bm.sum((y + y_next) * cross) / (6.0 * area))
    return abs(area), bm.asarray([cx, cy], dtype=float)


def _polygon_vertex_normals(points: Any) -> bm.ndarray:
    coords = bm.asarray(points, dtype=float)
    if coords.shape[0] == 0:
        return bm.zeros_like(coords)
    if coords.shape[0] == 1:
        return bm.asarray([[1.0, 0.0]], dtype=float)
    centroid = coords.mean(axis=0)
    normals = bm.zeros_like(coords, dtype=float)
    for index in range(coords.shape[0]):
        prev_point = coords[index - 1]
        current_point = coords[index]
        next_point = coords[(index + 1) % coords.shape[0]]
        prev_edge = current_point - prev_point
        next_edge = next_point - current_point
        tangent = bm.zeros(2, dtype=float)
        prev_norm = float(bm.linalg.norm(prev_edge))
        next_norm = float(bm.linalg.norm(next_edge))
        if prev_norm > 0.0:
            tangent += prev_edge / prev_norm
        if next_norm > 0.0:
            tangent += next_edge / next_norm
        if float(bm.linalg.norm(tangent)) <= 0.0:
            tangent = next_point - prev_point
        normal = bm.asarray([tangent[1], -tangent[0]], dtype=float)
        if float(bm.linalg.norm(normal)) <= 0.0:
            normal = current_point - centroid
        if float(bm.linalg.norm(normal)) <= 0.0:
            normal = bm.asarray([1.0, 0.0], dtype=float)
        if float(bm.dot(normal, current_point - centroid)) < 0.0:
            normal = -normal
        norm = float(bm.linalg.norm(normal))
        normals[index] = bm.asarray([1.0, 0.0], dtype=float) if norm <= 0.0 else normal / norm
    return normals


def _map_scalar_product(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        return float(sum(_map_scalar_product(left.get(key, 0.0), right.get(key, 0.0)) for key in keys))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return float(sum(_map_scalar_product(l_item, r_item) for l_item, r_item in zip(left, right)))
    if isinstance(left, Mapping):
        return float(sum(_map_scalar_product(item, right) for item in left.values()))
    if isinstance(right, Mapping):
        return float(sum(_map_scalar_product(left, item) for item in right.values()))
    if isinstance(left, (tuple, list)):
        return float(sum(_map_scalar_product(item, right) for item in left))
    if isinstance(right, (tuple, list)):
        return float(sum(_map_scalar_product(left, item) for item in right))
    return float(bm.asarray(left, dtype=float)) * float(bm.asarray(right, dtype=float))


def _vector_field_to_nodal_array(mesh: Any, vector_field: Any, dim: int = 2) -> bm.ndarray | None:
    nodes = get_mesh_nodes(mesh)
    if nodes is None:
        return None
    node_count = int(nodes.shape[0])

    if vector_field is None:
        return bm.zeros((node_count, dim), dtype=float)

    if isinstance(vector_field, Mapping):
        nodal_values = bm.zeros((node_count, dim), dtype=float)
        for node, value in vector_field.items():
            node_id = int(node)
            if 0 <= node_id < node_count:
                vec = bm.asarray(value, dtype=float).reshape(-1)
                if vec.size == 0:
                    continue
                if vec.size < dim:
                    padded = bm.zeros(dim, dtype=float)
                    padded[: vec.size] = vec
                    vec = padded
                nodal_values[node_id, :dim] = vec[:dim]
        return nodal_values

    values = bm.asarray(vector_field, dtype=float)
    if values.ndim == 1:
        if values.size == node_count * dim:
            return values.reshape(node_count, dim)
        if values.size == dim:
            return bm.tile(values.reshape(1, dim), (node_count, 1))
        return None

    if values.ndim >= 2 and values.shape[0] == node_count:
        if values.shape[1] < dim:
            padded = bm.zeros((node_count, dim), dtype=float)
            padded[:, : values.shape[1]] = values
            return padded
        return values[:, :dim]

    return None


def _prolong_nodal_vector_to_visualization_mesh(mesh: Any, nodal_vectors: bm.ndarray, vis_mesh: Any | None = None) -> bm.ndarray:
    values = bm.asarray(nodal_vectors, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[1] == 2:
        values = bm.concat(
        [values, bm.zeros((values.shape[0], 1), dtype=values.dtype)],axis=1,)
    elif values.shape[1] > 3:
        values = values[:, :3]
    elif values.shape[1] < 3:
        padded = bm.zeros((values.shape[0], 3), dtype=values.dtype)
        padded[:, : values.shape[1]] = values
        values = padded

    if vis_mesh is None:
        vis_mesh = _build_visualization_mesh(mesh)
    vis_node_count = int(bm.asarray(vis_mesh.entity("node")).shape[0])
    if values.shape[0] == vis_node_count:
        return values

    mesh_nodes = get_mesh_nodes(mesh)
    if mesh_nodes is None:
        raise ValueError("mesh does not expose node coordinates")
    if values.shape[0] != mesh_nodes.shape[0]:
        raise ValueError(
            "vector field cannot be prolonged to visualization mesh: "
            f"{values.shape[0]} base values vs {mesh_nodes.shape[0]} mesh nodes"
        )

    edges = bm.asarray(mesh.entity("edge"), dtype=int)
    midpoint_values = 0.5 * (values[edges[:, 0]] + values[edges[:, 1]])
    prolonged = bm.concatenate((values, midpoint_values), axis=0)
    if prolonged.shape[0] != vis_node_count:
        raise ValueError(
            "vector prolongation does not match visualization mesh node count: "
            f"{prolonged.shape[0]} != {vis_node_count}"
        )
    return prolonged


def _build_vtu_point_data(
    mesh: Any,
    state_result: Any,
    vis_mesh: Any | None = None,
    *,
    step_result: Any = None,
) -> dict[str, bm.ndarray]:
    point_data: dict[str, bm.ndarray] = {}
    velocity = getattr(state_result, "velocity", None)
    pressure = getattr(state_result, "pressure", None)

    if velocity is not None:
        velocity_array = bm.asarray(getattr(velocity, "array", velocity))
        if velocity_array.ndim == 1:
            velocity_array = velocity_array.reshape(-1, 1)
        dof_numel = int(getattr(getattr(velocity, "space", None), "dof_numel", velocity_array.shape[-1]))
        dof_priority = bool(getattr(getattr(velocity, "space", None), "dof_priority", True))
        vis_node_count = int(bm.asarray(vis_mesh.entity("node")).shape[0]) if vis_mesh is not None else None

        if velocity_array.shape[-1] == dof_numel and velocity_array.ndim == 2:
            point_vectors = velocity_array
        else:
            if vis_node_count is not None and velocity_array.size == vis_node_count * dof_numel:
                dof_count = vis_node_count
            elif velocity_array.size % dof_numel == 0:
                dof_count = velocity_array.size // dof_numel
            else:
                raise ValueError("velocity array cannot be reshaped to tensor point data")

            if dof_priority:
                point_vectors = bm.concat(
                    [velocity_array[i * dof_count : (i + 1) * dof_count] for i in range(dof_numel)], axis=1,
                )
            else:
                point_vectors = velocity_array.reshape(dof_count, dof_numel)

        if point_vectors.shape[1] == 1:
            point_data["u"] = point_vectors[:, 0]
        elif point_vectors.shape[1] == 2:
            point_data["u"] = bm.concat([point_vectors, bm.zeros((point_vectors.shape[0], 1), dtype=point_vectors.dtype)], axis=1)
        else:
            point_data["u"] = point_vectors[:, :3]

    if pressure is not None:
        pressure_array = bm.asarray(getattr(pressure, "array", pressure)).reshape(-1)
        if vis_mesh is None:
            vis_mesh = _build_visualization_mesh(mesh)
        vis_node_count = int(bm.asarray(vis_mesh.entity("node")).shape[0])
        if pressure_array.size == vis_node_count:
            point_data["p"] = pressure_array
        else:
            edge = bm.asarray(mesh.entity("edge"), dtype=int)
            midpoint_values = 0.5 * (pressure_array[edge[:, 0]] + pressure_array[edge[:, 1]])
            prolonged_pressure = bm.concatenate((pressure_array, midpoint_values), axis=0)
            if prolonged_pressure.size != vis_node_count:
                raise ValueError(
                    "pressure prolongation does not match visualization mesh node count: "
                    f"{prolonged_pressure.size} != {vis_node_count}"
                )
            point_data["p"] = prolonged_pressure

    boundary_displacement = None
    if step_result is not None:
        boundary_displacement = getattr(step_result, "boundary_displacement", None)
        if boundary_displacement is None and isinstance(step_result, Mapping):
            boundary_displacement = step_result.get("boundary_displacement")
    if boundary_displacement is not None or vis_mesh is not None:
        shape_vectors = _vector_field_to_nodal_array(mesh, boundary_displacement, dim=2)
        if shape_vectors is not None:
            shape_vectors = _prolong_nodal_vector_to_visualization_mesh(mesh, shape_vectors, vis_mesh=vis_mesh)
            point_data["shape_d"] = shape_vectors

    return point_data


def _write_vtu_frame(path: Path, mesh: Any, state_result: Any, step_result: Any = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis_mesh = _build_visualization_mesh(mesh)
    node, cell, cell_type, num_cells = vis_mesh.to_vtk(fname=None)
    point_data = _build_vtu_point_data(mesh, state_result, vis_mesh=vis_mesh, step_result=step_result)
    write_to_vtu(str(path), node, num_cells, cell_type, cell, nodedata=point_data)


def _compute_convergence_rates(epsilons: Sequence[float], residuals: Sequence[float]) -> list[float]:
    rates: list[float] = []
    for left, right, residual_left, residual_right in zip(epsilons[:-1], epsilons[1:], residuals[:-1], residuals[1:]):
        if left <= 0.0 or right <= 0.0 or residual_left <= 0.0 or residual_right <= 0.0:
            rates.append(float("nan"))
            continue
        rates.append(float(bm.log(residual_left / residual_right) / bm.log(left / right)))
    return rates


def _remesh_quality_parameters(profile: str) -> dict[str, float | str]:
    normalized = str(profile).casefold()
    if normalized in {"cashocs", "default"}:
        return {
            "mesh_quality_measure": "condition_number",
            "mesh_quality_type": "min",
            "tol_lower": 0.1,
            "tol_upper": 0.25,
        }
    if normalized == "stress":
        return {
            "mesh_quality_measure": "condition_number",
            "mesh_quality_type": "min",
            "tol_lower": 0.35,
            "tol_upper": 0.5,
        }
    raise ValueError(f"Unsupported remesh_quality_profile: {profile!r}")


__all__ = [
    "get_value",
    "_parse_bool",
    "_add_cli_argument",
    "get_mesh_nodes",
    "require_mesh_nodes",
    "get_boundary_node_mask",
    "get_role_nodes",
    "_build_visualization_mesh",
    "_polygon_area_and_centroid",
    "_polygon_vertex_normals",
    "_map_scalar_product",
    "_vector_field_to_nodal_array",
    "_prolong_nodal_vector_to_visualization_mesh",
    "_build_vtu_point_data",
    "_write_vtu_frame",
    "_compute_convergence_rates",
    "_remesh_quality_parameters",
]
