"""Post-processing helpers for the 2D lid-driven cavity benchmark."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def _as_numpy(values: TensorLike) -> np.ndarray:
    return np.asarray(bm.to_numpy(values))


def _as_backend(values: np.ndarray) -> TensorLike:
    return bm.array(values, dtype=bm.float64)


def sample_centerline(
    points: TensorLike,
    values: TensorLike,
    *,
    fixed_axis: int,
    fixed_value: float,
    coordinate_axis: int,
    atol: float | None = None,
) -> TensorLike:
    """Sample scalar cell values along the nearest available centerline."""
    point_array = _as_numpy(points)
    value_array = _as_numpy(values)
    distance = np.abs(point_array[:, fixed_axis] - fixed_value)
    if atol is None:
        mask = distance <= distance.min() + 1.0e-12
    else:
        mask = distance <= atol
        if not np.any(mask):
            mask = distance <= distance.min() + 1.0e-12

    coordinates = point_array[mask, coordinate_axis]
    selected_values = value_array[mask]
    unique_coordinates, inverse = np.unique(coordinates, return_inverse=True)
    averaged_values = np.array(
        [selected_values[inverse == i].mean() for i in range(unique_coordinates.size)]
    )
    profile = np.column_stack([unique_coordinates, averaged_values])
    return _as_backend(profile)


def centerline_velocity_profiles(
    points: TensorLike,
    velocity: TensorLike,
    *,
    center: tuple[float, float] = (0.5, 0.5),
    atol: float | None = None,
) -> tuple[TensorLike, TensorLike]:
    """Return ``u(x=center_x, y)`` and ``v(x, y=center_y)`` profiles."""
    u_profile = sample_centerline(
        points,
        velocity[:, 0],
        fixed_axis=0,
        fixed_value=center[0],
        coordinate_axis=1,
        atol=atol,
    )
    v_profile = sample_centerline(
        points,
        velocity[:, 1],
        fixed_axis=1,
        fixed_value=center[1],
        coordinate_axis=0,
        atol=atol,
    )
    return u_profile, v_profile


def primary_vortex_summary(
    points: TensorLike,
    velocity: TensorLike,
    *,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    boundary_margin: float = 0.0,
) -> dict[str, float | int]:
    """Approximate the primary vortex center by the minimum interior speed."""
    point_array = _as_numpy(points)
    velocity_array = _as_numpy(velocity)
    xmin, xmax, ymin, ymax = domain
    interior = (
        (point_array[:, 0] >= xmin + boundary_margin)
        & (point_array[:, 0] <= xmax - boundary_margin)
        & (point_array[:, 1] >= ymin + boundary_margin)
        & (point_array[:, 1] <= ymax - boundary_margin)
    )
    if not np.any(interior):
        interior = np.ones(point_array.shape[0], dtype=bool)

    speed = np.linalg.norm(velocity_array, axis=1)
    candidates = np.nonzero(interior)[0]
    cell_index = int(candidates[np.argmin(speed[candidates])])
    return {
        "cell_index": cell_index,
        "x": float(point_array[cell_index, 0]),
        "y": float(point_array[cell_index, 1]),
        "u": float(velocity_array[cell_index, 0]),
        "v": float(velocity_array[cell_index, 1]),
        "speed": float(speed[cell_index]),
    }


def write_profile_csv(
    path: str | Path,
    profile: TensorLike,
    coordinate_name: str,
    value_name: str,
) -> None:
    """Write a two-column centerline profile CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile_array = _as_numpy(profile)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([coordinate_name, value_name])
        for coordinate, value in profile_array:
            writer.writerow([f"{coordinate:.16g}", f"{value:.16g}"])


def write_dict_csv(path: str | Path, rows: Iterable[dict]) -> None:
    """Write a sequence of dictionaries as CSV."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solution_cell_fields(
    uh: TensorLike,
    vh: TensorLike,
    pressure: TensorLike,
    *,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure"),
    velocity_gradient=None,
) -> dict[str, TensorLike]:
    """Build selected cell fields for cavity VTU output."""
    velocity = bm.stack([uh, vh], axis=-1)
    available = {
        "velocity": velocity,
        "u": uh,
        "v": vh,
        "pressure": pressure,
        "speed": bm.sqrt(uh**2 + vh**2),
    }
    if "vorticity" in fields:
        if velocity_gradient is None:
            raise ValueError("velocity_gradient is required for vorticity output.")
        grad = velocity_gradient.cell_gradient(velocity)
        available["vorticity"] = grad[:, 1, 0] - grad[:, 0, 1]
    return {name: available[name] for name in fields}


def write_solution_vtk(
    mesh,
    uh: TensorLike,
    vh: TensorLike,
    pressure: TensorLike,
    path: str | Path,
    *,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure"),
    velocity_gradient=None,
) -> None:
    """Attach cell fields and write a VTU file through the mesh object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in solution_cell_fields(
        uh,
        vh,
        pressure,
        fields=fields,
        velocity_gradient=velocity_gradient,
    ).items():
        mesh.celldata[name] = value
    mesh.to_vtk(fname=str(path))
