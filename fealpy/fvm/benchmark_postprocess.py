"""Common output helpers for FVM benchmark examples."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def re_label(re: float) -> str:
    """Return a filesystem-stable Reynolds-number label."""
    text = f"{float(re):g}".replace(".", "p").replace("-", "m")
    return f"Re{text}"


def scalarize_value(value):
    """Convert backend scalar or array values to CSV-friendly objects."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    array = np.asarray(bm.to_numpy(value))
    if array.shape == ():
        return array.item()
    return array.tolist()


def scalarize_rows(rows: Iterable[dict]) -> list[dict]:
    """Convert backend scalar values in dictionaries to CSV-friendly objects."""
    return [{key: scalarize_value(value) for key, value in row.items()} for row in rows]


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
    """Build selected cell fields for VTU output."""
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
