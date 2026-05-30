"""Shared runner utilities for lid-driven cavity examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fealpy.backend import backend_manager as bm

from .lid_driven_cavity_postprocess import (
    centerline_velocity_profiles,
    primary_vortex_summary,
    write_dict_csv,
    write_profile_csv,
    write_solution_vtk,
)


@dataclass(frozen=True)
class CavityOutputConfig:
    output_dir: Path
    write_vtk: bool = True
    write_final: bool = True
    write_interval_steps: int | None = None
    write_interval_time: float | None = None
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure", "speed")


def re_label(re: float) -> str:
    """Return a filesystem-stable Reynolds-number label."""
    text = f"{float(re):g}".replace(".", "p").replace("-", "m")
    return f"Re{text}"


def mesh_label(mesh_type: str, nx: int, ny: int) -> str:
    """Return a compact mesh label for output directories."""
    prefix = mesh_type
    if prefix.startswith("uniform_"):
        prefix = prefix[len("uniform_") :]
    return f"{prefix}_{int(nx)}x{int(ny)}"


def default_output_dir(
    solver: str,
    re: float,
    mesh_type: str,
    nx: int,
    ny: int,
    *,
    root: str | Path = "output/lid_driven_cavity",
) -> Path:
    return Path(root) / solver / re_label(re) / mesh_label(mesh_type, nx, ny)


def _scalarize(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    array = np.asarray(bm.to_numpy(value))
    if array.shape == ():
        return array.item()
    return array.tolist()


def scalarize_rows(rows: Iterable[dict]) -> list[dict]:
    """Convert backend scalar values in dictionaries to CSV-friendly objects."""
    return [{key: _scalarize(value) for key, value in row.items()} for row in rows]


def should_write_snapshot(
    step: int,
    time: float,
    config: CavityOutputConfig,
    *,
    is_final: bool = False,
) -> bool:
    """Return whether a snapshot should be written for this time step."""
    if is_final:
        return bool(config.write_final)
    if config.write_interval_steps is not None and config.write_interval_steps > 0:
        if step % config.write_interval_steps == 0:
            return True
    if config.write_interval_time is not None and config.write_interval_time > 0.0:
        quotient = time / config.write_interval_time
        if abs(quotient - round(quotient)) <= 1.0e-12:
            return True
    return False


class CavitySnapshotWriter:
    """Write selected transient cavity snapshots and collect time history."""

    def __init__(
        self,
        config: CavityOutputConfig,
        *,
        domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        nt: int | None = None,
        boundary_margin: float = 0.05,
    ) -> None:
        self.config = config
        self.domain = domain
        self.nt = nt
        self.boundary_margin = boundary_margin
        self.history: list[dict] = []
        self._previous_velocity = None

    def __call__(
        self,
        *,
        step: int,
        time: float,
        model,
        cell_velocity,
        face_velocity=None,
        pressure=None,
        flux=None,
    ) -> None:
        uh = cell_velocity[:, 0]
        vh = cell_velocity[:, 1]
        velocity = bm.stack([uh, vh], axis=-1)
        speed = bm.sqrt(uh**2 + vh**2)
        vortex = primary_vortex_summary(
            model.mesh.entity_barycenter("cell"),
            velocity,
            domain=self.domain,
            boundary_margin=self.boundary_margin,
        )
        row = {
            "step": int(step),
            "time": float(time),
            "max_speed": _scalarize(bm.max(speed)),
            "velocity_update": self._velocity_update(cell_velocity),
            "mass_residual": self._mass_residual(model, flux),
            "vortex_x": vortex["x"],
            "vortex_y": vortex["y"],
        }
        self.history.append(row)

        is_final = self.nt is not None and int(step) == int(self.nt)
        if self.config.write_vtk and should_write_snapshot(
            step, time, self.config, is_final=is_final
        ):
            self.write_snapshot(model, step, cell_velocity, pressure)

        self._previous_velocity = bm.array(cell_velocity)

    def _velocity_update(self, cell_velocity):
        if self._previous_velocity is None:
            return 0.0
        delta = cell_velocity - self._previous_velocity
        return _scalarize(bm.max(bm.abs(delta)))

    @staticmethod
    def _mass_residual(model, flux):
        if flux is None or not hasattr(model, "divergence_from_flux"):
            return None
        imbalance = model.divergence_from_flux(flux)
        return _scalarize(bm.max(bm.abs(imbalance)))

    def write_snapshot(self, model, step: int, cell_velocity, pressure) -> None:
        snapshot_dir = self.config.output_dir / f"{int(step):06d}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        uh = cell_velocity[:, 0]
        vh = cell_velocity[:, 1]
        write_solution_vtk(
            model.mesh,
            uh,
            vh,
            pressure,
            snapshot_dir / "solution.vtu",
            fields=self.config.fields,
            velocity_gradient=getattr(model, "velocity_gradient", None),
        )

    def write_time_history(self) -> None:
        write_dict_csv(self.config.output_dir / "time_history.csv", self.history)


def write_benchmark_outputs(
    model,
    output_dir: str | Path,
    *,
    residuals: Iterable[dict] | None = None,
    domain: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    boundary_margin: float = 0.05,
    run_summary: dict | None = None,
    fields: tuple[str, ...] = ("velocity", "u", "v", "pressure", "speed"),
) -> dict:
    """Write standard cavity benchmark outputs for a solved model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = model.mesh.entity_barycenter("cell")
    velocity = bm.stack([model.uh, model.vh], axis=-1)
    u_profile, v_profile = centerline_velocity_profiles(points, velocity)
    vortex = primary_vortex_summary(
        points,
        velocity,
        domain=domain,
        boundary_margin=boundary_margin,
    )

    write_profile_csv(output_dir / "centerline_u.csv", u_profile, "y", "u")
    write_profile_csv(output_dir / "centerline_v.csv", v_profile, "x", "v")
    write_dict_csv(output_dir / "vortex_summary.csv", [vortex])
    write_solution_vtk(
        model.mesh,
        model.uh,
        model.vh,
        model.ph,
        output_dir / "solution.vtu",
        fields=fields,
        velocity_gradient=getattr(model, "velocity_gradient", None),
    )

    if residuals is not None:
        write_dict_csv(output_dir / "residual_history.csv", scalarize_rows(residuals))

    if run_summary is not None:
        lines = [f"{key}: {value}" for key, value in run_summary.items()]
        (output_dir / "run_summary.txt").write_text("\n".join(lines) + "\n")

    return {
        "output_dir": output_dir,
        "u_profile": u_profile,
        "v_profile": v_profile,
        "vortex": vortex,
    }
