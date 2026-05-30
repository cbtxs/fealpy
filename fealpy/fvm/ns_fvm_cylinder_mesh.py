"""Generate and plot the improved triangular mesh for cylinder-flow FVM tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fvm import CylinderFlowCase


def build_case(args) -> CylinderFlowCase:
    return CylinderFlowCase(
        re=args.re,
        rho=args.rho,
        mu=args.mu,
        mean_velocity=args.mean_velocity,
        mesh_size=args.mesh_size,
        cylinder_mesh_size=args.cylinder_mesh_size,
        wake_mesh_size=args.wake_mesh_size,
        cylinder_refine_radius=args.cylinder_refine_radius,
        wake_length=args.wake_length,
        wake_half_width=args.wake_half_width,
        outlet_velocity_policy=args.outlet_velocity_policy,
    )


def plot_mesh(mesh, case: CylinderFlowCase, output: Path | None, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    node = np.asarray(bm.to_numpy(mesh.entity("node")))
    cell = np.asarray(bm.to_numpy(mesh.entity("cell")), dtype=np.int64)

    fig, ax = plt.subplots(figsize=(12.0, 3.2))
    ax.triplot(node[:, 0], node[:, 1], cell, color="0.25", linewidth=0.25)
    ax.add_patch(
        plt.Circle(
            case.center,
            case.radius,
            fill=False,
            color="#b00020",
            linewidth=1.0,
        )
    )
    ax.set_xlim(case.box[0], case.box[1])
    ax.set_ylim(case.box[2], case.box[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("FEALPy FVM cylinder-flow improved triangular mesh")
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def print_quality(quality: dict) -> None:
    print(json.dumps(quality, indent=2, sort_keys=True))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the improved triangular mesh for cylinder-flow FVM tests."
    )
    parser.add_argument("--re", default=20.0, type=float)
    parser.add_argument("--rho", default=1.0, type=float)
    parser.add_argument("--mu", default=None, type=float)
    parser.add_argument("--mean_velocity", default=0.2, type=float)
    parser.add_argument("--mesh_size", default=0.04, type=float)
    parser.add_argument("--cylinder_mesh_size", default=0.006, type=float)
    parser.add_argument("--wake_mesh_size", default=0.02, type=float)
    parser.add_argument("--cylinder_refine_radius", default=None, type=float)
    parser.add_argument("--wake_length", default=None, type=float)
    parser.add_argument("--wake_half_width", default=None, type=float)
    parser.add_argument(
        "--outlet_velocity_policy",
        default="profile",
        choices=("profile", "zero"),
    )
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument(
        "--output",
        default="output/cylinder_flow/improved_tri_mesh.png",
        type=Path,
    )
    parser.add_argument("--vtk", default=None, type=Path)
    parser.add_argument("--show", default=False, action=argparse.BooleanOptionalAction)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    bm.set_backend(args.backend)

    case = build_case(args)
    mesh = case.init_mesh["improved_tri"]()
    quality = case.mesh_quality(mesh)
    print_quality(quality)
    plot_mesh(mesh, case, args.output, show=args.show)

    if args.vtk is not None:
        args.vtk.parent.mkdir(parents=True, exist_ok=True)
        mesh.to_vtk(fname=str(args.vtk))

    print(f"Mesh cells: {mesh.number_of_cells()}")
    print(f"Mesh figure: {args.output}")
    if args.vtk is not None:
        print(f"Mesh VTU: {args.vtk}")


if __name__ == "__main__":
    main()
