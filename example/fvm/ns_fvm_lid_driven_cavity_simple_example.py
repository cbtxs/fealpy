"""Run the lid-driven cavity benchmark with ``NSFVMSimpleModel``."""

from __future__ import annotations

import argparse
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMLinearSolverConfig, LidDrivenCavityCase, NSFVMSimpleModel
from fealpy.fvm.lid_driven_cavity_postprocess import (
    default_output_dir,
    write_benchmark_outputs,
)


def build_simple_options(
    case: LidDrivenCavityCase,
    *,
    nx: int,
    ny: int,
    mesh_type: str,
    backend: str,
    device: str,
    log_level: str,
    rho: float | None = None,
    mu: float | None = None,
    space_degree: int = 0,
    pbar_log: bool = False,
) -> dict:
    options = {
        "pde": case,
        "nx": int(nx),
        "ny": int(ny),
        "mesh_type": mesh_type,
        "space_degree": int(space_degree),
        "pbar_log": pbar_log,
        "log_level": log_level,
        "linear_solver_config": FVMLinearSolverConfig(
            backend=backend,
            device=device,
            solver="auto",
        ),
    }
    if rho is not None:
        options["rho"] = float(rho)
    if mu is not None:
        options["mu"] = float(mu)
    return options


def run_simple_cavity(args):
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    case = LidDrivenCavityCase(
        re=args.re,
        lid_velocity=args.lid_velocity,
        rho=args.rho,
        mu=args.mu,
    )
    options = build_simple_options(
        case,
        nx=args.nx,
        ny=args.ny,
        mesh_type=args.mesh_type,
        backend=args.backend,
        device=args.device,
        log_level=args.log_level,
        rho=args.rho,
        mu=args.mu,
        space_degree=args.space_degree,
        pbar_log=args.pbar_log,
    )
    model = NSFVMSimpleModel(options)
    model.solve(
        max_iter=args.max_iter,
        tol=args.tol,
        relax=args.relax,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir("simple", args.re, args.mesh_type, args.nx, args.ny)
    )
    outputs = write_benchmark_outputs(
        model,
        output_dir,
        residuals=model.residuals,
        domain=tuple(case.domain()),
        run_summary={
            "solver": "NSFVMSimpleModel",
            "re": case.re,
            "rho": case.rho,
            "mu": case.mu,
            "nu": case.nu,
            "nx": args.nx,
            "ny": args.ny,
            "mesh_type": args.mesh_type,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "relax": args.relax,
            "coefficient_note": "rho and mu are applied only in the momentum equation",
        },
        fields=tuple(args.output_fields),
    )
    return model, outputs


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lid-driven cavity benchmark with NSFVMSimpleModel"
    )
    parser.add_argument("--re", default=1.0, type=float)
    parser.add_argument("--lid_velocity", default=1.0, type=float)
    parser.add_argument("--rho", default=1.0, type=float)
    parser.add_argument("--mu", default=None, type=float)
    parser.add_argument("--nx", default=32, type=int)
    parser.add_argument("--ny", default=32, type=int)
    parser.add_argument("--mesh_type", default="uniform_quad", type=str)
    parser.add_argument("--space_degree", default=0, type=int)
    parser.add_argument("--max_iter", default=500, type=int)
    parser.add_argument("--tol", default=1.0e-6, type=float)
    parser.add_argument("--relax", default=0.03, type=float)
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--log_level", default="WARNING", type=str)
    parser.add_argument("--pbar_log", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["velocity", "u", "v", "pressure", "speed"],
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    model, outputs = run_simple_cavity(args)
    print(model)
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Primary vortex estimate: {outputs['vortex']}")


if __name__ == "__main__":
    main()
