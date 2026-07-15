import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMLinearSolverConfig, StokesFVMSimpleModel


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solve a Stokes manufactured solution with SIMPLE."
    )
    parser.add_argument(
        "--pde",
        default=1,
        type=int,
        help="Stokes PDE example ID.",
    )
    parser.add_argument(
        "--mesh-type",
        default=None,
        help="Override the mesh type supplied by the PDE model.",
    )
    parser.add_argument(
        "--mesh-refine",
        default=3,
        type=int,
        help="Uniform refinement levels applied to the PDE mesh.",
    )
    parser.add_argument(
        "--backend",
        default="numpy",
        help="FEALPy backend, such as numpy or pytorch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used by the selected backend.",
    )
    parser.add_argument(
        "--linear-solver",
        default="auto",
        choices=("auto", "mumps", "scipy", "cupy"),
        help="Sparse linear solver backend.",
    )
    parser.add_argument("--max-iter", default=3000, type=int)
    parser.add_argument("--tol", default=1.0e-6, type=float)
    parser.add_argument(
        "--relax",
        default=0.3,
        type=float,
        help="Pressure-correction relaxation factor.",
    )
    parser.add_argument(
        "--momentum-relaxation",
        default=0.9,
        type=float,
        help="Momentum-equation under-relaxation factor.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-iteration SIMPLE diagnostics.",
    )
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    if args.device != "cpu" and args.backend != "pytorch":
        raise ValueError("GPU execution is currently supported through pytorch backend.")
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    model_options = {
        "pde": args.pde,
        "mesh_refine": args.mesh_refine,
        "momentum_equation_relaxation": args.momentum_relaxation,
        "log_level": "WARNING" if args.quiet else "INFO",
        "linear_solver_config": FVMLinearSolverConfig(
            backend=args.backend,
            device=args.device,
            solver=args.linear_solver,
        ),
    }
    if args.mesh_type is not None:
        model_options["mesh_type"] = args.mesh_type

    model = StokesFVMSimpleModel(model_options)
    print(model)

    model.solve(max_iter=args.max_iter, tol=args.tol, relax=args.relax)
    status = "converged" if model.converged else "not converged"
    print(
        f"SIMPLE {status} after {model.outer_iterations} iterations "
        f"({model.termination_reason})."
    )

    errors = model.compute_error()
    for name, error in zip(("u", "v", "w"), errors[:-1]):
        print(f"L2 error ({name}) = {error}")
    print(f"L2 error (p) = {errors[-1]}")

    if args.plot:
        model.plot()
        model.plot_residual()


if __name__ == "__main__":
    main()
