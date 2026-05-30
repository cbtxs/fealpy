import argparse
from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMLinearSolverConfig, NSFVMSimpleModel

def main():
    parser = argparse.ArgumentParser(description="SIMPLE-based FVM Navier–Stokes Solver")

    parser.add_argument('--pde', default=1, type=int,
                         help='Navier–Stokes PDE example ID')

    parser.add_argument('--mesh_type', default="uniform_tri", type=str,
                        help='PDE mesh generator variant. Defaults to the PDE model default.')

    parser.add_argument('--mesh_refine', default=1, type=int,
                        help='Uniform refinement levels applied after the PDE default mesh is generated.')

    parser.add_argument('--backend', default='pytorch', type=str,
                        help="Backend: numpy, pytorch, tensorflow, or jax.")

    parser.add_argument('--device', default='cpu', type=str,
                        choices=("cpu", "cuda"),
                        help="Device used by the selected backend.")

    parser.add_argument('--pbar_log', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to show progress bar.')

    parser.add_argument('--log_level',
                        default='INFO', type=str,
                        help='Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.')

    parser.add_argument('--max_iter', default=500, type=int)

    parser.add_argument('--tol', default=1e-6, type=float)

    parser.add_argument('--relax', default=0.01, type=float)

    parser.add_argument('--plot', action='store_true')

    options = vars(parser.parse_args())

    backend = options.pop("backend")
    device = options.pop("device")
    if device != "cpu" and backend != "pytorch":
        raise ValueError("GPU execution is currently supported through pytorch backend.")

    bm.set_backend(backend)
    if backend == "pytorch":
        bm.set_default_device(device)

    solve_options = {
        "max_iter": options.pop("max_iter"),
        "tol": options.pop("tol"),
        "relax": options.pop("relax"),
    }
    plot = options.pop("plot")
    options["linear_solver_config"] = FVMLinearSolverConfig(
        backend=backend,
        device=device,
        solver="auto",
    )

    model = NSFVMSimpleModel(options)
    print(model)
    
    model.solve(**solve_options)
    uerror, verror, perror = model.compute_error()
    print(f"L2 error (u) = {uerror}")
    print(f"L2 error (v) = {verror}")
    print(f"L2 error (p) = {perror}")
    if plot:
        model.plot()
        model.plot_residual()

if __name__ == "__main__":
    main()
