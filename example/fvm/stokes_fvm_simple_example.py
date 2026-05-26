import argparse
from fealpy.backend import backend_manager as bm
from fealpy.fvm import StokesFVMSimpleModel

def main():
    parser = argparse.ArgumentParser(description="SIMPLE-based FVM Stokes Solver")

    parser.add_argument('--pde', default=1, type=int,
                        help='Stokes PDE example ID')

    parser.add_argument('--mesh_type', default="uniform_tri", type=str,
                        help='PDE mesh generator variant. Defaults to the PDE model default.')

    parser.add_argument('--mesh_refine', default=0, type=int,
                        help='Uniform refinement levels applied after the PDE default mesh is generated.')

    parser.add_argument('--backend', default='numpy', type=str,
                        help="Backend: numpy, torch, tensorflow, or jax.")

    parser.add_argument('--pbar_log', default=True, action=argparse.BooleanOptionalAction,
                        help='Whether to show progress bar.')

    parser.add_argument('--log_level',
                        default='INFO', type=str,
                        help='Log level: DEBUG, INFO, WARNING, ERROR, or CRITICAL.')

    parser.add_argument('--max_iter', default=400, type=int)

    parser.add_argument('--tol', default=1e-5, type=float)

    parser.add_argument('--relax', default=0.32, type=float)

    parser.add_argument('--plot', action='store_true')

    options = vars(parser.parse_args())

    bm.set_backend(options.pop("backend"))

    solve_options = {
        "max_iter": options.pop("max_iter"),
        "tol": options.pop("tol"),
        "relax": options.pop("relax"),
    }
    plot = options.pop("plot")

    model = StokesFVMSimpleModel(options)
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
