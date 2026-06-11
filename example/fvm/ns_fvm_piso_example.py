import argparse

from fealpy.backend import backend_manager as bm
from fealpy.fvm import NSFVMPISOModel


def main():
    parser = argparse.ArgumentParser(
        description="PISO-based FVM Navier-Stokes solver with Rhie-Chow interpolation")

    parser.add_argument("--pde", default=3, type=int,
                        help="Navier-Stokes PDE example ID")

    parser.add_argument("--nx", default=20, type=int,
                        help="Grid divisions in x")

    parser.add_argument("--ny", default=20, type=int,
                        help="Grid divisions in y")

    parser.add_argument("--nz", default=None, type=int,
                        help="Grid divisions in z for 3D examples")

    parser.add_argument("--mesh_type", default="uniform_quad", type=str,
                        help="Mesh type, e.g. uniform_quad or uniform_tri")

    parser.add_argument("--nt", default=40, type=int,
                        help="Number of time steps")

    parser.add_argument("--n_correctors", default=2, type=int,
                        help="Number of PISO pressure correctors per time step")

    parser.add_argument("--momentum_nonorthogonal_max_iter", default=None, type=int,
                        help="Max explicit non-orthogonal corrections for momentum diffusion")

    parser.add_argument("--pressure_nonorthogonal_max_iter", default=None, type=int,
                        help="Max explicit non-orthogonal corrections for pressure correction")

    parser.add_argument("--duration", nargs=2, default=(0.0, 1.0), type=float,
                        help="Start and end time")

    parser.add_argument("--space_degree", default=0, type=int,
                        help="Space degree")

    parser.add_argument("--backend", default="numpy", type=str,
                        help="Backend: numpy, torch, tensorflow, or jax.")

    parser.add_argument("--linear_solver", default="auto", type=str,
                        help="Sparse linear solver, e.g. auto, mumps, or scipy.")

    parser.add_argument("--pbar_log", default=True, type=bool,
                        help="Whether to show progress bar, default is True")

    parser.add_argument("--log_level", default="INFO", type=str,
                        help="Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL")

    parser.add_argument("--plot", action="store_true")

    options = vars(parser.parse_args())

    bm.set_backend(options["backend"])

    model = NSFVMPISOModel(options)
    print(model)

    model.solve()
    errors = model.compute_error()
    velocity_names = ("u", "v", "w")
    for name, error in zip(velocity_names, errors[:-1]):
        print(f"L2 error ({name}) = {error}")
    print(f"L2 error (p) = {errors[-1]}")
    # model.plot()
    if options["plot"]:
        model.plot()


if __name__ == "__main__":
    main()
