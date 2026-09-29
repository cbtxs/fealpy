import argparse


parser = argparse.ArgumentParser(
    description="Hu-Zhang FEM for a two-dimensional linear elasticity problem"
)
parser.add_argument(
    "--backend",
    default="numpy",
    help="Tensor backend, default is numpy",
)
parser.add_argument(
    "--pde",
    default=3,
    type=int,
    help="Linear elasticity PDE example, default is 3",
)
parser.add_argument(
    "--init_mesh",
    default="uniform_tri",
    help="Initial mesh generator, default is uniform_tri",
)
parser.add_argument(
    "--space_degree",
    default=3,
    type=int,
    help="Polynomial degree p of the Hu-Zhang stress space, default is 3",
)
parser.add_argument(
    "--maxit",
    default=4,
    type=int,
    help="Number of uniform refinement levels, default is 4",
)
parser.add_argument(
    "--solver",
    default="scipy",
    help="Sparse direct solver, default is scipy",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="Plot the convergence-rate curves",
)
args = parser.parse_args()


from fealpy.backend import backend_manager as bm

bm.set_backend(args.backend)

from fealpy.decorator import barycentric
from fealpy.fem import BilinearForm, BlockForm, LinearForm
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem.huzhang_mix_integrator import HuZhangMixIntegrator
from fealpy.fem.huzhang_stress_integrator import HuZhangStressIntegrator
from fealpy.functionspace import HuZhangFESpace
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.model import PDEModelManager
from fealpy.solver import spsolve
from fealpy.tools.show import show_error_table, showmultirate


# PDE data and the initial triangular mesh
pde = PDEModelManager("linear_elasticity").get_example(args.pde)
mesh = pde.init_mesh[args.init_mesh]()

p = args.space_degree
lambda0, lambda1 = pde.stress_matrix_coefficient()

error_type = [
    r"$|| \boldsymbol{\sigma} - \boldsymbol{\sigma}_h||_{L_2}$",
    r"$|| \boldsymbol{u} - \boldsymbol{u}_h||_{L_2}$",
]
error_matrix = bm.zeros((2, args.maxit), dtype=bm.float64)
h = bm.zeros(args.maxit, dtype=bm.float64)


for i in range(args.maxit):
    GD = mesh.geo_dimension()

    # Hu-Zhang stress space and discontinuous displacement space
    space_sigma = HuZhangFESpace(mesh, p=p)
    scalar_space_u = LagrangeFESpace(mesh, p=p - 1, ctype="D")
    space_u = TensorFunctionSpace(
        scalar_space=scalar_space_u,
        shape=(-1, GD),
    )

    # (A sigma_h, tau_h)
    stress_form = BilinearForm(space_sigma)
    stress_form.add_integrator(
        HuZhangStressIntegrator(lambda0=lambda0, lambda1=lambda1)
    )

    # (div tau_h, u_h)
    mixed_form = BilinearForm((space_u, space_sigma))
    mixed_form.add_integrator(HuZhangMixIntegrator())

    # [M B; B^T 0]
    system = BlockForm(
        [
            [stress_form, mixed_form],
            [mixed_form.T, None],
        ]
    )
    A = system.assembly()

    # (f, v_h)
    source_form = LinearForm(space_u)
    source_form.add_integrator(
        VectorSourceIntegrator(source=pde.body_force)
    )
    source = source_form.assembly()

    gdof_sigma = space_sigma.number_of_global_dofs()
    F = bm.zeros(A.shape[0], dtype=A.dtype)
    F[gdof_sigma:] = -source

    # Solve the saddle-point system and recover the finite element functions
    coefficient = spsolve(A, F, solver=args.solver)

    sigma_h = space_sigma.function()
    u_h = space_u.function()
    sigma_h[:] = coefficient[:gdof_sigma]
    u_h[:] = coefficient[gdof_sigma:]

    # The unified mesh integration interface passes simplex coordinates as a
    # one-element tuple.  Keep this compatibility local to this example.
    @barycentric
    def displacement_value(bcs):
        if isinstance(bcs, tuple):
            bcs = bcs[0]
        return u_h(bcs)

    error_matrix[0, i] = mesh.error(sigma_h, pde.stress)
    error_matrix[1, i] = mesh.error(displacement_value, pde.displacement)
    h[i] = bm.max(mesh.entity_measure("cell")) ** (1 / GD)

    residual = bm.linalg.norm(A @ coefficient - F, ord=2)
    print(
        f"level={i}, cells={mesh.number_of_cells()}, "
        f"stress_dofs={gdof_sigma}, "
        f"displacement_dofs={space_u.number_of_global_dofs()}, "
        f"residual={float(residual):.3e}"
    )

    if i < args.maxit - 1:
        mesh.uniform_refine()


show_error_table(h, error_type, error_matrix)

if args.plot and args.maxit > 1:
    import matplotlib.pyplot as plt

    showmultirate(
        plt,
        0,
        h,
        error_matrix,
        error_type,
        propsize=20,
    )
    plt.show()
