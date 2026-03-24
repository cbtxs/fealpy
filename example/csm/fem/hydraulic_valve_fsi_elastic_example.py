import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite element analysis for fluid-structure interaction (FSI) in hydraulic valve systems,
        with linear elasticity for structural deformation in steady-state conditions.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy")

parser.add_argument('--pde',
                    default=4, type=int,
                    help="index of the linear elasticity  model, default is 4")

parser.add_argument('--mesh_type',
                    default='uniform_tet', type=str,
                    help="Type of mesh, default is uniform_tet")

parser.add_argument('--space_degree',
        default=1, type=int,
        help="Degree of Lagrange finite element space, default is 1")

parser.add_argument('--E', 
                    default=70.0, type=float, 
                    help="Young's modulus (E) in GPa for the elastic material")

parser.add_argument('--nu',
                    default=0.3, type=float,
                    help="Poisson's ratio (nu) for the elastic material, default is 0.3")

# parser.add_argument('--neigen',
#         default=6, type=int,
#         help='Number of eigenvalues to compute, default is 6')

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.fem.linear_elasticity_lfem_model import  LinearElasticityLFEMModel
model = LinearElasticityLFEMModel()

model.set_pde(7)

#model.set_init_mesh(meshtype='uniform_tri')
model.set_init_mesh(meshtype='custom_hex')
# model.set_init_mesh(meshtype='uniform_tet', nx=10, ny=10, nz=10)


model.set_space_degree(p=1)

model.run['uniform_refine']()
print("-----------------------------")
