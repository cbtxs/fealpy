from typing import Any, Optional, Union

from fealpy.typing import TensorLike
from fealpy.backend import bm
from fealpy.model import ComputationalModel

from fealpy.mesh import Mesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material import LinearElasticMaterial
from fealpy.fem import BilinearForm, LinearForm
from fealpy.fem import LinearElasticityIntegrator, VectorSourceIntegrator
from fealpy.fem import ScalarMassIntegrator as MassIntegrator
from fealpy.fem import DirichletBC
from fealpy.solver import spsolve

from ..model.linear_elasticity import LinearElasticityPDEDataT
from ..model.model_manager import CSMModelManager


class HydraulicPipeLinearElasticModel(ComputationalModel):
    def __init__(self, options):
        self.options = options
        super().__init__(
                pbar_log=options['pbar_log'],
                log_level=options['log_level'])
        
        self.set_pde(options['pde'])
        # mesh = self.pde.init_mesh()
        # self.set_mesh(mesh)
        # self.set_space_degree(options['space_degree'])
        
        self.GD = self.pde.geo_dimension()

        self.E = options['E']
        self.nu = options['nu']
        self.rho = options['rho']
        
        # self.set_space()
        self.set_material()
        
    def set_pde(self, pde: Union[LinearElasticityPDEDataT, int] = 4) -> None:
        """Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            self.pde = CSMModelManager("linear_elasticity").get_example(pde)
        else:
            self.pde = pde
        #self.logger.info(self.pde.__str__())
        
    # def set_mesh(self, mesh: Mesh) -> None:
    #     """Set the mesh.

    #     Parameters:
    #         mesh (Mesh): The mesh object.
    #     """
    #     self.mesh = mesh
        
    # def set_space_degree(self, p: int) -> None:
    #     self.p = p
        
    # def set_space(self):
    #     """Initialize the finite element space."""
    #     mesh = self.mesh
    #     p = self.p
        
    #     scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
    #     self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD))

    def set_material(self) -> None:
        """Set material properties.

        Parameters:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
        """
        self.material = LinearElasticMaterial(name='single_pipe_Material',
                                    elastic_modulus=self.E,
                                    poisson_ratio=self.nu,
                                    density = self.rho)
        self.logger.info(self.material)
    
    # def linear_system(self):
    #     self.uh = self.space.function()

    #     bform = BilinearForm(self.tspace)
    #     LEI = LinearElasticityIntegrator(
    #                             material=self.material, q=self.p+3, method=None
    #                         )
    #     bform.add_integrator(LEI)

    #     lform = LinearForm(self.space)
    #     SI = VectorSourceIntegrator(self.pde.body_force)
    #     lform.add_integrator(SI)

    #     A = bform.assembly()
    #     F = lform.assembly()

    #     return A, F
    
    # def apply_bc(self, A, F):
    #     if hasattr(self.pde, 'displacement_bc'):
    #         A, F = DirichletBC(
    #                 self.space,
    #                 gd=self.pde.displacement_bc,
    #                 threshold=self.pde.is_displacement_boundary).apply(A, F)
    #     else:
    #         pass
    #     return A, F

    # def solve(self, A, F):
    #     """
    #     Solve the linear system and return the solution.

    #     Returns:
    #         uh: Solution vector.
    #     """
    #     uh = spsolve(A, F, solver='scipy')
    #     # self.logger.info(f"Solution : {uh}")
    #     return uh