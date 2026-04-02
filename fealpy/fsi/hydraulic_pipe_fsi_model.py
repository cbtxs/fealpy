from typing import Any, Optional, Union

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

from fealpy.csm.model.linear_elasticity import LinearElasticityPDEDataT
from fealpy.csm.model.model_manager import CSMModelManager

from .coupling_interface import CouplingInetrface
from .coupling_fsi_algorithm import CouplingFSIAlgorithm


class HydraulicPipeFSIModel(ComputationalModel):
    def __init__(self, options):
        self.options = options
        super().__init__(
                pbar_log=options['pbar_log'],
                log_level=options['log_level'])
        
        self.set_fluid_model(options['fluid_model'])
        self.set_solid_model(options['solid_model'])
        
        mesh = self.set_solid_model.init_mesh()
        self.set_mesh(mesh)
        self.set_space_degree(options['space_degree'])
        
        self.GD = self.pde.geo_dimension()

        self.E = options['E']
        self.nu = options['nu']
        self.rho = options['rho']
        
        self.set_space()
        self.set_material()
        
    def set_fluid_model(self, pde: Union[LinearElasticityPDEDataT, int] = 4) -> None:
        """Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            if pde not in [4, 5]:
                raise ValueError(f"Invalid PDE ID: {pde}. Must be 4, 5.")
            self.pde = CSMModelManager('linear_elasticity').get_example(pde)
        else:
            self.pde = pde
    
    def set_solid_model(self, pde: Union[LinearElasticityPDEDataT, int] = 4) -> None:
        """Set PDE parameters and update model.

        Parameters:
            pde: PDE data manager or int.
        """
        if isinstance(pde, int):
            if pde not in [4, 5]:
                raise ValueError(f"Invalid PDE ID: {pde}. Must be 4, 5.")
            self.pde = CSMModelManager('linear_elasticity').get_example(pde)
        else:
            self.pde = pde
            
    def set_mesh(self, mesh: Mesh) -> None:
        """Set the mesh.

        Parameters:
            mesh (Mesh): The mesh object.
        """
        self.mesh = mesh
        # self.logger.info(self.mesh)
        
    def set_space_degree(self, p: int) -> None:
        self.p = p
        
    def set_space(self):
        """Initialize the finite element space."""
        mesh = self.mesh
        p = self.p
        
        scalar_space = LagrangeFESpace(mesh, p=p, ctype='C')
        self.space = TensorFunctionSpace(scalar_space, shape=(-1, self.GD))

    def set_material(self) -> None:
        """Set material properties.

        Parameters:
            E (float): Young's modulus.
            nu (float): Poisson's ratio.
            rho(float): density.
        """
        self.material = LinearElasticMaterial(name='hydraulic_pipe_Material',
                                    elastic_modulus=self.E,
                                    poisson_ratio=self.nu,
                                    density = self.rho)
        #self.logger.info(self.material)
    