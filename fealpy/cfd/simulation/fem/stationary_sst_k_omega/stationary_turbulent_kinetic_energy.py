from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm
from fealpy.fem import (ScalarConvectionIntegrator, ScalarDiffusionIntegrator,
                     SourceIntegrator, ScalarMassIntegrator)
from fealpy.decorator import barycentric

from ..iterative_method import IterativeMethod 


class StationaryTurbulentKineticEnergyPicard(IterativeMethod):
    """
    Picard iteration method for stationary turbulent kinetic energy equation.
    """

    def BForm(self):
        kspace = self.pspace
        q = self.q

        A = BilinearForm(kspace)
        self.k_BC = ScalarConvectionIntegrator
        self.k_BD = ScalarDiffusionIntegrator
        self.k_BM = ScalarMassIntegrator

        A.add_integrator(self.k_BC)
        A.add_integrator(self.k_BD)
        A.add_integrator(self.k_BM)

        return A
    
    def LForm(self):
        kspace = self.pspace
        q = self.q

        L = LinearForm(kspace)
        self.k_source = SourceIntegrator(q=q)
        L.add_integrator(self.k_source)
        
        return L


