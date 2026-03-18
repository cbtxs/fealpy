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
        self.k_BC = ScalarConvectionIntegrator(q=q)
        self.k_BD = ScalarDiffusionIntegrator(q=q)
        self.k_BM = ScalarMassIntegrator(q=q)

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
    
    def update(self, u1, omega0):
        equation = self.equation
        cc = equation.coef_convection
        cd = equation.coef_diffusion
        cr = equation.coef_reaction
        cp = equation.coef_production
        
        ## BilinearForm
        @barycentric
        def k_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.k_BC.coef = k_BC_coef

        self.k_BD.coef = cd

        @barycentric
        def k_BM_coef(bcs, index):
            crcoef = cr(bcs, index)[..., bm.newaxis] if callable(cr) else cr
            return crcoef * omega0(bcs, index)
        self.k_BM.coef = k_BM_coef

        ## LinearForm
        self.k_source.source = cp

