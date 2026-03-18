from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm, BilinearForm
from fealpy.fem import (ScalarConvectionIntegrator, ScalarDiffusionIntegrator,
                     SourceIntegrator, ScalarMassIntegrator)
from fealpy.decorator import barycentric

from ..iterative_method import IterativeMethod 

class StationarySpecificDissipationRatePicard(IterativeMethod):
    """Stationary Specific Dissipation Rate Picard Iterative Method"""
    def BForm(self):
        omegasapce = self.pspace
        q = self.q

        A = BilinearForm(omegasapce)
        self.omega_BC = ScalarConvectionIntegrator(q=q)
        self.omega_BD = ScalarDiffusionIntegrator(q=q)
        self.omega_BM = ScalarMassIntegrator(q=q)
        self.omega_BCD = ScalarConvectionIntegrator(q=q)

        A.add_integrator(self.BC)
        A.add_integrator(self.BD)   
        A.add_integrator(self.BM)
        A.add_integrator(self.BCD)

        return A

    def LForm(self):
        omegasapce = self.pspace
        q = self.q

        L = LinearForm(omegasapce)
        self.omega_source = SourceIntegrator(q=q)
        L.add_integrator(self.S)

        return L
    
    def update(self, u1, k1, omega0):
        equation = self.equation 
        cc = equation.coef_convection
        cds = equation.coef_dissipation
        cd = equation.coef_diffusion
        ccd = equation.coef_cross_diffusion
        cp = equation.coef_production

        ## BilinearForm
        @barycentric
        def omega_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.omega_BC.coef = omega_BC_coef

        @barycentric
        def omega_BM_coef(bcs, index):
            cdscoef = cds(bcs, index)[bm.newaxis, bm.newaxis] if callable(cds) else cds
            return cdscoef * omega0(bcs, index)
        self.omega_BM.coef = omega_BM_coef

        self.omega_BD.coef = cd

        @barycentric
        def omega_BCD_coef(bcs, index):
            ccdcoef = ccd(bcs, index)[bm.newaxis, bm.newaxis] if callable(ccd) else ccd
            reciprocal_omega0 = 1/omega0
            ccdcoef *= reciprocal_omega0(bcs, index)
            ccdcoef *= k1.grad_value(bcs, index)
            return ccdcoef
        self.omega_BCD.coef = omega_BCD_coef

        ## LinearForm
        self.omega_source = cp




