from typing import Optional, Sequence
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian


class ElbowPipeModel:
    def __init__(self):
        super().__init__()

        self.A = self.cross_section_area()
        self.GD = self.geo_dimension()
        
        self.mesh = self.init_mesh()
    
    
    def __str__():
        pass
    
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
    def init_mesh(self):
        """
        """
        pass
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.ones(tuple(shp), dtype=p.dtype)

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        shp = list(p.shape[:-1]) + [2]
        return bm.zeros(tuple(shp), dtype=p.dtype)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        x, y = p[...,0], p[...,1]
        eps = 1e-12
        box = self.domain()  
        flag = (bm.abs(x-box[0])<eps) | (bm.abs(x-box[1])<eps) \
             | (bm.abs(y-box[2])<eps) | (bm.abs(y-box[3])<eps)
        return flag
    
    @cartesian
    def contact_condition(self, fluid_pressure: TensorLike, p: TensorLike) -> TensorLike:
        """
        Apply contact boundary condition for fluid-structure interaction.
        This represents the interaction between fluid and solid boundary (e.g., wall).
        """
        contact_force = fluid_pressure * self.is_dirichlet_boundary(p)  # Apply pressure at the interface
        
        return contact_force