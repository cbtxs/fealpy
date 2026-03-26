from typing import Optional, Sequence
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian


class HydraulicPipeModel:
    def __init__(self):
        super().__init__()

        self.A = self.cross_section_area()
        self.GD = self.geo_dimension()
        
        self.mesh = self.init_mesh()
        
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
    def init_mesh(self):
        """
        """
        pass
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        pass

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        pass

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
       pass