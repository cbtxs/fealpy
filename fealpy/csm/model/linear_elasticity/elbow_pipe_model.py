from typing import Optional, Sequence
from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian


class ElbowPipeModel:
    def __init__(self):
        super().__init__()

        self.D = 25e-3  # Inner diameter in meters (25 mm)
        self.wall_thickness = 5e-3  # Wall thickness in meters (5 mm)
        self.R_inner = 1.5 * self.D  # Bend radius (inner) in meters
        self.R_outer = self.R_inner + self.wall_thickness  # Outer radius considering wall thickness (5 mm)
        self.L_inlet = 5 * self.D  # Inlet length (5D) in meters
        self.L_outlet = 10 * self.D  # Outlet length (10D) in meters
        
        self.A = self.cross_section_area()
        self.GD = self.geo_dimension()
        self.mesh = self.init_mesh()
    
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing geometry & boundary conditions."""
        
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry Information ===\n"
        s += f"  Inner Diameter (D)       : {self.D * 1e3:.2f} mm\n"  # Inner diameter in mm
        s += f"  Outer Radius (R_outer)   : {self.R_outer * 1e3:.2f} mm\n"  # Outer radius in mm
        s += f"  Inlet Length (L_inlet)   : {self.L_inlet * 1e3:.2f} mm\n"  # Inlet length in mm
        s += f"  Outlet Length (L_outlet) : {self.L_outlet * 1e3:.2f} mm\n"  # Outlet length in mm
        s += f"  Wall Thickness           : {self.wall_thickness * 1e3:.2f} mm\n"  # Wall thickness in mm
        s += f"  Bend Radius (R_inner)   : {self.R_inner * 1e3:.2f} mm\n"  # Bend radius in mm
        s += f"  Mesh Type               : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes         : {self.mesh.number_of_nodes() if self.mesh else 'N/A'}\n"
        s += f"  Number of Elements      : {self.mesh.number_of_cells() if self.mesh else 'N/A'}\n"
        s += f"  Geo Dimension           : {self.geo_dimension()}\n"
        s += "\n  === Boundary Conditions ===\n"
        s += ")"
        
        return s
    
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
    def cross_section_area(self) -> float:
        """Calculate the cross-sectional area of the pipe."""
        radius_inner = self.R_inner
        radius_outer = self.R_outer
        area = bm.pi * (radius_outer ** 2 - radius_inner ** 2)
        return area
    
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
    
    @cartesian
    def contact_condition(self, fluid_pressure: TensorLike, p: TensorLike) -> TensorLike:
        """
        Apply contact boundary condition for fluid-structure interaction.
        This represents the interaction between fluid and solid boundary (e.g., wall).
        """
        pass