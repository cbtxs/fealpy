from typing import Optional, Dict, Any, List, Tuple
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.mesher import ElbowPipeMesher

import numpy as np

class ElbowPipeModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the solid model for an elbow pipe.
        
        Parameters:
            params: Dictionary of model geometry parameters (unit: m)
        """
        super().__init__()
        
        # Set default parameters if none provided
        if params is None:
            self.params = {
                "D": 0.025,           # Inner diameter 25 mm 
                "bend_angle": 90.0,
                "R_bend_inner": 1.5,
                "L_in_ratio": 5.0,
                "L_out_ratio": 10.0,
                "wall_thickness": 0.005, # Wall thickness 5 mm 
            }
        else:
            self.params = params
            
        # Initialize mesh
        self.GD = self.geo_dimension()
        self.mesh = self.set_mesh()
        
        # Extract boundary information
        self.mesh_data = self.mesher.mesh_data()
        self._extract_boundary_info()
        
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing geometry & boundary conditions."""
        
        D_mm = self.params["D"] 
        wt_mm = self.params["wall_thickness"] 
        L_in_mm = self.params["L_in_ratio"] * D_mm 
        L_out_mm = self.params["L_out_ratio"] * D_mm 
        R_bend_mm = self.params["R_bend_inner"] * D_mm 
        
        # Get boundary information
        dirichlet_nodes = self.get_dirichlet_nodes()
        fsi_triangles, fsi_nodes = self.get_fsi_interface()
        
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry Information ===\n"
        s += f"  Inner Diameter (D)         : {D_mm:.3f} m\n"
        s += f"  Bend Radius (R_bend_inner) : {R_bend_mm:.3f} m\n"
        s += f"  Inlet Length              : { L_in_mm:.3f} m\n" 
        s += f"  Outlet Length             : {L_out_mm :.3f} m\n" 
        s += f"  Wall Thickness             : {wt_mm:.3f} m\n"
        s += f"  Bend Angle                 : {self.params['bend_angle']:.1f} °\n"
        s += f"  Mesh Type                  : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes            : {self.mesh.number_of_nodes()}\n"
        s += f"  Number of Elements         : {self.mesh.number_of_cells()}\n"
        s += f"  Geo Dimension              : {self.GD}\n"
        s += "  [Note: Internal computation uses SI units (m)]\n"
    
        s += "\n  === FSI Boundary Conditions ===\n"
        s += "  Dirichlet BC (Fixed Supports):\n"
        s += f"    - solid_inlet_end (tag {self.solid_inlet_tag}): {len(dirichlet_nodes)} nodes\n"
        s += f"    - solid_outlet_end (tag {self.solid_outlet_tag}): Fixed displacement (0, 0, 0)\n"
        
        s += "  Neumann BC (FSI Coupling Interface):\n"
        s += f"    - fsi_interface (tag {self.fsi_interface_tag}): {fsi_triangles.shape[0]} triangles\n"
        s += ")"
        
        return s
    
    def geo_dimension(self) -> int:
        """Returns the geometric dimension of the domain."""
        return 3
    
    def set_mesh(self):
        """Initialize and return the tetrahedral mesh for solid domain.
        Uses elbow_mesher.init_mesh() to get the volume mesh with region labels.
        """
        self.mesher = ElbowPipeMesher(self.params)
        return self.mesher.init_mesh()
    
    def _extract_boundary_info(self):
        """Extract boundary information from mesh_data for FSI setup.
        """
        # Get physical name to tag mapping
        self.physical_name_to_dimtag = self.mesh_data["physical_name_to_dimtag"]
        
        # Extract physical tags for boundary conditions
        self.solid_inlet_tag = self.physical_name_to_dimtag['solid_inlet_end'][1]
        self.solid_outlet_tag = self.physical_name_to_dimtag['solid_outlet_end'][1]
        self.fsi_interface_tag = self.physical_name_to_dimtag['fsi_interface'][1]
        
        # Extract boundary triangles and markers
        self.boundary_tri = self.mesh_data["boundary_tri"]
        self.boundary_tri_marker = self.mesh_data["boundary_tri_marker"]
        
        # Extract interface triangles (FSI interface)
        self.interface_tri = self.mesh_data["interface_tri"]
        
        # Create mappings for efficient boundary queries
        self._create_boundary_mappings()
        
        # Identify Dirichlet boundary nodes (fixed supports)
        self.dirichlet_nodes = self._identify_dirichlet_nodes()
        all_nodes = self.mesh.entity('node')                 # (NN, GD)
        self.dirichlet_node_coords = all_nodes[self.dirichlet_nodes]  # (Nd, GD)
        
    def _create_boundary_mappings(self):
        """
        Create mappings between boundary markers and triangle indices.
        """
        self.marker_to_tri_indices = {}
        for i, marker in enumerate(self.boundary_tri_marker):
            if marker not in self.marker_to_tri_indices:
                self.marker_to_tri_indices[marker] = []
            self.marker_to_tri_indices[marker].append(i)
            
    def _identify_dirichlet_nodes(self) -> List[int]:
        """
        Identify nodes on Dirichlet boundaries (fixed supports).
        
        Returns:
            List of node indices on Dirichlet boundaries
        """
        dirichlet_nodes = set()
        
        # Get triangles on solid inlet and outlet ends
        inlet_tri_indices = self.marker_to_tri_indices.get(self.solid_inlet_tag, [])
        outlet_tri_indices = self.marker_to_tri_indices.get(self.solid_outlet_tag, [])
        
        # Collect all nodes on these boundaries
        for idx in inlet_tri_indices:
            dirichlet_nodes.update(self.boundary_tri[idx])
        for idx in outlet_tri_indices:
            dirichlet_nodes.update(self.boundary_tri[idx])
        
        return list(dirichlet_nodes)
    
    def get_dirichlet_nodes(self) -> List[int]:
        """
        Get all nodes on Dirichlet boundaries (fixed supports).
        
        Returns:
            List of node indices that should have fixed displacement
        """
        return self.dirichlet_nodes
    
    def get_fsi_interface(self) -> Tuple[TensorLike, TensorLike]:
        """Get the FSI interface information.
        
        Returns:
            Tuple of (interface triangles, interface node indices)
        """
        # Get all unique nodes on the FSI interface
        if hasattr(self.interface_tri, 'flatten'):
            # For numpy arrays and similar
            fsi_nodes = list(set(self.interface_tri.flatten()))
        else:
            # For lists or other iterables
            fsi_nodes = list(set(node_idx for tri in self.interface_tri for node_idx in tri))
        return self.interface_tri, fsi_nodes
    
    @cartesian
    def body_force(self, p: TensorLike) -> TensorLike:
        """Calculate the body force, such as gravity.
        Parameters:
            p: Coordinates of points inside the element, shape (N, 3), where N is the number of points,
           and 3 represents the x, y, and z coordinates.
    
        Returns:
            The body force density, shape (N, 3), representing the force [Fx, Fy, Fz] at each point.
            For example, under gravity, Fz = -rho * g, 
            where rho is the material density, and g is the gravitational acceleration.
        """
        rho = 7800  
        g = 9.8     
        force = bm.zeros_like(p)
        force[..., 2] = -rho * g  # Fz = -rho * g
        # force[..., 2] = 1
        return force 

    @cartesian
    def displacement_bc(self, p: TensorLike) -> TensorLike:
        """Return displacement values on Dirichlet boundaries.
        According to FSI theory, Dirichlet boundaries have fixed displacement (0, 0, 0).

        Parameters:
            p: Points on Dirichlet boundary

        Returns:
            Displacement values (Ux, Uy, Uz) = (0, 0, 0) for fixed boundaries.
        """
        
        return bm.zeros_like(p)
    
    @cartesian
    def is_displacement_boundary(self, p):
        """Determine if the given points lie on a Dirichlet boundary.
        
        Parameters:
            p: Coordinates of points, shape (N, GD), where N is the number of points,
                and GD represents the coordinate dimension (usually 3).
    
        Returns:
            A boolean array indicating whether each point lies on the Dirichlet boundary.
        """
        # p: (N, GD)
        # dirichlet_node_coords: (Nd, GD)
        diff = p[:, None, :] - self.dirichlet_node_coords[None, :, :]
        dist = bm.linalg.norm(diff, axis=-1)         # (N, Nd)
        min_dist = bm.min(dist, axis=1)              # (N,)
        tol = 1e-6  
        return min_dist < tol