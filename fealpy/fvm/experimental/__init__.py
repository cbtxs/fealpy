"""Experimental finite-volume model routes.

This subpackage contains routes that are useful for development and comparison
but are not part of the stable engineering-oriented FVM solver layer.
"""

from .ns_fvm_rc_model import NSFVMRCModel
from .ns_fvm_staggered_model import NSFVMStaggeredModel
from .ns_fvm_staggered_piso_model import NSFVMStaggeredPISOModel
from .ns_fvm_staggered_simple_model import NSFVMStaggeredSimpleModel
from .stokes_fvm_rc_model import StokesFVMRCModel
from .stokes_fvm_staggered_model import StokesFVMStaggeredModel
from .stokes_fvm_staggered_simple_model import StokesFVMStaggeredSimpleModel
from .staggered_mesh_manager import StaggeredMeshManager

__all__ = [
    "NSFVMRCModel",
    "NSFVMStaggeredModel",
    "NSFVMStaggeredPISOModel",
    "NSFVMStaggeredSimpleModel",
    "StokesFVMRCModel",
    "StokesFVMStaggeredModel",
    "StokesFVMStaggeredSimpleModel",
    "StaggeredMeshManager",
]
