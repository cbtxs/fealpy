import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry, reconstruct_face_gradient
from fealpy.mesh import TriangleMesh


def test_face_gradient_rejects_full_face_wise_patch_sn_grad():
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0][:2]
    cell_gradient = bm.ones((mesh.number_of_cells(), mesh.geo_dimension()))

    with pytest.raises(ValueError, match="boundary_sn_grad must have shape"):
        reconstruct_face_gradient(
            mesh,
            cell_gradient,
            geometry=geometry,
            boundary_faces=boundary_faces,
            boundary_sn_grad=bm.ones(mesh.number_of_faces()),
        )


def test_face_gradient_rejects_broadcast_patch_dirichlet_value():
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    geometry = FVMGeometry(mesh)
    boundary_faces = bm.nonzero(geometry.is_boundary)[0][:2]
    cell_gradient = bm.ones((mesh.number_of_cells(), mesh.geo_dimension()))
    cell_values = bm.ones(mesh.number_of_cells())

    with pytest.raises(ValueError, match="dirichlet_values must have shape"):
        reconstruct_face_gradient(
            mesh,
            cell_gradient,
            geometry=geometry,
            cell_values=cell_values,
            dirichlet_faces=boundary_faces,
            dirichlet_values=bm.array(1.0),
        )
