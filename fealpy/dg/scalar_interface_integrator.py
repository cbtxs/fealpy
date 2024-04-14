
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceIntegrator():
    def __init__(self, q: int, coef: Union[NDArray, float, None]=None) -> None:
        self.q = q
        self.coef = coef

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        q = self.q
        coef = self.coef
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        index = mesh.ds.boundary_face_flag()
        face2cell = mesh.ds.face_to_cell()[index, ...]
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        fn = mesh.entity_measure('face')
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, NF, GD)
        NQ, NF, GD = ps.shape

        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps, index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)
        gphil = np.sum(
            space.grad_basis(ps, index=face2cell[:, 0]) * fn[None, :, None, :],
            axis=-1
        ) # (NQ, NF, ldof)
        gphir = np.sum(
            space.grad_basis(ps, index=face2cell[in_face_flag, 1]) * fn[None, :, None, :],
            axis=-1
        ) # (NQ, in_NF, ldof)

        if coef is None:
            Al = np.einsum('q, qfj, qfj, f -> fj', ws, phil, gphil, optimize=True) # (NF, ldof)
            Ar = np.einsum('q, qfj, qfj, f -> fj', ws, phir, gphir, optimize=True) # (in_NF, ldof)
        elif np.isscalar(coef):
            Al = np.einsum('q, qfj, qfj, f -> fj', ws, phil, gphil, optimize=True) * coef # (NF, ldof)
            Ar = np.einsum('q, qfj, qfj, f -> fj', ws, phir, gphir, optimize=True) * coef # (in_NF, ldof)
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'c'
            elif coef.shape == (NQ, NF):
                coef_subs = 'qc'
            elif coef.shape == (GD, GD):
                coef_subs = 'ij'
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            Al = np.einsum(f'q, {coef_subs}, qfj, qfj, f -> fj', ws, coef, phil, gphil, optimize=True) # (NF, ldof)
            Ar = np.einsum(f'q, {coef_subs}, qfj, qfj, f -> fj', ws, coef, phir, gphir, optimize=True) # (in_NF, ldof)
        else:
            raise ValueError(f'coef type {type(coef)} is not supported.')

        Al += Al.T
        Ar += Ar.T
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[:, 0]], -Al)
            np.add.at(F, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[:, 0]], -Al)
            np.add.at(out, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return out
