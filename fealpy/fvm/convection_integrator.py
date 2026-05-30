"""Finite-volume convection integrator for face-velocity fluxes."""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator.variantmethod import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

class ConvectionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble a face-interpolation convection operator.

    ``coef`` is interpreted as the face velocity field.  The face mass flux is
    ``coef_f · S_f`` and multiplies the owner-neighbour face interpolation
    stencil.  Upwinding, limiters, and nonlinear iteration control are outside
    this low-level integrator.
    """

    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 interpolation: str="average",
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.interpolation = self._validate_interpolation(interpolation)
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @staticmethod
    def _validate_interpolation(interpolation: str) -> str:
        if interpolation not in {"average", "linear"}:
            raise ValueError(
                "interpolation must be 'average' or 'linear'."
            )
        return interpolation
        
    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.edge_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ConvectionIntegrator only supports spaces on "
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
        n = mesh.face_unit_normal(index=index)
        facemeasure = mesh.entity_measure('face', index=index)
        Sf = facemeasure[:, None] * n  # (NE, 2)
        q = self.q
        qf = mesh.quadrature_formula(q, 'face') 
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return Sf, index, bcs, phi

    def _owner_weight(self, space: _FS, Sf: TensorLike) -> TensorLike:
        if self.interpolation == "average":
            return 0.5 * bm.ones_like(Sf[:, 0])

        mesh = getattr(space, "mesh")
        index = self.index
        e2c = mesh.edge_to_cell(index=index)[:, :2]
        owner = e2c[:, 0]
        neighbour = e2c[:, 1]
        face_centers = mesh.entity_barycenter("face", index=index)
        cell_centers = mesh.entity_barycenter("cell")
        own = bm.abs(
            bm.einsum("ij,ij->i", Sf, face_centers - cell_centers[owner])
        )
        nei = bm.abs(
            bm.einsum("ij,ij->i", Sf, cell_centers[neighbour] - face_centers)
        )
        total = own + nei
        weight = bm.where(total > 0.0, nei / total, 0.5)
        return bm.where(owner == neighbour, 1.0, weight)
    
    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        Sf, _, _, phi = self.fetch(space)
        D = phi.shape[-1]
        eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
        owner_weight = self._owner_weight(space, Sf)
        neighbour_weight = 1.0 - owner_weight
        direction_matrix = bm.stack(
            [
                bm.stack([owner_weight, neighbour_weight], axis=-1),
                bm.stack([-owner_weight, -neighbour_weight], axis=-1),
            ],
            axis=1,
        )
        base_matrix = bm.einsum(
            "ij,fpq->fipjq", eye_D, direction_matrix
        ).reshape(-1, 2 * D, 2 * D)
        if coef is None:
            coef = bm.stack([bm.ones_like(Sf[:,0]), bm.zeros_like(Sf[:,0])], axis=1)
        integrator  = bm.einsum('ij,ij->i', Sf, coef)
        result = integrator[:, None, None] * base_matrix

        return result
