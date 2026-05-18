from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .vector_decomposition import VectorDecomposition
from .gradient_reconstruct import GradientReconstruct
from .nonorthogonal_geometry import NonOrthogonalGeometry

class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    def __init__(self, uh=None, grad_f=None, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 face_flux_correction=None,
                 correction_vector=None,
                 geometry=None,
                 correction_method: str="legacy",
                 limit_coeff: float=0.5,
                 limiter_small: float=1.0e-30,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.uh = uh
        self.grad_f = grad_f
        self.coef = coef
        self.face_flux_correction = face_flux_correction
        self.correction_vector = correction_vector
        self.geometry = geometry
        self.correction_method = correction_method
        self.limit_coeff = limit_coeff
        self.limiter_small = limiter_small
        self.q = 2 if q is None else q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
        Tf = VectorDecomposition(mesh).tangential_vector_calculation() # (NE, 2)
        edge_to_cell = mesh.edge_to_cell(index=index)[:,:2]
        NC = mesh.number_of_cells()
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights() 
        phi = space.basis(bcs, index=index) 
        return Tf,edge_to_cell,NC,phi
        

    def _coefficient_on_faces(self, space: _FS, face_flux: TensorLike) -> TensorLike:
        if self.coef is None:
            return bm.ones_like(face_flux[:, 0] if face_flux.ndim == 2 else face_flux, dtype=space.ftype)
        elif type(self.coef) in [int, float]:
            return bm.full_like(
                face_flux[:, 0] if face_flux.ndim == 2 else face_flux,
                fill_value=self.coef,
                dtype=space.ftype
            )
        return self.coef

    def _select_correction_vector(self, space: _FS, Tf: TensorLike) -> TensorLike:
        if self.correction_vector is not None:
            return bm.array(self.correction_vector, dtype=space.ftype)
        geometry = self.geometry
        if geometry is None and self.correction_method != "legacy":
            geometry = NonOrthogonalGeometry(getattr(space, "mesh"))
        if self.correction_method == "legacy":
            return geometry.legacy_tangential_vector() if geometry is not None else Tf
        if self.correction_method == "orthogonal":
            return geometry.zero_correction_vector()
        if self.correction_method == "openfoam_stabilized":
            return geometry.openfoam_correction_vector()
        if self.correction_method == "limited":
            return geometry.openfoam_correction_vector()
        raise ValueError(f"Unsupported correction_method: {self.correction_method!r}")

    def _apply_limiter(
            self, space: _FS, edge_to_cell: TensorLike, face_flux: TensorLike
    ) -> TensorLike:
        if self.correction_method != "limited":
            return face_flux
        if face_flux.ndim != 1:
            raise ValueError("limited correction currently supports scalar face flux only.")
        if self.uh is None:
            raise ValueError("uh is required when correction_method='limited'.")

        mesh = getattr(space, "mesh")
        geometry = self.geometry if self.geometry is not None else NonOrthogonalGeometry(mesh)
        uh = bm.array(self.uh, dtype=space.ftype)
        if uh.ndim != 1:
            raise ValueError("limited correction currently supports scalar cell values only.")

        is_boundary = edge_to_cell[:, 0] == edge_to_cell[:, 1]
        is_internal = ~is_boundary
        owner = edge_to_cell[:, 0]
        neighbour = edge_to_cell[:, 1]
        face_area = geometry.face_area_norm()
        delta_coeff = geometry.openfoam_delta_coeff()

        orthogonal_flux = bm.zeros_like(face_flux)
        cell_jump = bm.abs(uh[neighbour[is_internal]] - uh[owner[is_internal]])
        orthogonal_flux[is_internal] = (
            face_area[is_internal] * delta_coeff[is_internal] * cell_jump
        )
        limiter = bm.ones_like(face_flux)
        limited = self.limit_coeff * orthogonal_flux[is_internal]
        full = (1.0 - self.limit_coeff) * bm.abs(face_flux[is_internal])
        limiter[is_internal] = bm.minimum(
            limited / (full + self.limiter_small),
            bm.ones_like(full),
        )
        return face_flux * limiter

    def _compute_face_flux_correction(
            self, space: _FS, Tf: TensorLike, edge_to_cell: Optional[TensorLike]=None
    ) -> TensorLike:
        if self.face_flux_correction is not None:
            return bm.array(self.face_flux_correction, dtype=space.ftype)
        if self.grad_f is None:
            raise ValueError("grad_f is required when face_flux_correction is not provided.")
        correction_vector = self._select_correction_vector(space, Tf)
        if self.grad_f.ndim == 2:
            face_flux = bm.einsum('ij,ij->i', correction_vector, self.grad_f)
        elif self.grad_f.ndim == 3:
            face_flux = bm.einsum('ij,ikj->ik', correction_vector, self.grad_f)
        else:
            raise ValueError(f"Unsupported grad_f shape: {self.grad_f.shape}")
        coef = self._coefficient_on_faces(space, face_flux)
        face_flux = bm.einsum('i,i->i', coef, face_flux) if face_flux.ndim == 1 else face_flux * coef[:, None]
        if edge_to_cell is not None:
            face_flux = self._apply_limiter(space, edge_to_cell, face_flux)
        return face_flux

    def _scatter_face_flux_to_cells(
            self, edge_to_cell: TensorLike, NC: int, face_flux: TensorLike
    ) -> TensorLike:
        is_boundary = edge_to_cell[:, 0] == edge_to_cell[:, 1]
        is_internal = ~is_boundary
        if face_flux.ndim == 1:
            result = bm.zeros((NC,), dtype=face_flux.dtype)
            bm.add_at(result, edge_to_cell[:, 0], face_flux)
            bm.add_at(result, edge_to_cell[is_internal, 1], -face_flux[is_internal])
            return result
        elif face_flux.ndim == 2:
            result = bm.zeros((NC, face_flux.shape[1]), dtype=face_flux.dtype)
            bm.add_at(result, edge_to_cell[:, 0], face_flux)
            bm.add_at(result, edge_to_cell[is_internal, 1], -face_flux[is_internal])
            return result
        raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        Tf, edge_to_cell, NC, phi = self.fetch(space)
        face_flux = self._compute_face_flux_correction(space, Tf, edge_to_cell)
        return self._scatter_face_flux_to_cells(edge_to_cell, NC, face_flux)
