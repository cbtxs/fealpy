from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .fvm_geometry import FVMGeometry


class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble explicit non-orthogonal diffusion correction as a cell RHS.

    The implicit matrix part is handled by ``ScalarDiffusionIntegrator``.  This
    integrator only scatters a face correction flux,

        coef_f * C_f · grad(phi)_f,

    to owner/neighbour cells.  The default ``bounded_over_relaxed`` method uses
    the bounded over-relaxed ``T_f`` from ``FVMGeometry``.  For ordinary
    non-coupled boundary faces, that correction flux is zero by default because
    boundary diffusion is already handled by the boundary condition layer.
    """

    _SUPPORTED_CORRECTION_METHODS = {
        "orthogonal",
        "bounded_over_relaxed",
        "limited",
    }
    _BOUNDARY_POLICIES = {"all", "zero"}

    def __init__(
        self,
        uh=None,
        grad_f=None,
        coef: Optional[CoefLike]=None,
        q: Optional[int]=None,
        *,
        face_flux_correction=None,
        correction_vector=None,
        geometry=None,
        correction_method: str="bounded_over_relaxed",
        boundary_policy: Optional[str]=None,
        limit_coeff: float=0.5,
        limiter_small: float=1.0e-30,
        nonorthogonal_eps: float=0.05,
        index: Index=_S,
        batched: bool=False,
        method: Optional[str]=None,
    ) -> None:
        super().__init__()
        self.uh = uh
        self.grad_f = grad_f
        self.coef = coef
        self.face_flux_correction = face_flux_correction
        self.correction_vector = correction_vector
        self.geometry = geometry
        self.correction_method = self._validate_correction_method(correction_method)
        self.boundary_policy = self._validate_boundary_policy(boundary_policy)
        self.limit_coeff = limit_coeff
        self.limiter_small = limiter_small
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = nonorthogonal_eps
        self.q = 2 if q is None else q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @classmethod
    def _validate_correction_method(cls, correction_method: str) -> str:
        if correction_method not in cls._SUPPORTED_CORRECTION_METHODS:
            raise ValueError(f"Unsupported correction_method: {correction_method!r}")
        return correction_method

    @classmethod
    def _validate_boundary_policy(cls, boundary_policy: Optional[str]) -> Optional[str]:
        if boundary_policy is not None and boundary_policy not in cls._BOUNDARY_POLICIES:
            raise ValueError(f"Unsupported boundary_policy: {boundary_policy!r}")
        return boundary_policy

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = self._mesh(space)
        geometry = FVMGeometry(mesh, index=index)
        return geometry.face_to_cell, geometry

    @staticmethod
    def _mesh(space: _FS) -> HomogeneousMesh:
        mesh = getattr(space, "mesh", None)
        if isinstance(mesh, HomogeneousMesh):
            return mesh
        raise RuntimeError(
            "The ScalarCrossDiffusionIntegrator only supports spaces on "
            f"homogeneous meshes, but {type(mesh).__name__} is not a subclass "
            "of HomoMesh."
        )

    def _geometry(self, space: _FS) -> FVMGeometry:
        return (
            self.geometry
            if self.geometry is not None
            else FVMGeometry(self._mesh(space), index=self.index)
        )

    @staticmethod
    def _face_shape_source(face_flux: TensorLike) -> TensorLike:
        return face_flux[:, 0] if face_flux.ndim == 2 else face_flux

    def _coefficient_on_faces(self, space: _FS, face_flux: TensorLike) -> TensorLike:
        shape_source = self._face_shape_source(face_flux)
        if self.coef is None:
            return bm.ones_like(shape_source, dtype=space.ftype)
        if type(self.coef) in [int, float]:
            return bm.full_like(shape_source, fill_value=self.coef, dtype=space.ftype)
        return self.coef

    def _select_correction_vector(
            self, space: _FS, geometry: FVMGeometry
    ) -> TensorLike:
        if self.correction_vector is not None:
            return bm.array(self.correction_vector, dtype=space.ftype)

        if self.correction_method == "orthogonal":
            return bm.zeros_like(geometry.S_f)
        if self.correction_method in ("bounded_over_relaxed", "limited"):
            # Boundary handling is deliberately outside the geometry helper:
            # OpenFOAM sets non-coupled boundary correction fluxes to zero at
            # the scheme level, not by changing the raw geometric vector.
            return geometry.bounded_over_relaxed_decomposition(
                eps=self.nonorthogonal_eps
            )[2]
        raise ValueError(f"Unsupported correction_method: {self.correction_method!r}")

    def _apply_limiter(
            self,
            space: _FS,
            face_to_cell: TensorLike,
            face_flux: TensorLike,
            geometry: FVMGeometry,
    ) -> TensorLike:
        if self.correction_method != "limited":
            return face_flux
        if face_flux.ndim != 1:
            raise ValueError("limited correction currently supports scalar face flux only.")
        if self.uh is None:
            raise ValueError("uh is required when correction_method='limited'.")

        uh = self.uh
        if getattr(uh, "dtype", None) is None:
            uh = bm.array(uh, dtype=space.ftype)
        elif uh.dtype != space.ftype:
            uh = bm.astype(uh, space.ftype)
        if uh.ndim != 1:
            raise ValueError("limited correction currently supports scalar cell values only.")

        is_internal = self._internal_face_mask(face_to_cell)
        owner = face_to_cell[:, 0]
        neighbour = face_to_cell[:, 1]
        _, mag_E_f, _ = geometry.bounded_over_relaxed_decomposition(
            eps=self.nonorthogonal_eps
        )
        orthogonal_coeff = mag_E_f / geometry.mag_d_f

        orthogonal_flux = bm.zeros_like(face_flux)
        cell_jump = bm.abs(uh[neighbour[is_internal]] - uh[owner[is_internal]])
        orthogonal_flux = bm.set_at(
            orthogonal_flux,
            is_internal,
            orthogonal_coeff[is_internal] * cell_jump,
        )
        limiter = bm.ones_like(face_flux)
        limited = self.limit_coeff * orthogonal_flux[is_internal]
        full = (1.0 - self.limit_coeff) * bm.abs(face_flux[is_internal])
        limiter = bm.set_at(
            limiter,
            is_internal,
            bm.minimum(
                limited / (full + self.limiter_small),
                bm.ones_like(full),
            ),
        )
        return face_flux * limiter

    def _default_boundary_policy(self) -> str:
        if self.face_flux_correction is not None or self.correction_vector is not None:
            return "all"
        if self.correction_method in ("bounded_over_relaxed", "limited"):
            return "zero"
        return "all"

    def _effective_boundary_policy(self) -> str:
        return self.boundary_policy or self._default_boundary_policy()

    @staticmethod
    def _boundary_face_mask(face_to_cell: TensorLike) -> TensorLike:
        return face_to_cell[:, 0] == face_to_cell[:, 1]

    @classmethod
    def _internal_face_mask(cls, face_to_cell: TensorLike) -> TensorLike:
        return ~cls._boundary_face_mask(face_to_cell)

    def _apply_boundary_policy(
            self, face_to_cell: Optional[TensorLike], face_flux: TensorLike
    ) -> TensorLike:
        policy = self._effective_boundary_policy()
        if policy == "all":
            return face_flux
        if face_to_cell is None:
            raise ValueError("face_to_cell is required when boundary_policy='zero'.")

        is_boundary = self._boundary_face_mask(face_to_cell)
        if face_flux.ndim == 1:
            return bm.where(is_boundary, 0.0, face_flux)
        if face_flux.ndim == 2:
            return bm.where(is_boundary[:, None], 0.0, face_flux)
        raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")

    def _face_flux_from_gradient(self, space: _FS, geometry: FVMGeometry) -> TensorLike:
        if self.grad_f is None:
            raise ValueError("grad_f is required when face_flux_correction is not provided.")

        correction_vector = self._select_correction_vector(space, geometry)
        if self.grad_f.ndim == 2:
            return bm.einsum("ij,ij->i", correction_vector, self.grad_f)
        if self.grad_f.ndim == 3:
            return bm.einsum("ij,ikj->ik", correction_vector, self.grad_f)
        raise ValueError(f"Unsupported grad_f shape: {self.grad_f.shape}")

    def _apply_coefficient(self, space: _FS, face_flux: TensorLike) -> TensorLike:
        coef = self._coefficient_on_faces(space, face_flux)
        if face_flux.ndim == 1:
            return bm.einsum("i,i->i", coef, face_flux)
        return face_flux * coef[:, None]

    def _compute_face_flux_correction(
            self,
            space: _FS,
            face_to_cell: Optional[TensorLike]=None,
            geometry: Optional[FVMGeometry]=None,
    ) -> TensorLike:
        if self.face_flux_correction is not None:
            face_flux = bm.array(self.face_flux_correction, dtype=space.ftype)
            return self._apply_boundary_policy(face_to_cell, face_flux)

        geometry = geometry if geometry is not None else self._geometry(space)
        face_flux = self._face_flux_from_gradient(space, geometry)
        face_flux = self._apply_coefficient(space, face_flux)
        if face_to_cell is not None:
            face_flux = self._apply_limiter(space, face_to_cell, face_flux, geometry)
        return self._apply_boundary_policy(face_to_cell, face_flux)

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        face_to_cell, geometry = self.fetch(space)
        face_flux = self._compute_face_flux_correction(
            space,
            face_to_cell,
            geometry,
        )
        return geometry.scatter_face_flux_to_cells(face_flux)
