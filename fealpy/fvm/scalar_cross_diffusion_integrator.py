from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .nonorthogonal_geometry import NonOrthogonalGeometry


class ScalarCrossDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble explicit non-orthogonal diffusion correction as a cell RHS.

    The implicit matrix part is handled by ``ScalarDiffusionIntegrator``.  This
    integrator only scatters a face correction flux,

        coef_f * C_f · grad(phi)_f,

    to owner/neighbour cells.  The default ``openfoam_stabilized`` method uses
    the OpenFOAM-like bounded correction vector from ``NonOrthogonalGeometry``.
    For ordinary non-coupled boundary faces, that correction flux is zero by
    default because boundary diffusion is already handled by the boundary
    condition layer.
    """

    _SUPPORTED_CORRECTION_METHODS = {
        "orthogonal",
        "openfoam_stabilized",
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
        correction_method: str="openfoam_stabilized",
        boundary_policy: Optional[str]=None,
        limit_coeff: float=0.5,
        limiter_small: float=1.0e-30,
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
        edge_to_cell = mesh.edge_to_cell(index=index)[:, :2]
        NC = mesh.number_of_cells()
        return edge_to_cell, NC

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

    def _geometry(self, space: _FS) -> NonOrthogonalGeometry:
        return self.geometry if self.geometry is not None else NonOrthogonalGeometry(
            self._mesh(space)
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

    def _select_correction_vector(self, space: _FS) -> TensorLike:
        if self.correction_vector is not None:
            return bm.array(self.correction_vector, dtype=space.ftype)

        geometry = self._geometry(space)
        if self.correction_method == "orthogonal":
            return geometry.zero_correction_vector()
        if self.correction_method in ("openfoam_stabilized", "limited"):
            # Boundary handling is deliberately outside the geometry helper:
            # OpenFOAM sets non-coupled boundary correction fluxes to zero at
            # the scheme level, not by changing the raw geometric vector.
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

        geometry = self._geometry(space)
        uh = self.uh
        if getattr(uh, "dtype", None) is None:
            uh = bm.array(uh, dtype=space.ftype)
        elif uh.dtype != space.ftype:
            uh = bm.astype(uh, space.ftype)
        if uh.ndim != 1:
            raise ValueError("limited correction currently supports scalar cell values only.")

        is_internal = self._internal_face_mask(edge_to_cell)
        owner = edge_to_cell[:, 0]
        neighbour = edge_to_cell[:, 1]
        face_area = geometry.face_area_norm()
        delta_coeff = geometry.openfoam_delta_coeff()

        orthogonal_flux = bm.zeros_like(face_flux)
        cell_jump = bm.abs(uh[neighbour[is_internal]] - uh[owner[is_internal]])
        orthogonal_flux = bm.set_at(
            orthogonal_flux,
            is_internal,
            face_area[is_internal] * delta_coeff[is_internal] * cell_jump,
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
        if self.correction_method in ("openfoam_stabilized", "limited"):
            return "zero"
        return "all"

    def _effective_boundary_policy(self) -> str:
        return self.boundary_policy or self._default_boundary_policy()

    @staticmethod
    def _boundary_face_mask(edge_to_cell: TensorLike) -> TensorLike:
        return edge_to_cell[:, 0] == edge_to_cell[:, 1]

    @classmethod
    def _internal_face_mask(cls, edge_to_cell: TensorLike) -> TensorLike:
        return ~cls._boundary_face_mask(edge_to_cell)

    def _apply_boundary_policy(
            self, edge_to_cell: Optional[TensorLike], face_flux: TensorLike
    ) -> TensorLike:
        policy = self._effective_boundary_policy()
        if policy == "all":
            return face_flux
        if edge_to_cell is None:
            raise ValueError("edge_to_cell is required when boundary_policy='zero'.")

        is_boundary = self._boundary_face_mask(edge_to_cell)
        if face_flux.ndim == 1:
            return bm.where(is_boundary, 0.0, face_flux)
        if face_flux.ndim == 2:
            return bm.where(is_boundary[:, None], 0.0, face_flux)
        raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")

    def _face_flux_from_gradient(self, space: _FS) -> TensorLike:
        if self.grad_f is None:
            raise ValueError("grad_f is required when face_flux_correction is not provided.")

        correction_vector = self._select_correction_vector(space)
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
            self, space: _FS, edge_to_cell: Optional[TensorLike]=None
    ) -> TensorLike:
        if self.face_flux_correction is not None:
            face_flux = bm.array(self.face_flux_correction, dtype=space.ftype)
            return self._apply_boundary_policy(edge_to_cell, face_flux)

        face_flux = self._face_flux_from_gradient(space)
        face_flux = self._apply_coefficient(space, face_flux)
        if edge_to_cell is not None:
            face_flux = self._apply_limiter(space, edge_to_cell, face_flux)
        return self._apply_boundary_policy(edge_to_cell, face_flux)

    def _scatter_face_flux_to_cells(
            self, edge_to_cell: TensorLike, NC: int, face_flux: TensorLike
    ) -> TensorLike:
        is_internal = self._internal_face_mask(edge_to_cell)
        if face_flux.ndim == 1:
            result = bm.zeros((NC,), dtype=face_flux.dtype)
            result = bm.index_add(result, edge_to_cell[:, 0], face_flux, axis=0)
            result = bm.index_add(
                result,
                edge_to_cell[is_internal, 1],
                face_flux[is_internal],
                axis=0,
                alpha=-1,
            )
            return result
        elif face_flux.ndim == 2:
            result = bm.zeros((NC, face_flux.shape[1]), dtype=face_flux.dtype)
            result = bm.index_add(result, edge_to_cell[:, 0], face_flux, axis=0)
            result = bm.index_add(
                result,
                edge_to_cell[is_internal, 1],
                face_flux[is_internal],
                axis=0,
                alpha=-1,
            )
            return result
        raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        edge_to_cell, NC = self.fetch(space)
        face_flux = self._compute_face_flux_correction(space, edge_to_cell)
        return self._scatter_face_flux_to_cells(edge_to_cell, NC, face_flux)
