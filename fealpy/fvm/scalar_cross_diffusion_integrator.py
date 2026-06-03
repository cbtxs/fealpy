from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .fvm_geometry import FVMGeometry


_SUPPORTED_CORRECTION_METHODS = {
    "orthogonal",
    "bounded_over_relaxed",
    "limited",
}
_BOUNDARY_POLICIES = {"all", "zero"}


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
        if correction_method not in _SUPPORTED_CORRECTION_METHODS:
            raise ValueError(f"Unsupported correction_method: {correction_method!r}")
        if boundary_policy is not None and boundary_policy not in _BOUNDARY_POLICIES:
            raise ValueError(f"Unsupported boundary_policy: {boundary_policy!r}")
        self.correction_method = correction_method
        self.boundary_policy = boundary_policy
        self.limit_coeff = limit_coeff
        self.limiter_small = limiter_small
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = nonorthogonal_eps
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
        mesh = self._mesh(space)
        geometry = (
            self.geometry
            if self.geometry is not None
            else FVMGeometry(mesh, index=index)
        )
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

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        face_to_cell, geometry = self.fetch(space)
        face_flux = scalar_cross_diffusion_face_flux(
            space,
            geometry,
            face_to_cell,
            uh=self.uh,
            grad_f=self.grad_f,
            coef=self.coef,
            face_flux_correction=self.face_flux_correction,
            correction_vector=self.correction_vector,
            correction_method=self.correction_method,
            boundary_policy=self.boundary_policy,
            limit_coeff=self.limit_coeff,
            limiter_small=self.limiter_small,
            nonorthogonal_eps=self.nonorthogonal_eps,
        )
        return geometry.scatter_face_flux_to_cells(face_flux)


def scalar_cross_diffusion_face_flux(
    space: _FS,
    geometry: FVMGeometry,
    face_to_cell: TensorLike,
    *,
    uh=None,
    grad_f=None,
    coef: Optional[CoefLike]=None,
    face_flux_correction=None,
    correction_vector=None,
    correction_method: str="bounded_over_relaxed",
    boundary_policy: Optional[str]=None,
    limit_coeff: float=0.5,
    limiter_small: float=1.0e-30,
    nonorthogonal_eps: float=0.05,
) -> TensorLike:
    """Return owner-oriented explicit non-orthogonal correction flux.

    Future cleanup directions:

    - ``limited`` is intentionally kept in this function for now.  If the
      limiter becomes a stable production route, move only that formula into a
      clearly named top-level function rather than back into the integrator.
    - The validation here duplicates part of ``ScalarCrossDiffusionIntegrator``
      so that this function can be called directly by future solver kernels.
      Remove the duplication only after the intended public surface is fixed.
    - ``coef`` currently supports scalar values or face-wise arrays.  Extend
      this function deliberately if cell-wise, callable, or tensor coefficients
      become necessary.
    """
    if correction_method not in _SUPPORTED_CORRECTION_METHODS:
        raise ValueError(f"Unsupported correction_method: {correction_method!r}")
    if boundary_policy is not None and boundary_policy not in _BOUNDARY_POLICIES:
        raise ValueError(f"Unsupported boundary_policy: {boundary_policy!r}")
    if nonorthogonal_eps <= 0.0:
        raise ValueError("nonorthogonal_eps must be positive.")

    user_face_flux = face_flux_correction is not None
    user_correction_vector = correction_vector is not None
    if face_flux_correction is not None:
        face_flux = bm.array(face_flux_correction, dtype=space.ftype)
    else:
        if grad_f is None:
            raise ValueError(
                "grad_f is required when face_flux_correction is not provided."
            )
        if correction_vector is not None:
            correction_vector = bm.array(correction_vector, dtype=space.ftype)
        elif correction_method == "orthogonal":
            correction_vector = bm.zeros_like(geometry.S_f)
        else:
            # Boundary handling is a scheme-level decision; raw geometry still
            # exposes the bounded over-relaxed T_f on all faces.
            correction_vector = geometry.bounded_over_relaxed_decomposition(
                eps=nonorthogonal_eps
            )[2]

        if grad_f.ndim == 2:
            face_flux = bm.einsum("ij,ij->i", correction_vector, grad_f)
        elif grad_f.ndim == 3:
            face_flux = bm.einsum("ij,ikj->ik", correction_vector, grad_f)
        else:
            raise ValueError(f"Unsupported grad_f shape: {grad_f.shape}")

        shape_source = face_flux[:, 0] if face_flux.ndim == 2 else face_flux
        if coef is None:
            face_coef = bm.ones_like(shape_source, dtype=space.ftype)
        elif type(coef) in [int, float]:
            face_coef = bm.full_like(shape_source, fill_value=coef, dtype=space.ftype)
        else:
            face_coef = coef
        if face_flux.ndim == 1:
            face_flux = bm.einsum("i,i->i", face_coef, face_flux)
        else:
            face_flux = face_flux * face_coef[:, None]

        if correction_method == "limited":
            if face_flux.ndim != 1:
                raise ValueError(
                    "limited correction currently supports scalar face flux only."
                )
            if uh is None:
                raise ValueError("uh is required when correction_method='limited'.")

            if getattr(uh, "dtype", None) is None:
                uh = bm.array(uh, dtype=space.ftype)
            elif uh.dtype != space.ftype:
                uh = bm.astype(uh, space.ftype)
            if uh.ndim != 1:
                raise ValueError(
                    "limited correction currently supports scalar cell values only."
                )

            is_internal = face_to_cell[:, 0] != face_to_cell[:, 1]
            owner = face_to_cell[:, 0]
            neighbour = face_to_cell[:, 1]
            _, mag_E_f, _ = geometry.bounded_over_relaxed_decomposition(
                eps=nonorthogonal_eps
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
            limited = limit_coeff * orthogonal_flux[is_internal]
            full = (1.0 - limit_coeff) * bm.abs(face_flux[is_internal])
            limiter = bm.set_at(
                limiter,
                is_internal,
                bm.minimum(
                    limited / (full + limiter_small),
                    bm.ones_like(full),
                ),
            )
            face_flux = face_flux * limiter

    if boundary_policy is None:
        if user_face_flux or user_correction_vector:
            boundary_policy = "all"
        elif correction_method in ("bounded_over_relaxed", "limited"):
            boundary_policy = "zero"
        else:
            boundary_policy = "all"

    if boundary_policy == "all":
        return face_flux

    is_boundary = face_to_cell[:, 0] == face_to_cell[:, 1]
    if face_flux.ndim == 1:
        return bm.where(is_boundary, 0.0, face_flux)
    if face_flux.ndim == 2:
        return bm.where(is_boundary[:, None], 0.0, face_flux)
    raise ValueError(f"Unsupported face_flux_correction shape: {face_flux.shape}")
