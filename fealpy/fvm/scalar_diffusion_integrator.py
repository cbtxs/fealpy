"""Scalar finite-volume diffusion integrator for the orthogonal flux part."""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator import variantmethod

from fealpy.mesh import HomogeneousMesh
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functionspace.utils import to_tensor_dof

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .fvm_geometry import FVMGeometry


class ScalarDiffusionIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble the implicit two-point diffusion contribution.

    The local face matrix corresponds to the orthogonal finite-volume flux

        gamma_f |E_f| / |e_f| (phi_N - phi_P),

    where ``E_f`` is the projection of the face area vector onto the
    owner-neighbour centre line.  Non-orthogonal cross terms are intentionally
    not assembled here; they are handled explicitly by
    ``ScalarCrossDiffusionIntegrator``.
    """

    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        super().__init__()
        self.coef = coef
        self.q = 2 if q is None else q
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        mesh = getattr(space, "mesh", None)
        face_to_cell = mesh.face_to_cell()[self.index, :2]

        scalar_space = getattr(space, "scalar_space", None)
        if scalar_space is None:
            return face_to_cell

        return to_tensor_dof(
            face_to_cell,
            space.dof_numel,
            scalar_space.number_of_global_dofs(),
            space.dof_priority,
        )

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only supports spaces on "
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               " not a subclass of HomoMesh.")
        geometry = FVMGeometry(mesh, index=index)
        q = self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return geometry, index, bcs, phi

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        geometry, _, _, phi = self.fetch(space)
        return scalar_diffusion_local_matrix(space, geometry, phi, coef=self.coef)

def scalar_diffusion_local_matrix(
    space: _FS,
    geometry: FVMGeometry,
    phi: TensorLike,
    *,
    coef: Optional[CoefLike]=None,
) -> TensorLike:
    """Return local two-point matrices for the orthogonal diffusion flux.

    ``coef`` is interpreted as a constant or face-wise diffusion coefficient.
    Cell-wise coefficients must be interpolated to faces before calling this
    function.

    Future cleanup directions:

    - Add an explicit face-coefficient construction function if cell-wise or
      callable diffusion coefficients are needed.  The interpolation strategy
      should be selected deliberately, for example linear for smooth
      coefficients or harmonic for jump coefficients.
    - Keep ``over_relaxed_decomposition`` fixed for the current production
      route.  If minimum-correction, orthogonal-only, or bounded variants are
      needed, expose the decomposition strategy as an explicit option.
    - Revisit the local matrix layout before using this function with 3D,
      higher-order, or non-two-cell face spaces.
    - Consider caching ``mag_E_f / mag_d_f`` and the component base matrix only
      after profiler data shows repeated assembly cost is significant.
    """
    D = phi.shape[-1]
    _, mag_E_f, _ = geometry.over_relaxed_decomposition()
    if coef is None:
        face_coef = bm.ones_like(mag_E_f, dtype=space.ftype)
    elif isinstance(coef, (int, float)):
        face_coef = bm.full_like(mag_E_f, fill_value=coef, dtype=space.ftype)
    else:
        face_coef = bm.array(coef, dtype=space.ftype)
        if face_coef.shape == ():
            face_coef = bm.ones_like(mag_E_f, dtype=space.ftype) * face_coef
        elif face_coef.ndim != 1 or face_coef.shape[0] != mag_E_f.shape[0]:
            raise ValueError(
                "coef must be scalar or face-wise for ScalarDiffusionIntegrator."
            )

    face_strength = bm.einsum("i,i->i", mag_E_f / geometry.mag_d_f, face_coef)
    direction_matrix = bm.array([[1.0, -1.0], [-1.0, 1.0]], dtype=space.ftype)
    eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
    base_matrix = bm.einsum("ij,pq->ipjq", eye_D, direction_matrix).reshape(
        2 * D, 2 * D
    )
    return bm.einsum("i,ab->iab", face_strength, base_matrix)
