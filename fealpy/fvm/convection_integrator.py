"""Finite-volume convection integrator for face-velocity fluxes."""

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from fealpy.decorator.variantmethod import variantmethod

from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.functionspace.utils import to_tensor_dof

from fealpy.fem.integrator import LinearInt, OpInt, FaceInt, enable_cache

from .fvm_geometry import FVMGeometry, face_interpolation_owner_weight


class ConvectionIntegrator(LinearInt, OpInt, FaceInt):
    r"""Assemble the central finite-volume convection operator.

    ``coef`` is a face-wise convection velocity.  In incompressible momentum
    equations it may already include the density factor, so the face flux used
    by this integrator is always interpreted as

    .. math::

        \phi_f = \mathbf c_f \cdot \mathbf S_f .

    The ``interpolation`` option only selects the owner/neighbour weights used
    to reconstruct the central face value.  It does not switch to an upwind,
    bounded, or limited convection scheme.

    Boundary flux closure is deliberately outside this low-level operator.
    Dirichlet, Neumann, and natural outlet convection contributions are applied
    by the boundary-condition layer or by the flow solver that owns the case
    semantics.
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
            raise ValueError("interpolation must be 'average' or 'linear'.")
        return interpolation

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        mesh = getattr(space, "mesh", None)
        face_to_cell = FVMGeometry(mesh, index=self.index).face_to_cell

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
        Sf = FVMGeometry(mesh, index=index).S_f
        q = self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        return Sf, index, bcs, phi

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        Sf, _, _, phi = self.fetch(space)
        D = phi.shape[-1]
        eye_D = bm.eye(D, dtype=space.ftype, device=bm.get_device(space))
        mesh = getattr(space, "mesh")
        owner_weight = face_interpolation_owner_weight(mesh, method=self.interpolation, index=self.index)
        neighbour_weight = 1.0 - owner_weight
        direction_matrix = bm.stack(
            [
                bm.stack([owner_weight, neighbour_weight], axis=-1),
                bm.stack([-owner_weight, -neighbour_weight], axis=-1),
            ],
            axis=1,
        )
        base_matrix = bm.einsum("ij,fpq->fipjq", eye_D, direction_matrix).reshape(-1, 2 * D, 2 * D)
        if coef is None:
            coef = bm.stack([bm.ones_like(Sf[:, 0]), bm.zeros_like(Sf[:, 0])], axis=1)
        integrator = bm.einsum("ij,ij->i", Sf, coef)
        result = integrator[:, None, None] * base_matrix

        return result
