"""Explicit finite-volume source for the deviatoric viscous-stress correction."""

from typing import Literal, Optional

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.fem.integrator import FaceInt, LinearInt, OpInt, enable_cache
from fealpy.functionspace.space import FunctionSpace as _FS
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import CoefLike, Index, TensorLike, _S

from .fvm_geometry import FVMGeometry


def _deviatoric_stress_face_flux(
    grad_f: TensorLike,
    face_normal: TensorLike,
    coef: Optional[CoefLike] = None,
) -> TensorLike:
    r"""Return owner-oriented face flux for ``dev2(T(grad(U)))``.

    ``grad_f[f, i, j]`` stores ``d u_i / d x_j`` and ``face_normal`` stores the
    owner-oriented face vector ``S_f``.  The face flux is

        mu_f [grad(U)_f^T - 2/3 div(U)_f I] S_f,
    """
    grad_f = bm.array(grad_f)
    face_normal = bm.array(face_normal, dtype=grad_f.dtype)

    if grad_f.ndim != 3 or grad_f.shape[1] != grad_f.shape[2]:
        raise ValueError(f"grad_f must have shape (NF, GD, GD), got {grad_f.shape}.")
    if face_normal.ndim != 2 or grad_f.shape[:2] != face_normal.shape:
        raise ValueError(
            "face_normal must have shape (NF, GD) matching grad_f, got "
            f"{face_normal.shape}."
        )

    if coef is None:
        coef_f = bm.ones_like(face_normal[:, 0], dtype=grad_f.dtype)
    elif isinstance(coef, (int, float)):
        coef_f = bm.full_like(face_normal[:, 0], fill_value=coef, dtype=grad_f.dtype)
    else:
        coef_f = bm.array(coef, dtype=grad_f.dtype)
        if coef_f.shape == ():
            coef_f = bm.full_like(
                face_normal[:, 0],
                fill_value=float(coef_f),
                dtype=grad_f.dtype,
            )
        elif coef_f.shape[0] != face_normal.shape[0]:
            raise ValueError(
                f"coef has incompatible first dimension {coef_f.shape[0]}; expected "
                f"{face_normal.shape[0]} faces."
            )

    div_u = bm.einsum("fii->f", grad_f)
    return coef_f[:, None] * (
        bm.einsum("fji,fj->fi", grad_f, face_normal)
        - (2.0 / 3.0) * div_u[:, None] * face_normal
    )


def deviatoric_stress_integral(
    grad_f: TensorLike,
    face_normal: TensorLike,
    edge_to_cell: TensorLike,
    nc: int,
    coef: Optional[CoefLike] = None,
) -> TensorLike:
    r"""Return the cell-integrated ``dev2(T(grad(U)))`` source."""
    edge_to_cell = bm.array(edge_to_cell)
    face_flux = _deviatoric_stress_face_flux(grad_f, face_normal, coef)

    if edge_to_cell.ndim != 2 or edge_to_cell.shape[0] != face_flux.shape[0]:
        raise ValueError(
            "edge_to_cell must have shape (NF, 2) matching face_flux, got "
            f"{edge_to_cell.shape}."
        )

    is_internal = edge_to_cell[:, 0] != edge_to_cell[:, 1]
    result = bm.zeros((nc, face_flux.shape[1]), dtype=face_flux.dtype)
    result = bm.index_add(result, edge_to_cell[:, 0], face_flux, axis=0)
    return bm.index_add(
        result,
        edge_to_cell[is_internal, 1],
        face_flux[is_internal],
        axis=0,
        alpha=-1,
    )


class DeviatoricStressSourceIntegrator(LinearInt, OpInt, FaceInt):
    """Assemble the explicit deviatoric viscous-stress correction RHS."""

    def __init__(
        self,
        grad_f: TensorLike,
        coef: Optional[CoefLike] = None,
        *,
        index: Index = _S,
        region: Optional[Index] = None,
        batched: bool = False,
        method: Literal[None] = None,
    ) -> None:
        super().__init__()
        if region is not None and not (isinstance(index, slice) and index == _S):
            raise ValueError("Use either 'region' or legacy 'index', not both.")
        self.grad_f = grad_f
        self.coef = coef
        self.index = index
        self.batched = batched
        self.set_region(region)
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        # The integration region is face-based, but the scattered source lives
        # on cell unknowns.
        return space.cell_to_dof()

    @enable_cache
    def fetch(self, space: _FS, /, indices=None):
        mesh = space.mesh
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError(
                "DeviatoricStressSourceIntegrator only supports homogeneous "
                f"meshes, but got {type(mesh).__name__}."
            )

        region = self.get_region()
        if region is not None:
            if isinstance(region, slice) and indices is not None:
                index = bm.arange(mesh.number_of_faces())[region][indices]
            else:
                index = self.entity_selection(indices, mesh=mesh)
        elif indices is None:
            index = self.index
        elif isinstance(self.index, slice) and self.index == _S:
            index = indices
        elif isinstance(self.index, slice):
            index = bm.arange(mesh.number_of_faces())[self.index][indices]
        elif bm.is_tensor(self.index):
            if self.index.dtype == bm.bool:
                index = bm.nonzero(self.index)[0][indices]
            else:
                index = self.index[indices]
        else:
            raise TypeError(
                f"index of type '{self.index.__class__.__name__}' is not supported "
                "when local indices are given."
            )

        return mesh, FVMGeometry(mesh, index=index), index

    @variantmethod
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        mesh, geometry, index = self.fetch(space, indices)

        grad_f = bm.array(self.grad_f)
        if grad_f.shape[0] == mesh.number_of_faces():
            grad_f = grad_f[index]

        coef = self.coef
        if coef is not None and not isinstance(coef, (int, float)):
            coef = bm.array(coef)
            if coef.shape != () and coef.shape[0] == mesh.number_of_faces():
                coef = coef[index]

        face_flux = _deviatoric_stress_face_flux(grad_f, geometry.S_f, coef)
        return geometry.scatter_face_flux_to_cells(face_flux)
