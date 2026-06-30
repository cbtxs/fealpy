from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate, final, Literal, ParamSpec, TYPE_CHECKING

from ...backend import bm, Tensor, Index
from ..schema.entity_schema import EntityContext

if TYPE_CHECKING:
    from ..storage import EntitySector, MeshBlock, Relation
    from ..topology.boundary import BoundaryInfo

__all__ = ["EntityView"]

P = ParamSpec("P")

@final
class EntityView:
    """Provide a view of an entity sector, with user-friendly APIs to access its
    properties and relations."""
    def __init__(self, block: MeshBlock, sec: EntitySector):
        self.block = block
        self.sector = sec
        self.schema = sec.schema

    def __len__(self) -> int:
        return self.schema.size(self.context())

    def context(self) -> EntityContext:
        return EntityContext(self.block, self.sector)

    # User APIs

    def barycentric[**P, R](self, func: Callable[Concatenate[Tensor, P], R], /, *, index: Index | None = None):
        """Transform a function defined in cartesian coordinates to barycentric coordinates.

        Parameters:
            func (Callable): A function that takes cartesian coordinates as the first positional input.
            index (Index, optional): The index of the entities for which to compute the barycentric function.

        Returns:
            Callable: A function that takes barycentric coordinates as the first positional input.
        """
        return self.schema.barycentric(self.context(), func, index)

    def barycenter(self, *, index: Index | None = None) -> Tensor:
        return self.schema.barycenter(self.context(), index)

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        if not isinstance(bc, tuple):
            bc = (bc,)
        return self.schema.bc_to_point(self.context(), bc, index)

    def boundary(self) -> "BoundaryInfo":
        """Infer the boundary entities of this entity, and return a mapping from
        boundary name to its information.

        Returns:
            namedtuple:
            - index: Tensor of shape (num_boundary,) containing the indices
                of boundary entities.
            - mask: Tensor of shape (num_entity,) containing a boolean mask
                indicating which entities are boundary.
            - count: int, the number of references by the highest-dimensional
                entities.
        """
        return self.schema.boundary(self.context())

    def error(
        self,
        f1: Callable[..., Tensor],
        f2: Callable[..., Tensor],
        /,
        power: float = 2.0,
        q: int = 3,
        *,
        cell_axis: bool = False,
        index: Index | None = None
    ) -> Tensor:
        from ...decorator import barycentric
        if not getattr(f1, "coordtype", None) == "barycentric":
            f1 = self.barycentric(f1, index=index)
        if not getattr(f2, "coordtype", None) == "barycentric":
            f2 = self.barycentric(f2, index=index)
        @barycentric
        def integrand(bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
            v1 = f1(bcs)
            v2 = f2(bcs)
            return bm.abs(v1 - v2) ** power

        if cell_axis:
            return self.integral(integrand, q=q, index=index) ** (1.0 / power)
        return bm.sum(self.integral(integrand, q=q, index=index)) ** (1.0 / power)

    def geo_dimension(self) -> int:
        return self.schema.geo_dimension(self.context())

    def global_permutations(self, tgt_name: str) -> Tensor:
        return self.schema.global_permutations(self.context(), tgt_name)

    def grad_lambda(
        self,
        bcs: tuple[Tensor, ...] | None = None,
        index: Index | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        return self.schema.grad_lambda(self.context(), index, bcs=bcs, ref=ref) # type: ignore

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal["b", "u", "x"] = "u",
        mi = None
    ) -> Tensor:
        """Compute the gradient of shape functions.

        Parameters:
            bcs (Tensor or tuple of Tensor): The barycentric coordinates at which to evaluate the gradients.
            p (int or tuple of int): The order(s) of the shape functions.
            index (Index, optional): The index of the entities for which to compute the gradients.
            variables (str): The coordinate system for the gradients.
                "b" for barycentric, "u" for reference, "x" for cartesian.

        Returns:
            Tensor: The gradients of the shape functions evaluated at the given barycentric coordinates and order.
        """
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if isinstance(p, int):
            p = (p,)
        if variables == "b":
            return self.schema.grad_shape_function_barycentric(bcs=bcs, p=p)
        elif variables == "u":
            return self.schema.grad_shape_function_reference(bcs=bcs, p=p)
        elif variables == "x":
            return self.schema.grad_shape_function_cartesian(
                self.context(), bcs=bcs, p=p, index=index
            )
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    @property
    def indices(self) -> Tensor:
        return self.sector.indices

    def integral(
        self,
        func: Callable[..., Tensor],
        /,
        q: int = 3,
        *,
        index: Index | None = None
    ) -> Tensor:
        return self.schema.integral(self.context(), func, q, index)

    def jacobi_matrix(self, bcs: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        return self.schema.jacobi_matrix(self.context(), bcs, index)

    def measure(self, *, index: Index | None = None) -> Tensor:
        return self.schema.measure(self.context(), index)

    def multi_index_matrix(self, order: int | tuple[int, ...], *, internal: bool = False, tensorprod: bool = True):
        if isinstance(order, int):
            order = (order,)
        return self.schema.multi_index(order, internal=internal, tensorprod=tensorprod)

    def normal(self, *, index: Index | None = None) -> Tensor:
        return self.schema.normal(self.context(), index)

    def num_multi_index(self, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        if isinstance(order, int):
            order = (order,)
        return self.schema.num_multi_index(order, internal=internal)

    def quadrature_formula(self, q: int = 3, qtype: str = "legendre"):
        return self.schema.quadrature_formula(q, qtype)

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi = None
    ) -> Tensor:
        if isinstance(bcs, Tensor):
            bcs = (bcs,)
        if isinstance(p, int):
            p = (p,)
        val = self.schema.shape_function(bcs, p)
        if variables == "u":
            return val
        elif variables == "x":
            return val[None, ...] # type: ignore[return-value]
        else:
            raise ValueError(f"Unsupported variable type: {variables}")

    def size(self) -> int:
        return self.schema.size(self.context())

    def tangent(self, *, index: Index | None = None) -> Tensor:
        return self.schema.tangent(self.context(), index)

    def to(self, target: str | EntityView, /) -> Relation:
        """Compute the relation between this entity and the target entity."""
        if isinstance(target, EntityView):
            target = target.schema.name
        return self.schema.relation(self.context(), target)

    def top_dimension(self) -> int:
        return self.schema.top_dim
