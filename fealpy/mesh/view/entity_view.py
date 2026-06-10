from __future__ import annotations

from collections.abc import Callable
from typing import final, TYPE_CHECKING, ParamSpec

from ...backend import Tensor, Index
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

    def barycentric(self, func: Callable[[Tensor], Tensor], /, *, index: Index | None = None):
        return self.schema.barycentric(self.context(), index, func)

    def barycenter(self, *, index: Index | None = None) -> Tensor:
        return self.schema.barycenter(self.context(), index)

    def bc_to_point(self, bc: Tensor, *, index: Index | None = None) -> Tensor:
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

    def geo_dimension(self) -> int:
        return self.schema.geo_dimension(self.context())

    def grad_lambda(
        self,
        *,
        bcs: tuple[Tensor, ...] | None = None,
        index: Index | None = None,
        ref: bool = False,
    ) -> Tensor:
        return self.schema.grad_lambda(self.context(), index, bcs=bcs, ref=ref)

    def grad_shape_function(
        self,
        *,
        bcs: tuple[Tensor, ...] | None = None,
        p: int = 1,
        index: Index | None = None,
        variables: str = "u",
        mi = None
    ) -> Tensor:
        return self.schema.grad_shape_function(
            self.context(), bcs=bcs, p=p, index=index, variables=variables, mi=mi
        )

    @property
    def indices(self) -> Tensor:
        return self.sector.indices

    def integral(self, func: Callable[[Tensor], Tensor], /, *, index: Index | None = None, q: int = 3) -> Tensor:
        return self.schema.integral(self.context(), index, func, q)

    def jacobi_matrix(self, *, index: Index | None = None) -> Tensor:
        return self.schema.jacobi_matrix(self.context(), index)

    def measure(self, *, index: Index | None = None) -> Tensor:
        return self.schema.measure(self.context(), index)

    def multi_index_matrix(self, order: int | tuple[int, ...]):
        if isinstance(order, int):
            order = (order,)
        return self.schema.multi_index(order)

    def normal(self, *, index: Index | None = None) -> Tensor:
        return self.schema.normal(self.context(), index)

    def num_multi_index(self, order: int | tuple[int, ...]) -> int:
        if isinstance(order, int):
            order = (order,)
        return self.schema.num_multi_index(order)

    def quadrature_formula(self, q: int = 3, qtype: str = "legendre") -> tuple[Tensor, Tensor]:
        return self.schema.quadrature_formula(q, qtype)

    def shape_function(
        self,
        bcs: tuple[Tensor, ...],
        p: int = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi = None
    ) -> Tensor:
        return self.schema.shape_function(
            self.context(), bcs, p, index, variables=variables, mi=mi
        )

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

    def transform(self, func: Callable[P, Tensor], kind: str = "value") -> Callable[P, Tensor]:
        return self.schema.transform(self.context(), func, kind)
