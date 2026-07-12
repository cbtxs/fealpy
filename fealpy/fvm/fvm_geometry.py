"""Owner-oriented finite-volume face geometry."""

from inspect import signature

from fealpy.backend import backend_manager as bm
from fealpy.typing import Index, TensorLike, _S


def boundary_face_flag(points, threshold):
    """Evaluate a boundary-face threshold on face centers.

    ``lambda x`` receives x coordinates, ``lambda y`` receives y coordinates,
    ``lambda z`` receives z coordinates, and other one-argument callables such
    as ``lambda p`` receive the full point array.
    """
    if not callable(threshold):
        raise ValueError("threshold must be a callable boundary face selector.")

    argument = points
    try:
        params = list(signature(threshold).parameters.values())
    except (TypeError, ValueError):
        params = ()
    positional = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1:
        axis = {"x": 0, "y": 1, "z": 2}.get(positional[0].name)
        if axis is not None:
            if axis >= points.shape[1]:
                raise ValueError(
                    f"threshold requests coordinate axis {axis}, "
                    f"but boundary face centers have dimension {points.shape[1]}."
                )
            argument = points[:, axis]

    flag = bm.array(threshold(argument), dtype=bm.bool)
    if flag.shape == (points.shape[0],):
        return flag
    raise ValueError(
        "threshold must return a boolean array with one entry per boundary face."
    )


def selected_boundary_faces(geometry, threshold=None, *, default_all=False):
    """Return boundary faces selected by a face-center threshold callable.

    ``threshold=None`` returns ``None`` by default, matching solver boundary
    conditions where the absence of a selector means no patch.  Low-level
    reconstruction operators can pass ``default_all=True`` when boundary data
    without a selector should apply to all boundary faces.
    """
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    if threshold is None:
        return boundary_faces if default_all else None

    face_centers = geometry.face_center[boundary_faces]
    flag = boundary_face_flag(face_centers, threshold)
    if not bool(bm.to_numpy(bm.any(flag))):
        return boundary_faces[:0]
    return boundary_faces[flag]


class FVMGeometry:
    """Collect basic owner-oriented face geometry for FVM operators.

    The class intentionally stores only mesh geometry.  It does not define PDE
    coefficients, boundary values, gradient reconstruction, Rhie-Chow response,
    or pressure-correction data.
    """

    def __init__(self, mesh, *, index: Index = _S) -> None:
        self.mesh = mesh
        self.index = index

        required = ("geo_dimension", "entity_barycenter", "entity_measure", "number_of_cells")
        missing = [name for name in required if not hasattr(mesh, name)]
        if missing:
            raise RuntimeError(
                "FVMGeometry requires a mesh with FVM geometry methods: "
                + ", ".join(missing)
            )
        if not hasattr(mesh, "face_to_cell") and not hasattr(mesh, "edge_to_cell"):
            raise RuntimeError("FVMGeometry requires face_to_cell() or edge_to_cell().")

        if hasattr(mesh, "face_to_cell"):
            face_to_cell = mesh.face_to_cell(index=index)
        else:
            face_to_cell = mesh.edge_to_cell(index=index)
        if face_to_cell.ndim == 1:
            face_to_cell = face_to_cell[None, :]
        face_to_cell = face_to_cell[:, :2]
        self.face_to_cell = face_to_cell
        self.owner = face_to_cell[:, 0]
        self.neighbour = face_to_cell[:, 1]
        self.is_internal = self.owner != self.neighbour
        self.is_boundary = ~self.is_internal

        self.cell_center = mesh.entity_barycenter("cell")
        self.face_center = mesh.entity_barycenter("face", index=index)
        if self.face_center.ndim == 1:
            self.face_center = self.face_center[None, :]

        owner_to_neighbour = self.cell_center[self.neighbour] - self.cell_center[self.owner]
        owner_to_face = self.face_center - self.cell_center[self.owner]
        self.d_f = bm.where(self.is_internal[:, None], owner_to_neighbour, owner_to_face)
        self.mag_d_f = bm.linalg.norm(self.d_f, axis=1)
        if bm.any(self.mag_d_f <= 0.0):
            raise ValueError("face centre vector has zero length.")

        if mesh.geo_dimension() == 2:
            S_f = mesh.edge_normal(index=index)
        elif hasattr(mesh, "face_normal"):
            S_f = mesh.face_normal(index=index)
        else:
            mag_S_f = mesh.entity_measure("face", index=index)
            S_f = mag_S_f[:, None] * mesh.face_unit_normal(index=index)
        if S_f.ndim == 1:
            S_f = S_f[None, :]

        projection = bm.einsum("ij,ij->i", S_f, self.d_f)
        S_f = bm.where(projection[:, None] < 0.0, -S_f, S_f)
        self.S_f = S_f
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        if bm.any(self.mag_S_f <= 0.0):
            raise ValueError("face area vector has zero length.")
        self.n_f = self.S_f / self.mag_S_f[:, None]

        owner_projection = bm.einsum("ij,ij->i", self.S_f, self.d_f)
        if bm.any(owner_projection <= 0.0):
            raise ValueError("face area vector is not owner-oriented.")

        self.boundary_owner_to_face_vector = owner_to_face[self.is_boundary]
        boundary_normal = self.n_f[self.is_boundary]
        self.boundary_normal_distance = bm.einsum("ij,ij->i", self.boundary_owner_to_face_vector, boundary_normal)
        if bm.any(self.boundary_normal_distance <= 0.0):
            raise ValueError("boundary face has zero owner-normal distance.")

    def over_relaxed_decomposition(self):
        """Return ``(E_f, |E_f|, T_f)`` from over-relaxed decomposition.

        The decomposition is

            E_f = (S_f · S_f) / (d_f · S_f) d_f,
            T_f = S_f - E_f.

        It is kept outside ``__init__`` because it is a numerical
        decomposition strategy, not primitive mesh geometry.
        """
        S_dot_S = bm.einsum("ij,ij->i", self.S_f, self.S_f)
        d_dot_S = bm.einsum("ij,ij->i", self.d_f, self.S_f)
        if bm.any(d_dot_S <= 0.0):
            raise ValueError("over-relaxed decomposition has invalid d_f dot S_f.")
        E_f = (S_dot_S / d_dot_S)[:, None] * self.d_f
        mag_E_f = bm.linalg.norm(E_f, axis=1)
        T_f = self.S_f - E_f
        return E_f, mag_E_f, T_f

    def bounded_over_relaxed_decomposition(self, eps: float = 0.05):
        """Return bounded over-relaxed ``(E_f, |E_f|, T_f)``.

        The bounded form uses

            E_f = |S_f| / max(n_f · d_f, eps |d_f|) d_f,
            T_f = S_f - E_f.

        It is a non-orthogonal diffusion strategy, not primitive mesh geometry.
        """
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        projection = bm.einsum("ij,ij->i", self.n_f, self.d_f)
        denominator = bm.maximum(projection, eps * self.mag_d_f)
        E_f = (self.mag_S_f / denominator)[:, None] * self.d_f
        mag_E_f = bm.linalg.norm(E_f, axis=1)
        T_f = self.S_f - E_f
        return E_f, mag_E_f, T_f

    def normal_distance(self, faces: Index = _S):
        """Return the projection of ``d_f`` onto the owner-oriented unit normal."""
        d_f = self.d_f[faces]
        n_f = self.n_f[faces]
        if d_f.ndim == 1:
            return bm.einsum("i,i->", d_f, n_f)
        return bm.einsum("ij,ij->i", d_f, n_f)

    def linear_owner_weight(self):
        """Return owner-side linear interpolation weights on faces.

        Internal faces use owner/neighbour distances projected onto the face
        area vector direction.  Boundary faces return one because no real
        neighbour cell participates in the interpolation.
        """
        owner_dist = bm.abs(bm.einsum("ij,ij->i", self.S_f, self.face_center - self.cell_center[self.owner]))
        neighbour_dist = bm.abs(bm.einsum("ij,ij->i", self.S_f, self.cell_center[self.neighbour] - self.face_center))
        total_dist = owner_dist + neighbour_dist
        weight = bm.where(total_dist > 0.0, neighbour_dist / total_dist, 0.5)
        return bm.where(self.is_internal, weight, 1.0)

    def scatter_face_flux_to_cells(self, face_flux):
        """Scatter owner-oriented face fluxes to cell flux sums.

        ``face_flux[f]`` must be oriented with this geometry's ``S_f``.  For an
        internal face between owner ``P`` and neighbour ``N``, a positive
        flux means flow in the ``P -> N`` direction, so the cell flux sums are

            cell_sum[P] += face_flux[f],
            cell_sum[N] -= face_flux[f].

        For a boundary face, only the owner cell receives the face contribution.
        The returned value is the finite-volume boundary flux sum on each cell;
        it is not divided by cell volume and should not be interpreted as a
        cell-average divergence.
        """
        face_flux = bm.array(face_flux)
        if face_flux.ndim == 0:
            raise ValueError("face_flux must have at least one dimension.")
        if face_flux.shape[0] != self.owner.shape[0]:
            raise ValueError(
                f"face_flux has {face_flux.shape[0]} faces, expected "
                f"{self.owner.shape[0]}."
            )

        result = bm.zeros(
            (self.mesh.number_of_cells(),) + tuple(face_flux.shape[1:]),
            dtype=face_flux.dtype,
        )
        result = bm.index_add(result, self.owner, face_flux, axis=0)
        return bm.index_add(
            result,
            self.neighbour[self.is_internal],
            face_flux[self.is_internal],
            axis=0,
            alpha=-1,
        )


def face_interpolation_owner_weight(
    mesh,
    *,
    method: str = "linear",
    index: Index = _S,
    geometry: FVMGeometry | None = None,
) -> TensorLike:
    """Return owner-side face interpolation weights.

    ``linear`` is the geometry-consistent face interpolation used by the
    collocated FVM operators.  ``average`` is the arithmetic central weight.
    Boundary faces return one because no neighbour cell participates.
    """
    if method not in {"average", "linear"}:
        raise ValueError("method must be 'average' or 'linear'.")

    geometry = geometry if geometry is not None else FVMGeometry(mesh, index=index)
    if method == "linear":
        return geometry.linear_owner_weight()

    weight = 0.5 * bm.ones_like(geometry.mag_S_f)
    return bm.where(geometry.is_internal, weight, 1.0)
