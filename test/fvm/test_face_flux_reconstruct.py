import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fvm import FVMGeometry
from fealpy.mesh import HexahedronMesh, PolygonMesh, TetrahedronMesh


def _vector_quadratic(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return bm.stack(
        [
            1.0 + 2.0 * x - y + x * y + z**2,
            -0.5 + x + 3.0 * z + x**2 - 2.0 * y * z,
            2.0 - y + z + y**2 + x * z,
        ],
        axis=-1,
    )


def _divergence_free_quadratic(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return bm.stack(
        [x**2 + y**2, z**2 - 2.0 * x * y, x**2],
        axis=-1,
    )


def _entity_average(mesh, entity, function, q=5):
    quadrature = mesh.quadrature_formula(q, entity)
    bcs, weights = quadrature.get_quadrature_points_and_weights()
    points = mesh.bc_to_point(bcs)
    return bm.einsum("q,eqd->ed", weights, function(points))


def _vector_quadratic_2d(points):
    x = points[..., 0]
    y = points[..., 1]
    return bm.stack(
        [
            1.0 + 2.0 * x - y + x * y + y**2,
            -0.5 + x + 3.0 * y + x**2 - 2.0 * x * y,
        ],
        axis=-1,
    )


def _polygon_cell_average(mesh, function, q=5):
    integral = mesh.integral(
        lambda points, index: function(points),
        q=q,
        celltype=True,
    )
    return integral / mesh.entity_measure("cell")[:, None]


def _polygon_face_average(mesh, function, q=5):
    bcs, weights = mesh.quadrature_formula(q, "face").get_quadrature_points_and_weights()
    points = bm.einsum("qi,eid->eqd", bcs, mesh.entity("node")[mesh.entity("face")])
    return bm.einsum("q,eqd->ed", weights, function(points))


def test_cell_anchored_quadratic_face_flux_recovers_quadratic_face_average():
    bm.set_backend("numpy")
    from fealpy.fvm import CellAnchoredQuadraticFaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_average = _entity_average(mesh, "cell", _vector_quadratic)
    exact_face_average = _entity_average(mesh, "face", _vector_quadratic)

    reconstruct = CellAnchoredQuadraticFaceFluxReconstruct(
        mesh,
        geometry=geometry,
    )
    face_average = reconstruct.face_average(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
    )
    flux = reconstruct.reconstruct(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
    )

    np.testing.assert_allclose(
        bm.to_numpy(face_average),
        bm.to_numpy(exact_face_average),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        bm.to_numpy(flux),
        bm.to_numpy(bm.einsum("fi,fi->f", exact_face_average, geometry.S_f)),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    diagnostics = reconstruct.diagnostics()
    assert diagnostics["minimum_rank"] == 9
    assert diagnostics["failed_cell_count"] == 0
    assert diagnostics["maximum_stencil_layer"] <= 4


def test_face_flux_reconstruct_closes_divergence_free_quadratic_field():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_average = _entity_average(mesh, "cell", _divergence_free_quadratic)
    exact_face_average = _entity_average(
        mesh,
        "face",
        _divergence_free_quadratic,
    )
    base_face_velocity = 0.5 * (
        cell_average[geometry.owner] + cell_average[geometry.neighbour]
    )
    reconstruct = FaceFluxReconstruct(
        mesh,
        method="cell_anchored_quadratic",
        geometry=geometry,
    )

    correction = reconstruct.correction(
        cell_average,
        base_face_velocity,
        boundary_face_average=exact_face_average[geometry.is_boundary],
    )
    base_flux = bm.einsum("fi,fi->f", base_face_velocity, geometry.S_f)
    divergence = geometry.scatter_face_flux_to_cells(base_flux + correction)

    np.testing.assert_allclose(
        bm.to_numpy(divergence),
        0.0,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert reconstruct.diagnostics()["failed_cell_count"] == 0


def test_face_flux_reconstruct_none_variant_returns_zero_correction():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = TetrahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=1,
        ny=1,
        nz=1,
    )
    geometry = FVMGeometry(mesh)
    cell_velocity = bm.zeros(
        (mesh.number_of_cells(), mesh.geo_dimension()),
        dtype=geometry.cell_center.dtype,
    )
    face_velocity = bm.ones(
        (mesh.number_of_faces(), mesh.geo_dimension()),
        dtype=geometry.cell_center.dtype,
    )
    reconstruct = FaceFluxReconstruct(mesh, method="none", geometry=geometry)

    correction = reconstruct.correction(cell_velocity, face_velocity)

    np.testing.assert_allclose(bm.to_numpy(correction), 0.0)


def test_cell_anchored_quadratic_falls_back_to_zero_defect_when_stencil_is_rank_deficient():
    bm.set_backend("numpy")
    from fealpy.fvm import FaceFluxReconstruct

    mesh = HexahedronMesh.from_box(
        box=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        nx=2,
        ny=2,
        nz=2,
    )
    geometry = FVMGeometry(mesh)
    cell_velocity = bm.copy(geometry.cell_center)
    base_face_velocity = 0.5 * (
        cell_velocity[geometry.owner] + cell_velocity[geometry.neighbour]
    )
    reconstruct = FaceFluxReconstruct(
        mesh,
        method="cell_anchored_quadratic",
        geometry=geometry,
    )

    correction = reconstruct.correction(cell_velocity, base_face_velocity)
    diagnostics = reconstruct.diagnostics()

    np.testing.assert_allclose(bm.to_numpy(correction), 0.0)
    assert diagnostics["failed_cell_count"] == mesh.number_of_cells()
    assert diagnostics["fallback_face_count"] == mesh.number_of_faces()


def test_cell_anchored_quadratic_recovers_mixed_polygon_face_averages():
    bm.set_backend("numpy")
    from fealpy.fvm import CellAnchoredQuadraticFaceFluxReconstruct

    node = bm.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ])
    cells = (
        bm.array([
            0, 1, 4, 3,
            3, 4, 7, 6,
            1, 2, 5,
            1, 5, 4,
            4, 5, 8,
            4, 8, 7,
        ], dtype=bm.int64),
        bm.array([0, 4, 8, 11, 14, 17, 20], dtype=bm.int64),
    )
    mesh = PolygonMesh(node, cells)
    geometry = FVMGeometry(mesh)
    cell_average = _polygon_cell_average(mesh, _vector_quadratic_2d)
    exact_face_average = _polygon_face_average(mesh, _vector_quadratic_2d)
    reconstruct = CellAnchoredQuadraticFaceFluxReconstruct(
        mesh,
        geometry=geometry,
    )

    face_average = reconstruct.face_average(
        cell_average,
        boundary_face_average=exact_face_average[geometry.is_boundary],
    )

    np.testing.assert_allclose(
        bm.to_numpy(face_average),
        bm.to_numpy(exact_face_average),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert reconstruct.diagnostics()["failed_cell_count"] == 0
