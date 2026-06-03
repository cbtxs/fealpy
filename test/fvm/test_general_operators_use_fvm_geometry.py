import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh import QuadrangleMesh, TriangleMesh


class ShiftedOperatorGeometry:
    def __init__(self, mesh, *, index=slice(None)):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        real = FVMGeometry(mesh, index=index)
        is_boundary = np.asarray(real.is_boundary)

        face_center = np.asarray(real.face_center).copy()
        face_center[is_boundary, 0] += 3.0
        face_center[is_boundary, 1] -= 2.0

        S_f = np.asarray(real.S_f).copy()
        S_f *= 2.5

        self.mesh = mesh
        self.index = index
        self.face_to_cell = real.face_to_cell
        self.owner = real.owner
        self.neighbour = real.neighbour
        self.is_internal = real.is_internal
        self.is_boundary = real.is_boundary
        self.cell_center = real.cell_center
        self.face_center = bm.array(face_center)
        self.d_f = real.d_f
        self.mag_d_f = real.mag_d_f
        self.S_f = bm.array(S_f)
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        self.n_f = self.S_f / self.mag_S_f[:, None]
        self.boundary_owner_to_face_vector = (
            self.face_center[self.is_boundary] - self.cell_center[self.owner[self.is_boundary]]
        )
        self.boundary_normal_distance = bm.einsum(
            "ij,ij->i",
            self.boundary_owner_to_face_vector,
            self.n_f[self.is_boundary],
        )

    def linear_owner_weight(self):
        return bm.where(
            self.is_internal,
            0.25 * bm.ones_like(self.mag_S_f),
            1.0,
        )

    def bounded_over_relaxed_decomposition(self, eps=0.05):
        return self.S_f, self.mag_S_f, 0.4 * self.S_f

    def over_relaxed_decomposition(self):
        return self.S_f, self.mag_S_f, 0.4 * self.S_f

    def scatter_face_flux_to_cells(self, face_flux):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        return FVMGeometry(self.mesh, index=self.index).scatter_face_flux_to_cells(face_flux)


def _quad_mesh():
    bm.set_backend("numpy")
    return QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)


def test_gradient_reconstruct_boundary_data_uses_fvm_geometry(monkeypatch):
    import fealpy.fvm.gradient_reconstruct as gradient_module
    from fealpy.fvm import GradientReconstruct

    mesh = _quad_mesh()
    monkeypatch.setattr(gradient_module, "FVMGeometry", ShiftedOperatorGeometry)
    geometry = ShiftedOperatorGeometry(mesh)
    field = bm.zeros(mesh.number_of_cells())
    gd = lambda p: p[:, 0] - 0.5 * p[:, 1]

    grad = GradientReconstruct(mesh, gd=gd).cell_gradient(field)

    boundary_faces = np.flatnonzero(np.asarray(geometry.is_boundary))
    owner = np.asarray(geometry.owner)[boundary_faces]
    nonzero_owner = set(np.where(np.linalg.norm(np.asarray(grad), axis=1) > 1.0e-12)[0])

    assert nonzero_owner == set(owner.tolist())


def test_face_interpolation_average_and_distance_use_fvm_geometry(monkeypatch):
    import fealpy.fvm.face_interpolation as interpolation_module
    from fealpy.fvm import face_interpolation_owner_weight

    mesh = _quad_mesh()
    monkeypatch.setattr(
        interpolation_module,
        "FVMGeometry",
        ShiftedOperatorGeometry,
    )
    geometry = ShiftedOperatorGeometry(mesh)

    average = face_interpolation_owner_weight(mesh, method="average")
    distance = face_interpolation_owner_weight(mesh, method="distance")

    expected_average = np.where(np.asarray(geometry.is_internal), 0.5, 1.0)
    owner_dist = np.linalg.norm(
        np.asarray(geometry.face_center) - np.asarray(geometry.cell_center)[np.asarray(geometry.owner)],
        axis=1,
    )
    neighbour_dist = np.linalg.norm(
        np.asarray(geometry.cell_center)[np.asarray(geometry.neighbour)] - np.asarray(geometry.face_center),
        axis=1,
    )
    total = owner_dist + neighbour_dist
    expected_distance = np.where(total > 0.0, neighbour_dist / total, 0.5)
    expected_distance = np.where(np.asarray(geometry.is_internal), expected_distance, 1.0)

    np.testing.assert_allclose(np.asarray(average), expected_average)
    np.testing.assert_allclose(np.asarray(distance), expected_distance)


def test_convection_integrator_uses_fvm_geometry_face_area_vector(monkeypatch):
    import fealpy.fvm.convection_integrator as convection_module
    from fealpy.fvm import ConvectionIntegrator, face_interpolation_owner_weight

    mesh = _quad_mesh()
    monkeypatch.setattr(convection_module, "FVMGeometry", ShiftedOperatorGeometry)
    space = ScaledMonomialSpace2d(mesh, 0)

    local = ConvectionIntegrator(
        coef=bm.ones((mesh.number_of_faces(), mesh.geo_dimension())),
        interpolation="linear",
        q=1,
    ).assembly(space)

    geometry = ShiftedOperatorGeometry(mesh)
    flux = np.einsum(
        "ij,ij->i",
        np.ones((mesh.number_of_faces(), mesh.geo_dimension())),
        np.asarray(geometry.S_f),
    )
    owner_weight = np.asarray(face_interpolation_owner_weight(mesh, method="linear"))
    expected = flux[:, None, None] * np.array(
        [
            [[weight, 1.0 - weight], [-weight, weight - 1.0]]
            for weight in owner_weight
        ]
    )

    np.testing.assert_allclose(np.asarray(local), expected, rtol=1.0e-13, atol=1.0e-13)


def test_cross_diffusion_fetch_uses_fvm_geometry_face_to_cell(monkeypatch):
    import fealpy.fvm.scalar_cross_diffusion_integrator as cross_module
    from fealpy.fvm import ScalarCrossDiffusionIntegrator

    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=1)
    space = ScaledMonomialSpace2d(mesh, 0)
    monkeypatch.setattr(cross_module, "FVMGeometry", ShiftedOperatorGeometry)
    grad_f = bm.ones((mesh.number_of_faces(), mesh.geo_dimension()))

    rhs = LinearForm(space).add_integrator(
        ScalarCrossDiffusionIntegrator(
            bm.zeros(mesh.number_of_cells()),
            grad_f,
        )
    ).assembly()

    geometry = ShiftedOperatorGeometry(mesh)
    face_flux = np.einsum("ij,ij->i", 0.4 * np.asarray(geometry.S_f), np.ones_like(np.asarray(geometry.S_f)))
    face_flux[~np.asarray(geometry.is_internal)] = 0.0
    expected = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    np.testing.assert_allclose(np.asarray(rhs), expected, rtol=1.0e-13, atol=1.0e-13)


def test_collocated_divergence_and_mass_residual_use_fvm_geometry(monkeypatch):
    import fealpy.fvm.div_reconstruct as div_module
    import fealpy.fvm.simple_residual as residual_module
    from fealpy.fvm import DivergenceReconstruct, collocated_mass_residual

    mesh = _quad_mesh()
    monkeypatch.setattr(div_module, "FVMGeometry", ShiftedOperatorGeometry)
    monkeypatch.setattr(residual_module, "FVMGeometry", ShiftedOperatorGeometry)
    face_velocity = bm.ones((mesh.number_of_faces(), mesh.geo_dimension()))

    div = DivergenceReconstruct(mesh).Reconstruct(face_velocity)
    residual = collocated_mass_residual(mesh, face_velocity)

    geometry = ShiftedOperatorGeometry(mesh)
    face_flux = np.einsum("ij,ij->i", np.ones_like(np.asarray(geometry.S_f)), np.asarray(geometry.S_f))
    expected_div = np.asarray(geometry.scatter_face_flux_to_cells(face_flux))

    np.testing.assert_allclose(np.asarray(div), expected_div, rtol=1.0e-13, atol=1.0e-13)
    assert residual >= 0.0
