import numpy as np
import pytest

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


def test_face_gradient_average_interpolation_does_not_rebuild_geometry(monkeypatch):
    import fealpy.fvm.face_gradient as face_gradient_module
    from fealpy.fvm import reconstruct_face_gradient

    mesh = _quad_mesh()
    cell_gradient = bm.ones((mesh.number_of_cells(), mesh.geo_dimension()))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("face gradient should reuse its FVMGeometry instance")

    monkeypatch.setattr(
        face_gradient_module,
        "face_interpolation_owner_weight",
        fail_if_called,
        raising=False,
    )

    face_gradient = reconstruct_face_gradient(
        mesh,
        cell_gradient,
        interpolation_method="average",
    )

    assert face_gradient.shape == (mesh.number_of_faces(), mesh.geo_dimension())


def test_face_interpolation_average_uses_fvm_geometry_and_distance_is_removed(monkeypatch):
    import fealpy.fvm.fvm_geometry as geometry_module
    from fealpy.fvm import face_interpolation_owner_weight

    mesh = _quad_mesh()
    real_geometry_cls = geometry_module.FVMGeometry

    class FakeGeometry:
        def __init__(self, mesh, *, index=slice(None)):
            real = real_geometry_cls(mesh, index=index)
            self.is_internal = real.is_internal
            self.mag_S_f = 2.5 * real.mag_S_f

        def linear_owner_weight(self):
            raise AssertionError("average interpolation should not use linear weights")

    monkeypatch.setattr(geometry_module, "FVMGeometry", FakeGeometry)
    geometry = FakeGeometry(mesh)

    average = face_interpolation_owner_weight(mesh, method="average")

    expected_average = np.where(np.asarray(geometry.is_internal), 0.5, 1.0)

    np.testing.assert_allclose(np.asarray(average), expected_average)
    with pytest.raises(ValueError, match="average.*linear"):
        face_interpolation_owner_weight(mesh, method="distance")


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


def test_collocated_mass_residual_scatter_uses_single_geometry(monkeypatch):
    import fealpy.fvm.simple_residual as residual_module
    from fealpy.fvm import FVMGeometry, collocated_mass_residual

    mesh = _quad_mesh()
    face_velocity = bm.ones((mesh.number_of_faces(), mesh.geo_dimension()))
    calls = []

    class CountingGeometry(FVMGeometry):
        def scatter_face_flux_to_cells(self, face_flux):
            calls.append(np.asarray(face_flux).copy())
            return super().scatter_face_flux_to_cells(face_flux)

    monkeypatch.setattr(residual_module, "FVMGeometry", CountingGeometry)

    residual = collocated_mass_residual(mesh, face_velocity)

    assert residual >= 0.0
    assert len(calls) == 1


def test_rhie_chow_interpolation_reuses_instance_geometry_for_owner_weights(monkeypatch):
    import fealpy.fvm.collocated_face_velocity_reconstruct as face_velocity_module
    from fealpy.fvm import FVMGeometry, RhieChowInterpolation

    mesh = _quad_mesh()

    class WeightedGeometry(FVMGeometry):
        def linear_owner_weight(self):
            return bm.where(
                self.is_internal,
                0.125 * bm.ones_like(self.mag_S_f),
                1.0,
            )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("RhieChowInterpolation should reuse its FVMGeometry")

    monkeypatch.setattr(face_velocity_module, "FVMGeometry", WeightedGeometry)
    monkeypatch.setattr(
        face_velocity_module,
        "face_interpolation_owner_weight",
        fail_if_called,
        raising=False,
    )

    cell_velocity = np.arange(2 * mesh.number_of_cells(), dtype=float).reshape(
        mesh.number_of_cells(),
        2,
    )
    flat_velocity = cell_velocity.flatten(order="F")
    ap = bm.ones(2 * mesh.number_of_cells())

    face_velocity, _ = RhieChowInterpolation(
        mesh,
        velocity_interpolation="linear",
    ).cell_velocity_to_face(flat_velocity, ap)

    geometry = WeightedGeometry(mesh)
    weight = np.asarray(geometry.linear_owner_weight())
    owner = np.asarray(geometry.owner)
    neighbour = np.asarray(geometry.neighbour)
    expected = (
        weight[:, None] * cell_velocity[owner]
        + (1.0 - weight)[:, None] * cell_velocity[neighbour]
    )

    np.testing.assert_allclose(
        np.asarray(face_velocity),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_rhie_chow_pressure_gradient_reuses_instance_geometry():
    from fealpy.fvm import RhieChowInterpolation

    mesh = _quad_mesh()
    rhie_chow = RhieChowInterpolation(mesh)

    assert rhie_chow.gradient_reconstruct.fvm_geometry is rhie_chow.fvm_geometry
