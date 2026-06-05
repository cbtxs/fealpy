import numpy as np

from fealpy.backend import backend_manager as bm
from fealpy.mesh import QuadrangleMesh
from fealpy.sparse import spdiags


class ShiftedBoundaryGeometry:
    def __init__(self, mesh):
        from fealpy.fvm.fvm_geometry import FVMGeometry

        real = FVMGeometry(mesh)
        is_boundary = np.asarray(real.is_boundary)
        face_center = np.asarray(real.face_center).copy()
        face_center[is_boundary, 0] += 5.0
        face_center[is_boundary, 1] += 7.0

        S_f = np.asarray(real.S_f).copy()
        S_f[is_boundary] *= 3.0

        self.owner = real.owner
        self.neighbour = real.neighbour
        self.face_to_cell = real.face_to_cell
        self.is_internal = real.is_internal
        self.is_boundary = real.is_boundary
        self.d_f = real.d_f
        self.face_center = bm.array(face_center)
        self.S_f = bm.array(S_f)
        self.mag_S_f = bm.linalg.norm(self.S_f, axis=1)
        self.mag_d_f = real.mag_d_f
        self._mag_E_f = 2.0 * real.mag_d_f

    def over_relaxed_decomposition(self):
        return self.S_f, self._mag_E_f, bm.zeros_like(self.S_f)


def _mesh():
    bm.set_backend("numpy")
    return QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)


def _boundary_faces(geometry):
    return np.flatnonzero(np.asarray(geometry.is_boundary))


def test_neumann_diffusion_uses_fvm_geometry_for_boundary_face_integral(monkeypatch):
    import fealpy.fvm.neumann_bc as neumann_module
    from fealpy.fvm import NeumannBC

    mesh = _mesh()
    monkeypatch.setattr(
        neumann_module,
        "FVMGeometry",
        ShiftedBoundaryGeometry,
        raising=False,
    )
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    gd = lambda points: points[:, 0] - 0.5 * points[:, 1]

    actual = NeumannBC(mesh, gd).DiffusionApply(bm.zeros(mesh.number_of_cells()))

    expected = np.zeros(mesh.number_of_cells())
    np.add.at(
        expected,
        np.asarray(geometry.owner)[boundary_faces],
        np.asarray(gd(geometry.face_center[boundary_faces]))
        * np.asarray(geometry.mag_S_f)[boundary_faces],
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_dirichlet_diffusion_uses_fvm_geometry_for_face_points_and_owners(
    monkeypatch,
):
    import fealpy.fvm.dirichlet_bc as dirichlet_module
    from fealpy.fvm import DirichletBC

    mesh = _mesh()
    monkeypatch.setattr(dirichlet_module, "FVMGeometry", ShiftedBoundaryGeometry)
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    gd = lambda points: points[:, 0] + 0.25 * points[:, 1]
    A = spdiags(
        bm.zeros(mesh.number_of_cells()),
        0,
        mesh.number_of_cells(),
        mesh.number_of_cells(),
    )

    _, actual = DirichletBC(mesh, gd).DiffusionApply(
        A,
        bm.zeros(mesh.number_of_cells()),
    )

    expected = np.zeros(mesh.number_of_cells())
    np.add.at(
        expected,
        np.asarray(geometry.owner)[boundary_faces],
        2.0 * np.asarray(gd(geometry.face_center[boundary_faces])),
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_dirichlet_divergence_uses_fvm_geometry_for_boundary_flux(monkeypatch):
    import fealpy.fvm.dirichlet_bc as dirichlet_module
    from fealpy.fvm import DirichletBC

    mesh = _mesh()
    monkeypatch.setattr(dirichlet_module, "FVMGeometry", ShiftedBoundaryGeometry)
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)

    def gd(points):
        return bm.stack([points[:, 0] + 1.0, points[:, 1] - 2.0], axis=1)

    actual = DirichletBC(mesh, gd).DivApply(bm.zeros(2 * mesh.number_of_cells()))

    expected = np.zeros(2 * mesh.number_of_cells())
    owner = np.asarray(geometry.owner)[boundary_faces]
    value = np.asarray(gd(geometry.face_center[boundary_faces]))
    S_f = np.asarray(geometry.S_f)[boundary_faces]
    np.add.at(expected, owner, -value[:, 0] * S_f[:, 0])
    np.add.at(expected, owner + mesh.number_of_cells(), -value[:, 1] * S_f[:, 1])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_dirichlet_convection_uses_fvm_geometry_for_boundary_flux(monkeypatch):
    import fealpy.fvm.dirichlet_bc as dirichlet_module
    from fealpy.fvm import DirichletBC

    mesh = _mesh()
    monkeypatch.setattr(dirichlet_module, "FVMGeometry", ShiftedBoundaryGeometry)
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    coef = bm.ones((mesh.number_of_faces(), 2))

    def gd(points):
        return bm.stack([points[:, 0] - 2.0, 0.5 * points[:, 1]], axis=1)

    actual = DirichletBC(mesh, gd).ConvectionApply(
        bm.zeros(2 * mesh.number_of_cells()),
        coef,
    )

    expected = np.zeros(2 * mesh.number_of_cells())
    owner = np.asarray(geometry.owner)[boundary_faces]
    value = np.asarray(gd(geometry.face_center[boundary_faces]))
    flux = np.einsum(
        "ij,ij->i",
        np.asarray(coef)[boundary_faces],
        np.asarray(geometry.S_f)[boundary_faces],
    )
    np.add.at(expected, owner, -flux * value[:, 0])
    np.add.at(expected, owner + mesh.number_of_cells(), -flux * value[:, 1])
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)


def test_engineering_boundary_conditions_use_fvm_geometry_for_patch_faces(
    monkeypatch,
):
    import fealpy.fvm.engineering_boundary_conditions as bc_module
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    mesh = _mesh()
    monkeypatch.setattr(
        bc_module,
        "FVMGeometry",
        ShiftedBoundaryGeometry,
        raising=False,
    )
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    bc = EngineeringBoundaryConditions(
        mesh,
        patches=[BoundaryPatch("shifted", lambda p: p[:, 0] > 4.0)],
        conditions=[
            BoundaryCondition(
                "velocity",
                "shifted",
                "dirichlet",
                lambda p: bm.ones_like(p),
            )
        ],
    )

    selected_faces, selected_values = bc.boundary_face_velocity("velocity")

    np.testing.assert_array_equal(np.asarray(selected_faces), boundary_faces)
    np.testing.assert_allclose(
        np.asarray(selected_values),
        np.ones((boundary_faces.shape[0], mesh.geo_dimension())),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_boundary_condition_data_use_fvm_geometry_for_boundary_velocity():
    from fealpy.fvm import BoundaryConditionData, PDEBoundaryConditions

    mesh = _mesh()
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    bc = BoundaryConditionData(
        velocity_dirichlet=lambda p: bm.ones_like(p),
        velocity_dirichlet_threshold=lambda p: p[:, 0] > 4.0,
    ).to_pde_boundary(mesh)
    shifted_bc = PDEBoundaryConditions(
        mesh,
        velocity_dirichlet=bc.velocity_dirichlet,
        velocity_dirichlet_threshold=bc.velocity_dirichlet_threshold,
        geometry_class=ShiftedBoundaryGeometry,
    )

    selected_faces, selected_values = shifted_bc.boundary_face_velocity()

    np.testing.assert_array_equal(np.asarray(selected_faces), boundary_faces)
    np.testing.assert_allclose(
        np.asarray(selected_values),
        np.ones((boundary_faces.shape[0], mesh.geo_dimension())),
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_rhie_chow_pressure_dirichlet_partial_uses_fvm_geometry(monkeypatch):
    import fealpy.fvm.rhie_chow as rhie_chow_module
    from fealpy.fvm import RhieChowInterpolation

    mesh = _mesh()
    monkeypatch.setattr(rhie_chow_module, "FVMGeometry", ShiftedBoundaryGeometry)
    geometry = ShiftedBoundaryGeometry(mesh)
    boundary_faces = _boundary_faces(geometry)
    pressure_dirichlet = lambda p: p[:, 0] - 0.25 * p[:, 1]
    rhie_chow = RhieChowInterpolation(
        mesh,
        pressure_dirichlet=pressure_dirichlet,
        pressure_dirichlet_threshold=lambda p: p[:, 0] > 4.0,
    )

    actual = rhie_chow._apply_pressure_dirichlet_boundary_partial(
        bm.zeros(mesh.number_of_cells()),
        bm.zeros(mesh.number_of_faces()),
    )

    expected = np.zeros(mesh.number_of_faces())
    expected[boundary_faces] = (
        np.asarray(pressure_dirichlet(geometry.face_center[boundary_faces]))
        / np.asarray(geometry.mag_d_f)[boundary_faces]
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1.0e-13, atol=1.0e-13)
