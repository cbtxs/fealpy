import numpy as np

from fealpy.fem import LinearForm
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh import TriangleMesh
from fealpy.fvm import (
    ConvectionIntegrator,
    GradientReconstruct,
    NSFVMPISOModel,
    NSFVMSimpleModel,
    RhieChowInterpolation,
    ScalarCrossDiffusionIntegrator,
    reconstruct_face_gradient,
)


LINEAR_GRAD = np.array([1.25, -0.5])


def linear_pressure(points):
    return LINEAR_GRAD[0] * points[:, 0] + LINEAR_GRAD[1] * points[:, 1] + 0.75


def linear_velocity(points):
    u = points[:, 0] + 2.0 * points[:, 1] + 3.0
    v = -0.5 * points[:, 0] + 0.25 * points[:, 1] - 1.0
    return np.stack([u, v], axis=-1)


def _skew_two_cell_mesh():
    nodes = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.2, 1.0],
            [1.5, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TriangleMesh(nodes, cells)


def _openfoam_owner_weight(mesh, face):
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    owner, neighbour = e2c[face]
    face_center = np.asarray(mesh.entity_barycenter("face")[face])
    cell_center = np.asarray(mesh.entity_barycenter("cell"))
    sf = np.asarray(mesh.edge_normal()[face])
    own = abs(float(np.dot(sf, face_center - cell_center[owner])))
    nei = abs(float(np.dot(sf, cell_center[neighbour] - face_center)))
    return nei / (own + nei)


def test_convection_integrator_can_use_openfoam_linear_face_weights():
    mesh = _skew_two_cell_mesh()
    space = ScaledMonomialSpace2d(mesh, 0)
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    internal_face = int(np.flatnonzero(e2c[:, 0] != e2c[:, 1])[0])
    face_velocity = np.tile(np.array([[0.7, -0.2]]), (mesh.number_of_faces(), 1))

    local = np.asarray(
        ConvectionIntegrator(
            q=2,
            coef=face_velocity,
            interpolation="linear",
        ).assembly(space)
    )

    weight = _openfoam_owner_weight(mesh, internal_face)
    sf = np.asarray(mesh.edge_normal()[internal_face])
    flux = float(np.dot(sf, face_velocity[internal_face]))
    expected = flux * np.array(
        [
            [weight, 1.0 - weight],
            [-weight, -(1.0 - weight)],
        ]
    )

    assert abs(weight - 0.5) > 1.0e-3
    assert np.linalg.norm(local[internal_face] - expected) < 1.0e-12


def test_rhie_chow_can_use_openfoam_linear_velocity_interpolation():
    mesh = _skew_two_cell_mesh()
    e2c = np.asarray(mesh.edge_to_cell()[:, :2])
    internal_face = int(np.flatnonzero(e2c[:, 0] != e2c[:, 1])[0])
    weight = _openfoam_owner_weight(mesh, internal_face)
    velocity = np.array([[1.0, -2.0], [4.0, 3.0]])
    flat_velocity = velocity.flatten(order="F")
    ap = np.ones(2 * mesh.number_of_cells())

    uf, _ = RhieChowInterpolation(
        mesh,
        velocity_interpolation="linear",
    ).Ucell2edge(flat_velocity, ap)

    owner, neighbour = e2c[internal_face]
    expected = weight * velocity[owner] + (1.0 - weight) * velocity[neighbour]

    assert abs(weight - 0.5) > 1.0e-3
    assert np.linalg.norm(np.asarray(uf[internal_face]) - expected) < 1.0e-12


def test_simple_solver_pressure_gradient_integrator_is_exact_for_linear_pressure():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4, "space_degree": 0}),
    ]

    for model_cls, options in cases:
        model = model_cls({**options, "pbar_log": False})
        points = model.mesh.entity_barycenter("cell")
        pressure = linear_pressure(points)

        grad_p = GradientReconstruct(model.mesh).cell_gradient(pressure)
        pressure_integrator = np.concatenate([
            np.asarray(model.cm * grad_p[:, 0]),
            np.asarray(model.cm * grad_p[:, 1]),
        ])
        expected = np.concatenate([
            np.asarray(model.cm) * LINEAR_GRAD[0],
            np.asarray(model.cm) * LINEAR_GRAD[1],
        ])

        assert np.linalg.norm(pressure_integrator - expected) < 1.0e-12


def test_piso_velocity_pressure_correction_uses_lsq_gradient_and_flat_layout():
    model = NSFVMPISOModel({
        "pde": 3,
        "nx": 4,
        "ny": 4,
        "nt": 1,
        "duration": (0.0, 1.0),
        "space_degree": 0,
        "pbar_log": False,
    })
    points = model.mesh.entity_barycenter("cell")
    pressure_rate = linear_pressure(points)
    velocity = linear_velocity(points)
    u_flat = velocity.flatten(order="F")
    response = 0.25
    a_p = np.concatenate([
        np.asarray(model.cm) / response,
        np.asarray(model.cm) / response,
    ])

    corrected = np.asarray(
        model.velocity_pressure_correction(u_flat, pressure_rate, a_p)
    )

    expected = np.asarray(velocity) - response * LINEAR_GRAD
    assert np.linalg.norm(corrected - expected.flatten(order="F")) < 1.0e-12


def test_rhie_chow_gradient_difference_vanishes_for_linear_internal_pressure():
    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "space_degree": 0,
        "pbar_log": False,
    })
    points = model.mesh.entity_barycenter("cell")
    pressure = linear_pressure(points)
    e2c = model.mesh.edge_to_cell()[:, :2]
    is_internal = np.asarray(e2c[:, 0] != e2c[:, 1])

    grad_diff = np.asarray(RhieChowInterpolation(model.mesh).GradientDifference(pressure))

    assert np.max(np.abs(grad_diff[is_internal])) < 1.0e-12


def test_cross_diffusion_gradient_path_is_exact_for_linear_velocity():
    model = NSFVMSimpleModel({
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "space_degree": 0,
        "pbar_log": False,
    })
    points = model.mesh.entity_barycenter("cell")
    velocity = linear_velocity(points)
    exact_cell_grad = np.array([[1.0, 2.0], [-0.5, 0.25]])

    gradient = GradientReconstruct(
        model.mesh,
        method="green_gauss",
        gd=linear_velocity,
        bc_type="dirichlet",
    )
    cell_grad = np.asarray(
        gradient.cell_gradient(velocity)
    )
    face_grad = np.asarray(reconstruct_face_gradient(model.mesh, cell_grad))

    assert np.linalg.norm(cell_grad - exact_cell_grad) < 1.0e-12
    assert np.linalg.norm(face_grad - exact_cell_grad) < 1.0e-12


def _boundary_all_cross_diffusion(model, uh):
    space = getattr(model, "velocity_space", getattr(model, "uspace", None))
    lform = LinearForm(space)
    U = np.stack((uh[:model.NC], uh[model.NC:]), axis=1)
    grad_u = model.velocity_gradient.cell_gradient(U)
    grad_f = reconstruct_face_gradient(model.mesh, grad_u)
    lform.add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            boundary_policy="all",
        )
    )
    return np.asarray(lform.assembly())


def _cell_velocity_for_model(model, points):
    if hasattr(model.pde, "velocity_0"):
        return model.pde.velocity_0(points, getattr(model, "duration", (0.0,))[0])
    return model.pde.velocity(points)


def test_collocated_cross_diffusion_keeps_boundary_correction_for_current_bc_layer():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4, "space_degree": 0}),
        (
            NSFVMPISOModel,
            {
                "pde": 3,
                "nx": 4,
                "ny": 4,
                "nt": 1,
                "duration": (0.0, 1.0),
                "space_degree": 0,
            },
        ),
    ]

    for model_cls, options in cases:
        model = model_cls({**options, "pbar_log": False})
        points = model.mesh.entity_barycenter("cell")
        velocity = _cell_velocity_for_model(model, points)
        uh = np.asarray(velocity).flatten(order="F")

        actual = np.asarray(model.compute_cross_diffusion(uh))
        expected = _boundary_all_cross_diffusion(model, uh)

        assert np.linalg.norm(actual - expected) < 1.0e-12


def test_exact_pressure_lsq_gradient_decreases_on_solver_pde_meshes():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "space_degree": 0}),
    ]

    for model_cls, base_options in cases:
        errors = []
        for nx in (8, 16):
            model = model_cls({
                **base_options,
                "nx": nx,
                "ny": nx,
                "pbar_log": False,
            })
            points = model.mesh.entity_barycenter("cell")
            pressure = model.pde.pressure(points)
            grad_p = GradientReconstruct(model.mesh).cell_gradient(
                pressure
            )
            exact = model.pde.grad_pressure(points)
            errors.append(
                np.sqrt(
                    np.sum(
                        np.asarray(model.cm)[:, None]
                        * (np.asarray(grad_p) - np.asarray(exact)) ** 2
                    )
                )
            )

        assert errors[1] < 0.8 * errors[0]
