import numpy as np

from fealpy.fem import LinearForm
from fealpy.fvm import (
    GradientReconstruct,
    NSFVMPISOModel,
    NSFVMSimpleModel,
    RhieChowInterpolation,
    ScalarCrossDiffusionIntegrator,
    StokesFVMSimpleModel,
)


LINEAR_GRAD = np.array([1.25, -0.5])


def linear_pressure(points):
    return LINEAR_GRAD[0] * points[:, 0] + LINEAR_GRAD[1] * points[:, 1] + 0.75


def linear_velocity(points):
    u = points[:, 0] + 2.0 * points[:, 1] + 3.0
    v = -0.5 * points[:, 0] + 0.25 * points[:, 1] - 1.0
    return np.stack([u, v], axis=-1)


def test_simple_solver_pressure_gradient_integrator_is_exact_for_linear_pressure():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4, "space_degree": 0}),
        (StokesFVMSimpleModel, {"pde": 1, "nx": 4, "ny": 4, "space_degree": 0}),
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
    face_grad = np.asarray(gradient.face_gradient(cell_grad))

    assert np.linalg.norm(cell_grad - exact_cell_grad) < 1.0e-12
    assert np.linalg.norm(face_grad - exact_cell_grad) < 1.0e-12


def _boundary_all_cross_diffusion(model, uh):
    space = getattr(model, "velocity_space", getattr(model, "uspace", None))
    lform = LinearForm(space)
    U = np.stack((uh[:model.NC], uh[model.NC:]), axis=1)
    grad_u = model.velocity_gradient.cell_gradient(U)
    grad_f = model.velocity_gradient.face_gradient(grad_u)
    lform.add_integrator(
        ScalarCrossDiffusionIntegrator(
            uh,
            grad_f,
            boundary_policy="all",
        )
    )
    return np.asarray(lform.assembly())


def test_simple_cross_diffusion_keeps_boundary_correction_for_current_bc_layer():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "nx": 4, "ny": 4, "space_degree": 0}),
        (StokesFVMSimpleModel, {"pde": 1, "nx": 4, "ny": 4, "space_degree": 0}),
    ]

    for model_cls, options in cases:
        model = model_cls({**options, "pbar_log": False})
        points = model.mesh.entity_barycenter("cell")
        velocity = model.pde.velocity(points)
        uh = np.asarray(velocity).flatten(order="F")

        actual = np.asarray(model.compute_cross_diffusion(uh))
        expected = _boundary_all_cross_diffusion(model, uh)

        assert np.linalg.norm(actual - expected) < 1.0e-12


def test_exact_pressure_lsq_gradient_decreases_on_solver_pde_meshes():
    cases = [
        (NSFVMSimpleModel, {"pde": 6, "space_degree": 0}),
        (StokesFVMSimpleModel, {"pde": 1, "space_degree": 0}),
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
