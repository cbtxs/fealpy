import pytest

from fealpy.fvm import (
    PisoSolverControls,
    PoissonSolverControls,
    SimpleSolverControls,
)


@pytest.mark.parametrize(
    "controls_type",
    [SimpleSolverControls, PisoSolverControls, PoissonSolverControls],
)
def test_fvm_solver_controls_require_cellwise_p0_space(controls_type):
    with pytest.raises(ValueError, match="space_degree must be 0"):
        controls_type(space_degree=1)


def test_simple_solver_controls_construct_from_model_options():
    controls = SimpleSolverControls.from_mapping(
        {
            "pde": object(),
            "momentum_equation_relaxation": 0.6,
            "pressure_nonorthogonal_max_iter": None,
        }
    )

    assert controls.momentum_equation_relaxation == 0.6
    assert controls.pressure_nonorthogonal_max_iter == 50


def test_simple_solver_controls_default_to_current_recommended_policy():
    controls = SimpleSolverControls()

    assert controls.diffusion_method == "over_relaxed"
    assert controls.gradient_layer_weights == (1.0, 0.25)
    assert controls.gradient_boundary_weight == 1.0
    assert controls.pressure_constraint == "nullspace"
    assert controls.pressure_response_scheme == "simple"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"
    assert controls.momentum_nonorthogonal_atol == 1.0e-12
    assert controls.pressure_nonorthogonal_atol == 1.0e-12
    assert controls.face_flux_correction_scheme == "none"
    assert controls.face_flux_quadrature_order == 3


def test_simple_solver_controls_accept_gradient_reconstruction_weights():
    controls = SimpleSolverControls(
        gradient_layer_weights=(1.0, 0.05),
        gradient_boundary_weight=0.75,
    )

    assert controls.gradient_layer_weights == (1.0, 0.05)
    assert controls.gradient_boundary_weight == 0.75


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gradient_layer_weights": (1.0,)}, "gradient_layer_weights"),
        ({"gradient_layer_weights": (1.0, -0.1)}, "gradient_layer_weights"),
        ({"gradient_layer_weights": (0.0, 0.0)}, "gradient_layer_weights"),
        ({"gradient_boundary_weight": -0.1}, "gradient_boundary_weight"),
    ],
)
def test_simple_solver_controls_reject_invalid_gradient_reconstruction_weights(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        SimpleSolverControls(**kwargs)


def test_simple_solver_controls_accept_simplec_pressure_response_scheme():
    controls = SimpleSolverControls(pressure_response_scheme="simplec")

    assert controls.pressure_response_scheme == "simplec"


def test_simple_solver_controls_reject_unknown_pressure_response_scheme():
    with pytest.raises(ValueError, match="pressure_response_scheme"):
        SimpleSolverControls(pressure_response_scheme="unknown")


def test_simple_solver_controls_accept_second_order_rhie_chow_velocity_scheme():
    controls = SimpleSolverControls(
        rhie_chow_velocity_scheme="second_order_reconstructed"
    )

    assert controls.rhie_chow_velocity_scheme == "second_order_reconstructed"


def test_simple_solver_controls_reject_unknown_rhie_chow_velocity_scheme():
    with pytest.raises(ValueError, match="rhie_chow_velocity_scheme"):
        SimpleSolverControls(rhie_chow_velocity_scheme="unknown")


def test_simple_solver_controls_accept_cell_anchored_face_flux_correction():
    controls = SimpleSolverControls(
        face_flux_correction_scheme="cell_anchored_quadratic",
        face_flux_quadrature_order=4,
        face_flux_max_stencil_layers=5,
    )

    assert controls.face_flux_correction_scheme == "cell_anchored_quadratic"
    assert controls.face_flux_quadrature_order == 4
    assert controls.face_flux_max_stencil_layers == 5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"face_flux_correction_scheme": "unknown"}, "face_flux_correction_scheme"),
        ({"face_flux_quadrature_order": 1}, "face_flux_quadrature_order"),
        ({"face_flux_max_stencil_layers": 0}, "face_flux_max_stencil_layers"),
    ],
)
def test_simple_solver_controls_reject_invalid_face_flux_correction(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        SimpleSolverControls(**kwargs)


def test_piso_solver_controls_default_to_current_recommended_policy():
    controls = PisoSolverControls()

    assert controls.diffusion_method == "over_relaxed"
    assert controls.gradient_layer_weights == (1.0, 0.25)
    assert controls.gradient_boundary_weight == 1.0
    assert controls.pressure_constraint == "nullspace"
    assert controls.momentum_solve_strategy == "component"
    assert controls.momentum_component_matrix_policy == "shared"
    assert controls.momentum_linear_solver == "scipy_bicgstab"
    assert controls.pressure_nullspace_linear_solver == "petsc_gmres_hypre"
    assert controls.momentum_nonorthogonal_atol == 1.0e-12
    assert controls.pressure_nonorthogonal_atol == 1.0e-12


def test_piso_solver_controls_accept_gradient_reconstruction_weights():
    controls = PisoSolverControls(
        gradient_layer_weights=(1.0, 0.05),
        gradient_boundary_weight=0.75,
    )

    assert controls.gradient_layer_weights == (1.0, 0.05)
    assert controls.gradient_boundary_weight == 0.75


@pytest.mark.parametrize("controls_type", [SimpleSolverControls, PisoSolverControls])
def test_collocated_solver_controls_accept_complete_diffusion_variant(controls_type):
    controls = controls_type(
        diffusion_method="bounded_over_relaxed",
        diffusion_nonorthogonal_eps=0.1,
    )

    assert controls.diffusion_method == "bounded_over_relaxed"
    assert controls.diffusion_nonorthogonal_eps == 0.1


@pytest.mark.parametrize("controls_type", [SimpleSolverControls, PisoSolverControls])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diffusion_method": "unknown"}, "diffusion_method"),
        ({"diffusion_nonorthogonal_eps": 0.0}, "diffusion_nonorthogonal_eps"),
    ],
)
def test_collocated_solver_controls_reject_invalid_diffusion_variant(
    controls_type, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        controls_type(**kwargs)


@pytest.mark.parametrize("controls_type", [SimpleSolverControls, PisoSolverControls])
@pytest.mark.parametrize(
    "name",
    ["momentum_nonorthogonal_atol", "pressure_nonorthogonal_atol"],
)
def test_collocated_solver_controls_reject_negative_nonorthogonal_atol(
    controls_type, name
):
    with pytest.raises(ValueError, match=name):
        controls_type(**{name: -1.0})
