import numpy as np
import pytest

from fealpy.backend import backend_manager as bm


def test_collocated_solver_requires_pde_boundary_conditions():
    from fealpy.fvm import CollocatedSimpleSolver
    from fealpy.mesh import TriangleMesh

    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    with pytest.raises(TypeError, match="PDEBoundaryConditions"):
        CollocatedSimpleSolver(
            mesh=mesh,
            diffusion_coef=1.0,
            convection_coef=0.0,
            source=lambda points: bm.zeros_like(points),
            boundary_conditions=object(),
        )


def _constant_velocity(points):
    return bm.stack([bm.ones(points.shape[0]), bm.zeros(points.shape[0])], axis=-1)


def _left_right_engineering_bc(mesh, *, with_pressure=False):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    conditions = [
        BoundaryCondition("velocity", "left", "dirichlet", _constant_velocity),
        BoundaryCondition("velocity", "right", "natural", None),
    ]
    if with_pressure:
        conditions.append(BoundaryCondition("pressure", "right", "dirichlet", 0.0))

    return EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("left", lambda p: bm.abs(p[:, 0]) < 1.0e-12),
            BoundaryPatch("right", lambda p: bm.abs(p[:, 0] - 1.0) < 1.0e-12),
        ],
        conditions=conditions,
    )


def _left_dirichlet_right_neumann_bc(mesh):
    from fealpy.fvm import (
        BoundaryCondition,
        BoundaryPatch,
        EngineeringBoundaryConditions,
    )

    return EngineeringBoundaryConditions(
        mesh,
        patches=[
            BoundaryPatch("left", lambda p: bm.abs(p[:, 0]) < 1.0e-12),
            BoundaryPatch("right", lambda p: bm.abs(p[:, 0] - 1.0) < 1.0e-12),
        ],
        conditions=[
            BoundaryCondition("velocity", "left", "dirichlet", _constant_velocity),
            BoundaryCondition(
                "velocity",
                "right",
                "neumann",
                lambda p: bm.stack(
                    [
                        2.0 * bm.ones(p.shape[0], dtype=p.dtype),
                        3.0 * bm.ones(p.shape[0], dtype=p.dtype),
                    ],
                    axis=-1,
                ),
            ),
            BoundaryCondition("pressure", "right", "reference", 5.0),
        ],
    )


def _simple_model_with_left_right_bc(*, with_pressure=False, **options):
    from fealpy.fvm import NSFVMSimpleModel

    def build_bc(mesh, pde):
        return _left_right_engineering_bc(mesh, with_pressure=with_pressure)

    model_options = {
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "space_degree": 0,
        "log_level": "ERROR",
        "pbar_log": False,
        "boundary_conditions": build_bc,
    }
    model_options.update(options)
    return NSFVMSimpleModel(model_options)


def _simple_model_with_neumann_velocity(**options):
    from fealpy.fvm import NSFVMSimpleModel

    def build_bc(mesh, pde):
        return _left_dirichlet_right_neumann_bc(mesh)

    model_options = {
        "pde": 6,
        "nx": 4,
        "ny": 4,
        "space_degree": 0,
        "log_level": "ERROR",
        "pbar_log": False,
        "boundary_conditions": build_bc,
    }
    model_options.update(options)
    return NSFVMSimpleModel(model_options)


def test_engineering_boundary_conditions_select_only_dirichlet_patches():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)

    bc = _left_right_engineering_bc(mesh)
    pde_bc = bc.to_pde_boundary()

    boundary_faces = mesh.boundary_face_index()
    face_centers = mesh.entity_barycenter("face")[boundary_faces]
    flag = pde_bc.dirichlet_threshold("velocity")(face_centers)
    selected_faces, selected_values = pde_bc.boundary_face_velocity("velocity")

    assert int(bm.to_numpy(bm.sum(flag))) == 4
    assert selected_faces.shape[0] == 4
    assert selected_values.shape == (4, 2)
    assert bool(bm.all(mesh.entity_barycenter("face")[selected_faces][:, 0] < 1.0e-12))
    assert bool(bm.all(selected_values[:, 0] == 1.0))
    assert bool(bm.all(selected_values[:, 1] == 0.0))


def test_boundary_face_velocity_average_integrates_quadratic_dirichlet_data():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)

    def quadratic_velocity(points):
        x = points[..., 0]
        y = points[..., 1]
        return bm.stack([y**2, x**2], axis=-1)

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=quadratic_velocity,
        dirichlet_velocity_threshold=lambda p: bm.abs(p[:, 0]) < 1.0e-12,
    )
    faces, average = boundary.boundary_face_velocity_average(
        "velocity",
        mesh=mesh,
        quadrature_order=3,
    )
    nodes = mesh.entity("node")
    edge = mesh.entity("face")[faces]
    y0 = nodes[edge[:, 0], 1]
    y1 = nodes[edge[:, 1], 1]
    exact_first = (y0**2 + y0 * y1 + y1**2) / 3.0

    np.testing.assert_allclose(bm.to_numpy(average[:, 0]), bm.to_numpy(exact_first))
    np.testing.assert_allclose(bm.to_numpy(average[:, 1]), 0.0)


def test_boundary_face_velocity_average_supports_single_quadrangle():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=1, ny=1)

    def quadratic_velocity(points):
        x = points[..., 0]
        y = points[..., 1]
        return bm.stack([y**2, x**2], axis=-1)

    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_velocity=quadratic_velocity,
    )
    faces, average = boundary.boundary_face_velocity_average(
        "velocity",
        mesh=mesh,
        quadrature_order=3,
    )
    node = mesh.entity("node")
    edge = mesh.entity("face")[faces]
    x0 = node[edge[:, 0], 0]
    x1 = node[edge[:, 1], 0]
    y0 = node[edge[:, 0], 1]
    y1 = node[edge[:, 1], 1]
    exact = bm.stack(
        [
            (y0**2 + y0 * y1 + y1**2) / 3.0,
            (x0**2 + x0 * x1 + x1**2) / 3.0,
        ],
        axis=-1,
    )

    np.testing.assert_allclose(bm.to_numpy(average), bm.to_numpy(exact))


def test_engineering_boundary_face_velocity_average_uses_patch_selection():
    bm.set_backend("numpy")
    from fealpy.mesh import QuadrangleMesh

    mesh = QuadrangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=2, ny=2)
    boundary = _left_right_engineering_bc(mesh)
    pde_boundary = boundary.to_pde_boundary()

    faces, average = pde_boundary.boundary_face_velocity_average(
        "velocity",
        quadrature_order=3,
    )

    assert faces.shape[0] == 2
    assert average.shape == (2, 2)
    assert bool(bm.all(mesh.entity_barycenter("face")[faces][:, 0] < 1.0e-12))
    assert bool(bm.all(average[:, 0] == 1.0))
    assert bool(bm.all(average[:, 1] == 0.0))


def test_engineering_boundary_conditions_convert_to_pde_boundary_conditions():
    bm.set_backend("numpy")
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    engineering_bc = _left_right_engineering_bc(mesh, with_pressure=True)
    pde_bc = engineering_bc.to_pde_boundary()

    boundary_faces, boundary_value = pde_bc.boundary_face_velocity("velocity")
    face_centers = mesh.entity_barycenter("face")[boundary_faces]
    pressure_threshold = pde_bc.dirichlet_threshold("pressure")
    pressure_faces = mesh.boundary_face_index()
    pressure_points = mesh.entity_barycenter("face")[pressure_faces]

    assert isinstance(pde_bc, PDEBoundaryConditions)
    assert pde_bc.has_dirichlet("velocity")
    assert pde_bc.has_natural("velocity")
    assert pde_bc.has_dirichlet("pressure")
    assert boundary_faces.shape[0] == 4
    assert boundary_value.shape == (4, 2)
    assert bool(bm.all(face_centers[:, 0] < 1.0e-12))
    assert int(bm.to_numpy(bm.sum(pressure_threshold(pressure_points)))) == 4


def test_pressure_reference_maps_to_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    engineering_bc = _left_dirichlet_right_neumann_bc(mesh)
    pde_bc = engineering_bc.to_pde_boundary()
    boundary_faces = mesh.boundary_face_index()
    boundary_points = mesh.entity_barycenter("face")[boundary_faces]
    flag = pde_bc.dirichlet_threshold("pressure")(boundary_points)
    pressure_value = pde_bc.dirichlet_value("pressure")(boundary_points[flag])

    assert pde_bc.has_dirichlet("pressure")
    assert int(bm.to_numpy(bm.sum(flag))) == 4
    assert bool(bm.all(pressure_value == 5.0))


def test_dirichlet_pressure_without_selector_means_all_boundary_faces():
    from fealpy.fvm import PDEBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    boundary = PDEBoundaryConditions(
        mesh,
        dirichlet_pressure=lambda points: bm.zeros(points.shape[0]),
    )

    assert boundary.has_dirichlet("pressure")
    assert boundary.dirichlet_threshold("pressure") is None


def test_engineering_boundary_conditions_validate_reference_and_neumann_semantics():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import BoundaryCondition, BoundaryPatch, EngineeringBoundaryConditions
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    patches = [BoundaryPatch("right", lambda p: bm.abs(p[:, 0] - 1.0) < 1.0e-12)]

    with pytest.raises(ValueError, match="reference.*pressure"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("velocity", "right", "reference", 0.0)],
        )
    with pytest.raises(ValueError, match="pressure/neumann"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("pressure", "right", "neumann", 0.0)],
        )
    with pytest.raises(ValueError, match="Neumann"):
        EngineeringBoundaryConditions(
            mesh,
            patches,
            [BoundaryCondition("velocity", "right", "neumann", None)],
        )


def test_neumann_velocity_boundary_enters_pde_boundary_protocol():
    bm.set_backend("numpy")
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    pde_bc = _left_dirichlet_right_neumann_bc(mesh).to_pde_boundary()
    boundary_faces = mesh.boundary_face_index()
    boundary_points = mesh.entity_barycenter("face")[boundary_faces]
    flag = pde_bc.neumann_threshold("velocity")(boundary_points)
    neumann_value = pde_bc.neumann_value("velocity")(boundary_points[flag])

    assert pde_bc.has_neumann("velocity")
    assert int(bm.to_numpy(bm.sum(flag))) == 4
    assert neumann_value.shape == (4, 2)
    assert bool(bm.all(neumann_value[:, 0] == 2.0))
    assert bool(bm.all(neumann_value[:, 1] == 3.0))


def test_mesh_bound_boundary_conditions_reject_foreign_mesh():
    bm.set_backend("numpy")
    import pytest
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    other_mesh = pde.init_mesh["uniform_quad"](nx=2, ny=2)
    engineering_bc = _left_right_engineering_bc(mesh)
    pde_bc = engineering_bc.to_pde_boundary()

    with pytest.raises(ValueError, match="mesh-bound"):
        pde_bc.boundary_face_velocity("velocity", mesh=other_mesh)


def test_face_constraints_require_explicit_boundary_faces():
    import pytest
    from fealpy.fvm.engineering_boundary_conditions import (
        apply_boundary_flux_constraint,
        apply_face_velocity_constraint,
    )

    face_velocity = bm.zeros((3, 2))
    boundary_velocity = bm.ones((2, 2))
    face_flux = bm.zeros(3)
    face_normal = bm.ones((3, 2))

    with pytest.raises(ValueError, match="boundary_faces"):
        apply_face_velocity_constraint(face_velocity, None, boundary_velocity)
    with pytest.raises(ValueError, match="boundary_faces"):
        apply_boundary_flux_constraint(
            face_flux,
            None,
            boundary_velocity,
            face_normal,
        )


def test_simple_model_uses_engineering_boundary_conditions_for_face_constraints():
    bm.set_backend("numpy")
    from fealpy.fvm import EngineeringBoundaryConditions, PDEBoundaryConditions

    model = _simple_model_with_left_right_bc()

    boundary_faces, boundary_value = model.boundary_conditions.boundary_face_velocity(
        "velocity",
        mesh=model.mesh,
    )
    face_centers = model.mesh.entity_barycenter("face")[boundary_faces]

    assert isinstance(model.engineering_bc, EngineeringBoundaryConditions)
    assert isinstance(model.boundary_conditions, PDEBoundaryConditions)
    assert boundary_faces.shape[0] == 4
    assert boundary_value.shape == (4, 2)
    assert bool(bm.all(face_centers[:, 0] < 1.0e-12))
    assert model.velocity_gradient.boundary_threshold is not None


def test_neumann_velocity_boundary_contributes_momentum_rhs():
    bm.set_backend("numpy")
    model = _simple_model_with_neumann_velocity()
    source = model.neumann_velocity_diffusion_source(model.diffusion_coef)
    source_cell = model.dofs_to_cell_vector(source)
    right_faces = model.engineering_bc.patch_face_index("right")
    owner = model.fvm_geometry.owner[right_faces]
    expected = bm.zeros_like(source_cell)
    face_contribution = model.fvm_geometry.mag_S_f[right_faces, None] * bm.array([2.0, 3.0])
    expected = bm.index_add(expected, owner, face_contribution, axis=0)

    assert float(bm.max(bm.abs(source_cell - expected))) < 1.0e-12


def test_cylinder_case_engineering_boundary_conditions_excludes_outlet_velocity():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import CylinderFlowCase

    pytest.importorskip("gmsh")
    case = CylinderFlowCase(
        mesh_size=0.12,
        cylinder_mesh_size=0.03,
        wake_mesh_size=0.06,
    )
    mesh = case.init_mesh["improved_tri"]()
    bc = case.engineering_boundary_conditions(mesh)
    pde_bc = bc.to_pde_boundary()

    boundary_faces, _ = pde_bc.boundary_face_velocity("velocity")
    face_centers = mesh.entity_barycenter("face")[boundary_faces]

    assert bool(bm.any(case.is_inlet_boundary(face_centers)))
    assert bool(bm.any(case.is_wall_boundary(face_centers)))
    assert bool(bm.any(case.is_cylinder_boundary(face_centers)))
    assert not bool(bm.any(case.is_outlet_boundary(face_centers)))


def test_simple_dirichlet_pressure_outlet_contributes_pressure_correction_flux():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)
    p_corr = bm.ones(model.NC)
    response_coef = bm.ones(model.mesh.number_of_faces())
    flux = model.pressure_correction_flux(p_corr, response_coef)
    outlet_faces = model.engineering_bc.patch_face_index("right")

    assert float(bm.max(bm.abs(flux[outlet_faces]))) > 1.0e-12


def test_simple_pressure_correction_boundary_inherits_diffusion_configuration():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(
        with_pressure=True,
        diffusion_method="bounded_over_relaxed",
        diffusion_nonorthogonal_eps=0.1,
    )

    assert model.pressure_correction_bc.diffusion_method == "bounded_over_relaxed"
    assert model.pressure_correction_bc.nonorthogonal_eps == 0.1
    assert model.pressure_correction_gradient.boundary_value is not None
    assert model.pressure_correction_gradient.boundary_threshold is not None


def test_simple_pressure_correction_flux_matches_face_velocity_flux_change():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=False)
    cell_center = model.fvm_geometry.cell_center
    p_corr = cell_center[:, 0] - 0.4 * cell_center[:, 1]
    p_corr_gradient = model.pressure_gradient.cell_gradient(p_corr)
    response_coef = bm.ones(model.mesh.number_of_faces())
    boundary_faces, boundary_velocity = (
        model.boundary_conditions.boundary_face_velocity(
            "velocity", mesh=model.mesh
        )
    )
    face_velocity = bm.zeros((model.mesh.number_of_faces(), model.GD))
    face_velocity = bm.set_at(face_velocity, boundary_faces, boundary_velocity)
    before_flux = model.compute_face_flux(face_velocity)

    corrected = model.correct_face_velocity_with_pressure_correction(
        face_velocity,
        p_corr,
        response_coef,
        boundary_faces,
        boundary_velocity,
        pressure_gradient=p_corr_gradient,
    )
    actual_delta = model.compute_face_flux(corrected) - before_flux
    expected_delta = model.pressure_correction_flux(
        p_corr,
        response_coef,
        pressure_gradient=p_corr_gradient,
    )
    internal_faces = bm.nonzero(model.fvm_geometry.is_internal)[0]

    assert float(
        bm.max(bm.abs(actual_delta[internal_faces] - expected_delta[internal_faces]))
    ) < 1.0e-12
    assert float(bm.max(bm.abs(corrected[boundary_faces] - boundary_velocity))) < 1.0e-12


def test_simple_closed_face_flux_has_compatible_pressure_rhs():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=False)
    face_flux = bm.zeros(model.mesh.number_of_faces())
    internal_faces = bm.nonzero(model.fvm_geometry.is_internal)[0]
    face_flux = bm.set_at(
        face_flux,
        internal_faces,
        bm.arange(internal_faces.shape[0], dtype=face_flux.dtype) + 1.0,
    )
    rhs = -model.divergence_from_flux(face_flux)

    assert abs(float(bm.to_numpy(model.pressure_rhs_compatibility(rhs)))) < 1.0e-12

    incompatible = rhs + 1.0
    projected = model.project_pressure_rhs_to_range(incompatible)
    assert abs(float(bm.to_numpy(model.pressure_rhs_compatibility(projected)))) < 1.0e-12


def test_simple_pressure_gradient_uses_engineering_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)

    assert model.pressure_gradient.boundary_value is not None
    assert model.pressure_gradient.boundary_threshold is not None


def test_simple_gradient_methods_are_configurable_for_pressure_and_velocity():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import NSFVMSimpleModel

    with pytest.raises(ValueError, match="Unknown cell_gradient variant"):
        NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "space_degree": 0,
                "log_level": "ERROR",
                "pbar_log": False,
                "pressure_gradient_method": "not_a_gradient_method",
            }
        )

    with pytest.raises(ValueError, match="Unknown cell_gradient variant"):
        NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "space_degree": 0,
                "log_level": "ERROR",
                "pbar_log": False,
                "velocity_gradient_method": "not_a_gradient_method",
            }
        )


def test_simple_rhie_chow_keeps_independent_default_gradient_method():
    bm.set_backend("numpy")
    from fealpy.fvm import NSFVMSimpleModel

    model = NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
            "pressure_gradient_method": "face_weighted_lsq",
        }
    )
    rhie_chow = model.rhie_chow

    grad_diff = rhie_chow.pressure_gradient_difference(bm.ones(model.NC))
    assert grad_diff.shape == (model.mesh.number_of_faces(), 2)


def test_simple_rhie_chow_gradient_method_is_explicitly_configurable():
    bm.set_backend("numpy")
    import pytest
    from fealpy.fvm import NSFVMSimpleModel

    model = NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
            "rhie_chow_pressure_gradient_method": "face_weighted_lsq",
        }
    )
    rhie_chow = model.rhie_chow

    grad_diff = rhie_chow.pressure_gradient_difference(bm.ones(model.NC))
    assert grad_diff.shape == (model.mesh.number_of_faces(), 2)


def test_simple_face_interpolation_method_controls_response_and_rhie_chow():
    bm.set_backend("numpy")
    import numpy as np
    from fealpy.fvm import NSFVMSimpleModel

    model = NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
            "face_interpolation_method": "linear",
        }
    )
    ap = bm.ones(2 * model.NC)

    response = np.asarray(
        model.pressure_response_face_coefficient(
            ap,
            model.controls.face_interpolation("pressure_response_interpolation"),
        )
    )
    expected = np.asarray(
        model.face_interpolate_cell_scalar(model.cm / ap[: model.NC], method="linear")
    )

    assert np.linalg.norm(response - expected) < 1.0e-12
    assert model.rhie_chow.velocity_interpolation == "linear"


def test_rhie_chow_uses_engineering_dirichlet_pressure_boundary():
    bm.set_backend("numpy")
    model = _simple_model_with_left_right_bc(with_pressure=True)
    rhie_chow = model.rhie_chow
    pressure = bm.ones(model.NC)
    outlet_faces = model.engineering_bc.patch_face_index("right")
    normal_component = bm.einsum(
        "ij,ij->i",
        rhie_chow.pressure_gradient_difference(pressure)[outlet_faces],
        model.fvm_geometry.S_f[outlet_faces],
    )

    assert float(bm.max(bm.abs(normal_component))) > 1.0e-12


def test_simple_natural_velocity_outlet_adds_owner_convection_diagonal():
    bm.set_backend("numpy")
    from fealpy.sparse import spdiags

    model = _simple_model_with_left_right_bc(with_pressure=True)
    A = spdiags(bm.zeros(2 * model.NC), 0, 2 * model.NC, 2 * model.NC)
    uf = bm.zeros((model.mesh.number_of_faces(), 2))
    uf = bm.set_at(uf, (slice(None), 0), 1.0)

    A = model.add_natural_velocity_convection_diagonal(
        A,
        model.convection_coef * uf,
        model.natural_velocity_threshold,
    )
    diag = bm.array(A.to_scipy().diagonal())
    outlet_faces = model.engineering_bc.patch_face_index("right")
    outlet_owners = model.fvm_geometry.owner[outlet_faces]

    assert float(bm.min(diag[outlet_owners])) > 0.0
    assert float(bm.min(diag[outlet_owners + model.NC])) > 0.0
