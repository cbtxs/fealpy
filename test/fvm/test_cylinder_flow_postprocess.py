from pathlib import Path

from fealpy.backend import backend_manager as bm


def test_pressure_drop_uses_nearest_probe_points():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import pressure_drop

    points = bm.array([[0.15, 0.2], [0.25, 0.2], [1.0, 1.0]])
    pressure = bm.array([3.0, 1.25, -9.0])

    result = pressure_drop(
        points,
        pressure,
        upstream_point=(0.15, 0.2),
        downstream_point=(0.25, 0.2),
    )

    assert result["upstream_cell"] == 0
    assert result["downstream_cell"] == 1
    assert result["delta_p"] == 1.75


def test_zero_cylinder_fields_have_zero_force_coefficients():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import cylinder_force_coefficients

    class FakeCase:
        rho = 1.0
        mu = 1.0
        mean_velocity = 2.0
        radius = 0.5

        def is_cylinder_boundary(self, points):
            return bm.ones(points.shape[0], dtype=bm.bool)

    class FakeMesh:
        def boundary_face_index(self):
            return bm.array([0, 1], dtype=bm.int64)

        def entity_barycenter(self, entity):
            if entity == "face":
                return bm.array([[0.0, 0.0], [1.0, 0.0]])
            if entity == "cell":
                return bm.array([[0.0, 0.5], [1.0, 0.5]])
            raise KeyError(entity)

        def edge_to_cell(self):
            return bm.array([[0, 0], [1, 1]], dtype=bm.int64)

        def edge_normal(self):
            return bm.array([[0.0, -1.0], [0.0, -1.0]])

        def entity_measure(self, entity):
            if entity == "cell":
                return bm.array([1.0, 1.0])
            if entity == "face":
                return bm.array([1.0, 1.0])
            raise KeyError(entity)

    result = cylinder_force_coefficients(
        FakeMesh(),
        FakeCase(),
        uh=bm.zeros(2),
        vh=bm.zeros(2),
        pressure=bm.zeros(2),
        velocity_gradient=None,
    )

    assert result["force_x"] == 0.0
    assert result["force_y"] == 0.0
    assert result["drag_coefficient"] == 0.0
    assert result["lift_coefficient"] == 0.0


def test_cylinder_viscous_force_uses_wall_normal_sn_grad_by_default():
    bm.set_backend("numpy")
    from fealpy.fvm.cylinder_flow_postprocess import cylinder_force_coefficients

    class FakeCase:
        rho = 1.0
        mu = 0.5
        mean_velocity = 2.0
        radius = 0.5

        def is_cylinder_boundary(self, points):
            return bm.ones(points.shape[0], dtype=bm.bool)

    class FakeMesh:
        def boundary_face_index(self):
            return bm.array([0], dtype=bm.int64)

        def entity_barycenter(self, entity):
            if entity == "face":
                return bm.array([[1.0, 0.0]])
            if entity == "cell":
                return bm.array([[0.0, 0.0]])
            raise KeyError(entity)

        def edge_to_cell(self):
            return bm.array([[0, 0]], dtype=bm.int64)

        def edge_normal(self):
            return bm.array([[2.0, 0.0]])

        def entity_measure(self, entity):
            if entity == "cell":
                return bm.array([1.0])
            if entity == "face":
                return bm.array([2.0])
            raise KeyError(entity)

    result = cylinder_force_coefficients(
        FakeMesh(),
        FakeCase(),
        uh=bm.array([2.0]),
        vh=bm.array([0.0]),
        pressure=bm.zeros(1),
    )

    assert abs(result["viscous_force_x"] - 8.0 / 3.0) < 1.0e-12
    assert result["viscous_force_y"] == 0.0


def test_write_cylinder_outputs_creates_summary_and_vtu(tmp_path: Path):
    bm.set_backend("numpy")
    from fealpy.fvm import CylinderFlowCase, FVMLinearSolverConfig, NSFVMSimpleModel
    from fealpy.fvm.cylinder_flow_postprocess import write_cylinder_outputs

    case = CylinderFlowCase(
        mesh_size=0.16,
        cylinder_mesh_size=0.04,
        wake_mesh_size=0.08,
    )
    model = NSFVMSimpleModel(
        {
            "pde": case,
            "mesh_type": "improved_tri",
            "space_degree": 0,
            "boundary_conditions": case.engineering_boundary_conditions,
            "linear_solver_config": FVMLinearSolverConfig(solver="scipy"),
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    model.solve(max_iter=2, tol=1.0e-3)

    result = write_cylinder_outputs(
        model,
        case,
        tmp_path,
        residuals=model.residuals,
        run_summary={"solver": "NSFVMSimpleModel"},
    )

    assert (tmp_path / "solution.vtu").exists()
    assert (tmp_path / "flow_overview.png").exists()
    assert (tmp_path / "residual_history.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert "force" in result
    assert "pressure_drop" in result
