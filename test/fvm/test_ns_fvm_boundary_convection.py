from fealpy.backend import backend_manager as bm


def _boundary_convection_rhs(model, uf):
    from fealpy.fvm import FVMGeometry

    geometry = getattr(model, "fvm_geometry", FVMGeometry(model.mesh))
    boundary_faces = bm.nonzero(geometry.is_boundary)[0]
    owner = geometry.owner[boundary_faces]
    face_center = geometry.face_center[boundary_faces]
    velocity_bc = model.pde.dirichlet_velocity(face_center)
    Sf = geometry.S_f[boundary_faces]
    flux = bm.einsum("ij,ij->i", uf[boundary_faces], Sf)

    fx = bm.zeros(model.NC)
    fy = bm.zeros(model.NC)
    bm.add_at(fx, owner, -flux * velocity_bc[:, 0])
    bm.add_at(fy, owner, -flux * velocity_bc[:, 1])
    return bm.concatenate([fx, fy], axis=0)


def test_dirichlet_bc_convection_apply_handles_vector_dirichlet_data():
    bm.set_backend("numpy")
    from fealpy.fvm import DirichletBC
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=4, ny=4)
    face_center = mesh.entity_barycenter("face")
    uf = pde.velocity(face_center)

    class ModelView:
        pass

    model = ModelView()
    model.mesh = mesh
    model.pde = pde
    model.NC = mesh.number_of_cells()
    from fealpy.fvm import FVMGeometry
    model.fvm_geometry = FVMGeometry(mesh)

    b = bm.zeros(2 * model.NC)
    actual = DirichletBC(mesh, pde.dirichlet_velocity).apply_convection(b, uf)
    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(actual - expected))) < 1.0e-12


def test_simple_momentum_rhs_includes_dirichlet_boundary_convection(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "momentum_nonorthogonal_rhs",
        lambda self, uh: bm.zeros(2 * self.NC),
    )

    def first_momentum_rhs(uf):
        captured = []

        class CaptureSolver:
            def solve(self, A, b, *args, **kwargs):
                captured.append(b.copy())
                return bm.zeros(A.shape[0])

        model = simple_model.NSFVMSimpleModel(
            {
                "pde": 6,
                "nx": 4,
                "ny": 4,
                "space_degree": 0,
                "log_level": "ERROR",
                    "pbar_log": False,
                    "momentum_solve_strategy": "vector",
                    "momentum_nonorthogonal_max_iter": 0,
                    "linear_solver": CaptureSolver(),
            }
        )
        model.temporary_velocity(
            bm.zeros(model.NC),
            uf,
            bm.zeros((model.NC, model.GD)),
        )
        return model, captured[0]

    zero_model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 4,
            "ny": 4,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    face_center = zero_model.fvm_geometry.face_center
    uf = zero_model.pde.velocity(face_center)
    zero_uf = bm.zeros_like(uf)

    _, rhs_zero = first_momentum_rhs(zero_uf)
    model, rhs_with_boundary_flux = first_momentum_rhs(uf)

    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(rhs_with_boundary_flux - rhs_zero - expected))) < 1.0e-12


def test_simple_mass_residual_is_normalized_and_scale_invariant():
    bm.set_backend("numpy")
    from fealpy.fvm import collocated_mass_residual
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_quad"](nx=20, ny=20)
    face_velocity = pde.velocity(mesh.entity_barycenter("face"))

    residual = collocated_mass_residual(mesh, face_velocity)
    scaled_residual = collocated_mass_residual(mesh, 7.0 * face_velocity)

    assert residual < 1.0e-3
    assert abs(residual - scaled_residual) < 1.0e-12


def test_collocated_simple_records_common_residuals(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.simple_residual as simple_residual
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    class FakeRhieChow:
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh

        def cell_velocity_to_face(self, u, *, face_response_coefficient):
            return (
                bm.zeros((self.mesh.number_of_faces(), 2)),
                face_response_coefficient[:, None],
            )

        def pressure_gradient_difference(self, p, pressure_gradient=None):
            return bm.zeros((self.mesh.number_of_faces(), 2))

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_residual,
        "collocated_mass_metrics",
        lambda mesh, uf, **kwargs: {
            "relative_l1": 0.0,
            "relative_l2": 0.0,
            "divergence_l2": 0.0,
            "absolute_linf": 0.0,
        },
    )
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 0.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 0.0)
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "temporary_velocity",
        lambda self, p, uf, u0, **kwargs: simple_solver.MomentumPredictorResult(
            correction_denominator=bm.ones(2 * self.NC),
            spatial_diagonal=bm.ones(2 * self.NC),
            velocity=bm.zeros((self.NC, self.GD)),
        ),
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "pressure_correct",
        lambda self, ap, uf, *, response_coef=None: bm.zeros(self.NC),
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "steady_momentum_balance",
        lambda self, p, u, uf: {
            "lhs": bm.zeros(self.GD * self.NC),
            "rhs": bm.zeros(self.GD * self.NC),
            "residual": bm.zeros(self.GD * self.NC),
        },
    )

    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)

    model.solve(max_iter=5, tol=1.0e-5)

    assert len(model.residuals) == 1
    assert {
        key: model.residuals[0][key]
        for key in (
            "mass",
            "pressure_correction",
        )
    } == {
        "mass": 0.0,
        "pressure_correction": 0.0,
    }


def test_collocated_simple_updates_cell_velocity_after_pressure_correction(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.simple_residual as simple_residual
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    class FakeRhieChow:
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh

        def cell_velocity_to_face(self, u, *, face_response_coefficient):
            return (
                bm.zeros((self.mesh.number_of_faces(), 2)),
                face_response_coefficient[:, None],
            )

        def pressure_gradient_difference(self, p, pressure_gradient=None):
            return bm.zeros((self.mesh.number_of_faces(), 2))

    calls = []

    def fake_temporary_velocity(self, p, uf, u0, **kwargs):
        calls.append(u0.copy())
        response = bm.ones(2 * self.NC)
        return simple_solver.MomentumPredictorResult(
            correction_denominator=response,
            spatial_diagonal=response,
            velocity=bm.zeros((self.NC, self.GD)),
        )

    def fake_velocity_pressure_correction(
        self,
        cell_velocity,
        pressure_field,
        a_p,
        **kwargs,
    ):
        assert bm.max(bm.abs(pressure_field - 0.5)) < 1.0e-14
        return cell_velocity + 3.0

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_residual,
        "collocated_mass_metrics",
        lambda mesh, uf, **kwargs: {
            "relative_l1": 0.0,
            "relative_l2": 0.0,
            "divergence_l2": 0.0,
            "absolute_linf": 0.0,
        },
    )
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 1.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 1.0)
    monkeypatch.setattr(simple_model.NSFVMSimpleModel, "temporary_velocity", fake_temporary_velocity)
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "pressure_correct",
        lambda self, ap, uf, *, response_coef=None: bm.ones(self.NC),
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "correct_face_velocity_with_pressure_correction",
        lambda self, uf, p_corr, response_coef, boundary_faces, boundary_velocity, **kwargs: uf,
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "velocity_pressure_correction",
        fake_velocity_pressure_correction,
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "steady_momentum_balance",
        lambda self, p, u, uf: {
            "lhs": bm.ones(self.GD * self.NC),
            "rhs": bm.zeros(self.GD * self.NC),
            "residual": bm.ones(self.GD * self.NC),
        },
    )

    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)

    model.solve(
        max_iter=1,
        tol_mass=1.0e-99,
        tol_momentum=1.0e-12,
        relax=0.5,
    )

    assert len(calls) == 2
    assert bm.max(bm.abs(calls[1] - 3.0)) < 1.0e-14


def test_collocated_simple_avoids_duplicate_face_pressure_correction(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.simple_residual as simple_residual
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    class FakeRhieChow:
        def __init__(self, mesh, **kwargs):
            self.mesh = mesh

        def cell_velocity_to_face(self, u, *, face_response_coefficient):
            return (
                bm.zeros((self.mesh.number_of_faces(), 2)),
                face_response_coefficient[:, None],
            )

        def pressure_gradient_difference(self, p, pressure_gradient=None):
            return bm.zeros((self.mesh.number_of_faces(), 2))

    def fake_temporary_velocity(self, p, uf, u0, **kwargs):
        response = bm.ones(2 * self.NC)
        return simple_solver.MomentumPredictorResult(
            correction_denominator=response,
            spatial_diagonal=response,
            velocity=bm.zeros((self.NC, self.GD)),
        )

    def fake_correct_face_velocity(
        self,
        uf,
        p_corr,
        response_coef,
        boundary_faces,
        boundary_velocity,
        **kwargs,
    ):
        raise AssertionError(
            "the synchronized SIMPLE route must reconstruct the final face velocity"
        )

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(
        simple_residual,
        "collocated_mass_metrics",
        lambda mesh, uf, **kwargs: {
            "relative_l1": 0.0,
            "relative_l2": 0.0,
            "divergence_l2": 0.0,
            "absolute_linf": 0.0,
        },
    )
    monkeypatch.setattr(simple_residual, "cell_l2_norm", lambda mesh, value, **kwargs: 1.0)
    monkeypatch.setattr(simple_residual, "relative_l2_update", lambda mesh, update, p, **kwargs: 1.0)
    monkeypatch.setattr(simple_model.NSFVMSimpleModel, "temporary_velocity", fake_temporary_velocity)
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "pressure_correct",
        lambda self, ap, uf, *, response_coef=None: bm.ones(self.NC),
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "correct_face_velocity_with_pressure_correction",
        fake_correct_face_velocity,
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "steady_momentum_balance",
        lambda self, p, u, uf: {
            "lhs": bm.zeros(self.GD * self.NC),
            "rhs": bm.zeros(self.GD * self.NC),
            "residual": bm.zeros(self.GD * self.NC),
        },
    )

    model = simple_model.NSFVMSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "space_degree": 0,
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )
    model.pde.dirichlet_velocity = lambda points: bm.zeros_like(points)

    model.solve(max_iter=1, tol_mass=1.0e-99, tol_momentum=1.0e-12, relax=0.5)
