from fealpy.backend import backend_manager as bm


def _boundary_convection_rhs(model, uf):
    bd_face = model.mesh.boundary_face_index()
    owner = model.mesh.edge_to_cell()[bd_face, 0]
    face_center = model.mesh.entity_barycenter("face")[bd_face]
    velocity_bc = model.pde.dirichlet_velocity(face_center)
    Sf = model.mesh.edge_normal()[bd_face]
    flux = bm.einsum("ij,ij->i", uf[bd_face], Sf)

    fx = bm.zeros(model.NC)
    fy = bm.zeros(model.NC)
    bm.add_at(fx, owner, -flux * velocity_bc[:, 0])
    bm.add_at(fy, owner, -flux * velocity_bc[:, 1])
    return bm.concatenate([fx, fy], axis=0)


def test_rc_momentum_rhs_includes_dirichlet_boundary_convection():
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_rc_model import NSFVMRCModel

    model = NSFVMRCModel({"pde": 6, "nx": 4, "ny": 4, "log_level": "ERROR"})
    face_center = model.mesh.entity_barycenter("face")
    uf = model.pde.velocity(face_center)
    zero_uf = bm.zeros_like(uf)

    _, f_zero = model.assembly_velocity(zero_uf)
    _, f_with_boundary_flux = model.assembly_velocity(uf)

    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(f_with_boundary_flux - f_zero - expected))) < 1.0e-12


def test_dirichlet_bc_convection_apply_handles_vector_dirichlet_data():
    bm.set_backend("numpy")
    from fealpy.fvm import DirichletBC
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_qrad"](nx=4, ny=4)
    face_center = mesh.entity_barycenter("face")
    uf = pde.velocity(face_center)

    class ModelView:
        pass

    model = ModelView()
    model.mesh = mesh
    model.pde = pde
    model.NC = mesh.number_of_cells()

    b = bm.zeros(2 * model.NC)
    actual = DirichletBC(mesh, pde.dirichlet_velocity).ConvectionApply(b, uf)
    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(actual - expected))) < 1.0e-12


def test_dirichlet_bc_exposes_only_general_convection_apply():
    bm.set_backend("numpy")
    from fealpy.fvm import DirichletBC

    assert hasattr(DirichletBC, "ConvectionApply")
    assert not hasattr(DirichletBC, "ConvectionApplyX")
    assert not hasattr(DirichletBC, "ConvectionApplyY")


def test_simple_momentum_rhs_includes_dirichlet_boundary_convection(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "compute_cross_diffusion",
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
                "linear_solver": CaptureSolver(),
            }
        )
        model.temporary_velocity(bm.zeros(model.NC), uf, bm.zeros(2 * model.NC))
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
    face_center = zero_model.mesh.entity_barycenter("face")
    uf = zero_model.pde.velocity(face_center)
    zero_uf = bm.zeros_like(uf)

    _, rhs_zero = first_momentum_rhs(zero_uf)
    model, rhs_with_boundary_flux = first_momentum_rhs(uf)

    expected = _boundary_convection_rhs(model, uf)

    assert float(bm.max(bm.abs(expected[model.NC:]))) > 1.0e-12
    assert float(bm.max(bm.abs(rhs_with_boundary_flux - rhs_zero - expected))) < 1.0e-12


def test_staggered_divergence_includes_signed_boundary_fluxes():
    bm.set_backend("numpy")
    from fealpy.fvm import DivergenceReconstruct, StaggeredMeshManager
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    staggered_mesh = StaggeredMeshManager(pde.domain(), nx=20, ny=20)
    uh = pde.velocity_u(staggered_mesh.umesh.entity_barycenter("cell"))
    vh = pde.velocity_v(staggered_mesh.vmesh.entity_barycenter("cell"))
    edge_velocity, _ = staggered_mesh.map_velocity_uvcell_to_pedge(
        uh,
        vh,
        bm.ones_like(uh),
        bm.ones_like(vh),
    )

    div = DivergenceReconstruct(staggered_mesh.pmesh).StagReconstruct(edge_velocity)
    div_l2 = bm.sqrt(
        bm.sum(staggered_mesh.pmesh.entity_measure("cell") * div**2)
    )

    assert float(div_l2) < 1.0e-3


def test_simple_mass_residual_is_normalized_and_scale_invariant():
    bm.set_backend("numpy")
    from fealpy.fvm import collocated_mass_residual
    from fealpy.model import PDEModelManager

    pde = PDEModelManager("navier_stokes").get_example(6)
    mesh = pde.init_mesh["uniform_qrad"](nx=20, ny=20)
    face_velocity = pde.velocity(mesh.entity_barycenter("face"))

    residual = collocated_mass_residual(mesh, face_velocity)
    scaled_residual = collocated_mass_residual(mesh, 7.0 * face_velocity)

    assert residual < 1.0e-3
    assert abs(residual - scaled_residual) < 1.0e-12


def test_collocated_simple_records_common_residuals(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.collocated_simple_solver as simple_solver
    import fealpy.fvm.simple_iteration_control as simple_iteration
    import fealpy.fvm.ns_fvm_simple_model as simple_model

    class FakeRhieChow:
        def __init__(self, mesh):
            self.mesh = mesh

        def Interpolation(self, u, ap, p):
            return bm.zeros((self.mesh.number_of_faces(), 2))

    monkeypatch.setattr(simple_solver, "RhieChowInterpolation", FakeRhieChow)
    monkeypatch.setattr(simple_iteration, "collocated_mass_residual", lambda mesh, uf: 0.0)
    monkeypatch.setattr(simple_iteration, "cell_l2_norm", lambda mesh, value: 0.0)
    monkeypatch.setattr(simple_iteration, "relative_l2_update", lambda mesh, update, p: 0.0)
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "temporary_velocity",
        lambda self, p, uf, u0: (bm.ones(2 * self.NC), bm.zeros(2 * self.NC)),
    )
    monkeypatch.setattr(
        simple_model.NSFVMSimpleModel,
        "pressure_correct",
        lambda self, ap, uf, *, response_coef=None: bm.zeros(self.NC),
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
            "pressure_update",
            "pressure_correction",
            "pressure_relax",
        )
    } == {
        "mass": 0.0,
        "pressure_update": 0.0,
        "pressure_correction": 0.0,
        "pressure_relax": 0.03,
    }


def test_staggered_simple_records_common_residuals(monkeypatch):
    bm.set_backend("numpy")
    import fealpy.fvm.ns_fvm_staggered_simple_model as staggered_model

    monkeypatch.setattr(staggered_model, "staggered_mass_residual", lambda mesh, edge_velocity: 0.0)
    monkeypatch.setattr(staggered_model, "relative_l2_update", lambda mesh, update, p: 0.0)
    monkeypatch.setattr(
        staggered_model.NSFVMStaggeredSimpleModel,
        "compute_temporary_velocity_u",
        lambda self, p_u, uf: (
            bm.zeros(self.umesh.number_of_cells()),
            bm.ones(self.umesh.number_of_cells()),
        ),
    )
    monkeypatch.setattr(
        staggered_model.NSFVMStaggeredSimpleModel,
        "compute_temporary_velocity_v",
        lambda self, p_v, uf: (
            bm.zeros(self.vmesh.number_of_cells()),
            bm.ones(self.vmesh.number_of_cells()),
        ),
    )
    monkeypatch.setattr(
        staggered_model.NSFVMStaggeredSimpleModel,
        "correct_pressure_compute",
        lambda self, f, a_p_edge: bm.zeros(self.pmesh.number_of_cells()),
    )

    model = staggered_model.NSFVMStaggeredSimpleModel(
        {
            "pde": 6,
            "nx": 2,
            "ny": 2,
            "backend": "numpy",
            "log_level": "ERROR",
            "pbar_log": False,
        }
    )

    model.solve(max_iter=5, tol=1.0e-5, relax=0.32)

    assert len(model.residuals) == 1
    assert {
        key: model.residuals[0][key]
        for key in ("mass", "pressure_update", "pressure_correction")
    } == {
        "mass": 0.0,
        "pressure_update": 0.0,
        "pressure_correction": 0.0,
    }
