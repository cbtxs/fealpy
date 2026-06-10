import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh import TriangleMesh


def _box_space(nx=2, ny=1):
    bm.set_backend("numpy")
    mesh = TriangleMesh.from_box([0.0, 1.0, 0.0, 1.0], nx=nx, ny=ny)
    return mesh, ScaledMonomialSpace2d(mesh, 0)


def test_scalar_diffusion_local_matrix_matches_orthogonal_flux_formula():
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.scalar_diffusion_integrator import scalar_diffusion_local_matrix

    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    coef = np.linspace(0.8, 1.4, mesh.number_of_faces())
    qf = mesh.quadrature_formula(2, "face")
    bcs, _ = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)
    _, mag_E_f, _ = geometry.over_relaxed_decomposition()

    local = scalar_diffusion_local_matrix(space, geometry, phi, coef=coef)

    expected_coef = np.asarray(mag_E_f) / np.asarray(geometry.mag_d_f) * coef
    expected = expected_coef[:, None, None] * np.array(
        [[1.0, -1.0], [-1.0, 1.0]]
    )
    np.testing.assert_allclose(np.asarray(local), expected, rtol=1.0e-13, atol=1.0e-13)


def test_scalar_diffusion_rejects_cell_wise_coefficient():
    from fealpy.fvm import FVMGeometry
    from fealpy.fvm.scalar_diffusion_integrator import scalar_diffusion_local_matrix

    mesh, space = _box_space(nx=2, ny=1)
    geometry = FVMGeometry(mesh)
    qf = mesh.quadrature_formula(2, "face")
    bcs, _ = qf.get_quadrature_points_and_weights()
    phi = space.basis(bcs)

    with pytest.raises(ValueError, match="face-wise"):
        scalar_diffusion_local_matrix(
            space,
            geometry,
            phi,
            coef=np.ones(mesh.number_of_cells()),
        )


def test_scalar_diffusion_integrator_delegates_local_matrix_construction(monkeypatch):
    import fealpy.fvm.scalar_diffusion_integrator as diffusion_module
    from fealpy.fvm import ScalarDiffusionIntegrator

    mesh, space = _box_space(nx=1, ny=1)
    calls = []

    def counted_local_matrix(space_arg, geometry, phi, *, coef=None):
        calls.append((space_arg, geometry, phi, coef))
        return bm.zeros((mesh.number_of_faces(), 2, 2), dtype=space.ftype)

    monkeypatch.setattr(
        diffusion_module,
        "scalar_diffusion_local_matrix",
        counted_local_matrix,
    )

    BilinearForm(space).add_integrator(ScalarDiffusionIntegrator()).assembly()

    assert len(calls) == 1
    assert calls[0][0] is space
