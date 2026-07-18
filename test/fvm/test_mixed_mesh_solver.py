import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.fvm import (
    CollocatedSimpleSolver,
    FVMGeometry,
    FVMLinearSolverConfig,
    PDEBoundaryConditions,
    SimpleSolverControls,
)
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.topology.builder import TopologyBuilder
from fealpy.mesh.view import Mesh


def _mixed_tri_quad_mesh():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    block = MeshBlock(positions=points)
    block.add_sector(
        EntitySector(
            "tri",
            np.array([[0, 1, 3], [0, 3, 2]], dtype=np.int32),
        ),
        root=True,
    )
    block.add_sector(
        EntitySector(
            "quad",
            np.array([[1, 4, 3, 5]], dtype=np.int32),
        ),
        root=True,
    )
    TopologyBuilder.construct(block)
    return Mesh(block).fealpy_api()


def test_collocated_simple_zero_solution_on_mixed_tri_quad_mesh():
    bm.set_backend("numpy")
    mesh = _mixed_tri_quad_mesh()

    def zero_velocity(points):
        return bm.zeros_like(points)

    solver = CollocatedSimpleSolver(
        mesh=mesh,
        diffusion_coef=1.0,
        convection_coef=0.0,
        source=zero_velocity,
        boundary_conditions=PDEBoundaryConditions(
            mesh,
            dirichlet_velocity=zero_velocity,
        ),
        controls=SimpleSolverControls(
            pressure_constraint="gauge",
            rhie_chow_velocity_scheme="second_order_reconstructed",
        ),
        linear_solver_config=FVMLinearSolverConfig(solver="scipy"),
        log_level="ERROR",
    )

    velocity, pressure = solver.solve(max_iter=1, tol=1.0e-12, relax=0.3)

    assert velocity.shape == (3, 2)
    assert pressure.shape == (3,)
    assert solver.face_velocity.shape == (8, 2)
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(pressure, 0.0, atol=1.0e-13)


def test_solution_vtu_rejects_mixed_cell_sectors_until_mesh_writer_supports_them(
    tmp_path,
):
    from fealpy.fvm.benchmark_postprocess import write_solution_vtk

    mesh = _mixed_tri_quad_mesh()
    geometry = FVMGeometry(mesh)

    with pytest.raises(RuntimeError, match="mixed cell-sector VTU output"):
        write_solution_vtk(
            mesh,
            bm.zeros((geometry.NC, geometry.GD)),
            bm.zeros(geometry.NC),
            tmp_path / "mixed.vtu",
            geometry=geometry,
        )
