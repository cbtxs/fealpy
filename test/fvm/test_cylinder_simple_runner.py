from pathlib import Path

from fealpy.backend import backend_manager as bm


def test_cylinder_simple_runner_writes_standard_outputs(tmp_path: Path):
    bm.set_backend("numpy")
    from fealpy.fvm.ns_fvm_cylinder_simple import create_parser, run_simple_cylinder

    args = create_parser().parse_args(
        [
            "--mesh_size",
            "0.16",
            "--cylinder_mesh_size",
            "0.04",
            "--wake_mesh_size",
            "0.08",
            "--max_iter",
            "2",
            "--tol",
            "1e-3",
            "--linear_solver",
            "scipy",
            "--output_dir",
            str(tmp_path),
        ]
    )

    model, outputs = run_simple_cylinder(args)

    assert model.mesh.number_of_cells() > 0
    assert outputs["output_dir"] == tmp_path
    assert (tmp_path / "solution.vtu").exists()
    assert (tmp_path / "flow_overview.png").exists()
    assert (tmp_path / "summary.json").exists()
