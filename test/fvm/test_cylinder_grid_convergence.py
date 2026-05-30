from pathlib import Path


def test_cylinder_grid_convergence_runs_fealpy_levels(tmp_path, monkeypatch):
    from fealpy.fvm import ns_fvm_cylinder_grid_convergence as conv

    args = conv.create_parser().parse_args(
        [
            "--levels",
            "0.12:0.03:0.06",
            "0.08:0.02:0.04",
            "0.06:0.015:0.03",
            "--output_dir",
            str(tmp_path),
            "--max_iter",
            "3",
            "--linear_solver",
            "scipy",
        ]
    )
    calls = []

    def fake_run_simple_cylinder(case_args):
        calls.append(
            (
                case_args.mesh_size,
                case_args.cylinder_mesh_size,
                case_args.wake_mesh_size,
                Path(case_args.output_dir),
            )
        )
        value = 4.0 + case_args.mesh_size
        return object(), {
            "output_dir": Path(case_args.output_dir),
            "summary": {
                "cells": int(round(100 / case_args.mesh_size)),
                "iterations": case_args.max_iter,
                "force_drag_coefficient": value,
                "pressure_drop_delta_p": 0.1 + case_args.mesh_size,
            },
        }

    monkeypatch.setattr(conv, "run_simple_cylinder", fake_run_simple_cylinder)
    result = conv.run_grid_convergence(args)

    assert len(calls) == 3
    assert calls[0][0:3] == (0.12, 0.03, 0.06)
    assert result["summary_csv"] == tmp_path / "grid_summary.csv"
    assert result["convergence_csv"] == tmp_path / "grid_convergence.csv"
    assert result["summary_csv"].exists()
    assert result["summary_json"].exists()
    assert result["convergence_csv"].exists()
    assert result["convergence_json"].exists()

    summary_text = result["summary_csv"].read_text()
    assert "fealpy_default" in summary_text
    assert "force_drag_coefficient" in summary_text

    convergence_text = result["convergence_csv"].read_text()
    assert "successive_order" in convergence_text
    assert "force_drag_coefficient" in convergence_text


def test_successive_observed_orders_use_three_grid_levels():
    from fealpy.fvm.ns_fvm_cylinder_grid_convergence import (
        convergence_rows,
    )

    rows = [
        {
            "solver_case": "fealpy_default",
            "level": 1,
            "mesh_size": 0.12,
            "force_drag_coefficient": 5.08,
        },
        {
            "solver_case": "fealpy_default",
            "level": 2,
            "mesh_size": 0.06,
            "force_drag_coefficient": 5.02,
        },
        {
            "solver_case": "fealpy_default",
            "level": 3,
            "mesh_size": 0.03,
            "force_drag_coefficient": 5.005,
        },
    ]

    result = convergence_rows(rows, metrics=("force_drag_coefficient",))

    assert len(result) == 1
    assert result[0]["solver_case"] == "fealpy_default"
    assert result[0]["metric"] == "force_drag_coefficient"
    assert abs(result[0]["successive_order"] - 2.0) < 1.0e-12


def test_grid_convergence_can_write_openfoam_cases(tmp_path, monkeypatch):
    from fealpy.fvm import ns_fvm_cylinder_grid_convergence as conv

    args = conv.create_parser().parse_args(
        [
            "--levels",
            "0.12:0.03:0.06",
            "0.08:0.02:0.04",
            "--output_dir",
            str(tmp_path),
            "--write_openfoam_cases",
            "--max_iter",
            "0",
        ]
    )
    calls = []

    class FakeMesh:
        def number_of_cells(self):
            return 42

    class FakeCase:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @property
        def init_mesh(self):
            return {"improved_tri": lambda: FakeMesh()}

    def fake_writer(case, mesh, case_dir, **kwargs):
        calls.append((case.kwargs["mesh_size"], mesh.number_of_cells(), Path(case_dir)))
        Path(case_dir).mkdir(parents=True, exist_ok=True)
        return {"case_dir": str(case_dir), "cell2d": mesh.number_of_cells()}

    monkeypatch.setattr(conv, "CylinderFlowCase", FakeCase)
    monkeypatch.setattr(conv, "write_fealpy_cylinder_openfoam_case", fake_writer)
    monkeypatch.setattr(
        conv,
        "run_simple_cylinder",
        lambda case_args: (
            object(),
            {"summary": {"cells": 1, "force_drag_coefficient": 1.0}},
        ),
    )

    result = conv.run_grid_convergence(args)

    assert len(calls) == 2
    assert calls[0][0] == 0.12
    assert calls[0][1] == 42
    assert calls[0][2] == tmp_path / "openfoam_cases" / "level01_h0p12_hc0p03_hw0p06"
    assert "openfoam_case_dir" in result["rows"][0]
