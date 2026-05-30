from pathlib import Path

from fealpy.backend import backend_manager as bm


def test_openfoam_case_writer_exports_fealpy_prism_mesh(tmp_path: Path):
    import pytest

    pytest.importorskip("gmsh")
    bm.set_backend("numpy")

    from fealpy.fvm import CylinderFlowCase
    from fealpy.fvm.cylinder_openfoam_case import (
        write_fealpy_cylinder_openfoam_case,
    )

    case = CylinderFlowCase(
        mesh_size=0.18,
        cylinder_mesh_size=0.045,
        wake_mesh_size=0.09,
    )
    mesh = case.init_mesh["improved_tri"]()
    summary = write_fealpy_cylinder_openfoam_case(case, mesh, tmp_path)

    assert (tmp_path / "fealpy_cylinder_prism.msh").exists()
    assert (tmp_path / "0" / "U").exists()
    assert (tmp_path / "0" / "p").exists()
    assert (tmp_path / "constant" / "physicalProperties").exists()
    assert (tmp_path / "system" / "fvSchemes").exists()
    assert (tmp_path / "system" / "fvSolution").exists()
    change_dict = tmp_path / "system" / "changeDictionaryDict"
    assert change_dict.exists()
    change_text = change_dict.read_text()
    assert "frontAndBack" in change_text
    assert "type            empty;" in change_text
    assert (tmp_path / "fealpy_mesh_summary.json").exists()
    assert summary["cell2d"] == mesh.number_of_cells()
    assert summary["patch_counts"]["cylinder"] > 0
    assert "leastSquares" in (tmp_path / "system" / "fvSchemes").read_text()
