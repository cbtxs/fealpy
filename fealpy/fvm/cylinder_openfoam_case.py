"""OpenFOAM case writer for FEALPy cylinder-flow meshes.

The helpers in this module only generate a case directory and a Gmsh v2 prism
mesh from the current FEALPy two-dimensional triangular mesh.  They do not run
OpenFOAM commands.  This keeps FEALPy regression tests independent from a local
OpenFOAM installation while preserving a reproducible same-mesh comparison path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fealpy.backend import backend_manager as bm


def _as_numpy(value):
    return np.asarray(bm.to_numpy(value))


def oriented_triangles(node: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return counter-clockwise triangle connectivity."""
    cell = np.asarray(cell, dtype=np.int64).copy()
    p0 = node[cell[:, 0]]
    p1 = node[cell[:, 1]]
    p2 = node[cell[:, 2]]
    area2 = (
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    flip = area2 < 0.0
    cell[flip, 1], cell[flip, 2] = cell[flip, 2].copy(), cell[flip, 1].copy()
    return cell


def patch_name(case, point) -> str:
    """Return the OpenFOAM patch name for one FEALPy boundary face center."""
    p = bm.array(np.asarray(point, dtype=float).reshape(1, 2), dtype=bm.float64)
    flags = {
        "inlet": bool(_as_numpy(case.is_inlet_boundary(p))[0]),
        "outlet": bool(_as_numpy(case.is_outlet_boundary(p))[0]),
        "walls": bool(_as_numpy(case.is_wall_boundary(p))[0]),
        "cylinder": bool(_as_numpy(case.is_cylinder_boundary(p))[0]),
    }
    selected = [name for name, flag in flags.items() if flag]
    if len(selected) != 1:
        raise RuntimeError(f"boundary face at {point} has patch flags {flags}")
    return selected[0]


def write_prism_msh(case, mesh, path: str | Path, *, thickness: float = 0.01) -> dict:
    """Write a Gmsh v2 prism mesh by extruding the FEALPy triangle mesh."""
    path = Path(path)
    if thickness <= 0.0:
        raise ValueError("thickness must be positive.")

    raw_node2d = _as_numpy(mesh.entity("node"))
    raw_cell2d = _as_numpy(mesh.entity("cell")).astype(np.int64)
    raw_edge = _as_numpy(mesh.entity("edge")).astype(np.int64)
    bd_edge = _as_numpy(mesh.boundary_face_index()).astype(np.int64)
    face_centers = _as_numpy(mesh.entity_barycenter("face"))

    used_node = np.unique(raw_cell2d.reshape(-1))
    old_to_new = np.full(raw_node2d.shape[0], -1, dtype=np.int64)
    old_to_new[used_node] = np.arange(used_node.shape[0], dtype=np.int64)
    node2d = raw_node2d[used_node]
    cell2d = oriented_triangles(node2d, old_to_new[raw_cell2d])
    edge = old_to_new[raw_edge]
    if np.any(edge < 0):
        raise RuntimeError("A mesh edge references an unused node.")

    nnode = node2d.shape[0]
    nodes3d = np.vstack(
        [
            np.column_stack([node2d, np.zeros(nnode)]),
            np.column_stack([node2d, np.full(nnode, thickness)]),
        ]
    )

    physical_tags = {
        "frontAndBack": 1,
        "inlet": 2,
        "outlet": 3,
        "walls": 4,
        "cylinder": 5,
        "fluid": 6,
    }
    elements = []
    patch_counts = {name: 0 for name in physical_tags if name != "fluid"}

    def add_element(element_type: int, physical: str, nodes: list[int]) -> None:
        tag = physical_tags[physical]
        elements.append((element_type, tag, tag, nodes))
        if physical in patch_counts:
            patch_counts[physical] += 1

    for tri in cell2d:
        front = [int(tri[2]) + 1, int(tri[1]) + 1, int(tri[0]) + 1]
        back = [
            int(tri[0]) + nnode + 1,
            int(tri[1]) + nnode + 1,
            int(tri[2]) + nnode + 1,
        ]
        add_element(2, "frontAndBack", front)
        add_element(2, "frontAndBack", back)

    for edge_index in bd_edge:
        a, b = [int(v) for v in edge[edge_index]]
        name = patch_name(case, face_centers[edge_index])
        quad = [a + 1, b + 1, b + nnode + 1, a + nnode + 1]
        add_element(3, name, quad)

    for tri in cell2d:
        prism = [
            int(tri[0]) + 1,
            int(tri[1]) + 1,
            int(tri[2]) + 1,
            int(tri[0]) + nnode + 1,
            int(tri[1]) + nnode + 1,
            int(tri[2]) + nnode + 1,
        ]
        add_element(6, "fluid", prism)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        stream.write("$PhysicalNames\n6\n")
        stream.write('2 1 "frontAndBack"\n')
        stream.write('2 2 "inlet"\n')
        stream.write('2 3 "outlet"\n')
        stream.write('2 4 "walls"\n')
        stream.write('2 5 "cylinder"\n')
        stream.write('3 6 "fluid"\n')
        stream.write("$EndPhysicalNames\n")
        stream.write(f"$Nodes\n{nodes3d.shape[0]}\n")
        for index, point in enumerate(nodes3d, start=1):
            stream.write(
                f"{index} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n"
            )
        stream.write("$EndNodes\n")
        stream.write(f"$Elements\n{len(elements)}\n")
        for index, (etype, physical, elementary, nodes) in enumerate(
            elements,
            start=1,
        ):
            stream.write(
                f"{index} {etype} 2 {physical} {elementary} "
                + " ".join(str(node) for node in nodes)
                + "\n"
            )
        stream.write("$EndElements\n")

    return {
        "node2d": int(node2d.shape[0]),
        "cell2d": int(cell2d.shape[0]),
        "edge2d": int(edge.shape[0]),
        "boundary_edges": int(bd_edge.shape[0]),
        "nodes3d": int(nodes3d.shape[0]),
        "elements": int(len(elements)),
        "thickness": float(thickness),
        "patch_counts": patch_counts,
        "mesh_path": str(path),
    }


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_openfoam_case_files(
    case,
    case_dir: str | Path,
    *,
    thickness: float = 0.01,
    grad_scheme: str = "leastSquares",
    div_phi_u: str = "Gauss linear",
    sn_grad_scheme: str = "corrected",
    n_nonorthogonal_correctors: int = 2,
    pressure_relax: float = 0.3,
    velocity_equation_relax: float = 0.7,
    end_time: int = 3000,
    write_interval: int = 3000,
) -> None:
    """Write OpenFOAM dictionaries for the FEALPy cylinder benchmark."""
    case_dir = Path(case_dir)
    for rel in ("0", "constant", "system"):
        (case_dir / rel).mkdir(parents=True, exist_ok=True)

    diameter = 2.0 * float(case.radius)
    area_ref = diameter * float(thickness)
    xmin, xmax, ymin, ymax = case.box
    height = ymax - ymin

    _write_text(
        case_dir / "constant" / "physicalProperties",
        f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      physicalProperties;
}}

viscosityModel  constant;
nu              {float(case.nu):.17g};
""",
    )
    _write_text(
        case_dir / "constant" / "momentumTransport",
        """FoamFile
{
    format      ascii;
    class       dictionary;
    object      momentumTransport;
}

simulationType  laminar;
""",
    )
    _write_text(
        case_dir / "system" / "controlDict",
        f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {int(end_time)};
deltaT          1;
writeControl    timeStep;
writeInterval   {int(write_interval)};
purgeWrite      0;
writeFormat     ascii;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    forces
    {{
        type            forces;
        libs            ("libforces.so");
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          {float(case.rho):.17g};
        CofR            ({case.center[0]:.17g} {case.center[1]:.17g} 0);
        writeControl    timeStep;
        writeInterval   1;
    }}

    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (cylinder);
        rho             rhoInf;
        rhoInf          {float(case.rho):.17g};
        CofR            ({case.center[0]:.17g} {case.center[1]:.17g} 0);
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        magUInf         {float(case.mean_velocity):.17g};
        lRef            {diameter:.17g};
        Aref            {area_ref:.17g};
        writeControl    timeStep;
        writeInterval   1;
    }}
}}
""",
    )
    _write_text(
        case_dir / "system" / "fvSchemes",
        f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         {grad_scheme};
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_phi_u};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         {sn_grad_scheme};
}}
""",
    )
    _write_text(
        case_dir / "system" / "fvSolution",
        f"""FoamFile
{{
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-10;
        relTol          0.01;
        smoother        GaussSeidel;
    }}

    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors {int(n_nonorthogonal_correctors)};
    residualControl
    {{
        p               1e-7;
        U               1e-7;
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               {float(pressure_relax):.17g};
    }}
    equations
    {{
        U               {float(velocity_equation_relax):.17g};
    }}
}}
""",
    )
    _write_text(
        case_dir / "system" / "changeDictionaryDict",
        """FoamFile
{
    format      ascii;
    class       dictionary;
    object      changeDictionaryDict;
}

boundary
{
    frontAndBack
    {
        type            empty;
    }
}
""",
    )
    _write_text(
        case_dir / "0" / "U",
        f"""FoamFile
{{
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);

boundaryField
{{
    frontAndBack
    {{
        type            empty;
    }}
    inlet
    {{
        type            codedFixedValue;
        value           uniform (0 0 0);
        name            parabolicInlet;
        code
        #{{
            const scalar yMin = {float(ymin):.17g};
            const scalar H = {height:.17g};
            const scalar Umean = {float(case.mean_velocity):.17g};
            vectorField& field = *this;
            const vectorField& Cf = patch().Cf();
            forAll(field, i)
            {{
                const scalar y = Cf[i].y() - yMin;
                field[i] = vector(6.0*Umean*y*(H-y)/(H*H), 0, 0);
            }}
        #}};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    cylinder
    {{
        type            noSlip;
    }}
}}
""",
    )
    _write_text(
        case_dir / "0" / "p",
        """FoamFile
{
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    frontAndBack
    {
        type            empty;
    }
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
    cylinder
    {
        type            zeroGradient;
    }
}
""",
    )


def write_fealpy_cylinder_openfoam_case(
    case,
    mesh,
    case_dir: str | Path,
    *,
    mesh_filename: str = "fealpy_cylinder_prism.msh",
    thickness: float = 0.01,
    grad_scheme: str = "leastSquares",
    div_phi_u: str = "Gauss linear",
    sn_grad_scheme: str = "corrected",
) -> dict:
    """Write OpenFOAM files and the extruded FEALPy mesh for one case."""
    case_dir = Path(case_dir)
    write_openfoam_case_files(
        case,
        case_dir,
        thickness=thickness,
        grad_scheme=grad_scheme,
        div_phi_u=div_phi_u,
        sn_grad_scheme=sn_grad_scheme,
    )
    summary = write_prism_msh(
        case,
        mesh,
        case_dir / mesh_filename,
        thickness=thickness,
    )
    summary.update(
        {
            "case_dir": str(case_dir),
            "mesh_filename": mesh_filename,
            "grad_scheme": grad_scheme,
            "div_phi_u": div_phi_u,
            "sn_grad_scheme": sn_grad_scheme,
            "nu": float(case.nu),
            "rho": float(case.rho),
            "mean_velocity": float(case.mean_velocity),
            "re": float(case.re),
        }
    )
    (case_dir / "fealpy_mesh_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
