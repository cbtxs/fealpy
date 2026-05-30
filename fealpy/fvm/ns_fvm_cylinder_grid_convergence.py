"""Grid-convergence runner for the Re=20 cylinder-flow SIMPLE benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .cylinder_flow_case import CylinderFlowCase
from .cylinder_openfoam_case import write_fealpy_cylinder_openfoam_case
from .lid_driven_cavity_postprocess import write_dict_csv
from .lid_driven_cavity_runner import scalarize_rows
from .ns_fvm_cylinder_simple import create_parser as create_simple_parser
from .ns_fvm_cylinder_simple import run_simple_cylinder


DEFAULT_LEVELS = (
    "0.12:0.03:0.06",
    "0.08:0.02:0.04",
    "0.06:0.015:0.03",
)

DEFAULT_METRICS = (
    "force_drag_coefficient",
    "force_lift_coefficient",
    "pressure_drop_delta_p",
    "force_pressure_force_x",
    "force_viscous_force_x",
)


SOLVER_CASES = {
    "fealpy_default": {},
}


def _label(name: str, value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return f"{name}{text}"


def parse_level(level: str) -> tuple[float, float, float]:
    """Parse one ``mesh:cylinder:wake`` grid level specification."""
    parts = level.split(":")
    if len(parts) != 3:
        raise ValueError("grid level must be 'mesh_size:cylinder_size:wake_size'.")
    mesh_size, cylinder_mesh_size, wake_mesh_size = (
        float(part) for part in parts
    )
    if mesh_size <= 0.0 or cylinder_mesh_size <= 0.0 or wake_mesh_size <= 0.0:
        raise ValueError("all grid level sizes must be positive.")
    return mesh_size, cylinder_mesh_size, wake_mesh_size


def parse_levels(levels: list[str] | tuple[str, ...] | None) -> list[dict]:
    """Return normalized grid-level dictionaries from CLI level strings."""
    values = DEFAULT_LEVELS if levels is None else levels
    result = []
    for index, level in enumerate(values, start=1):
        mesh_size, cylinder_mesh_size, wake_mesh_size = parse_level(level)
        result.append(
            {
                "level": index,
                "mesh_size": mesh_size,
                "cylinder_mesh_size": cylinder_mesh_size,
                "wake_mesh_size": wake_mesh_size,
                "level_label": "_".join(
                    [
                        f"level{index:02d}",
                        _label("h", mesh_size),
                        _label("hc", cylinder_mesh_size),
                        _label("hw", wake_mesh_size),
                    ]
                ),
            }
        )
    return result


def _case_output_dir(root: Path, solver_case: str, level: dict) -> Path:
    return root / solver_case / level["level_label"]


def _with_solver_case_options(args, solver_case: str, level: dict, output_root: Path):
    case_args = argparse.Namespace(**vars(args))
    case_args.mesh_size = level["mesh_size"]
    case_args.cylinder_mesh_size = level["cylinder_mesh_size"]
    case_args.wake_mesh_size = level["wake_mesh_size"]
    case_args.output_dir = str(_case_output_dir(output_root, solver_case, level))
    for key, value in SOLVER_CASES[solver_case].items():
        setattr(case_args, key, value)
    return case_args


def _build_case_for_level(args, level: dict):
    """Build a cylinder case for a grid level without invoking the solver."""
    return CylinderFlowCase(
        re=args.re,
        rho=args.rho,
        mu=args.mu,
        mean_velocity=args.mean_velocity,
        mesh_size=level["mesh_size"],
        cylinder_mesh_size=level["cylinder_mesh_size"],
        wake_mesh_size=level["wake_mesh_size"],
        cylinder_refine_radius=args.cylinder_refine_radius,
        wake_length=args.wake_length,
        wake_half_width=args.wake_half_width,
        outlet_velocity_policy=args.outlet_velocity_policy,
    )


def _openfoam_case_dir(output_root: Path, level: dict) -> Path:
    return output_root / "openfoam_cases" / level["level_label"]


def write_openfoam_cases(args, levels: list[dict], output_root: Path) -> dict[int, dict]:
    """Write same-grid OpenFOAM case directories for all grid levels."""
    summaries = {}
    for level in levels:
        case = _build_case_for_level(args, level)
        mesh = case.init_mesh["improved_tri"]()
        case_dir = _openfoam_case_dir(output_root, level)
        summaries[level["level"]] = write_fealpy_cylinder_openfoam_case(
            case,
            mesh,
            case_dir,
            thickness=args.openfoam_thickness,
            grad_scheme=args.openfoam_grad_scheme,
            div_phi_u=args.openfoam_div_phi_u,
            sn_grad_scheme=args.openfoam_sn_grad_scheme,
        )
    return summaries


def _normalize_rows(rows: list[dict]) -> list[dict]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return [{key: row.get(key, "") for key in fieldnames} for row in rows]


def _as_float_or_none(value):
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def convergence_rows(
    rows: list[dict],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    solver_key: str = "solver_case",
) -> list[dict]:
    """Return three-level successive-difference observed orders.

    Without an analytical solution, one useful signal is the trend of a scalar
    benchmark quantity ``Q`` across three grids:

    ``p = log(|Q1-Q2| / |Q2-Q3|) / log(h1/h2)``.

    This is reported only when the two refinement ratios are equal.  Otherwise
    the row keeps the successive differences and marks ``ratio_consistent`` as
    false, because the simple three-level formula is not well-defined.
    """
    solver_cases = []
    for row in rows:
        solver_case = row.get(solver_key)
        if solver_case and solver_case not in solver_cases:
            solver_cases.append(solver_case)

    result = []
    for solver_case in solver_cases:
        case_rows = [
            row
            for row in rows
            if row.get(solver_key) == solver_case and row.get("status", "ok") == "ok"
        ]
        case_rows.sort(key=lambda row: float(row["mesh_size"]), reverse=True)
        for first, second, third in zip(case_rows, case_rows[1:], case_rows[2:]):
            h1 = float(first["mesh_size"])
            h2 = float(second["mesh_size"])
            h3 = float(third["mesh_size"])
            ratio12 = h1 / h2
            ratio23 = h2 / h3
            ratio_consistent = math.isclose(
                ratio12,
                ratio23,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for metric in metrics:
                q1 = _as_float_or_none(first.get(metric))
                q2 = _as_float_or_none(second.get(metric))
                q3 = _as_float_or_none(third.get(metric))
                if q1 is None or q2 is None or q3 is None:
                    continue
                diff12 = abs(q1 - q2)
                diff23 = abs(q2 - q3)
                order = ""
                if ratio_consistent and diff12 > 0.0 and diff23 > 0.0:
                    order = math.log(diff12 / diff23) / math.log(ratio12)
                result.append(
                    {
                        "solver_case": solver_case,
                        "metric": metric,
                        "level_coarse": first["level"],
                        "level_mid": second["level"],
                        "level_fine": third["level"],
                        "h_coarse": h1,
                        "h_mid": h2,
                        "h_fine": h3,
                        "ratio_coarse_to_mid": ratio12,
                        "ratio_mid_to_fine": ratio23,
                        "ratio_consistent": ratio_consistent,
                        "successive_difference_coarse_mid": diff12,
                        "successive_difference_mid_fine": diff23,
                        "successive_order": order,
                    }
                )
    return result


def run_grid_convergence(args) -> dict:
    """Run selected FEALPy cylinder grid levels and write comparison tables."""
    output_root = Path(args.output_dir or "output/cylinder_flow_grid_convergence")
    output_root.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.levels)
    openfoam_summaries = (
        write_openfoam_cases(args, levels, output_root)
        if args.write_openfoam_cases
        else {}
    )

    rows = []
    for solver_case in args.solver_cases:
        if solver_case not in SOLVER_CASES:
            raise ValueError(f"Unknown solver case: {solver_case}")
        for level in levels:
            row = {
                "solver_case": solver_case,
                "level": level["level"],
                "level_label": level["level_label"],
                "mesh_size": level["mesh_size"],
                "cylinder_mesh_size": level["cylinder_mesh_size"],
                "wake_mesh_size": level["wake_mesh_size"],
                "output_dir": str(_case_output_dir(output_root, solver_case, level)),
            }
            openfoam_summary = openfoam_summaries.get(level["level"])
            if openfoam_summary is not None:
                row.update(
                        {
                            "openfoam_case_dir": openfoam_summary.get("case_dir", ""),
                            "openfoam_mesh_path": openfoam_summary.get("mesh_path", ""),
                            "openfoam_cell2d": openfoam_summary.get("cell2d", ""),
                            "openfoam_grad_scheme": openfoam_summary.get(
                                "grad_scheme", ""
                            ),
                            "openfoam_div_phi_u": openfoam_summary.get(
                                "div_phi_u", ""
                            ),
                        }
                    )
            try:
                case_args = _with_solver_case_options(
                    args,
                    solver_case,
                    level,
                    output_root,
                )
                _, outputs = run_simple_cylinder(case_args)
                row["status"] = "ok"
                row.update(outputs.get("summary", {}))
            except Exception as exc:
                if args.stop_on_failure:
                    raise
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)

    summary_rows = scalarize_rows(_normalize_rows(rows))
    convergence = scalarize_rows(
        _normalize_rows(
            convergence_rows(rows, metrics=tuple(args.metrics))
        )
    )

    summary_csv = output_root / "grid_summary.csv"
    summary_json = output_root / "grid_summary.json"
    convergence_csv = output_root / "grid_convergence.csv"
    convergence_json = output_root / "grid_convergence.json"

    write_dict_csv(summary_csv, summary_rows)
    summary_json.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    write_dict_csv(convergence_csv, convergence)
    convergence_json.write_text(
        json.dumps(convergence, indent=2, sort_keys=True) + "\n"
    )

    return {
        "output_dir": output_root,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "convergence_csv": convergence_csv,
        "convergence_json": convergence_json,
        "rows": summary_rows,
        "convergence_rows": convergence,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = create_simple_parser()
    parser.description = "Grid convergence for the cylinder-flow SIMPLE benchmark"
    parser.set_defaults(output_dir="output/cylinder_flow_grid_convergence")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Grid levels as mesh_size:cylinder_mesh_size:wake_mesh_size.",
    )
    parser.add_argument(
        "--solver_cases",
        nargs="+",
        default=["fealpy_default"],
        choices=tuple(SOLVER_CASES),
        help="FEALPy diagnostic solver cases to run on each grid level.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Scalar summary metrics used for successive-difference orders.",
    )
    parser.add_argument(
        "--write_openfoam_cases",
        default=False,
        action="store_true",
        help="Write same-grid OpenFOAM case directories; does not run OpenFOAM.",
    )
    parser.add_argument("--openfoam_thickness", default=0.01, type=float)
    parser.add_argument("--openfoam_grad_scheme", default="leastSquares", type=str)
    parser.add_argument("--openfoam_div_phi_u", default="Gauss linear", type=str)
    parser.add_argument("--openfoam_sn_grad_scheme", default="corrected", type=str)
    parser.add_argument("--stop_on_failure", default=False, action="store_true")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    result = run_grid_convergence(args)
    print(f"Grid convergence output directory: {result['output_dir']}")
    print(f"Summary: {result['summary_csv']}")
    print(f"Convergence: {result['convergence_csv']}")


if __name__ == "__main__":
    main()
