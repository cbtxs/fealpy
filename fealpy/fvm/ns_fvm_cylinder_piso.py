"""Run the cylinder-flow benchmark with ``NSFVMPISOModel``."""

from __future__ import annotations

import argparse
from pathlib import Path

from fealpy.backend import backend_manager as bm
from fealpy.fvm import CylinderFlowCase, FVMLinearSolverConfig, NSFVMPISOModel
from fealpy.fvm.cylinder_flow_postprocess import (
    cylinder_force_coefficients,
    pressure_drop,
    write_cylinder_outputs,
)
from fealpy.fvm.lid_driven_cavity_postprocess import (
    write_dict_csv,
    write_solution_vtk,
)
from fealpy.fvm.lid_driven_cavity_runner import re_label, scalarize_rows


def default_output_dir(
    re: float,
    *,
    mesh_size: float,
    cylinder_mesh_size: float,
    root: str | Path = "output/cylinder_flow",
) -> Path:
    mesh_label = (
        f"tri_h{float(mesh_size):g}_hc{float(cylinder_mesh_size):g}"
        .replace(".", "p")
        .replace("-", "m")
    )
    return Path(root) / "piso" / re_label(re) / mesh_label


def build_case(args) -> CylinderFlowCase:
    return CylinderFlowCase(
        re=args.re,
        rho=args.rho,
        mu=args.mu,
        mean_velocity=args.mean_velocity,
        mesh_size=args.mesh_size,
        cylinder_mesh_size=args.cylinder_mesh_size,
        wake_mesh_size=args.wake_mesh_size,
        cylinder_refine_radius=args.cylinder_refine_radius,
        wake_length=args.wake_length,
        wake_half_width=args.wake_half_width,
        outlet_velocity_policy=args.outlet_velocity_policy,
    )


def build_piso_options(case: CylinderFlowCase, args) -> dict:
    options = {
        "pde": case,
        "mesh_type": "improved_tri",
        "space_degree": int(args.space_degree),
        "duration": tuple(float(value) for value in args.duration),
        "nt": int(args.nt),
        "n_correctors": int(args.n_correctors),
        "pbar_log": args.pbar_log,
        "log_level": args.log_level,
        "linear_solver_config": FVMLinearSolverConfig(
            backend=args.backend,
            device=args.device,
            solver=args.linear_solver,
        ),
        "pressure_gradient_method": args.pressure_gradient_method,
        "velocity_gradient_method": args.velocity_gradient_method,
        "rhie_chow_pressure_gradient_method": args.rhie_chow_pressure_gradient_method,
        "face_interpolation_method": args.face_interpolation_method,
        "momentum_explicit_correction": args.momentum_explicit_correction,
        "momentum_nonorthogonal_max_iter": args.momentum_nonorthogonal_max_iter,
        "pressure_nonorthogonal_max_iter": args.pressure_nonorthogonal_max_iter,
        "use_transient_flux_correction": args.use_transient_flux_correction,
        "transient_flux_correction_limiter": args.transient_flux_correction_limiter,
        "snapshot_interval": args.snapshot_interval,
        "snapshot_start_step": args.snapshot_start_step,
    }
    if args.engineering_boundary_conditions:
        options["boundary_conditions"] = case.engineering_boundary_conditions
    if args.rho is not None:
        options["rho"] = args.rho
    if args.mu is not None:
        options["mu"] = args.mu
    return options


class CylinderPISOHistory:
    """Collect transient scalar diagnostics from a PISO cylinder run."""

    def __init__(
        self,
        case: CylinderFlowCase,
        *,
        output_dir: Path,
        fields: tuple[str, ...],
        viscous_method: str,
        vtk_interval: int = 0,
        vtk_start_step: int = 1,
    ):
        if vtk_interval < 0:
            raise ValueError("vtk_interval must be non-negative.")
        if vtk_start_step < 1:
            raise ValueError("vtk_start_step must be positive.")
        self.case = case
        self.output_dir = Path(output_dir)
        self.fields = fields
        self.viscous_method = viscous_method
        self.vtk_interval = int(vtk_interval)
        self.vtk_start_step = int(vtk_start_step)
        self.rows: list[dict] = []
        self.force_rows: list[dict] = []
        self.corrector_rows: list[dict] = []
        self._previous_velocity = None

    def __call__(
        self,
        *,
        step: int,
        time: float,
        model,
        cell_velocity,
        face_velocity=None,
        pressure=None,
        flux=None,
    ) -> None:
        speed = bm.sqrt(cell_velocity[:, 0] ** 2 + cell_velocity[:, 1] ** 2)
        row = {
            "step": int(step),
            "time": float(time),
            "mass": self._mass_residual(model, flux),
            "velocity_update": self._velocity_update(cell_velocity),
            "speed_max": self._scalar(bm.max(speed)),
            "speed_mean": self._scalar(bm.mean(speed)),
        }
        row.update(self._outlet_flux_diagnostics(model, self.case, flux))
        self.rows.append(row)
        force = cylinder_force_coefficients(
            model.mesh,
            self.case,
            uh=cell_velocity[:, 0],
            vh=cell_velocity[:, 1],
            pressure=pressure,
            velocity_gradient=getattr(model, "velocity_gradient", None),
            viscous_method=self.viscous_method,
        )
        probes = pressure_drop(model.mesh.entity_barycenter("cell"), pressure)
        self.force_rows.append(
            {
                "step": int(step),
                "time": float(time),
                **force,
                **{f"pressure_drop_{key}": value for key, value in probes.items()},
            }
        )
        if (
            self.vtk_interval > 0
            and step >= self.vtk_start_step
            and (step - self.vtk_start_step) % self.vtk_interval == 0
        ):
            write_solution_vtk(
                model.mesh,
                cell_velocity[:, 0],
                cell_velocity[:, 1],
                pressure,
                self.output_dir / "snapshots" / f"solution_{int(step):06d}.vtu",
                fields=self.fields,
                velocity_gradient=getattr(model, "velocity_gradient", None),
            )
        self._previous_velocity = bm.array(cell_velocity)

    def record_corrector(self, **row) -> None:
        self.corrector_rows.append(row)

    def _velocity_update(self, cell_velocity):
        if self._previous_velocity is None:
            return 0.0
        return self._scalar(bm.max(bm.abs(cell_velocity - self._previous_velocity)))

    @classmethod
    def _mass_residual(cls, model, flux):
        if flux is None:
            return None
        return cls._scalar(bm.max(bm.abs(model.divergence_from_flux(flux))))

    @classmethod
    def _outlet_flux_diagnostics(cls, model, case, flux):
        if flux is None:
            return {
                "outlet_flux_total": None,
                "outlet_flux_min": None,
                "outlet_backflow_flux": None,
                "outlet_backflow_face_count": None,
            }

        boundary_faces = model.mesh.boundary_face_index()
        face_centers = model.mesh.entity_barycenter("face")[boundary_faces]
        outlet_faces = boundary_faces[case.is_outlet_boundary(face_centers)]
        if outlet_faces.shape[0] == 0:
            return {
                "outlet_flux_total": 0.0,
                "outlet_flux_min": 0.0,
                "outlet_backflow_flux": 0.0,
                "outlet_backflow_face_count": 0,
            }

        outlet_flux = flux[outlet_faces]
        backflow = bm.maximum(-outlet_flux, 0.0)
        return {
            "outlet_flux_total": cls._scalar(bm.sum(outlet_flux)),
            "outlet_flux_min": cls._scalar(bm.min(outlet_flux)),
            "outlet_backflow_flux": cls._scalar(bm.sum(backflow)),
            "outlet_backflow_face_count": int(
                cls._scalar(bm.sum(outlet_flux < 0.0))
            ),
        }

    @staticmethod
    def _scalar(value):
        array = bm.to_numpy(value)
        return float(array.item() if array.shape == () else array)


def run_piso_cylinder(args):
    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    case = build_case(args)
    model = NSFVMPISOModel(build_piso_options(case, args))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            case.re,
            mesh_size=args.mesh_size,
            cylinder_mesh_size=args.cylinder_mesh_size,
        )
    )
    output_fields = tuple(args.output_fields)
    history = CylinderPISOHistory(
        case,
        output_dir=output_dir,
        fields=output_fields,
        viscous_method=args.force_viscous_method,
        vtk_interval=args.vtk_interval,
        vtk_start_step=args.vtk_start_step,
    )
    model.solve(
        snapshot_callback=history,
        corrector_callback=(
            history.record_corrector if args.piso_corrector_diagnostics else None
        ),
    )

    corrector_diagnostics_path = None
    if args.piso_corrector_diagnostics:
        corrector_diagnostics_path = output_dir / "piso_corrector_diagnostics.csv"
        write_dict_csv(
            corrector_diagnostics_path,
            scalarize_rows(history.corrector_rows),
        )

    outputs = write_cylinder_outputs(
        model,
        case,
        output_dir,
        residuals=history.rows,
        force_history=history.force_rows,
        strouhal_start_time=args.strouhal_start_time,
        strouhal_min_lift_amplitude=args.strouhal_min_lift_amplitude,
        run_summary={
            "solver": "NSFVMPISOModel",
            "re": case.re,
            "rho": case.rho,
            "mu": case.mu,
            "nu": case.nu,
            "mean_velocity": case.mean_velocity,
            "mesh_type": "improved_tri",
            "mesh_size": args.mesh_size,
            "cylinder_mesh_size": args.cylinder_mesh_size,
            "wake_mesh_size": args.wake_mesh_size,
            "duration": tuple(float(value) for value in args.duration),
            "nt": args.nt,
            "n_correctors": args.n_correctors,
            "snapshot_interval": args.snapshot_interval,
            "snapshot_start_step": args.snapshot_start_step,
            "vtk_interval": args.vtk_interval,
            "vtk_start_step": args.vtk_start_step,
            "strouhal_start_time": args.strouhal_start_time,
            "strouhal_min_lift_amplitude": args.strouhal_min_lift_amplitude,
            "momentum_nonorthogonal_max_iter": args.momentum_nonorthogonal_max_iter,
            "momentum_explicit_correction": args.momentum_explicit_correction,
            "pressure_nonorthogonal_max_iter": args.pressure_nonorthogonal_max_iter,
            "engineering_boundary_conditions": args.engineering_boundary_conditions,
            "pressure_gradient_method": args.pressure_gradient_method,
            "velocity_gradient_method": args.velocity_gradient_method,
            "rhie_chow_pressure_gradient_method": (
                args.rhie_chow_pressure_gradient_method
            ),
            "face_interpolation_method": args.face_interpolation_method,
            "transient_flux_correction_limiter": (
                args.transient_flux_correction_limiter
            ),
            "piso_corrector_diagnostics": args.piso_corrector_diagnostics,
            "force_viscous_method": args.force_viscous_method,
            "linear_solver": args.linear_solver,
        },
        viscous_method=args.force_viscous_method,
        fields=output_fields,
    )
    if corrector_diagnostics_path is not None:
        outputs["piso_corrector_diagnostics"] = corrector_diagnostics_path
    return model, outputs


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cylinder-flow benchmark with NSFVMPISOModel"
    )
    parser.add_argument("--re", default=20.0, type=float)
    parser.add_argument("--rho", default=1.0, type=float)
    parser.add_argument("--mu", default=None, type=float)
    parser.add_argument("--mean_velocity", default=0.2, type=float)
    parser.add_argument("--mesh_size", default=0.04, type=float)
    parser.add_argument("--cylinder_mesh_size", default=0.006, type=float)
    parser.add_argument("--wake_mesh_size", default=0.02, type=float)
    parser.add_argument("--cylinder_refine_radius", default=None, type=float)
    parser.add_argument("--wake_length", default=None, type=float)
    parser.add_argument("--wake_half_width", default=None, type=float)
    parser.add_argument(
        "--outlet_velocity_policy",
        default="profile",
        choices=("profile", "zero"),
    )
    parser.add_argument("--space_degree", default=0, type=int)
    parser.add_argument("--nt", default=400, type=int)
    parser.add_argument("--duration", nargs=2, default=(0.0, 20.0), type=float)
    parser.add_argument("--n_correctors", default=4, type=int)
    parser.add_argument("--snapshot_interval", default=1, type=int)
    parser.add_argument("--snapshot_start_step", default=1, type=int)
    parser.add_argument("--vtk_interval", default=0, type=int)
    parser.add_argument("--vtk_start_step", default=1, type=int)
    parser.add_argument("--strouhal_start_time", default=None, type=float)
    parser.add_argument("--strouhal_min_lift_amplitude", default=1.0e-3, type=float)
    parser.add_argument(
        "--momentum_nonorthogonal_max_iter",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--momentum_explicit_correction",
        default="current",
        choices=("current", "openfoam"),
    )
    parser.add_argument(
        "--pressure_nonorthogonal_max_iter",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--pressure_gradient_method",
        default="extended_lsq",
        choices=(
            "extended_lsq",
            "face_lsq",
            "weighted_lsq",
            "green_gauss",
        ),
    )
    parser.add_argument(
        "--velocity_gradient_method",
        default="extended_lsq",
        choices=(
            "extended_lsq",
            "face_lsq",
            "weighted_lsq",
            "green_gauss",
        ),
    )
    parser.add_argument(
        "--rhie_chow_pressure_gradient_method",
        default="extended_lsq",
        choices=(
            "extended_lsq",
            "face_lsq",
            "weighted_lsq",
            "green_gauss",
        ),
    )
    parser.add_argument(
        "--face_interpolation_method",
        default="average",
        choices=("average", "linear"),
    )
    parser.add_argument(
        "--force_viscous_method",
        default="wall_sn_grad",
        choices=("wall_sn_grad", "cell_gradient", "none"),
    )
    parser.add_argument(
        "--engineering_boundary_conditions",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--use_transient_flux_correction",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--transient_flux_correction_limiter",
        default="none",
        choices=("openfoam", "none"),
    )
    parser.add_argument(
        "--piso_corrector_diagnostics",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--backend", default="numpy", type=str)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--linear_solver", default="auto", type=str)
    parser.add_argument("--log_level", default="WARNING", type=str)
    parser.add_argument(
        "--pbar_log",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument(
        "--output_fields",
        nargs="+",
        default=["velocity", "u", "v", "pressure", "speed"],
    )
    return parser


def main() -> None:
    args = create_parser().parse_args()
    model, outputs = run_piso_cylinder(args)
    print(model)
    print(f"Output directory: {outputs['output_dir']}")
    print(f"Summary: {outputs['summary']}")


if __name__ == "__main__":
    main()
