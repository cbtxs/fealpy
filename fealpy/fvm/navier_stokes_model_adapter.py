"""Manufactured Navier-Stokes model adapters for FVM solver frontends."""

from fealpy.model import PDEModelManager


class NavierStokesModelAdapter:
    """Shared PDE-example setup used by Navier-Stokes FVM model classes."""

    @staticmethod
    def _resolve_navier_stokes_pde(pde):
        return PDEModelManager("navier_stokes").get_example(pde) if isinstance(pde, int) else pde

    @staticmethod
    def _normalized_mesh_type(mesh_type: str) -> str:
        return "uniform_quad" if mesh_type == "uniform_qrad" else mesh_type

    def _init_momentum_coefficients(self, options) -> None:
        """Initialize scalar density and dynamic viscosity for momentum solves."""
        rho_value = options.get("rho", None)
        if rho_value is None:
            rho_value = getattr(self.pde, "rho", 1.0)

        mu_value = options.get("mu", None)
        if mu_value is None:
            for name in ("mu", "viscosity", "nu"):
                if hasattr(self.pde, name):
                    mu_value = getattr(self.pde, name)
                    break
            else:
                mu_value = 1.0

        self.rho = self._as_positive_scalar(rho_value, "rho")
        self.mu = self._as_positive_scalar(mu_value, "mu")

    def _init_navier_stokes_mesh(self, options, *, default_mesh_type: str, normalize_mesh_type: bool = False):
        """Build the PDE-example mesh for manufactured or benchmark adapters."""
        mesh_type = options.get("mesh_type") or getattr(self.pde, "default_mesh_type", default_mesh_type)
        if normalize_mesh_type:
            mesh_type = self._normalized_mesh_type(mesh_type)

        mesh_refine = int(options.get("mesh_refine", 0))
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")

        if getattr(self.pde, "supports_geometric_refine", False):
            return self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)

        mesh_options = {}
        if "nx" in options:
            mesh_options["nx"] = int(options["nx"])
        if "ny" in options:
            mesh_options["ny"] = int(options["ny"])
        if "nz" in options:
            mesh_options["nz"] = int(options["nz"])
        mesh = self.pde.init_mesh[mesh_type](**mesh_options)
        if mesh_refine == 0:
            return mesh

        if hasattr(mesh, "uniform_refine"):
            mesh.uniform_refine(mesh_refine)
            return mesh

        raise ValueError("mesh does not provide uniform_refine().")
