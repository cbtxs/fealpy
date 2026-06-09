import numpy as np
import matplotlib.pyplot as plt
from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.csm.material.hyperelastic_material import HyperElasticMaterial
from fealpy.csm.fem.hyperelastic_lfem_model import HyperElasticLFEMModel


# =====================================================
# 1. DOF helper
# =====================================================

def dof(node, direction, NN):

    return direction * NN + node

def compute_D1_from_poisson(C10, nu):

    inv_D1 = 2.0 * C10 * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
    return 1.0 / inv_D1

def make_yeoh_material(nu):
    C10 = 0.18
    C20 = -2.0e-3
    C30 = 5.0e-5

    D1 = compute_D1_from_poisson(C10, nu)

    material = HyperElasticMaterial(
        C10=C10,
        C20=C20,
        C30=C30,
        D1=D1,
    )

    return material, D1

# =====================================================
# 2. Mesh and space
# =====================================================

def create_one_hex_mesh_and_space():

    node = bm.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=bm.float64)

    cell = bm.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
    ], dtype=bm.int8)

    mesh = HexahedronMesh(node, cell)

    p = 1
    scalar_space = LagrangeFESpace(mesh, p=p)

    GD = mesh.geo_dimension()

    space = TensorFunctionSpace(
        scalar_space,
        shape=(GD, -1)
    )

    return mesh, scalar_space, space


# =====================================================
# 3. Boundary condition
# =====================================================

def make_displacement_bc(mesh, scalar_space, ubar):

    ipoints = scalar_space.interpolation_points()
    NN = scalar_space.number_of_global_dofs()

    tol = 1.0e-12

    left_nodes = []
    right_nodes = []

    for i in range(NN):
        x = float(ipoints[i, 0])

        if abs(x - 0.0) < tol:
            left_nodes.append(i)

        if abs(x - 1.0) < tol:
            right_nodes.append(i)

    dbc = {}

    # -------------------------------
    # Main displacement loading
    # -------------------------------

    # Left face: ux = 0
    for n in left_nodes:
        dbc[dof(n, 0, NN)] = 0.0

    # Right face: ux = ubar
    for n in right_nodes:
        dbc[dof(n, 0, NN)] = ubar

    # -------------------------------
    # Minimal constraints
    # remove rigid body motion
    # -------------------------------

    # node 0: uy = 0
    dbc[dof(0, 1, NN)] = 0.0

    # node 0: uz = 0
    dbc[dof(0, 2, NN)] = 0.0

    # node 3: uz = 0
    dbc[dof(3, 2, NN)] = 0.0

    return dbc


# =====================================================
# 4. Right-face reaction helper
# =====================================================

def get_right_x_dofs(scalar_space):
    """
    Find x-direction dofs on the right face X = 1.
    """

    ipoints = scalar_space.interpolation_points()
    NN = scalar_space.number_of_global_dofs()

    tol = 1.0e-12

    right_nodes = []

    for i in range(NN):
        x = float(ipoints[i, 0])
        if abs(x - 1.0) < tol:
            right_nodes.append(i)

    right_x_dofs = [dof(n, 0, NN) for n in right_nodes]

    return right_nodes, right_x_dofs


# =====================================================
# 5. Run one benchmark case
# =====================================================

def run_one_case(ubar, nu=0.495, q=2, nsteps=None, verbose=False):
    """
    Run one displacement-controlled Yeoh benchmark case.

    Parameters
    ----------
    ubar : float
        Prescribed x displacement on right face.
    nu : float
        Poisson ratio used to compute D1.
    q : int
        Quadrature order.
    nsteps : int or None
        Load steps. If None, chosen automatically.
    verbose : bool
        Print Newton iteration information.

    Returns
    -------
    result : dict
    """

    # -----------------------------
    # Mesh and finite element space
    # -----------------------------
    mesh = make_one_hex_mesh()

    scalar_space = LagrangeFESpace(mesh, p=1)

    # 注意：这里保持你当前已经验证过的自由度顺序
    # 即 ux: 0~NN-1, uy: NN~2NN-1, uz: 2NN~3NN-1
    space = TensorFunctionSpace(scalar_space, shape=(3, -1))

    NN = scalar_space.number_of_global_dofs()

    # -----------------------------
    # Material
    # -----------------------------
    material, D1 = make_yeoh_material(nu)

    # -----------------------------
    # Boundary condition
    # -----------------------------
    dbc, left_nodes, right_nodes = make_displacement_bc(
        mesh=mesh,
        scalar_space=scalar_space,
        ubar=ubar,
    )

    # -----------------------------
    # Load step choice
    # -----------------------------
    if nsteps is None:
        nsteps = max(5, int(abs(ubar) / 0.05) + 1)

        # 非常接近不可压缩时更难收敛，多给一些 load step
        if nu > 0.499:
            nsteps = max(nsteps, 20)

    # -----------------------------
    # Nonlinear model
    # -----------------------------
    model = HyperElasticLFEMModel(
        space=space,
        material=material,
        dbc=dbc,
        q=q,
    )

    converged, uh = model.solve(
        nsteps=nsteps,
        tol=1.0e-8,
        maxit=30,
        line_search=True,
        verbose=verbose,
    )

    # -----------------------------
    # Reaction force
    # -----------------------------
    reaction = model.reaction_force()

    right_ux_dofs = [dof(node, 0, NN) for node in right_nodes]

    Fx = float(np.sum(np.asarray(reaction)[right_ux_dofs]))

    # 初始截面积 A0 = 1 mm * 1 mm = 1 mm^2
    A0 = 1.0
    nominal_stress = Fx / A0

    # 初始长度 L0 = 1 mm
    L0 = 1.0
    stretch = 1.0 + ubar / L0

    return {
        "nu": nu,
        "D1": D1,
        "ubar": ubar,
        "stretch": stretch,
        "reaction_force": Fx,
        "nominal_stress": nominal_stress,
        "converged": converged,
        "nsteps": nsteps,
    }

def run_sweep_for_two_nu():
    nus = [0.495, 0.49999]

    stretch_values = np.linspace(0.7, 1.3, 13)
    # 后面稳定以后再改成：
    # stretch_values = np.linspace(0.5, 1.5, 21)

    all_results = []

    for nu in nus:
        print(f"\n==============================")
        print(f"Running sweep for nu = {nu}")
        print(f"==============================")

        for stretch in stretch_values:
            ubar = stretch - 1.0

            result = run_one_case(
                ubar=ubar,
                nu=nu,
                q=2,
                nsteps=None,
                verbose=False,
            )

            all_results.append(result)

            print(
                f"nu = {nu:.5f}, "
                f"stretch = {result['stretch']:.6f}, "
                f"stress = {result['nominal_stress']:.12e}, "
                f"converged = {result['converged']}"
            )

    return all_results

def save_results_to_csv(results, filename="yeoh_one_element_two_nu_sweep.csv"):
    import csv

    fieldnames = [
        "nu",
        "D1",
        "ubar",
        "stretch",
        "reaction_force",
        "nominal_stress",
        "converged",
        "nsteps",
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"\nSaved csv to: {filename}")


def plot_results(results, filename="yeoh_one_element_two_nu_stress_stretch.png"):
    import matplotlib.pyplot as plt

    nus = sorted(set(row["nu"] for row in results))

    plt.figure(figsize=(7, 5))

    for nu in nus:
        rows = [row for row in results if row["nu"] == nu]
        rows = sorted(rows, key=lambda r: r["stretch"])

        x = [row["stretch"] for row in rows]
        y = [row["nominal_stress"] for row in rows]

        plt.plot(
            x,
            y,
            marker="o",
            label=f"nu = {nu}",
        )

    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(1.0, linewidth=0.8)

    plt.xlabel("Stretch ratio")
    plt.ylabel("Nominal stress")
    plt.title("Yeoh one-element tension/compression benchmark")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    print(f"Saved figure to: {filename}")
# =====================================================
# 6. Main test
# =====================================================

if __name__ == "__main__":
    results = run_sweep_for_two_nu()

    save_results_to_csv(
        results,
        filename="yeoh_one_element_two_nu_sweep.csv",
    )

    plot_results(
        results,
        filename="yeoh_one_element_two_nu_stress_stretch.png",
    )