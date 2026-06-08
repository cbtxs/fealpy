from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace


# =====================================================
# 1. Mesh: one 1×1×1 hexahedral element
# =====================================================

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
# 确定形函数
p = 1
scalar_space = LagrangeFESpace(mesh, p=p)


# =====================================================
# 3. Vector displacement space
# =====================================================

GD = mesh.geo_dimension()

space = TensorFunctionSpace(
    scalar_space,
    shape=(GD, -1)
)

def dof(node, direction, NN):
    """
    Global vector dof number.

    direction:
        0 -> ux
        1 -> uy
        2 -> uz
    """
    return direction * NN + node


def make_displacement_bc(mesh, scalar_space, ubar):
    """
    Dirichlet boundary condition for one-element Yeoh benchmark.

    """

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
    # Main loading condition
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

ubar = 0.2

dbc = make_displacement_bc(mesh, scalar_space, ubar)

print("dbc =")
for k, v in dbc.items():
    print(k, v)