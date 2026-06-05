import numpy as np
from fealpy.backend import backend_manager as bm

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.csm.material.hyperelastic_material import HyperElasticMaterial
from fealpy.csm.fem.hyperelastic_residual_integrator import HyperElasticResidualIntegrator
from fealpy.csm.fem.hyperelastic_tangent_integrator import HyperElasticTangentIntegrator


# ===============================
# 1. Mesh
# ===============================
node = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
], dtype=float)

cell = np.array([[0, 1, 2]])

mesh = TriangleMesh(node, cell)


# ===============================
# 2. Space（关键：统一 vector space）
# ===============================
scalar_space = LagrangeFESpace(mesh, p=1)
space = TensorFunctionSpace(scalar_space, (-1,2))   # 2D elasticity


# ===============================
# 3. DOF
# ===============================
gdof = space.number_of_global_dofs()
uh = bm.zeros(gdof)


# ===============================
# 4. Material
# ===============================
material = HyperElasticMaterial(
    C10=1.0,
    C20=0.1,
    C30=0.01,
    D1=0.1
)


# ===============================
# 5. Operators
# ===============================
residual = HyperElasticResidualIntegrator(material,space)
tangent = HyperElasticTangentIntegrator(material,space)
# 6. Newton loop
# ===============================
for it in range(6):

    R = residual.assembly_cell_vector(space, uh)
    K = tangent.assembly_cell_matrix(space, uh)

    R = R.reshape(-1)
    K = K.reshape(R.size, R.size)

    normR = np.linalg.norm(R)
    print(f"iter {it}, ||R|| = {normR}")

    if normR < 1e-10:
        break

    du = np.linalg.solve(K, -R)
    uh += bm.array(du)


print("\nuh =", uh)