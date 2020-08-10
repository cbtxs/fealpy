
"""

Notes
-----

    1. 混合物的摩尔浓度 c 的计算公式为
        c = p/(ZRT), 
        Z^3 - (1 - B)Z^2 + (A - 3B^2 -2B)Z - (AB - B^2 -B^3) = 0
        A = a p/(R^2T^2)
        B = b p/(RT)
        其中:

        a = 3
        b = 1/3

        p 是压力 
        Z 是压缩系数 
        R 是气体常数
        T 是温度
        M_i 是组分 i 的摩尔质量
        
        混合物的密度计算公式为:

        rho = M_0 c_0 + M_1 c_1 + M_2 c_2 ...

    2. c_i : 组分 i 的摩尔浓度
       z_i : 组分 i 的摩尔分数

       c_i = z_i c

    3. 
    甲烷: methane,  CH_4,   16.04 g/mol,  0.42262 g/cm^3
    乙烷: Ethane, C_2H_6,   30.07 g/mol, 1.212 kg/m^3
    丙烷: Propane,  C_3H_8,  44.096 g/mol,  1.83 kg/m^3 
    (25度 100 kPa

    4. 气体常数  R = 8.31446261815324 	J/K/mol

    5. 6.02214076 x 10^{23}


References
----------
[1] https://www.sciencedirect.com/science/article/abs/pii/S0378381217301851

Authors
    Huayi Wei, weihuayi@xtu.edu.cn
"""
import numpy as np
from scipy.sparse import coo_matrix
from fealpy.mesh import MeshFactory
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.timeintegratoralg.timeline import UniformTimeLine


class Model_1():
    def __init__(self):
        self.m = [0.01604, 0.03007, 0.044096] # kg/mol 一摩尔质量
        self.R = 8.31446261815324 # J/K/mol
        self.T = 397 # K 绝对温度

    def init_pressure(self, pspace):
        """

        Notes
        ----
        目前压力用分片常数逼近。
        """
        ph = pspace.function()
        ph[:] = 50 # 初始压力
        return ph

    def init_molar_density(self, cspace):
        ch = cspace.function(dim=3)
        ch[:, 2] = cspace.local_projection(1)
        return ch

    def space_mesh(self, n=50):
        box = [0, 50, 0, 50]
        mf = MeshFactory()
        mesh = mf.boxmesh2d(box, nx=n, ny=n, meshtype='tri')
        return mesh

    def time_mesh(self, n=100):
        timeline = UniformTimeLine(0, 0.1, n)
        return timeline

    def molar_dentsity(self, p):
        """

        Notes
        ----
        给一个分片常数的压力，计算混合物的浓度 c
        """
        NC = len(p)
        t = self.R*self.T 
        A = 3*p/t**2
        B = p/t/3 
        
        a = np.ones((NC, 4), dtype=p.dtype)
        a[:, 1] = B - 1
        a[:, 2] = A - 3*B**2 - 2*B
        a[:, 3] = -A*B + B**2 - B**3
        r = list(map(np.roots, c))
        print(r)

class ShaleGasSolver():
    def __init__(self, model):
        self.model = model
        self.mesh = model.space_mesh()
        self.timeline =  model.time_mesh() 
        self.uspace = RaviartThomasFiniteElementSpace2d(self.mesh, p=0)
        self.cspace = ScaledMonomialSpace2d(self.mesh, p=1) # 线性间断有限元空间

        self.uh = self.uspace.function() # 速度
        self.ph = model.init_pressure(self.uspace.smspace) # 初始压力

        # 三个组分的摩尔密度, 只要算其中 c_0, c_1 
        self.ch = model.init_molar_density(self.cspace) 

        # TODO：初始化三种物质的浓度
        self.options = {
                'viscosity': 1.0,    # 粘性系数
                'permeability': 1.0, # 渗透率 
                'temperature': 397, # 初始温度 K
                'pressure': 50,   # 初始压力
                'porosity': 0.2,  # 孔隙度
                'injecttion_rate': 0.1,  # 注入速率
                'compressibility': 0.001, #压缩率
                'pmv': (1.0, 1.0, 1.0)} # 偏摩尔体积

        c = self.options['viscosity']/self.options['permeability']
        self.M = c*self.uspace.mass_matrix()
        self.B = -self.uspace.div_matrix()

        dt = self.timeline.dt
        c = self.options['porosity']*self.options['compressibility']/dt
        self.D = c*self.uspace.smspace.mass_matrix() 

    def get_current_pv_matrix(self, data):
        """
        (c_h[i] v_h\cdot n, w_h)_{\partial K}
        """

        vh = data[0]
        ph = data[1]
        ch = data[2]

        mesh = self.mesh
        edge2cell = mesh.ds.edge_to_cell()
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1]
        edge2dof = self.dof.edge_to_dof() 

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, etype='edge')
        en = mesh.edge_unit_normal() # (NE, 2)

        # 组分浓度在积分点上的值, 这里有 3 个组分
        lval = ch(ps, index=edge2cell[:, 0]) # (NQ, NE, 3) 
        rval = ch(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        rval[:, isBdEdge] = 0.0 # 边界值设为 0, 这样后面不产生贡献.

        # 速度在积分点上的基函数值
        phi = self.space0.edge_basis(ps) # (NQ, NE, 1, 2)

        # 压力测试函数空间基函数的值
        lphi = self.space0.smspace.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, 3)
        rphi = self.space0.smspace.basis(ps, index=edge2cell[:, 1]) # (NQ, NE, 3)
        measure = self.integralalg.edgemeasure

        LM = np.einsum('i, ijk, ijl, ijmn, jn, j->jlm', ws, lval, lphi, phi, en, measure)
        RM = np.einsum('i, ijk, ijl, ijmn, jn, j->jlm', ws, rval, rphi, phi, -en, measure)

        gdof0 = self.space0.smspace.number_of_global_dofs()
        gdof1 = self.space0.number_of_global_dofs()
        cell2dof = cell.space1.cell_to_dof()
        I = np.broadcast_to(edge2dof[:, :, None], LM.shape)

        J = np.broadcast_to(cell2dof[edge2cell[:, 0]][:, None, :], LM.shape)
        P = coo_matrix((LM.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        J = np.broadcast_to(cell2dof[edge2cell[:, 1]][:, None, :], RM.shape)
        P += coo_matrix((RM.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        return P.tocsr()

    def get_current_density_vector(self, data, timeline):
        vh = data[0]
        ph = data[1]
        ch = data[2]
        options  = self.options

        mesh = self.mesh

        # 0. 单元内的浓度方程积分
        qf = self.integralalg.cellintegrator if q is None else mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs, etype='cell')
        measure = self.integralalg.cellmeasure

        cval = ch.value(ps) # (NQ, NC, 3) 3 个组分的值
        vval = vh.value(bcs) # (NQ, NC, 2) 向量的值
        gphi = self.space1.grad_basis(ps) # (NQ, NC, 3, 2)

        # bb: (NC, 3, 3)
        bb = np.einsum('i, ijk, ijmn, ijln, j->jkl', ws, cval, vval, gphi, measure)

        # 1.  

        qf = self.integralalg.edgeintegrator if q is None else mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs, etype='edge')
        en = mesh.edge_unit_normal() # (NE, 2)
        
        lphi = np.space1.basis(ps, index=edge2cell[:, 0]) # (NQ, NC, 3)
        rphi = np.space1.basis(ps, index=edge2cell[:, 1]) # (NQ, NC, 3)

        M = np.einsum('', ws, lphi, rphi)
        

    def get_current_left_matrix(self, data, timeline):

        vh = data[0]
        ph = data[1]
        ch = data[2]
        options  = self.options

        A = self.A
        B = self.B
        P = self.get_current_pv_matrix()
        M = self.M
        AA = bmat([[A, -B], [P, M]], format='csr')
        return AA

    def get_current_right_vector(self, data, timeline):
        vh = data[0]
        ph = data[1]
        ch = data[2]
        options  = self.options
        return self.M@ph

    def solve(self, data, A, b, solver, timeline):
        pass



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    model = Model_1()

    solver = ShaleGasSolver(model)

    model.molar_dentsity(solver.ph)

    mesh = solver.mesh

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
