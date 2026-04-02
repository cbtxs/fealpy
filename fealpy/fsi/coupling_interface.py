from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike
from scipy.interpolate import Rbf # 也可以使用FEALPy内置的插值算子


class CouplingInetrface:
    """流固耦合界面处理类
    负责流体边界与固体边界之间载荷（压力）与运动（位移/速度）的传递
    """
    def __init__(self, fluid_model, solid_model, fluid_face_idx, solid_face_idx):
        """
        :param fluid_model: 流体模型实例 (需包含 mesh 对象)
        :param solid_model: 固体模型实例 (需包含 mesh 对象)
        :param fluid_face_idx: 流体侧耦合界面的索引
        :param solid_face_idx: 固体侧耦合界面的索引
        """
        self.fluid_model = fluid_model
        self.solid_model = solid_model
         
        # 获取界面坐标
        self.f_interface_coords = self._get_fluid_interface_coords(fluid_face_idx)
        self.s_interface_coords = self._get_solid_interface_coords(solid_face_idx)
        
        # 预计算映射矩阵或插值器（针对弯管这种拓扑不变的情况）
        # 这里以 RBF 插值为例，处理不匹配网格
        self.interp_p_to_s = None # 流体压力 -> 固体载荷 的插值器
        self.interp_d_to_f = None # 固体位移 -> 流体网格 的插值器
        
    def _get_fluid_interface_coords(self, idx):
        mesh = self.fluid_model.mesh
        node = mesh.entity('node')
        # 假设 idx 是界面节点的索引
        return node[idx]
    
    def _get_solid_interface_coords(self, idx):
        mesh = self.solid_model.mesh
        node = mesh.entity('node')
        return node[idx]

    def transfer_fluid_to_solid(self):
        """
        满足力平衡条件: σ·n = -p·n
        将流体压力(Scalar)映射为固体界面的等效节点力(Vector)
        """
        # 1. 从流体模型获取界面压力 p
        pressure = self.fluid_model.get_interface_pressure() # (N_f_interface,)
        
        # 2. 构建从流体节点到固体节点的映射 (若网格不匹配)
        # 简单示例：使用 RBF 将压力值插值到固体界面节点上
        rbf = Rbf(self.f_interface_coords[:, 0], 
                  self.f_interface_coords[:, 1], 
                  self.f_interface_coords[:, 2], 
                  pressure, function='multiquadric')
        p_on_solid = rbf(self.s_interface_coords[:, 0], 
                         self.s_interface_coords[:, 1], 
                         self.s_interface_coords[:, 2])
        
        # 3. 计算法向力：F = -p * n * Area
        # 注意：此处建议在固体模型侧根据单元法向量 n 转换为节点载荷
        self.solid_model.apply_surface_load(p_on_solid) 
        
    def transfer_solid_to_fluid(self, dt: float):
        """
        满足速度连续条件: u_f = u_s
        将固体位移映射回流体界面，并计算界面速度更新流体边界
        """
        # 1. 获取固体界面位移 d_s
        disp_s = self.solid_model.get_interface_displacement() # (N_s_interface, 3)
        
        # 2. 插值到位变流体界面节点 d_f
        # 同样使用 RBF 插值处理位移矢量
        disp_f = bm.zeros_like(self.f_interface_coords)
        for i in range(3): # 对 x, y, z 分量分别插值
            rbf = Rbf(self.s_interface_coords[:, 0], 
                      self.s_interface_coords[:, 1], 
                      self.s_interface_coords[:, 2], 
                      disp_s[:, i])
            disp_f[:, i] = rbf(self.f_interface_coords[:, 0], 
                               self.f_interface_coords[:, 1], 
                               self.f_interface_coords[:, 2])
        
        # 3. 计算界面速度 u_f = disp_f / dt (用于 Dirichlet 边界)
        velocity_f = disp_f / dt
        
        # 4. 更新流体模型：包括网格坐标更新和边界条件设置
        # 更新流体网格位置 (ALE 基础)
        self.fluid_model.update_mesh_displacement(disp_f)
        # 设置速度边界条件
        self.fluid_model.set_interface_velocity(velocity_f)