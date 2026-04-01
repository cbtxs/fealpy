from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike


class CouplingInetrface:
    """_summary_
    """
    def __init__(self, fluid_model, solid_model):
        self.fluid_model = fluid_model # 流体模型实例
        self.solid_model = solid_model # 固体模型实例
        
    def transfer_fluid_to_solid(self, pressure: TensorLike):
        """
        从流体模型传递压力到固体模型
        将流体内壁面上的压力场和壁面剪切力映射为结构载荷
        """
        # 将流体压力映射到固体模型的边界上
        self.solid_model.set_pressure(pressure)

    def transfer_solid_to_fluid(self, displacement: TensorLike):
        """
        从固体模型传递位移/速度到流体模型
        将壁面位移/速度传回流体侧，驱动流体网格运动
        """
        # 将固体模型的壁面位移/速度传递给流体模型
        self.fluid_model.update_boundary_conditions(displacement)