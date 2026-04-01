from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike

from .coupling_interface import CouplingInetrface


class CouplingFSIAlgorithm:
    """
    """
    def __init__(self, fsi_interface: CouplingInetrface, max_iter: int = 100, tolerance: float = 1e-6):
        self.fsi_interface = fsi_interface
        self.max_iter = max_iter  # 最大迭代次数
        self.tolerance = tolerance  # 收敛容忍度

    def run(self):
        """
        执行流固耦合算法
        """
        for iteration in range(self.max_iter):
            # 1. 流体 → 固体：将流体压力传递给固体模型
            pressure = self.fluid_model.get_pressure_field()  # 获取流体模型的压力场
            self.fsi_interface.transfer_fluid_to_solid(pressure)

            # 2. 固体 → 流体：将固体模型的位移传回流体模型
            displacement = self.solid_model.calculate_displacement()  # 固体模型计算位移
            self.fsi_interface.transfer_solid_to_fluid(displacement)

            # 3. 检查收敛性（可以使用位移变化、压力变化等作为标准）
            if self.check_convergence(pressure, displacement):
                print(f"Converged at iteration {iteration + 1}")
                break
            else:
                print(f"Iteration {iteration + 1} not converged.")

    def check_convergence(self, last_pressure, last_displacement):
        """
        检查流固耦合的收敛性
        """
        pressure_change = self.fluid_model.calculate_pressure_change(last_pressure)
        displacement_change = self.solid_model.calculate_displacement_change(last_displacement)

        return pressure_change < self.tolerance and displacement_change < self.tolerance