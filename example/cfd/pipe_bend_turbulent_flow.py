from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.solver import spsolve, cg, gmres
from fealpy.backend import backend_manager as bm
from fealpy.mesher import ElbowPipeMesher
from fealpy.mesh import TetrahedronMesh

options = {
    'backend': 'numpy',
    'pde': 1,
    'init_mesh': 'tri',
    'box': [0.0, 2.2, 0.0, 0.41],
    'center': (0.2, 0.2),
    'radius': 0.05,
    'n_circle': 1000,
    'lc': 0.004,
    'rho': 1.0,
    'mu': 1e-3,
    'method': 'Newton',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main_cylinder',
    'maxit': 1,
    'maxstep': 1000,
    'tol': 1e-10
}

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.3,     # 使用默认网格大小策略
    "mesh_size_bend": 0.3,
    "mesh_size_interface": 0.3,
}
mesher = ElbowPipeMesher(params)
mesh = mesher.init_mesh()
mesh.to_vtk("pipe_bend_mesh.vtu")

# geom = PipeGeometry()
# geom.build()
# mesher = PipeMesh(geom, mesh_size=0.3)
# mesh = mesher.generate_mesh()

# 网格可视化
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# plt.savefig("1.png")

pde = PipeBendTurbulentFlow()

equation = StationaryIncompressibleNS(pde=pde)
fem = Ossen(equation=equation, mesh=mesh)
# fem = Newton(equation=equation, mesh=mesh)

u0 = fem.uspace.function()
u1 = fem.uspace.function()
p0 = fem.pspace.function()
p1 = fem.pspace.function()

for i in range(100):
    BForm = fem.BForm()
    LForm = fem.LForm()
    fem.update(u0=u0)
    A = BForm.assembly() 
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde)
    if equation.pressure_neumann == True:
        A, b = fem.lagrange_multiplier(A, b)
    x = spsolve(A, b)

    ugdof = fem.uspace.number_of_global_dofs()
    
    u1[:] = x[:ugdof]
    if equation.pressure_neumann == True:
        p1[:] = x[ugdof:-1]
    else:
        p1[:] = x[ugdof:]

    mesh.nodedata["uh"] = u1.reshape(3, -1).T
    mesh.nodedata["ph"] = p1
    mesh.to_vtk(f"stationary_sst_k_omega_{i+1}.vtu")

    res_u = mesh.error(u0, u1)
    res_p = mesh.error(p0, p1)
    print("res_u", res_u)
    print("res_p", res_p)

    if res_u + res_p < 1e-8:
        break

    u0[:] = u1[:]
    p0[:] = p1[:]














exit()
from fealpy.backend import backend_manager as bm
from fealpy.cfd.equation.base import BaseEquation
from fealpy.cfd.simulation.fem.iterative_method import IterativeMethod
from typing import Union, Callable

CoefType = Union[int, float, Callable]

class IncompressibleRANS(BaseEquation):
    def __init__(self, pde, init_variables = False):
        super().__init__(pde)
        self._coefs = {
            'time_derivative': 1,  # 时间导数项系数
            'convection': 1,       # 对流项系数
            'viscosity': 1,        # 粘性项系数
            'pressure': 1,         # 压力项系数
            'body_force': 1,        # 外力项系数
        }
        self._variables = { 
            'velocity': None,     # 速度变量
            'pressure': None     # 压力变量
        }
        self.pde = pde
        if init_variables:
            self.initialize_from_pde(pde)
        
        if pde.is_pressure_boundary() == 0 :
            self.pressure_neumann = True
        else:
            self.pressure_neumann = False
    
    def initialize_from_pde(self, pde):
        """
        根据 pde 对象初始化系数和变量。

        参数:
            pde: PDE 对象，包含 rho, mu, mu_t, R, velocity, pressure, body_force 等属性

        处理:
            - 如果有 rho 和 mu，直接使用。
            - 如果只有 R，假设 rho=1，计算 mu=rho/R。
            - 如果两者都没有，使用默认值 rho=1, mu=1。
            - 如果有 body_force，使用其值，否则默认 0。
            - 设置 velocity 和 pressure 的初始值。
        """
        # 处理物理参数
        rho = pde.rho
        mu = pde.mu
    
        # 设置系数 
        self._coefs['time_derivative'] = rho
        self._coefs['convection'] = rho
        self._coefs['pressure'] = 1
        self._coefs['viscosity'] = mu
        self._coefs['body_force'] = getattr(pde, 'source', 0)

    # 定义属性访问
    @property
    def variables(self):
        """变量字典"""
        return self._variables

    @property
    def velocity(self):
        """速度变量"""
        return self._variables['velocity']

    @property
    def pressure(self):
        """压力变量"""
        return self._variables['pressure']
    
    @property
    def coefs(self):
        """系数字典"""
        return self._coefs
    
    @property
    def coef_time_derivative(self) -> float | Callable:
        """时间导数项系数"""
        return self._coefs['time_derivative']

    @property
    def coef_convection(self) -> CoefType:
        """对流项系数（惯性项）"""
        return self._coefs['convection']

    @property
    def coef_viscosity(self) -> CoefType:
        """粘性项系数"""
        return self._coefs['viscosity']

    @property
    def coef_pressure(self) -> CoefType:
        """压力项系数"""
        return self._coefs['pressure']
    
    @property
    def coef_body_force(self) -> CoefType:
        """外力项系数"""
        return self._coefs['body_force']
    
    def set_coefficient(
        self,
        term: str,
        value: CoefType
    ) -> None:
        """
        设置方程项的系数。
        
        参数:
            term: 方程项名称（'mass', 'inertia', 'viscosity', 'pressure'）
            value: 系数值，必须是标量（int/float）或可调用对象（callable）
        
        异常:
            ValueError: 如果值不是标量或可调用对象
            KeyError: 如果方程项名称不存在
        """
        if term not in self._coefs:
            raise KeyError(f"未知方程项: {term}。可选: {list(self._coefs.keys())}")
        
        if not (isinstance(value, (int, float)) or callable(value)):
            raise ValueError("系数必须是标量（int/float）或可调用对象（callable）")
        
        self._coefs[term] = value 
       
    def __str__(self) -> str:
        """返回所有方程项系数和变量的字符串表示"""
        terms_str = "\n".join(
            f"Term[{term}]: {'Callable' if callable(coeff) else coeff}"
            for term, coeff in self._coefs.items()
        )
        
        variables_str = "\n".join(
            f"Variable[{name}]: {type(value).__name__ if value is not None else 'None'}"
            for name, value in self._variables.items()
        )
        
        return f"=== IncompressibleNS ===\n" \
               f"coefficients:\n{terms_str}\n\n" \
               f"Variables:\n{variables_str}"
    
    def set_variable(self, name: str, value) -> None:
        """
        设置求解变量（速度或压力）

        Args:
            name: 变量名（'velocity' 或 'pressure'）
            value: 变量值

        Raises:
            KeyError: 如果变量名不存在
        """
        if name not in self._variables:
            raise KeyError(f"Invalid variable '{name}'. Valid variables: {list(self._variables.keys())}")
        self._variables[name] = value

    def set_coefs(self, **kwargs) -> None:
        """
        批量设置方程项系数
        
        Example:
            ns.set_coefs(viscosity=0.1, body_force=lambda x: x[0])
        """
        for term, value in kwargs.items():
            self.set_coefficient(term, value)

    def set_variables(self, **kwargs) -> None:
        """
        批量设置求解变量
        
        Example:
            ns.set_variables(velocity=u, pressure=p)
        """
        for name, value in kwargs.items():
            self.set_variable(name, value)
    
class IncompressibleTurbulentKineticEnergy(BaseEquation):
    def __init__(self, pde):
        super().__init__(pde)
        self._coefs = {
            'time_derivative': 1,   # 时间导数项系数
            'diffusion': 1,         # 扩散项系数
            'production': 1,        # 湍流产生项系数
            'reaction': 1,          # 生成项系数
        }
        self._variables = {
            'kinetic': None,       # 湍动能变量
        }
        self.pde = pde
        self.initialize_from_pde(pde)

        if pde.is_pressible_boundary() == 0:
            self.pressure_neumann = True
        else:
            self.pressure_neumann = False
    
    def initialize_from_pde(self, pde):
        # 处理物理参数
        rho = pde.rho
        mu = pde.mu
        beta_s = pde.beta_s

        # 设置系数 
        self._coefs['time_derivative'] = rho
        self._coefs['convection'] = rho
        self._coefs['reaction'] = beta_s * rho
        self._coefs['diffusion'] = mu
        self._coefs['production'] = 0.0
    
    # 定义属性访问
    @property
    def coefs(self):
        return self._coefs# 处理物理参数
        rho = pde.rho
        mu = pde.mu
        beta_s = pde.beta_s

        # 设置系数 
        self._coefs['convection'] = rho
        self._coefs['reaction'] = beta_s * rho
        self._coefs['diffusion'] = mu
        self._coefs['production'] = 0.0
    
    @property
    def coef_time_derivative(self) -> float | Callable:
        return self._coefs['time_derivative']

    @property
    def coef_convection(self) -> CoefType:
        return self._coefs['convection']

    @property
    def coef_diffusion(self) -> CoefType:
        return self._coefs['diffusion']
    
    @property
    def coef_production(self) -> CoefType:
        return self._coefs['production']
    
    @property
    def coef_reaction(self) -> CoefType:
        return self._coefs['reaction']
    
    def set_coefficient(
        self,
        term: str,
        value: CoefType
    ) -> None:
        """
        设置方程项的系数。
        
        参数:
            term: 方程项名称（'mass', 'inertia', 'viscosity', 'pressure'）
            value: 系数值，必须是标量（int/float）或可调用对象（callable）
        
        异常:
            ValueError: 如果值不是标量或可调用对象
            KeyError: 如果方程项名称不存在
        """
        if term not in self._coefs:
            raise KeyError(f"未知方程项: {term}。可选: {list(self._coefs.keys())}")
        
        if not (isinstance(value, (int, float)) or callable(value)):
            raise ValueError("系数必须是标量（int/float）或可调用对象（callable）")
        
        self._coefs[term] = value 
    
    def __str__(self) -> str:
        """返回所有方程项系数和变量的字符串表示"""
        terms_str = "\n".join(
            f"Term[{term}]: {'Callable' if callable(coeff) else coeff}"
            for term, coeff in self._coefs.items()
        )
        
        variables_str = "\n".join(
            f"Variable[{name}]: {type(value).__name__ if value is not None else 'None'}"
            for name, value in self._variables.items()
        )
        
        return f"=== StationaryTurbulentKineticEnergy ===\n" \
               f"coefficients:\n{terms_str}\n\n" \
               f"Variables:\n{variables_str}"
    
    def set_variable(self, name: str, value) -> None:
        """
        设置求解变量（速度或压力）

        Args:
            name: 变量名（'velocity' 或 'pressure'）
            value: 变量值

        Raises:
            KeyError: 如果变量名不存在
        """
        if name not in self._variables:
            raise KeyError(f"Invalid variable '{name}'. Valid variables: {list(self._variables.keys())}")
        self._variables[name] = value

    def set_coefs(self, **kwargs) -> None:
        """
        批量设置方程项系数
        
        Example:
            ns.set_coefs(viscosity=0.1, production=lambda x: x[0])
        """
        for term, value in kwargs.items():
            self.set_coefficient(term, value)

    def set_variables(self, **kwargs) -> None:
        """
        批量设置求解变量
        
        Example:
            ns.set_variables(velocity=u, pressure=p)
        """
        for name, value in kwargs.items():
            self.set_variable(name, value)

class SpecificDissipationRate(BaseEquation):
    def __init__(self, pde):
        super().__init__(pde)
        self._coefs = {
            'time_derivative': 1,   # 时间导数项系数
            'diffusion': 1,         # 扩散项系数
            'production': 1,        # 湍流产生项系数
            'convection': 1,        # 对流项系数
            'cross_diffusion': 1,   # 湍流交叉扩散项系数
        }
        self._variables = {
            'omega': None,       # 耗散率变量
        }
        self.pde = pde
        self.initialize_from_pde(pde)

        if pde.is_pressible_boundary() == 0:
            self.pressure_neumann = True
        else:
            self.pressure_neumann = False
    
    def initialize_from_pde(self, pde):
        """
        根据 pde 对象初始化系数和变量。

        参数:
            pde: PDE 对象，包含 rho, mu, mu_t, R, velocity, pressure, production 等属性

        处理:
            - 如果有 rho 和 mu，直接使用。
            - 如果只有 R，假设 rho=1，计算 mu=rho/R。
            - 如果两者都没有，使用默认值 rho=1, mu=1。
            - 如果有 production，使用其值，否则默认 0。
            - 设置 velocity 和 pressure 的初始值。
        """
        # 处理物理参数
        rho = pde.rho
        mu = pde.mu
        gamma = pde.gamma
        beta = pde.beta
        sigma_omega = pde.sigma_omega
        sigma_omega2 = pde.sigma_omega2

        # 设置系数 
        self._coefs['time_derivative'] = rho
        self._coefs['convection'] = rho
        self._coefs['dissipation'] = beta * rho
        self._coefs['diffusion'] = mu
        self._coefs['cross_diffusion'] = -2 * rho * sigma_omega2
        self._coefs['production'] = gamma * rho
    
    # 定义属性访问
    @property
    def coefs(self):
        return self._coefs
    
    @property
    def variables(self):
        return self._variables
    
    @property
    def coef_time_derivative(self) -> CoefType:
        return self._coefs['time_derivative']

    @property
    def coef_diffusion(self) -> CoefType:
        return self._coefs['diffusion']

    @property
    def coef_convection(self) -> CoefType:
        return self._coefs['convection']

    @property
    def coef_dissipation(self) -> CoefType:
        return self._coefs['dissipation']

    @property
    def coef_cross_diffusion(self) -> CoefType:
        return self._coefs['cross_diffusion']

    @property
    def coef_production(self) -> CoefType:
        return self._coefs['production']
    
    def set_coefficient(
        self,
        term: str,
        value: CoefType
    ) -> None:
        """
        设置方程项的系数。
        
        参数:
            term: 方程项名称（'mass', 'inertia', 'viscosity', 'pressure'）
            value: 系数值，必须是标量（int/float）或可调用对象（callable）
        
        异常:
            ValueError: 如果值不是标量或可调用对象
            KeyError: 如果方程项名称不存在
        """
        if term not in self._coefs:
            raise KeyError(f"未知方程项: {term}。可选: {list(self._coefs.keys())}")
        
        if not (isinstance(value, (int, float)) or callable(value)):
            raise ValueError("系数必须是标量（int/float）或可调用对象（callable）")
        
        self._coefs[term] = value 
    
    def __str__(self) -> str:
        """返回所有方程项系数和变量的字符串表示"""
        terms_str = "\n".join(
            f"Term[{term}]: {'Callable' if callable(coeff) else coeff}"
            for term, coeff in self._coefs.items()
        )
        
        variables_str = "\n".join(
            f"Variable[{name}]: {type(value).__name__ if value is not None else 'None'}"
            for name, value in self._variables.items()
        )
        
        return f"=== StationarySpecificDissipationRate ===\n" \
               f"coefficients:\n{terms_str}\n\n" \
               f"Variables:\n{variables_str}"
    
    def set_variable(self, name: str, value) -> None:
        """
        设置求解变量（速度或压力）

        Args:
            name: 变量名（'velocity' 或 'pressure'）
            value: 变量值

        Raises:
            KeyError: 如果变量名不存在
        """
        if name not in self._variables:
            raise KeyError(f"Invalid variable '{name}'. Valid variables: {list(self._variables.keys())}")
        self._variables[name] = value

    def set_coefs(self, **kwargs) -> None:
        """
        批量设置方程项系数
        
        Example:
            ns.set_coefs(viscosity=0.1, production=lambda x: x[0])
        """
        for term, value in kwargs.items():
            self.set_coefficient(term, value)

    def set_variables(self, **kwargs) -> None:
        """
        批量设置求解变量
        
        Example:
            ns.set_variables(velocity=u, pressure=p)
        """
        for name, value in kwargs.items():
            self.set_variable(name, value)

class PDE():
    from fealpy.decorator import cartesian
    from fealpy.backend import TensorLike
    from fealpy.typing import Index, _S
    def __init__(self):
        self.rho = 1.0
        self.mu = 1.0
        self.mu_t = 1.0
        self.R = 1.0
        self.gamma = 1.0
        self.beta = 1.0
        self.beta_s = 1.0
        self.sigma_omega = 1.0
        self.sigma_omega2 = 1.0
    
    @cartesian
    def distance_t0_wallline(self, p: TensorLike) -> TensorLike:
        """计算点到壁面的距离"""
        R = 2.8
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        
        # 情况1: 上游直管
        d_up = 0.5 - bm.minimum(bm.sqrt(x**2 + y**2), 0.5)
        
        # 情况2: 下游直管
        d_down = 0.5 - bm.minimum(bm.sqrt(y**2 + (z - R)**2), 0.5)
        
        # 情况3: 弯管段
        dist_to_wall_xz = bm.sqrt((x - R)**2 + z**2)
        dist_to_arc_xz = bm.abs(dist_to_wall_xz - R)
        d_bend = 0.5 - bm.minimum(bm.sqrt(dist_to_arc_xz**2 + y**2), 0.5)
        
        # 根据x坐标选择合适的距离
        d_tail = bm.where(x >= R, d_down, d_bend)

        # 根据z坐标选择合适的距离
        d = bm.where(z <= 0, d_up, d_tail)

        return d
    
    @cartesian
    def strain_rate(self, u0, bcs, index):
            grad_u = u0.grad_value(bcs, index)
            grad_u_T = bm.swapaxes(grad_u, -1, -2)

            S_ij = 1/2 * (grad_u + grad_u_T)
            return S_ij
    
    @cartesian
    def tur_mu(self, u0, k0, omega0, points, bcs, index: Index = _S):
        beta_s = self.beta_s
        mu = self.mu
        rho = self.rho
        a1 = self.a1
        d = self.distance_t0_wallline(points)

        def shear_stress_limit_function():
            k0_value = bm.maximum(k0(bcs, index), 1e-10)
            arg2 = bm.maximum(2 * bm.sqrt(k0_value)/(beta_s * omega0(bcs, index) * d),
                            500 * mu/(d**2 * rho * omega0(bcs, index)))
            F2 = bm.tanh(arg2**2)
            return F2
        F2 = shear_stress_limit_function()

        S_ij = self.strain_rate(u0, bcs, index)
        S = bm.sqrt(2 * bm.sum(S_ij * S_ij, axis=(2, 3)))
        mu_t = a1 * k0(bcs, index)
        mu_t /= bm.maximum(a1 * omega0(bcs, index), S * F2)
        return mu_t
    
    @cartesian
    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(z + 10) < atol)
        return on_boundary
    
    @cartesian
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(x - 17.8) < atol)
        return on_boundary
    
    @cartesian
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        r = 0.5
        d = self.distance_t0_wallline(p)
        atol = 1e-12
        on_boundary = (bm.abs(d) < atol)
        return on_boundary
    
    # 动量方程
    @cartesian
    def inlet_velocity(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 0.0
        u[..., 2] = 1.224*(1.0 - (0.5 - d)/R)**(1/7)
        return u
    
    @cartesian
    def outlet_velocity(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_wallline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 1.224*(1.0 - (0.5-d)/R)**(1/7)
        u[..., 1] = 0.0
        u[..., 2] = 0.0
        return u
    
    @cartesian
    def outlet_pressure(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pressure = bm.zeros(x.shape)
        return pressure
    
    @cartesian
    def wall_velocity(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape)
        return u
    
    @cartesian
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        # return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        return None
    
    @cartesian
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        # if p is None:
        #     return 1
        # return self.is_outlet_boundary(p)
        return 0
    
    @cartesian
    def velocity_dirichlet(self, p: TensorLike, t) -> TensorLike:
        result = bm.zeros(p.shape)
        inlet = self.inlet_velocity(p, t)
        outlet = self.outlet_velocity(p, t)
        wall = self.wall_velocity(p, t)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_outlet = self.is_outlet_boundary(p)

        result[is_inlet] = inlet[is_inlet]
        result[is_wall] = wall[is_wall]
        result[is_outlet] = outlet[is_outlet]
        return result
    
    @cartesian
    def pressure_dirichlet(self, p: TensorLike, t) -> TensorLike:
        return self.outlet_pressure(p, t)
    
    @cartesian
    def source(self, p: TensorLike, t) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape)
        return result
    
    # k 方程
    @cartesian
    def k_dirichlet(self, p: TensorLike, t) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        k = bm.zeros(p[..., 0].shape)
        k[is_inlet] = 0.00375
        k[is_outlet] = 0.00375
        k[is_wall] = 0
        return k
    
    @cartesian
    def is_k_boundary(self, p: TensorLike) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        return is_wall | is_inlet 
    
    @cartesian
    def production_k(self, u0, k0, omega0, mu_t, bcs, index) -> TensorLike:
        result_0 = self.production_omega(u0, k0, mu_t, bcs, index)
        result_1 = 10 * self.beta_s * self.rho * k0(bcs, index) * omega0(bcs, index)
        result = bm.minimum(result_0, result_1)
        return result
    
    # omega 方程
    @cartesian
    def omega_dirichlet(self, p: TensorLike, t) -> TensorLike:
        d = self.distance_t0_wallline(p)
        nu = self.mu/self.rho
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        omega = bm.zeros(p[..., 0].shape)
        omega[is_inlet] = 1.597
        omega[is_wall] = bm.minimum((60 * nu / (self.beta * d**2)), 2e6)[is_wall]
        return omega
    
    @cartesian
    def is_omega_boundary(self, p: TensorLike) -> TensorLike:
        is_inlet = self.is_inlet_boundary(p)
        is_outlet = self.is_outlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        return is_inlet | is_wall
    
    @cartesian
    def production_omega(self, u0, k0, mu_t, bcs, index) -> TensorLike:
        S_ij = self.strain_rate(u0, bcs, index)
        grad_u = u0.grad_value(bcs, index)
        
        P = mu_t * bm.sum(S_ij * grad_u, axis=(2, 3))
        P -= 2/3 * k0(bcs, index) * bm.einsum("...ii -> ...", grad_u)
        return P
    
    @cartesian
    def cross_diffuison_f1(self, k1, omega0, points, bcs, index):
        d = self.distance_t0_wallline(p=points)
        rho = self.rho
        k1_value = k1(bcs, index)
        k1_value = bm.maximum(k1_value, 1e-10)
        arg1_11 = bm.sqrt(k1_value)
        arg1_11 /= self.beta_s * omega0(bcs, index) * d
        arg1_12 = 500 * self.mu / (d**2 * rho * omega0(bcs, index))
        arg1_1 = bm.maximum(arg1_11, arg1_12)

        def cross_diddusion():
            CD1 = 2 * rho * self.sigma_omega2
            reciprocal_omega0 = 1/omega0
            CD1 *= reciprocal_omega0(bcs, index)
            grad_k1 = k1.grad_value(bcs, index)
            grad_omega0 = omega0.grad_value(bcs, index)
            CD1 *= bm.sum(grad_k1 * grad_omega0, axis=(2))

            CD2 = 10e-10

            CD = bm.maximum(CD1, CD2)
            return CD
        CD = cross_diddusion()
        arg1_2 = 4 * rho * self.sigma_omega2 * k1(bcs, index)
        arg1_2 /= CD * d**2

        arg1 = bm.minimum(arg1_1, arg1_2)

        F1 = bm.tanh(arg1**4)
        return F1
    
class FEMRANS(IterativeMethod):
    
    def BForm(self):
        from fealpy.fem import (BilinearForm, ScalarConvectionIntegrator, ScalarMassIntegrator,
                                ViscousWorkIntegrator, PressWorkIntegrator, BlockForm)
        pspace = self.pspace
        uspace = self.uspace
        q = self.q

        A00 = BilinearForm(uspace)
        self.u_BM = ScalarMassIntegrator(q=q)
        self.u_BC = ScalarConvectionIntegrator(q=q)
        self.u_BVW = ViscousWorkIntegrator(q=q)

        A00.add_integrator(self.u_BM)
        A00.add_integrator(self.u_BC)
        A00.add_integrator(self.u_BVW)

        A01 = BilinearForm((pspace, uspace))
        self.u_BPW = PressWorkIntegrator(q=q)
        A01.add_integrator(self.u_BPW)

        A10 = BilinearForm((pspace, uspace))
        self.p_BPW = PressWorkIntegrator(q=q)
        A10.add_integrator(self.p_BPW)

        A = BlockForm([[A00, A01], [A10.T, None]])
        return A
    
    def LForm(self):
        from fealpy.fem import (LinearForm, BlockForm, 
                                ScalarSourceIntegrator, LinearBlockForm)
        pspace = self.pspace
        uspace = self.uspace
        q = self.q

        L0 = LinearForm(uspace)
        self.u_LSI = ScalarSourceIntegrator(q=q)
        self.u_LSI_t = ScalarSourceIntegrator(q=q)
        L0.add_integrator(self.u_LSI)
        L0.add_integrator(self.u_LSI_f)
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L
    
    def update(self, uk, u0, k0, omega0): 
        from fealpy.decorator import barycentric
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative
        cv = equation.coef_viscosity
        cc = equation.coef_convection
        pc = equation.coef_pressure
        cbf = equation.coef_body_force
        
        ## BilinearForm
        self.u_BM.coef = ctd/dt
        self.u_BPW.coef = -pc

        @barycentric
        def u_BVM_coef(bcs, index):
            points = self.uspace.mesh.bc_to_point(bcs, index)
            mu_t = equation.pde.tur_mu(u0=u0, k0=k0, omega0=omega0, bcs=bcs, points= points)
            mu_t = bm.minimum(mu_t, 1000 * equation.pde.mu)
            self.mu_t = mu_t
            cvcoef = cv(bcs, index)[..., bm.newaxis] if callable(cv) else cv
            cvcoef += mu_t
            return cvcoef
        self.u_BVW.coef = u_BVM_coef

        @barycentric
        def u_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            cccoef *= u0(bcs, index)
            return cccoef
        self.u_BC.coef = u_BC_coef

        ## LinearForm 
        @barycentric
        def u_LSI_coef(bcs, index):
            scoef = -2/3 * self.equation.pde.rho
            scoef *= k0.grad_value(bcs, index)
            return scoef
        self.u_LSI.source = u_LSI_coef
        
        @barycentric
        def u_LSI_f_coef(bcs, index):
            ctdcoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = ctdcoef * uk(bcs, index) / dt
            return result
        self.u_LSI_t.source = u_LSI_f_coef

class FEMTurbulentKineticEnergy(IterativeMethod):
    def BForm(self):
        from fealpy.functionspace import LagrangeFESpace
        from fealpy.fem import (BilinearForm, ScalarConvectionIntegrator, 
                                ScalarMassIntegrator, ScalarDiffusionIntegrator)
        
        self.kspace = LagrangeFESpace(self.mesh, p=2)
        kspace = self.kspace
        q = 5

        A = BilinearForm(kspace)
        self.k_BM_t = ScalarMassIntegrator(q=q)
        self.k_BC = ScalarConvectionIntegrator(q=q)
        self.k_BD = ScalarDiffusionIntegrator(q=q)
        self.k_BM = ScalarMassIntegrator(q=q)

        A.add_integrator(self.k_BM_t)
        A.add_integrator(self.k_BC)
        A.add_integrator(self.k_BD)
        A.add_integrator(self.k_BM)

        return A
    
    def LForm(self):
        from fealpy.fem import (LinearForm, ScalarSourceIntegrator)
        kspace = self.kspace
        q = 5

        L = LinearForm(kspace)
        self.k_LSI = ScalarSourceIntegrator(q=q)
        self.k_LP = ScalarSourceIntegrator(q=q)
        L.add_integrator(self.k_LSI)
        L.add_integrator(self.k_LP)
        return L
    
    def update(self, kk, u1, k0, omega0, mu_t):
        from fealpy.decorator import barycentric
        equation = self.equation
        dt = self.dt
        ctd = equation.coef_time_derivative
        cc = equation.coef_convection
        cd = equation.coef_diffusion
        cr = equation.coef_reaction
        cp = equation.coef_production
        
        ## BilinearForm
        self.k_BM_t.coef = ctd/dt
        @barycentric
        def k_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.k_BC.coef = k_BC_coef

        @barycentric
        def k_BD_coef(bcs, index):
            cdcoef = cd(bcs, index)[..., bm.newaxis] if callable(cd) else cd
            cdcoef += equation.pde.sigma_k * mu_t
            return cdcoef
        self.k_BD.coef = k_BD_coef

        @barycentric
        def k_BM_coef(bcs, index):
            crcoef = cr(bcs, index)[..., bm.newaxis] if callable(cr) else cr
            return crcoef * omega0(bcs, index)
        self.k_BM.coef = k_BM_coef

        ## LinearForm
        @barycentric
        def k_LP_coef(bcs, index):
            result = equation.pde.production_k(u0 = u1, 
                                               k0 = k0, 
                                               omega0 = omega0, 
                                               mu_t = mu_t, 
                                               bcs = bcs, 
                                               index = index)
            return result
        self.k_LP.source = k_LP_coef

        @barycentric
        def k_LSI_coef(bcs, index):
            ctdcoef = ctd(bcs, index)[..., bm.newaxis] if callable(ctd) else ctd
            result = ctdcoef * kk(bcs, index) / dt
            return result
        self.k_LSI.source = k_LSI_coef

class FEMSpecificDissipationRate(IterativeMethod):
    def BForm(self):
        from fealpy.fem import (BilinearForm, ScalarConvectionIntegrator, 
                                ScalarMassIntegrator, ScalarDiffusionIntegrator)
        from fealpy.functionspace import LagrangeFESpace

        self.omega0space = LagrangeFESpace(self.mesh, p=2)
        omega0space = self.omega0space
        q = 5

        A = BilinearForm(omega0space)
        self.omega_BM_t = ScalarMassIntegrator(q=q)
        self.omega_BC = ScalarConvectionIntegrator(q=q)
        self.omega_BD = ScalarDiffusionIntegrator(q=q)
        self.omega_BM = ScalarMassIntegrator(q=q)
        self.omega_BCD = ScalarConvectionIntegrator(q=q)

        A.add_integrator(self.omega_BM_t)
        A.add_integrator(self.omega_BC)
        A.add_integrator(self.omega_BD)
        A.add_integrator(self.omega_BM)
        A.add_integrator(self.omega_BCD)

        return A

    def LForm(self):
        from fealpy.fem import (LinearForm, ScalarSourceIntegrator)
        
        omega0space = self.omega0space
        q = 5

        L = LinearForm(omega0space)
        self.omega_LSI = ScalarSourceIntegrator(q=q)
        self.omega_LP = ScalarSourceIntegrator(q=q)
        L.add_integrator(self.omega_LSI)
        L.add_integrator(self.omega_LP)
        return L
    
    def update(self, omegak, u1, k1, omega0, mu_t):
        from fealpy.decorator import barycentric
        equation = self.equation
        dt = self.dt 
        ctd = equation.coef_time_derivative
        cc = equation.coef_convection
        cds = equation.coef_dissipation
        cd = equation.coef_diffusion
        ccd = equation.coef_cross_diffusion
        cp = equation.coef_production

        ## BilinearForm
        self.u_BM_t.coef = ctd/dt
        @barycentric
        def omega_BC_coef(bcs, index):
            cccoef = cc(bcs, index)[..., bm.newaxis] if callable(cc) else cc
            return cccoef * u1(bcs, index)
        self.omega_BC.coef = omega_BC_coef

        @barycentric
        def omega_BM_coef(bcs, index):
            cdscoef = cds(bcs, index)[..., bm.newaxis] if callable(cds) else cds
            return cdscoef * omega0(bcs, index)
        self.omega_BM.coef = omega_BM_coef

        @barycentric
        def omega_BD_coef(bcs, index):
            cdcoef = cd(bcs, index)[..., bm.newaxis] if callable(cd) else cd
            cdcoef = equation.pde.sigma_omega * mu_t
            return cdcoef
        self.omega_BD.coef = omega_BD_coef

        @barycentric
        def omega_BCD_coef(bcs, index):
            ccdcoef = ccd(bcs, index)[bm.newaxis, bm.newaxis] if callable(ccd) else ccd
            points = self.omegaspace.mesh.bc_to_point(bcs, index)
            F1 = equation.pde.cross_diffuison_f1(k1=k1, 
                                                 omega0=omega0, 
                                                 points=points, 
                                                 bcs=bcs, 
                                                 index=index)
            ccdcoef *= (1 - F1)
            reciprocal_omega0 = 1/omega0
            ccdcoef *= reciprocal_omega0(bcs, index)
            ccdcoef = ccdcoef[..., None] * k1.grad_value(bcs, index)
            return ccdcoef
        self.omega_BCD.coef = omega_BCD_coef

        ## LinearForm
        @barycentric
        def omega_LP_coef(bcs, index):
            result = cp(bcs, index)[bm.newaxis, bm.newaxis] if callable(cp) else cp
            result /= mu_t
            result *= equation.pde.production_omega(u0 = u1, 
                                               k0 = k1, 
                                               mu_t = mu_t, 
                                               bcs = bcs, 
                                               index = index)
            return result
        self.omega_LP.source = omega_LP_coef

        @barycentric
        def omega_LSI_coef(bcs, index):
            ctdcoef = ctd(bcs, index)[bm.newaxis, bm.newaxis] if callable(ctd) else ctd
            result = ctdcoef * omegak(bcs, index) / dt  
            return result
        self.omega_LSI.source = omega_LSI_coef



