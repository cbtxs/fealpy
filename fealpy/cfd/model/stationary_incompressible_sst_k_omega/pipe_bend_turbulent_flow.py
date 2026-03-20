from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.typing import Index, _S

class PipeBendTurbulentFlow():
    def __init__(self):
        self.rho = 1.0
        self.mu = 2.3256e-5
        self.beta_s = 0.09
        self.a1 = 0.31
    
    def inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 1.0
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 0.0
        u[..., 2] = 1.224*(1.0 - bm.sqrt(x**2 + y**2)/R)**(1/7)
        return u

    def outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pressure = bm.zeros(x.shape)
        return pressure
    
    def wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        u = bm.zeros(p.shape)
        return u

    def is_inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(z + 10) < atol)
        return on_boundary
    
    def is_outlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        atol = 1e-12
        on_boundary = (bm.abs(x - 17.8) < atol)
        return on_boundary
    
    def is_wall_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        r = 0.5
        d = self.distance_t0_centerline(p)
        atol = 1e-12
        on_boundary = (bm.abs(d - r) < atol)
        return on_boundary


    def distance_t0_centerline(self, p: TensorLike) -> TensorLike:
        """计算点到中心轴的距离"""
        R = 2.8
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        
        # 情况1: 上游直管
        d_up = bm.sqrt(x**2 + y**2)
        
        # 情况2: 下游直管
        d_down = bm.sqrt(y**2 + (z - R)**2)
        
        # 情况3: 弯管段
        dist_to_center_xz = bm.sqrt((x - R)**2 + z**2)
        dist_to_arc_xz = bm.abs(dist_to_center_xz - R)
        d_bend = bm.sqrt(dist_to_arc_xz**2 + y**2)
        
        # 根据x坐标选择合适的距离
        d_tail = bm.where(x >= R, d_down, d_bend)

        # 根据z坐标选择合适的距离
        d = bm.where(z <= 0, d_up, d_tail)
        return d
    
    def tur_mu(self, u0, k0, omega0, points, bcs, index: Index = _S):
        beta_s = self.beta_s
        mu = self.mu
        rho = self.rho
        a1 = self.a1
        d = self.distance_t0_centerline(points)
        print("d", d.shape)
        def shear_stress_limit_function():
            arg2 = bm.maximum(2 * bm.sqrt(k0(bcs, index))/(beta_s * omega0(bcs, index) * d),
                            500 * mu/(d**2 * rho * omega0(bcs, index)))
            F2 = bm.tanh(arg2**2)
            return F2
        F2 = shear_stress_limit_function()
        print("F2", F2.shape)

        def strain_rate(bcs, index):
            c2d = k0.space.cell_to_dof()
            flat_ids = c2d.reshape(-1)
            GD = u0.space.mesh.GD
            grad_u = u0.grad_value(bcs, index)
            grad_u_T = bm.swapaxes(grad_u, -1, -2)

            # grad_u = grad_u.reshape((-1, GD, GD))
            # g_u = bm.zeros((len(points), GD, GD))
            # g_u[flat_ids] = grad_u
            # grad_u_T = grad_u_T.reshape((-1, GD, GD))
            # g_u_T = bm.zeros((len(points), GD, GD))
            # g_u_T[flat_ids] = grad_u_T

            # print("g_u", g_u.shape)
            # print("g_u_T", g_u_T.shape)
            S_ij = 1/2 * (grad_u + grad_u_T)
            print("S_ij", S_ij.shape)
            S = bm.sqrt(2 * bm.sum(S_ij * S_ij, axis=(2, 3)))
            print("S", S.shape)
            return S
        S = strain_rate(bcs, index)
        print("S", S)
        mu_t = a1 * k0(bcs, index)
        mu_t /= bm.maximum(a1 * omega0(bcs, index), S * F2)
        return mu_t
    
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
    
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        if p == None:
            return 1
        return self.is_outlet_boundary(p)
    
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.inlet_boundary(p) | self.wall_boundary(p)
    
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_boundary(p)
    




        
    

