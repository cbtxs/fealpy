from fealpy.backend import TensorLike
from fealpy.decorator import cartesian
from fealpy.backend import backend_manager as bm
from fealpy.typing import Index, _S

class PipeBendTurbulentFlow():
    def __init__(self):
        self.rho = 1.0
        self.mu = 2.3256e-5
        self.beta_s = 0.09
        self.beta = 0.079
        self.a1 = 0.31
        self.sigma_k = 1.0
        self.sigma_omega = 0.5
        self.sigma_omega2 = 0.856
        self.gamma = 0.5
    
    def inlet_boundary(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_centerline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 0.0
        u[..., 1] = 0.0
        u[..., 2] = 1.224*(1.0 - d/R)**(1/7)
        return u
    
    def outlet_velocity(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        R = 0.5
        d = self.distance_t0_centerline(p)
        u = bm.zeros(p.shape)
        u[..., 0] = 1.224*(1.0 - d/R)**(1/7)
        u[..., 1] = 0.0
        u[..., 2] = 0.0
        return u

    def outlet_pressure(self, p: TensorLike) -> TensorLike:
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
        d_up = bm.minimum(bm.sqrt(x**2 + y**2), 0.5)
        
        # 情况2: 下游直管
        d_down = bm.minimum(bm.sqrt(y**2 + (z - R)**2), 0.5)
        
        # 情况3: 弯管段
        dist_to_center_xz = bm.sqrt((x - R)**2 + z**2)
        dist_to_arc_xz = bm.abs(dist_to_center_xz - R)
        d_bend = bm.minimum(bm.sqrt(dist_to_arc_xz**2 + y**2), 0.5)
        
        # 根据x坐标选择合适的距离
        d_tail = bm.where(x >= R, d_down, d_bend)

        # 根据z坐标选择合适的距离
        d = bm.where(z <= 0, d_up, d_tail)

        return d
    
    def strain_rate(self, u0, bcs, index):
            grad_u = u0.grad_value(bcs, index)
            grad_u_T = bm.swapaxes(grad_u, -1, -2)

            S_ij = 1/2 * (grad_u + grad_u_T)
            return S_ij
    
    def tur_mu(self, u0, k0, omega0, points, bcs, index: Index = _S):
        beta_s = self.beta_s
        mu = self.mu
        rho = self.rho
        a1 = self.a1
        d = self.distance_t0_centerline(points)

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
    
    def is_velocity_boundary(self, p: TensorLike) -> TensorLike:
        # return self.is_inlet_boundary(p) | self.is_wall_boundary(p)
        return None
    
    def is_pressure_boundary(self, p: TensorLike = None) -> TensorLike:
        # if p is None:
        #     return 1
        # return self.is_outlet_boundary(p)
        return 0
    
    def velocity_dirichlet(self, p: TensorLike) -> TensorLike:
        result = bm.zeros(p.shape)
        inlet = self.inlet_boundary(p)
        outlet = self.outlet_velocity(p)
        wall = self.wall_boundary(p)
        is_inlet = self.is_inlet_boundary(p)
        is_wall = self.is_wall_boundary(p)
        is_outlet = self.is_outlet_boundary(p)

        result[is_inlet] = inlet[is_inlet]
        result[is_wall] = wall[is_wall]
        result[is_outlet] = outlet[is_outlet]
        return result
    
    def pressure_dirichlet(self, p: TensorLike) -> TensorLike:
        return self.outlet_pressure(p)
    
    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        result = bm.zeros(p.shape)
        return result
    
    @cartesian
    def production_k(self, u0, k0, omega0, mu_t, bcs, index) -> TensorLike:
        result_0 = self.production_omega(u0, k0, mu_t, bcs, index)
        result_1 = 10 * self.beta_s * self.rho * k0(bcs, index) * omega0(bcs, index)
        result = bm.minimum(result_0, result_1)
        return result
    
    @cartesian
    def production_omega(self, u0, k0, mu_t, bcs, index) -> TensorLike:
        S_ij = self.strain_rate(u0, bcs, index)
        grad_u = u0.grad_value(bcs, index)
        
        P = mu_t * bm.sum(S_ij * grad_u, axis=(2, 3))
        P -= 2/3 * k0(bcs, index) * bm.einsum("...ii -> ...", grad_u)
        return P
    
    @cartesian
    def cross_diffuison_f1(self, k1, omega0, points, bcs, index):
        d = self.distance_t0_centerline(p=points)
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

    

