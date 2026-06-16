import gmsh
import numpy as np

from math import tan, radians


class WingMeshConfig:
    """
    UAV 机翼壳网格参数配置。
    单位默认：mm。
    不使用 dataclass，不使用 typing。
    """

    def __init__(
        self,
        model_name="uav_wing_naca2412_shell",

        # 几何参数
        span=1200.0,
        sweep_deg=12.4,
        sweep_reference="leading_edge",

        # 弦长分布：(y, chord)
        chord_stations=None,

        # 翼肋位置
        rib_y=None,

        # 前后梁位置，x/c
        spar_xc=None,

        # NACA 四位翼型参数，默认 NACA 2412
        airfoil_m=0.02,
        airfoil_p=0.40,
        airfoil_t=0.12,
        n_profile=81,

        # 壳厚度参数
        shell_thickness=None,

        # 网格参数
        mesh_size=10.0,
        recombine=True,

        # 输出文件前缀
        output_basename="uav_wing_shell",
    ):
        self.model_name = model_name

        self.span = span
        self.sweep_deg = sweep_deg
        self.sweep_reference = sweep_reference

        if chord_stations is None:
            chord_stations = (
                (0.0, 200.0),
                (400.0, 150.0),
                (1200.0, 50.0),
            )
        self.chord_stations = chord_stations

        if rib_y is None:
            rib_y = (
                0.0,
                200.0,
                400.0,
                600.0,
                800.0,
                975.0,
                1200.0,
            )
        self.rib_y = rib_y

        if spar_xc is None:
            spar_xc = {
                "front_spar": 0.20,
                "rear_spar": 0.75,
            }
        self.spar_xc = spar_xc

        self.airfoil_m = airfoil_m
        self.airfoil_p = airfoil_p
        self.airfoil_t = airfoil_t
        self.n_profile = n_profile

        if shell_thickness is None:
            shell_thickness = {
                "skin": 2.0,
                "root_rib": 15.0,
                "ribs": 5.0,
                "front_spar": 5.0,
                "rear_spar": 3.0,
            }
        self.shell_thickness = shell_thickness

        self.mesh_size = mesh_size
        self.recombine = recombine
        self.output_basename = output_basename


class UAVWingShellMesher:
    """
    NACA 后掠锥形 UAV 机翼壳网格生成器。

    生成对象：
        skin       : 蒙皮壳面
        root_rib   : 根部翼肋壳面
        ribs       : 其余翼肋壳面
        front_spar : 前梁壳面
        rear_spar  : 后梁壳面

    注意：
        shell_thickness 只作为后续 FEALPy 赋厚度属性使用，
        不会改变 Gmsh 中的几何厚度。
    """

    def __init__(self, config=None):
        if config is None:
            config = WingMeshConfig()

        self.cfg = config

        self.occ = None

        self.x_af = None
        self.z_af = None
        self.x_base = None
        self.n_upper = None

        self.point_tags = []
        self.line_cache = {}

        self.skin_surfaces = []
        self.rib_surfaces = []
        self.spar_surface_groups = {}
        self.spar_lines_to_embed = {}
        self.physical_groups = {}

        self.geometry_built = False
        self.mesh_generated = False

    # ============================================================
    # 1. 配置工具
    # ============================================================

    @staticmethod
    def from_naca4(code, **kwargs):
        """
        通过 NACA 四位数创建配置。

        示例：
            cfg = UAVWingShellMesher.from_naca4("2412")
            cfg = UAVWingShellMesher.from_naca4("2415", mesh_size=8.0)
        """
        if len(code) != 4 or not code.isdigit():
            raise ValueError("NACA code must be a 4-digit string, e.g. '2412'.")

        m = int(code[0]) / 100.0
        p = int(code[1]) / 10.0
        t = int(code[2:]) / 100.0

        cfg = WingMeshConfig(
            airfoil_m=m,
            airfoil_p=p,
            airfoil_t=t,
            **kwargs
        )

        return cfg

    def _reset_model_data(self):
        self.occ = None

        self.x_af = None
        self.z_af = None
        self.x_base = None
        self.n_upper = None

        self.point_tags = []
        self.line_cache = {}

        self.skin_surfaces = []
        self.rib_surfaces = []
        self.spar_surface_groups = {}
        self.spar_lines_to_embed = {}
        self.physical_groups = {}

        self.geometry_built = False
        self.mesh_generated = False

    def _validate_config(self):
        if self.cfg.mesh_size <= 0:
            raise ValueError("mesh_size must be positive.")

        if self.cfg.n_profile < 11:
            raise ValueError("n_profile is too small. Use at least 11.")

        if abs(self.cfg.rib_y[0]) > 1.0e-9:
            raise ValueError("rib_y 的第一个位置应为 0.0，即根部翼肋。")

        if abs(self.cfg.rib_y[-1] - self.cfg.span) > 1.0e-9:
            raise ValueError("rib_y 的最后一个位置应等于 span，即翼尖翼肋。")

        for name in self.cfg.spar_xc:
            xc = self.cfg.spar_xc[name]
            if not (0.0 < xc < 1.0):
                raise ValueError(name + " 的弦向位置 xc 必须在 0 和 1 之间。")

        allowed = {
            "leading_edge",
            "quarter_chord",
            "trailing_edge",
        }

        if self.cfg.sweep_reference not in allowed:
            raise ValueError(
                "sweep_reference must be one of: "
                "'leading_edge', 'quarter_chord', 'trailing_edge'."
            )

    # ============================================================
    # 2. 翼型和机翼几何函数
    # ============================================================

    def chord_length(self, y):
        """
        按 chord_stations 进行分段线性插值。
        """
        stations = sorted(self.cfg.chord_stations, key=lambda item: item[0])

        ys = np.array([item[0] for item in stations], dtype=float)
        cs = np.array([item[1] for item in stations], dtype=float)

        if y < ys[0] - 1.0e-9 or y > ys[-1] + 1.0e-9:
            raise ValueError("y={} is outside chord station range.".format(y))

        return float(np.interp(y, ys, cs))

    def leading_edge_x(self, y):
        """
        根据后掠参考线计算前缘 x 坐标。

        sweep_reference = leading_edge:
            前缘后掠角为 sweep_deg。

        sweep_reference = quarter_chord:
            1/4 弦线后掠角为 sweep_deg，根部前缘 x=0。

        sweep_reference = trailing_edge:
            后缘后掠角为 sweep_deg，根部前缘 x=0。
        """
        c_root = self.chord_length(0.0)
        c_y = self.chord_length(y)
        sweep = tan(radians(self.cfg.sweep_deg))

        if self.cfg.sweep_reference == "leading_edge":
            return y * sweep

        if self.cfg.sweep_reference == "quarter_chord":
            x_q_root = 0.25 * c_root
            x_q_y = x_q_root + y * sweep
            return x_q_y - 0.25 * c_y

        if self.cfg.sweep_reference == "trailing_edge":
            x_te_root = c_root
            x_te_y = x_te_root + y * sweep
            return x_te_y - c_y

        raise RuntimeError("Invalid sweep_reference.")

    def naca4_closed_profile(self):
        """
        生成 NACA 四位翼型闭合轮廓。

        返回：
            x_closed: 闭合翼型 x/c
            z_closed: 闭合翼型 z/c
            x_base  : 上下表面计算前的基础 x/c
        """
        m = self.cfg.airfoil_m
        p = self.cfg.airfoil_p
        t = self.cfg.airfoil_t

        beta = np.linspace(0.0, np.pi, self.cfg.n_profile)
        x = 0.5 * (1.0 - np.cos(beta))

        # 强制包含 spar 的 x/c 位置，使 spar 与 skin 更容易共节点
        extra_x = np.array(list(self.cfg.spar_xc.values()), dtype=float)
        x = np.unique(np.r_[x, extra_x])
        x.sort()

        yt = 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

        if abs(m) < 1.0e-14 or abs(p) < 1.0e-14:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.where(
                x < p,
                m / p**2 * (2.0 * p * x - x**2),
                m / (1.0 - p) ** 2
                * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
            )

            dyc_dx = np.where(
                x < p,
                2.0 * m / p**2 * (p - x),
                2.0 * m / (1.0 - p) ** 2 * (p - x),
            )

        theta = np.arctan(dyc_dx)

        xu = x - yt * np.sin(theta)
        zu = yc + yt * np.cos(theta)

        xl = x + yt * np.sin(theta)
        zl = yc - yt * np.cos(theta)

        # 闭合顺序：上表面 LE -> TE，下表面 TE -> LE
        x_closed = np.r_[xu, xl[-2:0:-1]]
        z_closed = np.r_[zu, zl[-2:0:-1]]

        return x_closed, z_closed, x

    def station_point(self, y, xbar, zbar):
        """
        将无量纲翼型点映射到三维机翼坐标。
        """
        c = self.chord_length(y)
        x = self.leading_edge_x(y) + xbar * c
        z = zbar * c

        return x, y, z

    def nearest_profile_index(self, x_target):
        """
        找到闭合翼型中最接近给定 x/c 的上、下表面点。
        用于生成 spar 腹板面。
        """
        x_closed = self.x_af

        upper_candidates = np.arange(0, self.n_upper)
        lower_candidates = np.arange(self.n_upper, len(x_closed))

        iu = upper_candidates[
            np.argmin(np.abs(x_closed[upper_candidates] - x_target))
        ]

        il = lower_candidates[
            np.argmin(np.abs(x_closed[lower_candidates] - x_target))
        ]

        return int(iu), int(il)

    # ============================================================
    # 3. Gmsh 基础工具
    # ============================================================

    def _init_gmsh(self):
        """
        初始化 Gmsh。
        """
        self._reset_model_data()

        try:
            if gmsh.isInitialized():
                gmsh.clear()
            else:
                gmsh.initialize()
        except AttributeError:
            gmsh.initialize()

        gmsh.model.add(self.cfg.model_name)
        self.occ = gmsh.model.occ

    def close(self):
        try:
            gmsh.finalize()
        except Exception:
            pass

    @staticmethod
    def safe_set_number(option_name, value):
        try:
            gmsh.option.setNumber(option_name, value)
        except Exception:
            pass

    def hide_all_labels(self):
        """
        关闭几何和网格上的编号。
        """
        label_options = [
            "Geometry.PointLabels",
            "Geometry.CurveLabels",
            "Geometry.SurfaceLabels",
            "Geometry.VolumeLabels",
            "Geometry.PointNumbers",
            "Mesh.NodeLabels",
            "Mesh.LineLabels",
            "Mesh.SurfaceLabels",
            "Mesh.VolumeLabels",
        ]

        for opt in label_options:
            self.safe_set_number(opt, 0)

    def show_geometry_only(self):
        """
        只显示几何，不显示网格。
        """
        self.hide_all_labels()

        self.safe_set_number("Geometry.Points", 0)
        self.safe_set_number("Geometry.Curves", 1)
        self.safe_set_number("Geometry.Surfaces", 1)

        self.safe_set_number("Mesh.Nodes", 0)
        self.safe_set_number("Mesh.Lines", 0)
        self.safe_set_number("Mesh.SurfaceEdges", 0)
        self.safe_set_number("Mesh.SurfaceFaces", 0)

    def show_mesh_only(self):
        """
        显示网格，不显示编号。
        """
        self.hide_all_labels()

        self.safe_set_number("Geometry.Points", 0)
        self.safe_set_number("Geometry.Curves", 0)
        self.safe_set_number("Geometry.Surfaces", 0)

        self.safe_set_number("Mesh.Nodes", 0)
        self.safe_set_number("Mesh.Lines", 0)
        self.safe_set_number("Mesh.SurfaceEdges", 1)
        self.safe_set_number("Mesh.SurfaceFaces", 1)

    def line_between(self, pa, pb):
        """
        创建或复用两点之间的线。
        返回值可能为负，表示方向相反。
        """
        key = (pa, pb)
        rkey = (pb, pa)

        if key in self.line_cache:
            return self.line_cache[key]

        if rkey in self.line_cache:
            return -self.line_cache[rkey]

        ltag = self.occ.addLine(pa, pb)
        self.line_cache[key] = ltag

        return ltag

    def make_wire(self, curves):
        return self.occ.addWire(curves, checkClosed=True)

    def make_filling_surface(self, curves):
        wire = self.make_wire(curves)
        return self.occ.addSurfaceFilling(wire)

    def make_plane_surface(self, curves):
        wire = self.make_wire(curves)
        return self.occ.addPlaneSurface([wire])

    # ============================================================
    # 4. 几何生成
    # ============================================================

    def build_geometry(self):
        """
        生成几何模型，但不划分网格。
        """
        self._validate_config()
        self._init_gmsh()

        self.x_af, self.z_af, self.x_base = self.naca4_closed_profile()
        self.n_upper = len(self.x_base)

        station_y = list(self.cfg.rib_y)

        n_station = len(station_y)
        n_profile = len(self.x_af)

        # 4.1 生成各翼肋剖面的翼型点
        self.point_tags = []

        for y in station_y:
            row = []

            for xb, zb in zip(self.x_af, self.z_af):
                x, yy, z = self.station_point(y, xb, zb)

                ptag = self.occ.addPoint(
                    x,
                    yy,
                    z,
                    self.cfg.mesh_size,
                )

                row.append(ptag)

            self.point_tags.append(row)

        # 4.2 生成蒙皮 skin
        self.skin_surfaces = []

        for i in range(n_station - 1):
            for j in range(n_profile):
                j2 = (j + 1) % n_profile

                p00 = self.point_tags[i][j]
                p01 = self.point_tags[i][j2]
                p11 = self.point_tags[i + 1][j2]
                p10 = self.point_tags[i + 1][j]

                curves = [
                    self.line_between(p00, p01),
                    self.line_between(p01, p11),
                    self.line_between(p11, p10),
                    self.line_between(p10, p00),
                ]

                s = self.make_filling_surface(curves)
                self.skin_surfaces.append(s)

        # 4.3 生成翼肋 ribs
        self.rib_surfaces = []
        self.spar_lines_to_embed = {}

        for i in range(n_station):
            curves = []

            for j in range(n_profile):
                j2 = (j + 1) % n_profile

                curves.append(
                    self.line_between(
                        self.point_tags[i][j],
                        self.point_tags[i][j2],
                    )
                )

            s = self.make_plane_surface(curves)

            self.rib_surfaces.append(s)
            self.spar_lines_to_embed[s] = []

        # 4.4 生成翼梁 spars
        self.spar_surface_groups = {}

        for spar_name in self.cfg.spar_xc:
            xc = self.cfg.spar_xc[spar_name]
            iu, il = self.nearest_profile_index(xc)

            spar_surfaces = []

            for i in range(n_station - 1):
                p_u0 = self.point_tags[i][iu]
                p_u1 = self.point_tags[i + 1][iu]
                p_l1 = self.point_tags[i + 1][il]
                p_l0 = self.point_tags[i][il]

                vertical_0 = self.line_between(p_u0, p_l0)
                vertical_1 = self.line_between(p_u1, p_l1)

                curves = [
                    self.line_between(p_u0, p_u1),
                    vertical_1,
                    self.line_between(p_l1, p_l0),
                    -vertical_0,
                ]

                s = self.make_filling_surface(curves)
                spar_surfaces.append(s)

                # 把 spar 与 rib 的交线嵌入 rib 面，提升连接一致性
                self.spar_lines_to_embed[self.rib_surfaces[i]].append(vertical_0)
                self.spar_lines_to_embed[self.rib_surfaces[i + 1]].append(vertical_1)

            self.spar_surface_groups[spar_name] = spar_surfaces

        # 4.5 同步 OCC 几何
        self.occ.synchronize()

        # 4.6 嵌入 spar/rib 交线
        for rib_surface in self.spar_lines_to_embed:
            lines = self.spar_lines_to_embed[rib_surface]
            lines_unique = list(set(abs(l) for l in lines))

            if lines_unique:
                gmsh.model.mesh.embed(1, lines_unique, 2, rib_surface)

        # 4.7 创建物理分组
        self.create_physical_groups()

        self.geometry_built = True

        return self

    def create_physical_groups(self):
        """
        创建 Physical Groups，方便 FEALPy 后续识别部件。
        """
        self.physical_groups = {}

        pg_skin = gmsh.model.addPhysicalGroup(2, self.skin_surfaces)
        gmsh.model.setPhysicalName(2, pg_skin, "skin")
        self.physical_groups["skin"] = pg_skin

        # 根部翼肋单独分组，避免 root_rib 和 ribs 重叠
        pg_root_rib = gmsh.model.addPhysicalGroup(2, [self.rib_surfaces[0]])
        gmsh.model.setPhysicalName(2, pg_root_rib, "root_rib")
        self.physical_groups["root_rib"] = pg_root_rib

        if len(self.rib_surfaces) > 1:
            pg_ribs = gmsh.model.addPhysicalGroup(2, self.rib_surfaces[1:])
            gmsh.model.setPhysicalName(2, pg_ribs, "ribs")
            self.physical_groups["ribs"] = pg_ribs

        for spar_name in self.spar_surface_groups:
            surfaces = self.spar_surface_groups[spar_name]

            pg = gmsh.model.addPhysicalGroup(2, surfaces)
            gmsh.model.setPhysicalName(2, pg, spar_name)

            self.physical_groups[spar_name] = pg

    # ============================================================
    # 5. 网格生成、输出和显示
    # ============================================================

    def all_surfaces(self):
        surfaces = []

        surfaces.extend(self.skin_surfaces)
        surfaces.extend(self.rib_surfaces)

        for name in self.spar_surface_groups:
            surfaces.extend(self.spar_surface_groups[name])

        return surfaces

    def set_mesh_options(self):
        """
        设置 Gmsh 网格参数。
        """
        all_surfaces = self.all_surfaces()

        if self.cfg.recombine:
            for s in all_surfaces:
                gmsh.model.mesh.setRecombine(2, s)

            gmsh.option.setNumber("Mesh.RecombineAll", 1)

        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.cfg.mesh_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.cfg.mesh_size)

    def generate_mesh(self):
        """
        生成二维壳网格。
        """
        if not self.geometry_built:
            raise RuntimeError("请先调用 build_geometry() 再划分网格。")

        self.set_mesh_options()

        gmsh.model.mesh.generate(2)

        self.mesh_generated = True

        return self

    def write_geometry(self):
        """
        输出 STEP 几何文件。
        """
        if not self.geometry_built:
            raise RuntimeError("几何还没有生成。")

        filename = self.cfg.output_basename + "_geometry.step"

        gmsh.write(filename)

        print("Geometry written:", filename)

        return filename

    def write_mesh(self):
        """
        输出 MSH 网格文件。
        """
        if not self.mesh_generated:
            raise RuntimeError("网格还没有生成。")

        filename = self.cfg.output_basename + ".msh"

        gmsh.write(filename)

        print("Mesh written:", filename)

        return filename

    def view_geometry(self):
        """
        打开 Gmsh 图形窗口，只查看几何。
        """
        if not self.geometry_built:
            raise RuntimeError("几何还没有生成。")

        self.show_geometry_only()
        gmsh.fltk.run()

    def view_mesh(self):
        """
        打开 Gmsh 图形窗口，只查看网格。
        """
        if not self.mesh_generated:
            raise RuntimeError("网格还没有生成。")

        self.show_mesh_only()
        gmsh.fltk.run()

    def thickness_table(self):
        """
        返回壳厚度参数表。
        后续 FEALPy 可以根据 Physical Group 名称赋厚度。
        """
        return dict(self.cfg.shell_thickness)

    def print_summary(self):
        """
        打印模型参数摘要。
        """
        print()
        print("========== Wing Mesh Summary ==========")
        print("model_name:", self.cfg.model_name)
        print("span:", self.cfg.span, "mm")
        print("sweep_deg:", self.cfg.sweep_deg)
        print("sweep_reference:", self.cfg.sweep_reference)
        print("chord_stations:", self.cfg.chord_stations)
        print("rib_y:", self.cfg.rib_y)
        print("spar_xc:", self.cfg.spar_xc)
        print("mesh_size:", self.cfg.mesh_size)
        print("recombine:", self.cfg.recombine)

        print()
        print("Shell thickness table:")
        for name in self.cfg.shell_thickness:
            print("  {}: {} mm".format(name, self.cfg.shell_thickness[name]))

        print("=======================================")
        print()

    def run(
        self,
        view_geometry=True,
        view_mesh=True,
        write_files=True,
        finalize=True,
    ):
        """
        一键流程：

            1. 生成几何
            2. 输出几何 STEP
            3. 显示几何
            4. 划分二维壳网格
            5. 输出 MSH
            6. 显示网格
        """
        try:
            self.build_geometry()

            if write_files:
                self.write_geometry()

            if view_geometry:
                self.view_geometry()

            self.generate_mesh()

            if write_files:
                self.write_mesh()

            if view_mesh:
                self.view_mesh()

            self.print_summary()

        finally:
            if finalize:
                self.close()

        return self


# ============================================================
# 6. 使用示例
# ============================================================

if __name__ == "__main__":

    # 默认参数：对应论文中的 NACA 2412 UAV 机翼壳网格模型
    cfg = WingMeshConfig(
        mesh_size=30.0,
        output_basename="uav_wing_shell",
    )
    
    # cfg.shell_thickness["skin"] = 1.0
    # cfg.spar_xc["front_spar"] = 0.25
    # cfg.spar_xc["rear_spar"] = 0.70

    mesher = UAVWingShellMesher(cfg)
    mesher.run(
        view_geometry=True,
        view_mesh=True,
        write_files=True,
        finalize=True,
    )