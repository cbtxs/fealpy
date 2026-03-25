import gmsh
import math


class PipeGeometry:
    def __init__(self):
        self.R = 0.5
        self.D = 1.0
        self.Rc = 2.8
        self.L_up = 10.0
        self.L_down = 15.0
        self.wall_thickness = 0.005
        self.volume_tags = []

    def build(self):
        gmsh.initialize()
        gmsh.model.add("BendPipe_WallSolid")

        inner_r = self.R - self.wall_thickness
        if inner_r <= 0:
            raise ValueError("wall_thickness 太大，导致内半径 <= 0")

        # 1) 上游直管（沿 z）
        up_outer = gmsh.model.occ.addCylinder(0, 0, -self.L_up, 0, 0, self.L_up, self.R)
        up_inner = gmsh.model.occ.addCylinder(0, 0, -self.L_up, 0, 0, self.L_up, inner_r)
        up_wall, _ = gmsh.model.occ.cut([(3, up_outer)], [(3, up_inner)], removeObject=True, removeTool=True)

        # 2) 弯管段：将环形面绕 y 轴旋转 90 度生成体
        c_out = gmsh.model.occ.addCircle(0, 0, 0, self.R)
        c_in = gmsh.model.occ.addCircle(0, 0, 0, inner_r)
        loop_out = gmsh.model.occ.addCurveLoop([c_out])
        loop_in = gmsh.model.occ.addCurveLoop([c_in])
        annulus = gmsh.model.occ.addPlaneSurface([loop_out, loop_in])
        bend_out = gmsh.model.occ.revolve([(2, annulus)], self.Rc, 0, 0, 0, 1, 0, math.pi / 2.0)
        bend_wall = [(d, t) for d, t in bend_out if d == 3]

        # 3) 下游直管（沿 x）
        down_outer = gmsh.model.occ.addCylinder(self.Rc, 0, self.Rc, self.L_down, 0, 0, self.R)
        down_inner = gmsh.model.occ.addCylinder(self.Rc, 0, self.Rc, self.L_down, 0, 0, inner_r)
        down_wall, _ = gmsh.model.occ.cut([(3, down_outer)], [(3, down_inner)], removeObject=True, removeTool=True)

        up_wall = [(d, t) for d, t in up_wall if d == 3]
        down_wall = [(d, t) for d, t in down_wall if d == 3]
        if not up_wall or not bend_wall or not down_wall:
            raise RuntimeError("分段建模失败，未得到完整管壁实体。")

        # 4) 做 fragment 以保证三段在交界处共形，不强制 fuse 成单体
        frag1, _ = gmsh.model.occ.fragment(up_wall, bend_wall, removeObject=True, removeTool=True)
        frag1_vols = [(d, t) for d, t in frag1 if d == 3]
        frag2, _ = gmsh.model.occ.fragment(frag1_vols, down_wall, removeObject=True, removeTool=True)
        wall_vols = [(d, t) for d, t in frag2 if d == 3]
        if not wall_vols:
            raise RuntimeError("三段拼接失败，未得到管壁实体。")
        self.volume_tags = [t for d, t in wall_vols]

        gmsh.model.occ.synchronize()
        self.classify_boundaries()

    def classify_boundaries(self):
        surfaces = gmsh.model.getBoundary([(3, tag) for tag in self.volume_tags], oriented=False)

        inlet_tags, outlet_tags, wall_tags = [], [], []
        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if abs(com[2] + self.L_up) < 1e-3:
                inlet_tags.append(tag)
            elif abs(com[0] - (self.Rc + self.L_down)) < 1e-3:
                outlet_tags.append(tag)
            else:
                wall_tags.append(tag)

        if inlet_tags:
            gmsh.model.addPhysicalGroup(2, inlet_tags, name="Inlet")
        if outlet_tags:
            gmsh.model.addPhysicalGroup(2, outlet_tags, name="Outlet")
        if wall_tags:
            gmsh.model.addPhysicalGroup(2, wall_tags, name="Wall")

        gmsh.model.addPhysicalGroup(3, self.volume_tags, name="WallDomain")
        print("管壁实体已生成")

if __name__ == "__main__":
    geom = PipeGeometry()
    geom.build()

    gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)

    # 关键：生成3D网格，GUI里就不会只像一根线
    gmsh.model.mesh.generate(3)

    # 可选隐藏点线
    gmsh.option.setNumber("Geometry.Points", 0)
    gmsh.option.setNumber("Geometry.Curves", 0)

    gmsh.fltk.run()
    gmsh.finalize()
    