import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import gmsh

class PipeGeometry:
    def __init__(self):
        # 几何数据参数化
        self.R = 0.5        # 管道半径 0.5m
        self.D = 1.0        # 管道直径 1.0m
        self.Rc = 2.8       # 曲率半径 2.8 * D = 2.8m
        self.L_up = 10.0    # 上游直管段长度 10m
        self.L_down = 15.0  # 下游直管段长度 15m
        
        self.volume_tag = None

    def build(self):
        """原生构建 Z 轴入口的 90 度弯管几何模型 (中心线扫掠法)"""
        gmsh.initialize()
        gmsh.model.add("Benchmark_90_Degree_Bend_Native_Z")

        # ==========================================
        # 1. 构建中心线轨迹 (Wire)
        # ==========================================
        # 入口起点：位于 Z 轴负方向，向原点流动
        p1 = gmsh.model.occ.addPoint(0, 0, -self.L_up)
        # 弯管起点：原点
        p2 = gmsh.model.occ.addPoint(0, 0, 0)
        # 弯管圆心：向 X 轴正向偏移 Rc
        p_center = gmsh.model.occ.addPoint(self.Rc, 0, 0)
        # 弯管终点：转 90 度后切向变为 X 轴
        p3 = gmsh.model.occ.addPoint(self.Rc, 0, self.Rc)
        # 出口终点：沿 X 轴正向延伸 L_down
        p4 = gmsh.model.occ.addPoint(self.Rc + self.L_down, 0, self.Rc)

        # 连成线段与圆弧
        l1 = gmsh.model.occ.addLine(p1, p2)                     # 入口直段 (沿 Z 轴)
        arc = gmsh.model.occ.addCircleArc(p2, p_center, p3)     # 90 度弯管弧
        l2 = gmsh.model.occ.addLine(p3, p4)                     # 出口直段 (沿 X 轴)

        # 将线段和圆弧组合成一条平滑的迹线 (Wire)
        wire = gmsh.model.occ.addWire([l1, arc, l2])

        # ==========================================
        # 2. 构建截面并扫掠成体 (Pipe)
        # ==========================================
        # 在入口端 (0, 0, -L_up) 创建一个圆面。
        disk = gmsh.model.occ.addDisk(0, 0, -self.L_up, self.R, self.R)

        # 沿中心迹线扫掠生成 3D 管道实体
        pipe = gmsh.model.occ.addPipe([(2, disk)], wire)
        
        gmsh.model.occ.synchronize()
        
        # 提取生成的 3D 体标签
        self.volume_tag = pipe[0][1]
        
        # 进行边界判断和命名
        self.classify_boundaries()

    def classify_boundaries(self):
        """判断并划分 入口、出口 和 壁面边界"""
        surfaces = gmsh.model.getBoundary([(3, self.volume_tag)], oriented=False)
        
        inlet_tags = []
        outlet_tags = []
        wall_tags = []

        for dim, tag in surfaces:
            # 获取每个面的质心坐标
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            
            # 判断逻辑：入口面质心的 Z 坐标位于 -L_up 处
            if abs(com[2] - (-self.L_up)) < 1e-3:
                inlet_tags.append(tag)
            # 出口面质心的 X 坐标位于 Rc + L_down 处
            elif abs(com[0] - (self.Rc + self.L_down)) < 1e-3:
                outlet_tags.append(tag)
            # 剩余的柱面/环面均判定为管壁
            else:
                wall_tags.append(tag)

        # 添加 Physical Groups 供 CFD 读取
        gmsh.model.addPhysicalGroup(2, inlet_tags, name="Inlet")
        gmsh.model.addPhysicalGroup(2, outlet_tags, name="Outlet")
        gmsh.model.addPhysicalGroup(2, wall_tags, name="Wall")
        
        # 添加流体域
        gmsh.model.addPhysicalGroup(3, [self.volume_tag], name="FluidDomain")
        print("几何构建（原生Z轴扫掠）与边界划分完成！")


class PipeMesh:
    def __init__(self, geometry: PipeGeometry, mesh_size=1.0):
        self.geom = geometry
        self.mesh_size = mesh_size
        self.mesh = None  # Initialize mesh attribute

    def generate_mesh(self):
        """基于传入的几何对象生成四面体网格"""
        if self.geom.volume_tag is None:
            raise ValueError("几何未构建！请先调用 geometry.build()")

        print(f"开始生成 3D 四面体网格，全局最大尺寸设定为: {self.mesh_size} ...")
        # 设置全局网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size / 5.0)
        
        # 优化网格质量（针对 3D 流体网格） 
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # 采用 HXT 算法
        
        # 生成 3D 网格
        gmsh.model.mesh.generate(3)
        print("网格生成完毕！")
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        
        # 将一维坐标数组 reshape 成 (N, 3) 的矩阵
        nodes = np.array(nodeCoords).reshape(-1, 3)
        
        # ★ 关键步骤：建立 nodeTag 到 numpy 数组索引 (0, 1, 2...) 的映射
        tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}
        
        # 2. 获取所有的 3D 单元 (对于本模型，就是四面体单元)
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
        
        # 我们提取 3D 单元的节点组成信息 (由于只有四面体，取索引 [0])
        cells_tags = np.array(elemNodeTags[0]).reshape(-1, 4)
        
        # 使用向量化操作，将包含 Gmsh Tag 的 cells 转换为包含 Numpy 0-based 索引的 cells
        mapper = np.vectorize(lambda tag: tag_to_index[tag])
        cells = mapper(cells_tags)
        
        from fealpy.mesh import TetrahedronMesh
        self.mesh = TetrahedronMesh(nodes, cells)  # Assign mesh to the object
        
        return self.mesh

    def plot_mesh(self):
        """使用 Matplotlib 可视化网格，并区分流体域和固体域"""
        if self.mesh is None:
            raise ValueError("网格未生成！请先调用 generate_mesh()")
        
        # 获取节点和单元数据
        nodes = np.array(self.mesh.node)
        cells = np.array(self.mesh.cell)
        
        # 计算流体域（管道内腔）和固体域（管道壁面）
        fluid_nodes = nodes[nodes[:, 0] < self.geom.Rc + self.geom.L_down]  # 假设管道内的流体区域
        solid_nodes = nodes[nodes[:, 0] >= self.geom.Rc + self.geom.L_down]  # 假设管道壁面是固体域
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制流体域的节点（假设流体节点的 X 坐标小于 Rc + L_down）
        ax.scatter(fluid_nodes[:, 0], fluid_nodes[:, 1], fluid_nodes[:, 2], c='b', marker='o', label="Fluid Domain")
        
        # 绘制固体域的节点（假设固体节点的 X 坐标大于等于 Rc + L_down）
        ax.scatter(solid_nodes[:, 0], solid_nodes[:, 1], solid_nodes[:, 2], c='r', marker='^', label="Solid Domain")
        
        # 绘制单元边界（四面体）
        for cell in cells:
            x = nodes[cell, 0]
            y = nodes[cell, 1]
            z = nodes[cell, 2]
            # 使用多边形连接四面体的节点
            ax.plot_trisurf(x, y, z, color='g', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Pipe Mesh with Fluid and Solid Domains')
        
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # 1. 实例化几何类并构建
    geom = PipeGeometry()
    geom.build()

    # 2. 实例化网格类（传入几何对象），设定网格尺寸
    mesher = PipeMesh(geom, mesh_size=0.8)  # 初始网格密度设为 0.8

    # 3. 生成网格
    mesh = mesher.generate_mesh()
    
    # 4. 可视化检查边界与网格
    mesher.plot_mesh()