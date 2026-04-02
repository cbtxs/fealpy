from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["ElbowPipeMesh", "TeePipeMesh"]

class ElbowPipeMesh(CNodeType):
    r"""Create a mesh for an elbow pipe in 3D.

    Inputs:
    D (float): The inner diameter of the elbow pipe.
    bend_angle (float): The bend angle in degrees.
    R_bend_inner (float): The inner radius ratio of the bend.
    L_in_ratio (float): The inlet straight pipe length ratio.
    L_out_ratio (float): The outlet straight pipe length ratio.
    wall_thickness (float): The wall thickness of the pipe.
    mesh_size_global (float): The global background mesh size.
    mesh_size_bend (float): The local mesh size in the bend region.
    mesh_size_interface (float): The local mesh size near the fluid-structure interaction interface.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "单弯管网格建模"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS = [
        PortConf("D", DataType.FLOAT, 0, title="弯管内径", default= 25.0),
        PortConf("bend_angle", DataType.FLOAT, 1, title="弯角（度）", default=90.0),
        PortConf("R_bend_inner", DataType.FLOAT, 1, title="内弯半径比例", default=1.5),
        PortConf("L_in_ratio", DataType.FLOAT, 1, title="入口直管长度比例", default=5.0),
        PortConf("L_out_ratio", DataType.FLOAT, 1, title="出口直管长度相比例", default=10.0),
        PortConf("wall_thickness",DataType.FLOAT, 1, title="壁厚", default=5.0),
        PortConf("mesh_size_global",DataType.FLOAT, 1, title="全局背景网格尺寸", default=7.5),
        PortConf("mesh_size_bend",DataType.FLOAT, 1, title="弯头区域局部网格尺寸", default=5),
        PortConf("mesh_size_interface",DataType.FLOAT, 1, title="流固界面附近局部网格尺寸", default=3.75)
    ]
    
    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ] 
    
    @staticmethod
    def run(**options):
        from fealpy.mesher import ElbowPipeMesher
        params = {
            "D": options.get("D"),
            "bend_angle": options.get("bend_angle"),
            "R_bend_inner": options.get("R_bend_inner"),
            "L_in_ratio": options.get("L_in_ratio"),
            "L_out_ratio": options.get("L_out_ratio"),
            "wall_thickness": options.get("wall_thickness"),
            "mesh_size_global": options.get("mesh_size_global"),
            "mesh_size_bend": options.get("mesh_size_bend"),
            "mesh_size_interface": options.get("mesh_size_interface"),
        }

        remesher = ElbowPipeMesher(params)
        mesh = remesher.init_mesh()
        
        return mesh


class TeePipeMesh(CNodeType):
    r"""Create a mesh for a tee pipe in 3D.

    Inputs:
    D_main (float): The inner diameter of the main pipe.
    D_branch (float): The inner diameter of branch pipes.
    intersect_angle (float): The intersection angle in degrees.
    R_fillet (float): The fillet radius at the junction.
    L_in_ratio (float): The inlet straight pipe length ratio.
    L_out_ratio (float): The outlet straight pipe length ratio.
    wall_thickness (float): The wall thickness of the pipe.
    mesh_size_global (float): The global background mesh size.
    mesh_size_junction (float): The local mesh size in junction region.
    mesh_size_interface (float): The local mesh size near FSI interface.

    Outputs:
        mesh (MeshType): The mesh object created.
    """
    TITLE: str = "三通管网格建模"
    PATH: str = "preprocess.mesher"
    INPUT_SLOTS = [
        PortConf("D_main", DataType.FLOAT, 0, title="主管内径", default=32.0),
        PortConf("D_branch", DataType.FLOAT, 1, title="支管内径", default=32.0),
        PortConf("intersect_angle", DataType.FLOAT, 1, title="相交角（度）", default=90.0),
        PortConf("R_fillet", DataType.FLOAT, 1, title="连接圆角半径", default=12.25),
        PortConf("L_in_ratio", DataType.FLOAT, 1, title="入口直管长度比例", default=5.0),
        PortConf("L_out_ratio", DataType.FLOAT, 1, title="出口直管长度比例", default=10.0),
        PortConf("wall_thickness", DataType.FLOAT, 1, title="壁厚", default=5.0),
        PortConf("mesh_size_global", DataType.FLOAT, 1, title="全局背景网格尺寸", default=6.0),
        PortConf("mesh_size_junction", DataType.FLOAT, 1, title="三通交汇区局部网格尺寸", default=4.0),
        PortConf("mesh_size_interface", DataType.FLOAT, 1, title="流固界面附近局部网格尺寸", default=3.0),
    ]

    OUTPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, title="网格"),
    ]

    @staticmethod
    def run(**options):
        from fealpy.mesher import TeePipeMesher

        params = {
            "D_main": options.get("D_main"),
            "D_branch": options.get("D_branch"),
            "intersect_angle": options.get("intersect_angle"),
            "R_fillet": options.get("R_fillet"),
            "L_in_ratio": options.get("L_in_ratio"),
            "L_out_ratio": options.get("L_out_ratio"),
            "wall_thickness": options.get("wall_thickness"),
            "mesh_size_global": options.get("mesh_size_global"),
            "mesh_size_junction": options.get("mesh_size_junction"),
            "mesh_size_interface": options.get("mesh_size_interface"),
        }

        remesher = TeePipeMesher(params)
        mesh = remesher.init_mesh()

        return mesh
    

