
from fealpy.mesh import write_mesh_to_vtu
from fealpy.mesher import Box3d


box = Box3d(box=[0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
mesh = box.prismatize()

prism_view = mesh.sector("prism")
tri_view = mesh.sector("tri")
quad_view = mesh.sector("quad")
edge_view = mesh.sector("edge")
storage = mesh.block

print("geo dim =", mesh.geo_dimension())
print("top dim =", mesh.top_dimension())
print("prism indices:\n", prism_view.indices)
print("tri indices:\n", tri_view.indices)
print("quad indices:\n", quad_view.indices)
print("edge indices:\n", edge_view.indices)
print("prism -> tri relation:\n", storage.relations[("prism", "tri")].tgt_indices)
print("prism -> quad relation:\n", storage.relations[("prism", "quad")].tgt_indices)

write_mesh_to_vtu("box_prism.vtu", mesh)
