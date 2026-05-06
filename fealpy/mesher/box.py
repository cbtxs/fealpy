from ..backend import bm
from ..mesh import Mesh, MeshBlock, EntitySector, TopologyBuilder


class Box2d:
    def __init__(
        self,
        box: list[float] = [0, 1, 0, 1],
        nx: int = 10,
        ny: int = 10,
    ) -> None:
        self.box = box
        self.nx = nx
        self.ny = ny

    def initialize(self):
        box = self.box
        nx = self.nx
        ny = self.ny

        NN = (nx + 1) * (ny + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64)
        X, Y = bm.meshgrid(x, y, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32), (nx + 1, ny + 1))

        cell0 = idx[:-1, :-1]
        cell1 = cell0 + ny + 1
        cell2 = cell1 + 1
        cell3 = cell0 + 1
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
            ),
            axis=1,
        )
        return node, cell

    def triangulate(self):
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 3],
            [1, 2, 3],
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 3))

        storage = MeshBlock(positions=node)
        storage.add_sector(EntitySector("tri", cell), root=True)
        TopologyBuilder.construct(storage)
        mesh = Mesh(storage)

        return mesh


class Box3d:
    def __init__(
        self,
        box: list[float] = [0, 1, 0, 1, 0, 1],
        nx: int = 10,
        ny: int = 10,
        nz: int = 10,
    ) -> None:
        self.box = box
        self.nx = nx
        self.ny = ny
        self.nz = nz

    def initialize(self):
        box = self.box
        nx = self.nx
        ny = self.ny
        nz = self.nz

        NN = (nx + 1) * (ny + 1) * (nz + 1)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=bm.float64)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=bm.float64)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=bm.float64)
        X, Y, Z = bm.meshgrid(x, y, z, indexing="ij")

        node = bm.concat(
            (
                bm.reshape(X, (-1, 1)),
                bm.reshape(Y, (-1, 1)),
                bm.reshape(Z, (-1, 1)),
            ),
            axis=1,
        )
        idx = bm.reshape(bm.arange(NN, dtype=bm.int32), (nx + 1, ny + 1, nz + 1))

        nyz = (ny + 1) * (nz + 1)
        cell0 = idx[:-1, :-1, :-1]
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concat(
            (
                bm.reshape(cell0, (-1, 1)),
                bm.reshape(cell1, (-1, 1)),
                bm.reshape(cell2, (-1, 1)),
                bm.reshape(cell3, (-1, 1)),
                bm.reshape(cell4, (-1, 1)),
                bm.reshape(cell5, (-1, 1)),
                bm.reshape(cell6, (-1, 1)),
                bm.reshape(cell7, (-1, 1)),
            ),
            axis=1,
        )
        return node, cell

    def tetrahedralize(self):
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 4))

        storage = MeshBlock(positions=node)
        storage.add_sector(EntitySector("tet", cell), root=True)
        TopologyBuilder.construct(storage)
        mesh = Mesh(storage)

        return mesh

    def prismatize(self):
        node, cell = self.initialize()
        local_cell = bm.asarray([
            [0, 1, 2, 4, 5, 6],
            [0, 2, 3, 4, 6, 7]
        ], dtype=bm.int32)
        cell = bm.reshape(cell[:, local_cell], (-1, 6))

        storage = MeshBlock(positions=node)
        storage.add_sector(EntitySector("prism", cell), root=True)
        TopologyBuilder.construct(storage)
        mesh = Mesh(storage)

        return mesh
