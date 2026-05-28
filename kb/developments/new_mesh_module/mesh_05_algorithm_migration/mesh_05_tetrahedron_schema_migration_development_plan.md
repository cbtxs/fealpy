# 四面体 Schema 算法迁移实施计划

**目标：** 将旧网格模块 `fealpy/mesh_old/tetrahedron_mesh.py` 中属于“四面体实体自身”的算法迁移到新网格模块 `fealpy/mesh/schema/tetrahedron.py`，并为 `TetrahedronSchema` 内所有本任务负责的接口补齐单元测试。

**架构原则：** 严格遵守新 mesh 模块的分层边界：`TetrahedronSchema` 只负责四面体这一实体形状的规则和算法，算法输入来自 `EntityContext`、`ctx.block.positions` 和 `ctx.sector.indices`。网格生成、网格加密、全局插值点编号、跨实体 relation 便利接口和 IO 都不进入本任务。

**技术栈：** FEALPy backend manager `bm`，新网格模块 `MeshBlock` / `EntitySector` / `Mesh` / `EntityView`，`pytest`，现有 quadrature 工具。

---

## 一、任务范围

本子任务属于 `mesh_05_algorithm_migration`，只实现四面体形状 Schema 的算法迁移，不迁移旧 `TetrahedronMesh` 的整类能力。

### 1.1 纳入范围

- 修改 `fealpy/mesh/schema/tetrahedron.py`
  - 四面体局部拓扑元数据
  - 四面体重心 `barycenter`
  - 四面体测度 `measure`
  - 四面体重心坐标到物理坐标映射 `bc_to_point`
  - 四面体重心坐标梯度 `grad_lambda`
  - 四面体仿射映射 Jacobian
  - 四面体积分公式 `quadrature_formula`
  - 四面体实体上的 `multi_index`
  - 按 handoff 约定实现四面体 `normal()` 与 `tangent()`
- 新增 `tests/mesh/unit/schema/test_tetrahedron_schema.py`
  - 覆盖本任务实现的每一个 `TetrahedronSchema` 接口
  - 测试必须通过新网格入口构造，即 `MeshBlock`、`EntitySector`、`Mesh`、`mesh.sector("tet")`
- 保留迁移说明
  - 记录旧方法迁移到哪个新接口
  - 记录不迁移的方法及原因

### 1.2 排除范围

- 网格生成
  - 旧方法：`from_box`、`from_unit_cube`、`from_unit_sphere_gmsh`、`from_cylinder_gmsh`、`from_vtu`、`from_medit`
  - 归属：`fealpy/mesher/` 或后续 IO 任务
- 网格加密
  - 旧方法：`uniform_refine`、`uniform_bisect`、`bisect`、`label`、`bisect_options`、`interpolation_with_HB`
  - 原因：这些方法修改全局拓扑，并且跨越 node、edge、tri、tet 等多个实体形状
- 全局插值点编号
  - 旧方法：`number_of_global_ipoints`、`interpolation_points`、`face_to_ipoint`、`cell_to_ipoint`、`prolongation_matrix`
  - 原因：这些方法依赖全局网格拓扑和有限元空间编号，不属于单个实体 Schema
- relation 与 boundary 便利接口
  - 旧方法：`cell_to_face_sign`、`cell_to_edge_sign`、`face_to_edge_sign`、`boundary_edge_flag`
  - 原因：这些方法依赖 `Relation`、边界推断或跨实体方向约定
- 面和边的算法
  - 旧方法：`face_area`、`face_normal`、`face_unit_normal`、edge length 相关逻辑
  - 原因：三角形面由 `TriangleSchema` 负责，边由 `EdgeSchema` 负责

## 二、当前代码状态

`TetrahedronSchema` 当前已有：

- `local_faces`
- `ccw`
- `barycenter`
- `geo_dimension`
- `grad_lambda`
- `measure`

相对 Entity Schema 接口与本任务需求，仍缺少或不完整：

- `bc_to_point`
- `quadrature_formula`
- `multi_index`
- `multi_index_sort`
- `num_multi_index`
- `normal`
- `tangent`
- `jacobi_matrix` 或等价的四面体仿射映射辅助方法

当前 `tests/mesh/unit/schema/` 下没有四面体 Schema 的单元测试。

## 三、旧方法到新接口的迁移映射

| 旧实现来源 | 新目标 | 处理方式 |
|---|---|---|
| `cell_volume()` | `TetrahedronSchema.measure()` | 已有实现，需要测试和加固 |
| 基类 `entity_barycenter()` 行为 | `TetrahedronSchema.barycenter()` | 已有实现，需要测试 |
| 基类 `bc_to_point()` | `TetrahedronSchema.bc_to_point()` | 本任务实现 |
| `grad_lambda(..., TD=3)` | `TetrahedronSchema.grad_lambda()` | 已有实现，需要用不变量测试 |
| `jacobi_matrix()` | `TetrahedronSchema.jacobi_matrix()` | 本任务实现为 Schema 层仿射映射辅助方法 |
| `quadrature_formula(q, etype="cell")` | `TetrahedronSchema.quadrature_formula(q, qtype="legendre")` | 只迁移四面体分支，不保留 `etype` 分派 |
| `multi_index_matrix(p, 3)` | `TetrahedronSchema.multi_index((p,))` | 基于 `InterpolationPoints.multi_index_matrix(p, 4)` 实现 |
| 单纯形局部自由度数量 | `TetrahedronSchema.num_multi_index((p,))` | 本任务实现 |
| `localFace`、`localEdge`、`localFace2edge`、`localEdge2face` | Schema 元数据与 `local_entity()` 推断测试 | 部分已有，补测试 |
| `face_area`、`face_normal`、`face_unit_normal` | 不迁移 | 属于 `TriangleSchema` |
| `from_*` 生成器 | 不迁移 | 属于 `mesher` 或 IO |
| 加密方法 | 不迁移 | 属于后续 mesh editing/refinement 任务 |
| 全局插值点编号方法 | 不迁移 | 属于后续有限元空间或全局编号任务 |

## 四、接口设计决策

### 4.1 `multi_index`

`TetrahedronSchema.multi_index(p)` 只接受一元整数元组，例如 `(2,)`。这是 `mesh_05_algorithm_migration_team_handoff.md` 中约定的接口形式。

标量 `2` 必须抛出 `TypeError`，负数阶数必须抛出 `ValueError`。

对次数 `p`，返回所有四分量非负整数指标，且每一行和为 `p`。返回形状为：

```text
((p + 1) * (p + 2) * (p + 3) // 6, 4)
```

### 4.2 `quadrature_formula`

`TetrahedronSchema.quadrature_formula(q, qtype="legendre")` 只返回四面体实体自身的求积公式。

不保留旧接口中的 `etype` 参数，也不在四面体 Schema 中分派 face/edge 求积。三角形面求积归 `TriangleSchema`，边求积归 `EdgeSchema`。

### 4.3 `normal()` 与 `tangent()`

按 handoff 约定：

```text
法向数量 = G - T
切向数量 = T
返回形状 = [实体数量, 方向数量, 几何维数]
```

对三维空间中的四面体，`G = 3`，`T = 3`：

- `normal()` 返回形状 `(N, 0, 3)`。
- `tangent()` 返回形状 `(N, 3, 3)`。

`tangent()` 的方向取四面体仿射映射的三列基：

```text
[x1 - x0, x2 - x0, x3 - x0]
```

这些方向不单位化，与 handoff 中“默认返回值不做单位化”的约定一致。

### 4.4 `jacobi_matrix`

在 `TetrahedronSchema` 中增加 `jacobi_matrix(ctx, index)`。虽然当前 `EntityView` 还没有包装该接口，但它是四面体实体自身的仿射映射算法，适合放在 Schema 层，也便于后续 view 包装。

返回形状：

```text
(N, G, 3)
```

本任务只支持有效三维四面体，即 `G == 3`。如果 `G != 3`，抛出清晰的 `ValueError`。

## 五、文件计划

- 修改：`fealpy/mesh/schema/tetrahedron.py`
  - 补齐缺失的 Schema 算法
  - 增加必要的输入检查辅助函数
  - 所有方法保持 `@classmethod`，不引入实例状态和副作用
- 新建：`tests/mesh/unit/schema/test_tetrahedron_schema.py`
  - 用一个标准直角四面体构造最小单元测试
  - 用 `Box3d(nx=1, ny=1, nz=1).tetrahedralize()` 做一个轻量集成式验证
  - 对已通过 `EntityView` 包装的接口从 `tet_view` 调用，对尚未包装的接口从 `tet_view.schema` 调用
- 可选文档更新：
  - 如果实现过程中发现 handoff 文档需要同步接口决策，再更新 `mesh_05_algorithm_migration_team_handoff.md`

## Chunk 1：元数据与测试夹具

### Task 1：建立最小四面体 Schema 测试夹具

**涉及文件：**

- 新建：`tests/mesh/unit/schema/test_tetrahedron_schema.py`
- 参考：`tests/mesh/unit/schema/test_node_schema.py`
- 参考：`fealpy/mesher/box.py`

- [ ] **Step 1：创建测试文件导入与断言辅助函数**

沿用 `test_node_schema.py` 的风格，导入：

```python
from fealpy.backend import backend_manager as bm
from fealpy.mesh.schema import TetrahedronSchema
from fealpy.mesh.storage import EntitySector, MeshBlock
from fealpy.mesh.view import Mesh
```

添加 `_as_list()`、`_assert_allclose()`、`_assert_equal()`、`_assert_shape()` 等辅助函数。

- [ ] **Step 2：添加 `_build_single_tet_view()`**

使用一个直角四面体：

```python
positions = bm.asarray(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=bm.float64,
)
tet = bm.asarray([[0, 1, 2, 3]], dtype=bm.int64)
block = MeshBlock(positions=positions)
block.add_sector(EntitySector("tet", tet), root=True)
mesh = Mesh(block)
tet_view = mesh.sector("tet")
```

- [ ] **Step 3：编写元数据测试**

断言：

- `tet_view.schema is TetrahedronSchema`
- `tet_view.top_dimension() == 3`
- `tet_view.geo_dimension() == 3`
- `TetrahedronSchema.local_entity("tri")` 有 4 个面
- `TetrahedronSchema.local_entity("edge")` 有 6 条边
- `TetrahedronSchema.ccw["tri"]` 有 4 个有向面

- [ ] **Step 4：运行元数据测试**

运行：

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

预期：

- 如果只使用已有元数据，该测试可能已经通过。
- 如果失败，失败点应指向 local edge 推断或测试夹具假设。

## Chunk 2：多重指标

### Task 2：实现并测试四面体 `multi_index`

**涉及文件：**

- 修改：`fealpy/mesh/schema/tetrahedron.py`
- 测试：`tests/mesh/unit/schema/test_tetrahedron_schema.py`
- 参考：`fealpy/mesh/topology/ipoints.py`

- [ ] **Step 1：先写失败测试**

测试内容：

- `multi_index((0,)) == [[0, 0, 0, 0]]`
- `multi_index((1,))` 形状为 `(4, 4)`，每行和为 `1`
- `multi_index((2,))` 形状为 `(10, 4)`，每行和为 `2`
- `multi_index(2)` 抛出 `TypeError`
- 负数阶数抛出 `ValueError`
- `num_multi_index((p,))` 等于 `(p + 1) * (p + 2) * (p + 3) // 6`

- [ ] **Step 2：运行测试确认失败**

运行：

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

预期：

- 当前会因为缺少多重指标方法而失败。

- [ ] **Step 3：实现最小代码**

在 `TetrahedronSchema` 中添加解析阶数的私有辅助方法：

```python
@classmethod
def _parse_order(cls, p: tuple[int, ...]) -> int:
    if not isinstance(p, tuple):
        raise TypeError(...)
    if len(p) != 1:
        raise ValueError(...)
    order = p[0]
    if not isinstance(order, int):
        raise TypeError(...)
    if order < 0:
        raise ValueError(...)
    return order
```

然后用：

```python
InterpolationPoints.multi_index_matrix(order, 4)
```

实现 `multi_index()`。

- [ ] **Step 4：运行测试确认通过**

再次运行四面体 Schema 测试。

## Chunk 3：核心几何算法

### Task 3：测试并加固 `measure`、`barycenter`、`bc_to_point`、`grad_lambda`

**涉及文件：**

- 修改：`fealpy/mesh/schema/tetrahedron.py`
- 测试：`tests/mesh/unit/schema/test_tetrahedron_schema.py`

- [ ] **Step 1：编写 `barycenter` 与 `measure` 测试**

对直角四面体：

- 重心应为 `[0.25, 0.25, 0.25]`
- 体积应为 `1.0 / 6.0`

- [ ] **Step 2：编写 `bc_to_point()` 测试**

使用重心坐标：

```python
bcs = (
    bm.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=bm.float64,
    ),
)
```

期望物理点：

- 第一个点为顶点 `[0, 0, 0]`
- 第二个点为重心 `[0.25, 0.25, 0.25]`

- [ ] **Step 3：编写 `grad_lambda()` 测试**

对直角四面体，期望梯度为：

```text
lambda0: [-1, -1, -1]
lambda1: [ 1,  0,  0]
lambda2: [ 0,  1,  0]
lambda3: [ 0,  0,  1]
```

同时验证不变量：

```text
sum_i grad(lambda_i) == [0, 0, 0]
```

- [ ] **Step 4：运行测试确认失败**

`bc_to_point()` 当前应失败，因为尚未实现。

- [ ] **Step 5：实现 `bc_to_point()`**

检查：

- `bcs` 必须是 tuple
- tuple 长度必须为 1
- 最后一维必须为 4

计算：

```python
points = ctx.block.positions[tet]
return bm.einsum("...j,cjd->c...d", bcs[0], points)
```

- [ ] **Step 6：增加清晰的几何维数检查**

对四面体几何算法，要求 `G == 3`。否则抛出：

```python
ValueError("tetrahedron geometry requires GD == 3, got ...")
```

- [ ] **Step 7：运行核心几何测试**

运行：

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

预期：核心几何测试通过。

## Chunk 4：Jacobian、切向、法向与求积

### Task 4：实现剩余四面体 Schema 算法

**涉及文件：**

- 修改：`fealpy/mesh/schema/tetrahedron.py`
- 测试：`tests/mesh/unit/schema/test_tetrahedron_schema.py`

- [ ] **Step 1：编写 `jacobi_matrix()` 失败测试**

对直角四面体，期望：

- 形状为 `(1, 3, 3)`
- Jacobian 为单位矩阵

- [ ] **Step 2：编写 `tangent()` 与 `normal()` 失败测试**

对 `tet_view.tangent()`：

- 形状为 `(1, 3, 3)`
- 与 `jacobi_matrix` 的方向约定一致

对 `tet_view.normal()`：

- 形状为 `(1, 0, 3)`

- [ ] **Step 3：编写 `quadrature_formula()` 失败测试**

通过 schema 调用：

```python
qf = tet_view.schema.quadrature_formula(1, qtype="legendre")
bcs, weights = qf.get_quadrature_points_and_weights()
```

断言：

- `bcs` 与四面体重心坐标兼容
- `weights` 非空

- [ ] **Step 4：实现 `jacobi_matrix()`**

```python
v1 = points[:, 1, :] - points[:, 0, :]
v2 = points[:, 2, :] - points[:, 0, :]
v3 = points[:, 3, :] - points[:, 0, :]
return bm.stack([v1, v2, v3], axis=-1)
```

- [ ] **Step 5：实现 `tangent()`**

返回与 `jacobi_matrix()` 一致的三条仿射基方向。

- [ ] **Step 6：实现 `normal()`**

对 `G == T == 3`：

```python
return bm.zeros((tet.shape[0], 0, 3), dtype=ctx.block.positions.dtype)
```

- [ ] **Step 7：实现 `quadrature_formula()`**

使用现有 FEALPy 求积：

- `q > 7` 时使用 `StroudQuadrature(3, q)`
- 其他情况使用 `TetrahedronQuadrature(q)`

不要引入 `etype` 参数。

- [ ] **Step 8：运行测试确认通过**

运行：

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

预期：四面体 Schema 测试全部通过。

## Chunk 5：与现有新网格生成入口的轻量验证

### Task 5：用 `Box3d.tetrahedralize()` 验证新 Schema

**涉及文件：**

- 测试：`tests/mesh/unit/schema/test_tetrahedron_schema.py`
- 参考：`fealpy/mesher/box.py`

- [ ] **Step 1：增加基于 `Box3d(nx=1, ny=1, nz=1).tetrahedralize()` 的测试**

断言：

- 四面体数量为 6
- 所有四面体测度为正
- 总体积为 `1.0`
- `barycenter()` 形状为 `(6, 3)`
- `grad_lambda()` 形状为 `(6, 4, 3)`

- [ ] **Step 2：运行测试**

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

预期：通过。

## Chunk 6：最终验证与交付检查

### Task 6：运行相关测试并记录结果

**涉及文件：**

- 测试：`tests/mesh/unit/schema/test_tetrahedron_schema.py`
- 测试：`tests/mesh/unit/schema/test_node_schema.py`
- 测试：`kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py`

- [ ] **Step 1：运行四面体 Schema 测试**

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

- [ ] **Step 2：运行 node Schema 测试**

```bash
python -m pytest tests/mesh/unit/schema/test_node_schema.py -q
```

当前已知风险：

- `NodeSchema.quadrature_formula()` 的签名可能仍不接受 `qtype`，这属于相邻接口一致性问题，不是本四面体子任务的直接内容。

- [ ] **Step 3：运行 Line Walk 验证**

```bash
python -m pytest kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py -q
```

- [ ] **Step 4：检查工作区**

```bash
git status --short
```

预期改动文件：

- `fealpy/mesh/schema/tetrahedron.py`
- `tests/mesh/unit/schema/test_tetrahedron_schema.py`
- 本计划文档，如果尚未提交

不要回退无关的已有工作区改动。

## 七、完成判据

本子任务完成时必须同时满足：

- `TetrahedronSchema` 实现所有纳入范围内的 Schema 层方法。
- 每个已实现方法都有聚焦单元测试。
- 测试通过新网格模块入口构造对象，不依赖旧 `TetrahedronMesh`。
- 没有把网格生成、加密、IO、全局插值点编号迁入 `TetrahedronSchema`。
- 本文档已明确记录旧方法的迁移目标或排除原因。
- 使用项目测试 Python 环境运行四面体 Schema 测试通过。

## 八、当前复核记录（2026-05-28）

本节记录本次开发后的完成度复核结果，用于避免把四面体子任务状态扩展成整个新网格模块状态。

### 8.1 四面体子任务已覆盖内容

- `fealpy/mesh/schema/tetrahedron.py` 已实现本计划纳入范围内的实体级接口：
  - `barycenter`
  - `bc_to_point`
  - `geo_dimension`
  - `grad_lambda`
  - `jacobi_matrix`
  - `measure`
  - `multi_index`
  - `multi_index_sort`
  - `num_multi_index`
  - `normal`
  - `quadrature_formula`
  - `tangent`
- `tests/mesh/unit/schema/test_tetrahedron_schema.py` 已覆盖：
  - schema 分派和 `local_entity` / `ccw` 元数据
  - 多重指标数量、行和、非法阶数、排序语义
  - 重心、测度、重心坐标到物理坐标映射
  - `grad_lambda`、`jacobi_matrix`、`normal`、`tangent`
  - 标量 `index` 下保持单实体轴的形状语义
  - 四面体求积公式
  - `Box3d(nx=1, ny=1, nz=1).tetrahedralize()` 生成的 6 个四面体总体积与梯度形状

### 8.2 复核命令

已运行并通过：

```bash
python -m pytest tests/mesh/unit/schema/test_tetrahedron_schema.py -q
```

结果：`12 passed`。

已运行并通过：

```bash
python -m pytest kb/developments/new_mesh_module/mesh_01_validation/triangle_line_walk_case_on_new_mesh/tests/test_triangle_line_walk_on_new_mesh.py -q
```

结果：`6 passed`。

已运行：

```bash
python -m pytest tests/mesh/unit/schema -q
```

结果：`1 failed, 21 passed`。失败项为 `NodeSchema.quadrature_formula()` 不接受 `qtype` 参数，属于 node schema 接口一致性问题，不属于本四面体子任务的直接修改范围。

已运行：

```bash
git diff --check
```

结果：通过，无空白格式错误。

### 8.3 仍不宣称完成的范围

- 不宣称整个 `tests/mesh/unit/schema` 已全绿，因为当前仍有 node schema 的 `qtype` 签名失败。
- 不宣称新网格模块已经满足完整有限元求解链路，因为全局插值点编号、自由度映射、有限元空间、边界条件处理、面法向和跨实体关系仍属于其他层或后续任务。
- 不把网格生成、加密、IO、全局插值点编号迁入 `TetrahedronSchema`。
