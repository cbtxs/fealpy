# FEALPy | Development Task | 新 mesh 模块开发总任务 | 目标资产设计

- **任务编号**：new_mesh_module
- **任务类型**：Development Task
- **设计目的**：定义 01~08 全链路任务完成后应沉淀的关键目标资产

## 一、目标资产分层

- Governance Asset：任务治理、验收与协作规范资产。
- Knowledge Asset：架构结论、迁移经验、测试方法沉淀。
- Code Asset：接口完善、迁移实现、测试代码与脚本。
- Capability Asset：团队对新 mesh 模块开发与测试协作的稳定能力。

## 二、关键资产清单（建议）

### 1. 总任务治理资产

- 建议位置：`kb/developments/new_mesh_module/`
- 主要对象：
  - `new_mesh_module_task_brief.md`
  - `new_mesh_module_task_object_boundary.md`
  - `new_mesh_module_task_requirement.md`
  - `new_mesh_module_task_network_design.md`
  - `new_mesh_module_task_target_asset_design.md`
- 职责：提供总领任务的对象定义、边界、需求、网络与资产口径。

### 2. 子任务资产

- 建议位置：`kb/developments/new_mesh_module/mesh_01_validation/`、`kb/developments/new_mesh_module/mesh_02_test_system/` 及后续 03~08 子目录
- 职责：承接各节点的设计、执行、验证与收尾结果。

### 3. 代码与测试资产

- 建议位置：`fealpy/mesh/`、`test/`、`example/`
- 职责：承接接口完善、算法迁移、自动化测试、覆盖补强和回归验证。

### 4. 验收与复盘资产

- 建议位置：`kb/developments/new_mesh_module/`
- 职责：承接 07、08 阶段的查漏结论、验收结论和后续建议。

## 三、验收关注点

- 资产是否能从总任务追溯到子任务，再追溯到代码与测试证据。
- 资产职责是否清晰，是否存在本体、边界、需求、网络混写。
- 资产是否支持后续二阶段任务直接复用，而非一次性文档。