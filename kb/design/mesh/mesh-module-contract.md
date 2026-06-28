# FEALPy | 网格模块约定

## 一、Schema 函数

### 1. 接口设计原则

Schema 函数指网格模块针对各类上层模块需求，统一规定的一套算法函数库。这些函数承载算法功能，供网格计算视图调取，**不作为用户接口**。制定函数表时，应遵循这些原则：

- 必须是纯函数。
- 必须只包含算法本身。
- 是上层模块/计算视图的普遍需求。

### 2. 标准函数表

这些函数按分类和字母表顺序，定义在基类文件 `entity_schema.py` 中。

| 分类  | 函数                                | 功能                   |
| --- | --------------------------------- | -------------------- |
| 拓扑  | `boundary`                        | 获得边界信息               |
| 拓扑  | `local_entity`                    | 局部子实体编号              |
| 拓扑  | `relation`                        | 获取拓扑关系               |
| 拓扑  | `size`                            | 实体数量                 |
| 插值点 | `multi_index`                     | 获取多重指标，0-轴广播形式/张量积形式 |
| 插值点 | `num_multi_index`                 | 计算多重指标数量             |
| 插值点 | `global_permutations`             | 子实体局部朝向到全局朝向的顶点置换矩阵  |
| 插值点 | `vo_to_do`                        | 从顶点置换矩阵获得多重指标置换矩阵    |
| 几何  | `barycenter`                      | 单元重心                 |
| 几何  | `barycentric`                     | 装饰器，把笛卡尔坐标函数变为重心坐标函数 |
| 几何  | `bc_to_point`                     | 重心坐标转换为笛卡尔坐标         |
| 几何  | `geo_dimension`                   | 几何维数                 |
| 几何  | `grad_shape_function_barycentric` | 形函数对重心坐标的导数          |
| 几何  | `grad_shape_function_cartesian`   | 形函数对笛卡尔坐标的导数         |
| 几何  | `grad_shape_function_reference`   | 形函数对参考坐标的导数          |
| 几何  | `integral`                        | 计算输入函数的数值积分          |
| 几何  | `jacobi_matrix`                   | 雅可比矩阵，物理坐标关于参考坐标的梯度  |
| 几何  | `measure`                         | 单元测度                 |
| 几何  | `normal`                          | 单元法向                 |
| 几何  | `quadrature_formula`              | 获取积分公式               |
| 几何  | `shape_function`                  | 计算形函数值               |
| 几何  | `tangent`                         | 单元切向                 |

## 二、计算视图函数

### 1. 实体视图接口表

| 函数                     | 功能                   |
| ---------------------- | -------------------- |
| `barycentric`          | 装饰器，把笛卡尔坐标函数变为重心坐标函数 |
| `barycenter`           | 计算单元重心               |
| `bc_to_point`          | 重心坐标转换为笛卡尔坐标         |
| `boundary`             | 获得边界信息               |
| `error`                | 计算两个函数之间的误差          |
| `geo_dimension`        | 几何维数                 |
| `global_permutations`  | 子实体局部朝向到全局朝向的顶点置换矩阵  |
| `grad_shape_function`  | 计算形函数的梯度             |
| (*property*) `indices` | 单元顶点的全局编号            |
| `integral`             | 计算函数积分               |
| `jacobi_matrix`        | 雅可比矩阵                |
| `measure`              | 计算单元测度               |
| `multi_index_matrix`   | 获取多重指标，0-轴广播形式/张量积形式 |
| `normal`               | 计算单元法向               |
| `num_multi_index`      | 计算多重指标数量             |
| `quadrature_formula`   | 获取积分公式               |
| `shape_function`       | 计算形函数值               |
| `size`                 | 获取实体数量               |
| `tangent`              | 计算单元切向               |
| `to`                   | 获取到另一实体的拓扑关系         |
| `top_dimension`        | 获取拓扑维数               |

### 2. 网格视图接口表

网格视图接口承载的是“把网格对象当作视图来读”的能力。这里分成两层：一层是新的标准网格接口，另一层是为了兼容 FEALPy 既有调用习惯而保留的老式接口。标准接口面向 `Mesh` 基类，老式接口面向 `FEALPyMesh` 兼容视图。

**标准网格接口**

| 函数                | 功能                | 备注                                                 |
| ----------------- | ----------------- | -------------------------------------------------- |
| `entity_count`    | 获取指定维数实体的数量       | 参数可用 `str` 或 `int`；支持 `cell/face/edge/node` 以及负维度。 |
| `entities`        | 获取所有指定维数实体的顶点全局编号 | `node` 返回的是位置数组，其余维数返回索引数组的迭代器。                    |
| `entity_views`    | 获取所有指定维数实体的视图     | 返回 `EntityView` 迭代器，适合逐个实体族处理。                     |
| `relation`        | 获取源实体到目标实体的拓扑关系   | 返回 `AdjointRelation` 迭代器，按视图对逐对生成。                 |
| `geo_dimension`   | 获取几何维数            | 与位置张量的列数一致。                                        |
| `top_dimension`   | 获取拓扑维数            | 没有根实体时返回 `-1`。                                     |
| `sector`          | 获取指定名字的实体视图       | 直接按 sector 名称取视图。                                  |
| `is_simplex_mesh` | 判断是否为单纯形网格        | 仅检查 sector 名称集合，不做几何合法性验证。                         |
| `is_tensor_mesh`  | 判断是否为张量网格         | 仅检查 sector 名称集合，不做几何合法性验证。                         |
| `is_elemental`    | 判断是否只有一个根实体类型     | 可选传入 `entity_name` 进一步约束根实体名称。                     |
| `uniform_refine`  | 均匀加密网格            | 当前为占位接口，尚未实现。                                      |
| `add_plot`        | 获取绘图接口            | 返回 `MeshPloter`。                                   |

**FEALPy 旧式接口**

| 函数                         | 功能           | 备注                                                |
| -------------------------- | ------------ | ------------------------------------------------- |
| `entity`                   | 获取指定实体的索引/位置 | 与 `entities` 类似，但返回单个张量；`cell/face/edge/node` 可用。 |
| `cell`                     | 获取单元顶点全局编号   | 等价于 `entity('cell')`。                             |
| `face`                     | 获取面顶点全局编号    | 等价于 `entity('face')`。                             |
| `edge`                     | 获取边顶点全局编号    | 等价于 `entity('edge')`。                             |
| `node`                     | 获取节点坐标       | 等价于 `entity('node')`。                             |
| `localEdge`                | 获取单元局部边编号    | 旧 FEALPy 风格属性，由单元 schema 的局部实体关系生成。               |
| `localFace`                | 获取单元局部面编号    | 旧 FEALPy 风格属性，由单元 schema 的局部实体关系生成。               |
| `entity_view`              | 获取单个实体族视图    | 若同一维度存在多个实体族，则抛出异常。                               |
| `entity_barycenter`        | 计算指定实体的重心    | 支持按实体类型和索引切片。                                     |
| `shape_function`           | 计算单元形函数值     | 作用于单元视图；`cell_shape_function` 是同义别名。              |
| `face_shape_function`      | 计算面形函数值      | 作用于面视图。                                           |
| `edge_shape_function`      | 计算边形函数值      | 作用于边视图。                                           |
| `grad_shape_function`      | 计算单元形函数梯度    | 作用于单元视图。                                          |
| `number_of_cells`          | 获取单元数量       | 与 `cell` 视图大小一致。                                  |
| `number_of_faces`          | 获取面数量        | 与 `face` 视图大小一致。                                  |
| `number_of_edges`          | 获取边数量        | 与 `edge` 视图大小一致。                                  |
| `number_of_nodes`          | 获取节点数量       | 直接返回位置数组第一维长度。                                    |
| `number_of_global_ipoints` | 获取全局插值点总数    | 按所有 sector 汇总。                                    |
| `number_of_local_ipoints`  | 获取局部插值点数量    | 默认按 `cell` 计算。                                    |
| `multi_index_matrix`       | 获取多重指标矩阵     | 直接委派给对应实体视图。                                      |
| `quadrature_formula`       | 获取积分公式       | 直接委派给对应实体视图。                                      |
| `cell_to_face`             | 获取单元到面的拓扑映射  | 返回目标实体索引。                                         |
| `cell_to_edge`             | 获取单元到边的拓扑映射  | 返回目标实体索引。                                         |
| `face_to_edge`             | 获取面到边的拓扑映射   | 返回目标实体索引。                                         |
| `boundary_cell_flag`       | 获取单元边界标记     | 返回布尔掩码。                                           |
| `boundary_face_flag`       | 获取边界面标记      | 返回布尔掩码。                                           |
| `boundary_edge_flag`       | 获取边界边标记      | 返回布尔掩码。                                           |
| `boundary_node_flag`       | 获取边界节点标记     | 返回布尔掩码。                                           |
| `boundary_cell_index`      | 获取边界单元索引     | 返回边界实体下标。                                         |
| `boundary_face_index`      | 获取边界面索引      | 返回边界实体下标。                                         |
| `boundary_edge_index`      | 获取边界边索引      | 返回边界实体下标。                                         |
| `boundary_node_index`      | 获取边界节点索引     | 返回边界实体下标。                                         |
| `cell_to_edge_sign`        | 获取单元到边的方向符号  | 当前实现中仍保留在兼容层。                                     |
| `face_to_edge_sign`        | 获取面到边的方向符号   | 当前实现中仍保留在兼容层。                                     |
| `cell_to_ipoint`           | 获取单元到插值点的映射  | 返回插值点索引张量。                                        |
| `face_to_ipoint`           | 获取面到插值点的映射   | 返回插值点索引张量。                                        |
| `interpolation_points`     | 获取插值点坐标      | 可按实体类型批量获取。                                       |
| `bc_to_point`              | 重心坐标转笛卡尔坐标   | 支持分量元组输入。                                         |
| `entity_measure`           | 获取指定实体的测度    | 节点返回零。                                            |
| `edge_tangent`             | 获取边切向量       | 作用于边视图。                                           |
| `edge_unit_tangent`        | 获取边单位切向量     | 对切向量做归一化。                                         |
| `error`                    | 计算两个函数之间的误差  | 委派给单元视图。                                          |
| `face_normal`              | 获取面法向量       | 作用于面视图。                                           |
| `face_unit_normal`         | 获取面单位法向量     | 对法向量做归一化。                                         |
| `grad_lambda`              | 获取重心坐标梯度     | 支持按拓扑维数选择实体视图。                                    |
| `grad_face_lambda`         | 获取面重心坐标梯度    | `grad_lambda(..., TD=top_dimension()-1)` 的封装。     |
