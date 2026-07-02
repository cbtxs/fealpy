# 网格拓扑关系

## 一、拓扑关系对象

> [!INFO] 代码文件 `fealpy\mesh\storage\relation.py`

拓扑关系接口总是返回拓扑关系对象 `Relation`，该对象具有以下属性：

| 属性              | 类型             | 含义          |
| --------------- | -------------- | ----------- |
| `src_name`      | str            | 映射关系源的实体名称  |
| `tgt_name`      | str            | 映射关系目标的实体名称 |
| `tgt_indices`   | Tensor         | 目标实体索引      |
| `src_indices`   | Tensor \| None | 源实体索引       |
| `local_indices` | Tensor \| None | （待补充）       |

### 1.1 布局

针对不同的拓扑关系，Relation 有两种数据布局方式：

（1）**同质性布局**（稠密）。只有当每个源实体映射到相同数量的目标实体的情形时，才可使用这种方式；此时

- `tgt_indices` 为 2 维张量；
- `src_indices` 为 None。

（2）**异质性布局**（稀疏）。这种方式适用于任意关系，此时 `tgt_indices` 和 `src_indices`

- 均为 1 维张量；
- 长度相等，都等于映射关系的数量。

### 1.2 视图与操作

用户可以将 Relation 转化成特定格式以便参与后续计算。

```python
relation.as_array() # 仅同质性布局可用
relation.as_coo()
relation.as_csr()
```

`inverse` 方法可以方便地逆向拓扑关系，就像把 cell_to_face 变成 face_to_cell。

```python
inv_relation = relation.inverse()
```

> [!INFO] 逆映射的布局变化
> 经 `inverse` 逆向以后的拓扑关系总是异质性布局的。

## 二、使用拓扑关系接口

实体视图和兼容性网格视图中均提供了直观的拓扑关系接口。结合实体视图，我们一般可以这样使用：

（1）**指定实体形状名字**

```python
tet2tri = mesh.Tet.to("tri").as_array()
# 或者
tet2tri = mesh.Tet.to(mesh.Tri).as_array()
```

（2）**指定实体维数**

```python
hex2quad = mesh.Cell[0].to(mesh.Face[0]).as_array()
```

## 三、拓扑构造器

