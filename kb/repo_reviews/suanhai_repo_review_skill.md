# Suanhai | Skill | Repo Review Skill

- **版本**：v0.1
- **状态**：生效
- **入库位置**：`suanhai/skills/nanobot/suanhai_repo_review_skill.md`
- **启用条件**：当需要由 agent、nanobot 或其他受约束工程协作者型大模型，对算海体系内任意仓库执行仓库级 review，并生成可入库的 review 文档时，本 Skill 必须启用
- **适用范围**：算海体系内所有仓库；仓库级 review 任务；以全仓扫描、指定目录扫描或目的驱动自适应扫描方式执行的 review 场景

本文档是算海体系内的 Repo Review Skill。本文档用于为 agent、nanobot 或其他受约束工程协作者型大模型，提供一套可复用、可约束、可扩展的仓库级 review 能力，用于快速理解仓库现状，识别结构、实现、设计、脚本、测试与示例之间的关系，形成结构化 review 文档，并在多轮 review 中维持统一的归档与输出纪律。本文档不替代具体仓库的本地 review 规范文档，不替代具体仓库的设计文档，不替代 Issue / PR 工作流中的正式裁决，也不替代运行验证、构建验证、性能验证或安全验证等专项验证活动。

## 一、Skill 定位与职责边界

### 1.1 Skill 角色

本 Skill 的角色是：

- 面向算海所有仓库的通用仓库级 review Skill

本 Skill 主要负责：

- 对指定仓库执行受约束的结构化 review
- 根据仓库类型、用户目标与仓库本地规范，自适应确定重点扫描目录与 review 主题
- 读取仓库入口文档、本地 review 规范文档与设计文档
- 识别当前现状、设计与实现的一致性、主要问题、风险与改进方向建议
- 生成带日期与主题的 review 文档
- 在生成新 review 文档前，将旧 review 文档归档到 `history/` 目录

### 1.2 本 Skill 不负责的事项

本 Skill 不直接负责：

- 替人类做最终工程判断或路线裁决
- 将 review 中的建议直接推进为工程变更
- 替代具体仓库的本地规则资产
- 执行代码修改、依赖安装、构建、运行、测试或部署
- 宣告某项问题已经被修复
- 将静态扫描结果包装为已验证事实
- 修改与本次 review 无关的文件

### 1.3 与相邻对象的关系

本 Skill 与相邻对象之间的关系如下：

- `<repo>/docs/repo_review/` 是仓库本地 review 规范目录；若存在，本 Skill 必须优先读取并服从该目录中的本地约束
- `<repo>/docs/design/` 是仓库设计文档目录；本 Skill 必须显式关注该目录，并将设计与实现的一致性作为 review 的核心维度之一
- `<repo>/README.md` 是仓库入口文档；本 Skill 应将其作为快速理解仓库角色、目录职责与入口说明的重要输入
- `<repo>/kb/repo_reviews/` 是 review 文档输出目录；本 Skill 的输出文档必须进入该目录
- `<repo>/kb/repo_reviews/history/` 是 review 文档归档目录；本 Skill 在生成新 review 文档前必须将旧 review 文档归档到该目录
- Issue、PR、Review、Validation、Evidence 等对象可作为本 Skill 输出的后续接口，但本 Skill 不替代这些对象本身

## 二、适用仓库与适用对象

### 2.1 适用仓库类型

本 Skill 适用于算海体系内的各类仓库，包括但不限于：

- 交付型仓库
- 研究型仓库
- 平台型仓库
- 工具链仓库
- 应用型仓库
- 混合型仓库

本 Skill 不要求在开始前对仓库类型做绝对精确分类，但必须先形成当前仓库类型的工作判断，并据此调整 review 重点。

### 2.2 适用对象

本 Skill 同时面向：

- nanobot
- 其他 agent
- 受约束工程协作者型大模型
- 使用统一 review 能力对仓库进行盘点、复查、对齐或演进检查的人类协作者

## 三、输入对象与输入边界

### 3.1 显式输入

本 Skill 的显式输入包括：

- 仓库根路径 `<repo>/`
- 当前日期
- 用户指定的 review 目标
- 用户指定的扫描目录
- 是否执行旧 review 归档
- 是否存在仓库本地 review 规范

### 3.2 隐式输入

本 Skill 执行时必须主动识别并尽量读取以下隐式输入：

- `<repo>/docs/repo_review/`
- `<repo>/docs/design/`
- `<repo>/README.md`
- `<repo>/kb/repo_reviews/` 当前已有的 review 文档
- `<repo>/kb/repo_reviews/history/` 中可比较的历史 review 文档
- 仓库主要目录结构

### 3.3 输入优先级

当多个输入对象同时存在时，本 Skill 应按以下优先级处理：

1. `<repo>/docs/repo_review/`
2. `<repo>/docs/design/`
3. `<repo>/README.md`
4. 用户显式指定的扫描目录与 review 目标
5. 仓库目录结构本身
6. 历史 review 文档

若仓库本地 review 规范与统一默认行为存在张力，应优先服从仓库本地 review 规范，除非该规范与上位治理资产冲突。

## 四、扫描模式

### 4.1 Full Repository Scan

当用户未指定目录，且 review 目标为快速了解仓库整体现状时，本 Skill 应进入 Full Repository Scan 模式。

该模式下，本 Skill 应：

- 扫描仓库主要目录
- 主动关注 `docs/repo_review/`、`docs/design/`、`README.md`
- 形成整体性、主题为 `full_repo` 的 review 文档

### 4.2 Scoped Directory Scan

当用户明确指定需要 review 的目录时，本 Skill 应进入 Scoped Directory Scan 模式。

该模式下，本 Skill 应：

- 聚焦指定目录
- 为理解结构关系而进行最小必要扩展扫描
- 将 review 主题收敛到指定目录的职责或目标上
- 不得将未扫描目录写成已确认事实

### 4.3 Goal-Driven Adaptive Scan

当用户给出 review 目标，而未明确给出目录时，本 Skill 应进入 Goal-Driven Adaptive Scan 模式。

该模式下，本 Skill 应：

- 先识别当前 review 的目标
- 再主动选择重点扫描目录与辅助输入文档
- 形成与当前目标相匹配的主题化 review 文档

## 五、默认重点目录与自适应扩展原则

### 5.1 默认重点目录集合

当无更具体约束时，本 Skill 默认优先关注以下目录：

- `src/`
- `include/`
- `tests/`
- `examples/`
- `tools/`

### 5.2 必须优先关注的文档目录

无论采用何种扫描模式，本 Skill 都必须优先关注：

- `<repo>/docs/repo_review/`
- `<repo>/docs/design/`
- `<repo>/README.md`

### 5.3 自适应扩展原则

当以下条件成立时，本 Skill 可以主动扩大扫描范围：

- 指定目录不足以解释实现主线
- 设计文档明确引用了其他关键目录
- review 目标需要追踪配置、脚本、接口或依赖链
- 当前实现与设计文档或 README 之间存在明显张力，需要补充证据

即使发生扩展扫描，也必须在最终 review 文档中显式说明扩展范围，而不得伪装为默认扫描范围。

## 六、仓库类型判断与 review 目标判断

### 6.1 仓库类型工作判断

在正式扫描前，本 Skill 必须先基于以下输入形成当前仓库类型的工作判断：

- README
- 目录结构
- `docs/repo_review/`
- `docs/design/`

该工作判断可包括但不限于：

- 交付型
- 研究型
- 平台型
- 工具链型
- 应用型
- 混合型

### 6.2 review 目标判断

在正式生成 review 文档前，本 Skill 必须判断当前任务更偏向哪类目标。若用户未明确指定，则由本 Skill 主动收敛。

推荐默认目标集合包括：

- `full_repo`
- `implementation`
- `design_consistency`
- `build_and_ops`
- `testing_and_examples`
- `structure`

### 6.3 默认主题

当用户未指定更明确的 review 目标时，本 Skill 的默认主题应为：

- `full_repo`

## 七、核心执行流程

### 7.1 读取本地约束与入口材料

本 Skill 在执行前必须尽量按以下顺序读取和理解输入材料：

1. `<repo>/docs/repo_review/`
2. `<repo>/docs/design/`
3. `<repo>/README.md`

### 7.2 确定扫描模式与重点目录

本 Skill 必须根据：

- 当前仓库类型的工作判断
- 用户指定目录
- 用户 review 目标
- 本地 review 规范
- 设计文档要求

确定：

- 当前采用的扫描模式
- 必扫目录
- 重点扫目录
- 最小必要扩展目录

### 7.3 扫描并提取证据

本 Skill 在执行 review 时，应尽量提取以下类型的证据：

- 目录职责
- 接口与实现映射
- 示例与测试角色
- `tools/` 中环境、构建、运维、打包、诊断或平台适配脚本的作用
- 设计文档与实现之间的一致性线索
- README 与当前目录职责描述的对齐情况
- 本地 review 规范与当前仓库状态的张力

### 7.4 识别旧 review 文档

在生成新 review 文档前，本 Skill 必须检查：

- `<repo>/kb/repo_reviews/` 当前目录下已有的 review 实例文档
- `<repo>/kb/repo_reviews/history/` 中可比较的历史 review 文档

### 7.5 归档旧 review 文档

若当前目录中已存在旧的 review 实例文档，则本 Skill 必须先将其移动到：

- `<repo>/kb/repo_reviews/history/`

归档时：

- 必须保持原有文件名
- 不得覆盖已有历史文件
- 不得移动模板、README、提示词、说明文件或其他非实例文档

### 7.6 生成新 review 文档

完成扫描与必要归档后，本 Skill 必须生成新的 review 文档，并写入：

- `<repo>/kb/repo_reviews/`

## 八、输出命名与归档规则

### 8.1 输出命名规则

本 Skill 生成的新 review 文档文件名必须使用以下格式：

- `<repo>_<YYYY>_<MM>_<DD>_<theme>_review.md`

其中：

- `<repo>` 表示仓库名
- `<YYYY>_<MM>_<DD>` 表示本次 review 日期
- `<theme>` 表示本次 review 主题

### 8.2 输出目录

本 Skill 的默认输出目录为：

- `<repo>/kb/repo_reviews/`

### 8.3 归档目录

本 Skill 的默认归档目录为：

- `<repo>/kb/repo_reviews/history/`

### 8.4 归档对象范围

本 Skill 只应归档 review 实例文档，不得归档：

- 模板文件
- 技能文档
- 提示词文件
- README
- 设计文档
- 本地 review 规范文档
- 其他非 review 实例文件

## 九、默认输出对象与默认文档骨架

### 9.1 默认输出对象

本 Skill 的默认输出对象是：

- 主题化的 Repo Review 文档

### 9.2 默认文档骨架

若仓库本地 review 规范未提供更具体结构，本 Skill 应尽量使用以下默认骨架：

1. Review 定位与本轮范围
2. 输入材料与扫描基线
3. 与上一轮相比的变化摘要
4. 当前现状
5. 结构理解与一致性观察
6. 当前主要问题
7. 改进方向建议
8. 后续动作建议

### 9.3 与本地模板的关系

若仓库存在本地 review 模板或本地 review 规范对文档结构有更具体要求，本 Skill 应优先服从仓库本地约束，而不是强制套用统一默认骨架。

## 十、判断纪律与禁止性行为

### 10.1 必须遵守的判断纪律

本 Skill 必须始终遵守以下纪律：

- 不将未扫描内容写成已确认事实
- 不将静态扫描结果写成运行验证结论
- 不将改进建议写成裁决
- 不伪造上一轮比较
- 不删除不确定性说明
- 不删除未纳入范围说明
- 不擅自改模板
- 不擅自污染无关文件
- 不忽略 `docs/repo_review/`
- 不忽略 `docs/design/`

### 10.2 禁止性行为

本 Skill 不得：

- 跳过本地 review 规范而直接自由输出
- 跳过设计文档而只看代码
- 将 README 视为全部事实来源
- 将 review 结论写成工程授权
- 在未说明依据时给出确定性架构裁决
- 扩大扫描范围却不在最终文档中说明
- 将历史归档动作应用到非 review 实例文档

## 十一、典型适用场景与不适用场景

### 11.1 典型适用场景

本 Skill 适用于以下场景：

- 需要快速了解某个仓库当前现状
- 需要为某个仓库生成结构化 review 文档
- 需要检查设计与实现是否一致
- 需要检查 `tools/`、构建、运维或环境脚本体系
- 需要在多轮 review 中保持命名、归档与结构一致
- 需要先由 agent 做受约束 review，再供人类继续决策

### 11.2 不适用场景

本 Skill 不适用于以下场景：

- 需要做最终架构裁决
- 需要做真实性能验证
- 需要做运行正确性验证
- 需要直接修改代码
- 需要替代 Issue / PR 工作流推进变更
- 需要输出与 review 无关的治理资产

## 十二、Skill 使用提示

### 12.1 典型触发语句

以下触发语句适合调用本 Skill：

- “扫描一下这个仓库，生成当前 repo review”
- “按设计文档和实现做一次一致性 review”
- “重点 review 一下这个仓库的构建和运维脚本体系”
- “把旧 review 归档后，生成今天的新 review”
- “按本地 review 规范，对这个仓库做一次结构化盘点”

### 12.2 输出前自检要求

在输出最终 review 文档前，本 Skill 应自检以下事项：

- 是否已读取仓库本地 review 规范目录
- 是否已关注设计文档目录
- 是否已识别当前 review 目标与主题
- 是否已说明未纳入范围
- 是否已说明不确定性
- 是否已完成旧 review 归档
- 是否仅修改了目标 review 文档
- 是否误将建议写成裁决
- 是否误将静态扫描写成运行验证

## 附录 A：本文件版本演进记录

- **v0.1**：
  - 变更人：魏华祎
  - 变更时间：2026-03-15
  - 变更摘要：
    - 初始建立 Suanhai Repo Review Skill
    - 将能力范围从单仓库实现评审扩展为面向算海所有仓库的通用 repo review 能力
    - 引入 Full Repository Scan、Scoped Directory Scan 与 Goal-Driven Adaptive Scan 三种扫描模式
    - 冻结 `<repo>_<YYYY>_<MM>_<DD>_<theme>_review.md` 的输出命名规则与 `<repo>/kb/repo_reviews/history/` 归档规则
    - 明确要求优先关注 `<repo>/docs/repo_review/` 与 `<repo>/docs/design/`
    - 冻结默认主题 `full_repo` 与推荐主题集合
    - 冻结 repo review 的默认骨架、判断纪律与归档纪律

