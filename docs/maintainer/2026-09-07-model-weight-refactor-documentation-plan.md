# 模型与权重重构的文档分工与编写计划

> 状态：纲领及模块目标文档已建立，并完成最终审查修订；迁移执行计划尚未编写，实现尚未开始。
> 本文是重构准备阶段的临时记录。

以[目标架构](model-weight-execution.md)为纲领，组织一份纲领、六份详细模块文档和一份临时迁移
执行计划。本文确定这些文档的职责、相互引用和编写顺序，不展开 v3 字段设计或实施任务。
DFlash2 与 MTP、DFlash 统一纳入已落地后端的迁移范围。

全局目标与模块关系以[重构纲领](model-weight-execution.md)为入口。纲领按七个职责部分解释
解耦，其中固定模型执行与 Engine/Frontend 接入合在一份详细文档中；职责数量不等于文档、类或
框架层的数量。本文只记录文档分工，不以文档列表替代核心验收目标。

目标文档按已确认的尺度展开：大部分数学、组件交接和状态规则写在模型代码中，config 保留少量
实例维度和必要参数；实际表示与绑定承担 recipe 的变化。详细文档区分代码规则与持久字段，
以直接函数、循环、有限分支和现有生命周期接口说明实现。

## 1. 文档集合

以下路径均相对于 `docs/maintainer/`。新增文档的文件名按此分工采用；尚未建立的文件不预先
创建空壳，也不建立无效链接。

| 文档 | 状态与定位 | 核心问题 |
|---|---|---|
| [model-weight-execution.md](model-weight-execution.md) | 持续维护的重构纲领 | 原始目标、模块关系、领域边界、所有权和核心约束 |
| [model-contracts.md](model-contracts.md) 及其架构专属定义 | 目标领域规范已建立 | 公共协作边界，以及各架构 config、逻辑数据和状态的确切含义 |
| [artifact-container.md](artifact-container.md) | v3 目标规范已建立 | 模型事实与权重描述如何精确序列化 |
| [weight-conversion.md](weight-conversion.md) | 目标实现文档已建立 | 从源 checkpoint、组件选择和 recipe 生成 v3 |
| [weight-loading.md](weight-loading.md) | 目标实现文档已建立 | 从文件到稳定权重、逻辑绑定与 Op 原生参数 |
| [model-runtime.md](model-runtime.md) | 目标实现文档已建立 | 固定模型执行、Program 组织与 Engine 接入 |
| [program-resources.md](program-resources.md) | 目标实现文档已建立 | 实际绑定如何决定状态、workspace 和 CUDA Graph |
| 迁移执行计划 | 后续建立的临时指引 | 工作包、依赖、验证、旧路径移除和切换条件 |

## 2. 各文档的内容与边界

### 2.1 纲领

`model-weight-execution.md` 继续拥有模型架构、配置、训练实例、物理表示与执行之间的边界，
以及静态执行、融合、可选组件、Op 支持判定和 v2 离线升级等已定要求。

保留能检验跨层关系的全链路例子。详细文档形成后，将重复的字段、接口和局部算法细节收敛到
对应权威，纲领保留摘要与链接，成为整个文档组的入口，不另写一份“最终架构”。

按已确认的纲领目标、职责与改造尺度统筹迁移任务。V3 分片规则已确定，默认文件上限为
32 GB，包含 framing/metadata；自定义模板的存储已允许，运行时保持现有识别和渲染行为。
这些配套沿所属模块接入，以权重与计算解耦为核心的验收目标保持不变。

### 2.2 模型配置与逻辑参数合同

[model-contracts.md](model-contracts.md) 是 converter 与 runtime 共同解释模型事实的目标权威，规定：

- 公共约定与架构专属定义的分工；新增架构向 converter、binder、固定执行与 Program 交付的内容。
- 固定模型代码与小的实例 config 的边界；保留字段沿用对应上游名称，说明实际用途。
- 逻辑参数、持久语义数据、物理表示和使用辅助输入的关系。
- 主模型必需、组件可选的规则，以及架构专属输入、状态和阶段如何接入公共生命周期。
- Frontend、训练配对和产品身份的关联；Qwen4Exp、DeepSeek V4.1 的全链路扩展检查。

各架构文档分别说明固定数学、小的实例字段和派生值；参数目录、组件输入与状态章节用于指导
直接代码。Converter 核对固定语义、解析源组织并提取必要字段，binder 根据实例参数直接绑定。
已建立的 [Qwen3.5 合同](qwen3_5-model-contracts.md)覆盖现有 Dense/MoE 及其组件组合，并引用
现有模型数学文档和正式 DFlash2 合同。文档集合按责任组织，一个模块可有必要的架构专属参考。
例如 Q 的 shape 由公式推导，target 共用维度由引用取得；源 Q/gate 行交错和源分片在 converter 解析。

### 2.3 v3 容器规范

`artifact-container.md` 应足够精确，使 reader 与 writer 可以独立实现，规定：

- Magic、版本、metadata 编码与长度、对齐、payload 起点和 offset 含义。
- 根结构、字段类型、必需/可选字段、缺省和未知字段处理。
- 对象目录、codec/layout 描述、payload 范围及资源引用。
- 逻辑绑定、parent/片段、行范围、共享引用和辅助值关联的序列化。
- 组件存在信息、数据完整性规则及 reader/binder 的检查分界。
- 完整的 Text-only、split/fused 示例和代表性非法情况。

架构 config 只承载精简后的实例参数，固定公式、组件交接和状态程序由代码定义。
字段含义引用对应架构合同；codec/layout 的字节解释引用
[数值格式](tensor-formats.md)与[存储布局](storage-layouts.md)。本规范不再维护另一份模型
shape 公式，也不规定某种组合对应哪个 Op 或 kernel。

单文件与续卷共用一个逻辑 payload 地址空间，对象可跨文件；writer 使用规范命名，reader
按记录路径定位。模板资源承载由本规范定义，运行时行为按
[Frontend 合同](model-runtime.md#62-frontend-资源和模板范围)保持现状。

### 2.4 Converter 实现

`weight-conversion.md` 从用户输入一直讲到文件产物，规定：

- 产物组件选择及必要源输入；未选组件不要求源参数或私有资源齐全。
- 源数学核对、少量实例参数提取、源组织解析、逻辑参数映射与已量化源访问。
- Recipe 的实际写法、默认分配、范围覆盖、冲突处理和辅助值生成要求。
- 转换作业、对象与绑定描述、流式生成和容器写入的组织。
- 量化、融合、拆分、packing、layout 转换与已量化值保留的边界。
- Python/C++ 的共享合同、默认与混合 recipe 例子、错误及转换证据。

临时 v2 升级脚本在本文设独立小节，说明硬编码元数据来源、原 payload 保留、标准库实现与验证，
不另建大型设计文档。面向用户的命令在工具说明中记录。

### 2.5 权重加载与 Op 参数准备

`weight-loading.md` 解释文件到原生调用之间的实现，规定：

- Reader 输出如何进入小的 config 解析、直接语义绑定与所选组件的准备代码。
- Owning storage、物理对象、逻辑引用、共享与 view 的组织。
- Code/high-bit/scale plane、范围、对齐以及 host scalar 的生命周期。
- 上传前的描述准备与上传后的地址绑定。
- Op 如何把逻辑引用整理为原生参数，保存稳定准备结果，并按实际 shape 分派。
- 数据与参数错误、资源错误、Op 不支持的传播边界。

以 attention 单/双 parent 给出具体调用示例，再用 GDN 和 MoE 检验多参数与状态输入。
Binder 不建立格式到重载的全模型选择表；Op 参数准备不发展成图级规划器。

### 2.6 模型运行时与 Engine 接入

`model-runtime.md` 讲清实际执行代码的组织，规定：

- 从架构入口取得固定实现，实例维度与编译期专用化的配合。
- Text、Vision 和各 spec 阶段的固定调用结构、融合调用及模型/Op 分工。
- Program 对不可变模型描述的使用。
- Verification、接受前缀、commit、continuation 与输出的执行顺序。
- 批处理、prefix reuse、评分如何接入同一执行实现。
- Engine 构造、Frontend 初始化、sampling defaults、公开名称和诊断接口的改动。

[Engine 架构](engine-architecture.md)继续拥有顶层请求控制面。本文是其下方的模型执行实现
说明，不复制 Scheduler、ResourceManager 和输出发布合同。各 spec 后端采用统一的文档组织，
具体数学与状态语义引用各自模型合同。

### 2.7 Program 资源与 CUDA Graph

`program-resources.md` 独立解释实际绑定到物理资源构造的过程，规定：

- 模型描述、绑定与启动选项如何成为资源准备输入。
- 对象去重、功能依赖、设备驻留与权重容量。
- 跨 Op 激活存活期、scratch 查询、对齐、复用和峰值组合。
- KV/GDN、draft context、replay records、pending features 与 Vision handoff 的资源需求。
- 数学维度、物理表示、容量和生命周期的归属。
- 启动分配与地址稳定性、capture 范围、frontier、graph key 和 replay 数据更新。
- 容量查询与执行的一致性，以及资源准备失败的处理。

用 Dense scratch 复用、GDN record 生命周期、DFlash2 context 与临时 query 数据给出具体例子。
[资源调度与上下文缓存](resource-scheduling-and-context-cache.md)继续拥有缓存保留、回收与
admission 合同，[Paged KV](paged-kv-cache.md)继续拥有物理 KV 合同，本文引用它们。

### 2.8 临时迁移执行计划

规范明确后，迁移计划记录工作包、依赖、联动修改、首条端到端路径、完整功能覆盖顺序、各阶段
产物与验证条件、旧路径移除、文档切换、升级脚本交付，以及未解决实施问题的影响。

工作包引用对应规范和实现文档，不复制字段、公式或 Op 合同。本文是文档组织记录，不替代这份
后续执行计划。

## 3. 详细文档的深度与现有权威

每份模块文档应给出关键输入输出、所有权与生命周期、具体处理流程、必要接口草案、错误边界，
以及少量能贯通实现的例子；不能只把纲领拆成几份重复说明。精确字段与规则各有一个权威位置。
字段表以持久数据为范围，公式、参数需求、功能条件和状态事务以代码行为说明。
对实例例子同时核对保留字段和固定/派生规则能否共同确定完整数学与数据需求。

现有模型数学、[DFlash2 算法](qwen3.8-27b-dflash2.md)、codec/layout、
[Op 合同与开发规则](op-development.md)、KV 和 Engine 控制面文档继续保留各自职责。
Checkpoint artifact 文档中的源映射、默认 recipe 与旧 v2 inventory 分别收敛到适当位置，
不再承担未来运行时完整 profile 的权威。

编写期间明确标注目标规范与实现状态。容器文档分清现行 v2 与目标 v3；切换后以 v3 为主规范，
旧输入解释仅保留升级所需部分。正常源码、工具和文档的切换由迁移计划安排。

## 4. 编写顺序

1. 已完成纲领的目标、职责与改造尺度。
2. 已完成模型合同和 v3 容器规范，统一数据含义与序列化。
3. 已完成 converter 与加载文档，明确重写范围及可复用基础。
4. 已完成模型运行时与 Program 资源文档，并按最终审查修订跨模块合同。
5. 下一步编写迁移执行计划，用已经确定的合同划分依赖和完成条件。

保留独立的模型合同，避免 writer 与 binder 各自解释模型；保留独立的 Program 资源文档，避免
执行与容量规划形成两套事实。这两项是本次文档分工的重要决定。

## 5. 生命周期

当文档组已建立、分工和导航已落入对应文档及[文档索引](../README.md)，并由迁移执行计划承接
实施统筹后，删除本文及其临时索引入口。迁移执行计划则在迁移完成或放弃后删除；稳定合同留在
各自权威中，不保留历史计划树。
