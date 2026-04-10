# 概率性自适应思维图系统 (Probabilistic Adaptive Thought Graph, PATG)

## 1. 系统概述

本系统设计为一个**基于有向无环图 (DAG) 的概率性推理引擎**。与传统确定性搜索不同，本系统引入概率论机制模拟“量子态”的演化过程：通过**扩散**探索解空间，通过**干涉**融合不同路径信息，通过**演化**评估状态质量，最后通过**坍缩**剪枝低优路径。

系统核心目标是解决复杂、非结构化问题，支持动态图结构生成，具备自适应收敛能力，防止陷入局部最优。

---

## 2. 核心概念与术语映射

为了便于工程实现，我们将“量子”隐喻映射为标准工程术语：

| 原概念 | 工程术语 | 定义 |
| :--- | :--- | :--- |
| **扩散 (Diffusion)** | **Branching Operator (分支算子)** | 从父节点生成多个子节点的生成操作。 |
| **演化 (Evolution)** | **Scoring & Refinement Operator (评分与优化算子)** | 对节点内容进行质量评估 (0-1) 和微调的操作。 |
| **干涉 (Interference)** | **Cross-Pollination Operator (交叉融合算子)** | 随机选取同层或跨层的多个节点，将其内容合成新节点的操作。 |
| **坍缩 (Collapse)** | **Probabilistic Pruning Operator (概率剪枝算子)** | 基于动态阈值和概率分布，丢弃低分节点的操作。 |
| **波函数** | **State Distribution (状态分布)** | 当前图中所有存活节点的集合及其概率幅。 |
| **时间/深度** | **Depth Decay (深度衰减)** | 随着图深度增加，生存阈值自动提高的机制。 |

---

## 3. 系统架构设计

系统采用 **Controller-Operator-Graph** 三层架构。

### 3.1 数据模型 (Data Models)

#### 3.1.1 Node (思维节点)
```python
class Node:
    id: str                  # 唯一标识
    content: str             # 思维内容/状态
    depth: int               # 在图中的深度 (根节点为 0)
    score: float             # 质量评分 [0.0, 1.0]
    parents: List[str]       # 父节点 ID 列表
    children: List[str]      # 子节点 ID 列表
    is_active: bool          # 是否存活 (未被坍缩)
    metadata: Dict           # 额外信息 (如生成耗时、Token 消耗)
```

#### 3.1.2 ThoughtGraph (思维图)
```python
class ThoughtGraph:
    nodes: Dict[str, Node]   # 所有节点索引
    roots: List[str]         # 根节点 ID 列表
    active_leaves: List[str] # 当前活跃叶节点 (可用于下一步扩散)
    
    def add_node(node: Node)
    def link_edges(parent_id: str, child_id: str)
    def get_nodes_at_depth(depth: int) -> List[Node]
    def prune_node(node_id: str) # 标记为非活跃
```

### 3.2 配置参数 (Configuration)

系统行为由以下超参数控制，支持运行时动态调整：

```python
class SystemConfig:
    # 扩散控制
    max_branch_factor: int = 5        # 单次扩散最大子节点数
    min_branch_factor: int = 2        # 单次扩散最小子节点数
    
    # 演化控制
    refinement_enabled: bool = True   # 是否启用生成后微调
    
    # 干涉控制
    interference_probability: float = 0.2  # 每步触发干涉的概率
    max_interference_sources: int = 3      # 干涉操作最大输入节点数
    min_interference_sources: int = 2      # 干涉操作最小输入节点数
    
    # 坍缩控制 (核心概率模型)
    base_threshold: float = 0.3       # 基础生存阈值 (T_base)
    depth_decay_rate: float = 0.15    # 深度衰减系数 (lambda)
    temperature: float = 1.0          # 温度系数 (控制随机性，越高越随机)
    
    # 终止条件
    max_depth: int = 10               # 最大深度限制 (硬停止)
    max_total_nodes: int = 500        # 最大节点总数限制
    target_score: float = 0.95        # 达到此分数立即停止
```

---

## 4. 核心算子实现逻辑

### 4.1 分支算子 (Branching Operator)
**输入**: 父节点 $P$
**输出**: 子节点集合 $\{C_1, C_2, ..., C_k\}$

**逻辑**:
1. 调用 LLM Prompt: "基于当前状态，生成 $k$ 种不同的推导方向/解决方案"。
2. $k$ 在 `[min_branch_factor, max_branch_factor]` 之间随机采样。
3. 创建新节点，设置 `depth = P.depth + 1`，建立父子关系。
4. 初始 `score = 0.0` (待演化算子评估)。

### 4.2 评分与优化算子 (Scoring & Refinement Operator)
**输入**: 节点 $N$
**输出**: 更新后的 $N.score$ 和 $N.content$

**逻辑**:
1. **评分**: 调用 LLM 或 验证函数，输出 $S \in [0, 1]$。
   - Prompt: "评估该解决方案的有效性、逻辑性和完整性，返回 0-1 之间的分数"。
2. **优化 (可选)**: 若 $S < 0.8$ 但 $> T_{base}$，可触发一次自我修正。
   - Prompt: "指出上述方案的缺陷并尝试改进"。
3. 更新节点状态。

### 4.3 交叉融合算子 (Cross-Pollination Operator)
**输入**: 当前层级的活跃节点集合 $L_d$
**输出**: 新生成的融合节点 $N_{new}$

**逻辑**:
1. **触发判断**: 生成随机数 $r \in [0, 1]$，若 $r < interference\_probability$ 则执行。
2. **采样**: 从 $L_d$ 中随机抽取 $m$ 个节点 ($m \in [min\_sources, max\_sources]$)。
   - *策略*: 可加权采样 (高分节点被选概率大) 或 纯随机 (增加多样性)。
3. **合成**: 调用 LLM。
   - Prompt: "综合以下 $m$ 个观点的精华，构建一个新的、更完善的解决方案..."
4. **创建**: 生成新节点 $N_{new}$，其父节点指向被采样的 $m$ 个节点，深度设为 $d+1$ (或保持 $d$，视作同级增强)。

### 4.4 概率剪枝算子 (Probabilistic Pruning Operator)
**输入**: 节点 $N$ (含 score $S$, depth $d$)
**输出**: Boolean (是否保留)

**核心算法**:
1. **计算动态阈值** $T(d)$:
   $$ T(d) = T_{base} + (1 - T_{base}) \times (1 - e^{-\lambda \cdot d}) $$
   - 当 $d=0$, $T \approx T_{base}$
   - 当 $d \to \infty$, $T \to 1.0$
   
2. **计算生存概率** $P_{keep}$:
   - 如果 $S < T(d)$: $P_{keep} = 0$ (直接丢弃)
   - 如果 $S \ge T(d)$:
     $$ P_{keep} = \frac{1}{1 + e^{-\frac{S - T(d)}{temperature}}} $$
     *(使用 Sigmoid 函数平滑过渡，temperature 控制曲线陡峭度)*

3. **执行采样**:
   - 生成随机数 $r \in [0, 1]$。
   - 若 $r > P_{keep}$，标记节点为 `is_active = False` (坍缩)。

---

## 5. 执行流程 (Controller Loop)

系统采用 **BFS (广度优先) + 动态调度** 混合模式。

```python
def run_system(initial_prompt):
    # 1. 初始化
    graph = ThoughtGraph()
    root = create_root_node(initial_prompt)
    graph.add_node(root)
    queue = [root]
    
    while queue is not empty:
        current_depth = queue[0].depth
        
        # A. 检查全局终止条件
        if current_depth >= config.max_depth or graph.total_nodes >= config.max_total_nodes:
            break
            
        # B. 处理当前层级 (Level-wise Processing)
        current_level_nodes = get_all_active_nodes_at_depth(current_depth)
        
        # 1. 演化阶段：对所有新节点进行评分和优化
        for node in current_level_nodes:
            if node.score == 0: # 未评分
                scoring_operator.execute(node)
                if node.score >= config.target_score:
                    return node # 找到完美解
        
        # 2. 干涉阶段：尝试跨节点融合
        if random() < config.interference_probability:
            new_fused_node = cross_pollination_operator.execute(current_level_nodes)
            if new_fused_node:
                graph.add_node(new_fused_node)
                # 融合节点加入下一轮队列
                queue.append(new_fused_node)
        
        # 3. 坍缩阶段：应用概率剪枝
        for node in current_level_nodes:
            if not pruning_operator.should_keep(node):
                graph.prune_node(node.id)
                # 从队列移除被剪枝的节点，不再向下扩散
                if node in queue: queue.remove(node)
        
        # 4. 扩散阶段：对存活节点进行分支
        next_level_queue = []
        for node in current_level_nodes:
            if node.is_active:
                children = branching_operator.execute(node)
                graph.add_nodes(children)
                next_level_queue.extend(children)
        
        # 更新队列
        queue = sorted(next_level_queue, key=lambda x: x.score, reverse=True)
        
    # 6. 输出结果
    return get_best_node(graph)
```

---

## 6. 关键特性说明

### 6.1 为什么需要“干涉”？
传统树搜索 (ToT) 的分支是独立的，容易形成“信息孤岛”。
- **场景**: 分支 A 解决了“逻辑严密性”，分支 B 解决了“创意新颖性”。
- **干涉作用**: 将 A 和 B 结合，可能产生既严密又创新的分支 C，这是单一路径无法到达的“涌现”解。

### 6.2 深度衰减的物理意义
- **浅层 (Low Depth)**: 阈值低，允许大量“粗糙”的想法存在，鼓励**探索 (Exploration)**。
- **深层 (High Depth)**: 阈值趋近 1.0，只有近乎完美的想法才能继续，强制**利用 (Exploitation)** 和收敛。
- 这模拟了人类思考过程：先头脑风暴 (发散)，再逐步聚焦细节 (收敛)。

### 6.3 温度的作用
- **High Temperature**: 即使分数略低于阈值，也有概率存活。防止过早杀死“大器晚成”的节点。
- **Low Temperature**: 严格遵循阈值，快速剪枝，节省资源。

---

## 7. 扩展性建议

1. **自定义评分函数**: 除了 LLM 评分，可接入单元测试、代码执行器、外部 API 验证作为评分依据。
2. **多模态支持**: 节点内容不仅可以是文本，也可以是代码片段、JSON 结构或图像描述。
3. **记忆机制**: 在图中引入“全局记忆节点”，所有分支均可读取/写入共享上下文。
4. **可视化调试**: 由于是图结构，建议开发前端可视化工具，实时展示节点的生成、融合与剪枝过程。

---

## 8. 总结

本设计方案将抽象的“量子思维”转化为可落地的工程模块。通过**分支、评分、融合、剪枝**四个标准算子的组合，配合**深度相关的动态概率模型**，构建了一个既能广泛探索又能高效收敛的智能推理系统。开发者只需关注算子的具体提示词工程和参数调优，即可适应不同领域的复杂任务。
