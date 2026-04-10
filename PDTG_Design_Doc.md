# 概率性动态思维图系统 (Probabilistic Dynamic Thought Graph, PDTG)

## 1. 系统概述

本系统是一个基于有向无环图 (DAG) 的推理框架，引入了概率性剪枝和跨分支合成机制。与传统的静态图不同，本系统的图结构在运行时动态生成，通过**扩散**、**演化**、**干涉**和**坍缩**四种核心操作，模拟量子力学中的波函数演化过程，实现高效的问题求解空间搜索。

### 核心特性
- **动态拓扑**：图结构由模型在运行时决定，而非预定义。
- **概率生存**：节点基于评分和深度动态计算生存概率，避免硬阈值截断。
- **跨层干涉**：允许同一层级的不同分支进行信息合成，打破树形结构的隔离。
- **深度衰减**：随着推理深度增加，生存标准自动提高，强制系统收敛。

---

## 2. 核心概念定义

### 2.1 思维节点 (ThoughtNode)
代表推理过程中的一个状态快照。
- **id**: 唯一标识符
- **content**: 思维内容 (字符串)
- **depth**: 当前节点在图中的深度 (根节点为 0)
- **score**: 质量评分 (0.0 ~ 1.0)
- **probability**: 生存概率 (0.0 ~ 1.0)
- **parents**: 父节点 ID 列表 (支持多父节点，用于干涉操作)
- **is_alive**: 布尔值，标记是否被坍缩操作丢弃

### 2.2 系统参数配置 (SystemConfig)
控制整个演化过程的超参数。
```python
class SystemConfig:
    # 扩散控制
    max_branching_factor: int = 5       # 单个节点最大子节点数
    base_diffusion_rate: float = 0.8    # 基础扩散概率
    
    # 演化控制
    score_alpha: float = 1.5            # 评分敏感度 (越高越精英主义)
    
    # 坍缩控制 (时间/深度约束)
    depth_decay_lambda: float = 0.3     # 深度衰减系数 (越大收敛越快)
    base_survival_threshold: float = 0.2 # 基础生存阈值
    
    # 干涉控制
    interference_probability: float = 0.3 # 触发干涉操作的概率
    max_interference_parents: int = 3     # 合成操作最大父节点数
    
    # 全局终止条件
    max_depth: int = 10                 # 最大递归深度
    max_total_nodes: int = 200          # 最大节点总数 (防爆炸)
    target_score: float = 0.95          # 达到此分数立即停止
```

---

## 3. 核心操作流程

系统执行是一个循环过程，维护一个“活跃节点池 (Active Pool)”。每轮迭代包含以下四个阶段：

### 3.1 扩散 (Diffusion)
**目标**：从当前活跃节点生成新的候选解，扩大搜索空间。
- **输入**：当前活跃节点集合 $S_t$
- **逻辑**：
  1. 遍历 $S_t$ 中的每个节点 $n$。
  2. 调用 LLM 生成 $k$ 个子节点 ($1 \le k \le \text{max\_branching\_factor}$)。
  3. 子节点内容基于父节点内容进行推导、分解或发散。
  4. 新节点深度 $d_{new} = d_{parent} + 1$。
- **输出**：新生成的候选节点集合 $C_{diff}$。

### 3.2 演化 (Evolution)
**目标**：评估新生成节点的质量，并赋予生存概率幅。
- **输入**：候选节点集合 $C_{diff}$
- **逻辑**：
  1. **评分**：调用 LLM 或验证函数对每个节点打分 $s \in [0, 1]$。
  2. **计算生存概率**：应用深度衰减公式。
     $$ P_{survive} = \sigma(\alpha \cdot s - \beta \cdot d - \gamma) $$
     其中：
     - $\sigma$ 是 Sigmoid 函数，将结果映射到 (0, 1)。
     - $\alpha$ 是评分权重 (`score_alpha`)。
     - $\beta$ 是深度惩罚 (`depth_decay_lambda`)。
     - $\gamma$ 是基础偏置，由 `base_survival_threshold` 推导。
     - *直观解释*：分数越高存活率越高；深度越深，对分数要求越苛刻。
  3. 更新节点的 `probability` 字段。
- **输出**：已评分的节点集合 $C_{eval}$。

### 3.3 干涉 (Interference)
**目标**：打破独立分支的隔离，通过随机合成产生涌现解。
- **输入**：已评分节点集合 $C_{eval}$
- **逻辑**：
  1. 以 `interference_probability` 的概率决定是否触发本轮干涉。
  2. 若触发，从 $C_{eval}$ 中随机采样 $m$ 个节点 ($2 \le m \le \text{max\_interference\_parents}$)。
     - *采样策略*：可按概率加权采样，高分节点更容易被选中参与合成。
  3. 调用 LLM 执行“合成”指令：综合这 $m$ 个节点的观点，生成一个新的融合节点。
  4. 新节点的深度取父节点深度的最大值 + 1 (或保持当前层级，视策略而定)。
  5. 对新节点执行**演化**步骤（评分 + 计算概率）。
- **输出**：融合后的增强节点集合 $C_{final}$。

### 3.4 坍缩 (Collapse)
**目标**：根据概率分布剪枝，保留高潜力路径，防止资源爆炸。
- **输入**：最终候选节点集合 $C_{final}$
- **逻辑**：
  1. 对每个节点 $n$，生成一个随机数 $r \sim U(0, 1)$。
  2. 若 $r < n.\text{probability}$，标记 `is_alive = True`，加入下一轮活跃池 $S_{t+1}$。
  3. 若 $r \ge n.\text{probability}$，标记 `is_alive = False`，该分支终止。
  4. **额外保护机制**：如果某一层所有节点都被坍缩，强制保留概率最高的一个节点（避免全灭死锁）。
- **输出**：下一代活跃节点池 $S_{t+1}$。

---

## 4. 系统架构设计

### 4.1 模块划分

#### A. 控制器 (Orchestrator)
- **职责**：主循环调度，维护全局状态，检查终止条件。
- **关键方法**：
  - `run(initial_prompt)`: 启动流程。
  - `check_termination()`: 检查是否达到最大深度、最大节点数或找到最优解。
  - `step()`: 执行一次完整的 [扩散->演化->干涉->坍缩] 循环。

#### B. 图管理器 (GraphManager)
- **职责**：维护节点对象和边关系，提供查询接口。
- **数据结构**：
  - `nodes`: Dict[ID, ThoughtNode]
  - `adjacency_list`: Dict[ID, List[ChildID]]
  - `active_pool`: List[ID] (当前存活的节点 ID)
- **关键方法**：
  - `add_node(...)`: 创建并注册节点。
  - `get_layer_nodes(depth)`: 获取特定深度的所有节点 (用于干涉采样)。
  - `prune_dead_nodes()`: 清理被坍缩的节点以节省内存。

#### C. 算子引擎 (OperatorEngine)
- **职责**：封装与 LLM 的交互逻辑，执行具体操作。
- **关键方法**：
  - `diffuse(node)`: 生成子节点提示词构建与解析。
  - `evaluate(node)`: 评分提示词构建与解析。
  - `interfere(nodes_list)`: 合成提示词构建与解析。

#### D. 概率计算器 (ProbabilityCalculator)
- **职责**：纯数学计算，无副作用。
- **关键方法**：
  - `calculate_survival_prob(score, depth, config)`: 执行核心公式计算。

---

## 5. 详细算法流程 (伪代码)

```python
def run_pdtg(initial_prompt, config):
    # 初始化
    graph = GraphManager()
    root = graph.add_node(content=initial_prompt, depth=0, parents=[])
    active_pool = [root.id]
    
    while not is_finished(graph, config):
        # 1. 扩散阶段
        new_candidates = []
        for node_id in active_pool:
            node = graph.get_node(node_id)
            children = engine.diffuse(node, config.max_branching_factor)
            for child in children:
                graph.add_node(child, depth=node.depth + 1, parents=[node.id])
                new_candidates.append(child.id)
        
        if not new_candidates:
            break
            
        # 2. 演化阶段 (评分 & 计算概率)
        for node_id in new_candidates:
            node = graph.get_node(node_id)
            score = engine.evaluate(node)
            node.score = score
            node.probability = prob_calc.calculate(score, node.depth, config)
            
        # 3. 干涉阶段 (随机合成)
        if random() < config.interference_probability:
            # 从当前层随机选 m 个节点
            current_depth = graph.get_node(new_candidates[0]).depth
            layer_nodes = graph.get_layer_nodes(current_depth)
            
            if len(layer_nodes) >= 2:
                parents = random_sample(layer_nodes, config.max_interference_parents)
                fused_content = engine.interfere(parents)
                fused_node = graph.add_node(
                    content=fused_content, 
                    depth=current_depth + 1, 
                    parents=[p.id for p in parents]
                )
                # 对新节点也进行演化
                fused_node.score = engine.evaluate(fused_node)
                fused_node.probability = prob_calc.calculate(fused_node.score, fused_node.depth, config)
                new_candidates.append(fused_node.id)
        
        # 4. 坍缩阶段 (概率剪枝)
        next_active_pool = []
        for node_id in new_candidates:
            node = graph.get_node(node_id)
            if random() < node.probability:
                node.is_alive = True
                next_active_pool.append(node_id)
            else:
                node.is_alive = False
        
        # 防止全灭保护
        if not next_active_pool and new_candidates:
            best_node_id = max(new_candidates, key=lambda id: graph.get_node(id).probability)
            graph.get_node(best_node_id).is_alive = True
            next_active_pool = [best_node_id]
            
        active_pool = next_active_pool
        
        # 检查是否找到解
        best_score = max([graph.get_node(id).score for id in active_pool], default=0)
        if best_score >= config.target_score:
            return graph.get_best_solution()

    return graph.get_best_solution()
```

---

## 6. 关键实现细节建议

### 6.1 提示词工程 (Prompt Engineering)
- **扩散提示词**：强调多样性 ("Generate 3 distinct approaches...")。
- **评分提示词**：要求输出结构化 JSON (`{"score": 0.8, "reason": "..."}`)，便于解析。
- **干涉提示词**：强调综合与去重 ("Synthesize the following viewpoints into a coherent solution, resolving contradictions...")。

### 6.2 性能优化
- **异步执行**：扩散和评分阶段可以完全并行化 (AsyncIO)，大幅降低延迟。
- **缓存机制**：对相同的思维内容缓存评分结果，避免重复计算。
- **流式处理**：如果节点数量巨大，可以采用分批处理策略。

### 6.3 可观测性
- **日志记录**：记录每一代的节点数量、平均分数、最大深度。
- **可视化**：生成 Graphviz 或 HTML 文件，展示节点的生存/死亡状态及血缘关系。

---

## 7. 扩展方向
1. **自适应参数**：让模型根据当前搜索态势动态调整 `depth_decay_lambda` (例如：发现陷入局部最优时，降低衰减率以增加探索)。
2. **记忆库 (Memory Bank)**：将历史上被坍缩但分数尚可的节点存入“潜意识库”，在干涉阶段有极低概率被重新唤醒。
3. **多模态支持**：节点内容不仅限于文本，可以是代码、图像描述或结构化数据。

---

## 8. 总结
本设计文档描述了一个**概率性动态思维图系统**。它摒弃了固定的流程图，转而采用类似生物进化或量子演化的动态机制。通过**扩散**探索空间，**演化**评估质量，**干涉**促进创新，**坍缩**控制资源，该系统能够在保证收敛的前提下，最大化地挖掘复杂问题的潜在解空间。
