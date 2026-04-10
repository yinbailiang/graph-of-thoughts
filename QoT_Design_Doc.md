# QoT (Quantum of Thought) System Design Document

## 1. 系统概述 (Overview)

**QoT (Quantum of Thought)** 是一个自适应的元推理引擎。它不仅能模拟传统的线性 (CoT)、树形 (ToT) 和静态图 (GoT) 推理，更引入了概率性、纠缠态和外部知识增强机制，实现了对复杂问题的动态求解。

### 核心理念：量子穿越势垒
系统将推理过程类比为**量子粒子穿越多重势垒**的物理过程：
1.  **波函数扩散 (Diffusion)**：在不确定空间中广泛探索，生成多种可能性（QoT 模式）。
2.  **量子纠缠 (Entanglement)**：通过 RAG 检索和内部关联，在不同思维节点间建立强耦合联系。
3.  **干涉合成 (Interference)**：跨分支、跨层级融合信息，打破思维孤岛，产生涌现解。
4.  **观测坍缩 (Collapse)**：基于概率幅和深度衰减机制，收敛到最优解路径。
5.  **路径固化 (Distillation & Caching)**：将成功的随机探索路径剪枝、蒸馏为确定性的 GoT 模板并缓存，供未来高效复用。

### 系统目标
- **适应性 (Adaptivity)**：根据问题复杂度自动选择推理模式 (CoT/ToT/GoT/QoT)。
- **创新性 (Innovation)**：利用随机性和干涉算子突破局部最优，解决开放性问题。
- **经济性 (Efficiency)**：通过“探索 - 蒸馏 - 缓存”机制，将高成本的探索转化为低成本的确定性执行。
- **知识增强 (Knowledge-Augmented)**：实时引入外部知识库，通过纠缠机制锚定事实，减少幻觉。

---

## 2. 核心概念与术语映射

| 原概念 | 工程术语 | 定义 |
| :--- | :--- | :--- |
| **扩散** | **Branching Operator** | 从父节点生成多个子节点的生成操作。 |
| **演化** | **Scoring & Refinement** | 对节点内容进行质量评估 (0-1) 和微调。 |
| **干涉** | **Cross-Pollination** | 随机选取多个节点（同层或跨层），合成新节点。 |
| **坍缩** | **Probabilistic Pruning** | 基于动态阈值和概率分布，丢弃低分节点。 |
| **纠缠** | **Entanglement Link** | 显式记录两个节点间的强关联（内部逻辑或外部知识）。 |
| **势垒** | **Complexity Barrier** | 推理过程中的难点，需消耗更多资源跨越。 |
| **路径固化** | **Graph Distillation** | 从随机图中提取最优子图，固化为确定性模板。 |

---

## 3. 系统架构设计

系统采用 **Meta-Controller - Operator - Graph - Cache** 四层架构。

### 3.1 数据模型 (Data Models)

#### 3.1.1 ThoughtNode (思维节点)
```python
class ThoughtNode:
    id: str
    content: str
    depth: int               # 深度
    score: float             # 质量评分 [0.0, 1.0]
    survival_prob: float     # 当前生存概率
    parents: List[str]       # 父节点 ID (支持多父节点，DAG)
    children: List[str]      # 子节点 ID
    entangled_with: List[str] # 纠缠节点 ID 列表
    source_type: str         # 来源：LLM_GENERATED, RAG_ENHANCED, CROSS_POLLENATED
    metadata: Dict
```

#### 3.1.2 EntanglementLink (纠缠链)
```python
class EntanglementLink:
    id: str
    node_a: str              # 节点 A ID
    node_b: str              # 节点 B ID
    link_type: str           # INTERNAL_LOGIC, EXTERNAL_KNOWLEDGE
    relation_desc: str       # 关系描述 (自然语言)
    strength: float          # 纠缠强度 [0.0, 1.0]
    external_ref: str        # 若为外部知识，存储引用源 (如文档 ID)
```

#### 3.1.3 CognitiveGraph (认知图)
```python
class CognitiveGraph:
    nodes: Dict[str, ThoughtNode]
    links: Dict[str, EntanglementLink]  # 显式管理纠缠关系
    roots: List[str]
    active_leaves: List[str]
    history_snapshots: List[GraphState] # 用于回溯和蒸馏
```

#### 3.1.4 GoT_Template_Cache (GoT 模板缓存)
```python
class GoTTemplate:
    problem_signature: str   # 问题特征向量/哈希
    graph_structure: Dict    # 固化的图结构 (节点类型、连接方式)
    prompts: Dict            # 每个节点的最佳 Prompt 模板
    success_rate: float      # 历史成功率
    avg_cost: float          # 平均执行成本
```

---

## 4. 核心算子 (Core Operators)

### 4.1 Meta-Selector (元选择器)
**功能**：在推理开始前，分析问题类型，决定初始策略。
- **简单/确定性问题** → 选择 **CoT** (单链)
- **多解/博弈问题** → 选择 **ToT** (树形搜索)
- **复杂/多步依赖** → 选择 **GoT** (预定义图)
- **开放/创新/未知问题** → 选择 **QoT** (概率图探索)

### 4.2 Branching Operator (扩散)
- **输入**：父节点集合
- **逻辑**：调用 LLM 生成 $k$ 个子节点，$k \sim Poisson(\lambda_{branch})$
- **输出**：新节点集合，深度 = max(parent_depth) + 1

### 4.3 RAG Entanglement Operator (RAG 纠缠)
- **触发**：以概率 $P_{rag}$ 对活跃节点触发。
- **逻辑**：
  1. 提取节点关键词/向量。
  2. 检索外部知识库 (Top-K 文档)。
  3. 创建新节点或增强现有节点，并建立 `EXTERNAL_KNOWLEDGE` 类型的纠缠链。
- **作用**：引入事实锚点，纠正幻觉。

### 4.4 Internal Entanglement Operator (内部纠缠)
- **触发**：定期扫描图结构。
- **逻辑**：计算节点间语义相似度，若超过阈值且无直接父子关系，建立 `INTERNAL_LOGIC` 纠缠链。
- **作用**：发现隐含联系，为干涉做准备。

### 4.5 Cross-Pollination Operator (干涉)
- **输入**：随机采样 $m$ 个节点 (可跨层，可来自不同分支)。
- **逻辑**：
  1. 若采样到纠缠对，优先打包输入。
  2. 构造 Prompt："结合以下 $m$ 个思维片段，生成一个新的综合见解..."
  3. LLM 生成新节点，父节点指向这 $m$ 个节点。
- **输出**：融合节点 (DAG 中的跨层连接)。

### 4.6 Scoring & Refinement Operator (演化)
- **逻辑**：
  1. LLM 评分：$S_{llm} \in [0, 1]$
  2. 自我修正：若 $S_{llm} < 0.5$，尝试一次 Refine 生成新版本。
  3. 最终评分 $S_{final}$。

### 4.7 Probabilistic Pruning Operator (坍缩)
- **核心公式**：
  $$ P_{keep} = \sigma(\alpha \cdot S - \beta \cdot depth - \gamma) + \epsilon_{tunnel} $$
  - $\sigma$: Sigmoid 函数
  - $\alpha$: 评分敏感度
  - $\beta$: 深度衰减系数 (时间箭头)
  - $\gamma$: 基础阈值偏移
  - $\epsilon_{tunnel}$: **量子隧穿项**。若节点有强纠缠 ($strength > 0.8$)，增加额外生存概率，防止误杀重要线索。
- **执行**：对每个节点生成 $r \sim Uniform(0,1)$，若 $r > P_{keep}$ 则标记为 inactive。

### 4.8 Graph Distillation Operator (路径固化)
- **触发**：当 QoT 成功解决问题后。
- **逻辑**：
  1. **回溯**：从最终解节点反向追踪到根节点，保留所有贡献路径。
  2. **剪枝**：移除所有未贡献的旁支节点。
  3. **抽象**：将具体节点内容替换为 Prompt 模板变量。
  4. **缓存**：将生成的 `GoTTemplate` 存入缓存池，键为问题特征签名。
- **复用**：下次遇到相似问题时，直接加载模板，以确定性 GoT 模式执行，大幅降低成本。

---

## 5. 执行流程 (Execution Flow)

```python
def run_qot_system(problem):
    # 1. 模式选择
    mode = meta_selector.analyze(problem)
    
    if mode == "DETERMINISTIC":
        return execute_simple_cot(problem)
    
    # 2. 初始化图
    graph = CognitiveGraph()
    graph.add_root(problem)
    
    # 3. QoT 主循环 (最大迭代 T)
    for t in range(T):
        # A. 扩散
        if should_branch(graph):
            graph.apply(BranchingOperator())
        
        # B. RAG 纠缠 (外部知识注入)
        if random() < config.rag_probability:
            graph.apply(RAGEntanglementOperator(kb_client))
            
        # C. 内部纠缠 (发现联系)
        graph.apply(InternalEntanglementOperator())
        
        # D. 干涉 (跨层融合)
        if random() < config.interference_probability:
            graph.apply(CrossPollinationOperator())
            
        # E. 演化 (评分)
        graph.apply(ScoringOperator())
        
        # F. 坍缩 (剪枝)
        graph.apply(ProbabilisticPruningOperator(depth_decay=config.beta))
        
        # G. 终止检查
        if graph.has_solution():
            best_solution = graph.get_best_solution()
            
            # H. 路径固化 (关键步骤)
            template = distill_graph_to_template(graph, best_solution)
            cache.save(problem.signature, template)
            
            return best_solution
            
        if graph.is_empty():
            return "No solution found (All paths collapsed)"
            
    return "Timeout"
```

---

## 6. 优势分析：QoT vs CoT/ToT/GoT

| 特性 | CoT | ToT | GoT (Static) | **QoT (Proposed)** |
| :--- | :--- | :--- | :--- | :--- |
| **结构** | 线性链 | 树形 | 预定义 DAG | **动态 DAG + 纠缠网** |
| **探索性** | 低 (单一路径) | 中 (广度/深度优先) | 固定 | **极高 (概率扩散 + 干涉)** |
| **抗局部最优** | 弱 | 中 | 中 | **强 (量子隧穿 + 跨层跳跃)** |
| **知识融合** | 无 | 弱 | 需硬编码 | **原生支持 (RAG 纠缠)** |
| **复用性** | 低 | 低 | 高 | **极高 (自动蒸馏缓存)** |
| **适用场景** | 简单推理 | 游戏/搜索 | 固定流程任务 | **开放创新/复杂科研/未知问题** |
| **成本** | 低 | 中 | 中 | **高 (探索期) / 低 (复用期)** |

### 核心优势总结
1.  **动态适应性**：不再依赖人工设计图结构，系统自动“生长”出适合当前问题的拓扑。
2.  **涌现能力**：干涉算子允许“风马牛不相及”的节点碰撞，常能产生人类意想不到的创新解。
3.  **自我进化**：通过“路径固化”，系统越用越聪明，将昂贵的试错成本转化为永久资产。
4.  **事实锚定**：纠缠机制让外部知识不再是简单的 Context 拼接，而是成为图结构的一部分，参与推理演化。

---

## 7. 潜在挑战与对策

| 挑战 | 描述 | 对策 |
| :--- | :--- | :--- |
| **不可复现性** | 概率性导致相同输入可能产生不同输出 | 记录随机种子；对关键任务运行多次取最优；利用固化机制将成功路径确定性化。 |
| **资源爆炸** | 扩散过快导致节点数指数级增长 | 严格的深度衰减剪枝；设置最大节点总数上限；动态调整 $\lambda_{branch}$。 |
| **调试困难** | 复杂的跨层连接难以追踪错误源头 | 提供可视化图谱工具，高亮显示纠缠链和干涉路径；记录完整的血缘谱系。 |
| **延迟较高** | RAG 和多轮 LLM 调用耗时 | 异步并行执行算子；使用小模型进行预评分；缓存中间结果。 |

---

## 8. 实现路线图 (Roadmap)

### Phase 1: 基础骨架 (Core Engine)
- [ ] 实现 `ThoughtNode`, `CognitiveGraph` 数据结构。
- [ ] 实现 `Branching`, `Scoring`, `Pruning` 基础算子。
- [ ] 跑通纯概率搜索流程 (无 RAG, 无干涉)。

### Phase 2: 高级算子 (Advanced Operators)
- [ ] 实现 `Cross-Pollination` (干涉)。
- [ ] 实现 `EntanglementLink` 管理及内部纠缠发现。
- [ ] 引入深度衰减公式和量子隧穿逻辑。

### Phase 3: 知识集成 (RAG Integration)
- [ ] 接入向量数据库。
- [ ] 实现 `RAG Entanglement Operator`。
- [ ] 测试外部知识对评分和生存率的影响。

### Phase 4: 元控制与固化 (Meta & Distillation)
- [ ] 实现 `Meta-Selector` 模式切换。
- [ ] 开发 `Graph Distillation` 算法，提取最优子图。
- [ ] 构建 `GoT_Template_Cache` 及匹配检索机制。

### Phase 5: 可视化与优化 (UI & Opt)
- [ ] 开发 Web 可视化界面，动态展示图的生长、纠缠和坍缩。
- [ ] 超参数自动调优 (Auto-tuning $\alpha, \beta, \lambda$)。

---

## 9. 结语

QoT 系统不仅仅是一个推理框架，它是一个**认知进化引擎**。它模拟了人类创造性思维的“发散 - 联想 - 收敛”过程，并通过工程化的手段将其规模化、自动化。通过引入“路径固化”机制，它解决了概率性系统难以落地的痛点，实现了从“灵光一现”到“生产力工具”的闭环。
