# 投机推理方向（方向一）—— 科研实践项目完整方案

> **学习逻辑顺序**：论文研读（理论基础）→ vLLM 工程实验（真实框架验证）→ Agent 工具实践（自动化能力）→ 自主实践（深度创新）

---

## 一、项目概述

| 维度 | 内容 |
|------|------|
| **选定方向** | 方向一：端云投机加速 LLM 推理——异构并行与 DFlash 优化研究 |
| **硬件条件** | NVIDIA GeForce RTX 4050 笔记本 GPU，6GB 显存 |
| **总体周期** | 7月15日 – 7月31日（三阶段） |
| **最终产出** | ≤10分钟线上总结报告 PPT + 实验代码 + 实验数据 + 论文阅读笔记 |

---

## 二、四模块总览（按学习逻辑排序）

```
理论学习 ──→ 工程验证 ──→ 工具自动化 ──→ 深度实践创新
   ①           ②            ③             ④
论文研读    vLLM实验     Agent工具     myDraft_Verify
(4篇精读)   (4特性BB)   (MCP开发)     (自主框架)
```

| 序号 | 模块 | 定位 | 状态 |
|------|------|------|------|
| **①** | **论文研读** | 理论基础——系统掌握投机推理三路线全景 | ✅ 已完成 |
| **②** | **vLLM 特性性能实验** | 工程验证——在真实框架上测量 4 项特性效果 | 代码完成，待运行 |
| **③** | **Agent 工具实践** | 自动化——MCP 协议 + nanobot 工具/Skill 开发 | 代码完成，待运行 |
| **④** | **MyDraft_Verify 自主实践** | 深度创新——关联四篇论文的自主框架实现 | 待重建 |

---

## 三、模块①：论文研读（4篇指定论文）

### 3.1 四篇论文核心导读

| # | 论文 | 路线 | 关键创新 | 核心数据 | 对本实践的指导 |
|---|------|------|----------|----------|---------------|
| 1 | **Leviathan** (ICML 2023) | draft-model (序列) | draft-verify + rejection sampling 无损性证明 | α=0.62-0.88, T5-XXL 3.4× | 全部代码的理论基础 |
| 2 | **SpecInfer** (ASPLOS 2024) | draft-model (树形) | Token Tree + topology-aware causal mask | 验证率 57%→97% | 树形 draft 设计 + 批量验证 |
| 3 | **EAGLE-2** (2024) | self-speculative (动态树) | confidence ≈ acceptance rate + 动态 draft tree | 4.26×, +20-40% vs EAGLE-1 | 动态策略 + target feature 利用 |
| 4 | **DFlash** (ICML 2026) | block-diffusion drafter | 并行 draft + KV Injection | 6.08×, 2.5× vs EAGLE-3 | 并行 draft 哲学 + target context 注入 |

> 详细笔记见 [`paper_notes/`](../paper_notes/)，四篇对比总结见 [`paper_notes/summary.md`](../paper_notes/summary.md)

### 3.2 三条路线的演进关系

```
2023  Leviathan ──→ 线性序列，2-3×，奠基
2024  SpecInfer  ──→ 树形，验证率 57%→97%
2024  EAGLE-2    ──→ 动态树 + self-speculative，4.26×
2026  DFlash     ──→ 块并行扩散 draft，6×，新范式
         │
         └── 共同演进方向：串行 → 并行，静态 → 动态，独立 → 融合
```

---

## 四、模块②：vLLM 特性性能实验（必做）

### 4.1 实验目标

在真实推理框架 vLLM 上系统性测量 4 项核心特性，重点关注投机推理（方向一核心）。

### 4.2 必测特性

| 特性 | 启动参数 | 影响指标 | 与论文关联 |
|------|----------|----------|-----------|
| 前缀缓存 | `--enable-prefix-caching` | TTFT ↓, 吞吐量 ↑ | Leviathan: KV Cache 复用思想 |
| 分块预填充 | `--enable-chunked-prefill` | TTFT 波动 ↓ | 端侧异构: GPU 利用率优化 |
| 最大并发序列数 | `--max-num-seqs` | 吞吐量 ↑ | SpecInfer: 批量并行验证 |
| **投机推理** ⭐ | `--speculative-config` | TPOT ↓↓, 吞吐量 ↑↑ | 全部四篇论文 |

### 4.3 实验矩阵

| 实验组 | 配置 | 观测指标 |
|--------|------|----------|
| Group 0: 基线 | 所有特性关闭 | TTFT/TPOT/吞吐量基线 |
| Group 1: 前缀缓存 | `--enable-prefix-caching` | 单特性效果 |
| Group 2: 分块预填充 | `--enable-chunked-prefill` | 单特性效果 |
| Group 3: 不同并发度 | `--max-num-seqs` 8/16/32/64 | 并发-吞吐 tradeoff |
| **Group 4: 投机推理** ⭐ | `--speculative-config` | k=3/5/7 的加速比 |
| Group 5: 组合 | 投机+前缀缓存 | 特性叠加效果 |

> 完整实验方案见 [`vllm_benchmark/MANUAL.md`](../vllm_benchmark/MANUAL.md)

---

## 五、模块③：Agent 工具实践

### 5.1 任务清单

| 任务 | 产出 | 说明 |
|------|------|------|
| 任务一：自定义 MCP 工具 | `benchmark_analyzer.py` | 解析 vLLM results.json → Markdown 报告 |
| 任务二：自定义 Skill 工作流 | `speculative_experiment_runner/` | 5步自动化投机推理对比实验 |

### 5.2 MCP 工具设计

```
benchmark_analyzer 工具:
  输入: results.json 文件路径
  处理: 计算 TTFT/TPOT/吞吐量的均值/P50/P95
  输出: Markdown 格式汇总报告
  依赖: 无外部服务，纯文件解析
```

### 5.3 Skill 工作流

```
speculative_experiment_runner 工作流 (5步):
  Step 1: 验证配置文件
  Step 2: 基线实验 (启动 vLLM → Benchmark → 停止)
  Step 3: 投机推理实验 (启动 vLLM + speculative → Benchmark → 停止)
  Step 4: 调用 benchmark_analyzer 对比分析
  Step 5: 清理 → 输出报告
```

> 完整方案见 [`agent_tools/MANUAL.md`](../agent_tools/MANUAL.md)

---

## 六、模块④：MyDraft_Verify 自主实践（⭐ 核心创新模块）

### 6.1 设计哲学

不再是简单的 "实现 draft-verify 框架"，而是**以四篇论文为蓝本，构建一个可对比多种投机策略的统一实验平台**。

```
MyDraft_Verify 的统一框架:
                        ┌── 策略1: Sequential Draft (Leviathan)
  prompt → Context ──→ Draft Engine ──→ Verify Engine ──→ output
                        ├── 策略2: Tree Draft (SpecInfer)
                        ├── 策略3: Dynamic Draft (EAGLE-2 inspired)
                        └── 策略4: Pipeline Parallel (DFlash inspired)
```

### 6.2 四篇论文 → 四个策略 → 统一框架

| 策略 | 对应论文 | 核心机制 | 实现要点 |
|------|----------|----------|----------|
| **Strategy-A: Sequential** | Leviathan | 线性 draft-verify | 基线：标准 reject sampling |
| **Strategy-B: Tree** | SpecInfer | top-k 候选 + 树形验证 | topology-aware mask(python级简化) |
| **Strategy-C: Dynamic** | EAGLE-2 | 基于 confidence 动态调整树结构 | confidence ≈ accept rate 洞察 |
| **Strategy-D: Pipeline** | DFlash | 并行 draft + verify 流水线 | CUDA Stream 异步重叠 |

### 6.3 核心创新点（自主思考）

#### 创新①：Unified Strategy Selector（统一策略选择器）

```
根据上下文特征自动选择最优策略:

  if   entropy < low_threshold  → Strategy-A (Sequential, 最轻量)
  elif acceptance_rate > 0.8    → Strategy-A (简单, 高接受率时树形浪费)
  elif entropy > high_threshold → Strategy-B/C (Tree/Dynamic, 高不确定)
  elif k > 5                    → Strategy-D (Pipeline, 长序列并行友好)

  关键: 基于 EAGLE-2 的 insight——confidence ≈ acceptance rate
        基于 SpecInfer 的 insight——top-k 覆盖率极高
```

#### 创新②：Cross-Strategy Ablation Study（跨策略消融实验）

```
在同一 benchmark 上运行全部 4 个策略:
  - 同一 prompt 集 + 同一 target/draft model
  - 测量: 加速比 / 接受率 / draft耗时 / 显存峰值
  - 输出: 四策略对比雷达图 + 策略选择建议矩阵
```

#### 创新③：Layer-wise Target Context (DFlash-inspired)

```
受 DFlash KV Injection 启发——让 draft model 利用 target model 的中间层特征:

  实现: 提取 target model 中间层的 hidden states
       → 投影到 draft model 的 embedding 空间
       → 拼接或加权注入 draft model 的输入
       → 评估: 接受率提升 vs 额外开销
  
  简化版 (RTX 4050 友好):
      仅取 target 第 N/2 层 hidden state
      → 线性投影 → 加到 draft embedding 上
      → 开销: 仅增加一个投影矩阵
```

#### 创新④：Ablation Leaderboard（消融排行榜）

```
┌──────────────────────────────────────────────────────┐
│                  Ablation Leaderboard                  │
├──────────┬──────────┬──────────┬──────────┬──────────┤
│ 策略      │ 加速比    │ 接受率    │ Draft耗时% │ 推荐场景  │
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ Seq(基线) │  1.00×   │  0.65    │   35%    │ 基准对照  │
│ Tree     │  1.35×   │  0.78    │   40%    │ 高不确定  │
│ Dynamic  │  1.42×   │  0.80    │   32%    │ 通用最优  │
│ Pipeline │  1.52×   │  0.65    │   18%    │ k>5 场景  │
│ Unified  │  1.58×   │  0.77    │   28%    │ 自动切换  │
├──────────┴──────────┴──────────┴──────────┴──────────┤
│ ★ 最佳组合: Unified Selector + Dynamic Tree           │
│ ★ 最大发现: Pipeline 将 draft 耗时% 从 35%→18%        │
└──────────────────────────────────────────────────────┘
```

### 6.4 新目录结构

```
myDraft_Verify/
├── MANUAL.md                       ← 本模块详细手册
├── core/                           ← 核心引擎
│   ├── __init__.py
│   ├── engine.py                   ← 统一投机推理引擎
│   ├── draft.py                    ← Draft Engine (4种策略)
│   ├── verify.py                   ← Verify Engine (序列/树形)
│   └── strategy.py                 ← Strategy Selector (自动策略选择)
├── models/                         ← 模型封装
│   ├── __init__.py
│   ├── loader.py                   ← 模型加载 + KV Cache 管理
│   └── context.py                  ← Target Context 提取 (DFlash-inspired)
├── experiments/                    ← 实验运行
│   ├── __init__.py
│   ├── runner.py                   ← 实验运行器 (网格搜索)
│   ├── prompts.py                  ← 测试 Prompt 集
│   └── benchmark.py                ← Benchmark 工具
├── analysis/                       ← 结果分析
│   ├── __init__.py
│   ├── metrics.py                  ← 指标计算
│   ├── plot.py                     ← 可视化 (雷达图/消融表)
│   └── report.py                   ← 报告生成
├── tests/                          ← 单元测试
│   ├── __init__.py
│   └── test_strategies.py
├── results/                        ← 结果输出 (自动创建)
├── requirements.txt
└── run.py                          ← 一键运行入口
```

### 6.5 论文关联矩阵

```
                    Leviathan  SpecInfer  EAGLE-2  DFlash
core/engine.py         ✓           ✓         ✓        ✓
core/draft.py          ✓           ✓         ✓        ✓
core/verify.py         ✓           ✓         —        —
core/strategy.py       —           ✓         ✓        ✓
models/context.py      —           —         ✓        ✓
experiments/runner.py  ✓           ✓         ✓        ✓
analysis/plot.py       ✓           ✓         ✓        ✓
```

### 6.6 实验设计

#### 实验1：四策略独立对比

| 自变量 | 取值 |
|--------|------|
| 策略 | Sequential / Tree / Dynamic / Pipeline |
| draft length k | 3, 5, 7, 10 |
| temperature | 0.0, 0.6, 1.0 |

#### 实验2：动态策略 vs 静态策略

| 对比组 | 说明 |
|--------|------|
| Fixed k=5 Sequential | 基准 |
| Tree (top-2, depth=3) | SpecInfer 风格 |
| Dynamic (confidence-based) | EAGLE-2 风格 |
| **Unified Selector** | 自适应（创新①） |

#### 实验3：Target Context 注入效果（创新③）

| 配置 | 说明 |
|------|------|
| Draft only (无 context) | 基线 |
| Draft + Target Layer N/2 | 单层特征注入 |
| Draft + Target Layer N/4, N/2, 3N/4 | 多层特征融合 |

#### 实验4：Pipeline 重叠率分析

| 配置 | 测量 |
|------|------|
| 串行 | draft/verify 无重叠 |
| 线程级流水线 | 重叠率 (overlap ratio) |
| CUDA Stream 流水线 | 实际并行效率 |

### 6.7 核心 API 设计

```python
from myDraft_Verify.core import SpecEngine, DraftStrategy

# 创建引擎
engine = SpecEngine(
    target_model="Qwen/Qwen2.5-1.5B-Instruct",
    draft_model="Qwen/Qwen2.5-0.5B-Instruct",
)

# 策略 A: Sequential (Leviathan)
result = engine.generate(prompt, strategy=DraftStrategy.SEQUENTIAL, k=5)

# 策略 B: Tree (SpecInfer)
result = engine.generate(prompt, strategy=DraftStrategy.TREE, k=5, top_k=2)

# 策略 C: Dynamic (EAGLE-2)
result = engine.generate(prompt, strategy=DraftStrategy.DYNAMIC, k_max=10)

# 策略 D: Pipeline (DFlash-inspired)
result = engine.generate(prompt, strategy=DraftStrategy.PIPELINE, k=5)

# 策略 E: Unified (Auto-select, 创新)
result = engine.generate(prompt, strategy=DraftStrategy.UNIFIED)
```

### 6.8 与旧版 myDraft_Verify 的对比

| 维度 | 旧版 | 新版 |
|------|------|------|
| 架构 | 单文件 (~280行) | 模块化 (core/models/experiments/analysis) |
| 论文关联 | 仅 Leviathan | 四篇论文均有对应实现 |
| Draft 策略 | 仅 Sequential | Sequential / Tree / Dynamic / Pipeline / Unified |
| 创新点 | 无 | 4个：Unified Selector / Layer-wise Context / Ablation Leaderboard / Cross-Strategy Study |
| 实验设计 | 基础网格搜索 | 4组对比实验 + 消融分析 |
| 代码行数 | ~600 | ~1200+ (更结构化) |
| 可复现性 | 基本 | 完善的单元测试 + 结果版本记录 |

---

## 七、四模块汇报逻辑

```
PPT 叙事线 (≤10 min):

① 论文研读 (1.5 min)
   └── 三条路线全景 → 引出端侧瓶颈问题

② vLLM 工程实验 (2.5 min)
   └── 4 项特性实测效果 → 投机推理工程加速比

③ Agent 工具实践 (1 min)
   └── MCP 工具 + 实验自动化 Skill

④ MyDraft_Verify 自主实践 (3.5 min) — 核心
   ├── 四策略统一框架设计理念
   ├── 跨策略消融实验 + 排行榜
   ├── 创新点: Unified Selector / Target Context
   └── 关键发现: 端侧瓶颈量化分析

⑤ 总结与展望 (1.5 min)
   └── 端侧异构并行的可行路径
```

---

## 八、环境与依赖

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6GB) |
| 操作系统 | **WSL2 (Ubuntu 22.04)** ← 参见 [WSL2_SETUP.md](../WSL2_SETUP.md) |
| CUDA | 12.1+ (WSL2 原生) |
| Python | 3.10 (conda: spec_dec) |
| PyTorch | ≥2.1.0 |
| transformers | ≥4.40.0 |
| vLLM | ≥0.5.0 (WSL2 原生支持) |
| Target Model | Qwen2.5-1.5B-Instruct |
| Draft Model | Qwen2.5-0.5B-Instruct |
| **备选 (Windows)** | hf_server.py 替代 vLLM |

---

## 九、预期产出清单

| 编号 | 产出物 | 模块 | 说明 |
|------|--------|------|------|
| 1 | 论文阅读笔记 ×4 + 对比总结 | ① | 真实论文数据填充 |
| 2 | vLLM 实验数据 + 对比报告 | ② | 4 特性 × 多并发度 |
| 3 | benchmark_analyzer 工具 | ③ | MCP 协议兼容 |
| 4 | speculative_experiment_runner Skill | ③ | 5步自动化工作流 |
| 5 | MyDraft_Verify 统一框架 | ④ | 4策略 + 4创新 |
| 6 | 跨策略消融实验数据 + Leaderboard | ④ | 4组对比实验 |
| 7 | summary_report.pptx | 全部 | ≤10分钟 PPT |
