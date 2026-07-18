# 投机推理方向（方向一）—— 科研实践项目完整方案

---

## 一、项目概述

| 维度 | 内容 |
|------|------|
| **选定方向** | 方向一：端云投机加速 LLM 推理——异构并行与 DFlash 优化研究 |
| **硬件条件** | NVIDIA GeForce RTX 4050 笔记本 GPU，6GB 显存 |
| **总体周期** | 7月15日 – 7月31日（三阶段） |
| **最终产出** | ≤10分钟线上总结报告 PPT + 实验代码 + 实验数据 + 论文阅读笔记 |

---

## 二、总体任务清单（来自 `暑期科研论文及实践.md`）

方向一共包含三大模块，均需完成：

| 模块 | 内容 | 状态 |
|------|------|------|
| **一、论文研读** | 精读 4 篇投机推理方向论文 | 待完成 |
| **二、vLLM 特性性能实验（必做）** | 4 项 vLLM 特性的 Benchmark | 待完成 |
| **三、Agent 工具实践** | 部署 nanobot + MCP 协议开发 | 待完成 |
| **四、手写投机推理框架（自主深化）** | 方案B：从零实现 draft-verify | ✅ 代码已完成 |

---

## 三、论文研读（4 篇指定论文）

### 3.1 必读论文清单

| 序号 | 论文 | 核心方向 | 阅读重点 |
|------|------|----------|----------|
| 1 | **Fast Inference from Transformers via Speculative Decoding** (Leviathan et al., ICML 2023) | 投机推理奠基 | draft-verify 框架、接受率理论、rejection sampling 无损性证明 |
| 2 | **SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification** (Miao et al., 2024) | 树形投机推理 | token tree verifier、树形 draft 构建、批量验证机制 |
| 3 | **EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees** (2024) | Self-Speculative | feature-level 草稿生成、动态 draft tree、无需独立 draft model |
| 4 | **DFlash: Block Diffusion for Flash Speculative Decoding** (2024) | 块并行解码 | 打破因果 mask、块并行生成、接受率波动问题、训练成本 |

### 3.2 论文阅读笔记模板

每篇论文按以下结构整理：

```
论文标题: xxx
作者/会议: xxx
核心问题: (一句话描述要解决什么问题)
核心方法: (用 3-5 句话描述方法的关键思路)
关键结果: (列出 2-3 个最重要的实验发现/数据)
与方向一的关联: (这篇论文对端侧投机推理有何启发?)
个人思考: (哪些设计可以借鉴到自己的实践中?)
```

### 3.3 论文关联关系

```
Leviathan (论文1) ──→ 奠基：draft-verify 框架 + 接受率理论
    ├── SpecInfer (论文2) ──→ 改进 draft：树形结构 → 提升接受率
    ├── EAGLE-2 (论文3) ──→ 改进 draft：feature-level → 更准确的草稿
    └── DFlash (论文4) ──→ 替代方案：块并行 → 一次生成多个 token
```

---

## 四、vLLM 特性性能实验（必做）

### 4.1 实验目标

使用 vLLM 推理框架，系统性测量 4 项核心特性对 LLM 推理性能的影响，重点关注投机推理（方向一核心特性）。

### 4.2 必测特性与启动参数

| 特性 | 启动参数 | 作用机制 | 影响指标 |
|------|----------|----------|----------|
| **前缀缓存** | `--enable-prefix-caching` | 复用相同前缀的 KV Cache，减少重复计算 | TTFT ↓, 吞吐量 ↑ |
| **分块预填充** | `--enable-chunked-prefill` | 将长 prompt 的 prefill 阶段切分为多块，提升 GPU 利用率 | TTFT 波动 ↓, 并发能力 ↑ |
| **最大并发序列数** | `--max-num-seqs` | 控制同时处理的请求数量上限 | 吞吐量 ↑, TPOT ↑ (tradeoff) |
| **投机推理** ⭐ | `--speculative-config` | draft model 预猜 + target model 并行验证 | TPOT ↓, 吞吐量 ↑ |

> ⭐ = 方向一核心特性，需重点分析

### 4.3 核心观测指标

| 指标 | 全称 | 含义 | 采集方式 |
|------|------|------|----------|
| **TTFT** | Time To First Token | 从请求发出到第一个 token 生成的时间 | vLLM `metrics` 端点 |
| **TPOT** | Time Per Output Token | 除首个 token 外，每个后续 token 的平均生成时间 | vLLM `metrics` 端点 |
| **吞吐量** | Throughput | 单位时间内系统处理的总 token 数 | `tokens/s` 汇总 |

### 4.4 实验设计方案

#### 环境配置

```
GPU:      NVIDIA GeForce RTX 4050 (6GB)
vLLM:     >= 0.5.0
模型:     Qwen2.5-1.5B-Instruct  (target)
          Qwen2.5-0.5B-Instruct  (draft, 投机推理用)
Python:   >= 3.10
```

#### 基线设置

| 配置项 | 基线值 |
|--------|--------|
| 模型 | Qwen2.5-1.5B-Instruct |
| 并发数 | 1 |
| 最大序列长度 | 2048 |
| 所有特性关闭 | 基线 (Baseline) |

#### 测试数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| ShareGPT 数据集 | 通用对话场景测试 | 多轮对话、不同序列长度 |
| 自建 prefix-heavy prompts | 前缀缓存测试 | 共享长前缀的多个请求 |
| 自建 mixed-length prompts | 分块预填充测试 | 短/中/长 prompt 混合 |

#### 实验矩阵

```
实验组设计（每个实验组独立运行 3 次取均值）:

Group 0: 基线 (所有特性关闭)
Group 1: --enable-prefix-caching                          (单特性: 前缀缓存)
Group 2: --enable-chunked-prefill                         (单特性: 分块预填充)
Group 3: --max-num-seqs {8, 16, 32, 64}                  (单特性: 不同并发度)
Group 4: --speculative-config {...}                       (单特性: 投机推理) ⭐
Group 5: --speculative-config {...} + --enable-prefix-caching (组合: 投机+前缀)
Group 6: 全特性组合                                        (All features)
```

#### 投机推理配置细节（Group 4, 5 核心）

```bash
# vLLM 投机推理启动命令示例
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-config '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 5,
        "method": "draft_model"
    }' \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048
```

| 投机推理参数 | 测试值 | 说明 |
|-------------|--------|------|
| `num_speculative_tokens` (k) | 3, 5, 7 | 不同 draft length 的效果 |
| `method` | `draft_model` | 使用独立小模型做 draft |

#### 特性-指标关联分析框架

| 特性 | TTFT | TPOT | 吞吐量 | 影响机制 |
|------|------|------|--------|----------|
| 前缀缓存 | ↓↓ (显著降低) | — | ↑ | 共享前缀的请求跳过重复 KV 计算 |
| 分块预填充 | ↓ (波动减小) | — | ↑↑ | 短 prompt 不再被长 prompt 阻塞 |
| 最大并发序列数 | ↑ (排队增加) | ↑ | ↑↑ | 更多请求并发 → GPU 利用率上升 |
| 投机推理 ⭐ | — | ↓↓ (显著降低) | ↑↑ | draft-verify 减少 serial forward 次数 |

### 4.5 实验汇报清单

按 `暑期科研论文及实践.md` 要求，最终汇报需覆盖：

1. **环境配置**：显卡型号 (RTX 4050 6GB)、vLLM 版本、模型选型
2. **实验设计**：实验方案逻辑（单特性 → 组合）、基线设置、测试数据集说明
3. **特性-指标关联**：分析各特性分别影响哪些性能指标，为什么（影响机制）

---

## 五、手写投机推理框架（方案B，✅ 代码已完成）

> 代码位置: [`myDraft_Verify/`](myDraft_Verify/speculative_decoding.py)

### 5.1 目标

从零实现完整 draft-verify 循环，深入理解投机推理机制。与 vLLM 的投机推理实验形成"原理验证 + 工程应用"的双层实验结构。

### 5.2 核心算法流程

```
while len(generated_tokens) < max_new_tokens:
    Step 1: DRAFT  — draft_model 自回归生成 k 个 token (复用 KV Cache)
    Step 2: VERIFY — target_model 一次并行前向验证全部 draft tokens
    Step 3: ACCEPT/REJECT — Rejection Sampling (保证无损)
    Step 4: 拼接已接受的 token，继续下一轮
```

### 5.3 三种采样策略

| 策略 | 说明 | 质量保证 |
|------|------|----------|
| Greedy | draft & target 均取 argmax | 无损 |
| Standard Rejection Sampling | 完整 rejection sampling | 严格无损（分布等价） |
| Entropy-Adaptive | 标准采样 + 基于接受率动态调整 k | 严格无损 |

### 5.4 实验变量

| 自变量 | 取值 |
|--------|------|
| draft length k | 3, 5, 7, 10 |
| temperature | 0.0, 0.6, 0.8, 1.0 |
| 采样策略 | greedy / standard / entropy-adaptive |
| KV Cache | 无 / 标准 / 增量复用 |

### 5.5 已实现的优化点

| 编号 | 优化点 | 难度 | 预期收益 |
|------|--------|------|----------|
| ① | KV Cache 增量复用 | ⭐ | Draft 加速 40-60% |
| ② | Draft 长度动态调整 | ⭐⭐ | 减少浪费 15-30% |
| ③ | Tree-Structured Draft | ⭐⭐ | 接受率 +10-20% |
| ④ | Draft 量化加速 (INT8) | ⭐⭐ | Draft 速度 +30-50% |
| ⑤ | Pipeline 并行（模拟异构） | ⭐⭐⭐ | 端到端 +20-40% |
| ⑥ | Tree Attention 批量验证 | ⭐⭐⭐ | 验证效率 ×2-3 |

---

## 六、Agent 工具实践（nanobot + MCP）

### 6.1 目标

部署 nanobot 开源 Agent 框架，基于 MCP 协议完成自定义扩展开发。

### 6.2 环境搭建

```bash
# 克隆 nanobot 仓库
git clone https://github.com/nanobot-project/nanobot.git
cd nanobot

# 安装依赖
pip install -r requirements.txt

# 验证部署成功
python -m nanobot.cli --help
```

### 6.3 任务一：新增 1 个自定义工具

**工具设计**：为 nanobot 添加一个与 LLM 推理性能分析相关的工具。

```
工具名称: benchmark_analyzer
功能描述: 读取 vLLM 实验输出的 metrics 数据（JSON/CSV），
         自动分析 TTFT、TPOT、吞吐量并生成摘要报告
输入参数:
  - metrics_file: str (metrics 数据文件路径)
  - format: str ("json" | "csv")
输出: 包含各指标统计值的结构化摘要
```

**实现要点**：

1. 在 nanobot 的 `tools/` 目录下新增 `benchmark_analyzer.py`
2. 实现 MCP Tool 接口：`name`, `description`, `input_schema`, `execute()`
3. 注册到 nanobot 工具链中
4. 测试：用 vLLM 实验产出的真实 metrics 数据验证

### 6.4 任务二：新增 1 个 Skill 工作流

**Skill 设计**：创建一个端到端的"投机推理实验自动化"工作流。

```
Skill 名称: speculative_experiment_runner
功能描述: 自动化执行投机推理对比实验流程
工作流步骤:
  1. 加载配置 (基线参数 vs 投机推理参数)
  2. 依次启动 vLLM 服务 (基线模式 → 投机推理模式)
  3. 运行 Benchmark 脚本 (使用 benchmark_analyzer 工具)
  4. 收集 metrics (TTFT, TPOT, 吞吐量)
  5. 生成对比报告 (Markdown 表格 + 加速比计算)
  6. 清理并输出结果
```

**实现要点**：

1. 在 nanobot 的 `skills/` 目录下新增 `speculative_experiment_runner/`
2. 实现 skill 定义文件：`skill.yaml` (名称、描述、工具依赖)
3. 实现工作流编排逻辑 (`workflow.py`)
4. 确保每个步骤有错误处理和日志

### 6.5 MCP 协议对接

```
┌──────────────────────────────────────────┐
│  nanobot Agent                            │
│    ├── tools/benchmark_analyzer.py        │  ← 自定义工具 (任务一)
│    └── skills/speculative_experiment_runner/│  ← 自定义技能 (任务二)
│           ├── skill.yaml                  │
│           └── workflow.py                 │
├──────────────────────────────────────────┤
│  MCP Protocol (Model Context Protocol)    │
│    ├── Tool Discovery                     │
│    ├── Tool Execution                     │
│    └── Result Return                      │
├──────────────────────────────────────────┤
│  External Resources                       │
│    ├── vLLM Server (metrics endpoint)     │
│    ├── Benchmark Scripts                  │
│    └── Result Storage                     │
└──────────────────────────────────────────┘
```

---

## 七、四个模块间的关系与汇报逻辑

```
总结报告叙事线:

  ┌─────────────────────────────────────────────────────────────┐
  │ 论文研读 (4篇)                                                │
  │   ├── 掌握投机推理理论全貌 (Leviathan → SpecInfer → EAGLE-2 → DFlash) │
  │   └── 为实验提供理论基础                                      │
  ├─────────────────────────────────────────────────────────────┤
  │ 手写框架 (方案B)                                              │
  │   ├── 从零实现 draft-verify → 深入理解算法细节                │
  │   ├── 在 RTX 4050 上测量接受率/加速比/瓶颈                    │
  │   └── 证明: draft model 在端侧成为瓶颈 → 引出异构并行         │
  ├─────────────────────────────────────────────────────────────┤
  │ vLLM 特性实验 (必做)                                          │
  │   ├── 在真实推理框架上验证 4 项特性的工程效果                  │
  │   ├── 重点: 投机推理在 vLLM 中的端到端加速比                  │
  │   └── 对比手写框架 vs vLLM 投机推理的异同                     │
  ├─────────────────────────────────────────────────────────────┤
  │ Agent 工具实践                                                │
  │   ├── 将实验流程自动化 (benchmark_analyzer 工具)              │
  │   ├── 构建可复用的实验工作流 (speculative_experiment_runner)  │
  │   └── 体现工程化思维 + MCP 协议应用能力                       │
  └─────────────────────────────────────────────────────────────┘
```

---

## 八、PPT 汇报建议框架（≤10 分钟）

```
1. 研究背景 (1.5 min)
   ├── LLM 推理时延问题
   └── 投机推理基本原理 (draft → verify → accept/reject)

2. 论文研读综述 (2 min)
   ├── 4篇论文核心贡献速览
   └── 三条路线的设计哲学对比 (draft-model / self-speculative / 块并行)

3. 系统实践 (4 min) — 核心
   ├── 手写 draft-verify 框架：实现要点 + RTX 4050 实验数据
   │   └── 关键发现：draft 耗时占比 + 最佳 draft length
   ├── vLLM 特性实验：4 项特性性能对比
   │   └── 投机推理 vs 基线的加速比 (TTFT/TPOT/吞吐量)
   └── 特性-指标关联分析

4. Agent 工具实践 (1 min)
   ├── nanobot 部署 + MCP 工具开发
   └── 自动化实验工作流

5. 总结与展望 (1.5 min)
   ├── 核心发现：端侧 draft 成为瓶颈 → 异构并行
   └── 后续工作方向
```

---

## 九、环境与依赖

### 硬件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| 显存 | 6GB GDDR6 |
| RAM | ≥16GB |

### 软件环境 (Conda)

```bash
conda create -n spec_dec python=3.10 -y
conda activate spec_dec

# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 手写框架
pip install transformers>=4.40.0 accelerate>=0.28.0 matplotlib>=3.7.0

# vLLM 实验
pip install vllm>=0.5.0

# Agent 工具实践
pip install mcp httpx
git clone https://github.com/nanobot-project/nanobot.git
cd nanobot && pip install -r requirements.txt
```

---

## 十、预期产出清单

| 编号 | 产出物 | 模块 | 说明 |
|------|--------|------|------|
| 1 | 论文阅读笔记 ×4 | 论文研读 | 每篇论文按模板整理 |
| 2 | `speculative_decoding.py` | 手写框架 | ✅ 已完成，~280行 |
| 3 | `run_experiments.py` | 手写框架 | ✅ 已完成，实验运行器 |
| 4 | `experiment_results.csv` | 手写框架 | 接受率/加速比原始数据 |
| 5 | `vllm_benchmark/` | vLLM 实验 | vLLM 启动脚本 + Benchmark 脚本 + 原始数据 |
| 6 | `vllm_analysis.md` | vLLM 实验 | 4 项特性性能对比 + 特性-指标关联分析 |
| 7 | `tools/benchmark_analyzer.py` | Agent 实践 | nanobot 自定义工具 |
| 8 | `skills/speculative_experiment_runner/` | Agent 实践 | nanobot 自定义 Skill |
| 9 | `summary_report.pptx` | 总结报告 | ≤10分钟线上汇报 PPT |
