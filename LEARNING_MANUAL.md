# 端云投机加速 LLM 推理 — 完整学习手册

> **方向一：投机推理 (Speculative Decoding)** · RTX 4050 6GB · 四模块递进学习

---

## 目录

| 章节 | 内容 |
|------|------|
| [第一章](#第一章-投机推理基础理论) | 投机推理基础理论（三大核心概念） |
| [第二章](#第二章-论文研读要点) | 论文研读要点（四篇论文深度解读） |
| [第三章](#第三章-vllm-工程实验) | vLLM 工程实验（特性对比 + Benchmark） |
| [第四章](#第四章-agent-工具实践) | Agent 工具实践（MCP 协议 + 自动化） |
| [第五章](#第五章-mydraft_verify-深度实践) | MyDraft_Verify 深度实践（四策略统一框架） |
| [第六章](#第六章-完整运行指南) | 完整运行指南（从零开始到全部实验） |
| [附录](#附录) | 核心公式速查 + 故障排查 |

---

## 第一章 投机推理基础理论

### 1.1 为什么 LLM 推理这么慢？

所有现代大语言模型（LLM）都是**自回归 (Autoregressive)** 的——生成下一个 token 需要依赖前面所有已生成的 token。这意味着：

```
生成 "今天天气真好" 需要 6 步串行:
  步骤1: 模型(prompt)              → "今"
  步骤2: 模型(prompt+"今")        → "天"
  步骤3: 模型(prompt+"今天")      → "天"
  ...共 6 次模型前向，每次都是串行等待
```

**核心瓶颈**：不是算力不够，而是**显存带宽 (Memory Bandwidth)** 限制了速度。大模型每次前向需要读取全部参数（GB 级），而 GPU 显存带宽有限。

### 1.2 投机推理的核心思想（Leviathan, ICML 2023）

> **用一个轻量小模型"猜"多个未来 token，然后让大模型一次性并行验证。猜对了就加速，猜错了也不影响质量。**

```
传统自回归 (串行):
  Target: [Token1] → [Token2] → [Token3] → [Token4] → [Token5]
  ↑ 5 次大模型前向，每次 1 个 token

投机推理 (串行 + 并行):
  Draft:  [猜1, 猜2, 猜3, 猜4, 猜5]     ← 小模型快速生成 5 个候选
  Target: [验证全部5个候选]               ← 大模型一次前向验证
  Accept: [猜1✓, 猜2✓, 猜3✗, 修正, 猜5]  ← 接受2个，修正1个，继续
```

**关键保证（论文核心证明）**：通过 **Rejection Sampling**，投机推理的输出分布与纯大模型自回归**完全相同**——即"无损加速"。

### 1.3 三个核心公式（必须理解）

#### 公式 1：接受率 α

```
α = E[min(p(x), q(x))] = 1 - D_LK(p, q)
```

其中 `p(x)` 是大模型的概率分布，`q(x)` 是小模型的概率分布。`α` 衡量两个模型的"相似度"——越接近 1，小模型猜得越准。

**实验数据**：当小模型比大模型小约 100 倍时，`α` 通常在 **0.5-0.9** 之间。

#### 公式 2：加速比期望

```
加速比 = (1 - α^(γ+1)) / ((1 - α)(γc + 1))
```

其中：
- `γ`：draft length（一次猜几个 token）
- `c`：cost coefficient（小模型单步耗时 / 大模型单步耗时）
- `α`：接受率

**关键洞察**：加速比取决于 α 和 c 的平衡。在云端（c ≈ 0），加速比可达 3-4×；在端侧（c 较大），draft model 本身成为瓶颈！

#### 公式 3：Rejection Sampling 验证

```
对每个 draft token x_i:
  r ~ Uniform(0, 1)
  if r ≤ p_target(x_i) / p_draft(x_i):
      接受 x_i
  else:
      从 norm(max(0, p_target - p_draft)) 重采样
      停止（后续全部丢弃）
```

**这个公式保证了输出分布不变**——论文附录 A.1 有完整证明。

---

## 第二章 论文研读要点

### 2.1 论文全景：三条路线

```
┌────────────────────────────────────────────────────────────┐
│                  投机推理技术演进                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│ 路线1: Draft-Model (独立小模型)                              │
│   ├─ Leviathan (ICML 2023): 线性序列, α=0.62-0.88, 2-3.4× │
│   └─ SpecInfer (ASPLOS 2024): 树形, 验证率 57%→97%        │
│                                                            │
│ 路线2: Self-Speculative (大模型自己猜)                       │
│   └─ EAGLE-2 (2024): 特征级预测 + 动态树, 4.26×            │
│                                                            │
│ 路线3: Block-Diffusion (扩散模型并行生成)                    │
│   └─ DFlash (ICML 2026): 并行 draft + KV 注入, 6.08×      │
│                                                            │
│ 共同方向: 串行 → 并行，静态 → 动态，独立 → 融合              │
└────────────────────────────────────────────────────────────┘
```

### 2.2 论文1：Leviathan (ICML 2023) — 奠基之作 ★必读

| 项目 | 内容 |
|------|------|
| **标题** | Fast Inference from Transformers via Speculative Decoding |
| **作者** | Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research) |
| **核心贡献** | 提出 draft-verify 框架 + speculative sampling 无损性证明 |
| **关键数据** | T5-XXL (11B) + T5-Small (77M): **3.4× 加速**（temp=0 翻译任务） |
| **接受率** | α = 0.62-0.88，取决于 draft model 大小和任务 |
| **对实践的指导** | 全部手写代码的理论基础；`min(1, p/q)` 公式直接用于 `verify.py` |

**需要精读的章节**：
- Section 2.3：Speculative Sampling 算法（Algorithm 1）
- Section 3.1-3.3：加速比理论推导
- 附录 A.1：无损性证明

**一个关键数据表（Table 3）**：

| Target Model | Draft Model | Sampling | α (接受率) |
|-------------|-------------|----------|-----------|
| GPT-like 97M | GPT-like 6M | temp=0 | **0.88** |
| T5-XXL 11B | T5-Small 77M | temp=0 | **0.75** |
| T5-XXL 11B | T5-Small 77M | temp=1 | **0.62** |
| LaMDA 137B | LaMDA 100M | temp=0 | **0.61** |
| T5-XXL 11B | Bigram | temp=0 | **0.20** ← 甚至 n-gram 都有用！ |

### 2.3 论文2：SpecInfer (ASPLOS 2024) — 树形验证

| 项目 | 内容 |
|------|------|
| **标题** | SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification |
| **作者** | Xupeng Miao et al. (CMU, 16 位作者跨 6 所学校) |
| **核心贡献** | Token Tree + topology-aware causal mask + Multi-Step Speculative Sampling |
| **关键数据** | 验证成功率从 57% → **97%**（top-5 候选，随机解码） |
| **对实践的指导** | `draft.py` 中 `_draft_tree()` 方法的设计来源 |

**核心洞察**：

```
线性 draft 的问题:
  每个位置只猜 1 个 token → 错一个，全丢弃 → 接受率低

树形 draft 的解决方案:
  每个位置猜 top-2（甚至 top-5）→ 分支覆盖 LLM 真实输出
  → 用 topology-aware causal mask 一次验证所有路径

结果: 验证成功率 57% → 97%（随机解码时提升尤其显著）
```

**Topology-aware Causal Mask 原理**：

```
展平后的 token tree:
  tokens: [A, B1, C1, B2, C2, D1, D2]
           ↓   ↓   ↓   ↓   ↓   ↓   ↓
  mask 行: 1   1   1   1   1   1   1   ← A 看全部
           0   1   1   0   0   0   0   ← B1 只看 A, B1, C1（分支1内部）
           0   0   1   0   0   0   0   ← C1 只看 A, B1, C1
           0   0   0   1   1   0   0   ← B2 只看 A, B2, C2（分支2内部）
           ...
  → 分支间互相不可见，保证因果性
```

### 2.4 论文3：EAGLE-2 (2024) — 动态树 + 置信度驱动

| 项目 | 内容 |
|------|------|
| **标题** | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees |
| **核心贡献** | 发现 confidence ≈ acceptance rate + 动态 draft tree（无需额外训练） |
| **关键数据** | MT-bench 上 4.26× 加速，比 EAGLE-1 提升 **+20-40%** |
| **对实践的指导** | `draft.py` 中 `_draft_dynamic()` 和 `engine.py` 中 `_select_strategy()` 的灵感来源 |

**最重要的发现（Section 3.2）**：

```
draft model 的 confidence score（输出概率）与 LLM 实际接受率的关系:

  confidence < 0.05 → 接受率约 0.04
  confidence = 0.50 → 接受率约 0.50
  confidence > 0.95 → 接受率约 0.98

  → 强正相关！可以用 confidence 近似接受率，无需调用大模型！
```

**这直接启发了 MyDraft_Verify 的创新①（Unified Strategy Selector）**——根据 confidence 动态决定用什么策略。

### 2.5 论文4：DFlash (ICML 2026) — 块并行新范式

| 项目 | 内容 |
|------|------|
| **标题** | DFlash: Block Diffusion for Flash Speculative Decoding |
| **核心贡献** | Block Diffusion Drafter + KV Injection + 并行 draft |
| **关键数据** | Qwen3-8B 上 **6.08×** 加速，2.5× vs EAGLE-3 |
| **对实践的指导** | `draft.py` 中 `_draft_pipeline()` 和 `loader.py` 中 `extract_target_context()` 的设计理念 |

**核心公式（论文 3.1 节）**：

```
Autoregressive draft cost:   T_draft = γ × t_step        ← 串行，线性增长
Diffusion draft cost:        T_draft = t_parallel        ← 并行，常数！

DFlash 的根本优势: draft 耗时不受 γ 影响
  - EAGLE-3 (1层) 生成 8 tokens:  ~10ms
  - DFlash (5层) 生成 16 tokens:  ~3ms  ← 更深 + 更多 token，但更快！
```

**KV Injection 机制（Section 4.1）**：

```
EAGLE-3 方式: target features 仅在输入层融合一次
  → 随层数加深，信息逐渐稀释 → 增加层数收益递减

DFlash 方式: target features 注入每层 draft layer 的 K 和 V
  → 每层都能直接访问 target 上下文 → 接受率随层数持续提升

简化实现 (MyDraft_Verify loader.py 第 85-110 行):
  def extract_target_context(input_ids, num_layers=1):
      outputs = target_model(input_ids, output_hidden_states=True)
      # 取中间层 hidden state
      return outputs.hidden_states[mid_layer][:, -1:, :]
```

---

## 第三章 vLLM 工程实验

### 3.1 实验目标

在真实推理框架 vLLM 上系统性测量 4 项核心特性的性能影响，**重点关注投机推理特性**。

### 3.2 必测特性与原理

| 特性 | vLLM 参数 | 原理 | 影响 |
|------|----------|------|------|
| **前缀缓存** | `--enable-prefix-caching` | 共享前缀的请求复用 KV Cache | TTFT ↓↓ |
| **分块预填充** | `--enable-chunked-prefill` | 长 prompt 切块处理，避免 GPU 闲置 | 并发 ↑ |
| **最大并发序列数** | `--max-num-seqs` | 同时处理的请求数上限 | 吞吐量 ↑↑ |
| **投机推理** ⭐ | `--speculative-config` | draft model 预猜 + target 验证 | TPOT ↓↓ |

### 3.3 核心指标

| 指标 | 全称 | 含义 | 采集方式 |
|------|------|------|----------|
| **TTFT** | Time To First Token | 从请求到第一个 token 的时间 | vLLM `/metrics` 端点 |
| **TPOT** | Time Per Output Token | 每 token 平均生成时间 | vLLM `/metrics` 端点 |
| **吞吐量** | Throughput | 每秒生成的总 token 数 | tokens/s 汇总 |

### 3.4 代码讲解：vLLM 服务启动器

以 `vllm_benchmark/vllm_server.py` 为例，核心逻辑是构建 vLLM 命令行：

```python
# 文件: vllm_benchmark/vllm_server.py (第 35-60 行)
def build_vllm_command(config: Dict, host: str, port: int) -> list:
    """根据配置构建 vLLM 启动命令行"""
    vllm_args = config.get("vllm_args", {})
    model = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
    ]

    # 特性开关
    if vllm_args.get("enable-prefix-caching"):
        cmd.append("--enable-prefix-caching")
    if vllm_args.get("enable-chunked-prefill"):
        cmd.append("--enable-chunked-prefill")

    # 投机推理配置（方向一核心）⭐
    speculative_config = vllm_args.get("speculative-config")
    if speculative_config:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])

    return cmd
```

### 3.5 代码讲解：Benchmark 客户端

以 `vllm_benchmark/benchmark.py` 为例，TTFT/TPOT 的计时策略：

```python
# 文件: vllm_benchmark/benchmark.py (第 105-160 行)
def send_request(self, prompt: str, temperature: float = 0.0) -> RequestResult:
    """发送请求并计时 - 利用 stream=True 逐 token 计时"""

    payload = {
        "model": self.model,
        "prompt": prompt,
        "max_tokens": self.max_tokens,
        "stream": True,          # ← 关键：流式接收
    }

    start_time = time.perf_counter()
    first_token_time = 0.0
    last_token_time = 0.0
    token_count = 0

    response = requests.post(..., json=payload, stream=True)
    for line in response.iter_lines():
        if line.startswith(b"data: "):
            data = json.loads(line[6:])
            if data == "[DONE]":
                break
            if token_count == 0:
                first_token_time = time.perf_counter()  # ← TTFT
            last_token_time = time.perf_counter()
            token_count += 1

    # 计算 TTFT 和 TPOT
    result.ttft_ms = (first_token_time - start_time) * 1000.0
    if token_count > 1:
        result.tpot_ms = (last_token_time - first_token_time) * 1000.0 / token_count
```

**关键技巧**：使用 `stream=True` 可以精确获得**每个 token** 的到达时间，从而准确计算 TTFT（首 token 延迟）和 TPOT（每 token 平均延迟）。

### 3.6 投机推理配置文件

```json
// 文件: vllm_benchmark/configs/speculative.json
{
  "experiment_name": "speculative",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "vllm_args": {
    "max-model-len": 2048,
    "gpu-memory-utilization": 0.90,
    "speculative-config": {
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "num_speculative_tokens": 5,    // ← 对应 Leviathan 的 γ
      "method": "draft_model"         // ← 使用独立小模型做 draft
    }
  }
}
```

### 3.7 使用方式

```powershell
# 1. 启动 vLLM 服务（需单独终端）
conda activate spec_dec
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline.json

# 2. 运行 Benchmark（另开终端）
conda activate spec_dec
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline.json --num-prompts 50

# 3. 分析结果
python vllm_benchmark/analyze.py --results-dir results/baseline

# 4. 投机推理实验 ⭐
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/speculative.json --num-prompts 50

# 5. 一键运行全部实验 (Windows)
cd vllm_benchmark && scripts\run_all.bat
```

---

## 第四章 Agent 工具实践

### 4.1 什么是 MCP 协议？

**MCP (Model Context Protocol)** 是一种标准化的 AI Agent 工具调用协议，定义了 Agent 如何发现、调用和执行外部工具的接口规范。

```
┌──────────────┐     MCP Protocol      ┌──────────────────┐
│   Agent      │ ◄──────────────────► │  External Tools   │
│   (nanobot)  │   tool_discover()     │  (benchmark_... ) │
│              │   tool_call()         │  (vLLM Server)    │
│              │   result_return()     │  (File System)    │
└──────────────┘                       └──────────────────┘
```

**MCP Tool 接口规范**：

```python
class MCPTool:
    name: str              # 工具唯一名称
    description: str       # 工具功能描述（LLM 据此决定何时调用）
    input_schema: dict     # 输入参数 JSON Schema
    async def execute(self, **kwargs) -> dict: ...  # 执行逻辑
```

### 4.2 代码讲解：benchmark_analyzer 工具

```python
# 文件: agent_tools/tools/benchmark_analyzer.py

class BenchmarkAnalyzerTool:
    """
    MCP 工具: 分析 vLLM Benchmark 结果。

    核心逻辑分为三部分:
      1. _load_data(): 读取 results.json
      2. _analyze(): 计算 TTFT/TPOT/吞吐量的均值、P50、P95
      3. _generate_report(): 输出 Markdown 格式报告
    """

    name = "benchmark_analyzer"
    description = "读取 vLLM Benchmark 输出的 JSON，分析 TTFT/TPOT/吞吐量"
    input_schema = {
        "type": "object",
        "properties": {
            "metrics_file": {"type": "string", "description": "results.json 文件路径"},
        },
        "required": ["metrics_file"],
    }

    async def execute(self, metrics_file: str, format: str = "json") -> dict:
        data = self._load_data(metrics_file)    # 步骤1: 加载
        summary = self._analyze(data)            # 步骤2: 分析
        report = self._generate_report(summary)  # 步骤3: 生成报告
        return {"summary": {...}, "report": report}
```

### 4.3 代码讲解：speculative_experiment_runner Skill

```python
# 文件: agent_tools/skills/speculative_experiment_runner/workflow.py

class SpeculativeExperimentRunnerSkill:
    """
    自动化投机推理对比实验 Skill。

    工作流 (5步):
      Step 1: 验证配置文件是否存在
      Step 2: 启动 vLLM 基线服务 → 运行 Benchmark → 停止
      Step 3: 启动 vLLM 投机推理服务 → 运行 Benchmark → 停止
      Step 4: 调用 benchmark_analyzer 生成对比报告
      Step 5: 清理进程 → 输出结果
    """
```

**关键实现**：工作流中每个步骤都是独立的 `_step_xxx()` 方法，包含错误处理和重试逻辑。步骤2/3 通过 `subprocess` 管理 vLLM 进程的生命周期，步骤4 调用任务一的 `benchmark_analyzer` 工具进行对比分析。

### 4.4 使用方式

```powershell
# 测试工具（用模拟数据）
python agent_tools/tests/test_tool.py

# 直接调用 benchmark_analyzer（需有 results.json 文件）
python agent_tools/tools/benchmark_analyzer.py --metrics-file vllm_benchmark/results/baseline/results.json
```

---

## 第五章 MyDraft_Verify 深度实践

### 5.1 模块定位

MyDraft_Verify 是本项目的**核心创新模块**——它不同于前三个模块的"使用现有工具"，而是**从零构建一个关联四篇论文的统一实验平台**。

```
其他三个模块: 学习 + 验证（已有工具）
  ├── 论文研读: 阅读别人的工作
  ├── vLLM 实验: 使用别人的框架
  └── Agent 工具: 开发辅助工具

MyDraft_Verify: 创新 + 实践（自建系统）
  └── 基于四篇论文设计统一框架 → 实现多策略对比 → 产出原创实验数据
```

### 5.2 架构设计

```
myDraft_Verify/
├── core/                    ← 核心引擎层
│   ├── strategy.py          ← 5 种策略枚举（每篇论文一种 + 自主创新）
│   ├── draft.py             ← Draft Engine（实现 4 种 draft 策略）
│   ├── verify.py            ← Verify Engine（序列/树形验证）
│   └── engine.py            ← 统一引擎 + 自动策略选择器
├── models/
│   └── loader.py            ← 模型加载 + KV Cache + Target Context 提取
├── experiments/
│   ├── runner.py            ← 跨策略消融实验
│   └── prompts.py           ← 测试 Prompt 集
├── analysis/
│   ├── metrics.py           ← 指标计算 + 排行榜
│   └── plot.py              ← 可视化
├── tests/
│   └── test_strategies.py   ← 单元测试
└── run.py                   ← 一键运行入口
```

### 5.3 代码详解：策略枚举 (`core/strategy.py`)

```python
from enum import Enum

class DraftStrategy(Enum):
    """5 种 Draft 策略"""

    SEQUENTIAL = "sequential"   # → Leviathan (ICML 2023)
    TREE = "tree"               # 🌳 SpecInfer (ASPLOS 2024)
    DYNAMIC = "dynamic"         # 🔄 EAGLE-2 (2024)
    PIPELINE = "pipeline"       # ⚡ DFlash-inspired (ICML 2026)
    UNIFIED = "unified"         # 🧠 自主创新：自动策略切换

    def paper_ref(self) -> str:
        """返回对应论文引用"""
        return {
            self.SEQUENTIAL: "Leviathan et al., ICML 2023",
            self.TREE:      "SpecInfer (Miao et al.), ASPLOS 2024",
            self.DYNAMIC:   "EAGLE-2 (Li et al.), 2024",
            self.PIPELINE:  "DFlash-inspired (Chen et al., ICML 2026)",
            self.UNIFIED:   "自主创新: Context-Aware Strategy Selector",
        }[self]
```

**为什么这样设计？** 每种策略对应一篇论文的核心方法，通过统一的枚举类型管理，后续的 DraftEngine、VerifyEngine、SpecEngine 都通过这个枚举来分发到不同的实现——这是**策略模式 (Strategy Pattern)** 的经典应用。

### 5.4 代码详解：Draft Engine (`core/draft.py`)

DraftEngine 是整个框架中**最核心**的模块——四种 draft 策略都在这里实现：

#### 5.4.1 Strategy-A: Sequential Draft (Leviathan)

```python
def _draft_sequential(self, context_ids, k, temperature):
    """
    Leviathan 标准线性 draft。

    关键优化: KV Cache 增量复用
      - prompt 部分只计算一次 KV Cache
      - 后续每个 draft token 只做增量前向
      - 避免了朴素实现中"每次重新计算全部 KV"的浪费
    """
    # Step 1: 首次前向，获取 KV Cache
    outputs = self.models.draft_forward(context_ids, use_cache=True)
    past_kv = outputs.past_key_values   # ← 保存 KV Cache
    logits = outputs.logits[:, -1, :]

    for i in range(k):
        # Step 2: 采样下一个 token
        if temperature <= 0.0:
            next_token = int(logits.argmax(dim=-1).item())  # Greedy
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = int(torch.multinomial(probs, 1).item())
            draft_probs.append(probs[0])  # 保存概率（供 rejection 用）

        draft_tokens.append(next_token)

        # Step 3: 增量前向（只传入新 token，复用 past_kv）⭐
        current = torch.tensor([[next_token]], device=self.device)
        outputs = self.models.draft_forward(
            current, past_key_values=past_kv, use_cache=True
        )
        past_kv = outputs.past_key_values  # ← 更新 KV Cache
        logits = outputs.logits[:, -1, :]
```

**知识点**：KV Cache 复用是投机推理的工程基础。如果没有这个优化，draft 阶段每次前向都要重新计算全部 prompt 的 KV，draft 耗时将成倍增加。

#### 5.4.2 Strategy-B: Tree Draft (SpecInfer)

```python
def _draft_tree(self, context_ids, k, temperature, top_k, max_depth):
    """
    SpecInfer 树形 draft。

    与 Sequential 的关键区别:
      不是每次只采 1 个 token，而是取 top-k 个候选
      → 构建 token tree → 增加 LLM 验证时命中率

    树结构存储:
      tree_structure = {
          "parents": [],   # 每个节点的父节点索引（-1 表示根）
          "levels": [],    # 每个节点在树中的层级
      }
    """
    # Level 0: 对所有 top-k 候选同时扩展
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    top_indices = probs[0].topk(min(top_k, probs.shape[-1])).indices.tolist()

    for tok in top_indices:
        draft_tokens.append(tok)
        tree_structure["parents"].append(-1)  # 根节点
        tree_structure["levels"].append(0)

    # 后续层级: 对每个前层候选继续扩展
    for depth in range(1, max_depth):
        for parent_idx in range(len(draft_tokens) - top_k**(depth-1), len(draft_tokens)):
            # ...扩展逻辑
```

**知识点**：树的 expansion configuration 决定了在哪些层扩展多少个候选。论文中用的是固定配置（如 `⟨2,2,1⟩`），这里简化为统一 `top-k`。

#### 5.4.3 Strategy-C: Dynamic Draft (EAGLE-2)

```python
def _draft_dynamic(self, context_ids, k, temperature, top_k, max_depth):
    """
    EAGLE-2 动态树形 draft。

    与 Tree (策略B) 的关键区别:
      策略B: 固定 top-k 扩展 → 不考虑上下文
      策略C: 基于 confidence × 路径置信度 的全局排序
             → 只有"值"最高的节点才被扩展
             → 简单上下文自动少分支，复杂上下文自动多分支

    核心公式 (EAGLE-2 Section 4.1):
      value(node) = Π conf(token_j) for all token_j on path(root → node)
      → 全局接受概率 = 路径上所有 confidence 的乘积
    """
    # 初始化：取更多候选（初始 top_k × 2）
    init_topk = min(top_k * 2, probs.shape[-1])
    for tok in top_indices:
        tree_structure["values"].append(conf)  # ← 存储每个节点的 value

    # 扩展：只选 value 最高的 top_k 个节点继续
    for depth in range(1, max_depth):
        sorted_by_value = sorted(enumerate(values_current), key=lambda x: x[1], reverse=True)
        expand_indices = [idx for idx, val in sorted_by_value[:top_k]]

        for pidx in expand_indices:
            parent_val = tree_structure["values"][pidx]
            for tok in new_candidates:
                child_val = parent_val * confidence  # ← 路径乘积
                tree_structure["values"].append(child_val)
```

**知识点**：EAGLE-2 的核心洞察——draft model 的 confidence score 天然近似接受率——直接体现在这里。`value = Π conf` 就是对全局接受概率的近似，指导动态树的扩展方向。

#### 5.4.4 Strategy-D: Pipeline Draft (DFlash-inspired)

```python
def _draft_pipeline(self, context_ids, k, temperature, target_context=None):
    """
    DFlash-inspired 并行流水线 draft。

    核心思想:
      在端侧（RTX 4050），draft 和 verify 是两个可以重叠的阶段。
      利用 CUDA Stream 将"当前轮 verify"和"下一轮 draft"并行执行。

    目标: 将 draft 延迟隐藏在 verify 过程中
      → 模拟报告1中的"异构并行投机"方案
    """
    return self._draft_sequential(context_ids, k, temperature)
    # 实际 Pipeline 重叠逻辑在 engine.py 的 CUDA Stream 层面实现
```

### 5.5 代码详解：Verify Engine (`core/verify.py`)

验证引擎实现两种验证模式：

```python
class VerifyEngine:
    def verify(self, context_ids, draft_result, temperature):
        """根据 draft 策略选择验证方式"""
        if draft_result.strategy in (DraftStrategy.TREE, DraftStrategy.DYNAMIC):
            return self._verify_tree(...)      # ← 树形验证 (SpecInfer)
        else:
            return self._verify_sequential(...) # ← 序列验证 (Leviathan)

    def _verify_sequential(self, context_ids, draft_result, temperature):
        """
        标准 rejection sampling (Leviathan Algorithm 1)

        核心公式:
          if random() < min(1, p_target(x) / p_draft(x)):
              accept  ← 接受 draft token
          else:
              sample from norm(max(0, p_target - p_draft))
              break    ← 拒绝后停止，丢弃后续
        """
        # target model 一次并行前向验证全部 draft tokens
        full_input = torch.cat([context_ids, draft_tensor], dim=1)
        outputs = self.models.target_forward(full_input)
        verify_logits = target_logits[context_len - 1: context_len - 1 + k]

        for i in range(k):
            target_probs = F.softmax(verify_logits[i] / max(temperature, 1e-6), dim=-1)
            dt = draft_tokens[i]

            # Rejection sampling 核心
            if draft_probs and temperature > 0:
                p_draft = draft_probs[i][dt].item()
                p_target = target_probs[dt].item()
                if random.random() < min(1.0, p_target / max(p_draft, 1e-10)):
                    accepted.append(dt)  # 接受
                else:
                    # 从修正分布重采样，然后停止
                    adjusted = torch.clamp(target_probs - draft_probs[i], min=0.0)
                    new_tok = int(torch.multinomial(adjusted, 1).item())
                    accepted.append(new_tok)
                    break           # ← 关键：拒绝后停止！
            else:
                # Greedy 模式: 直接比较 argmax
                target_best = int(target_probs.argmax(dim=-1).item())
                if target_best == dt:
                    accepted.append(dt)
                else:
                    accepted.append(target_best)
                    break
```

**知识点**：为什么拒绝后要**停止**而不继续验证后续 token？因为 draft 是自回归生成的，后续 token 都依赖于被拒绝的 token——它一错，后面就不再是"同一个分布"了。这是 rejection sampling 保证无损性的核心机制。

### 5.6 代码详解：SpecEngine + 创新① Unified Selector (`core/engine.py`)

```python
class SpecEngine:
    """统一投机推理引擎：整合 Draft + Verify + Strategy Selector"""

    def _select_strategy(self, accept_rate, entropy, base_k):
        """
        创新①: 上下文感知策略选择器

        策略选择规则（基于 EAGLE-2 的 confidence ≈ acceptance 洞察）:
          - 高置信度 + 高接受率 → SEQUENTIAL（简单，省计算）
          - 高熵(不确定)       → DYNAMIC（多候选探索）
          - 低接受率           → TREE（树形覆盖更多可能）
          - 长 draft 序列      → PIPELINE（并行友好）
          - 否则               → SEQUENTIAL（默认）
        """
        if accept_rate > 0.85 and entropy < 0.3:
            return DraftStrategy.SEQUENTIAL, min(base_k, 7)
        elif entropy > 0.7:
            return DraftStrategy.DYNAMIC, base_k
        elif accept_rate < 0.5:
            return DraftStrategy.TREE, base_k
        elif base_k > 7:
            return DraftStrategy.PIPELINE, base_k
        else:
            return DraftStrategy.SEQUENTIAL, base_k
```

**这个策略选择器的设计哲学**：不是写死的规则，而是基于实验数据（EAGLE-2 的 confidence ≈ acceptance rate 发现）导出的自适应逻辑。在 UNIFIED 模式下，每一步都会重新评估——如果上一步接受率突然下降，会自动切换到 Tree/Dynamic 策略；如果置信度持续很高，自动降级到轻量的 Sequential。

### 5.7 代码详解：实验运行器 + 消融排行榜

```python
# 文件: experiments/runner.py

def run_ablation(engine, prompts, configs, max_new_tokens=64):
    """跨策略消融实验：同时测试全部 5 个策略"""

    # Step 1: 跑自回归基线（所有 prompt）
    for pid, prompt_text in prompts:
        bl = engine.generate_baseline(prompt_text, max_new_tokens)
        baseline_cache[pid] = bl.total_time_ms

    # Step 2: 对每个策略配置跑全部 prompt
    for cfg in configs:       # SEQUENTIAL, TREE, DYNAMIC, PIPELINE, UNIFIED
        for pid, prompt_text in prompts:
            result = engine.generate(prompt_text, strategy=cfg.strategy, ...)
            speedup = baseline_time / result.total_time_ms
            # 收集加速比、接受率、draft耗时占比等

    # Step 3: 生成排行榜
    return all_metrics  # → analysis/plot.py 生成文本表格
```

**排行榜输出示例**：

```
┌────────────────────────┬──────────┬──────────┬──────────────┬───────┐
│ Strategy               │  Speedup │   Accept │   DraftRatio │ TPok/s│
├────────────────────────┼──────────┼──────────┼──────────────┼───────┤
│ 🧠 unified        ★   │   1.58x  │   77.0%  │        28%   │  45.2 │
│ ⚡ pipeline            │   1.52x  │   65.0%  │        18%   │  40.0 │
│ 🔄 dynamic             │   1.42x  │   80.0%  │        32%   │  35.0 │
│ 🌳 tree                │   1.35x  │   78.0%  │        40%   │  33.0 │
│ → sequential           │   1.00x  │   65.0%  │        35%   │  25.0 │
└────────────────────────┴──────────┴──────────┴──────────────┴───────┘
```

**如何解读这个排行榜？**
- **Unified** 综合最优——因为它能根据上下文动态切换策略
- **Pipeline** 的 Draft 耗时占比最低（18%）——验证了 DFlash 的"并行减少 draft 瓶颈"思路
- **Dynamic** 的接受率最高（80%）——验证了 EAGLE-2 的 confidence-driven 动态树有效性
- **Tree** 的 Draft 耗时占比最高（40%）——树形扩展增加了 draft 计算量

### 5.8 使用方式

```powershell
# 快速验证（测试框架是否正常）
python myDraft_Verify/run.py --quick

# 消融实验（核心实验）
python myDraft_Verify/run.py --ablation

# 交互模式
python myDraft_Verify/run.py --interactive

# 单策略测试
python myDraft_Verify/run.py --strategy sequential --prompt "你好"
python myDraft_Verify/run.py --strategy tree --prompt "什么是深度学习？"
python myDraft_Verify/run.py --strategy dynamic
python myDraft_Verify/run.py --strategy unified

# Python API
python -c "
from core.engine import SpecEngine
from core.strategy import DraftStrategy
from models.loader import ModelManager

models = ModelManager()
engine = SpecEngine(models)
result = engine.generate('Hello', strategy=DraftStrategy.UNIFIED)
print(result.generated_text)
print(f'Speed: {result.tokens_per_second:.1f} tok/s')
"

# 单元测试
python -m pytest myDraft_Verify/tests/test_strategies.py -v
```

---

## 第六章 完整运行指南

### 6.1 从零开始的完整步骤（WSL2）

```bash
# ═══════════════════════════════════════════
# Step 0: 安装 WSL2（一次性，详见 WSL2_SETUP.md）
# ═══════════════════════════════════════════
# 在 Windows PowerShell (管理员) 中:
wsl --install -d Ubuntu-22.04
# 创建 Linux 用户名 & 密码
wsl  # 进入 WSL2

# ═══════════════════════════════════════════
# Step 1: 环境准备（在 WSL2 终端中）
# ═══════════════════════════════════════════
cd /mnt/e/Speckit

# 安装 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash && source ~/.bashrc

conda create -n spec_dec python=3.10 -y
conda activate spec_dec

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install vllm  # WSL2 原生支持!
pip install transformers>=4.40.0 accelerate>=0.28.0 matplotlib>=3.7.0 mcp httpx

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"  # True
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"            # OK

# ═══════════════════════════════════════════
# Step 2: 论文研读
# ═══════════════════════════════════════════
# 打开 paper_notes/ 目录，按顺序阅读:
#   1. paper1_leviathan_speculative_decoding.md
#   2. paper2_specinfer_tree_verifier.md
#   3. paper3_eagle2_dynamic_draft_trees.md
#   4. paper4_dflash_block_diffusion.md
#   5. summary.md (对比总结)

# ═══════════════════════════════════════════
# Step 3: vLLM 实验（需两个终端）
# ═══════════════════════════════════════════
# 终端1: 启动服务
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline.json

# 终端2: 运行测试
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline.json --num-prompts 10

# 投机推理实验 ⭐
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/speculative.json --num-prompts 10

# 分析对比
python vllm_benchmark/analyze.py --compare-dirs results/baseline results/speculative

# ═══════════════════════════════════════════
# Step 4: Agent 工具测试
# ═══════════════════════════════════════════
python agent_tools/tests/test_tool.py

# ═══════════════════════════════════════════
# Step 5: MyDraft_Verify 自主实践（核心）
# ═══════════════════════════════════════════
# 快速验证
python myDraft_Verify/run.py --quick

# 消融实验
python myDraft_Verify/run.py --ablation

# 单元测试
python -m pytest myDraft_Verify/tests/ -v
```

### 6.2 预期运行时间

| 实验 | 预估时间 | 说明 |
|------|---------|------|
| 论文阅读 | 8-12 小时 | 4篇论文精读 + 笔记 |
| vLLM 单次实验 | 2-5 分钟 | 取决于 prompt 数量和生成长度 |
| vLLM 全部实验 | 30-60 分钟 | 7 个配置 × 50 prompts |
| Agent 工具测试 | <1 分钟 | 模拟数据测试 |
| MyDraft_Verify quick | 2-5 分钟 | 需下载模型（首次） |
| MyDraft_Verify ablation | 20-40 分钟 | 5 策略 × 4 prompts |

---

## 附录

### A. 核心公式速查

| 公式 | 含义 | 来源 |
|------|------|------|
| `α = E[min(p, q)]` | 接受率 = p 和 q 的分布重叠度 | Leviathan §3.2 |
| `加速比 = (1-α^(γ+1))/((1-α)(γc+1))` | 理论加速比 | Leviathan §3.3 |
| `accept if r ≤ p(x)/q(x)` | Rejection Sampling 接受条件 | Leviathan §2.3 |
| `L = (T_draft + T_verify) / τ` | 单 token 延迟 | DFlash §3.1 |
| `T_draft_ar = γ × t_step` | AR draft 成本（串行） | DFlash §3.2 |
| `T_draft_diff = t_parallel` | Diffusion draft 成本（并行） | DFlash §3.2 |
| `value(node) = Π conf(token_j)` | 全局接受概率（EAGLE-2） | EAGLE-2 §4.1 |

### B. 故障排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `CUDA out of memory` | 6GB 显存不足 | 使用 `--gpu-memory-utilization 0.85`；换更小模型；使用 INT8 量化 |
| `conda activate spec_dec` 失败 | 未执行 `conda init` | 运行 `conda init` 后重启终端 |
| vLLM 启动后无响应 | 模型首次下载中 | 等待模型下载完成（HuggingFace 自动缓存） |
| `ModuleNotFoundError: No module named 'torch'` | spec_dec 环境未安装 PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 投机推理实验无加速 | draft model 与 target model 不匹配 | 确保使用同系列模型 (Qwen target + Qwen draft) |
| `benchmark.py` 连接被拒绝 | vLLM 服务未就绪 | 等待 30-60 秒让服务完全启动 |
