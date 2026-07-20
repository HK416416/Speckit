# Agent 工具实践模块 (agent_tools/)

> 部署 nanobot 开源 Agent 框架，基于 MCP 协议完成自定义工具和 Skill 开发
> **运行环境：WSL2 (Ubuntu 22.04)** ← [WSL2_SETUP.md](../WSL2_SETUP.md)

---

## 文件结构

```
agent_tools/
├── MANUAL.md                         ← 本手册
├── tools/
│   └── benchmark_analyzer.py         ← 任务一：自定义 MCP 工具
├── skills/
│   └── speculative_experiment_runner/ ← 任务二：自定义 Skill 工作流
│       ├── skill.yaml                ← Skill 定义
│       └── workflow.py               ← 工作流编排逻辑
└── tests/
    └── test_tool.py                  ← 工具单元测试
```

---

## 环境要求

### 1. Conda 环境

```powershell
conda activate spec_dec

# 确认 Python 版本
python --version  # 应输出 Python 3.10.x
```

### 2. 安装 nanobot 和 MCP SDK

```powershell
# 安装 MCP Python SDK
pip install mcp httpx

# 注意: nanobot 为假设性开源框架名，本模块实现的是其工具/Skill 接口规范
# 真实部署时请替换为实际框架
```

---

## 模块一：自定义 MCP 工具 — benchmark_analyzer.py

### 功能说明

`benchmark_analyzer` 是一个 MCP 协议兼容的工具，专门用于分析 vLLM Benchmark 输出的 `results.json` 文件。

### MCP Tool 接口规范

每个 MCP 工具需实现以下接口：

```python
class MCPTool:
    name: str                               # 工具唯一名称
    description: str                        # 工具功能描述
    input_schema: dict                      # 输入参数 JSON Schema
    async def execute(self, **kwargs) -> dict: ...  # 执行逻辑
```

### 工具注册到 nanobot

```python
# 在 nanobot 的 tools/__init__.py 中注册
from agent_tools.tools.benchmark_analyzer import BenchmarkAnalyzerTool

nanobot.register_tool(BenchmarkAnalyzerTool())
```

### 代码解读

`benchmark_analyzer.py` 核心逻辑分为三部分：

1. **数据加载** (`_load_data`): 读取 `results.json` 文件，解析实验配置和性能指标
2. **统计分析** (`_analyze`): 计算 TTFT/TPOT/吞吐量的均值、P50、P95
3. **报告生成** (`_generate_report`): 输出 Markdown 格式的摘要报告

```
数据流:
  results.json ──→ _load_data() ──→ _analyze() ──→ _generate_report() ──→ Markdown Report
                      │                  │
                      │                  ├── avg_ttft, p50_ttft, p95_ttft
                      │                  ├── avg_tpot
                      │                  └── throughput
                      │
                      └── experiment_name, config, per_request[]
```

---

## 模块二：自定义 Skill — speculative_experiment_runner

### 功能说明

`speculative_experiment_runner` 是一个自动化工作流 Skill，编排端到端的投机推理对比实验流程。

### Skill 工作流步骤

```
Step 1: 加载配置
  ├── 读取 baseline.json 和 speculative.json
  └── 验证模型路径、端口等参数

Step 2: 基线实验
  ├── 启动 vLLM (baseline 配置)
  ├── 等待服务就绪 (health check)
  ├── 运行 benchmark.py
  ├── 收集 results.json
  └── 停止 vLLM 服务

Step 3: 投机推理实验
  ├── 启动 vLLM (speculative 配置)
  ├── 等待服务就绪
  ├── 运行 benchmark.py
  ├── 收集 results.json
  └── 停止 vLLM 服务

Step 4: 分析对比
  ├── 调用 benchmark_analyzer 工具 (任务一的工具)
  ├── 计算加速比
  └── 生成对比报告

Step 5: 清理 & 输出
  ├── 确保 vLLM 进程已停止
  └── 返回报告路径
```

### Skill 配置文件 (skill.yaml)

```yaml
name: speculative_experiment_runner
version: "1.0.0"
description: >
  自动化执行投机推理对比实验:
  基线 vs 投机推理的 TTFT/TPOT/吞吐量对比
dependencies:
  tools:
    - benchmark_analyzer        # 依赖任务一的自定义工具
  scripts:
    - vllm_benchmark/vllm_server.py
    - vllm_benchmark/benchmark.py
triggers:
  - type: manual                # 手动触发
  - type: scheduled             # 可选：定时触发
```

---

## 使用方式

### 测试 benchmark_analyzer 工具

```powershell
cd agent_tools

# 用模拟数据测试工具
python tests/test_tool.py
```

### 在 nanobot 中使用

```python
# 启动 nanobot Agent
from nanobot import Agent

agent = Agent()

# 注册自定义工具
from agent_tools.tools.benchmark_analyzer import BenchmarkAnalyzerTool
agent.register_tool(BenchmarkAnalyzerTool())

# 注册自定义 Skill
from agent_tools.skills.speculative_experiment_runner import SpeculativeExperimentRunnerSkill
agent.register_skill(SpeculativeExperimentRunnerSkill())

# 通过自然语言调用
response = agent.chat("请分析 results/baseline 目录下的实验结果")
print(response)

# 触发工作流
response = agent.chat("运行投机推理对比实验")
print(response)
```

---

## MCP 协议交互流程

```
┌──────────────┐     MCP Protocol      ┌──────────────────┐
│   nanobot    │ ◄──────────────────► │  External Server  │
│   Agent      │                       │  (vLLM / Files)  │
└──────┬───────┘                       └──────────────────┘
       │
       ├── tool_discover()
       │   └── 发现可用的 benchmark_analyzer 工具
       │
       ├── tool_call(name="benchmark_analyzer", args={...})
       │   └── 执行分析逻辑
       │   └── 返回结构化结果
       │
       └── skill_execute(name="speculative_experiment_runner")
           └── 按 workflow.py 中定义步骤执行
           └── 每步有错误处理和重试
```

---

## 验证检查清单

- [ ] `benchmark_analyzer.py` 能正确解析 `results.json` 并生成 Markdown 报告
- [ ] `speculative_experiment_runner` 工作流能按顺序执行所有步骤
- [ ] 工具和 Skill 在 nanobot 中注册后能被自然语言调用
- [ ] 错误处理：服务不可用、文件不存在等异常场景有适当反馈
