# 端云投机加速 LLM 推理 —— 科研实践项目

> **方向一：投机推理 (Speculative Decoding)**
> **运行环境：WSL2 (Ubuntu 22.04)** ← 参见 [WSL2_SETUP.md](WSL2_SETUP.md)
> RTX 4050 Laptop GPU (6GB) | 周期：2025.07.15 – 07.31

---

## 项目概述

四大模块，按学习逻辑组织：**论文研读 → vLLM 工程实验 → Agent 工具开发 → 自主深度实践**。

```
理论学习 ──→ 工程验证 ──→ 工具自动化 ──→ 深度实践创新
   ①           ②            ③             ④
```

---

## 项目结构

```
e:/Speckit/
├── README.md                          ← 本文件
├── Require/                           ← 需求文档 & 方案设计
│   ├── project_design.md              ← 完整方案（含四模块顺序 + 创新设计）
│   ├── 暑期科研论文及实践.md/pptx     ← 方向一全部要求
│   └── LLM科研实践日程.pdf            ← 三阶段日程
├── Papers/                            ← 四篇指定论文 PDF
├── paper_notes/                       ← ① 论文研读笔记
├── vllm_benchmark/                    ← ② vLLM 特性性能实验
├── agent_tools/                       ← ③ Agent 工具实践
└── myDraft_Verify/                    ← ④ MyDraft_Verify 自主实践 ⭐
```

---

## 快速开始

```bash
# ══ WSL2 环境（推荐）══
# 详见 WSL2_SETUP.md
wsl --install -d Ubuntu-22.04         # 一次性安装 WSL2
wsl                                    # 进入 WSL2

# 在 WSL2 中:
cd /mnt/e/Speckit
conda create -n spec_dec python=3.10 -y
conda activate spec_dec
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers>=4.40.0 accelerate matplotlib>=3.7.0 mcp httpx

# ══ Windows 备用 ══
# 同上但用 hf_server.py 替代 vllm
```

---

## 四模块速览

| 序号 | 模块 | 核心内容 | 状态 |
|------|------|----------|------|
| ① | **论文研读** | 4篇论文精读笔记 + 对比总结 | ✅ 已完成 |
| ② | **vLLM 实验** | 4项特性 Benchmark + 投机推理 | 代码完成 |
| ③ | **Agent 工具** | MCP 工具 + Skill 工作流 | 代码完成 |
| ④ | **MyDraft_Verify** ⭐ | 4策略统一框架 + 4个创新点 | ✅ 新架构 |

### ④ MyDraft_Verify 核心特色

| 策略 | 论文 | 机制 |
|------|------|------|
| Sequential | Leviathan (ICML 2023) | 标准线性 draft-verify |
| Tree | SpecInfer (ASPLOS 2024) | top-k 候选 + 树形验证 |
| Dynamic | EAGLE-2 (2024) | confidence-driven 动态树 |
| Pipeline | DFlash (ICML 2026) | 并行流水线 draft |
| **Unified** | 🧠 自主创新 | 根据上下文自动切换最优策略 |

**创新点**：Unified Strategy Selector / Cross-Strategy Ablation / Target Context Injection / Ablation Leaderboard

---

## 论文关联全景

```
Leviathan (2023) ──→ 理论基础: draft-verify + rejection sampling
    │
SpecInfer (2024) ──→ 树形扩展: Token Tree + 批量验证
    │
EAGLE-2 (2024) ──→ 动态策略: confidence ≈ acceptance rate
    │
DFlash (2026)   ──→ 并行范式: block diffusion drafting + KV Injection
    │
    └──→ MyDraft_Verify 统一框架整合全部 4 条路线
```

---

## 核心指标

| 指标 | 含义 |
|------|------|
| TTFT | 首 Token 生成时间 |
| TPOT | 每 Token 平均生成时间 |
| 加速比 | 投机推理 vs 自回归基线的耗时比 |
| 接受率 | draft token 被 target 接受的比例 |
| Draft 耗时占比 | draft 阶段耗时 / 总耗时 |

---

## 环境

| 项目 | 配置 |
|------|------|
| GPU | RTX 4050 Laptop (6GB) |
| CUDA | 12.1 |
| Python | 3.10 (conda: spec_dec) |
| PyTorch | ≥2.1.0 |
| vLLM | ≥0.5.0 |
| 模型 | Qwen2.5-1.5B (target) + Qwen2.5-0.5B (draft) |
