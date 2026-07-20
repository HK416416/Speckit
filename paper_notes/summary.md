# 四篇论文对比总结

> 方向一：投机推理方向论文研读 — 基于真实论文内容

---

## 一、四篇论文速览

| | 论文1: Leviathan | 论文2: SpecInfer | 论文3: EAGLE-2 | 论文4: DFlash |
|---|---|---|---|---|
| **会议** | ICML 2023 | ASPLOS 2024 | 2024 (ICML相关) | ICML 2026 |
| **路线** | draft-model (序列) | draft-model (树形) | self-speculative (动态树) | block-diffusion drafter |
| **Draft 来源** | 独立小模型 (如 T5-Small 77M) | 独立小模型 (如 LLaMA-68M) | target model hidden states + draft head | 轻量 block diffusion model (5层) |
| **Draft 结构** | 线性序列 | 树形 (expansion config) | 动态树 (context-aware, confidence-based) | 块 (一次并行生成 16 tokens) |
| **是否需要额外训练** | 无需 | 无需（预训练 SSM） | 需要训练 draft head (EAGLE-1) | 需要训练 drafter (800K+ 样本) |
| **加速比 (最佳)** | **3.4×** (T5-XXL + T5-Small) | **2.8×** (LLaMA-65B 多节点) | **4.26×** (Vicuna 13B) | **6.08×** (Qwen3-8B Math500) |
| **接受率/接受长度** | α = 0.62-0.88 | 验证成功率 57%→97% (top-5) | τ ≈ 4.65 tokens/cycle | τ ≈ 6.5 tokens/cycle |
| **端侧适配** | ⭐⭐⭐ 简单但 draft 成瓶颈 | ⭐⭐ 树形增加 KV cache | ⭐ 需训练 draft head | ⭐ 训练成本高 |

---

## 二、三条路线的设计哲学对比

```
┌─────────────────────────────────────────────────────────────────┐
│                     投机推理加速 LLM 推理                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  路线1: Draft-Model (Leviathan → SpecInfer)                     │
│    ├── 哲学: 用另一个模型"猜"，猜对了加速，猜错了无损            │
│    ├── Leviathan: 线性序列, α=0.62-0.88, 2-3× 加速              │
│    ├── SpecInfer: 树形序列, 验证率 57%→97%, +1.2-1.5× 额外提升 │
│    ├── 优势: 无需训练 target model, 即插即用                    │
│    ├── 劣势: 额外显存 + draft model 串行瓶颈                    │
│    └── 端侧瓶颈: draft model 成为瓶颈 ← 方向一核心问题          │
│                                                                 │
│  路线2: Self-Speculative (EAGLE → EAGLE-2)                     │
│    ├── 哲学: 大模型用自身特征"猜"自己，feature-level 更准确     │
│    ├── EAGLE-2: 动态树, confidence ≈ acceptance, 3-4× 加速     │
│    ├── 核心发现: draft model 的 confidence score 天然近似接受率 │
│    ├── 优势: 无需独立 draft model, 端侧显存友好                 │
│    ├── 劣势: 需要训练 draft head (但 EAGLE-2 无需额外训练)     │
│    └── 端侧适配: 显存最优，但需训练                             │
│                                                                 │
│  路线3: Block-Diffusion Drafter (DFlash)                        │
│    ├── 哲学: 扩散模型并行生成 + 自回归模型验证 → 完美分工       │
│    ├── DFlash: 5层 drafter, KV Injection, 16 token block, 6×   │
│    ├── 关键: T_draft = t_parallel (常数) vs AR = γ × t_step    │
│    ├── 优势: draft 耗时不随 block size 增长, 理论上限最高       │
│    ├── 劣势: 需要大规模训练 (800K+ 样本, H200 GPU)             │
│    └── 端侧适配: 训练成本过高，更关注其设计哲学                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、关键数字对比

| 维度 | Leviathan | SpecInfer | EAGLE-2 | DFlash |
|------|-----------|-----------|---------|--------|
| **速度天花板** | ~3.4× | ~2.8× (系统级) | ~4.26× | **~6.1×** |
| **Draft 并行度** | 串行 (γ 步) | 串行 + 并行验证 | 串行（每层 batch） | **全并行 (1 步)** |
| **每步生成 token** | ~1-2 | ~3-4 | ~4.65 | ~6.5 |
| **额外参数** | 独立小模型 | 独立小模型 + SSM | target 的 draft head | 5 层 draft model |
| **训练需求** | 无 | 无（可选 boost-tuning） | 需要（但可用已有） | 需要大量数据 |

---

## 四、各论文对实践的直接指导

| 论文 | 指导内容 | 对应实验模块 |
|------|----------|-------------|
| Leviathan | draft-verify 循环逻辑 + rejection sampling 公式 | `myDraft_Verify/` 全部代码 |
| Leviathan | 加速比公式 `(1-α^(γ+1))/((1-α)(γc+1))` | 理论加速比验证 |
| SpecInfer | 树形 draft + topology-aware mask 设计 | `myDraft_Verify/` 优化点③⑥ |
| EAGLE-2 | confidence ≈ acceptance rate 洞察 | 优化点② (动态 k 调整) |
| EAGLE-2 | 动态树 vs 静态树的对比思路 | Self-Speculative 对比实验 |
| DFlash | 并行 draft 的设计哲学 | 优化点⑤ Pipeline 并行 |
| DFlash | KV injection / target context | Self-Speculative early exit 设计 |
| DFlash | SGLang/vLLM 集成验证结果 | vLLM 实验对比讨论 |

---

## 五、四篇论文的时间线与演进关系

```
2023 ─ Leviathan (ICML)
        │  奠基: draft-verify + rejection sampling
        │  线性序列, α=0.62-0.88, 2-3× 加速
        │
2024 ─ SpecInfer (ASPLOS) ── EAGLE-2
        │  树形: token tree           │  动态树: context-aware
        │  验证: topology mask        │  发现: confidence ≈ accept rate
        │  多 SSM: collective boost   │  无需: 独立 draft model
        │                             │  提升: +20-40% vs EAGLE-1
        │                             │
2026 ─ ─ ─ ─ ─ ─ ─ ─ ─ DFlash (ICML) ─ ─ ─ ─ ─ ─ ─ ─ ─
                         块并行: block diffusion drafter
                         KV Injection: 每层注入 target features
                         天花板: 6× 加速, 2.5× vs EAGLE-3
                         范式: "扩散 draft + 自回归 verify"
```

---

## 六、阅读完成检查清单

- [x] 论文1: Leviathan — 理解 draft-verify 循环和 rejection sampling (α=0.62-0.88, 2-3×)
- [x] 论文2: SpecInfer — 理解 token tree verifier 和 topology-aware mask (57%→97%)
- [x] 论文3: EAGLE-2 — 理解 feature-level draft 和 confidence ≈ acceptance rate (4.26×)
- [x] 论文4: DFlash — 理解 block diffusion drafting 和 KV Injection 机制 (6×)
- [x] 完成对比总结 (本文件)
- [x] 提炼出 3 条可以写入 PPT 的关键洞察:
  1. **串行 → 并行**是三条路线的共同演进方向 (线性→树形→动态树→块并行)
  2. **Draft model 成为瓶颈**是方向一的核心问题——DFlash 用扩散并行、EAGLE 用 self-speculative 都在试图解决
  3. **基于真实论文数据的加速上限**：Leviathan 3.4× → EAGLE-2 4.26× → DFlash 6×
