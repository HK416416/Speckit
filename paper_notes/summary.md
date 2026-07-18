# 四篇论文对比总结

> 方向一：投机推理方向论文研读

---

## 一、四篇论文速览

| | 论文1: Leviathan | 论文2: SpecInfer | 论文3: EAGLE-2 | 论文4: DFlash |
|---|---|---|---|---|
| **路线** | draft-model | draft-model (树形) | self-speculative | 块并行 |
| **Draft 来源** | 独立小模型 | 独立小模型 | target model 的 hidden states | target model 自身（打破因果） |
| **Draft 结构** | 线性序列 | 树形 (top-k 分支) | 动态树 (自适应) | 块 (并行生成) |
| **是否需要额外训练** | 否（选已有模型） | 否（选已有模型） | 是（需训练 draft head） | 是（从零或微调） |
| **端侧适配** | ⭐⭐⭐ 适合 | ⭐⭐ 树形增加显存 | ⭐ 需额外训练 | ⭐ 训练成本高 |

---

## 二、三条路线的设计哲学对比

```
┌─────────────────────────────────────────────────────────────┐
│                   投机推理加速 LLM 推理                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  路线1: Draft-Model (Leviathan, SpecInfer)                  │
│    ├── 哲学: 用另一个模型来"猜"，猜对了加速，猜错了无损      │
│    ├── 优势: 实现简单，不需要训练 target model               │
│    ├── 劣势: 需要额外显存存放 draft model                    │
│    └── 端侧瓶颈: draft model 本身成为瓶颈 ← 方向一核心问题   │
│                                                             │
│  路线2: Self-Speculative (EAGLE-2, Medusa)                  │
│    ├── 哲学: 大模型自己"猜"自己，利用内部特征                │
│    ├── 优势: 不需要独立 draft model，节省显存                │
│    ├── 劣势: 需要额外训练 (draft heads / early exit 层)     │
│    └── 端侧适配: 显存友好，但额外计算量需评估                │
│                                                             │
│  路线3: 块并行解码 (DFlash, Domino, DSpark)                 │
│    ├── 哲学: 不猜了，直接打破因果依赖，一次生成多个 token    │
│    ├── 优势: 理论上加速上限更高 (不依赖 "猜对" 概率)         │
│    ├── 劣势: 接受率波动大，需要大规模训练                    │
│    └── 端侧适配: 训练成本高，更适合云端                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、各论文对实践的直接指导

| 论文 | 指导内容 | 对应实验模块 |
|------|----------|-------------|
| Leviathan | draft-verify 核心循环逻辑 | `myDraft_Verify/` — 手写框架的基础 |
| SpecInfer | 树形 draft 结构、tree attention mask | `myDraft_Verify/` — 优化点③⑥ |
| EAGLE-2 | Self-Speculative early exit 思路 | `myDraft_Verify/` — Self-Speculative 对比实验 |
| DFlash | 打破因果 mask 的设计哲学 | vLLM 实验中对比投机推理效果 |

---

## 四、阅读完成检查清单

- [ ] 论文1: Leviathan — 理解 draft-verify 循环和 rejection sampling
- [ ] 论文2: SpecInfer — 理解 token tree verifier 和 tree attention mask
- [ ] 论文3: EAGLE-2 — 理解 feature-level draft 和 dynamic draft tree
- [ ] 论文4: DFlash — 理解块并行解码和因果 mask 打破机制
- [ ] 完成对比总结 (本文件)
- [ ] 提炼出 3 条可以写入 PPT 的关键洞察
