# Paper 3: EAGLE-2 — Faster Inference of Language Models with Dynamic Draft Trees

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees |
| **作者** | Yuhui Li et al. |
| **会议/期刊** | arXiv 2024 (后续被会议接收) |
| **年份** | 2024 |
| **链接** | https://arxiv.org/abs/2406.16858 |
| **阅读日期** | [TODO: YYYY-MM-DD] |
| **阅读耗时** | [TODO: X 小时] |

---

## 阅读前置知识

在阅读 EAGLE-2 之前，建议先了解：

1. EAGLE (前作) — 首次提出 feature-level 草稿生成
2. 标准投机推理 (Leviathan) — 理解 draft model 路线的局限
3. Medusa — 对比：用 extra decoding heads 做 self-speculative

---

## 1. 核心问题 (1-2句)

> 这篇论文要解决什么核心问题？

[TODO: 独立 draft model 需要额外显存和训练 → Self-Speculative 路线如何做得更好？EAGLE-2 相比 EAGLE-1 改进了什么？]

---

## 2. 核心方法 (3-5句关键思路)

> 怎么做？关键设计是什么？

[TODO: 
- Feature-level 草稿：不是预测 token，而是预测 feature → 更准确的草稿
- Dynamic draft tree：根据上下文动态调整树的结构，不是固定宽度/深度
- 无需独立 draft model：利用 target model 自身的 hidden states 做草稿
]

---

## 3. 方法详解

### 3.1 Feature-Level Draft 原理

```
传统 token-level draft (Leviathan):
  draft_model 直接预测 token → 误差大，因为 token 空间离散

EAGLE-2 feature-level draft:
  target_model 的 hidden states (连续) → 预测下一个 feature → 再用 lm_head 得到 token
  → 连续空间预测比离散 token 预测更准确
```

### 3.2 动态 Draft Tree

[TODO: EAGLE-2 如何根据上下文不确定性动态调整 draft tree 的拓扑？与固定宽度的 SpecInfer 有何不同？]

### 3.3 关键技术创新点

| 创新点 | 解决了什么问题 | 具体做法 |
|--------|---------------|----------|
| Feature-level 草稿 | token-level 预测不够准确 | 在连续特征空间做预测 |
| Dynamic Draft Trees | 固定 tree 结构不适应不同上下文 | 根据不确定性动态调整 |
| [TODO] | [TODO] | [TODO] |

---

## 4. 关键实验结果

| 实验 | 基线方法 | 本方法结果 | 提升幅度 |
|------|---------|-----------|----------|
| [TODO] | [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] | [TODO] |

**最重要的一个数据**：[TODO: 推测 EAGLE-2 的加速比和接受率相比 EAGLE-1 的提升]

---

## 5. 与方向一的关联

> 这篇论文对「端侧投机推理加速」有什么启发？

[TODO: 
- Self-Speculative 路线避免了加载独立 draft model → 节省端侧显存
- feature-level 预测在 RTX 4050 上是否可模拟？（用 target 前几层）
- 与手写框架中 Self-Speculative 对比实验直接相关
]

---

## 6. 个人思考

### 可借鉴的设计
[TODO: 如何在手写框架中模拟 feature-level draft (early exit)]

### 局限性
[TODO: EAGLE-2 需要训练 extra components，在手写框架中如何简化？]

### 疑问点
[TODO]

---

## 7. 一句话总结

> [TODO]
