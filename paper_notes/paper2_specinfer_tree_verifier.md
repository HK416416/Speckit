# Paper 2: SpecInfer — Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification |
| **作者** | Xupeng Miao et al. (CMU / PKU) |
| **会议/期刊** | USENIX ATC / OSDI 相关 |
| **年份** | 2024 |
| **链接** | https://arxiv.org/abs/2305.09781 |
| **阅读日期** | [TODO: YYYY-MM-DD] |
| **阅读耗时** | [TODO: X 小时] |

---

## 1. 核心问题 (1-2句)

> 这篇论文要解决投机推理的什么问题？

[TODO: 线性 draft 的"全有或全无"问题 — 一个 token 被拒，后续全部作废]

---

## 2. 核心方法 (3-5句关键思路)

> 怎么做？关键设计是什么？

[TODO: Token Tree — 每个位置生成多个候选 → 树形结构 → 批量验证 → 选最优路径]

---

## 3. 方法详解

### 3.1 整体架构

```
[TODO: 画出 token tree 的构建和验证流程]

线性 draft (Leviathan):
  [A] → [B] → [C] → [D] → [E]     若 B 被拒 → 全部作废

树形 draft (SpecInfer):
              ┌── [B1] → [C1] → [D1]
  [A] ──┤
              └── [B2] → [C2] → [D2]
  → target model 一次验证所有路径 → 选接受最多的分支
```

### 3.2 关键技术创新点

| 创新点 | 解决了什么问题 | 具体做法 |
|--------|---------------|----------|
| Token Tree Verifier | 线性 draft 浪费算力 | 树形结构 + 批量验证 |
| Tree Attention Mask | 树内各分支需要独立 | 非因果 mask 限制分支间可见性 |
| [TODO] | [TODO] | [TODO] |

### 3.3 Token Tree Verifier 详解

[TODO: 解释 tree attention mask 是如何构建的，为什么分支之间不能互相看到]

---

## 4. 关键实验结果

| 实验 | 基线方法 | 本方法结果 | 提升幅度 |
|------|---------|-----------|----------|
| [TODO] | [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] | [TODO] |

**最重要的一个数据**：[TODO]

---

## 5. 与方向一的关联

> 这篇论文对「端侧投机推理加速」有什么启发？

[TODO: 
- Tree-structured draft 在端侧 (RTX 4050) 上是否可行？
- 树形 draft 的 batch 验证会增加多少显存开销？
- 与手写框架的优化点③ (Tree-Structured Draft) 直接对应
]

---

## 6. 个人思考

### 可借鉴的设计
[TODO: token tree 构建逻辑、tree attention mask 实现]

### 局限性
[TODO: 树形结构在端侧小显存设备上的适配问题]

### 疑问点
[TODO]

---

## 7. 一句话总结

> [TODO]
