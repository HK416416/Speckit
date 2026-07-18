# Paper 1: Fast Inference from Transformers via Speculative Decoding

> ★ 投机推理奠基论文 — 必读最先

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | Fast Inference from Transformers via Speculative Decoding |
| **作者** | Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research) |
| **会议/期刊** | ICML 2023 |
| **年份** | 2023 |
| **链接** | https://arxiv.org/abs/2302.01318 |
| **阅读日期** | [TODO: YYYY-MM-DD] |
| **阅读耗时** | [TODO: X 小时] |

---

## 1. 核心问题 (1-2句)

> 这篇论文要解决什么问题？

[TODO: 用自己的话概括 — 提示：自回归解码为什么慢？投机推理的核心洞察是什么？]

---

## 2. 核心方法 (3-5句关键思路)

> 怎么做？关键设计是什么？

[TODO: 用自己的理解描述 draft-verify 框架的核心流程]

---

## 3. 方法详解

### 3.1 整体架构

```
[TODO: 画出 draft-verify 循环的流程图]

Draft Model (轻量)              Target Model (大模型)
     │                                │
     ├─ generate k tokens ──→ draft_tokens
     │                                │
     │                        ┌───────┘
     │                        ├─ forward(prompt + draft_tokens)
     │                        ├─ 验证每个 draft token
     │                        └─ rejection sampling
     │                                │
     └──── 下一轮 ←── accepted tokens ┘
```

### 3.2 关键技术创新点

| 创新点 | 解决了什么问题 | 具体做法 |
|--------|---------------|----------|
| draft-verify 框架 | 自回归解码串行瓶颈 | 小模型预猜 + 大模型并行验证 |
| Rejection Sampling | 保证生成质量无损 | 基于概率比接受/拒绝 draft tokens |
| 接受率理论分析 | 预测加速上限 | 推导加速比与 draft 准确率的关系 |

### 3.3 接受率理论（核心公式理解）

[TODO: 用自己的话解释为什么 rejection sampling 能保证分布无损，写出关键不等式]

---

## 4. 关键实验结果

| 实验 | 基线方法 | 本方法结果 | 提升幅度 |
|------|---------|-----------|----------|
| [TODO] | [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] | [TODO] |
| [TODO] | [TODO] | [TODO] | [TODO] |

**最重要的一个数据**：[TODO]

---

## 5. 与方向一的关联

> 这篇论文对「端侧投机推理加速」有什么启发？

[TODO: 
- 端侧场景下 draft model 和 target model 的时延关系如何？
- 接受率理论对选择最佳 draft length k 有什么指导？
- 在 RTX 4050 上手写实现时需要注意什么？
]

---

## 6. 个人思考

### 可借鉴的设计
[TODO: draft-verify 循环结构直接指导手写框架实现]

### 局限性
[TODO: draft model 本身成为新的瓶颈 — 这是方向一的核心问题]

### 疑问点
[TODO]

---

## 7. 一句话总结

> [TODO: 用一句话告诉别人这篇论文做了什么、为什么重要]
