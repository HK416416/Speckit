# Paper 1: Fast Inference from Transformers via Speculative Decoding

> ★ 投机推理奠基论文 — ICML 2023, Google Research

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | Fast Inference from Transformers via Speculative Decoding |
| **作者** | Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research) |
| **会议/期刊** | ICML 2023 |
| **年份** | 2023 |
| **链接** | https://arxiv.org/abs/2211.17192 |
| **阅读日期** | 2025-07-18 |
| **阅读耗时** | ~2 小时 |

---

## 1. 核心问题 (1-2句)

自回归大模型推理慢 —— 解码 K 个 token 需要 K 次串行前向。能否在**不改变模型架构、不重新训练、不改变输出分布**的前提下加速？

---

## 2. 核心方法 (3-5句关键思路)

利用两个观察：(1) 许多推理步骤是"简单的"，可以被更高效的模型近似；(2) 大模型推理通常瓶颈在显存带宽而非算术运算，有闲置的并行计算资源可用。

**Draft-Verify 框架**：用轻量近似模型 (draft model, M_q) 自回归生成 γ 个候选 token → 大模型 (target model, M_p) 一次并行前向验证全部候选 → 通过 rejection sampling 接受/拒绝 → 保证输出分布与大模型自回归**完全相同**。

核心公式：
- 接受率 `α = E(min(p, q))`（p 和 q 越接近，α 越接近 1）
- 加速比期望：`(1-α^(γ+1)) / ((1-α)(γc+1))`，其中 c 是 draft/target 单步耗时比
- 接受率与 draft 模型大小的关系：draft 约小 2 个数量级时 α 在 0.5-0.9 之间

Rejection sampling 保证无损的关键：对每个 draft token，以概率 `min(1, p(x)/q(x))` 接受；若拒绝，从修正分布 `norm(max(0, p(x)-q(x)))` 重采样。

---

## 3. 方法详解

### 3.1 整体架构

```
输入: prefix, M_p (target), M_q (draft), γ (draft length)

1. DRAFT: M_q 自回归生成 γ 个 token x_1,...,x_γ，记录概率 q_i(x)
2. VERIFY: M_p 一次并行前向，计算 p_1(x),...,p_{γ+1}(x)
3. ACCEPT/REJECT: 对 i=1..γ
     r_i ~ U(0,1)
     if r_i ≤ p_i(x_i)/q_i(x_i): 接受 x_i，继续下一个
     else: 从 norm(max(0, p_{i+1} - q_{i+1})) 采样替换，停止
4. 返回 prefix + [accepted tokens]
```

### 3.2 关键技术创新点

| 创新点 | 解决了什么问题 | 具体做法 |
|--------|---------------|----------|
| Speculative Sampling | 如何在随机采样中保证分布无损 | rejection sampling 的特殊形式 |
| α-E(β) 理论分析 | 预测加速上限 | 推导加速比与接受率、cost coefficient 的关系 |
| 最优 γ 选择 | 不同 α 和 c 下该生成多少个 draft token | 数值求解最大化加速比公式 |

### 3.3 近似模型选择

论文测试了多种近似模型：
- **同类小模型**（如 T5-small 77M 对 T5-XXL 11B）：α 约 0.62-0.75，加速 2-3×
- **n-gram 模型**：c≈0 的"零成本模型"，α 约 0.2，仍有 1.25× 加速
- **随机模型**：理论上总是有加速（虽然很小）

关键发现：approximation model 约比 target model 小 2 个数量级时，α 在 0.5-0.9 之间，c < 0.05（几乎可以忽略）。

---

## 4. 关键实验结果

| 实验 | 基线方法 | 本方法结果 | 提升幅度 |
|------|---------|-----------|----------|
| EnDe Translation (T5-XXL + T5-Small, temp=0) | T5X 自回归 | 3.4× walltime | T5-Small 77M: α=0.75, γ=7 |
| EnDe Translation (T5-XXL + T5-Small, temp=1) | T5X 自回归 | 2.6× walltime | T5-Small 77M: α=0.62, γ=7 |
| CNN/DM Summarization (T5-XXL + T5-Small, temp=0) | T5X 自回归 | 3.1× walltime | α=0.65, γ=5 |
| GPT-like 97M + 6M (temp=0) | 自回归 | α=0.88 | 极高的接受率 |
| LaMDA 137B + LaMDA 2B (temp=0) | 自回归 | α=0.71 | 2B draft 对 137B target |

**最重要的一个数据**：T5-XXL + T5-Small 在 EnDe 翻译任务上实现 **3.4×** 端到端加速（temp=0），且输出完全相同。

---

## 5. 与方向一的关联

- draft-verify 框架是**手写投机推理代码 (myDraft_Verify/)** 的直接理论基础
- 接受率理论解释了为什么端侧 draft model 会成为瓶颈（端侧 c 值高，draft 耗时接近 verify）
- α = E(min(p, q)) 公式可直接嵌入代码计算理论加速比
- 论文明确指出 memory bandwidth 是瓶颈时投机推理最有效 → 端侧 (RTX 4050) 典型场景
- 最优 γ 的选择公式指导实验中 k ∈ {3, 5, 7, 10} 的取值

---

## 6. 个人思考

### 可借鉴的设计
- draft-verify 循环的伪代码 (Algorithm 1) 直接指导手写框架实现
- rejection sampling 的 `min(1, p(x)/q(x))` 公式直接用于代码
- i.i.d. 假设下的加速比公式可用来验证实验结果是否合理

### 局限性
- **draft model 本身成为瓶颈**：论文假设 c ≈ 0（draft 成本可忽略），但在端侧 (RTX 4050) 上 c 不可忽略
- 需要额外的存储和显存存放 draft model
- i.i.d. β 假设是简化，实际 β 有上下文相关性
- 未涉及批量推理（batch size > 1）的讨论

### 疑问点
- 端侧 draft model 耗时占比高（c 不可忽略），如何通过异构并行降低 c？
- 论文指出"如果有闲置计算资源"，RTX 4050 单 GPU 场景下还有闲置资源吗？

---

## 7. 一句话总结

> 用轻量 draft model 预猜多个 token，大模型一次并行验证，通过 rejection sampling 实现**无损**的 2-3× 推理加速——不需要改模型、不需要重新训练。
