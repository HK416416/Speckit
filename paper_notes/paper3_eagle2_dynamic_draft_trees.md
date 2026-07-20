# Paper 3: EAGLE-2 — Faster Inference of Language Models with Dynamic Draft Trees

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **论文标题** | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees |
| **作者** | Yuhui Li (Peking University), Fangyun Wei (Microsoft Research), Chao Zhang, Hongyang Zhang (Waterloo/Vector Institute) |
| **会议/期刊** | ICML 相关 (2024) |
| **年份** | 2024 |
| **链接** | https://arxiv.org/abs/2406.16858 |
| **阅读日期** | 2025-07-18 |
| **阅读耗时** | ~2 小时 |

---

## 阅读前置知识

EAGLE 系列的关键特点：
1. **Feature-level draft**：在特征空间（LM Head 之前的 hidden states）做自回归预测，而非 token 空间
2. **无需独立 draft model**：利用 target model 的 hidden states + 轻量 draft head
3. EAGLE → EAGLE-2：从静态 draft tree 升级为**动态 draft tree**

---

## 1. 核心问题 (1-2句)

现有方法（包括 EAGLE、Medusa、SpecInfer）都使用**静态 draft tree**——隐式假设 draft token 的接受率只取决于它在树中的位置。但实验发现：**接受率高度依赖于上下文**。"10+2=" 下一个 token 几乎一定是 "1"，不需要多个候选；而 "10+2" 则难以预测，需要更多候选。

---

## 2. 核心方法 (3-5句关键思路)

**关键发现**：EAGLE 的 draft model 是 **well-calibrated** 的——draft model 给出的 confidence score（输出概率）与 token 被 LLM 实际接受的概率有**极强的正相关**（confidence < 0.05 时接受率 ~0.04，confidence > 0.95 时接受率 ~0.98）。

基于此，EAGLE-2 用 confidence score 近似接受率，动态调整 draft tree 结构：

1. **Expansion Phase（扩展阶段）**：从当前层选择 top-k 个全局接受概率（value = 路径置信度乘积）最高的节点，输入 draft model 扩展
2. **Reranking Phase（重排序阶段）**：对所有 draft token 按全局接受概率排序，选 top-m 个形成最终 draft

---

## 3. 方法详解

### 3.1 Feature-Level Draft（继承自 EAGLE）

```
传统 token-level draft:
  draft_model 直接预测: token_1 → token_2 → token_3
  问题: token 空间离散 → 小误差被放大

EAGLE feature-level draft:
  target hidden state → draft model 预测下一层 feature → LM Head → token
  在连续特征空间预测 → 更准确
  同时输入前一步的采样结果（消除不确定性）
```

### 3.2 动态 Draft Tree 构建

```
Value of node t_i = Π_{t_j ∈ Path(root, t_i)} c_j
  其中 c_j = draft model 对 token t_j 的 confidence score

Expansion Phase (top-k=2):
  ┌─ It (1.0) ─┬─ is (0.6) ─── a (0.48)     ← value 最高，被选择扩展
  │            └─ has (0.2) ─── to (0.14)    ← 也被选择
  │
  扩展后得到: good (0.34), be (0.08), do (0.03), a (0.02), ...

Reranking Phase (top-m=8):
  从所有节点选 value 最高的 8 个 → 形成最终 draft
  保证父节点始终在子节点之前（按树结构 flatten）
```

### 3.3 关键技术创新点

| 创新点 | 解决了什么问题 | 具体做法 |
|--------|---------------|----------|
| Context-Aware Dynamic Tree | 静态树不适应不同上下文 | 用 confidence score 近似接受率，动态选择扩展/保留哪些分支 |
| Value-based Selection | 仅看局部 confidence 不够 | 全局接受概率 = 路径上所有 confidence 的乘积 |
| Reranking | 扩展可能遗漏浅层优秀节点 | 扩展后全局重排序，选最优的 m 个 |
| 无需额外训练 | 用已有 EAGLE draft model | 动态树结构完全基于推理时 confidence 调整 |

---

## 4. 关键实验结果

| 实验 | 基线方法 | 本方法结果 | 提升幅度 |
|------|---------|-----------|----------|
| Vicuna 13B MT-bench (temp=0) | EAGLE 3.07× | EAGLE-2 4.26× | **+39%** |
| Vicuna 7B MT-bench (temp=1) | EAGLE 2.13× | EAGLE-2 3.05× | **+43%** |
| LLaMA2-Chat 7B GSM8K | EAGLE 2.91× | EAGLE-2 3.52× | **+21%** |
| LLaMA3-Instruct 8B | EAGLE 2.72× | EAGLE-2 3.46× | **+27%** |
| 平均接受长度 τ | EAGLE ~3.8 | EAGLE-2 ~4.65 | 每轮多接受 ~1 token |
| vs Medusa (Vicuna 13B) | 2.07× | 4.26× | **~2× faster** |
| vs Lookahead (Vicuna 13B) | 1.65× | 4.26× | **~2.6× faster** |

**最重要的数据**：在 MT-bench 上 EAGLE-2 相比 EAGLE-1 提升 20-40%，且**不需要额外训练**——开箱即用。

---

## 5. 与方向一的关联

- EAGLE-2 是 Self-Speculative 路线的代表作 → 与手写框架中 **Self-Speculative 对比实验**直接相关
- 不需要独立 draft model → 对**端侧显存受限场景**非常友好
- "draft model confidence ≈ acceptance rate" 的发现 → 可用于手写框架的优化点②（动态 draft length 调整）
- Dynamic tree 的思想比 SpecInfer 的静态 expansion configuration 更灵活
- 但 EAGLE 系列需要训练 draft model，在手写框架中通过 early exit 方式近似模拟

---

## 6. 个人思考

### 可借鉴的设计
- "confidence ≈ acceptance rate" 的洞察可直接用于手写框架中动态调整 k
- Global acceptance value (路径乘积) 的概念可以简化应用

### 局限性
- 需要额外训练 draft model（虽然 EAGLE-2 不额外训练，但需要 EAGLE-1 的 draft model）
- 在 RTX 4050 上手写框架中难以完全复现 feature-level draft（需要修改模型内部结构）
- 论文在高端 GPU 上测试，树的大小（60 个节点）在端侧可能过大

### 疑问点
- Confidence score 与接受率的强相关性在**跨模型系列**（如 Qwen target + Llama draft）时是否仍然成立？
- 在手写框架中能否通过 target model 前几层的 early exit 模拟 feature-level draft？

---

## 7. 一句话总结

> EAGLE-2 发现 draft model 的 confidence score 天然近似接受率，据此动态调整 draft tree——简单上下文多用单分支、复杂上下文多用多分支——在 EAGLE-1 基础上再提速 20-40%，且无需额外训练。
