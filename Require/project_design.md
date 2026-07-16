# 端云投机加速 LLM 推理 —— 科研实践项目设计方案

---

## 一、项目概述

| 维度 | 内容 |
|------|------|
| **选定方向** | 报告1：端云投机加速LLM推理——异构并行与DFlash优化研究 |
| **硬件条件** | NVIDIA GeForce RTX 4050 笔记本 GPU，约 6GB 显存 |
| **总体周期** | 2025年7月15日 – 7月31日（三阶段） |
| **最终产出** | ≤10分钟线上总结报告 PPT + 实验代码 + 实验数据 |
| **实践目标** | 从零实现投机推理（Speculative Decoding）框架，在 RTX 4050 上完成系统性 Benchmark，产出有真实数据支撑的技术报告 |

---

## 二、三阶段总体安排

### 第一阶段：参加学术报告（7.15 – 7.16）

| 日期 | 时间 | 内容 | 腾讯会议 |
|------|------|------|----------|
| 7.15 | 14:00-14:30 | 大模型推理性能优化概述 | 341-252-035 |
| 7.15 | 14:30-15:30 | **报告1：端云投机加速LLM推理** ✅ | 同上 |
| 7.15 | 15:30-16:30 | 报告2：面向生成式推荐模型的多芯异构推理调度优化 | 同上 |
| 7.16 | 9:30-10:30 | 报告3：面向多阶段智能体回溯推理加速的上下文优化 | 548-685-977 |
| 7.16 | 10:30-11:30 | 报告4：GUI/CUA Agents环境、轨迹与强化学习 | 同上 |

### 第二阶段：科研实践（7.17 – 7.30）

自主阅读论文 + 系统实践，可与报告研究生讨论，以自主开展为主。本项目选定 **方案 B + A 组合**（详见第四节）。

### 第三阶段：线上总结报告（7.31）

- 形式：线上 PPT 报告，不超过 10 分钟，严格计时
- 框架：标题 → 研究背景 → 进展综述 → 后续工作
- 标题要求：准确表达内容，避免过于宽泛

---

## 三、前期研究背景

### 3.1 投机推理（Speculative Decoding）基础

投机推理的核心思想：用轻量级 draft model 预测多个未来 token，target model 一次并行验证，通过 rejection sampling 保证生成质量无损。

| 论文 | 核心贡献 |
|------|----------|
| Leviathan et al. (ICML 2023) — Fast Inference from Transformers via Speculative Decoding | 奠基之作，draft-verify 框架 |
| Chen et al. (NeurIPS 2023) — Accelerating Large Language Model Decoding with Speculative Sampling | 严格无偏采样证明 |
| Miao et al. (2024) — SpecInfer | tree-structured speculative decoding + token tree verifier |

**关键变体**：

| 路线 | 代表工作 | 特点 |
|------|----------|------|
| Self-Speculative | Medusa (2024) | 在大模型上加 extra decoding heads，并行预测多个后续 token |
| Self-Speculative | EAGLE / EAGLE-2 (2024) | feature-level 草稿生成，比 token-level 更准确 |
| 块并行解码 | DFlash | 打破因果 mask，一次前向生成多个 token |
| 块并行解码 | Domino / DSpark | 改进接受率稳定性 / 降低训练成本 |

### 3.2 端侧 + 异构并行推理

端侧设备（手机、PC）的核心挑战：draft model 推理延迟接近 target model 验证延迟，草稿阶段成为新瓶颈。

| 工作 | 要点 |
|------|------|
| PowerInfer (2024) | 利用 CPU/GPU 异构 + activation sparsity |
| LLM in a Flash (Apple, 2024) | 端侧闪存-内存混合推理 |
| DejaVu (2023) | 基于稀疏激活预测，跳过不参与计算的 MLP 块 |

**与本项目关联**：异构并行投机思路——在端侧利用 AICPU/NPU 等协处理器，在大模型验证的同时并行执行下一轮 draft，将草稿延迟隐藏在验证过程中。这是总结报告 "后续工作" 部分的自然延伸方向。

### 3.3 块并行解码（DFlash 系列）

DFlash 打破因果依赖，一次并行生成多个 token。核心挑战：
- 接受率波动大：打破因果依赖后 token 被拒绝概率上升
- 需要大规模训练：从零训练块并行模型成本高
- 线上部署复杂：需平衡 draft/verify 算力分配和延迟

---

## 四、实践方案：B + A 组合（已确定）

### 4.1 模型选型

| 角色 | 模型 | 预估显存 (FP16) |
|------|------|-----------------|
| Target Model | Qwen2.5-1.5B 或 Llama-3.2-1B | ~3-4GB |
| Draft Model | Qwen2.5-0.5B 或 Llama-3.2-1B 的前几层 | ~1-2GB |
| **总计** | | **~5-6GB** |

> 若显存不足，使用 8-bit 量化（bitsandbytes）腾出余量。

### 4.2 方案 B：手写最小投机推理框架（第一周：7.17 – 7.23）

#### 目标

从零实现完整 draft-verify 循环，深入理解投机推理机制，掌握所有核心细节。

#### 核心算法流程

```
输入: prompt, draft_model, target_model, max_new_tokens, k (draft length)

while len(generated_tokens) < max_new_tokens:
    ┌─────────────────────────────────────────────┐
    │ Step 1: DRAFT — draft_model 自回归生成 k 个 token  │
    │   draft_tokens = []                              │
    │   draft_input = prompt + generated_tokens        │
    │   for i in range(k):                             │
    │       logits = draft_model.forward(draft_input)   │
    │       next_token = sample(logits[-1])             │
    │       draft_tokens.append(next_token)             │
    │       draft_input.append(next_token)              │
    └─────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────┐
    │ Step 2: VERIFY — target_model 一次并行前向      │
    │   full_input = prompt + generated_tokens + draft_tokens │
    │   target_logits = target_model.forward(full_input) │
    │   # 提取对应位置的 logits 用于验证                │
    │   verify_logits = target_logits[-(k+1):-1]       │
    └─────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────┐
    │ Step 3: ACCEPT/REJECT — Rejection Sampling    │
    │   for i in range(k):                             │
    │       draft_token = draft_tokens[i]              │
    │       p_draft = draft_model_probs[i][draft_token] │
    │       p_target = target_probs[i][draft_token]     │
    │       if random() < min(1, p_target / p_draft): │
    │           accept draft_token                       │
    │       else:                                       │
    │           sample new_token ~ norm(max(0, p_target - p_draft)) │
    │           accept new_token; break                  │
    └─────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────┐
    │ Step 4: 拼接已接受的 token，继续下一轮          │
    │   generated_tokens.extend(accepted_tokens)      │
    └─────────────────────────────────────────────┘
```

#### 三种采样策略实现

| 策略 | 难度 | 说明 |
|------|------|------|
| **贪婪解码** | ⭐ | draft 和 target 都取 argmax，无随机性，接受率 100%（draft 正确时） |
| **标准投机采样** | ⭐⭐ | 完整的 rejection sampling，保证与 target model 自回归完全相同的分布 |
| **基于熵的动态 draft length** | ⭐⭐⭐ | 根据上一步接受情况或当前预测熵动态调整 k，减少浪费 |

#### 实验变量与测量指标

| 自变量 | 取值 |
|--------|------|
| draft length k | 3, 5, 7, 10 |
| 温度参数 temperature | 0.0 (greedy), 0.6, 0.8, 1.0 |
| 采样策略 | greedy / rejection sampling / entropy-adaptive |
| **KV Cache 策略（优化点）** | 无 KV Cache / 标准 KV Cache / 增量 KV Cache 复用 |

| 测量指标 | 说明 |
|----------|------|
| **加速比 (Speedup)** | 每次 draft-verify 循环生成的平均 token 数 × (target forward 耗时 / 总耗时) |
| **接受率 (Acceptance Rate)** | 每轮 draft 中被目标模型接受的 token 占比 |
| **平均接受长度** | 每次 draft 中连续被接受的 token 数量（期望值） |
| **浪费率 (Waste Rate)** | draft 中生成的但被拒绝的 token 占比 |
| **Draft 耗时占比** | draft 前向时间占总时间的百分比 |
| **显存峰值 (Peak VRAM)** | 不同配置下的 GPU 显存占用峰值 |
| **吞吐量 (Tokens/s)** | 端到端每秒生成 token 数 |

#### 技术实现要点

```
实现细节:
├── 模型加载: transformers.AutoModelForCausalLM + AutoTokenizer
├── 前向传播: 手动调用 model.model(embed + past_key_values) 获取 hidden states，再用 lm_head 得到 logits
│   └── 重点: 复用 KV Cache（past_key_values）避免重复计算
├── 采样: torch.multinomial / torch.argmax
├── 日志: 每轮记录 draft_tokens / accepted_tokens / 各阶段耗时
└── 测试用例: 准备 10-20 个 prompt，覆盖短/中/长不同长度
```

#### 交付物

1. `speculative_decoding.py` — 完整的手写投机推理框架代码（~200行）
2. `experiment_week1.ipynb` — 实验记录，包含所有自变量组合的测量结果
3. `results_week1/` — 原始数据 CSV + 可视化图表（加速比曲线、接受率曲线）

---

### 4.3 方案 A 精简版：工程化 + 对比实验（第二周：7.24 – 7.30）

#### 目标

在手写版本基础上加入两个关键对比维度，提升实验的深度和报告的工程说服力。

#### 对比实验 1：Self-Speculative Decoding

**思路**：不用独立 draft model，用 target model 的前 N 层做 early exit 预测，模拟 Self-Speculative 方案。

```
实现方式:
1. 加载 target model 完整权重
2. 截取前 1/3 层作为 "draft model":
   ─ 方案 A: 直接取 model.layers[:N]，接一个简单的 token prediction head
   ─ 方案 B (更轻量): 将 target model 的 lm_head 连接到前 N 层的 hidden states 输出
3. 后续流程与方案 B 相同，比较:
   ├── 接受率 vs 独立 draft model
   ├── draft 生成速度（前 N 层比完整小模型更快或更慢？）
   └── 显存占用对比（无需额外加载独立模型）
```

#### 对比实验 2：Draft 耗时精细分解

**思路**：用 `torch.cuda.Event` 或 `time.perf_counter` 精确计时，分解出各阶段耗时。

```
计时拆分:
┌──────────────────────────────────────────────┐
│ Total time per iteration                        │
│   ├── Draft phase                                │
│   │   ├── draft_model.forward() × k 次          │
│   │   └── draft sampling 开销                    │
│   ├── Verify phase                               │
│   │   ├── target_model.forward() 一次           │
│   │   └── rejection sampling 开销               │
│   └── 其他开销（tokenizer, tensor 搬运等）       │
└──────────────────────────────────────────────┘

输出: 饼图 + 堆叠柱状图，直观展示瓶颈所在
预期发现: RTX 4050 上 draft model 耗时极可能成为主要瓶颈 → 自然引出端侧异构并行方案
```

#### 补充实验（可选，视时间而定）

- **KV Cache 影响分析**：对比关闭/开启 KV Cache 情况下，不同序列长度对 draft/verify 速度的影响
- **Batch 实验**：尝试 batch_size > 1 的并行验证（同时对多个 draft 序列做 verify）

#### 交付物

1. `self_speculative.py` — self-speculative 实现
2. `profiling.py` — 耗时分解工具
3. `experiment_week2.ipynb` — 第二周实验记录
4. `results_week2/` — 对比数据 + 分析图表

---

### 4.4 实验中的优化点探索（贯穿两周）

以下 6 个优化点从简单到复杂递增，在实验过程中逐项实现和测量，量化每个优化点的独立收益。这些构成总结报告的"实验发现"核心素材。

#### 优化点 ①：KV Cache 增量复用（难度 ⭐ | 预期收益：Draft 阶段加速 40-60%）

**问题**：Draft model 每生成一个 token 都要完整前向一次，其中 prompt 部分的 KV Cache 被重复计算 k 次。

**优化方案**：
```
朴素实现:
  for i in range(k):
      logits = draft_model(prompt + draft_tokens[:i])  # 每次都重新计算全部 KV

优化后:
  past_kv = draft_model(prompt)                    # 只计算一次 prompt 的 KV
  for i in range(k):
      logits, past_kv = draft_model(draft_token_i, past_key_values=past_kv)  # 增量计算
```

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| 无 KV Cache | draft 阶段耗时基线 |
| 标准 KV Cache（每轮 reset） | draft 阶段耗时 |
| **增量 KV Cache 复用**（优化后） | draft 阶段耗时、加速比提升 |

**实现要点**：使用 HuggingFace `past_key_values` 参数，在 draft 循环内部增量传递。Verify 阶段同样复用 prompt 部分的 KV Cache。

---

#### 优化点 ②：Draft 长度动态调整（难度 ⭐⭐ | 预期收益：减少浪费 15-30%）

**问题**：固定 draft length k 在不同上下文下效果差异大——简单 token（如标点、常用词）容易被猜中，复杂 token 则不然。固定 k 导致"猜错浪费"或"猜少效率不足"。

**优化方案**：

1. **基于上轮接受率的自适应调整**（简单）：
   ```
   if prev_accept_rate > 0.8:
       k = min(k + 1, k_max)
   elif prev_accept_rate < 0.5:
       k = max(k - 1, k_min)
   else:
       k = k  # 保持不变
   ```

2. **基于预测熵的自适应调整**（进阶）：
   ```
   对 draft model 输出的概率分布计算熵:
   entropy = -sum(p_i * log(p_i))
   if entropy > threshold:  # 高熵 = 不确定 → 缩短 draft
       k = k_min
   else:                    # 低熵 = 确定 → 增长 draft
       k = k_max
   ```

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| 固定 k = 3 / 5 / 7 / 10 | 接受率、加速比基线 |
| 基于接受率自适应 | 平均 k 值、实际加速比、浪费率 |
| 基于熵自适应 | 平均 k 值、实际加速比、浪费率 |

---

#### 优化点 ③：Tree-Structured Draft（难度 ⭐⭐ | 预期收益：接受率提升 10-20%）

**问题**：标准投机推理中，draft model 只生成一条线性序列，一旦某个位置被拒绝，后续所有 draft token 全部作废。这种"全有或全无"的模式浪费了 draft 算力。

**优化方案**（参考 SpecInfer 论文）：
```
线性 draft（朴素）:
  draft_model 生成: [A] → [B] → [C] → [D] → [E]
  若 B 被拒绝:     [A] → [✗] → 全部作废

树形 draft（优化）:
  draft_model 在每个位置生成 top-2 候选:
              ┌── [B1] → [C1] → [D1]
  [A] ──┤
              └── [B2] → [C2] → [D2]
  target model 一次验证所有路径，选接受率最高的分支
```

**实现要点**：
- 构建 token tree：用 draft model 在每个位置保留 top-2 候选
- 将 tree 展开为 batch，target model 一次前向验证所有路径
- 用 tree attention mask（非因果 mask）保证各分支独立
- 选择接受 token 最多的路径

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| 线性 draft（k=5） | 接受率、加速比（基线） |
| 树形 draft（top-2, k=5） | 接受率、加速比、显存开销 |

---

#### 优化点 ④：Draft 阶段量化加速（难度 ⭐⭐ | 预期收益：Draft 前向速度 +30-50%，显存 -50%）

**问题**：Draft model 虽然小，但 FP16 精度下仍占 ~1-2GB 显存，且前向计算受限于显存带宽。

**优化方案**：
```
draft_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    load_in_8bit=True,          # 8-bit 量化
    device_map="auto"
)
# 或使用 GPTQ/AWQ 4-bit 量化
```

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| draft FP16 | 前向耗时、显存占用（基线） |
| draft INT8 (bitsandbytes) | 前向耗时、显存占用、接受率是否下降 |
| draft INT4 (GPTQ) | 前向耗时、显存占用、接受率是否下降 |

---

#### 优化点 ⑤：Pipeline 并行（模拟端侧异构）（难度 ⭐⭐⭐ | 预期收益：端到端加速 20-40%）

**问题**：当前 draft-verify 是串行的——先 draft 完 k 个 token，再 verify。Draft 和 Verify 阶段无法重叠，是"draft 成为瓶颈"的根本原因。

**优化方案**（模拟报告1中的异构并行思路）：
```
串行（朴素）:
  │ Draft(t) │ Verify(t) │ Draft(t+1) │ Verify(t+1) │ ...
  
流水线（优化后）:
  │ Draft(t) │ Draft(t+1) │ Draft(t+2) │ ...
  │          │ Verify(t)  │ Verify(t+1)│ Verify(t+2)│ ...
```

**在 RTX 4050 上的模拟实现**：
```
方案：利用 Python 的 threading 实现流水线重叠

1. 第 t 轮: draft_model 生成 draft_tokens(t)
2. 启动 Thread-1: target_model.forward(verify(t))  # 异步验证
3. 同时启动 Thread-2: draft_model.generate(draft(t+1))  # 并行生成下一轮 draft
4. 等待 Thread-1 完成 → 接受/拒绝 token
5. 等待 Thread-2 完成 → draft_tokens(t+1) 已就绪
6. 循环

注意: 在单 GPU 上线程并发会争抢 CUDA 资源（受 Python GIL 影响较小，因为 CUDA 操作会释放 GIL）。
实际重叠取决于 CUDA stream 调度。可使用 torch.cuda.Stream 实现真正的异步。
```

**CUDA Stream 实现方案**（更精确）：
```python
stream_draft = torch.cuda.Stream()
stream_verify = torch.cuda.Stream()

# 第 t 轮 draft 完成时立即启动 verify
with torch.cuda.stream(stream_verify):
    verify_output = target_model(draft_tokens_t)  # 异步验证

# 同时在另一 stream 上启动下一轮 draft
with torch.cuda.stream(stream_draft):
    draft_tokens_next = draft_model(next_input)   # 异步草稿

# 等待 verify 完成，接受 token
torch.cuda.synchronize()
```

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| 串行 draft-verify | 端到端加速比（基线） |
| 线程级流水线 | 端到端加速比、draft/verify 时间重叠率 |
| CUDA Stream 流水线 | 端到端加速比、实际并行效率 |

---

#### 优化点 ⑥：Tree Attention + SpecInfer 简化实现（难度 ⭐⭐⭐ | 预期收益：验证效率提升 2-3×）

**问题**：标准 verify 阶段，k 个 draft token 在 target model 中被逐个串行验证。将树形 draft 与批量验证结合，可以一次验证多条候选路径。

**优化方案**（基于 SpecInfer 的 Token Tree Verifier 思想）：
```
1. Draft 阶段生成 token tree（每个位置 top-2 候选）
2. 将所有候选路径展平为 batch
3. 构建 tree attention mask（非因果 mask 限制各分支独立）
4. target model 一次 batch 前向验证所有路径
5. 选择接受 token 最多的路径

关键: tree attention mask 的构建
  - 对于同一路径上的 token，使用因果 mask
  - 对于不同分支的 token，互相不可见
  - 通过调整 attention mask 实现并行
```

**实验对比**：
| 配置 | 测量指标 |
|------|----------|
| 标准线性 verify | verify 耗时（基线） |
| Tree batch verify | verify 耗时、总吞吐量、显存峰值 |

---

#### 优化点汇总与优先级

| 编号 | 优化点 | 难度 | 预期收益 | 建议时间 | 论文参考 |
|------|--------|------|----------|----------|----------|
| ① | KV Cache 增量复用 | ⭐ | Draft 加速 40-60% | 第1周 | 基础优化 |
| ② | Draft 长度动态调整 | ⭐⭐ | 减少浪费 15-30% | 第1周 | Leviathan 原始理论 |
| ③ | Tree-Structured Draft | ⭐⭐ | 接受率 +10-20% | 第1周末 | SpecInfer |
| ④ | Draft 量化加速 | ⭐⭐ | Draft 速度 +30-50% | 第2周 | GPTQ / AWQ |
| ⑤ | Pipeline 并行（异构模拟） | ⭐⭐⭐ | 端到端 +20-40% | 第2周 | 报告1核心思路 |
| ⑥ | Tree Attention 批量验证 | ⭐⭐⭐ | 验证效率 ×2-3 | 第2周 | SpecInfer / Medusa |

> **建议策略**：优化点 ①② 必做（收益明确、实现简单），优化点 ③④⑤ 选做至少 2 个（体现深度），优化点 ⑥ 视时间选做（技术最复杂，但报告价值最高）。
---

## 五、推荐论文阅读路线

### 第1周（基础，配合方案 B + 优化点①②③）

```
Leviathan et al. (ICML 2023)          ★必读 — 投机推理奠基论文
  ├── 精读: draft-verify 框架、接受率理论推导
  ├── 理解: 为什么 rejection sampling 能保证无损？
  └── 对照: Chen et al. (NeurIPS 2023) 的独立证明

SpecInfer (Miao et al., 2024)         ★配合优化点③⑥
  └── Tree-structured speculative decoding + token tree verifier

综述: "Unlocking Efficiency in LLMs" (arXiv 2024)
  └── 建立全景认知，了解 LLM 推理优化的各大方向
```

### 第2周（深入，配合方案 A + 优化点④⑤⑥）

```
Medusa (2024)
  └── 理解 self-speculative 路线：如何用 extra heads 做多 token 预测

EAGLE / EAGLE-2 (2024)
  └── feature-level 草稿生成为何比 token-level 更准确？

DFlash + Domino + DSpark
  └── 块并行路线：如何打破因果 mask？接受率波动如何解决？

PowerInfer (2024)
  └── 异构并行：CPU/GPU 如何分工？activation sparsity 如何利用？

GPTQ / AWQ 量化论文（配合优化点④）
  └── 理解训练后量化对推理速度和精度的影响
```

---

## 六、总结报告叙事线（PPT 框架）

```
┌──────────────────────────────────────────────────────────┐
│ 报告标题（自拟，参考格式）                                   │
│ 示例: "端侧投机推理性能分析：draft-verify 框架的                                 │
│        实现、瓶颈与异构并行优化初探"                          │
├──────────────────────────────────────────────────────────┤
│                                                             │
│  1. 研究背景 (2 min)                                         │
│     ├── LLM 自回归推理的时延瓶颈                              │
│     ├── 投机推理的基本思想：draft → verify → accept/reject  │
│     └── 端侧部署的特殊挑战：draft model 本身成为瓶颈          │
│                                                             │
│  2. 进展综述 (2 min)                                         │
│     ├── Speculative Decoding 三大家族                       │
│     │   ├── draft-model 路线 (Leviathan, Chen)              │
│     │   ├── self-speculative 路线 (Medusa, EAGLE)           │
│     │   └── 块并行路线 (DFlash, Domino, DSpark)            │
│     └── 端侧异构并行 (PowerInfer, LLM in a Flash)          │
│                                                             │
│  3. 实验与发现 (4 min) — 核心部分                            │
│     ├── 手写 draft-verify 框架的实现要点                      │
│     ├── 优化实验递进（Ablation Study）                        │
│     │   ├── 基础基线：加速比 vs draft length / 接受率 vs temp │
│     │   ├── 优化点①：KV Cache 增量复用 → draft 阶段加速收益  │
│     │   ├── 优化点②：动态 draft length → 浪费率降低效果      │
│     │   ├── 优化点③：Tree-Structured Draft → 接受率提升      │
│     │   ├── 优化点④：Draft 量化 → 显存/速度 tradeoff         │
│     │   └── 优化点⑤：Pipeline 并行模拟 → 端到端加速收益      │
│     ├── 各优化点独立收益对比柱状图                             │
│     ├── Self-Speculative vs 独立 Draft Model 对比          │
│     ├── 耗时分解：瓶颈定位（draft model 成为瓶颈）           │
│     └── 关键发现总结 + 优化建议优先级                         │
│                                                             │
│  4. 后续工作 (2 min)                                         │
│     ├── 针对 "draft 成为瓶颈" 的优化方向                     │
│     │   ├── 端侧异构并行：利用 AICPU/NPU 隐藏 draft 延迟    │
│     │   ├── Pipelined Speculative Decoding：流水线化        │
│     │   └── 更高效的 draft model 设计（EAGLE, Medusa）      │
│     ├── 本实验各优化点在真实端侧部署中的延伸                  │
│     └── 未来可探索的工程方案                                 │
│                                                             │
└──────────────────────────────────────────────────────────┘
```

---

## 七、环境与依赖

### 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| 显存 | 6GB GDDR6 |
| CPU | 笔记本平台 CPU |
| RAM | ≥16GB（推荐） |

### 软件环境

```
Python:         >= 3.10
PyTorch:        >= 2.1.0 (CUDA 12.1)
transformers:   >= 4.40.0
bitsandbytes:   >= 0.43.0 (8-bit 量化备选)
matplotlib:     >= 3.7.0  (可视化)
numpy:          >= 1.24.0
jupyter:        用于实验记录
```

### 候选模型（HuggingFace）

| 用途 | 模型 ID | 参数量 |
|------|---------|--------|
| Target | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B |
| Target (备选) | `meta-llama/Llama-3.2-1B-Instruct` | 1B |
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B |
| Draft (备选) | 手动截取 Llama-3.2-1B 前 N 层 | ~0.3B |

---

## 八、预期产出清单

| 编号 | 产出物 | 类型 | 说明 |
|------|--------|------|------|
| 1 | `speculative_decoding.py` | 代码 | 手写投机推理框架（~200行） |
| 2 | `self_speculative.py` | 代码 | Self-Speculative 实现 |
| 3 | `profiling.py` | 代码 | 耗时精细分解工具 |
| 4 | `experiment_week1.ipynb` | 实验记录 | 第一周完整实验 |
| 5 | `experiment_week2.ipynb` | 实验记录 | 第二周完整实验 |
| 6 | `results_week1/` | 数据 | 加速比/接受率原始 CSV + 图表 |
| 7 | `results_week2/` | 数据 | 对比实验数据 + 瓶颈分析图 |
| 8 | `summary_report.pptx` | PPT | ≤10分钟线上总结报告 |
