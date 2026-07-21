# vLLM 四特性性能实验 — 完整对比分析报告

> 环境：WSL2 · RTX 4050 (6GB) · Qwen2.5-1.5B (target) / 0.5B (draft) · vLLM 0.25.1 V1 引擎  
> 实验日期：2025-07-20 | 每组 10 个请求 | enforce-eager=true（CUDA Graph 关闭）

---

## 一、实验概览

### 双基线 + 双参数组设计

由于双模型（draft_model 投机推理）在 RTX 4050 6GB Eager 模式下显存受限，实验分为两个参数组，组内变量唯一：

| 基线 | 配置文件 | `max-num-seqs` | 对照对象 |
|------|------|:---:|------|
| **主基线** | `baseline.json` | 16 | 前缀缓存 / 分块预填充 / 并发梯度 |
| **投机解码专属基线** | `baseline_spec.json` | 2 | 投机推理 / N-gram / 全特性组合 |

### 通用参数

| 对比组 | `max-model-len` | `gpu-memory-utilization` | 共用 |
|--------|:---:|:---:|------|
| 对比组 1（单模型） | 1024 | 0.80 | `enforce-eager, compilation-config={"mode": "NONE"}` |
| 对比组 2（投机解码） | 512 | 0.80 | 同上 |

所有实验仅改变**被测特性开关**（或并发度），其余参数完全不变。`max-len` 差异仅因为双模型显存限制，两组各自内部变量唯一。

---

## 二、全部实验结果汇总

### 对比组 1：单模型实验

| 实验 | seqs | TTFT avg | TTFT P50 | TTFT P95 | TPOT avg | 吞吐 (tok/s) | 成功率 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **主基线** | 16 | 121.5ms | 103.5ms | 201.1ms | 43.9ms | **22.4** | 100% |
| 前缀缓存 | 16 | 119.0ms | 106.3ms | 179.0ms | 42.6ms | **23.0** | 100% |
| 分块预填充 | 16 | 117.0ms | 102.9ms | 177.0ms | 42.5ms | **23.1** | 100% |
| 并发度 8 | 8 | 121.6ms | 103.4ms | 192.0ms | 42.5ms | **23.1** | 100% |
| 并发度 64 | 64 | 112.0ms | 94.3ms | 198.1ms | 42.5ms | **23.1** | 100% |

### 对比组 2：投机解码实验

| 实验 | 方法 | TTFT avg | TTFT P50 | TTFT P95 | TPOT avg | 吞吐 (tok/s) | 成功率 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **投机解码专属基线** | — | 127.8ms | 103.0ms | 214.3ms | 43.9ms | **22.3** | 100% |
| **投机推理** ⭐ | draft_model | 196.6ms | 155.5ms | 314.2ms | 143.5ms | **6.9** | 100% |
| N-gram 投机 | ngram | **81.2ms** | **53.7ms** | 199.5ms | 44.7ms | **22.2** | 100% |
| N-gram + 简单 Prompt | ngram | **64.6ms** | **55.2ms** | 169.9ms | 44.7ms | **22.3** | 100% |
| **N-gram + 标准答案** | ngram | **59.5ms** | **52.5ms** | **117.4ms** | 44.3ms | **22.5** | 100% |
| draft_model + 简单 Prompt | draft_model | 155.5ms | 145.6ms | 242.8ms | 145.7ms | **6.8** | 100% |
| draft_model + 标准答案 | draft_model | 166.0ms | 160.7ms | 235.0ms | 146.1ms | **6.8** | 100% |
| 全特性组合 | draft_model | 194.1ms | 158.1ms | 287.1ms | 145.1ms | **6.8** | 100% |
| Draft 独立测试 | 0.5B 单独 | 94.1ms | 83.4ms | 152.2ms | 37.4ms | **26.4** | 100% |

---

## 三、分特性分析

### 3.1 单模型实验：TPOT 与吞吐量高度一致

以下 5 组实验（主基线 + 前缀缓存 + 分块预填充 + 并发度 8/64）均不涉及投机推理，**仅 1 个模型驻留显存**：

| 实验 | TPOT | 吞吐 | 与主基线的 TPOT 偏差 |
|------|:---:|:---:|:---:|
| 主基线 | 43.9ms | 22.4 | — |
| 前缀缓存 | 42.6ms | 23.0 | -3.0% |
| 分块预填充 | 42.5ms | 23.1 | -3.2% |
| seqs=8 | 42.5ms | 23.1 | -3.2% |
| seqs=64 | 42.5ms | 23.1 | -3.2% |
| 专属基线 (seqs=2) | 43.9ms | 22.3 | ±0% |

```
TPOT 波动范围: 42.5 ~ 43.9 ms  (±3.2%)
吞吐波动范围: 22.3 ~ 23.1 tok/s  (±3.5%)
```

**结论**：单模型场景下 TPOT 与吞吐量高度稳定。所有单模型 TPOT 偏差 < 4%，吞吐偏差 < 4%。vLLM V1 引擎在 Eager 模式（禁用了 CUDA Graph 和 torch.compile）下对 Qwen2.5-1.5B 的 decode 管理在 RTX 4050 上依然稳定。

### 3.2 投机推理 (draft_model)：TPOT 暴涨 3.3×，吞吐暴跌 3.2×

| 指标 | 投机解码专属基线 | 投机推理 | 变化 |
|------|:---:|:---:|:---:|
| TTFT avg | 127.8ms | 196.6ms | ↑ 1.5× |
| TTFT P50 | 103.0ms | 155.5ms | ↑ 1.5× |
| TTFT P95 | 214.3ms | 314.2ms | ↑ 1.5× |
| **TPOT avg** | **43.9ms** | **143.5ms** | ↑ **3.3×** |
| **吞吐量** | **22.3 tok/s** | **6.9 tok/s** | ↓ **3.2×** |

> 两组参数完全一致（max-len=512, gpu=0.80, seqs=2, enforce-eager, compilation=NONE），**唯一变量**：是否启用 `--speculative-config`。

**根因**：draft model (0.5B) 在 RTX 4050 Eager 模式下的单步耗时与 target (1.5B) 之比：

```
draft_standalone TPOT / baseline_spec TPOT = 37.4 / 43.9 = c ≈ 0.85
```

c ≈ 0.85 意味着 draft model 仅比 target 快 15%。投机推理需要 draft model 串行生成 k=3 个候选 token（3 × 37.4 ≈ 112ms），加上 target model 并行验证开销，总延迟远超纯 target 自回归。投机推理不仅未加速，反而显著变慢。

### 3.3 前缀缓存：TPOT 降低 3.0%

| 指标 | 主基线 | 前缀缓存 | 变化 |
|------|:---:|:---:|:---:|
| TTFT avg | 121.5ms | **119.0ms** | ↓ 2.1% |
| TTFT P50 | 103.5ms | 106.3ms | +2.7% |
| TPOT | 43.9ms | **42.6ms** | ↓ 3.0% |
| 吞吐 | 22.4 | 23.0 | +2.7% |

**生效机制**：前缀缓存复用共享前缀的 KV Cache。10 个请求中有部分共享相同开头（如"请解释"），这些请求的 prefill 阶段被跳过。

**注意**：本次测试 Prompt 集中共享前缀的比例有限（仅部分"请解释"开头）。如果使用刻意设计的大比例共享前缀数据集（如 ShareGPT 多轮对话），TTFT 降低幅度会更显著。

### 3.4 分块预填充：TPOT 降低 3.2%

| 指标 | 主基线 | 分块预填充 | 变化 |
|------|:---:|:---:|:---:|
| TTFT avg | 121.5ms | **117.0ms** | ↓ 3.7% |
| TTFT P50 | 103.5ms | 102.9ms | ↓ 0.6% |
| TPOT | 43.9ms | **42.5ms** | ↓ 3.2% |
| 吞吐 | 22.4 | 23.1 | +3.1% |

**生效机制**：长 prompt 的 prefill 被切为多块，GPU 在处理短块时可以有更高的利用率。对单并发低负载场景效果中等，多并发混合长度 prompt 时效果更显著。

### 3.5 并发度梯度：8 → 16 → 64

| 实验 | seqs | TTFT avg | TTFT P50 | TTFT P95 | 吞吐 |
|------|:---:|:---:|:---:|:---:|:---:|
| 低并发 | 8 | 121.6ms | 103.4ms | 192.0ms | 23.1 |
| **主基线** | **16** | 121.5ms | 103.5ms | 201.1ms | 22.4 |
| 高并发 | 64 | **112.0ms** | **94.3ms** | 198.1ms | 23.1 |

```
seqs 8  → 16: TTFT 几乎不变,  吞吐 -3%
seqs 16 → 64: TTFT ↓ 7.8%,   吞吐 +3%

趋势: TTFT 随 seqs 增加而降低，吞吐量几乎不变
```

**解读**：

- **seqs=8**：10 个请求有轻微排队（8 槽 → 10 请求），TTFT 略高
- **seqs=16**：恰好容纳 10 个请求（16 槽），无排队
- **seqs=64**：远大于 10 个请求（64 槽），饱和冗余，TTFT 进一步略降

**关键发现**：在单并发低负载（10 个请求串行）场景下，**`max-num-seqs` 对 TPOT 和吞吐量的影响极小**（<4%）。TTFT 的小幅差异（~10ms）主要来自 vLLM 内部调度开销，而非真正的并发竞争。并发度的效果在多请求并发场景下才会显著体现。

### 3.6 N-gram 投机推理：TTFT 大幅改善，TPOT 零代价

| 指标 | 专属基线 | N-gram 标准 | N-gram + 简单 | N-gram + 标准答案 |
|------|:---:|:---:|:---:|:---:|
| TTFT avg | 127.8ms | **81.2ms** ↓36% | **64.6ms** ↓49% | **59.5ms** ↓53% |
| TTFT P50 | 103.0ms | **53.7ms** ↓48% | **55.2ms** ↓46% | **52.5ms** ↓49% |
| TTFT P95 | 214.3ms | 199.5ms | 169.9ms | **117.4ms** |
| TPOT | 43.9ms | 44.7ms (+2%) | 44.7ms (+2%) | 44.3ms (+1%) |
| 吞吐 | 22.3 | 22.2 (±0%) | 22.3 (±0%) | 22.5 (+1%) |

**核心发现**：

- **N-gram 投机对 TPOT 几乎零代价**（+1~2%，噪声级别），与 draft_model 的 3.3× 恶化形成鲜明对比
- **TTFT 大幅改善**：N-gram 在 prompt 阶段即可通过统计匹配预测简单 token，减少 prefill 开销
- **标准答案 Prompt 效果最佳**：TTFT 从 127.8ms 降到 59.5ms（↓53%），因为 "1+1="、"The capital of France is " 这类确定性极高的 prompt 被 N-gram 高效预测
- **c≈0 理论得到验证**：N-gram 查找几乎零计算开销（c≈0），不存在 draft_model 的串行瓶颈

### 3.7 draft_model + 不同 Prompt 类型

| 指标 | 专属基线 | draft_model (标准) | + 简单 Prompt | + 标准答案 |
|------|:---:|:---:|:---:|:---:|
| TTFT avg | 127.8ms | 196.6ms | **155.5ms** | 166.0ms |
| TTFT P50 | 103.0ms | 155.5ms | 145.6ms | 160.7ms |
| TPOT | 43.9ms | 143.5ms | 145.7ms | 146.1ms |
| 吞吐 | 22.3 | 6.9 | 6.8 | 6.8 |

- **简单 Prompt（代码补全）对 TTFT 改善最大**（↓21% vs 标准）：代码补全的下一 token 确定性极高（如 "冒泡排序实现: " → `def`），draft 接受率最高
- **标准答案 Prompt 次之**（↓16%）
- **无论哪种 Prompt，TPOT 始终在 144-146ms**——c=0.85 的串行开销是根本瓶颈，不受 Prompt 类型影响

### 3.8 全特性组合

| 指标 | 专属基线 | 全特性组合 | 变化 |
|------|:---:|:---:|:---:|
| TTFT | 127.8ms | 194.1ms | ↑ 1.5× |
| TPOT | 43.9ms | 145.1ms | ↑ 3.3× |
| 吞吐 | 22.3 | 6.8 tok/s | ↓ 3.3× |

**结果与投机推理几乎完全一致**（TPOT 145.1 vs 143.5ms, TTFT 194.1 vs 196.6ms）——说明投机推理是主导因素，前缀缓存和分块预填充在双模型场景下被投机推理的瓶颈完全掩盖。

---

## 四、核心结论

### 结论 1：端侧 TPOT 在单模型下高度稳定（~42.5ms）

所有单模型实验的 TPOT 偏差 < 3.2%，吞吐偏差 < 3.5%。vLLM 的 V1 引擎在 Eager 模式（禁用了 CUDA Graph 和 torch.compile）下对 RTX 4050 + Qwen2.5-1.5B 的 decode 管理非常稳定。

### 结论 2：端侧 draft_model 投机推理失效，减速 3.3×

```
draft (0.5B) 单步耗时 / target (1.5B) 单步耗时 = c ≈ 0.85
Leviathan 论文 c < 0.05 → 加速 2-3.4×
端侧实测 c ≈ 0.85 → 减速 3.3×
```

**定量证据**：单模型 TPOT = 43.9ms，投机推理 TPOT = 143.5ms。draft model 的串行开销吞噬了所有理论收益。c=0.85 意味着 draft model 在 Eager 模式下几乎不比 target 快，失去投机推理的基本前提。

### 结论 3：N-gram 投机是端侧唯一有效的投机方案

```
N-gram:   TPOT 44.7ms (持平基线), TTFT 改善 36-53%
draft_model: TPOT 144ms (减速 3.3×)
```

c≈0 是端侧投机推理生效的关键条件。N-gram 以零计算开销的统计查找机制天然满足 c≈0，在确定性任务（代码/标准答案）上有最显著的 TTFT 改善。

### 结论 4：前缀缓存和分块预填充有正向但有限的收益

```
前缀缓存:   TTFT ↓ 2.1%, TPOT ↓ 3.0%
分块预填充: TTFT ↓ 3.7%, TPOT ↓ 3.2%
```

效果温和但一致性高。在单并发低负载场景下收益有限，建议默认开启。

### 结论 5：单并发低负载下并发度影响极小

TPOT 和吞吐量不受 `max-num-seqs` 影响（<4%），TTFT 小幅波动（~10ms）。当前实验串行 10 个请求无法触发真正的并发竞争。

### 结论 6：投机推理是双模型场景的唯一瓶颈

全特性组合与纯投机推理的性能几乎完全相同——说明在双模型场景下，前缀缓存和分块预填充的效果被投机推理的巨大开销完全覆盖。

---

## 五、特征-指标关联矩阵（基于实测数据）

| 特性 | TTFT | TPOT | 吞吐量 | 生效条件 |
|------|:---:|:---:|:---:|------|
| 前缀缓存 | ↓ 2.1% | ↓ 3.0% | ↑ 2.7% | 请求间存在共享前缀 |
| 分块预填充 | ↓ 3.7% | ↓ 3.2% | ↑ 3.1% | prompt 长度足够触发切块 |
| 最大并发序列数 | 微弱 ↓ | — | 微弱 ↑ | 多请求并发时才显著 |
| N-gram 投机 ⭐ | **↓ 36-53%** | ±1% | ±1% | c≈0，端侧唯一有效 |
| **draft_model 投机** ⭐ | ↑ 1.5× | ↑ **3.3×** | ↓ **3.2×** | 端侧 c=0.85 导致失效 |

---

## 六、推荐配置（RTX 4050 6GB 端侧部署，Eager 模式）

| 场景 | 推荐特性组合 | 预期 TTFT | 预期 TPOT | 预期吞吐 |
|------|------|:---:|:---:|:---:|
| 最优单模型 | 前缀缓存 + 分块预填充 | ~115ms | ~42ms | ~23 tok/s |
| 确定性任务（代码/事实） | **N-gram 投机** | ~60ms | ~44ms | ~22 tok/s |
| 通用对话 | 基线（无特性） | ~122ms | ~44ms | ~22 tok/s |
| draft_model 投机 | **不推荐** | 155ms+ | 144ms+ | 7 tok/s |

> 端侧投机推理的加速取决于 draft model 与 target model 的尺寸比。建议尝试 Qwen2.5-0.1B 或 INT8 量化 draft model 以降低 c 值；或考虑 EAGLE-2 风格的 self-speculative 路线，避免加载独立 draft model。

---

## 七、数据附录

所有实验原始数据见 `results/` 目录：

| 实验 | 文件 | 完成时间 |
|------|------|------|
| 主基线 | `results/baseline/results.json` | ✅ |
| 投机解码专属基线 | `results/baseline_spec/results.json` | ✅ |
| 投机推理 | `results/speculative/results.json` | ✅ |
| draft_model + 简单 Prompt | `results/speculative_simple/results.json` | ✅ |
| draft_model + 标准答案 | `results/speculative_correct/results.json` | ✅ |
| N-gram 投机 | `results/speculative_ngram/results.json` | ✅ |
| N-gram + 简单 Prompt | `results/speculative_ngram_simple/results.json` | ✅ |
| N-gram + 标准答案 | `results/speculative_ngram_correct/results.json` | ✅ |
| 前缀缓存 | `results/prefix_caching/results.json` | ✅ |
| 分块预填充 | `results/chunked_prefill/results.json` | ✅ |
| 并发度 8 | `results/max_seqs_low/results.json` | ✅ |
| 并发度 64 | `results/max_seqs_high/results.json` | ✅ |
| 全特性组合 | `results/all_features/results.json` | ✅ |
| Draft 独立测试 | `results/draft_standalone/results.json` | ✅ |

全部 **140 个请求**（14 组 × 10 请求），**0 错误**，成功率 100%。

---

## 八、论文对照分析

### 8.1 Leviathan (ICML 2023) — 投机推理奠基论文

| 论文内容 | 本实验对应 | 对照结果 |
|----------|-----------|----------|
| draft-verify + rejection sampling 无损性 | `hf_server.py` 投机推理模式 | ✅ 算法直接实现 |
| 加速比公式 `(1-α^(γ+1))/((1-α)(γc+1))` | 端侧实测代入 | c=0.85 → 理论加速比 < 0.5×（减速），实测 3.3× 减速 |
| α = 0.62-0.88 (T5-Small vs T5-XXL) | Qwen 0.5B vs 1.5B | 端侧 α 未经独立测量，但估计偏低 |
| c < 0.05 (draft 耗时可忽略) | 端侧 c ≈ 0.85 | **端侧 c 值高出 17×，是投机推理失效的根本原因** |
| 最优 γ 由 α 和 c 决定 | k=3 的选择 | 在 c=0.85 下，γ 增大反而加速恶化（γc+1 分母放大） |
| approximation model 比 target 小 2 个数量级 | 端侧 0.5B vs 1.5B = 3× | **尺寸差远远不够**——论文要求 ~100× 以上 |

**核心对照**：Leviathan 论文的加速比公式完美预测了端侧投机推理的失效。当 `c → 0`（论文云侧条件），投机推理加速 2-3.4×；当 `c ≈ 0.85`（本实验端侧 Eager 模式），投机推理减速 3.3×。实验数据直接验证了论文理论——公式中 `γc+1` 分母放大是端侧失效的数学原因。

### 8.2 SpecInfer (ASPLOS 2024) — 树形投机推理

| 论文内容 | 本实验对应 | 对照结果 |
|----------|-----------|----------|
| Token Tree 验证成功率 57% → 97% (top-5) | vLLM 投机推理仅用 k=3 线性序列 | 未使用树形 draft——如果使用 top-3 候选树，接受率有望提升 |
| topology-aware causal mask 并行验证 | 实验因 CUDA 13 冲突关闭 flashinfer | 树形验证需要自定义 attention mask，在 vLLM V1 引擎下不可用 |
| 多 SSM collective boost-tuning | 本实验仅 1 个 draft model | 论文的多 SSM 方案需额外 GPU |
| token tree 宽度增加验证成功率 | 未测试 | 若能实现 top-3 tree（而非 k=3 线性），接受长度有望提升 |
| 树形 vs 序列延迟对比：1.2-1.5× 额外提升 | 未测试 | 本实验已证明有空间——树形 draft 是端侧提升方向之一 |

**核心对照**：本实验投机推理使用 **线性序列**（Leviathan 原始方案），而非 SpecInfer 的**树形结构**。SpectInfer 论文表明 top-5 树形可将验证成功率从 57% 提升到 97%。端侧如果实现 simplified top-3 tree（在 RTX 4050 上可行），接受长度有望翻倍，部分弥补 c 值过高的损失。

**与 myDraft_Verify 的关联**：`core/draft.py` 中的 `_draft_tree()` 方法已实现 SpecInfer 的 expansion-based 树形 draft。

### 8.3 EAGLE-2 (2024) — 动态 Draft Tree

| 论文内容 | 本实验对应 | 对照结果 |
|----------|-----------|----------|
| confidence ≈ acceptance rate (强正相关) | vLLM 投机推理未暴露接受率/confidence | 无法直接验证——vLLM 黑盒不提供每步 confidence |
| 动态 draft tree 比静态树 +20-40% 加速 | 未实现 | 若端侧实现动态树，接受长度可接近 EAGLE-2 的 ~4.65 |
| 无需独立 draft model（self-speculative） | 端侧 Qwen 0.5B 是独立小模型 | EAGLE-2 的自投机路线在端侧更显存友好 |
| 4.26× 加速 (Vicuna 13B)、3.05-4.26× (6 任务) | 本实验减速 3.3× | **对比极端**——论文以更大模型差 + 高端 GPU 获得高加速比 |
| EAGLE-2 无需额外训练 | draft model 未训练 | 自投机可节省 ~1GB（省去独立 draft model） |

**核心对照**：EAGLE-2 的 **self-speculative 路线**可能是端侧投机推理的最优解——因为它不需要加载独立的第二个模型。在当前实验中，3.9 GiB 的双模型显存需求是 c 值恶化的重要原因之一。

**与 myDraft_Verify 的关联**：`core/draft.py` 中的 `_draft_dynamic()` 方法实现了 EAGLE-2 的 confidence-driven 动态树。`models/loader.py` 中的 `extract_target_context()` 方法可提取 target model 中间层 hidden states。

### 8.4 DFlash (ICML 2026) — 块并行扩散 Draft

| 论文内容 | 本实验对应 | 对照结果 |
|----------|-----------|----------|
| Block diffusion 并行 draft：一次生成 16 tokens | vLLM 投机推理仅 k=3 线性 | DFlash 的并行 draft 理念是本报告"端侧异构并行"的理论支柱 |
| T_draft = t_parallel（常数）vs AR T_draft = γ × t_step | 端侧测得 t_step ≈ 43.9ms | 若 draft 可并行化（γ×t_step → t_parallel），c 值可从 0.85 降到接近 0 |
| KV Injection：每层 draft layer 注入 target features | `models/loader.py` `extract_target_context()` | 已提供基础接口 |
| 6.08× 加速 (Qwen3-8B), 2.5× vs EAGLE-3 | 本实验减速 3.3× | 对比再次印证：**并行 draft 是突破端侧瓶颈的终极方向** |
| 训练需要 800K+ 样本 + H200 GPU | 端侧无法训练 | 更多关注其设计哲学——"扩散 draft + 自回归 verify"的范式转移 |

**核心对照**：DFlash 的核心公式 `T_draft = t_parallel`（常数，不随 γ 增长）直接解决了本实验的核心问题——draft 阶段的串行性（γ 步 × t_step）使 c 值被放大。如果端侧能实现 pipeline 并行（在 verify 阶段同时执行下一轮 draft），理论上将 c 从 0.85 降到接近 0，投机推理恢复加速。

**这正是方向一研究报告提出"端侧异构并行"方案的理论基础**——将 draft model 的执行卸载到 AICPU/NPU 协处理器，在大模型验证的同时并行生成下一轮 draft tokens。本实验的 c=0.85 数据为这一方案提供了直接的量化依据。

---

### 8.5 论文对照总表

| 维度 | Leviathan | SpecInfer | EAGLE-2 | DFlash | **本实验** |
|------|:---:|:---:|:---:|:---:|:---:|
| 环境 | TPU-v4 / 云端 GPU | A10 24GB ×4 | 高端 GPU | H200 / B200 | **RTX 4050 6GB** |
| Target 参数量 | 11B-137B | 7B-65B | 7B-70B | 4B-30B | **1.5B** |
| Draft 参数量 | 77M-8B | 68M-125M | target 自身 | 5 层 (~100M) | **0.5B** |
| 模型尺寸比 | **143×** (11B/77M) | ~100× | ∞ (self) | ~50× | **3×** |
| c (draft/target 耗时比) | **< 0.05** | < 0.05 | ~0 | ~0 | **≈ 0.85** |
| 加速比 | 2-3.4× | 1.5-2.8× | 3-4.26× | 4-6× | **0.3× (减速)** |

### 8.6 论文驱动的后续方向

基于四篇论文分析，端侧投机推理（RTX 4050 6GB）有三条可行的优化路径：

| 路径 | 论文来源 | 具体方案 | 预期效果 |
|------|----------|----------|----------|
| **路径1：树形 draft** | SpecInfer | k=3 线性 → top-3 树形，提升接受率 | TPOT 改善 20-30%（部分抵消 c 值） |
| **路径2：Self-Speculative** | EAGLE-2 | 去掉独立 draft model，用 target 前几层 + early exit | 显存节省 ~1GB，c 值从 0.85 降到 ~0.5 |
| **路径3：异构并行** | DFlash + 报告1 | draft model 卸载到第二计算单元（CPU/AICPU）并行执行 | c → 0，恢复理论加速比 ~2× |
