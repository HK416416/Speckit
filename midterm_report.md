# 中期报告：投机推理端侧性能实验

> 方向一：端侧投机加速 LLM 推理 — RTX 4050 Laptop GPU (6GB)  
> 周期：2025.07.15 – 07.20（第一阶段）

---

## 一、从论文中学到了什么

### 1.1 投机推理的核心逻辑

通过精读四篇论文（Leviathan ICML 2023 → SpecInfer ASPLOS 2024 → EAGLE-2 2024 → DFlash ICML 2026），我理解了投机推理加速 LLM 推理的完整演进脉络：

```
串行自回归 decode 的痛点:
  每生成一个 token 都需要一次完整的前向传播
  → GPU 计算单元大量空闲等待
  → 延迟完全由 TPOT（Time Per Output Token）决定

投机推理的解法:
  用一个小/快的 draft model 先"猜"几个 token
  → target model 一次并行验证所有候选
  → 猜对的 token 被接受，猜错的被丢弃并重新采样
  → 理论加速 = 接受长度 / (1 + γ × c)
     其中 γ = 每轮候选数, c = draft耗时 / target耗时
```

### 1.2 三条路线的演进

| 路线 | 代表论文 | 核心思想 | 瓶颈 |
|------|----------|----------|------|
| **Draft-Model** (独立小模型) | Leviathan → SpecInfer | 用另一个模型"猜" | draft 自身串行开销 |
| **Self-Speculative** (自投机) | EAGLE-2 | 用 target 自身特征"猜"自己 | 需训练 draft head |
| **Block-Diffusion** (扩散并行) | DFlash | 扩散模型一次生成多个 token | 训练成本极高 |

### 1.3 最关键的公式——Leviathan 加速比

```
实际加速比 = (1 - α^(γ+1)) / ((1 - α) × (γ × c + 1))

其中:
  α = draft model 的 token 接受率 (0~1)
  γ = 每轮草稿 token 数
  c = draft model 单步耗时 / target model 单步耗时
```

**论文的隐含前提**：c < 0.05（云端：draft 比 target 快 20× 以上）。但论文中从未讨论"如果 c 比较大，会发生什么"。这直接引出了我的实验设计。

---

## 二、实验动机：验证 c 值对端侧投机推理的影响

### 2.1 为什么做这个实验

论文中所有实验的 c 值都极小（< 0.05），这是云端环境的天然条件：
- Target 模型巨大（11B-137B），draft 模型极小（68M-8B）
- 硬件为 TPU-v4/H200/A10×4，算力充裕

**但端侧（RTX 4050 6GB）完全不同**：
- 只能放下 Qwen2.5-1.5B (target) + Qwen2.5-0.5B (draft)
- 两个模型的尺寸比仅为 **3×**（而非论文的 100×+）
- 小模型在端侧 GPU 上并非"瞬间完成"

### 2.2 实验设计

我选择了 Qwen2.5-1.5B-Instruct (target) 和 Qwen2.5-0.5B-Instruct (draft)，在 vLLM 0.25.1 V1 引擎上测量：

- **4 项核心特性**：前缀缓存、分块预填充、并发度梯度、投机推理
- **3 类投机方法**：draft_model（独立小模型）、N-gram（统计查找）、全特性组合
- **3 种 Prompt 类型**：标准对话、代码补全、标准答案（极端确定性）

实验环境严格遵循单一变量原则——每个对比组内仅改变被测特性，其余参数完全一致。

---

## 三、实验中遇到的问题

### 3.1 vLLM 版本问题：V1 引擎无法使用 V2 特性

vLLM 0.25.1 默认使用 V1 引擎。通过设置环境变量 `VLLM_USE_V2_MODEL_RUNNER=0` 尝试切换 V2 失败——V2 在 WSL2 环境下有 pinned memory 不兼容问题。最终全程使用 V1 引擎 + Eager 模式。

### 3.2 enforce-eager 参数被静默忽略（已修复）

**问题**：所有配置文件中都设置了 `"enforce-eager": true`（关闭 CUDA Graph），但 `vllm_server.py` 的 `build_vllm_command()` 函数完全没有处理这个参数，导致 vLLM 实际在 CUDA Graph 模式下运行。

**影响**：初次测试的 baseline TPOT = 17.9ms，修复后 TPOT = 43.9ms——差距 2.5×。CUDA Graph 对小模型的加速尤其显著（0.5B draft: 6.7ms → 37.4ms，差距 5.6×）。

**修复**：在 `build_vllm_command()` 中添加了 `enforce-eager` 和 `compilation-config` 的处理逻辑。

### 3.3 双模型显存不足（OOM）

**问题**：draft_model 投机推理需要同时加载两个模型（1.5B + 0.5B），占用约 3.9 GiB。RTX 4050 只有 6GB，其中 ~1 GiB 被 WSL2 图形栈占用，实际可用约 4.95 GiB。

**修复**：将对比组 2 的 `max-model-len` 从 1024 降到 512，`max-num-seqs` 从 4 降到 2，同时将 `gpu-memory-utilization` 从 0.80 调整到合适值（最终维持 0.80），确保 KV Cache 有足够空间。

### 3.4 流式输出计时精度问题（hf_server.py）

`hf_server.py` 原本先生成完整文本，再在流式输出中人为插入 `asyncio.sleep(0.01)`。修复为真正的逐 token 生成迭代器 `generate_stream()`，使 benchmark 的 TTFT/TPOT 计时精确反映真实模型推理耗时。

### 3.5 其他修复

- 三处 `[HFSever]` → `[HFServer]` 拼写错误
- `benchmark.py` TPOT 计算公式：分母 `token_count` → `token_count - 1`
- `analyze.py` 加速比分析的基线查找：从硬编码 `experiments[0]` 改为按名称匹配
- `run_all.bat`（Windows）→ `run_all.sh`（WSL2/Linux），从 7 组扩展到 14 组实验

---

## 四、核心实验结果

### 4.1 单模型性能基线（Eager 模式）

| 实验 | TTFT | TPOT | 吞吐 |
|------|:---:|:---:|:---:|
| 主基线 (seqs=16) | 121.5ms | 43.9ms | 22.4 tok/s |
| 前缀缓存 | 119.0ms (↓2.1%) | 42.6ms (↓3.0%) | 23.0 tok/s |
| 分块预填充 | 117.0ms (↓3.7%) | 42.5ms (↓3.2%) | 23.1 tok/s |
| seqs=8 / 16 / 64 | 112~122ms | 42.5~43.9ms | 22.4~23.1 tok/s |

**结论**：单模型场景下 TPOT 高度稳定（偏差 < 3.2%）。前缀缓存和分块预填充有温和的正向收益。

### 4.2 投机推理对比 ⭐

| 方案 | TTFT | TPOT | 吞吐 | vs 基线 |
|------|:---:|:---:|:---:|:---:|
| **专属基线** | 127.8ms | **43.9ms** | **22.3** | — |
| draft_model | 196.6ms (↑54%) | **143.5ms** (↑227%) | **6.9** (↓69%) | 减速 3.3× |
| N-gram 标准 | **81.2ms** (↓36%) | 44.7ms (+2%) | 22.2 (±0%) | 持平 |
| N-gram + 代码 | **64.6ms** (↓49%) | 44.7ms (+2%) | 22.3 (±0%) | 持平 |
| **N-gram + 标准答案** | **59.5ms** (↓53%) | 44.3ms (+1%) | 22.5 (+1%) | 持平 |

### 4.3 c 值计算

```
c = draft_standalone TPOT / baseline_spec TPOT
  = 37.4ms / 43.9ms
  = 0.85
```

draft model (0.5B) 在 Eager 模式下仅比 target (1.5B) 快 15%。投机推理需要 draft 串行生成 k=3 个候选（3 × 37.4 = 112ms），加上 target 并行验证，总延迟远超纯 target 自回归（43.9ms/tok）。

### 4.4 关键数据验证

**高并发 TTFT 反而最低（112ms）**：`max-num-seqs=64` 时为 64 个序列预分配 KV Cache 池。虽然是 10 个串行请求，但较大的空槽位池减少了内存碎片整理和调度锁争用，导致请求间调度间隙更短。仅适用于极低压场景。

**全特性组合 TPOT=145.1ms ≈ 纯投机 TPOT=143.5ms**：前缀缓存节省的 ~2ms Prefill 时间被 draft_model 的串行开销完全淹没——投机推理是双模型场景的唯一瓶颈。

---

## 五、未来实验思考

### 5.1 内存带宽墙的发现

Draft 独立测试显示 0.5B 模型 TPOT=37.4ms，仅比 1.5B (43.9ms) 快 15%。但参数量是 3× 关系——理论速度差应为 3×。这说明 **RTX 4050 上的小模型瓶颈不在算力 (Compute-bound)，而在内存带宽 (Memory-bandwidth-bound)**。

- 0.5B 需要搬动 ~3GB 权重进出显存
- 1.5B 需要搬动 ~9GB
- 显存带宽是固定的（RTX 4050: ~192 GB/s）
- 权重搬运时间在总延迟中占比很高

**启示**：后续优化 draft_model 不能靠换更小的模型（如 0.1B）——权重搬运仍是瓶颈。必须走更根本的路线。

### 5.2 选择方向：异构并行（GPU + CPU）

基于实验数据，我排除了三条路线中的两条：

| 路线 | 排除/选择原因 |
|------|--------------|
| ❌ 树形 Draft | 接受率提升（~50%）无法抵消 c=0.85 的串行开销。k=3 线性 143ms → top-3 树形仍需 ~80ms（仍有 2× 减速） |
| ❌ Self-Speculative | 需要大量训练数据，且 EAGLE-2 的 draft head 仍需在 GPU 上运行——仍是 memory-bandwidth-bound |
| ✅ **异构并行** | GPU 上 target 自回归时，CPU 同时跑 draft——**两个计算单元并行工作，c_effective → 0** |

**选择异构并行的核心理由**：当前实验的 c=0.85 瓶颈来自 draft 必须在同一个 GPU 上**串行等待**。CPU 免费且空闲——把 draft 移过去，GPU 做一轮 verify 的 43.9ms 内，CPU 有足够时间做完 3 步 draft。

### 5.3 异构并行可行性分析

**CPU 规格**：AMD Ryzen 7 7735H（8 核 16 线程，16MB L3，AVX2）

**0.5B 模型在 CPU 上的推理估算**：

```
模型大小:    0.92 GiB (float16) → 完全放在内存中，无需 GPU 显存
每次 token 的权重搬运: ~0.92 GB / 16 层 ≈ 58 MB/层
CPU 内存带宽:   DDR5-4800 双通道 ≈ 51.2 GB/s (理论)
单 token 预估延迟: 30~60ms（8 核 + AVX2 加速矩阵运算）
```

**异构并行的关键时间线**：

```
GPU (Target 1.5B):
  │── verify 43.9ms ──│── verify 43.9ms ──│── verify 43.9ms ──│
  │    等待 draft      │
  
CPU (Draft 0.5B):
  │── draft 3 tokens ──│── draft 3 tokens ──│
  │   ~3 × 40ms = 120ms  │
  
由于 GPU verify 和下一代 CPU draft 可以完全并行:
  GPU 看到的有效 c 值 = 0（draft 耗时被 verify 掩盖）
  理论加速比 ≈ 接受长度 = k × α
```

**成功条件**：

```
CPU 3×draft 耗时 < GPU 1×verify 耗时 + 传输延迟
        120ms      <     43.9ms       +   ~5ms (CPU→GPU 拷贝)
                   <     48.9ms

当前条件：120ms > 48.9ms → CPU 是瓶颈，需要优化
```

**优化方案**：

| 优化 | 方法 | CPU draft 延迟预期 |
|------|------|:---:|
| 量化 | 0.5B 模型 INT8 量化（0.5 GiB） | ~20ms/tok × 3 = 60ms |
| **量化 + 减层** | 仅用前 8 层做 draft | **~12ms/tok × 3 = 36ms ✅** |
| ONNX Runtime | 用 ONNX 优化 CPU 推理图 | 额外 20-30% 提升 |

**最优方案（量化 + 减层）**：CPU 36ms < GPU 43.9ms，draft 完全被 GPU verify 掩盖，c_effective = 0。

### 5.4 预期效果

```
当前 (纯 GPU draft_model):   TPOT = 143.5ms, 吞吐 = 6.9 tok/s, 减速 3.3×
异构并行 (CPU draft):         TPOT ≈ 43.9ms, 吞吐 ≈ 22 tok/s, 持平基线
异构并行 (k=5 候选):          TPOT ≈ 25ms,   吞吐 ≈ 40 tok/s, 加速 ~1.8×
```

**实验计划**：

1. 在 CPU 上跑 Qwen2.5-0.5B-Instruct（量化），实测单 token 延迟验证估算
2. 实现 `hf_server.py` 的异构版本：GPU 上 target autoregressive + CPU 上 draft 并行
3. 对比不同 k 值（3/5/8）的加速效果
4. 与纯 GPU baseline 对比，验证 c_effective → 0 的假设

后续计划在 [`myDraft_Verify/`](../myDraft_Verify/) 中实现异构并行投机推理框架。

---

## 附录：实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4050 Laptop (6GB, ~192 GB/s 带宽) |
| CPU | AMD Ryzen 7 7735H (8 核 16 线程, 16MB L3, AVX2, DDR5-4800) |
| 驱动 | 595.97, CUDA 13.2 |
| 系统 | WSL2 Ubuntu 22.04 |
| 推理框架 | vLLM 0.25.1 V1 引擎 |
| Target 模型 | Qwen/Qwen2.5-1.5B-Instruct (~2.88 GiB) |
| Draft 模型 | Qwen/Qwen2.5-0.5B-Instruct (~0.92 GiB) |
| 模式 | enforce-eager=true, compilation=NONE, torch.bfloat16 |
| 测试规模 | 14 组实验 × 10 请求 = 140 请求, 0 错误 |
