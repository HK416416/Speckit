# 模块②：LLM 特性性能实验 (vllm_benchmark/)

> 方向一必做实验：在 RTX 4050 (6GB) + WSL2 上使用 vLLM 测量 4 项核心特性性能

---

## 实验设计：双基线架构

| 基线 | 配置 | 用途 | `max-num-seqs` |
|------|------|------|:---:|
| **主基线** | `baseline.json` | 前缀缓存 / 分块预填充 / 并发梯度实验的对照组 | **16** |
| **投机解码专属基线** | `baseline_spec.json` | 仅与投机推理组做严格单一变量对照 | **4** |

### 通用参数（两条基线及所有实验共享）

| 参数 | 值 |
|------|-----|
| `max-model-len` | 1024 |
| `gpu-memory-utilization` | 0.80 |
| `enforce-eager` | true |
| `compilation-config` | `{"mode": "NONE"}` |
| `enable-prefix-caching` | false |
| `enable-chunked-prefill` | false |

### 启动方式

`vllm_server.py` 自动注入 `VLLM_USE_V2_MODEL_RUNNER=0` 和 `VLLM_USE_FLASHINFER_SAMPLER=0`。

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/<实验>.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/<实验>.json --num-prompts 10
```

### 前置依赖

```bash
sudo apt update && sudo apt install -y build-essential
pip install nvidia-cuda-nvcc-cu12
```

---

## 实验一：主基线 

**配置文件**：`configs/baseline.json`（seqs=16）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline.json --num-prompts 10
```

---

## 实验二：投机解码专属基线 + 投机推理 

**原理**：draft model 预猜 + target 并行验证。**影响**：TPOT ↓↓（端侧实测 ↑，证实 draft bottleneck）。

### Step 1：专属基线

**配置文件**：`configs/baseline_spec.json`（seqs=4，与投机推理组参数完全一致，仅无 speculative-config）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline_spec.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline_spec.json --num-prompts 10
```

### Step 2：投机推理 

**配置文件**：`configs/speculative.json`（seqs=4，仅多 speculative-config）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/speculative.json --num-prompts 10
```

> 对比分析见 [`results/comparison_report.md`](results/comparison_report.md)

---

## 实验三：前缀缓存（待完成）

**原理**：共享前缀的请求复用 KV Cache。**影响**：TTFT ↓↓。

**配置文件**：`configs/prefix_caching.json`（与主基线对齐，仅 `enable-prefix-caching: true`）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/prefix_caching.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/prefix_caching.json --num-prompts 10
```

---

## 实验四：分块预填充（待完成）

**原理**：将长 prompt 的 prefill 切块，避免 GPU 闲置。**影响**：并发 ↑。

**配置文件**：`configs/chunked_prefill.json`（与主基线对齐，仅 `enable-chunked-prefill: true`）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/chunked_prefill.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/chunked_prefill.json --num-prompts 10
```

---

## 实验五：最大并发序列数（待完成）

**原理**：控制同时处理的请求数。**影响**：吞吐量 ↑↑（16→64 时趋于饱和）。

**并发梯度**：

| 等级 | seqs | 配置 | 预期行为 |
|------|:---:|------|------|
| 低并发（排队） | 8 | `max_seqs_low.json` | 轻微排队，TTFT 略高 |
| **主基线（饱和）** | **16** | `baseline.json` | 无排队，GPU 充分利用 |
| 高并发（冗余） | 64 | `max_seqs_high.json` | 远大于请求数，与 16 性能趋近 |

```bash
# 低并发
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/max_seqs_low.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/max_seqs_low.json --num-prompts 10

# 高并发
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/max_seqs_high.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/max_seqs_high.json --num-prompts 10
```

> 三组数据形成完整梯度：8 → 16 → 64，验证「并发数上升 → 吞吐量先升后稳、TTFT 先降后稳」。

---

## 四特性汇总

| 实验 | 配置 | 基线 | 与基线的差异 | 状态 |
|------|------|:---:|------|:---:|
| 主基线 | `baseline.json` | — | seqs=16 | ✅ |
| 投机解码专属基线 | `baseline_spec.json` | — | seqs=4 | 待测 |
| 投机推理 ⭐ | `speculative.json` | baseline_spec | `+speculative-config` | ✅ |
| 前缀缓存 | `prefix_caching.json` | baseline | `prefix-caching=true` | ⏳ |
| 分块预填充 | `chunked_prefill.json` | baseline | `chunked-prefill=true` | ⏳ |
| 并发度 8 | `max_seqs_low.json` | baseline | `max-num-seqs=8` | ⏳ |
| 并发度 64 | `max_seqs_high.json` | baseline | `max-num-seqs=64` | ⏳ |
| 全特性 | `all_features.json` | baseline_spec | 全部开启 | ⏳ |

## 检查清单

- [x] 主基线 ✅
- [ ] 投机解码专属基线
- [x] 投机推理 ✅
- [ ] 前缀缓存
- [ ] 分块预填充
- [ ] 并发度 8 / 16(基线) / 64
- [ ] 全特性组合
- [x] 对比分析 ✅
