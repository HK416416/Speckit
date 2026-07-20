# 模块②：LLM 特性性能实验 (vllm_benchmark/)

> 方向一必做实验：在 RTX 4050 (6GB) + WSL2 上使用 vLLM 测量 4 项核心特性性能

---

## 实验设计：双基线架构

| 基线 | 配置 | `max-num-seqs` | 用途 |
|------|------|:---:|------|
| **主基线** | `baseline.json` | 16 | 前缀缓存 / 分块预填充 / 并发梯度 |
| **投机解码专属基线** | `baseline_spec.json` | 4 | 所有投机推理实验的严格对照 |

### 通用参数

`max-model-len=1024, gpu-memory-utilization=0.80, enforce-eager=true, compilation-config={"mode": "NONE"}`

### 启动方式

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/<实验>.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/<实验>.json --num-prompts 10
```

---

## 实验一：主基线

**配置**：`configs/baseline.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline.json --num-prompts 10
```

---

## 实验二：投机解码专属基线 + 投机推理

### 专属基线

**配置**：`configs/baseline_spec.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/baseline_spec.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/baseline_spec.json --num-prompts 10
```

### 投机推理 (draft_model)

**配置**：`configs/speculative.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/speculative.json --num-prompts 10
```

---

## 实验三：前缀缓存

**配置**：`configs/prefix_caching.json`（与主基线对齐，仅 `enable-prefix-caching: true`）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/prefix_caching.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/prefix_caching.json --num-prompts 10
```

---

## 实验四：分块预填充

**配置**：`configs/chunked_prefill.json`（与主基线对齐，仅 `enable-chunked-prefill: true`）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/chunked_prefill.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/chunked_prefill.json --num-prompts 10
```

---

## 实验五：并发度梯度

### 低并发

**配置**：`configs/max_seqs_low.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/max_seqs_low.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/max_seqs_low.json --num-prompts 10
```

### 高并发

**配置**：`configs/max_seqs_high.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/max_seqs_high.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/max_seqs_high.json --num-prompts 10
```

---

## 实验六：轻量级投机推理优化（N-gram + 简单 Prompt）

> **目标**：验证 c≈0（N-gram）和 α↑（简单 Prompt）对端侧投机推理的改善效果

### 6.1 Draft Model 独立测试

**配置**：`configs/draft_standalone.json`（0.5B 独立运行，计算 c 值）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/draft_standalone.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/draft_standalone.json --num-prompts 10
```

### 6.2 N-gram + 标准 Prompt

**配置**：`configs/speculative_ngram.json`（c≈0，vLLM 内置 n-gram 引擎，标准 Prompt）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative_ngram.json
python vllm_benchmark/benchmark.py \
  --config vllm_benchmark/configs/speculative_ngram.json \
  --prompts-file vllm_benchmark/prompts/test_prompts.jsonl --num-prompts 10
```

### 6.3 N-gram + 简单 Prompt（代码补全）

**配置**：`configs/speculative_ngram_simple.json`（c≈0 + α↑，代码补全场景）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative_ngram_simple.json
python vllm_benchmark/benchmark.py \
  --config vllm_benchmark/configs/speculative_ngram_simple.json \
  --prompts-file vllm_benchmark/prompts/simple_prompts.jsonl --num-prompts 10
```

### 6.4 draft_model + 简单 Prompt（代码补全）

**配置**：`configs/speculative_simple.json`（0.5B draft + 代码补全 Prompt）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative_simple.json
python vllm_benchmark/benchmark.py \
  --config vllm_benchmark/configs/speculative_simple.json \
  --prompts-file vllm_benchmark/prompts/simple_prompts.jsonl --num-prompts 10
```

### 6.5 N-gram + 标准答案 Prompt

**配置**：`configs/speculative_ngram_correct.json`（c≈0 + 极端确定性，如 "1+1="、"The capital of France is "）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative_ngram_correct.json
python vllm_benchmark/benchmark.py \
  --config vllm_benchmark/configs/speculative_ngram_correct.json \
  --prompts-file vllm_benchmark/prompts/correct_prompts.jsonl --num-prompts 10
```

### 6.6 draft_model + 标准答案 Prompt（新）

**配置**：`configs/speculative_correct.json`（c≈0.37 + 极端确定性，"1+1="、"The capital of France is "）

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/speculative_correct.json
python vllm_benchmark/benchmark.py \
  --config vllm_benchmark/configs/speculative_correct.json \
  --prompts-file vllm_benchmark/prompts/correct_prompts.jsonl --num-prompts 10
```

---

## 实验七：全特性组合

**配置**：`configs/all_features.json`

```bash
python vllm_benchmark/vllm_server.py --config vllm_benchmark/configs/all_features.json
python vllm_benchmark/benchmark.py --config vllm_benchmark/configs/all_features.json --num-prompts 10
```

---

## 四特性汇总

| 实验 | 配置 | 对照基线 | 状态 | TTFT | TPOT | 吞吐 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| 主基线 | `baseline.json` | — | ✅ | 51ms | 17.9ms | 54.4 |
| 投机解码专属基线 | `baseline_spec.json` | — | ✅ | 62ms | 17.9ms | 54.1 |
| 投机推理 (draft_model) | `speculative.json` | baseline_spec | ✅ | 131ms | 84.2ms | 11.5 |
| N-gram + 标准 | `speculative_ngram.json` | baseline_spec | ✅ | 44ms | 23.7ms | 42.4 |
| N-gram + 代码 | `speculative_ngram_simple.json` | baseline_spec | ✅ | 31ms | 22.2ms | 44.5 |
| draft_model + 代码 | `speculative_simple.json` | baseline_spec | ✅ | 101ms | 81.4ms | 12.0 |
| **N-gram + 标准答案** | `speculative_ngram_correct.json` | baseline_spec | ✅ | 32ms | 22.2ms | 44.6 |
| **draft_model + 标准答案** | `speculative_correct.json` | baseline_spec | &#9203; | — | — | — |
| Draft 独立测试 | `draft_standalone.json` | — | ✅ c=0.37 | 43ms | 6.7ms | 143.1 |
| 前缀缓存 | `prefix_caching.json` | baseline | ✅ | 43ms | 17.8ms | 54.7 |
| 分块预填充 | `chunked_prefill.json` | baseline | ✅ | 46ms | 17.9ms | 54.6 |
| 并发度 8/64 | `max_seqs_low/high.json` | baseline | ✅ | — | — | — |
| 全特性 | `all_features.json` | baseline_spec | ✅ | 131ms | 83.3ms | 11.6 |

## 检查清单

- [x] 主基线 ✅
- [x] 投机解码专属基线 ✅
- [x] 投机推理 (draft_model) ✅
- [x] N-gram + 标准 Prompt ✅
- [x] N-gram + 简单 Prompt ✅
- [x] draft_model + 简单 Prompt ✅
- [x] N-gram + 标准答案 Prompt ✅
- [x] Draft 独立测试 ✅ c=0.37
- [x] 全特性组合 ✅
