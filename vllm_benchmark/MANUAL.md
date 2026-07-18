# vLLM 特性性能实验模块 (vllm_benchmark/)

> 方向一必做实验：在 RTX 4050 (6GB) 上测量 vLLM 4 项核心特性的性能影响

---

## 文件结构

```
vllm_benchmark/
├── MANUAL.md              ← 本手册：模块使用说明
├── vllm_server.py         ← vLLM 服务启动器（支持不同特性配置）
├── benchmark.py           ← Benchmark 工具（发送请求 + 收集指标）
├── analyze.py             ← 结果分析工具（生成对比表格 + 图表）
├── configs/               ← 实验配置文件
│   ├── baseline.json      ← 基线配置（所有特性关闭）
│   ├── prefix_caching.json
│   ├── chunked_prefill.json
│   ├── max_seqs.json
│   ├── speculative.json   ← 投机推理配置 ⭐
│   └── all_features.json  ← 全特性组合
├── prompts/               ← 测试 Prompt 集
│   └── test_prompts.jsonl
├── scripts/               ← 自动化脚本
│   ├── run_all.bat        ← Windows 一键运行
│   └── run_all.sh         ← Linux/Mac 一键运行
└── results/               ← 结果输出目录（自动创建）
```

---

## 环境要求

### 1. Conda 环境

```powershell
# 激活已创建的 spec_dec 环境
conda activate spec_dec

# 确认 PyTorch CUDA 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# 预期输出: CUDA: True
```

### 2. 安装 vLLM

```powershell
# vLLM 0.5.x 或更新版本
pip install vllm>=0.5.0
```

### 3. 验证 GPU

```powershell
nvidia-smi
# 预期: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
```

---

## 使用方式

### 快速验证（推荐先做）

测试 vLLM 是否能正常启动并响应请求：

```powershell
# 1. 启动 vLLM 服务 (基线配置，后台运行)
python vllm_server.py --config configs/baseline.json

# 2. 在另一个终端中运行 Benchmark
python benchmark.py --config configs/baseline.json --num-prompts 10

# 3. 分析结果
python analyze.py --results-dir results/baseline
```

### 完整实验（所有特性逐个测试）

```powershell
# Windows: 一键运行全部实验
scripts\run_all.bat

# 或手动逐个运行每个配置
python vllm_server.py --config configs/baseline.json
python benchmark.py --config configs/baseline.json --num-prompts 50
# ... 等待完成后再启动下一个配置 ...
python vllm_server.py --config configs/speculative.json
python benchmark.py --config configs/speculative.json --num-prompts 50
```

> **重要**：由于 RTX 4050 只有 6GB 显存，**每次只能运行一个 vLLM 实例**。实验需串行执行。

---

## 命令行参数

### vllm_server.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 实验配置文件路径 (JSON) | 必填 |
| `--port` | 服务端口 | 8000 |
| `--host` | 服务地址 | 127.0.0.1 |

### benchmark.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 实验配置文件路径 | 必填 |
| `--num-prompts` | 测试 prompt 数量 | 50 |
| `--concurrency` | 并发请求数 | 1 |
| `--output-dir` | 结果输出目录 | 自动（基于配置名） |
| `--api-base` | vLLM API 地址 | http://127.0.0.1:8000 |

### analyze.py

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--results-dir` | 结果数据目录 | 必填 |
| `--compare-dirs` | 多个结果目录对比 | 无 |
| `--output` | 分析报告输出路径 | 自动 |

---

## 配置文件说明

### baseline.json（基线）

```json
{
  "experiment_name": "baseline",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "vllm_args": {
    "max-model-len": 2048,
    "gpu-memory-utilization": 0.90,
    "enable-prefix-caching": false,
    "enable-chunked-prefill": false,
    "max-num-seqs": 256
  }
}
```

### speculative.json（投机推理 ⭐ 核心）

```json
{
  "experiment_name": "speculative",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "vllm_args": {
    "max-model-len": 2048,
    "gpu-memory-utilization": 0.90,
    "speculative-config": {
      "model": "Qwen/Qwen2.5-0.5B-Instruct",
      "num_speculative_tokens": 5,
      "method": "draft_model"
    }
  }
}
```

---

## 核心指标采集方式

vLLM 提供了 `/metrics` 端点（Prometheus 格式），Benchmark 脚本会自动从中提取：

| 指标 | Metrics 字段 | 含义 |
|------|-------------|------|
| TTFT | `vllm:time_to_first_token_seconds_sum` / `_count` | 首 token 平均延迟 |
| TPOT | `vllm:time_per_output_token_seconds_sum` / `_count` | 每 token 平均生成时间 |
| 吞吐量 | `vllm:generation_tokens_total` / 时间间隔 | 每秒生成 token 数 |

---

## 实验执行检查清单

- [ ] 确认 GPU 驱动正常 (`nvidia-smi`)
- [ ] 确认 spec_dec 环境已激活
- [ ] 下载 Qwen2.5-1.5B 和 Qwen2.5-0.5B 模型
- [ ] 测试基线配置 `baseline.json`
- [ ] 测试前缀缓存 `prefix_caching.json`
- [ ] 测试分块预填充 `chunked_prefill.json`
- [ ] 测试不同并发度 `max_seqs.json` (--max-num-seqs 8/16/32)
- [ ] 测试投机推理 `speculative.json` ⭐
- [ ] 测试全特性组合 `all_features.json`
- [ ] 运行 `analyze.py --compare-dirs` 生成对比报告
