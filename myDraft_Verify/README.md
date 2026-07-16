# myDraft_Verify — 手写投机推理 (Speculative Decoding) 框架

方案B: 从零实现 draft-verify 框架，在 RTX 4050 (6GB) 上系统性测量投机推理性能。

## 项目结构

```
myDraft_Verify/
├── speculative_decoding.py   # 核心框架: SpeculativeDecoder 类
├── run_experiments.py        # 实验运行器: 网格搜索 + 基线对比 + 绘图
├── requirements.txt          # Python 依赖列表
└── README.md                 # 本文件
```

## 快速开始

### 1. 初始化 Conda（PowerShell）

首次在 PowerShell 中使用 conda，需先执行初始化并重启终端：

```powershell
# 第一步：初始化 conda（仅需执行一次）
conda init

# 第二步：关闭当前 PowerShell 窗口，重新打开一个新的 PowerShell 窗口
# （conda init 修改了 PowerShell profile，重启后生效）

# 第三步：验证 conda 可用
conda --version
```

### 2. 创建 Conda 虚拟环境并安装依赖

```powershell
# 创建名为 spec_dec 的虚拟环境 (Python 3.10)
conda create -n spec_dec python=3.10 -y

# 激活环境
conda activate spec_dec

# 安装 PyTorch (CUDA 12.1 版本, 适配 RTX 4050 6GB)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其余依赖
pip install -r requirements.txt
```

> **验证安装是否成功**:
> ```powershell
> python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}')"
> python -c "import transformers; print(f'transformers {transformers.__version__}')"
> ```
> 预期输出: `PyTorch 2.x.x, CUDA=True`

> **如果没有 GPU 或使用 CPU 模式**，安装 PyTorch CPU 版本:
> ```powershell
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. 快速验证 (单组参数, ~2分钟)

```bash
python run_experiments.py --quick --verbose
```

### 3. 完整实验

```bash
python run_experiments.py
```

### 4. CPU-only 模式 (无 GPU)

```bash
python run_experiments.py --device cpu --quick --max_new_tokens 32
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target_model` | `Qwen/Qwen2.5-1.5B-Instruct` | Target model HuggingFace ID |
| `--draft_model` | `Qwen/Qwen2.5-0.5B-Instruct` | Draft model HuggingFace ID |
| `--output_dir` | `./results` | 结果输出目录 |
| `--device` | `cuda` | 推理设备 (`cuda` / `cpu`) |
| `--quick` | - | 快速模式: 仅跑 2 个 prompt 单组参数 |
| `--no-viz` | - | 跳过 matplotlib 绘图 |
| `--max_new_tokens` | `128` | 最大生成 token 数 |
| `--verbose` | - | 打印每轮 draft/verify 详情 |

## 代码 API 使用

```python
from speculative_decoding import SpeculativeDecoder, SamplingStrategy

# 初始化
decoder = SpeculativeDecoder(
    target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cuda",
)

# 投机推理
result = decoder.generate(
    prompt="请解释什么是机器学习？",
    max_new_tokens=128,
    draft_k=5,
    temperature=0.0,
    strategy=SamplingStrategy.STANDARD,
)

print(f"生成文本: {result.generated_text}")
print(f"耗时: {result.total_time_ms:.0f}ms")
print(f"吞吐: {result.tokens_per_second:.1f} tok/s")
print(f"接受率: {result.avg_acceptance_rate:.1%}")

# 自回归基线 (用于计算加速比)
baseline = decoder.generate_autoregressive_baseline(
    prompt="请解释什么是机器学习？",
    max_new_tokens=128,
)
speedup = baseline.total_time_ms / result.total_time_ms
print(f"加速比: {speedup:.2f}x")
```

## 三种采样策略

| 策略 | 枚举值 | 原理 | 质量保证 |
|------|--------|------|----------|
| **贪婪解码** | `SamplingStrategy.GREEDY` | draft & target 均取 argmax | 无损 (draft 正确时) |
| **标准投机采样** | `SamplingStrategy.STANDARD` | 完整 rejection sampling | 严格无损 (分布等价) |
| **熵自适应** | `SamplingStrategy.ENTROPY_ADAPTIVE` | 标准采样 + 动态调整 k | 严格无损 |

## 已实现的优化点

| 优化点 | 位置 | 说明 |
|--------|------|------|
| ① KV Cache 增量复用 | `_draft()` 方法 | 避免重复计算 prompt KV |
| ② Draft 长度动态调整 | `_adjust_k()` 方法 | 基于接受率自适应 |

## 实验输出

运行 `run_experiments.py` 后，在 `--output_dir` (默认 `./results`) 中生成:

- `experiment_results.csv` — 所有实验原始数据
- `experiment_summary.json` — 汇总统计 + 最佳配置
- `experiment_plots.png` — 4张图表 (加速比/接受率/温度影响/吞吐对比)

## 模型选型建议 (RTX 4050 6GB)

| 角色 | 模型 | 显存 |
|------|------|------|
| Target | `Qwen/Qwen2.5-1.5B-Instruct` | ~3GB |
| Draft | `Qwen/Qwen2.5-0.5B-Instruct` | ~1GB |

备选: Llama-3.2 系列 (`meta-llama/Llama-3.2-1B-Instruct` + `Llama-3.2-3B` 截取前几层)

## 论文参考

- Leviathan et al. (ICML 2023) — Fast Inference from Transformers via Speculative Decoding
- Chen et al. (NeurIPS 2023) — Accelerating Large Language Model Decoding with Speculative Sampling
