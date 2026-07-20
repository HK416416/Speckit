# myDraft_Verify — 投机推理统一实验平台

> 模块④：自主实践 → 基于四篇论文构建四策略统一框架 + 创新实验
> **运行环境：WSL2 (Ubuntu 22.04)** ← [WSL2_SETUP.md](../WSL2_SETUP.md)

---

## 架构概览

```
myDraft_Verify/
├── MANUAL.md                       ← 本手册
├── run.py                          ← 一键运行入口
├── requirements.txt                ← Python 依赖
├── core/                           ← 核心引擎
│   ├── strategy.py                 ← 5种策略枚举 + 论文引用
│   ├── draft.py                    ← Draft Engine (4种策略)
│   ├── verify.py                   ← Verify Engine (序列/树形)
│   └── engine.py                   ← 统一引擎 + 自动策略选择
├── models/
│   └── loader.py                   ← 模型加载 + KV Cache + Target Context 提取
├── experiments/
│   ├── runner.py                   ← 跨策略消融实验运行器
│   └── prompts.py                  ← 测试 Prompt 集
├── analysis/
│   ├── metrics.py                  ← 指标计算 + 排行榜生成
│   └── plot.py                     ← 可视化
└── tests/
    └── test_strategies.py          ← 单元测试
```

---

## 论文关联矩阵

| 模块文件 | Leviathan | SpecInfer | EAGLE-2 | DFlash |
|----------|:---------:|:---------:|:-------:|:------:|
| `core/strategy.py` | ✓ | ✓ | ✓ | ✓ |
| `core/draft.py` | Strategy-A: Sequential | Strategy-B: Tree | Strategy-C: Dynamic | Strategy-D: Pipeline |
| `core/verify.py` | Rejection Sampling | Tree Verify | — | — |
| `core/engine.py` | generate() | generate() | generate() | generate() + Unified |
| `models/loader.py` | KV Cache | — | — | Target Context 提取 |
| `experiments/runner.py` | ✓ | ✓ | ✓ | ✓ |
| `analysis/plot.py` | ✓ | ✓ | ✓ | ✓ |

---

## 四个创新点

### 创新①：Unified Strategy Selector

根据上下文特征自动选择最优策略（`core/engine.py` → `_select_strategy()`）：

```
if accept_rate > 0.85 and entropy < 0.3 → SEQUENTIAL (省计算)
elif entropy > 0.7                     → DYNAMIC (多候选探索)
elif accept_rate < 0.5                  → TREE (高不确定)
elif k > 7                              → PIPELINE (并行友好)
else                                    → SEQUENTIAL (默认)
```

**理论依据**: EAGLE-2 的置信度-接受率强相关性 + SpecInfer 的 top-k 覆盖率分析。

### 创新②：Cross-Strategy Ablation Study

同一 benchmark 上运行全部 4 个策略 + Unified，输出消融排行榜（`experiments/runner.py`）。

### 创新③：Layer-wise Target Context (DFlash-inspired)

提取 target model 中间层 hidden states → 注入 draft model 输入（`models/loader.py` → `extract_target_context()`）。

### 创新④：Ablation Leaderboard

自动生成策略对比排行榜，标注最佳策略（`analysis/plot.py`）。

---

## 使用方式

### 快速验证

```bash
conda activate spec_dec
python run.py --quick
```

### 消融实验

```bash
python run.py --ablation
```

输出将包含全部 5 个策略的加速比/接受率/Draft耗时占比排行榜。

### 交互模式

```bash
python run.py --interactive

# 交互命令:
#   /strat 1-5    → 切换策略
#   /k 5          → 设置 draft length
#   help          → 帮助
#   quit          → 退出
```

### 单策略测试

```bash
python run.py --strategy sequential --prompt "请解释机器学习。"
python run.py --strategy tree --prompt "什么是深度学习？"
python run.py --strategy dynamic
python run.py --strategy pipeline
python run.py --strategy unified
```

### Python API 调用

```python
from core.engine import SpecEngine
from core.strategy import DraftStrategy
from models.loader import ModelManager

models = ModelManager(
    target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
    draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
)
engine = SpecEngine(models)

# 策略 A: Sequential (Leviathan)
result = engine.generate(prompt="Hello", strategy=DraftStrategy.SEQUENTIAL, k=5)

# 策略 B: Tree (SpecInfer)
result = engine.generate(prompt="Hello", strategy=DraftStrategy.TREE, k=5, top_k=2)

# 策略 C: Dynamic (EAGLE-2)
result = engine.generate(prompt="Hello", strategy=DraftStrategy.DYNAMIC, k=5)

# 策略 D: Pipeline (DFlash-inspired)
result = engine.generate(prompt="Hello", strategy=DraftStrategy.PIPELINE, k=7)

# 策略 E: Unified (Auto-select, 创新)
result = engine.generate(prompt="Hello", strategy=DraftStrategy.UNIFIED)

# 自回归基线
baseline = engine.generate_baseline(prompt="Hello")
speedup = baseline.total_time_ms / result.total_time_ms
print(f"加速比: {speedup:.2f}x")
```

---

## 运行单元测试

```bash
python -m pytest tests/test_strategies.py -v
# 或
python tests/test_strategies.py
```

---

## 预期实验输出

运行 `python run.py --ablation` 后，在 `results/` 目录生成：

- `ablation_leaderboard.md` — 纯文本排行榜
- `metrics_raw.json` — 原始指标 JSON

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Ablation Leaderboard                           │
├────────────────────────┬─────────────┬──────────┬───────────────┬─────┤
│ Strategy               │      Speedup │   Accept │    DraftRatio │ TPok/s│
├────────────────────────┼─────────────┼──────────┼───────────────┼─────┤
│ 🧠 unified        ★   │     1.58x │   77.0% │         28% │ 45.2 │
│ ⚡ pipeline            │     1.52x │   65.0% │         18% │ 40.0 │
│ 🔄 dynamic             │     1.42x │   80.0% │         32% │ 35.0 │
│ 🌳 tree                │     1.35x │   78.0% │         40% │ 33.0 │
│ → sequential           │     1.00x │   65.0% │         35% │ 25.0 │
└────────────────────────┴─────────────┴──────────┴───────────────┴─────┘
```
