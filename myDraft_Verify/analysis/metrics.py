"""
metrics.py — 指标计算与汇总
============================
"""
from __future__ import annotations
from typing import List
from dataclasses import dataclass, field


@dataclass
class ExperimentMetrics:
    """单组实验的汇总指标"""
    experiment_name: str = ""
    strategy: str = ""
    draft_k: int = 5
    temperature: float = 0.0
    num_runs: int = 0
    avg_speedup: float = 1.0
    avg_accept_rate: float = 0.0
    avg_tokens_per_sec: float = 0.0
    avg_draft_ratio: float = 0.0  # draft 耗时占比
    avg_steps: int = 0
    total_generated_tokens: int = 0


def compute_speedup(baseline_time_ms: float, spec_time_ms: float) -> float:
    """计算加速比"""
    if spec_time_ms <= 0:
        return 0.0
    return baseline_time_ms / spec_time_ms


def percentile(values: List[float], p: float) -> float:
    """计算百分位数"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(k, len(sorted_vals) - 1)]


def generate_leaderboard(
    results: List[ExperimentMetrics],
) -> str:
    """生成消融排行榜（Markdown 表格）"""
    lines = []
    lines.append("# Ablation Leaderboard")
    lines.append("")
    lines.append("| 策略 | 加速比 | 接受率 | Draft耗时% | 吞吐量(tok/s) | 推荐场景 |")
    lines.append("|------|--------|--------|-----------|-------------|---------|")

    sorted_results = sorted(results, key=lambda r: r.avg_speedup, reverse=True)
    for r in sorted_results:
        best = "⭐" if r.avg_speedup == sorted_results[0].avg_speedup else ""
        scenario = _recommend_scenario(r)
        lines.append(
            f"| {r.strategy} {best} | {r.avg_speedup:.2f}x | {r.avg_accept_rate:.1%} | "
            f"{r.avg_draft_ratio:.0%} | {r.avg_tokens_per_sec:.1f} | {scenario} |"
        )

    lines.append("")
    return "\n".join(lines)


def _recommend_scenario(metrics: ExperimentMetrics) -> str:
    """根据指标推荐使用场景"""
    if metrics.avg_speedup >= 1.5:
        return "通用最优"
    elif metrics.avg_draft_ratio < 0.2:
        return "k>5 长序列"
    elif metrics.avg_accept_rate > 0.8:
        return "高确定性任务"
    elif metrics.avg_accept_rate < 0.5:
        return "高不确定性任务"
    return "基准对照"
