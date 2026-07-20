"""
plot.py — 可视化
=================
消融排行榜 + 雷达图 + 加速比条形图
"""
from __future__ import annotations
import os
from typing import List
from analysis.metrics import ExperimentMetrics


def plot_leaderboard_text(metrics: List[ExperimentMetrics]) -> str:
    """纯文本排行榜（无需 matplotlib）"""
    lines = []
    lines.append("┌" + "─" * 68 + "┐")
    lines.append("│" + " Ablation Leaderboard".center(66) + "│")
    lines.append("├" + "─" * 22 + "┬" + "─" * 11 + "┬" + "─" * 9 + "┬" + "─" * 13 + "┬" + "─" * 9 + "┤")
    lines.append("│ {:<20} │ {:>9} │ {:>7} │ {:>11} │ {:>7} │".format(
        "Strategy", "Speedup", "Accept", "DraftRatio", "TPok/s"))
    lines.append("├" + "─" * 22 + "┼" + "─" * 11 + "┼" + "─" * 9 + "┼" + "─" * 13 + "┼" + "─" * 9 + "┤")

    sorted_m = sorted(metrics, key=lambda r: r.avg_speedup, reverse=True)
    for m in sorted_m:
        best = " ★" if m.avg_speedup == sorted_m[0].avg_speedup else "  "
        lines.append("│ {:<20} │ {}{:>7.2f}x │ {:>6.1%} │ {:>10.0%} │ {:>6.1f} │".format(
            m.strategy[:20], best, m.avg_speedup, m.avg_accept_rate, m.avg_draft_ratio, m.avg_tokens_per_sec))

    lines.append("└" + "─" * 22 + "┴" + "─" * 11 + "┴" + "─" * 9 + "┴" + "─" * 13 + "┴" + "─" * 9 + "┘")
    return "\n".join(lines)
