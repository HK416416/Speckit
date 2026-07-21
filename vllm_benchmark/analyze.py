"""
analyze.py — vLLM Benchmark 结果分析工具
===========================================
对比多个实验配置的性能指标，生成 Markdown 报告和图表。

用法:
  python analyze.py --results-dir results/baseline
  python analyze.py --compare-dirs results/baseline results/speculative --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List


def load_experiment(results_dir: str) -> Dict:
    """加载单个实验目录中的 results.json"""
    path = os.path.join(results_dir, "results.json")
    if not os.path.exists(path):
        print(f"  ⚠ 未找到 {path}，跳过")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_table_row(name: str, data: Dict) -> str:
    """格式化单行对比表格"""
    s = data.get("summary", {})
    return (
        f"| {name:<25s} | "
        f"{s.get('avg_ttft_ms', 0):>8.1f} | "
        f"{s.get('p50_ttft_ms', 0):>8.1f} | "
        f"{s.get('p95_ttft_ms', 0):>8.1f} | "
        f"{s.get('avg_tpot_ms', 0):>8.1f} | "
        f"{s.get('avg_throughput_tps', 0):>8.1f} | "
        f"{s.get('num_success', 0):>6d} |"
    )


def generate_report(
    experiments: List[tuple],
    output_path: str,
) -> str:
    """生成 Markdown 格式的实验对比报告"""

    lines = []
    lines.append("# vLLM 特性性能实验 — 对比报告")
    lines.append("")
    lines.append(f"实验时间: 自动生成")
    lines.append(f"实验数量: {len(experiments)}")
    lines.append("")

    # ── 性能对比表 ──
    lines.append("## 1. 核心指标对比")
    lines.append("")
    lines.append("| 实验配置 | TTFT avg(ms) | TTFT P50(ms) | TTFT P95(ms) | TPOT avg(ms) | 吞吐量(tok/s) | 成功数 |")
    lines.append("|-----------|-------------|-------------|-------------|-------------|--------------|-------|")

    baseline_tps = 0.0
    for name, data in experiments:
        lines.append(format_table_row(name, data))
        if "baseline" in name.lower():
            baseline_tps = data.get("summary", {}).get("avg_throughput_tps", 0.0)

    lines.append("")

    # ── 加速比分析 ──
    if baseline_tps > 0:
        # ── 按名称查找基线实验（而非假设 experiments[0] 是基线）──
        baseline_data = None
        for name, data in experiments:
            if "baseline" in name.lower():
                baseline_data = data
                break
        baseline_ttft = baseline_data.get("summary", {}).get("avg_ttft_ms", 1) if baseline_data else 1
        baseline_tpot = baseline_data.get("summary", {}).get("avg_tpot_ms", 1) if baseline_data else 1

        lines.append("## 2. 加速比分析 (vs 基线)")
        lines.append("")
        lines.append("| 实验配置 | 吞吐量 (tok/s) | 加速比 | TTFT 变化 | TPOT 变化 |")
        lines.append("|-----------|---------------|--------|-----------|-----------|")
        for name, data in experiments:
            s = data.get("summary", {})
            tps = s.get("avg_throughput_tps", 0.0)
            speedup = tps / baseline_tps if baseline_tps > 0 else 1.0

            ttft_change = (s.get("avg_ttft_ms", 0) - baseline_ttft) / baseline_ttft * 100 if baseline_ttft else 0
            tpot_change = (s.get("avg_tpot_ms", 0) - baseline_tpot) / baseline_tpot * 100 if baseline_tpot else 0

            lines.append(
                f"| {name:<25s} | {tps:>13.1f} | {speedup:>6.2f}x | "
                f"{ttft_change:>+8.1f}% | {tpot_change:>+8.1f}% |"
            )
        lines.append("")

    # ── 特性-指标关联分析 ──
    lines.append("## 3. 特性-指标关联分析")
    lines.append("")
    lines.append("| 特性 | TTFT | TPOT | 吞吐量 | 影响机制 |")
    lines.append("|------|------|------|--------|----------|")
    lines.append("| 前缀缓存 | ↓↓ (共享前缀跳过重复 KV 计算) | — | ↑ | 复用相同前缀的 KV Cache |")
    lines.append("| 分块预填充 | ↓ (减少长 prompt 阻塞) | — | ↑ | 将 prefill 切块，提升 GPU 利用率 |")
    lines.append("| 最大并发序列数 | ↑ (排队增加) | ↑ | ↑↑ | 更多请求并发 → GPU 利用率上升 |")
    lines.append("| 投机推理 ⭐ | — | ↓↓ (draft-verify 减少串行) | ↑↑ | 小模型预猜 + 大模型并行验证 |")
    lines.append("")

    # ── 建议 ──
    lines.append("## 4. 结论与建议")
    lines.append("")
    lines.append("[TODO: 根据实验结果填写具体结论]")
    lines.append("")
    lines.append("请在完成实验后，根据实际数据更新此部分：")
    lines.append("1. 哪个特性对吞吐量提升最大？")
    lines.append("2. 投机推理在 RTX 4050 上的实际加速比是多少？")
    lines.append("3. 前缀缓存和投机推理组合使用的效果如何？")
    lines.append("4. 端侧部署的推荐配置是什么？")
    lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Benchmark 结果分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, help="单个实验结果目录")
    parser.add_argument("--compare-dirs", type=str, nargs="+", help="多个实验结果目录进行对比")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径 (默认自动生成)")
    args = parser.parse_args()

    if args.compare_dirs:
        dirs = args.compare_dirs
    elif args.results_dir:
        dirs = [args.results_dir]
    else:
        print("请指定 --results-dir 或 --compare-dirs")
        sys.exit(1)

    # 加载所有实验数据
    experiments = []
    for d in dirs:
        data = load_experiment(d)
        if data:
            name = data.get("experiment_name", os.path.basename(d))
            experiments.append((name, data))

    if not experiments:
        print("未找到任何实验结果")
        sys.exit(1)

    # 输出路径
    output_path = args.output
    if not output_path:
        if args.compare_dirs:
            output_path = os.path.join(os.path.commonpath(args.compare_dirs) if len(args.compare_dirs) > 1 else ".", "comparison_report.md")
        else:
            output_path = os.path.join(dirs[0], "analysis_report.md")

    report = generate_report(experiments, output_path)

    print("=" * 60)
    print("  vLLM 实验分析")
    print("=" * 60)
    print(report)
    print(f"\n  报告已保存: {output_path}")


if __name__ == "__main__":
    main()
