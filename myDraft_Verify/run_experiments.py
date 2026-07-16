"""
run_experiments.py — 投机推理实验运行脚本
=============================================
运行方案B完整实验:
  1. 基线实验: draft_k ∈ {3, 5, 7, 10} × temperature ∈ {0.0, 0.6, 0.8, 1.0}
  2. 采样策略对比: greedy vs standard vs entropy-adaptive
  3. 自回归基线对比 (加速比计算)
  4. 结果导出 CSV + 自动绘图

用法:
  python run_experiments.py
  python run_experiments.py --quick       # 快速验证 (单组参数)
  python run_experiments.py --no-viz      # 跳过绘图
  python run_experiments.py --output_dir ./results_week1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from speculative_decoding import (
    SpeculativeDecoder,
    SamplingStrategy,
    DecodeResult,
)


# ──────────────────────────────────────────────────────────────────────
# 测试 Prompt 集
# ──────────────────────────────────────────────────────────────────────

TEST_PROMPTS = [
    # 短 prompt
    ("short_1", "请用一句话介绍人工智能。"),
    ("short_2", "什么是深度学习？"),
    ("short_3", "请解释机器学习的监督学习和无监督学习。"),
    # 中 prompt
    ("medium_1", "请详细说明Transformer模型中的自注意力机制是如何工作的。"),
    ("medium_2", "请比较CPU和GPU在深度学习推理中的优劣，至少包含三个方面。"),
    ("medium_3", "请解释什么是梯度下降算法，以及随机梯度下降与批量梯度下降的区别。"),
    # 长 prompt (需要更丰富的生成)
    ("long_1", "假设你是一名计算机科学家，请向非技术人员解释大语言模型的工作原理，包括训练和推理两个阶段。请用通俗易懂的语言。"),
    ("long_2", "请从性能、成本和部署难度三个维度，对比分析云计算和边缘计算在AI推理场景下的适用性。"),
]


# ──────────────────────────────────────────────────────────────────────
# 数据类型
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """单次实验的完整结果"""
    prompt_id: str = ""
    prompt: str = ""
    strategy: str = ""
    draft_k: int = 5
    temperature: float = 0.0
    max_new_tokens: int = 128
    num_generated: int = 0
    total_time_ms: float = 0.0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    avg_accept_rate: float = 0.0
    tokens_per_second: float = 0.0
    num_steps: int = 0
    generated_text: str = ""
    speedup_vs_baseline: float = 1.0


# ──────────────────────────────────────────────────────────────────────
# 实验运行器
# ──────────────────────────────────────────────────────────────────────

class ExperimentRunner:
    """实验运行器: 管理参数网格搜索、基线对比和结果导出"""

    def __init__(
        self,
        decoder: SpeculativeDecoder,
        output_dir: str = "./results",
        prompt_subset: Optional[List[str]] = None,
    ):
        self.decoder = decoder
        self.output_dir = output_dir
        self.results: List[ExperimentResult] = []
        self.baseline_cache: Dict[str, DecodeResult] = {}  # prompt_id → 基线结果

        # 筛选 prompt
        if prompt_subset:
            self.prompts = [(p[0], p[1]) for p in TEST_PROMPTS if p[0] in prompt_subset]
        else:
            self.prompts = list(TEST_PROMPTS)

        os.makedirs(output_dir, exist_ok=True)

    # ── 公开 API ────────────────────────────────────────────────────

    def run_baseline(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> None:
        """运行自回归基线 (所有 prompt)"""
        print("\n" + "=" * 60)
        print("  自回归基线 (Autoregressive Baseline)")
        print("=" * 60)

        for prompt_id, prompt_text in self.prompts:
            print(f"\n  [{prompt_id}] {prompt_text[:60]}...")
            result = self.decoder.generate_autoregressive_baseline(
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            self.baseline_cache[prompt_id] = result
            print(f"    → {len(result.generated_token_ids)} tokens, {result.total_time_ms:.0f}ms, "
                  f"{result.tokens_per_second:.1f} tok/s")
            print(f"    → {result.generated_text[-120:]}...")

    def run_grid_search(
        self,
        k_values: List[int] = (3, 5, 7, 10),
        temperatures: Tuple[float, ...] = (0.0, 0.6, 0.8, 1.0),
        strategies: Tuple[SamplingStrategy, ...] = (
            SamplingStrategy.GREEDY,
            SamplingStrategy.STANDARD,
            SamplingStrategy.ENTROPY_ADAPTIVE,
        ),
        max_new_tokens: int = 128,
        verbose: bool = False,
    ) -> None:
        """运行网格搜索实验"""
        total = len(self.prompts) * len(k_values) * len(temperatures) * len(strategies)
        count = 0

        print("\n" + "=" * 60)
        print(f"  网格搜索实验 (共 {total} 组)")
        print("=" * 60)

        for prompt_id, prompt_text in self.prompts:
            for k in k_values:
                for strategy in strategies:
                    for temp in temperatures:
                        # GREEDY 策略下温度无意义，跳过 temp > 0
                        if strategy == SamplingStrategy.GREEDY and temp > 0.0:
                            continue

                        count += 1
                        print(f"\n[{count}/{total}] "
                              f"prompt={prompt_id} | k={k} | "
                              f"strategy={strategy.value} | temp={temp}")

                        try:
                            result = self.decoder.generate(
                                prompt=prompt_text,
                                max_new_tokens=max_new_tokens,
                                draft_k=k,
                                temperature=temp,
                                strategy=strategy,
                                verbose=verbose,
                                return_steps=True,
                            )
                        except Exception as e:
                            print(f"    ⚠ 实验失败: {e}")
                            continue

                        # 计算加速比
                        baseline = self.baseline_cache.get(prompt_id)
                        speedup = 1.0
                        if baseline and baseline.total_time_ms > 0:
                            speedup = baseline.total_time_ms / result.total_time_ms

                        exp_result = ExperimentResult(
                            prompt_id=prompt_id,
                            prompt=prompt_text[:100],
                            strategy=strategy.value,
                            draft_k=k,
                            temperature=temp,
                            max_new_tokens=max_new_tokens,
                            num_generated=len(result.generated_token_ids),
                            total_time_ms=result.total_time_ms,
                            draft_time_ms=result.total_draft_time_ms,
                            verify_time_ms=result.total_verify_time_ms,
                            avg_accept_rate=result.avg_acceptance_rate,
                            tokens_per_second=result.tokens_per_second,
                            num_steps=len(result.steps),
                            generated_text=result.generated_text[-200:],
                            speedup_vs_baseline=speedup,
                        )
                        self.results.append(exp_result)

                        print(f"    → {exp_result.num_generated} tokens, "
                              f"{exp_result.total_time_ms:.0f}ms, "
                              f"{exp_result.tokens_per_second:.1f} tok/s, "
                              f"accept={exp_result.avg_accept_rate:.1%}, "
                              f"speedup={speedup:.2f}x")

        print(f"\n  实验完成! 共收集 {len(self.results)} 条结果")

    def export_csv(self, filename: str = "experiment_results.csv") -> str:
        """导出结果为 CSV"""
        path = os.path.join(self.output_dir, filename)
        if not self.results:
            print("  无结果可导出")
            return path

        fieldnames = [
            "prompt_id", "strategy", "draft_k", "temperature",
            "max_new_tokens", "num_generated", "total_time_ms",
            "draft_time_ms", "verify_time_ms", "avg_accept_rate",
            "tokens_per_second", "num_steps", "speedup_vs_baseline",
            "generated_text",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "prompt_id": r.prompt_id,
                    "strategy": r.strategy,
                    "draft_k": r.draft_k,
                    "temperature": r.temperature,
                    "max_new_tokens": r.max_new_tokens,
                    "num_generated": r.num_generated,
                    "total_time_ms": round(r.total_time_ms, 2),
                    "draft_time_ms": round(r.draft_time_ms, 2),
                    "verify_time_ms": round(r.verify_time_ms, 2),
                    "avg_accept_rate": round(r.avg_accept_rate, 4),
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "num_steps": r.num_steps,
                    "speedup_vs_baseline": round(r.speedup_vs_baseline, 4),
                    "generated_text": r.generated_text[:200],
                })

        print(f"  CSV 已导出: {path} ({len(self.results)} 行)")
        return path

    def export_summary(self, filename: str = "experiment_summary.json") -> str:
        """导出汇总统计为 JSON"""
        path = os.path.join(self.output_dir, filename)

        summary = {
            "total_experiments": len(self.results),
            "model_info": {
                "target": self.decoder.target_model_name,
                "draft": self.decoder.draft_model_name,
            },
            "prompts": [{"id": p[0], "text": p[1]} for p in self.prompts],
            "baselines": {
                pid: {
                    "time_ms": r.total_time_ms,
                    "tokens": len(r.generated_token_ids),
                    "tok_per_sec": r.tokens_per_second,
                }
                for pid, r in self.baseline_cache.items()
            },
            "best_configs": self._find_best_configs(),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"  JSON 已导出: {path}")
        return path

    def plot_results(self) -> Optional[str]:
        """使用 matplotlib 绘制实验图表"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # 非交互式后端
            import matplotlib.pyplot as plt
        except ImportError:
            print("  ⚠ matplotlib 未安装，跳过绘图")
            return None

        if not self.results:
            print("  无结果可绘图")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Speculative Decoding Experiments — RTX 4050", fontsize=14, fontweight="bold")

        # ── 图1: 加速比 vs draft_k (按策略分组) ──
        ax1 = axes[0, 0]
        for strategy in ["greedy", "standard", "entropy_adaptive"]:
            data = [r for r in self.results if r.strategy == strategy and r.temperature == 0.0]
            if data:
                xs = sorted(set(r.draft_k for r in data))
                ys = []
                for k in xs:
                    k_data = [r.speedup_vs_baseline for r in data if r.draft_k == k]
                    ys.append(sum(k_data) / len(k_data) if k_data else 0)
                ax1.plot(xs, ys, "o-", label=strategy, linewidth=2, markersize=8)
        ax1.set_xlabel("Draft Length k", fontsize=11)
        ax1.set_ylabel("Speedup vs Baseline", fontsize=11)
        ax1.set_title("Speedup vs Draft Length (temp=0)", fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

        # ── 图2: 接受率 vs draft_k ──
        ax2 = axes[0, 1]
        for strategy in ["greedy", "standard", "entropy_adaptive"]:
            data = [r for r in self.results if r.strategy == strategy and r.temperature == 0.0]
            if data:
                xs = sorted(set(r.draft_k for r in data))
                ys = []
                for k in xs:
                    k_data = [r.avg_accept_rate for r in data if r.draft_k == k]
                    ys.append(sum(k_data) / len(k_data) if k_data else 0)
                ax2.plot(xs, ys, "s-", label=strategy, linewidth=2, markersize=8)
        ax2.set_xlabel("Draft Length k", fontsize=11)
        ax2.set_ylabel("Acceptance Rate", fontsize=11)
        ax2.set_title("Acceptance Rate vs Draft Length", fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ── 图3: 接受率 vs temperature ──
        ax3 = axes[1, 0]
        temps = sorted(set(r.temperature for r in self.results if r.temperature >= 0))
        for k in [3, 5, 7, 10]:
            xs = []
            ys = []
            for temp in temps:
                data = [r for r in self.results
                        if r.draft_k == k and r.temperature == temp
                        and r.strategy == "standard"]
                if data:
                    xs.append(temp)
                    ys.append(sum(r.avg_accept_rate for r in data) / len(data))
            if xs:
                ax3.plot(xs, ys, "o-", label=f"k={k}", linewidth=2, markersize=6)
        ax3.set_xlabel("Temperature", fontsize=11)
        ax3.set_ylabel("Acceptance Rate", fontsize=11)
        ax3.set_title("Acceptance Rate vs Temperature (standard)", fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # ── 图4: 吞吐量对比 (bar chart) ──
        ax4 = axes[1, 1]
        strategies = ["autoregressive", "greedy", "standard", "entropy_adaptive"]
        labels = ["Baseline\n(Autoregressive)", "Greedy\n(k=5)", "Standard\n(k=5)", "Entropy\nAdaptive"]
        values = []
        # Baseline throughput
        baseline_tps = []
        for pid, br in self.baseline_cache.items():
            if br.tokens_per_second > 0:
                baseline_tps.append(br.tokens_per_second)
        avg_baseline = sum(baseline_tps) / len(baseline_tps) if baseline_tps else 0
        values.append(avg_baseline)

        for strategy in ["greedy", "standard", "entropy_adaptive"]:
            data = [r for r in self.results
                    if r.strategy == strategy and r.draft_k == 5 and r.temperature == 0.0]
            if data:
                values.append(sum(r.tokens_per_second for r in data) / len(data))
            else:
                values.append(0)

        colors = ["#999999", "#2196F3", "#FF9800", "#4CAF50"]
        bars = ax4.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax4.set_ylabel("Tokens per Second", fontsize=11)
        ax4.set_title("Throughput Comparison (k=5, temp=0)", fontsize=12)
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "experiment_plots.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  图表已保存: {path}")
        return path

    # ── 内部方法 ────────────────────────────────────────────────────

    def _find_best_configs(self) -> Dict:
        """找出每种策略下的最优配置"""
        best = {}
        for strategy in ["greedy", "standard", "entropy_adaptive"]:
            data = [r for r in self.results if r.strategy == strategy]
            if not data:
                continue
            best_run = max(data, key=lambda r: r.speedup_vs_baseline)
            best[strategy] = {
                "draft_k": best_run.draft_k,
                "temperature": best_run.temperature,
                "speedup": best_run.speedup_vs_baseline,
                "accept_rate": best_run.avg_accept_rate,
                "tokens_per_second": best_run.tokens_per_second,
            }
        return best


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="投机推理实验运行器 — 方案B完整实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target_model", type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Target model HuggingFace ID"
    )
    parser.add_argument(
        "--draft_model", type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Draft model HuggingFace ID"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="结果输出目录"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式: 仅跑一组参数验证流程"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="跳过 matplotlib 绘图"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cuda / cpu)"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=128,
        help="最大生成 token 数"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="打印每轮投机推理详情"
    )
    args = parser.parse_args()

    # ── 初始化 Decoder ──────────────────────────────────────────────
    print("=" * 60)
    print("  投机推理实验 — 方案B: 手写 Draft-Verify 框架")
    print("=" * 60)
    print(f"  Target Model:  {args.target_model}")
    print(f"  Draft Model:   {args.draft_model}")
    print(f"  Device:        {args.device}")
    print(f"  Mode:          {'快速验证' if args.quick else '完整网格搜索'}")
    print(f"  Output Dir:    {args.output_dir}")

    decoder = SpeculativeDecoder(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        device=args.device,
    )

    runner = ExperimentRunner(
        decoder=decoder,
        output_dir=args.output_dir,
        prompt_subset=["short_1", "medium_1"] if args.quick else None,
    )

    # ── 运行实验 ────────────────────────────────────────────────────
    print("\n[1/3] 运行自回归基线...")
    runner.run_baseline(max_new_tokens=args.max_new_tokens)

    if args.quick:
        print("\n[2/3] 快速验证 (单组参数)...")
        runner.run_grid_search(
            k_values=[5],
            temperatures=(0.0,),
            strategies=(SamplingStrategy.GREEDY, SamplingStrategy.STANDARD),
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )
    else:
        print("\n[2/3] 网格搜索实验...")
        runner.run_grid_search(
            k_values=[3, 5, 7, 10],
            temperatures=(0.0, 0.6, 0.8, 1.0),
            strategies=(SamplingStrategy.GREEDY, SamplingStrategy.STANDARD, SamplingStrategy.ENTROPY_ADAPTIVE),
            max_new_tokens=args.max_new_tokens,
            verbose=args.verbose,
        )

    # ── 导出 ────────────────────────────────────────────────────────
    print("\n[3/3] 导出结果...")
    runner.export_csv()
    runner.export_summary()

    if not args.no_viz:
        runner.plot_results()

    # ── 输出关键发现 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  实验总结")
    print("=" * 60)

    if runner.results:
        # 最佳加速比
        best = max(runner.results, key=lambda r: r.speedup_vs_baseline)
        print(f"\n  最佳加速比: {best.speedup_vs_baseline:.2f}x")
        print(f"    配置: k={best.draft_k}, strategy={best.strategy}, temp={best.temperature}")
        print(f"    接受率: {best.avg_accept_rate:.1%}, 吞吐量: {best.tokens_per_second:.1f} tok/s")

        # Draft 耗时占比
        avg_draft_ratio = sum(r.draft_time_ms / max(r.total_time_ms, 1) for r in runner.results) / len(runner.results)
        print(f"\n  平均 Draft 耗时占比: {avg_draft_ratio:.1%}")
        if avg_draft_ratio > 0.3:
            print(f"    ⚠ Draft 阶段是主要瓶颈 → 需端侧异构并行优化 (报告1核心)")

    print(f"\n  所有结果已保存至: {os.path.abspath(args.output_dir)}")
    print("  实验完成! ✓")


if __name__ == "__main__":
    main()
