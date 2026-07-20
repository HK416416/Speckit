"""
runner.py — 实验运行器
=======================
跨策略消融实验：在同一个 benchmark 上运行全部 4(5) 个策略。

论文关联: 全部四篇论文的对比实验
创新点: ② Cross-Strategy Ablation Study
"""

from __future__ import annotations

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Optional
from dataclasses import dataclass, field

from core.engine import SpecEngine, SpecResult
from core.strategy import DraftStrategy
from models.loader import ModelManager
from analysis.metrics import ExperimentMetrics, compute_speedup, generate_leaderboard
from experiments.prompts import TEST_PROMPTS


@dataclass
class RunConfig:
    """单次实验运行配置"""
    name: str = ""
    strategy: DraftStrategy = DraftStrategy.SEQUENTIAL
    k: int = 5
    temperature: float = 0.0
    top_k: int = 2
    max_depth: int = 3
    use_target_context: bool = False


def run_ablation(
    engine: SpecEngine,
    prompts: list,
    configs: List[RunConfig],
    max_new_tokens: int = 64,
    verbose: bool = True,
) -> List[ExperimentMetrics]:
    """
    运行跨策略消融实验。

    对每个配置，在全部 prompt 上运行，收集加速比、接受率等指标。
    """
    all_metrics: List[ExperimentMetrics] = []

    # 先跑自回归基线
    print("\n[Runner] 计算自回归基线...")
    baseline_cache: Dict[str, float] = {}
    for pid, prompt_text in prompts:
        bl = engine.generate_baseline(prompt_text, max_new_tokens=max_new_tokens, temperature=0.0)
        baseline_cache[pid] = bl.total_time_ms
        print(f"  {pid}: {bl.tokens_per_second:.1f} tok/s, {bl.total_time_ms:.0f}ms")

    # 对每个配置运行实验
    for cfg in configs:
        print(f"\n[Runner] 策略: {cfg.strategy.value} ({cfg.strategy.paper_ref()})")
        print(f"          k={cfg.k}, temp={cfg.temperature}")

        speedups: List[float] = []
        accept_rates: List[float] = []
        draft_ratios: List[float] = []
        tps_list: List[float] = []
        total_tokens = 0

        for pid, prompt_text in prompts:
            result = engine.generate(
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
                k=cfg.k,
                temperature=cfg.temperature,
                strategy=cfg.strategy,
                top_k=cfg.top_k,
                max_depth=cfg.max_depth,
                use_target_context=cfg.use_target_context,
                verbose=False,
            )

            bl_time = baseline_cache.get(pid, 1000.0)
            speedups.append(compute_speedup(bl_time, result.total_time_ms))
            accept_rates.append(result.avg_accept_rate)
            draft_ratios.append(
                result.total_draft_time_ms / max(result.total_time_ms, 1)
            )
            tps_list.append(result.tokens_per_second)
            total_tokens += len(result.generated_token_ids)

        n = len(speedups)
        metrics = ExperimentMetrics(
            experiment_name=f"{cfg.strategy.value}_k{cfg.k}_t{cfg.temperature}",
            strategy=f"{cfg.strategy.icon()} {cfg.strategy.value}",
            draft_k=cfg.k,
            temperature=cfg.temperature,
            num_runs=n,
            avg_speedup=sum(speedups) / n,
            avg_accept_rate=sum(accept_rates) / n,
            avg_tokens_per_sec=sum(tps_list) / n,
            avg_draft_ratio=sum(draft_ratios) / n,
            total_generated_tokens=total_tokens,
        )
        all_metrics.append(metrics)

        if verbose:
            print(f"  → speedup={metrics.avg_speedup:.2f}x, "
                  f"accept={metrics.avg_accept_rate:.1%}, "
                  f"draft_ratio={metrics.avg_draft_ratio:.0%}, "
                  f"tps={metrics.avg_tokens_per_sec:.1f}")

    return all_metrics


def main():
    """运行全策略消融实验"""
    print("=" * 60)
    print("  MyDraft_Verify — 跨策略消融实验")
    print("=" * 60)

    # 加载模型
    models = ModelManager(
        target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda",
    )

    engine = SpecEngine(models, device="cuda")

    # 只测试短 prompt（快速验证时）
    prompts = [(pid, t) for pid, t in TEST_PROMPTS if pid.startswith("short") or pid.startswith("medium")]

    # 实验配置
    configs = [
        RunConfig("seq_k5", DraftStrategy.SEQUENTIAL, k=5),
        RunConfig("tree_k5", DraftStrategy.TREE, k=5, top_k=2, max_depth=3),
        RunConfig("dynamic_k5", DraftStrategy.DYNAMIC, k=5, top_k=2, max_depth=4),
        RunConfig("pipeline_k7", DraftStrategy.PIPELINE, k=7),
        RunConfig("unified_k5", DraftStrategy.UNIFIED, k=5),
    ]

    metrics = run_ablation(engine, prompts, configs, max_new_tokens=64, verbose=True)

    # 输出排行榜
    print("\n" + "=" * 60)
    leaderboard = generate_leaderboard(metrics)
    print(leaderboard)

    # 保存结果
    os.makedirs("results", exist_ok=True)
    result_path = "results/ablation_leaderboard.md"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(leaderboard)
    print(f"\n排行榜已保存: {result_path}")

    # 保存 JSON
    json_path = "results/metrics_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "strategy": m.strategy, "speedup": round(m.avg_speedup, 3),
                "accept_rate": round(m.avg_accept_rate, 3),
                "draft_ratio": round(m.avg_draft_ratio, 3),
                "tps": round(m.avg_tokens_per_sec, 1),
            }
            for m in metrics
        ], f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存: {json_path}")


if __name__ == "__main__":
    main()
