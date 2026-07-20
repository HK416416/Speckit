#!/usr/bin/env python3
"""
run.py — MyDraft_Verify 一键运行入口
======================================
支持:
  - 单策略测试:   python run.py --strategy sequential --prompt "Hello"
  - 消融实验:     python run.py --ablation
  - 快速验证:     python run.py --quick
  - 交互模式:     python run.py --interactive

论文关联: 全部四篇
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import SpecEngine
from core.strategy import DraftStrategy
from models.loader import ModelManager
from experiments.runner import run_ablation, RunConfig
from experiments.prompts import TEST_PROMPTS
from analysis.plot import plot_leaderboard_text


def cmd_quick():
    """快速验证：Sequential + Tree 策略各跑一次"""
    models = ModelManager(device="cuda")
    engine = SpecEngine(models)

    for strat in [DraftStrategy.SEQUENTIAL, DraftStrategy.TREE]:
        print(f"\n{'='*50}")
        print(f"  {strat.icon()} Strategy: {strat.value}  ({strat.paper_ref()})")
        print(f"{'='*50}")
        result = engine.generate(
            prompt="请用一句话介绍人工智能。",
            max_new_tokens=64,
            strategy=strat,
            k=5,
            top_k=2,
            verbose=True,
        )
        print(f"\n  → {result.tokens_per_second:.1f} tok/s, "
              f"accept={result.avg_accept_rate:.1%}")

    # Baseline
    print(f"\n{'='*50}")
    print(f"  Baseline (Autoregressive)")
    print(f"{'='*50}")
    bl = engine.generate_baseline("请用一句话介绍人工智能。", max_new_tokens=64)
    print(f"  → {bl.tokens_per_second:.1f} tok/s")


def cmd_single(strategy_name: str, prompt: str):
    """单策略测试"""
    models = ModelManager(device="cuda")
    engine = SpecEngine(models)

    strat_map = {
        "sequential": DraftStrategy.SEQUENTIAL,
        "tree": DraftStrategy.TREE,
        "dynamic": DraftStrategy.DYNAMIC,
        "pipeline": DraftStrategy.PIPELINE,
        "unified": DraftStrategy.UNIFIED,
    }
    strat = strat_map.get(strategy_name, DraftStrategy.SEQUENTIAL)

    print(f"\n  Strategy: {strat.icon()} {strat.value} ({strat.paper_ref()})")
    result = engine.generate(
        prompt=prompt, max_new_tokens=64, strategy=strat, k=5, top_k=2, verbose=True,
    )
    print(f"\n  Generated: {result.generated_text[-200:]}")
    print(f"  Speed: {result.tokens_per_second:.1f} tok/s, "
          f"accept={result.avg_accept_rate:.1%}, "
          f"draft_ratio={result.total_draft_time_ms / max(result.total_time_ms, 1):.0%}")


def cmd_ablation():
    """消融实验"""
    models = ModelManager(device="cuda")
    engine = SpecEngine(models)
    prompts = [(pid, t) for pid, t in TEST_PROMPTS[:4]]

    configs = [
        RunConfig("seq", DraftStrategy.SEQUENTIAL, k=5),
        RunConfig("tree", DraftStrategy.TREE, k=5, top_k=2, max_depth=3),
        RunConfig("dyn", DraftStrategy.DYNAMIC, k=5, top_k=2, max_depth=4),
        RunConfig("pipe", DraftStrategy.PIPELINE, k=7),
        RunConfig("unified", DraftStrategy.UNIFIED, k=5),
    ]

    metrics = run_ablation(engine, prompts, configs, max_new_tokens=64)
    print("\n" + plot_leaderboard_text(metrics))


def cmd_interactive():
    """交互模式"""
    models = ModelManager(device="cuda")
    engine = SpecEngine(models)

    print("\n  MyDraft_Verify 交互模式")
    print("  输入 'quit' 退出, 'help' 查看帮助")
    print("  策略: 1=Sequential 2=Tree 3=Dynamic 4=Pipeline 5=Unified")

    current_strategy = DraftStrategy.UNIFIED
    current_k = 5

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() == "quit":
            break
        if user_input.lower() == "help":
            print("  策略切换: /strat 1-5  |  k值: /k 5  |  当前策略: "
                  f"{current_strategy.value}, k={current_k}")
            continue
        if user_input.startswith("/strat"):
            try:
                idx = int(user_input.split()[1])
                current_strategy = list(DraftStrategy)[idx - 1]
                print(f"  → 策略切换为: {current_strategy.value}")
            except (IndexError, ValueError):
                print("  用法: /strat 1-5")
            continue
        if user_input.startswith("/k"):
            try:
                current_k = int(user_input.split()[1])
                print(f"  → k 设置为: {current_k}")
            except (IndexError, ValueError):
                print("  用法: /k 5")
            continue

        result = engine.generate(
            prompt=user_input, max_new_tokens=128, strategy=current_strategy,
            k=current_k, verbose=False,
        )
        print(f"[Bot] {result.generated_text}")
        print(f"  → {result.tokens_per_second:.1f} tok/s, "
              f"accept={result.avg_accept_rate:.1%}, "
              f"strategy={result.strategy_used.value}")


def main():
    parser = argparse.ArgumentParser(
        description="MyDraft_Verify — 统一投机推理实验平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="快速验证（单轮测试）")
    parser.add_argument("--ablation", action="store_true", help="运行消融实验")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--strategy", type=str, help="单策略名: sequential/tree/dynamic/pipeline/unified")
    parser.add_argument("--prompt", type=str, default="请用一句话介绍人工智能。", help="输入 prompt")
    args = parser.parse_args()

    if args.quick:
        cmd_quick()
    elif args.ablation:
        cmd_ablation()
    elif args.interactive:
        cmd_interactive()
    elif args.strategy:
        cmd_single(args.strategy, args.prompt)
    else:
        print("用法: python run.py --quick | --ablation | --interactive | --strategy <name>")
        print("策略: sequential | tree | dynamic | pipeline | unified")


if __name__ == "__main__":
    main()
