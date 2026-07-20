"""
engine.py — 统一投机推理引擎
===============================
整合 Draft Engine + Verify Engine + Strategy Selector，
提供统一的 generate() API。

论文关联: 全部四篇论文
创新点:
  ① Unified Strategy Selector — 根据上下文自动选择最优策略
  ② Pipeline Parallel (CUDA Stream) — DFlash 启发
  ③ Ablation Leaderboard — 全策略对比实验
"""

from __future__ import annotations

import time
import torch
from typing import Optional, List
from dataclasses import dataclass, field

from .strategy import DraftStrategy
from .draft import DraftEngine, DraftResult
from .verify import VerifyEngine


@dataclass
class SpecResult:
    """统一生成结果"""
    generated_text: str = ""
    generated_token_ids: List[int] = field(default_factory=list)
    strategy_used: DraftStrategy = DraftStrategy.UNIFIED
    total_time_ms: float = 0.0
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    num_steps: int = 0
    avg_accept_rate: float = 0.0
    tokens_per_second: float = 0.0
    strategy_switches: List[str] = field(default_factory=list)  # UNIFIED 模式的策略切换记录


class SpecEngine:
    """
    统一投机推理引擎。

    用法:
        engine = SpecEngine(models)
        result = engine.generate(prompt, strategy=DraftStrategy.SEQUENTIAL, k=5)
        result = engine.generate(prompt, strategy=DraftStrategy.UNIFIED)  # 自动选择
    """

    def __init__(self, models, device: str = "cuda"):
        self.models = models
        self.device = device
        self.tokenizer = models.tokenizer
        self.draft_engine = DraftEngine(models, device)
        self.verify_engine = VerifyEngine(models, device)

        # UNIFIED 模式的策略状态
        self._prev_accept_rate = 0.7
        self._prev_entropy = 0.5
        self._current_k = 5

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        k: int = 5,
        temperature: float = 0.0,
        strategy: DraftStrategy = DraftStrategy.SEQUENTIAL,
        top_k: int = 2,
        max_depth: int = 3,
        use_target_context: bool = False,
        verbose: bool = True,
    ) -> SpecResult:
        """
        执行投机推理生成。

        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            k: draft length
            temperature: 采样温度
            strategy: Draft 策略（SEQUENTIAL/TREE/DYNAMIC/PIPELINE/UNIFIED）
            top_k: Tree/Dynamic 策略的候选数
            max_depth: Tree/Dynamic 策略的最大深度
            use_target_context: 是否使用 DFlash-inspired target context 注入
            verbose: 是否打印每步详情
        """
        if strategy == DraftStrategy.UNIFIED:
            return self._generate_unified(
                prompt, max_new_tokens, k, temperature, top_k, max_depth,
                use_target_context, verbose,
            )

        context_ids = self.models.encode(prompt)
        generated: List[int] = []
        total_draft_time = 0.0
        total_verify_time = 0.0
        total_start = time.perf_counter()
        steps = 0
        total_accepted = 0
        total_drafted = 0

        # DFlash: 提取 target context（可选）
        target_context = None
        if use_target_context:
            target_context = self.models.extract_target_context(context_ids, num_layers=1)

        while len(generated) < max_new_tokens:
            steps += 1
            if generated:
                input_ids = torch.cat([
                    context_ids,
                    torch.tensor([generated], device=self.device),
                ], dim=1)
            else:
                input_ids = context_ids

            # Draft
            draft_result = self.draft_engine.generate(
                context_ids=input_ids, k=k, temperature=temperature,
                strategy=strategy, top_k=top_k, max_depth=max_depth,
                target_context=target_context,
            )
            total_draft_time += draft_result.draft_time_ms

            # Verify
            accepted, v_time = self.verify_engine.verify(
                input_ids, draft_result, temperature,
            )
            total_verify_time += v_time
            total_accepted += len(accepted)
            total_drafted += draft_result.num_candidates

            generated.extend(accepted)
            if verbose:
                acc_text = self.tokenizer.decode(accepted) if accepted else "<eos>"
                print(
                    f"  [Step {steps:3d}] {strategy.icon()} k={k:2d} | "
                    f"acc={len(accepted)}/{draft_result.num_candidates} | "
                    f"draft={draft_result.draft_time_ms:.1f}ms | "
                    f"verify={v_time:.1f}ms | → {acc_text!r}"
                )

            if not accepted or accepted[-1] == self.tokenizer.eos_token_id:
                break

        total_time = (time.perf_counter() - total_start) * 1000.0
        full_ids = context_ids[0].tolist() + generated
        gen_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)

        return SpecResult(
            generated_text=gen_text,
            generated_token_ids=generated,
            strategy_used=strategy,
            total_time_ms=total_time,
            total_draft_time_ms=total_draft_time,
            total_verify_time_ms=total_verify_time,
            num_steps=steps,
            avg_accept_rate=total_accepted / max(total_drafted, 1),
            tokens_per_second=len(generated) / max(total_time / 1000.0, 0.001),
        )

    @torch.no_grad()
    def generate_baseline(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> SpecResult:
        """自回归基线：用于加速比计算"""
        context_ids = self.models.encode(prompt)
        generated: List[int] = []
        total_start = time.perf_counter()
        past_kv = None
        current = context_ids

        for _ in range(max_new_tokens):
            outputs = self.models.target_forward(current, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature <= 0.0:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())

            generated.append(nxt)
            current = torch.tensor([[nxt]], device=self.device)
            if nxt == self.tokenizer.eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - total_start) * 1000.0
        full_ids = context_ids[0].tolist() + generated

        return SpecResult(
            generated_text=self.tokenizer.decode(full_ids, skip_special_tokens=False),
            generated_token_ids=generated,
            strategy_used=DraftStrategy.SEQUENTIAL,
            total_time_ms=total_time,
            tokens_per_second=len(generated) / max(total_time / 1000.0, 0.001),
        )

    # ── UNIFIED 模式（创新①）────────────────────────────────────────

    @torch.no_grad()
    def _generate_unified(
        self, prompt: str, max_new_tokens: int, k: int, temperature: float,
        top_k: int, max_depth: int, use_target_context: bool, verbose: bool,
    ) -> SpecResult:
        """
        创新①：Unified Strategy Selector

        策略选择逻辑（基于 EAGLE-2 的 confidence ≈ acceptance rate 洞察）:
          - 如果 draft model 最近 2 步的 confidence 极高 (>0.9) → SEQUENTIAL
          - 如果 entropy 高 (>0.8) → TREE 或 DYNAMIC
          - 如果 k > 5 且 GPU 利用率低 → PIPELINE
          - 否则根据接受率微调
        """
        context_ids = self.models.encode(prompt)
        generated: List[int] = []
        total_draft_time = 0.0
        total_verify_time = 0.0
        total_start = time.perf_counter()
        steps = 0
        total_accepted = 0
        total_drafted = 0
        switches: List[str] = []

        current_strategy = DraftStrategy.SEQUENTIAL
        current_k = k

        while len(generated) < max_new_tokens:
            steps += 1

            # 策略决策
            if steps == 1:
                current_strategy = DraftStrategy.SEQUENTIAL
                current_k = k
            else:
                current_strategy, current_k = self._select_strategy(
                    self._prev_accept_rate, self._prev_entropy, k,
                )
                if verbose and (not switches or switches[-1] != current_strategy.value):
                    switches.append(f"Step{steps}:{current_strategy.value}")

            if generated:
                input_ids = torch.cat([
                    context_ids,
                    torch.tensor([generated], device=self.device),
                ], dim=1)
            else:
                input_ids = context_ids

            draft_result = self.draft_engine.generate(
                context_ids=input_ids, k=current_k, temperature=temperature,
                strategy=current_strategy, top_k=top_k, max_depth=max_depth,
            )
            total_draft_time += draft_result.draft_time_ms

            accepted, v_time = self.verify_engine.verify(input_ids, draft_result, temperature)
            total_verify_time += v_time
            total_accepted += len(accepted)
            total_drafted += draft_result.num_candidates

            # 更新状态
            self._prev_accept_rate = len(accepted) / max(draft_result.num_candidates, 1)
            self._prev_entropy = 1.0 - draft_result.avg_confidence

            generated.extend(accepted)

            if verbose:
                print(
                    f"  [Step {steps:3d}] 🧠→{current_strategy.value[:3]} k={current_k} | "
                    f"acc={len(accepted)}/{draft_result.num_candidates} | "
                    f"draft={draft_result.draft_time_ms:.1f}ms | "
                    f"conf={draft_result.avg_confidence:.2f}"
                )

            if not accepted or accepted[-1] == self.tokenizer.eos_token_id:
                break

        total_time = (time.perf_counter() - total_start) * 1000.0
        full_ids = context_ids[0].tolist() + generated

        return SpecResult(
            generated_text=self.tokenizer.decode(full_ids, skip_special_tokens=False),
            generated_token_ids=generated,
            strategy_used=DraftStrategy.UNIFIED,
            total_time_ms=total_time,
            total_draft_time_ms=total_draft_time,
            total_verify_time_ms=total_verify_time,
            num_steps=steps,
            avg_accept_rate=total_accepted / max(total_drafted, 1),
            tokens_per_second=len(generated) / max(total_time / 1000.0, 0.001),
            strategy_switches=switches,
        )

    def _select_strategy(
        self, accept_rate: float, entropy: float, base_k: int,
    ) -> tuple[DraftStrategy, int]:
        """
        上下文感知策略选择器。

        规则:
          - 高置信度 + 高接受率 → SEQUENTIAL (省计算)
          - 高熵(不确定) → DYNAMIC (多候选探索)
          - 中熵 → TREE (平衡)
          - 长序列 → PIPELINE (并行友好)
        """
        if accept_rate > 0.85 and entropy < 0.3:
            return DraftStrategy.SEQUENTIAL, min(base_k, 7)
        elif entropy > 0.7:
            return DraftStrategy.DYNAMIC, base_k
        elif accept_rate < 0.5:
            return DraftStrategy.TREE, base_k
        elif base_k > 7:
            return DraftStrategy.PIPELINE, base_k
        else:
            return DraftStrategy.SEQUENTIAL, base_k
