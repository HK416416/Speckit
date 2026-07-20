"""
verify.py — Verify Engine: 序列验证 + 树形批量验证
=====================================================
论文关联:
  - Leviathan (ICML 2023): 标准 rejection sampling（序列验证）
  - SpecInfer (ASPLOS 2024): tree-based parallel decoding + topology-aware verify

支持两种验证模式:
  1. Sequential Verify: 逐 token rejection sampling（Leviathan）
  2. Tree Verify: 树形结构的批量验证（SpecInfer inspired，简化版）
"""

from __future__ import annotations

import random
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

from .strategy import DraftStrategy
from .draft import DraftResult


class VerifyEngine:
    """验证引擎: 对 draft candidates 执行 rejection sampling 验证"""

    def __init__(self, models, device: str = "cuda"):
        self.models = models
        self.device = device
        self.tokenizer = models.tokenizer

    def verify(
        self,
        context_ids: torch.Tensor,
        draft_result: DraftResult,
        temperature: float,
    ) -> Tuple[List[int], float]:
        """
        统一验证入口: 根据 draft 策略选择验证方式。

        Returns:
            (accepted_tokens, verify_time_ms)
        """
        if draft_result.strategy in (DraftStrategy.TREE, DraftStrategy.DYNAMIC):
            return self._verify_tree(context_ids, draft_result, temperature)
        else:
            return self._verify_sequential(context_ids, draft_result, temperature)

    # ── 序列验证 (Leviathan, ICML 2023) ─────────────────────────────

    def _verify_sequential(
        self, context_ids: torch.Tensor, draft_result: DraftResult, temperature: float,
    ) -> Tuple[List[int], float]:
        """
        标准 rejection sampling 验证（Leviathan Algorithm 1）。

        核心公式:
          accept if random() < min(1, p_target(x) / p_draft(x))
          on reject: sample from norm(max(0, p_target - p_draft))
        """
        import time
        start = time.perf_counter()

        draft_tokens = draft_result.draft_tokens
        draft_probs = draft_result.draft_probs
        k = len(draft_tokens)

        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        full_input = torch.cat([context_ids, draft_tensor], dim=1)

        outputs = self.models.target_forward(full_input, use_cache=False)
        target_logits = outputs.logits[0]
        context_len = context_ids.shape[1]
        verify_logits = target_logits[context_len - 1: context_len - 1 + k]

        accepted: List[int] = []

        for i in range(k):
            target_probs = F.softmax(verify_logits[i] / max(temperature, 1e-6), dim=-1)
            dt = draft_tokens[i]

            if draft_probs and i < len(draft_probs) and temperature > 0:
                p_draft = draft_probs[i][dt].item()
                p_target = target_probs[dt].item()
                if p_draft > 0 and random.random() < min(1.0, p_target / max(p_draft, 1e-10)):
                    accepted.append(dt)
                else:
                    adjusted = torch.clamp(target_probs - draft_probs[i], min=0.0)
                    denom = adjusted.sum()
                    new_tok = int(torch.multinomial(adjusted, 1).item()) if denom > 0 else int(target_probs.argmax().item())
                    accepted.append(new_tok)
                    break
            else:
                target_best = int(target_probs.argmax(dim=-1).item())
                if target_best == dt:
                    accepted.append(dt)
                else:
                    accepted.append(target_best)
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        return accepted, elapsed

    # ── 树形验证 (SpecInfer inspired, ASPLOS 2024) ──────────────────

    def _verify_tree(
        self, context_ids: torch.Tensor, draft_result: DraftResult, temperature: float,
    ) -> Tuple[List[int], float]:
        """
        SpecInfer 风格的树形验证（简化版）。

        由于 RTX 4050 显存和 Python 层实现限制，不做完整的 topology-aware
        causal mask kernel，而是用简化的路径遍历验证。

        核心思想（论文 Section 4.3）:
          - 从根节点开始，沿树路径验证
          - 每条路径独立做 rejection sampling
          - 选接受 token 最多/confidence 最高的路径
        """
        import time
        start = time.perf_counter()

        draft_tokens = draft_result.draft_tokens
        draft_probs = draft_result.draft_probs
        tree_structure = draft_result.tree_structure

        if not tree_structure or "parents" not in tree_structure:
            # Fallback to sequential if no tree structure
            return self._verify_sequential(context_ids, draft_result, temperature)

        # 简化: 用 Sequential verify 验证整个展平序列
        # 实际树形验证需要更复杂的 attention mask，这里作为概念实现
        full_text_ids = context_ids.tolist()[0] + draft_tokens
        full_tensor = torch.tensor([full_text_ids], device=self.device)
        outputs = self.models.target_forward(full_tensor, use_cache=False)
        target_logits = outputs.logits[0]
        context_len = context_ids.shape[1]
        verify_logits = target_logits[context_len - 1: context_len - 1 + len(draft_tokens)]

        accepted: List[int] = []

        for i in range(len(draft_tokens)):
            target_probs = F.softmax(verify_logits[i] / max(temperature, 1e-6), dim=-1)
            dt = draft_tokens[i]

            if draft_probs and i < len(draft_probs) and temperature > 0:
                p_draft = draft_probs[i][dt].item()
                p_target = target_probs[dt].item()
                if p_draft > 0 and random.random() < min(1.0, p_target / max(p_draft, 1e-10)):
                    accepted.append(dt)
                else:
                    adjusted = torch.clamp(target_probs - draft_probs[i], min=0.0)
                    denom = adjusted.sum()
                    new_tok = int(torch.multinomial(adjusted, 1).item()) if denom > 0 else int(target_probs.argmax().item())
                    accepted.append(new_tok)
                    break
            else:
                target_best = int(target_probs.argmax(dim=-1).item())
                if target_best == dt:
                    accepted.append(dt)
                else:
                    accepted.append(target_best)
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0
        return accepted, elapsed
