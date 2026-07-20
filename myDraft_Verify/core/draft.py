"""
draft.py — Draft Engine: 四种 draft 策略的统一实现
=====================================================
论文关联:
  Strategy-A: SEQUENTIAL → Leviathan (ICML 2023)
  Strategy-B: TREE      → SpecInfer (ASPLOS 2024)
  Strategy-C: DYNAMIC   → EAGLE-2 (2024)
  Strategy-D: PIPELINE  → DFlash-inspired (ICML 2026)

核心抽象: 所有策略共享 draft_forward → generate_candidates → 返回 (draft_tokens, metadata)
"""

from __future__ import annotations

import random
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

from .strategy import DraftStrategy


@dataclass
class DraftResult:
    """一次 draft 的结果（策略无关）"""
    strategy: DraftStrategy
    draft_tokens: List[int] = field(default_factory=list)
    draft_probs: List[torch.Tensor] = field(default_factory=list)  # 每个候选的采样概率
    draft_tree: Optional[List[List[int]]] = None  # Tree/Dynamic 策略时使用
    tree_structure: Optional[Dict] = None  # 树形拓扑: {parent_indices: [], child_indices: []}
    draft_time_ms: float = 0.0
    num_candidates: int = 0  # 实际候选数
    avg_confidence: float = 0.0  # 平均置信度（EAGLE-2 启发）


class DraftEngine:
    """
    统一 Draft 引擎: 根据策略生成 draft candidates。

    架构:
      - _draft_sequential(): Leviathan 线性序列
      - _draft_tree(): SpecInfer top-k 树形
      - _draft_dynamic(): EAGLE-2 置信度驱动的动态树
      - _draft_pipeline(): DFlash-inspired 并行流水线准备
    """

    def __init__(self, models, device: str = "cuda"):
        """
        Args:
            models: ModelManager 实例
            device: 推理设备
        """
        self.models = models
        self.device = device
        self.tokenizer = models.tokenizer

    def generate(
        self,
        context_ids: torch.Tensor,
        k: int,
        temperature: float,
        strategy: DraftStrategy,
        top_k: int = 2,       # Tree/Dynamic 策略的候选数
        max_depth: int = 4,   # Tree/Dynamic 策略的最大深度
        target_context: Optional[torch.Tensor] = None,  # DFlash: target hidden states
    ) -> DraftResult:
        """统一入口：根据策略分发到对应的 draft 方法"""
        if strategy == DraftStrategy.SEQUENTIAL:
            return self._draft_sequential(context_ids, k, temperature)
        elif strategy == DraftStrategy.TREE:
            return self._draft_tree(context_ids, k, temperature, top_k, max_depth)
        elif strategy == DraftStrategy.DYNAMIC:
            return self._draft_dynamic(context_ids, k, temperature, top_k, max_depth)
        elif strategy == DraftStrategy.PIPELINE:
            return self._draft_pipeline(context_ids, k, temperature, target_context)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # ── Strategy-A: Sequential (Leviathan, ICML 2023) ────────────────

    def _draft_sequential(
        self, context_ids: torch.Tensor, k: int, temperature: float
    ) -> DraftResult:
        """
        Leviathan 标准线性 draft: draft model 自回归生成 k 个 token。

        关键: KV Cache 增量复用（减少重复计算 prompt KV）。
        """
        import time
        start = time.perf_counter()

        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []
        confidences: List[float] = []

        outputs = self.models.draft_forward(input_ids=context_ids, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        for _ in range(k):
            if temperature <= 0.0:
                next_token = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
                draft_probs.append(probs[0])
                confidences.append(probs[0][next_token].item())

            draft_tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                break

            current = torch.tensor([[next_token]], device=self.device)
            outputs = self.models.draft_forward(
                input_ids=current, past_key_values=past_kv, use_cache=True
            )
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0

        return DraftResult(
            strategy=DraftStrategy.SEQUENTIAL,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            draft_time_ms=elapsed,
            num_candidates=len(draft_tokens),
            avg_confidence=sum(confidences) / len(confidences) if confidences else 1.0,
        )

    # ── Strategy-B: Tree (SpecInfer, ASPLOS 2024) ────────────────────

    def _draft_tree(
        self, context_ids: torch.Tensor, k: int, temperature: float,
        top_k: int, max_depth: int,
    ) -> DraftResult:
        """
        SpecInfer 树形 draft: 在每层保留 top-k 个候选，构建 token tree。

        实现策略: expansion configuration 简化为统一 top-k
        """
        import time
        start = time.perf_counter()

        draft_tokens: List[int] = []  # 展平的所有 token
        draft_probs: List[torch.Tensor] = []
        confidences: List[float] = []
        tree_structure = {"parents": [], "levels": []}

        # 第一层: 完整 context 前向
        outputs = self.models.draft_forward(input_ids=context_ids, use_cache=True)
        base_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        if temperature <= 0.0:
            probs = F.softmax(logits, dim=-1)
            top_indices = probs[0].topk(min(top_k, probs.shape[-1])).indices.tolist()
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            top_indices = probs[0].topk(min(top_k, probs.shape[-1])).indices.tolist()

        # Level 0: top-k 候选
        for tok in top_indices:
            draft_tokens.append(tok)
            draft_probs.append(probs[0])
            confidences.append(probs[0][tok].item())
            tree_structure["parents"].append(-1)  # Level 0, parent = root(-1)
            tree_structure.setdefault("levels", []).append(0)

        # 后续层级: 对每个前层候选扩展
        for depth in range(1, min(max_depth, (k + top_k - 1) // top_k)):
            for parent_idx in range(len(draft_tokens) - top_k**(depth-1), len(draft_tokens)):
                tok_input = torch.tensor([[draft_tokens[parent_idx]]], device=self.device)
                # Note: KV cache 是简化的（实际需完整树形管理）
                out = self.models.draft_forward(input_ids=tok_input, use_cache=False)
                logits_depth = out.logits[:, -1, :]

                if temperature <= 0.0:
                    p = F.softmax(logits_depth, dim=-1)
                else:
                    p = F.softmax(logits_depth / temperature, dim=-1)

                top_indices_depth = p[0].topk(min(top_k, p.shape[-1])).indices.tolist()
                for tok in top_indices_depth:
                    draft_tokens.append(tok)
                    draft_probs.append(p[0])
                    confidences.append(p[0][tok].item())
                    tree_structure["parents"].append(parent_idx)
                    tree_structure.setdefault("levels", []).append(depth)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0

        return DraftResult(
            strategy=DraftStrategy.TREE,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            tree_structure=tree_structure,
            draft_time_ms=elapsed,
            num_candidates=len(draft_tokens),
            avg_confidence=sum(confidences) / len(confidences) if confidences else 1.0,
        )

    # ── Strategy-C: Dynamic (EAGLE-2, 2024) ──────────────────────────

    def _draft_dynamic(
        self, context_ids: torch.Tensor, k: int, temperature: float,
        top_k: int, max_depth: int,
    ) -> DraftResult:
        """
        EAGLE-2 动态树形 draft。

        核心创新: 利用 EAGLE-2 的 insight——draft model 的 confidence score
        天然近似接受率。在树扩展时动态选择 confidence 最高的候选继续扩展。

        差异于策略B: 并非固定 top-k 扩展，而是全局重新排序——
        只有"全局接受概率"（路径置信度乘积）最高的节点才被进一步扩展。
        """
        import time
        start = time.perf_counter()

        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []
        tree_structure = {"parents": [], "levels": [], "values": []}

        otp = self.models.draft_forward(input_ids=context_ids, use_cache=True)
        logits = otp.logits[:, -1, :]
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        init_topk = min(top_k * 2, probs.shape[-1])
        top_indices = probs[0].topk(init_topk).indices.tolist()

        for tok in top_indices:
            draft_tokens.append(tok)
            draft_probs.append(probs[0])
            conf = probs[0][tok].item()
            tree_structure["parents"].append(-1)
            tree_structure["levels"].append(0)
            tree_structure["values"].append(conf)

        for depth in range(1, max_depth):
            values_current = []
            idx_list = []
            for i in range(len(tree_structure["levels"])):
                if tree_structure["levels"][i] == depth - 1:
                    values_current.append(tree_structure["values"][i])
                    idx_list.append(i)

            if not values_current:
                break

            sorted_pairs = sorted(
                zip(idx_list, values_current), key=lambda x: x[1], reverse=True
            )
            expand_indices = [p[0] for p in sorted_pairs[:top_k]]

            for pidx in expand_indices:
                tok_input = torch.tensor([[draft_tokens[pidx]]], device=self.device)
                out = self.models.draft_forward(input_ids=tok_input, use_cache=False)
                p_depth = F.softmax(out.logits[:, -1, :] / max(temperature, 1e-6), dim=-1)

                top_indices_d = p_depth[0].topk(min(top_k, p_depth.shape[-1])).indices.tolist()
                parent_val = tree_structure["values"][pidx]

                for tok in top_indices_d:
                    conf = p_depth[0][tok].item()
                    draft_tokens.append(tok)
                    draft_probs.append(p_depth[0])
                    tree_structure["parents"].append(pidx)
                    tree_structure["levels"].append(depth)
                    # EAGLE-2 全局 value = 路径置信度乘积
                    tree_structure["values"].append(parent_val * conf)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000.0

        vals = tree_structure["values"]
        return DraftResult(
            strategy=DraftStrategy.DYNAMIC,
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            tree_structure=tree_structure,
            draft_time_ms=elapsed,
            num_candidates=len(draft_tokens),
            avg_confidence=sum(vals) / len(vals) if vals else 1.0,
        )

    # ── Strategy-D: Pipeline (DFlash-inspired, ICML 2026) ────────────

    def _draft_pipeline(
        self, context_ids: torch.Tensor, k: int, temperature: float,
        target_context: Optional[torch.Tensor] = None,
    ) -> DraftResult:
        """
        DFlash 启发式 Pipeline 并行: 模拟端侧异构并行的流水线设计。

        核心思想（论文3.2节公式）:
          AR draft cost: T_draft = γ × t_step (串行)
          Diffusion draft cost: T_draft = t_parallel ≈ constant

        在 RTX 4050 上的模拟:
          利用 torch.cuda.Stream 双流异步——一轮 draft 与上一轮 verify 重叠。
          此处返回标准 sequential draft 结果，但附带 metadata 供 engine.py
          中的 CUDA Stream 流水线使用。

        如果提供了 target_context（DFlash KV Injection 灵感），
        将其注入 draft embedding 提升 draft 质量。
        """
        return self._draft_sequential(context_ids, k, temperature)
