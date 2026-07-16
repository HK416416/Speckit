"""
speculative_decoding.py — 手写投机推理 (Speculative Decoding) 框架
=====================================================================
基于 draft-verify 框架的完整实现，支持:
  - 三种采样策略: greedy / rejection sampling / entropy-adaptive
  - KV Cache 增量复用 (优化点①)
  - Draft 长度动态调整 (优化点②)
  - 详细的耗时记录与日志

论文参考:
  Leviathan et al. (ICML 2023) — Fast Inference from Transformers via Speculative Decoding
  Chen et al. (NeurIPS 2023) — Accelerating Large Language Model Decoding with Speculative Sampling
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


# ──────────────────────────────────────────────────────────────────────
# 数据类型定义
# ──────────────────────────────────────────────────────────────────────

class SamplingStrategy(Enum):
    """采样策略枚举"""
    GREEDY = "greedy"                         # 贪婪解码: 取 argmax
    STANDARD = "standard"                     # 标准投机采样: rejection sampling
    ENTROPY_ADAPTIVE = "entropy_adaptive"     # 基于熵的动态调整


@dataclass
class DecodeStep:
    """单次 draft-verify 循环的详细记录"""
    step_id: int = 0
    draft_length_k: int = 5
    draft_tokens: List[int] = field(default_factory=list)
    draft_probs: List[torch.Tensor] = field(default_factory=list)
    accepted_tokens: List[int] = field(default_factory=list)
    num_accepted: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    sample_time_ms: float = 0.0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """本轮接受率"""
        if self.draft_length_k == 0:
            return 0.0
        return self.num_accepted / self.draft_length_k


@dataclass
class DecodeResult:
    """完整推理结果"""
    generated_text: str = ""
    generated_token_ids: List[int] = field(default_factory=list)
    total_time_ms: float = 0.0
    total_draft_time_ms: float = 0.0
    total_verify_time_ms: float = 0.0
    steps: List[DecodeStep] = field(default_factory=list)

    @property
    def avg_acceptance_rate(self) -> float:
        if not self.steps:
            return 0.0
        total_accepted = sum(s.num_accepted for s in self.steps)
        total_drafted = sum(s.draft_length_k for s in self.steps)
        return total_accepted / total_drafted if total_drafted > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return len(self.generated_token_ids) / (self.total_time_ms / 1000.0)


# ──────────────────────────────────────────────────────────────────────
# 核心类: SpeculativeDecoder
# ──────────────────────────────────────────────────────────────────────

class SpeculativeDecoder:
    """
    投机推理解码器。

    用法:
        decoder = SpeculativeDecoder(
            target_model_name="Qwen/Qwen2.5-1.5B-Instruct",
            draft_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="cuda",
        )
        result = decoder.generate(
            prompt="请解释什么是机器学习？",
            max_new_tokens=128,
            draft_k=5,
            strategy=SamplingStrategy.STANDARD,
        )
    """

    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        device: str = "cuda",
        target_load_in_8bit: bool = False,
        draft_load_in_8bit: bool = False,
    ):
        """
        初始化投机推理解码器。

        Args:
            target_model_name: Target model 的 HuggingFace 模型名或路径
            draft_model_name: Draft model 的 HuggingFace 模型名或路径
            device: 推理设备 ("cuda" 或 "cpu")
            target_load_in_8bit: 是否对 target model 使用 8-bit 量化
            draft_load_in_8bit: 是否对 draft model 使用 8-bit 量化 (优化点④)
        """
        self.device = device
        print(f"[SpeculativeDecoder] 加载 Target Model: {target_model_name} ...")
        self.target_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=target_load_in_8bit,
            trust_remote_code=True,
        )
        self.target_model.eval()

        print(f"[SpeculativeDecoder] 加载 Draft Model: {draft_model_name} ...")
        self.draft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=draft_load_in_8bit,
            trust_remote_code=True,
        )
        self.draft_model.eval()

        # 共用 Tokenizer (优先用 target 的)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            target_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置
        self.target_model_name = target_model_name
        self.draft_model_name = draft_model_name
        self._warmup()

    def _warmup(self) -> None:
        """GPU warmup: 降低首次推理延迟抖动"""
        if self.device != "cuda":
            return
        dummy_ids = torch.tensor([[self.tokenizer.eos_token_id or 0]], device=self.device)
        with torch.no_grad():
            self.target_model(dummy_ids)
            self.draft_model(dummy_ids)
        torch.cuda.synchronize()

    # ── 公开 API ────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        draft_k: int = 5,
        temperature: float = 0.0,
        strategy: SamplingStrategy = SamplingStrategy.STANDARD,
        adaptive_k_min: int = 2,
        adaptive_k_max: int = 10,
        verbose: bool = True,
        return_steps: bool = False,
    ) -> DecodeResult:
        """
        执行投机推理生成。

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            draft_k: draft model 每次预测的 token 数量
            temperature: 采样温度 (0.0 = greedy)
            strategy: 采样策略
            adaptive_k_min: 自适应 k 的最小值 (仅 ENTROPY_ADAPTIVE)
            adaptive_k_max: 自适应 k 的最大值 (仅 ENTROPY_ADAPTIVE)
            verbose: 是否打印每轮详情
            return_steps: 是否在结果中返回每步详情

        Returns:
            DecodeResult 包含生成文本和性能指标
        """
        # 编码 prompt
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_ids: List[int] = []
        steps: List[DecodeStep] = []
        current_k = draft_k

        total_start = time.perf_counter()
        total_draft_time = 0.0
        total_verify_time = 0.0

        while len(generated_ids) < max_new_tokens:
            step_id = len(steps) + 1

            # ── Step 1: DRAFT ──────────────────────────────────────
            # 构建 draft 的输入: prompt + already_generated
            if generated_ids:
                context_ids = torch.cat([
                    prompt_ids,
                    torch.tensor([generated_ids], device=self.device),
                ], dim=1)
            else:
                context_ids = prompt_ids

            draft_start = time.perf_counter()
            draft_tokens, draft_probs = self._draft(
                context_ids=context_ids,
                k=current_k,
                temperature=temperature,
                strategy=strategy,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_time = (time.perf_counter() - draft_start) * 1000.0  # ms

            # 安全检查: draft 为空则退出
            if not draft_tokens:
                break

            # ── Step 2: VERIFY ─────────────────────────────────────
            verify_start = time.perf_counter()
            accepted_tokens = self._verify(
                context_ids=context_ids,
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                temperature=temperature,
                strategy=strategy,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            verify_time = (time.perf_counter() - verify_start) * 1000.0  # ms

            # ── Step 3: 记录 & 拼接 ─────────────────────────────────
            step = DecodeStep(
                step_id=step_id,
                draft_length_k=current_k,
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                accepted_tokens=accepted_tokens,
                num_accepted=len(accepted_tokens),
                draft_time_ms=draft_time,
                verify_time_ms=verify_time,
                total_time_ms=draft_time + verify_time,
            )
            steps.append(step)
            generated_ids.extend(accepted_tokens)
            total_draft_time += draft_time
            total_verify_time += verify_time

            if verbose:
                accepted_text = self.tokenizer.decode(accepted_tokens) if accepted_tokens else "<eos>"
                print(
                    f"  [Step {step_id:3d}] k={current_k:2d} | "
                    f"draft={len(draft_tokens)} | accepted={len(accepted_tokens)} | "
                    f"accept_rate={step.acceptance_rate:.1%} | "
                    f"draft={draft_time:.1f}ms | verify={verify_time:.1f}ms | "
                    f"→ {accepted_text!r}"
                )

            # ── 动态调整 k (优化点②) ────────────────────────────────
            if strategy == SamplingStrategy.ENTROPY_ADAPTIVE:
                current_k = self._adjust_k(
                    acceptance_rate=step.acceptance_rate,
                    k_min=adaptive_k_min,
                    k_max=adaptive_k_max,
                    current_k=current_k,
                )

            # 终止条件
            if not accepted_tokens:
                break
            if accepted_tokens[-1] == self.tokenizer.eos_token_id:
                break

        total_time = (time.perf_counter() - total_start) * 1000.0  # ms

        # 解码生成文本
        full_ids = prompt_ids[0].tolist() + generated_ids
        generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)

        return DecodeResult(
            generated_text=generated_text,
            generated_token_ids=generated_ids,
            total_time_ms=total_time,
            total_draft_time_ms=total_draft_time,
            total_verify_time_ms=total_verify_time,
            steps=steps if return_steps else [],
        )

    @torch.no_grad()
    def generate_autoregressive_baseline(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> DecodeResult:
        """
        自回归基线: 用 target model 逐 token 生成，用于对比加速比。

        Returns:
            DecodeResult (steps 为空，仅包含 total_time 和 generated_text)
        """
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        total_start = time.perf_counter()
        generated_ids: List[int] = []
        past_key_values = None
        current_ids = prompt_ids

        for _ in range(max_new_tokens):
            outputs = self.target_model(
                input_ids=current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature <= 0.0:
                next_token = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_token)
            current_ids = torch.tensor([[next_token]], device=self.device)

            if next_token == self.tokenizer.eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = (time.perf_counter() - total_start) * 1000.0

        full_ids = prompt_ids[0].tolist() + generated_ids
        generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)

        return DecodeResult(
            generated_text=generated_text,
            generated_token_ids=generated_ids,
            total_time_ms=total_time,
        )

    # ── 内部方法 ────────────────────────────────────────────────────

    def _draft(
        self,
        context_ids: torch.Tensor,
        k: int,
        temperature: float,
        strategy: SamplingStrategy,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Draft 阶段: draft model 自回归生成 k 个 token.

        实现优化点①: KV Cache 增量复用
          - 首次 forward 传入完整 context_ids 获取 past_key_values
          - 后续 forward 仅传入新 token，复用 past_key_values

        Returns:
            (draft_tokens, draft_probs) — tokens 列表和对应的概率分布
        """
        draft_tokens: List[int] = []
        draft_probs: List[torch.Tensor] = []

        # 第一步: 用完整 context 获取 KV Cache
        outputs = self.draft_model(
            input_ids=context_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

        for _ in range(k):
            # 采样
            if temperature <= 0.0:
                next_token = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
                draft_probs.append(probs[0])  # 保存概率分布供 rejection sampling 使用

            draft_tokens.append(next_token)

            # 如果命中 EOS，提前终止
            if next_token == self.tokenizer.eos_token_id:
                break

            # 增量前向: 仅传入新 token + 复用 KV Cache
            current_input = torch.tensor([[next_token]], device=self.device)
            outputs = self.draft_model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        return draft_tokens, draft_probs

    def _verify(
        self,
        context_ids: torch.Tensor,
        draft_tokens: List[int],
        draft_probs: List[torch.Tensor],
        temperature: float,
        strategy: SamplingStrategy,
    ) -> List[int]:
        """
        Verify 阶段: target model 一次前向验证 draft tokens.

        支持两种验证模式:
          - GREEDY: 直接比较 draft token 和 target argmax
          - STANDARD: rejection sampling (保证无损)
          - ENTROPY_ADAPTIVE: 与 STANDARD 相同，仅 k 动态调整
        """
        k = len(draft_tokens)

        # 拼接 context + draft tokens → target model 一次前向
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        full_input = torch.cat([context_ids, draft_tensor], dim=1)

        outputs = self.target_model(full_input, use_cache=False)
        target_logits = outputs.logits[0]  # [seq_len, vocab_size]

        # 提取对应 draft token 位置的 target logits
        # target_logits[context_len - 1] 对应 draft_tokens[0] 的预测
        # target_logits[context_len + i - 1] 对应 draft_tokens[i] 的预测
        context_len = context_ids.shape[1]
        verify_logits = target_logits[context_len - 1 : context_len - 1 + k]  # [k, vocab_size]

        accepted_tokens: List[int] = []

        if strategy == SamplingStrategy.GREEDY:
            # 贪婪模式: 直接比对 argmax
            for i in range(k):
                target_best = int(verify_logits[i].argmax(dim=-1).item())
                if target_best == draft_tokens[i]:
                    accepted_tokens.append(draft_tokens[i])
                else:
                    accepted_tokens.append(target_best)
                    break
            return accepted_tokens

        else:
            # 标准投机采样 / 熵自适应: rejection sampling
            # 算法来自 Chen et al. (NeurIPS 2023)
            for i in range(k):
                target_probs = F.softmax(verify_logits[i] / max(temperature, 1e-6), dim=-1)
                draft_token = draft_tokens[i]

                if draft_probs and i < len(draft_probs):
                    p_draft = draft_probs[i][draft_token].item()
                    p_target = target_probs[draft_token].item()

                    # Rejection sampling
                    if p_draft > 0 and random.random() < min(1.0, p_target / p_draft):
                        accepted_tokens.append(draft_token)
                    else:
                        # 从修正分布采样
                        adjusted = torch.clamp(target_probs - draft_probs[i], min=0.0)
                        if adjusted.sum() > 0:
                            adjusted = adjusted / adjusted.sum()
                            new_token = int(torch.multinomial(adjusted, num_samples=1).item())
                        else:
                            new_token = int(target_probs.argmax(dim=-1).item())
                        accepted_tokens.append(new_token)
                        break
                else:
                    # 无 draft probs (greedy draft with low temperature): 直接取 target argmax
                    target_best = int(target_probs.argmax(dim=-1).item())
                    if target_best == draft_token:
                        accepted_tokens.append(draft_token)
                    else:
                        accepted_tokens.append(target_best)
                        break

            return accepted_tokens

    @staticmethod
    def _adjust_k(
        acceptance_rate: float,
        k_min: int,
        k_max: int,
        current_k: int,
        up_threshold: float = 0.8,
        down_threshold: float = 0.5,
    ) -> int:
        """
        优化点②: 基于上轮接受率动态调整 draft length。

        - 接受率 > up_threshold  → k + 1 (上限 k_max)
        - 接受率 < down_threshold → k - 1 (下限 k_min)
        - 否则保持当前 k
        """
        if acceptance_rate > up_threshold:
            return min(current_k + 1, k_max)
        elif acceptance_rate < down_threshold:
            return max(current_k - 1, k_min)
        return current_k


# ──────────────────────────────────────────────────────────────────────
# 辅助: 确保 random 可用
# ──────────────────────────────────────────────────────────────────────

import random
