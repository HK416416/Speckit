"""
loader.py — 模型加载与 KV Cache 管理
======================================
职责:
  1. 加载 target model 和 draft model（支持 8-bit 量化）
  2. 管理 KV Cache 的复用与增量更新
  3. Tokenizer 管理

论文关联:
  - Leviathan: KV Cache 复用是 draft-verify 框架的必备工程基础
  - DFlash: 提供了 target context feature 提取接口（第 6.3 节创新③）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ModelManager:
    """
    统一管理 target model 和 draft model 的加载、前向、KV Cache。

    继承自旧版 SpeculativeDecoder 的模型管理逻辑，重构为独立管理类。
    """

    def __init__(
        self,
        target_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda",
        draft_load_in_8bit: bool = False,
    ):
        self.device = device

        print(f"[ModelManager] 加载 Target Model: {target_model_name}")
        self.target_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=False,
            trust_remote_code=True,
        )
        self.target_model.eval()

        print(f"[ModelManager] 加载 Draft Model: {draft_model_name}")
        self.draft_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            load_in_8bit=draft_load_in_8bit,
            trust_remote_code=True,
        )
        self.draft_model.eval()

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            target_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 模型配置信息
        self.target_num_layers = self._get_num_layers(self.target_model)
        self.draft_hidden_size = self.draft_model.config.hidden_size
        self.target_hidden_size = self.target_model.config.hidden_size

        print(f"  Target: {self.target_num_layers} layers, "
              f"hidden_size={self.target_hidden_size}")
        print(f"  Draft: hidden_size={self.draft_hidden_size}")

        self._warmup()

    def _get_num_layers(self, model: PreTrainedModel) -> int:
        """获取模型的 Transformer 层数"""
        config = model.config
        for attr in ["num_hidden_layers", "n_layer", "num_layers"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 0

    def _warmup(self) -> None:
        """GPU warmup"""
        if self.device != "cuda":
            return
        dummy = torch.tensor([[self.tokenizer.eos_token_id or 0]], device=self.device)
        with torch.no_grad():
            self.target_model(dummy)
            self.draft_model(dummy)
        torch.cuda.synchronize()

    def encode(self, text: str) -> torch.Tensor:
        """编码文本为 token IDs"""
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """解码 token IDs 为文本"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special)

    @torch.no_grad()
    def target_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = True,
    ):
        """
        Target model 单次前向。

        同时返回 hidden_states，供 DFlash-inspired target context 提取使用。
        """
        outputs = self.target_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,  # DFlash: 提取中间层 hidden states
        )
        return outputs

    @torch.no_grad()
    def draft_forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[tuple] = None,
        use_cache: bool = True,
    ):
        """Draft model 前向（KV Cache 增量复用 — Leviathan 基础）"""
        return self.draft_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @torch.no_grad()
    def extract_target_context(
        self,
        input_ids: torch.Tensor,
        num_layers: int = 1,
    ) -> torch.Tensor:
        """
        提取 target model 的中间层 hidden states（DFlash 启发）。

        DFlash 的核心创新是 KV Injection——将 target model 的多个中间层
        hidden features 注入 draft model 的每一层。本方法提供基础接口，
        将中间层 hidden states 提出来供 draft 阶段使用。

        Args:
            input_ids: 输入 token IDs
            num_layers: 提取的层数（默认取中间 1 层）

        Returns:
            target context tensor: (batch, seq_len, target_hidden_size)
            或平均融合后的: (batch, seq_len, hidden_size)
        """
        outputs = self.target_model(
            input_ids=input_ids,
            output_hidden_states=True,
        )

        if not outputs.hidden_states:
            return None

        # 选择中间 num_layers 层
        total_layers = len(outputs.hidden_states) - 1  # 去掉 embedding 层
        if total_layers <= 0:
            return None

        indices = []
        step = total_layers // (num_layers + 1)
        for i in range(1, num_layers + 1):
            indices.append(i * step)

        if num_layers == 1:
            # 单层: 直接返回该层 hidden states
            return outputs.hidden_states[indices[0]][:, -1:, :]  # 最后一个 token

        # 多层: 取平均值融合
        stacked = torch.stack(
            [outputs.hidden_states[i][:, -1:, :] for i in indices], dim=0
        )
        return stacked.mean(dim=0)  # (1, 1, hidden_size)

    @torch.no_grad()
    def project_context_to_draft(
        self,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 target context 投影到 draft model 的特征空间（DFlash 启发）。

        使用简单的线性投影将 target hidden_size 映射到 draft hidden_size。
        这个投影矩阵可以在实现中动态创建。

        Args:
            target_context: (1, 1, target_hidden_size)

        Returns:
            projected: (1, 1, draft_hidden_size)
        """
        # 简易线性投影（更复杂可用 MLP）
        weight = torch.randn(
            self.target_hidden_size, self.draft_hidden_size,
            device=self.device, dtype=torch.float16,
        ) * 0.02  # 小随机初始化

        return target_context.to(torch.float16) @ weight
