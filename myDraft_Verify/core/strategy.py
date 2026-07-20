"""
strategy.py — Draft Strategy 枚举定义
=======================================
四篇论文对应的四种策略 + 创新统一选择器。

论文关联:
  SEQUENTIAL → Leviathan (ICML 2023)
  TREE      → SpecInfer (ASPLOS 2024)
  DYNAMIC   → EAGLE-2 (2024)
  PIPELINE  → DFlash inspired (ICML 2026)
  UNIFIED   → 自主创新: 上下文感知自动策略切换
"""

from __future__ import annotations
from enum import Enum


class DraftStrategy(Enum):
    """
    Draft 策略枚举。

    - SEQUENTIAL: 标准线性 draft-verify（Leviathan, 2023）
    - TREE: 树形 draft，每个位置 top-k 候选（SpecInfer, 2024）
    - DYNAMIC: 基于 confidence 动态调整树结构（EAGLE-2, 2024）
    - PIPELINE: 并行流水线 draft-verify（DFlash 启发, 2026）
    - UNIFIED: 根据上下文特征自动选择最优策略（自主创新）
    """
    SEQUENTIAL = "sequential"
    TREE = "tree"
    DYNAMIC = "dynamic"
    PIPELINE = "pipeline"
    UNIFIED = "unified"

    def paper_ref(self) -> str:
        """返回对应论文引用"""
        refs = {
            DraftStrategy.SEQUENTIAL: "Leviathan et al., ICML 2023",
            DraftStrategy.TREE: "SpecInfer (Miao et al.), ASPLOS 2024",
            DraftStrategy.DYNAMIC: "EAGLE-2 (Li et al.), 2024",
            DraftStrategy.PIPELINE: "DFlash-inspired (Chen et al., ICML 2026)",
            DraftStrategy.UNIFIED: "自主创新: Context-Aware Strategy Selector",
        }
        return refs.get(self, "Unknown")

    def icon(self) -> str:
        """策略对应的 emoji 图标"""
        icons = {
            DraftStrategy.SEQUENTIAL: "→",
            DraftStrategy.TREE: "🌳",
            DraftStrategy.DYNAMIC: "🔄",
            DraftStrategy.PIPELINE: "⚡",
            DraftStrategy.UNIFIED: "🧠",
        }
        return icons.get(self, "❓")

    @classmethod
    def all_strategies(cls) -> list[DraftStrategy]:
        """返回所有可用策略"""
        return [
            cls.SEQUENTIAL, cls.TREE, cls.DYNAMIC,
            cls.PIPELINE, cls.UNIFIED,
        ]
