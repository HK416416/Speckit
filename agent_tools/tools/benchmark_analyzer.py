"""
benchmark_analyzer.py — MCP 自定义工具
========================================
功能: 读取 vLLM Benchmark 输出的 results.json，
      自动分析 TTFT、TPOT、吞吐量并生成 Markdown 摘要报告。

MCP Tool 接口:
  - name: "benchmark_analyzer"
  - input_schema: { metrics_file: str, format: "json" | "csv" }
  - output: 包含各指标统计值的结构化摘要

用法 (直接测试):
  python tools/benchmark_analyzer.py --metrics-file results/baseline/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────
# MCP Tool 接口定义
# ──────────────────────────────────────────────────────────────────────

TOOL_NAME = "benchmark_analyzer"
TOOL_DESCRIPTION = (
    "读取 vLLM Benchmark 输出的 metrics 数据（JSON 格式），"
    "自动分析 TTFT、TPOT、吞吐量等性能指标，"
    "并生成 Markdown 格式的结构化摘要报告。"
    "适用于投机推理实验的性能对比分析场景。"
)

TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "metrics_file": {
            "type": "string",
            "description": "vLLM Benchmark 输出的 results.json 文件路径",
        },
        "format": {
            "type": "string",
            "enum": ["json", "csv"],
            "description": "数据文件格式，默认 json",
            "default": "json",
        },
    },
    "required": ["metrics_file"],
}


# ──────────────────────────────────────────────────────────────────────
# 数据模型
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MetricSummary:
    """性能指标汇总"""
    experiment_name: str = "unknown"
    num_requests: int = 0
    num_success: int = 0
    avg_ttft_ms: float = 0.0
    p50_ttft_ms: float = 0.0
    p95_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_throughput_tps: float = 0.0
    total_tokens: int = 0
    errors: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# 核心分析逻辑
# ──────────────────────────────────────────────────────────────────────

class BenchmarkAnalyzerTool:
    """
    MCP 工具: 分析 vLLM Benchmark 结果。

    实现 MCP Tool 接口:
      - name: str
      - description: str
      - input_schema: dict
      - execute(**kwargs) -> dict
    """

    name = TOOL_NAME
    description = TOOL_DESCRIPTION
    input_schema = TOOL_INPUT_SCHEMA

    async def execute(self, metrics_file: str, format: str = "json") -> dict:
        """
        MCP Tool execute 方法入口。

        Args:
            metrics_file: results.json 文件路径
            format: 文件格式 ("json" | "csv")

        Returns:
            包含 summary 和 report 的字典
        """
        if format != "json":
            return {"error": f"暂不支持的格式: {format}，当前仅支持 JSON"}

        # 1. 加载数据
        data = self._load_data(metrics_file)
        if "error" in data:
            return data

        # 2. 分析统计
        summary = self._analyze(data)

        # 3. 生成报告
        report = self._generate_report(summary)

        return {
            "summary": {
                "experiment_name": summary.experiment_name,
                "avg_ttft_ms": round(summary.avg_ttft_ms, 2),
                "p50_ttft_ms": round(summary.p50_ttft_ms, 2),
                "p95_ttft_ms": round(summary.p95_ttft_ms, 2),
                "avg_tpot_ms": round(summary.avg_tpot_ms, 2),
                "avg_throughput_tps": round(summary.avg_throughput_tps, 2),
                "num_success": summary.num_success,
                "num_requests": summary.num_requests,
            },
            "report": report,
        }

    # ── 内部方法 ────────────────────────────────────────────────────

    def _load_data(self, metrics_file: str) -> dict:
        """加载并验证 results.json 文件"""
        if not os.path.exists(metrics_file):
            return {"error": f"文件不存在: {metrics_file}"}

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return {"error": f"JSON 解析失败: {e}"}
        except Exception as e:
            return {"error": f"文件读取失败: {e}"}

        # 验证必要字段
        if "summary" not in data:
            return {"error": "results.json 缺少 'summary' 字段"}
        if "per_request" not in data:
            return {"error": "results.json 缺少 'per_request' 字段"}

        return data

    def _analyze(self, data: dict) -> MetricSummary:
        """从原始数据中提取并计算性能指标"""
        s = data.get("summary", {})
        per_request = data.get("per_request", [])

        # 提取 TTFT 列表
        ttft_values = []
        tpot_values = []
        throughput_values = []
        errors = []
        total_tokens = 0

        for req in per_request:
            if req.get("error"):
                errors.append(req["error"])
                continue
            if req.get("ttft_ms", 0) > 0:
                ttft_values.append(req["ttft_ms"])
            if req.get("tpot_ms", 0) > 0:
                tpot_values.append(req["tpot_ms"])
            if req.get("tokens_per_second", 0) > 0:
                throughput_values.append(req["tokens_per_second"])
            total_tokens += req.get("completion_tokens", 0)

        # 计算百分位
        def percentile(sorted_vals: List[float], p: float) -> float:
            if not sorted_vals:
                return 0.0
            k = int(len(sorted_vals) * p / 100)
            return sorted(sorted_vals)[k] if k < len(sorted_vals) else sorted_vals[-1]

        return MetricSummary(
            experiment_name=data.get("experiment_name", "unknown"),
            num_requests=len(per_request),
            num_success=len(per_request) - len(errors),
            avg_ttft_ms=s.get("avg_ttft_ms", 0.0),
            p50_ttft_ms=s.get("p50_ttft_ms", percentile(ttft_values, 50)),
            p95_ttft_ms=s.get("p95_ttft_ms", percentile(ttft_values, 95)),
            avg_tpot_ms=s.get("avg_tpot_ms", 0.0),
            avg_throughput_tps=s.get("avg_throughput_tps", 0.0),
            total_tokens=total_tokens,
            errors=errors,
        )

    def _generate_report(self, summary: MetricSummary) -> str:
        """生成 Markdown 格式的摘要报告"""
        lines = []
        lines.append(f"# Benchmark 分析报告: {summary.experiment_name}")
        lines.append("")
        lines.append("## 1. 基本统计")
        lines.append("")
        lines.append(f"- **请求总数**: {summary.num_requests}")
        lines.append(f"- **成功数**: {summary.num_success}")
        lines.append(f"- **总生成 Token 数**: {summary.total_tokens}")
        if summary.errors:
            lines.append(f"- **错误数**: {len(summary.errors)}")
        lines.append("")
        lines.append("## 2. 性能指标")
        lines.append("")
        lines.append("| 指标 | 值 | 单位 |")
        lines.append("|------|----|------|")
        lines.append(f"| 平均 TTFT | {summary.avg_ttft_ms:.1f} | ms |")
        lines.append(f"| P50 TTFT | {summary.p50_ttft_ms:.1f} | ms |")
        lines.append(f"| P95 TTFT | {summary.p95_ttft_ms:.1f} | ms |")
        lines.append(f"| 平均 TPOT | {summary.avg_tpot_ms:.1f} | ms |")
        lines.append(f"| 平均吞吐量 | {summary.avg_throughput_tps:.1f} | tok/s |")
        lines.append("")
        lines.append("## 3. 关键发现")
        lines.append("")
        if summary.avg_tpot_ms > 0:
            lines.append(f"- 首 Token 延迟 (TTFT): **{summary.avg_ttft_ms:.0f}ms**")
            lines.append(f"- 每 Token 生成时间 (TPOT): **{summary.avg_tpot_ms:.0f}ms**")
            lines.append(f"- 系统吞吐量: **{summary.avg_throughput_tps:.1f} tok/s**")
        lines.append("")
        if summary.errors:
            lines.append(f"⚠ 存在 {len(summary.errors)} 个请求失败，请检查日志。")
        else:
            lines.append("✅ 所有请求均成功完成。")
        lines.append("")

        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# 命令行入口 (独立测试)
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Analyzer — 分析 vLLM Benchmark 结果",
    )
    parser.add_argument(
        "--metrics-file", type=str, required=True,
        help="results.json 文件路径"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="报告输出路径 (默认打印到 stdout)"
    )
    args = parser.parse_args()

    tool = BenchmarkAnalyzerTool()

    # 同步调用 execute (命令行环境)
    import asyncio
    result = asyncio.run(tool.execute(metrics_file=args.metrics_file))

    if "error" in result:
        print(f"❌ 错误: {result['error']}")
        sys.exit(1)

    print(result["report"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["report"])
        print(f"\n报告已保存: {args.output}")


if __name__ == "__main__":
    main()
