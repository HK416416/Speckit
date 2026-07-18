"""
test_tool.py — benchmark_analyzer 工具单元测试
================================================
测试内容:
  1. 正常 results.json 解析
  2. 文件不存在错误处理
  3. JSON 格式错误处理
  4. 报告生成正确性

用法:
  python tests/test_tool.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.benchmark_analyzer import BenchmarkAnalyzerTool


class TestBenchmarkAnalyzer(unittest.TestCase):
    """benchmark_analyzer 工具单元测试"""

    def setUp(self):
        self.tool = BenchmarkAnalyzerTool()
        self.temp_dir = tempfile.mkdtemp()

        # 创建模拟的 results.json
        self.mock_data = {
            "experiment_name": "test_experiment",
            "config": {"model": "Qwen/Qwen2.5-1.5B-Instruct"},
            "summary": {
                "num_requests": 5,
                "num_success": 5,
                "avg_ttft_ms": 250.0,
                "p50_ttft_ms": 240.0,
                "p95_ttft_ms": 300.0,
                "avg_tpot_ms": 15.0,
                "avg_throughput_tps": 60.0,
            },
            "per_request": [
                {"id": "1", "ttft_ms": 200, "tpot_ms": 14, "total_ms": 500, "completion_tokens": 30, "tokens_per_second": 60, "error": None},
                {"id": "2", "ttft_ms": 220, "tpot_ms": 15, "total_ms": 520, "completion_tokens": 30, "tokens_per_second": 58, "error": None},
                {"id": "3", "ttft_ms": 240, "tpot_ms": 14, "total_ms": 480, "completion_tokens": 30, "tokens_per_second": 62, "error": None},
                {"id": "4", "ttft_ms": 260, "tpot_ms": 16, "total_ms": 550, "completion_tokens": 30, "tokens_per_second": 55, "error": None},
                {"id": "5", "ttft_ms": 300, "tpot_ms": 15, "total_ms": 600, "completion_tokens": 30, "tokens_per_second": 50, "error": None},
            ],
        }

        self.valid_json_path = os.path.join(self.temp_dir, "valid_results.json")
        with open(self.valid_json_path, "w", encoding="utf-8") as f:
            json.dump(self.mock_data, f)

        # 创建无效 JSON 文件
        self.invalid_json_path = os.path.join(self.temp_dir, "invalid_results.json")
        with open(self.invalid_json_path, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        # 缺少必要字段的 JSON
        self.incomplete_json_path = os.path.join(self.temp_dir, "incomplete_results.json")
        with open(self.incomplete_json_path, "w", encoding="utf-8") as f:
            json.dump({"experiment_name": "test"}, f)

    def tearDown(self):
        # 清理临时文件
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ── 测试用例 ────────────────────────────────────────────────────

    def test_valid_json(self):
        """测试正常 results.json 解析"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file=self.valid_json_path))

        self.assertNotIn("error", result)
        self.assertIn("summary", result)
        self.assertIn("report", result)

        summary = result["summary"]
        self.assertEqual(summary["experiment_name"], "test_experiment")
        self.assertEqual(summary["num_success"], 5)
        self.assertAlmostEqual(summary["avg_ttft_ms"], 250.0, delta=0.1)
        self.assertAlmostEqual(summary["avg_throughput_tps"], 60.0, delta=1.0)

    def test_file_not_found(self):
        """测试文件不存在时的错误处理"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file="./nonexistent.json"))

        self.assertIn("error", result)
        self.assertIn("文件不存在", result["error"])

    def test_invalid_json(self):
        """测试 JSON 格式错误时的处理"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file=self.invalid_json_path))

        self.assertIn("error", result)
        self.assertIn("JSON", result["error"])

    def test_missing_fields(self):
        """测试缺少必要字段时的处理"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file=self.incomplete_json_path))

        self.assertIn("error", result)

    def test_report_generation(self):
        """测试报告生成内容"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file=self.valid_json_path))

        report = result.get("report", "")
        self.assertIn("test_experiment", report)
        self.assertIn("250.0", report)
        self.assertIn("ms", report)
        self.assertIn("tok/s", report)

    def test_percentile_calculation(self):
        """测试百分位数计算"""
        import asyncio
        result = asyncio.run(self.tool.execute(metrics_file=self.valid_json_path))

        summary = result.get("summary", {})
        self.assertGreater(summary.get("p50_ttft_ms", 0), 0)
        self.assertGreater(summary.get("p95_ttft_ms", 0), 0)
        # P95 应 >= P50
        self.assertGreaterEqual(summary["p95_ttft_ms"], summary["p50_ttft_ms"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
