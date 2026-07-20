"""
test_strategies.py — 策略单元测试
==================================
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategy import DraftStrategy


class TestDraftStrategy(unittest.TestCase):

    def test_paper_refs(self):
        """验证每个策略都有论文引用"""
        for strat in DraftStrategy:
            ref = strat.paper_ref()
            self.assertIsNotNone(ref)
            self.assertGreater(len(ref), 10)

    def test_all_strategies_count(self):
        """验证 all_strategies 返回 5 个策略"""
        all_s = DraftStrategy.all_strategies()
        self.assertEqual(len(all_s), 5)

    def test_strategy_values(self):
        """验证策略值正确"""
        self.assertEqual(DraftStrategy.SEQUENTIAL.value, "sequential")
        self.assertEqual(DraftStrategy.TREE.value, "tree")
        self.assertEqual(DraftStrategy.DYNAMIC.value, "dynamic")
        self.assertEqual(DraftStrategy.PIPELINE.value, "pipeline")
        self.assertEqual(DraftStrategy.UNIFIED.value, "unified")

    def test_icon_mapping(self):
        """验证每个策略有 icon"""
        for strat in DraftStrategy:
            icon = strat.icon()
            self.assertIsNotNone(icon)
            self.assertGreater(len(icon), 0)


class TestAnalysisMetrics(unittest.TestCase):

    def test_compute_speedup(self):
        from analysis.metrics import compute_speedup
        self.assertAlmostEqual(compute_speedup(1000, 500), 2.0)
        self.assertAlmostEqual(compute_speedup(1000, 1000), 1.0)
        self.assertEqual(compute_speedup(1000, 0), 0.0)

    def test_percentile(self):
        from analysis.metrics import percentile
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(percentile(vals, 50), 5.0)
        self.assertEqual(percentile(vals, 95), 9.0)
        self.assertEqual(percentile([], 50), 0.0)

    def test_generate_leaderboard(self):
        from analysis.metrics import ExperimentMetrics, generate_leaderboard
        metrics = [
            ExperimentMetrics(strategy="→ sequential", avg_speedup=1.0, avg_accept_rate=0.65),
            ExperimentMetrics(strategy="🌳 tree", avg_speedup=1.35, avg_accept_rate=0.78),
        ]
        board = generate_leaderboard(metrics)
        self.assertIn("sequential", board)
        self.assertIn("tree", board)
        self.assertIn("⭐", board)  # 最佳策略有星标


if __name__ == "__main__":
    unittest.main(verbosity=2)
