"""
workflow.py — speculative_experiment_runner Skill 工作流编排
=============================================================
自动化执行投机推理对比实验的完整工作流。

Skill 接口:
  - name: "speculative_experiment_runner"
  - execute(config) -> dict

使用方式:
  在 nanobot 中注册此 Skill 后，通过自然语言触发:
  "请运行投机推理对比实验"
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 添加 agent_tools 到路径（以便导入 benchmark_analyzer）
_AGENT_TOOLS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _AGENT_TOOLS_DIR not in sys.path:
    sys.path.insert(0, _AGENT_TOOLS_DIR)

from tools.benchmark_analyzer import BenchmarkAnalyzerTool

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 数据类型
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """工作流中单个步骤的执行结果"""
    step_name: str = ""
    success: bool = True
    output: str = ""
    error: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class WorkflowResult:
    """完整工作流的执行结果"""
    skill_name: str = "speculative_experiment_runner"
    success: bool = True
    steps: List[StepResult] = field(default_factory=list)
    baseline_report: Optional[str] = None
    speculative_report: Optional[str] = None
    comparison_summary: Optional[Dict] = None
    error: Optional[str] = None

    @property
    def total_duration(self) -> float:
        return sum(s.duration_seconds for s in self.steps)


# ──────────────────────────────────────────────────────────────────────
# Skill 实现
# ──────────────────────────────────────────────────────────────────────

class SpeculativeExperimentRunnerSkill:
    """
    投机推理对比实验自动化 Skill。

    工作流步骤:
      Step 1: 加载配置
      Step 2: 基线实验 (启动 vLLM → Benchmark → 停止)
      Step 3: 投机推理实验 (启动 vLLM → Benchmark → 停止)
      Step 4: 分析对比 (调用 benchmark_analyzer)
      Step 5: 清理 & 输出结果
    """

    name = "speculative_experiment_runner"
    description = (
        "自动化执行投机推理对比实验: "
        "依次运行基线 (baseline) 和投机推理 (speculative) 配置的 vLLM 服务，"
        "收集 TTFT/TPOT/吞吐量指标，生成对比报告。"
    )
    version = "1.0.0"

    # 依赖的工具
    dependencies = ["benchmark_analyzer"]

    def __init__(self):
        self.analyzer = BenchmarkAnalyzerTool()

    async def execute(
        self,
        baseline_config: str = "../../vllm_benchmark/configs/baseline.json",
        speculative_config: str = "../../vllm_benchmark/configs/speculative.json",
        num_prompts: int = 50,
        port: int = 8000,
    ) -> dict:
        """
        Skill 入口: 执行完整的投机推理对比实验工作流。

        Args:
            baseline_config: 基线配置文件路径
            speculative_config: 投机推理配置文件路径
            num_prompts: 每组实验的测试 prompt 数量
            port: vLLM 服务端口

        Returns:
            包含工作流执行结果和对比报告的字典
        """
        result = WorkflowResult()
        logger.info(f"启动 Skill: {self.name}")

        # 解析相对路径 (以 agent_tools/ 为基准)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        baseline_config = os.path.normpath(os.path.join(base_dir, baseline_config))
        speculative_config = os.path.normpath(os.path.join(base_dir, speculative_config))
        vllm_server_script = os.path.normpath(os.path.join(base_dir, "vllm_benchmark", "vllm_server.py"))
        benchmark_script = os.path.normpath(os.path.join(base_dir, "vllm_benchmark", "benchmark.py"))

        # ── Step 1: 验证配置 ───────────────────────────────────────
        step = await self._step_validate_configs(
            baseline_config, speculative_config,
            vllm_server_script, benchmark_script,
        )
        result.steps.append(step)
        if not step.success:
            result.success = False
            result.error = step.error
            return self._to_dict(result)

        # ── Step 2: 基线实验 ───────────────────────────────────────
        step = await self._step_run_experiment(
            config_path=baseline_config,
            vllm_server_script=vllm_server_script,
            benchmark_script=benchmark_script,
            num_prompts=num_prompts,
            port=port,
            experiment_name="baseline",
        )
        result.steps.append(step)
        if not step.success:
            result.success = False
            result.error = step.error
            self._cleanup(port)
            return self._to_dict(result)

        # ── Step 3: 投机推理实验 ──────────────────────────────────
        step = await self._step_run_experiment(
            config_path=speculative_config,
            vllm_server_script=vllm_server_script,
            benchmark_script=benchmark_script,
            num_prompts=num_prompts,
            port=port,
            experiment_name="speculative",
        )
        result.steps.append(step)
        if not step.success:
            result.success = False
            result.error = step.error
            self._cleanup(port)
            return self._to_dict(result)

        # ── Step 4: 分析对比 ──────────────────────────────────────
        step = await self._step_compare_results(result)
        result.steps.append(step)

        # ── Step 5: 清理 ──────────────────────────────────────────
        self._cleanup(port)
        step = StepResult(step_name="Step 5: 清理", success=True, output="vLLM 进程已停止")
        step.duration_seconds = 0.0
        result.steps.append(step)

        logger.info(f"Skill 完成: {self.name}")
        return self._to_dict(result)

    # ── 步骤实现 ────────────────────────────────────────────────────

    async def _step_validate_configs(
        self,
        baseline_config: str,
        speculative_config: str,
        vllm_server: str,
        benchmark: str,
    ) -> StepResult:
        """Step 1: 验证所有配置文件和脚本是否存在"""
        start = time.perf_counter()
        result = StepResult(step_name="Step 1: 验证配置")

        missing = []
        for path, label in [
            (baseline_config, "基线配置"),
            (speculative_config, "投机推理配置"),
            (vllm_server, "vllm_server.py"),
            (benchmark, "benchmark.py"),
        ]:
            if not os.path.exists(path):
                missing.append(f"{label}: {path}")

        if missing:
            result.success = False
            result.error = "缺少以下文件:\n  " + "\n  ".join(missing)
        else:
            result.output = "所有配置文件验证通过"

        result.duration_seconds = time.perf_counter() - start
        logger.info(f"  {result.step_name}: {'✅' if result.success else '❌'} ({result.duration_seconds:.1f}s)")
        return result

    async def _step_run_experiment(
        self,
        config_path: str,
        vllm_server_script: str,
        benchmark_script: str,
        num_prompts: int,
        port: int,
        experiment_name: str,
    ) -> StepResult:
        """Step 2/3: 启动 vLLM 服务 → 运行 Benchmark → 停止服务"""
        start = time.perf_counter()
        step_label = f"Step: {experiment_name} 实验"
        result = StepResult(step_name=step_label)

        try:
            # 2a. 启动 vLLM 服务（后台子进程）
            logger.info(f"  启动 vLLM 服务 ({experiment_name})...")
            server_proc = subprocess.Popen(
                [
                    sys.executable, vllm_server_script,
                    "--config", config_path,
                    "--port", str(port),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # 2b. 等待服务就绪（轮询 health check）
            logger.info("  等待服务就绪...")
            health_url = f"http://127.0.0.1:{port}/health"
            max_wait = 120  # 最多等 2 分钟
            wait_interval = 5
            service_ready = False

            for attempt in range(max_wait // wait_interval):
                await asyncio.sleep(wait_interval)
                try:
                    import urllib.request
                    urllib.request.urlopen(health_url, timeout=3)
                    service_ready = True
                    break
                except Exception:
                    # 轮询状态码以判断是否就绪
                    poll = server_proc.poll()
                    if poll is not None:
                        stderr_output = server_proc.stderr.read().decode(errors="ignore") if server_proc.stderr else ""
                        result.success = False
                        result.error = f"vLLM 进程意外退出 (code={poll}): {stderr_output[-200:]}"
                        break

            if not result.success:
                return result
            if not service_ready:
                result.success = False
                result.error = "vLLM 服务启动超时"
                self._cleanup(port)
                return result

            logger.info("  ✅ 服务就绪")

            # 2c. 运行 Benchmark
            logger.info(f"  运行 Benchmark ({num_prompts} prompts)...")
            bench_proc = subprocess.run(
                [
                    sys.executable, benchmark_script,
                    "--config", config_path,
                    "--num-prompts", str(num_prompts),
                    "--api-base", f"http://127.0.0.1:{port}",
                ],
                capture_output=True,
                text=True,
                timeout=300,  # 最多等 5 分钟
            )

            if bench_proc.returncode != 0:
                result.success = False
                result.error = f"Benchmark 执行失败: {bench_proc.stderr[-300:]}"
                self._cleanup(port)
                return result

            result.output = f"{experiment_name} 实验完成\n{bench_proc.stdout[-500:]}"

        except subprocess.TimeoutExpired:
            result.success = False
            result.error = "Benchmark 执行超时 (5分钟)"
        except Exception as e:
            result.success = False
            result.error = f"实验异常: {e}"
        finally:
            self._cleanup(port)

        result.duration_seconds = time.perf_counter() - start
        logger.info(f"  {result.step_name}: {'✅' if result.success else '❌'} ({result.duration_seconds:.1f}s)")
        return result

    async def _step_compare_results(self, workflow_result: WorkflowResult) -> StepResult:
        """Step 4: 调用 benchmark_analyzer 分析两组实验结果"""
        start = time.perf_counter()
        result = StepResult(step_name="Step 4: 分析对比")

        # 查找两组实验的结果文件
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        baseline_dir = os.path.join(base_dir, "vllm_benchmark", "results", "baseline")
        speculative_dir = os.path.join(base_dir, "vllm_benchmark", "results", "speculative")

        baseline_file = os.path.join(baseline_dir, "results.json")
        speculative_file = os.path.join(speculative_dir, "results.json")

        comparison = {}
        reports = []

        # 分析基线
        if os.path.exists(baseline_file):
            baseline_analysis = await self.analyzer.execute(metrics_file=baseline_file)
            reports.append(f"## 基线实验 (Baseline)\n\n{baseline_analysis.get('report', '')}")
            comparison["baseline"] = baseline_analysis.get("summary", {})
        else:
            reports.append(f"⚠ 基线结果文件不存在: {baseline_file}")
            comparison["baseline"] = {"error": "文件不存在"}

        # 分析投机推理
        if os.path.exists(speculative_file):
            speculative_analysis = await self.analyzer.execute(metrics_file=speculative_file)
            reports.append(f"## 投机推理实验 (Speculative)\n\n{speculative_analysis.get('report', '')}")
            comparison["speculative"] = speculative_analysis.get("summary", {})
        else:
            reports.append(f"⚠ 投机推理结果文件不存在: {speculative_file}")
            comparison["speculative"] = {"error": "文件不存在"}

        # 计算加速比
        baseline_tps = comparison.get("baseline", {}).get("avg_throughput_tps", 0)
        speculative_tps = comparison.get("speculative", {}).get("avg_throughput_tps", 0)
        if baseline_tps and speculative_tps and baseline_tps > 0:
            speedup = speculative_tps / baseline_tps
            comparison["speedup"] = round(speedup, 2)
            reports.append(f"\n## 加速比\n\n投机推理吞吐量加速比: **{speedup:.2f}x**")

        workflow_result.baseline_report = reports[0] if len(reports) > 0 else ""
        workflow_result.speculative_report = reports[1] if len(reports) > 1 else ""
        workflow_result.comparison_summary = comparison

        result.output = "\n\n".join(reports)
        result.duration_seconds = time.perf_counter() - start
        logger.info(f"  {result.step_name}: {'✅' if result.success else '❌'} ({result.duration_seconds:.1f}s)")
        return result

    def _cleanup(self, port: int) -> None:
        """清理 vLLM 进程"""
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "vllm*"],
                    capture_output=True,
                )
        except Exception:
            pass

    def _to_dict(self, result: WorkflowResult) -> dict:
        """将 WorkflowResult 转为可序列化的字典"""
        return {
            "skill_name": result.skill_name,
            "success": result.success,
            "error": result.error,
            "total_duration_seconds": round(result.total_duration, 1),
            "steps": [
                {
                    "step": s.step_name,
                    "success": s.success,
                    "duration_seconds": round(s.duration_seconds, 1),
                    "output": s.output[:200] if s.output else "",
                    "error": s.error,
                }
                for s in result.steps
            ],
            "comparison_summary": result.comparison_summary,
        }


# ──────────────────────────────────────────────────────────────────────
# 命令行入口 (独立测试)
# ──────────────────────────────────────────────────────────────────────

async def main():
    """独立运行 Skill 工作流进行测试"""
    print("=" * 60)
    print("  投机推理对比实验 Skill — 独立测试")
    print("=" * 60)

    skill = SpeculativeExperimentRunnerSkill()
    result = await skill.execute(
        num_prompts=5,  # 测试模式: 少量 prompt
    )

    print("\n执行结果:")
    for step in result.get("steps", []):
        status = "✅" if step["success"] else "❌"
        print(f"  {status} {step['step']}: {step['duration_seconds']:.1f}s")

    if result.get("comparison_summary", {}).get("speedup"):
        print(f"\n  投机推理加速比: {result['comparison_summary']['speedup']}x")
    if result.get("error"):
        print(f"\n  ❌ 错误: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
