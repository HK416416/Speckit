"""
benchmark.py — vLLM Benchmark 客户端
======================================
向 vLLM 服务发送请求，采集 TTFT、TPOT、吞吐量等性能指标。

用法:
  python benchmark.py --config configs/baseline.json --num-prompts 50
  python benchmark.py --config configs/speculative.json --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests


# ──────────────────────────────────────────────────────────────────────
# 测试 Prompt 加载
# ──────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS = [
    # 短 prompt (10-50 tokens)
    "请用一句话介绍人工智能。",
    "什么是深度学习？请简要回答。",
    "请解释机器学习的监督学习和无监督学习。",
    "Python 中的列表和元组有什么区别？",
    "请用三句话描述太阳系。",
    # 中 prompt (50-150 tokens)
    "请详细说明 Transformer 模型中的自注意力机制是如何工作的，以及它相比 RNN 的优势。",
    "请比较 CPU 和 GPU 在深度学习推理中的优劣，至少包含计算模式、内存带宽和延迟三个方面。",
    "请解释什么是梯度下降算法，以及随机梯度下降与批量梯度下降的区别，并说明各自的适用场景。",
    "请描述云计算和边缘计算的主要区别，并给出各自适合的应用场景。",
    "请解释数据库中的 ACID 特性，并说明为什么它们对事务处理很重要。",
    # 长 prompt (150+ tokens)
    "假设你是一名计算机科学家，请向非技术人员解释大语言模型的工作原理，包括训练阶段（预训练、监督微调、RLHF）和推理阶段（自回归解码、KV Cache）。请用通俗易懂的语言，避免过多专业术语。",
    "请从性能、成本和部署难度三个维度，对比分析云计算和边缘计算在 AI 推理场景下的适用性。请给出具体的应用场景和建议。",
]


def load_prompts(prompts_file: Optional[str] = None) -> List[str]:
    """加载测试 Prompt 列表"""
    if prompts_file and os.path.exists(prompts_file):
        prompts = []
        with open(prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    prompts.append(data.get("text", ""))
        return prompts
    return list(DEFAULT_PROMPTS)


# ──────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    """单次请求的性能数据"""
    request_id: str = ""
    prompt: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0         # Time to First Token (ms)
    tpot_ms: float = 0.0         # Time per Output Token (ms)
    total_time_ms: float = 0.0   # 端到端延迟 (ms)
    tokens_per_second: float = 0.0
    generated_text: str = ""
    error: Optional[str] = None


@dataclass
class ExperimentReport:
    """单次实验的汇总报告"""
    experiment_name: str = ""
    config: Dict = field(default_factory=dict)
    results: List[RequestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def num_success(self) -> int:
        return sum(1 for r in self.results if r.error is None)

    @property
    def avg_ttft_ms(self) -> float:
        vals = [r.ttft_ms for r in self.results if r.error is None and r.ttft_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_tpot_ms(self) -> float:
        vals = [r.tpot_ms for r in self.results if r.error is None and r.tpot_ms > 0]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def avg_throughput_tps(self) -> float:
        vals = [r.tokens_per_second for r in self.results if r.error is None]
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def p50_ttft_ms(self) -> float:
        vals = sorted([r.ttft_ms for r in self.results if r.error is None and r.ttft_ms > 0])
        return vals[len(vals) // 2] if vals else 0.0

    @property
    def p95_ttft_ms(self) -> float:
        vals = sorted([r.ttft_ms for r in self.results if r.error is None and r.ttft_ms > 0])
        return vals[int(len(vals) * 0.95)] if vals else 0.0


# ──────────────────────────────────────────────────────────────────────
# Benchmark 核心
# ──────────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """vLLM Benchmark 运行器"""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        max_tokens: int = 128,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens

    def send_request(self, prompt: str, temperature: float = 0.0) -> RequestResult:
        """
        发送单个 completion 请求并计时。

        计时策略:
          - TTFT: 从请求发出到收到第一个 token 的时间
          - TPOT: 首 token 之后的平均每 token 时间
          - 使用 stream=True 逐个接收 token 实现精确计时
        """
        result = RequestResult(
            request_id=str(uuid.uuid4())[:8],
            prompt=prompt[:100],
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        start_time = time.perf_counter()
        first_token_time = 0.0
        last_token_time = 0.0
        token_count = 0
        generated_text = ""

        try:
            response = requests.post(
                f"{self.api_base}/v1/completions",
                json=payload,
                stream=True,
                timeout=120,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices and choices[0].get("text"):
                            token_time = time.perf_counter()
                            if token_count == 0:
                                first_token_time = token_time
                            last_token_time = token_time
                            token_count += 1
                            generated_text += choices[0]["text"]
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            result.error = str(e)
            return result
        except Exception as e:
            result.error = str(e)
            return result

        total_time = time.perf_counter()

        # 计算结果
        result.ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time > 0 else 0.0
        result.total_time_ms = (total_time - start_time) * 1000.0

        if token_count > 0:
            result.tpot_ms = (last_token_time - first_token_time) * 1000.0 / token_count if token_count > 1 else 0.0
            result.completion_tokens = token_count
            result.generated_text = generated_text[:200]
            result.tokens_per_second = token_count / (result.total_time_ms / 1000.0) if result.total_time_ms > 0 else 0.0

        return result

    def run(
        self,
        prompts: List[str],
        experiment_name: str = "unknown",
        config: Dict = None,
        repeat: int = 1,
    ) -> ExperimentReport:
        """在多个 prompt 上运行 Benchmark"""
        report = ExperimentReport(
            experiment_name=experiment_name,
            config=config or {},
        )

        all_prompts = prompts * repeat

        print(f"\n  发送 {len(all_prompts)} 个请求 (repeat={repeat})...")
        print(f"  {'ID':>8s} | {'Tokens':>6s} | {'TTFT(ms)':>10s} | {'TPOT(ms)':>10s} | {'Total(s)':>8s} | {'tok/s':>7s}")
        print("  " + "-" * 65)

        for i, prompt in enumerate(all_prompts):
            result = self.send_request(prompt)
            report.results.append(result)

            if result.error:
                report.errors.append(f"[{i}] {result.error}")
                print(f"  [{i:3d}]     ERROR: {result.error[:50]}")
            else:
                print(
                    f"  {result.request_id:>8s} | {result.completion_tokens:>6d} | "
                    f"{result.ttft_ms:>10.1f} | {result.tpot_ms:>10.1f} | "
                    f"{result.total_time_ms/1000:>8.2f} | {result.tokens_per_second:>7.1f}"
                )

        return report


# ──────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="vLLM Benchmark 客户端 — 发送请求并采集性能指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="实验配置文件 (JSON)")
    parser.add_argument("--prompts-file", type=str, help="自定义 Prompt 文件 (JSONL)")
    parser.add_argument("--num-prompts", type=int, default=50, help="测试 prompt 数量 (默认50)")
    parser.add_argument("--concurrency", type=int, default=1, help="并发请求数 (默认1)")
    parser.add_argument("--max-tokens", type=int, default=128, help="每个请求最大生成 token 数")
    parser.add_argument("--repeat", type=int, default=1, help="重复轮次 (默认1)")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:8000", help="vLLM API 地址")
    parser.add_argument("--output-dir", type=str, help="结果输出目录 (默认自动生成)")
    args = parser.parse_args()

    # 加载配置
    config = {}
    experiment_name = "unknown"
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        experiment_name = config.get("experiment_name", os.path.splitext(os.path.basename(args.config))[0])

    # 输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("results", experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 加载 prompts
    prompts = load_prompts(args.prompts_file)
    if len(prompts) > args.num_prompts:
        prompts = prompts[:args.num_prompts]
    elif len(prompts) < args.num_prompts:
        # 循环重复 prompt 以达到目标数量
        repeats_needed = (args.num_prompts + len(prompts) - 1) // len(prompts)
        prompts = (prompts * repeats_needed)[:args.num_prompts]

    print("=" * 60)
    print("  vLLM Benchmark")
    print("=" * 60)
    print(f"  实验名称:  {experiment_name}")
    print(f"  API 地址:  {args.api_base}")
    print(f"  Prompt 数: {len(prompts)}")
    print(f"  并发数:    {args.concurrency}")
    print(f"  输出目录:  {output_dir}")
    print("=" * 60)

    # 预热: 发送一个简单请求确保服务已就绪
    model_name = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    print("\n  预热检查...")
    runner = BenchmarkRunner(api_base=args.api_base, model=model_name, max_tokens=args.max_tokens)

    for attempt in range(5):
        warmup = runner.send_request("Hello, how are you?")
        if warmup.error is None:
            print(f"  ✅ 服务就绪 (warmup TTFT: {warmup.ttft_ms:.0f}ms)")
            break
        print(f"  ⏳ 等待服务就绪 (尝试 {attempt+1}/5)...")
        time.sleep(5)
    else:
        print("  ❌ 服务未就绪，请确认 vLLM 已启动")
        sys.exit(1)

    # 运行基准测试
    report = runner.run(
        prompts=prompts,
        experiment_name=experiment_name,
        config=config,
        repeat=args.repeat,
    )

    # 保存详细结果
    detail_path = os.path.join(output_dir, "results.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_name": experiment_name,
            "config": config,
            "summary": {
                "num_requests": len(prompts),
                "num_success": report.num_success,
                "num_errors": len(report.errors),
                "avg_ttft_ms": round(report.avg_ttft_ms, 2),
                "avg_tpot_ms": round(report.avg_tpot_ms, 2),
                "p50_ttft_ms": round(report.p50_ttft_ms, 2),
                "p95_ttft_ms": round(report.p95_ttft_ms, 2),
                "avg_throughput_tps": round(report.avg_throughput_tps, 2),
            },
            "per_request": [
                {
                    "id": r.request_id,
                    "ttft_ms": round(r.ttft_ms, 2),
                    "tpot_ms": round(r.tpot_ms, 2),
                    "total_ms": round(r.total_time_ms, 2),
                    "completion_tokens": r.completion_tokens,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "error": r.error,
                }
                for r in report.results
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n  详细结果已保存: {detail_path}")

    # 打印汇总
    print("\n  " + "=" * 50)
    print(f"  实验汇总: {experiment_name}")
    print("  " + "=" * 50)
    print(f"  成功请求数:     {report.num_success}/{len(prompts)}")
    print(f"  平均 TTFT:      {report.avg_ttft_ms:.1f} ms")
    print(f"  P50 TTFT:       {report.p50_ttft_ms:.1f} ms")
    print(f"  P95 TTFT:       {report.p95_ttft_ms:.1f} ms")
    print(f"  平均 TPOT:      {report.avg_tpot_ms:.1f} ms")
    print(f"  平均吞吐量:     {report.avg_throughput_tps:.1f} tok/s")

    if report.errors:
        print(f"\n  ⚠ 错误数: {len(report.errors)}")
        for err in report.errors[:5]:
            print(f"    - {err}")


if __name__ == "__main__":
    main()
