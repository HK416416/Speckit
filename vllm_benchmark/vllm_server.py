"""
vllm_server.py — vLLM 服务启动器
=================================
根据配置文件启动 vLLM OpenAI-compatible API 服务，支持各种特性组合。

用法:
  python vllm_server.py --config configs/baseline.json
  python vllm_server.py --config configs/speculative.json --port 8001
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict


def load_config(config_path: str) -> Dict:
    """加载实验配置 JSON 文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def build_vllm_command(config: Dict, host: str, port: int) -> list:
    """
    根据配置构建 vLLM 启动命令行。

    配置格式（JSON）:
    {
      "experiment_name": "baseline",
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "vllm_args": {
        "max-model-len": 2048,
        "gpu-memory-utilization": 0.90,
        "enable-prefix-caching": false,
        "enable-chunked-prefill": false,
        "max-num-seqs": 256,
        "speculative-config": {                        // 可选: 投机推理配置
          "model": "Qwen/Qwen2.5-0.5B-Instruct",
          "num_speculative_tokens": 5,
          "method": "draft_model"
        }
      }
    }
    """
    vllm_args = config.get("vllm_args", {})
    model = config.get("model", "Qwen/Qwen2.5-1.5B-Instruct")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
    ]

    # 基础参数
    if "max-model-len" in vllm_args:
        cmd.extend(["--max-model-len", str(vllm_args["max-model-len"])])
    if "gpu-memory-utilization" in vllm_args:
        cmd.extend(["--gpu-memory-utilization", str(vllm_args["gpu-memory-utilization"])])
    if "max-num-seqs" in vllm_args:
        cmd.extend(["--max-num-seqs", str(vllm_args["max-num-seqs"])])

    # 执行模式（enforce-eager 关闭 CUDA Graph，对 RTX 4050 6GB 至关重要）
    if vllm_args.get("enforce-eager"):
        cmd.append("--enforce-eager")
    if "compilation-config" in vllm_args:
        cmd.extend(["--compilation-config", json.dumps(vllm_args["compilation-config"])])

    # 特性开关
    if vllm_args.get("enable-prefix-caching"):
        cmd.append("--enable-prefix-caching")
    if vllm_args.get("enable-chunked-prefill"):
        cmd.append("--enable-chunked-prefill")

    # 投机推理配置 ⭐
    speculative_config = vllm_args.get("speculative-config")
    if speculative_config:
        cmd.extend(["--speculative-config", json.dumps(speculative_config)])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="vLLM 服务启动器 — 根据配置文件启动实验服务",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python vllm_server.py --config configs/baseline.json
  python vllm_server.py --config configs/speculative.json --port 8001
        """,
    )
    parser.add_argument("--config", type=str, required=True, help="实验配置文件 (JSON)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config.get("experiment_name", "unknown")
    model = config.get("model", "unknown")

    print("=" * 60)
    print("  vLLM 实验服务启动")
    print("=" * 60)
    print(f"  实验名称: {experiment_name}")
    print(f"  模型:      {model}")
    print(f"  地址:      {args.host}:{args.port}")
    print(f"  配置:      {json.dumps(config.get('vllm_args', {}), indent=2)}")
    print(f"  配置文件:  {args.config}")
    print("=" * 60)

    cmd = build_vllm_command(config, args.host, args.port)

    print(f"\n  启动命令: {' '.join(cmd)}")
    print()

    # 启动 vLLM 服务（前台运行，Ctrl+C 停止）
    try:
        env = os.environ.copy(); env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0"); env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0"); nvcc_path = os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib", "python3.10", "site-packages", "nvidia", "cu13"); env.setdefault("CUDA_HOME", nvcc_path); env["PATH"] = nvcc_path + "/bin:" + env.get("PATH", ""); subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        print("\n  vLLM 服务已停止.")


if __name__ == "__main__":
    main()
