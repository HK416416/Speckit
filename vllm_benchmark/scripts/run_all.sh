#!/bin/bash
# run_all.sh — vLLM 特性性能实验一键运行脚本（WSL2/Linux）
# ============================================================
# 逐个启动 vLLM 配置 → 运行 Benchmark → 分析结果
# 由于 RTX 4050 6GB 显存限制，实验需串行执行

set -euo pipefail

PYTHON="${CONDA_PREFIX:-/usr}/bin/python"
BENCHMARK_ARGS="--num-prompts 50 --max-tokens 128"
OUTPUT_BASE="results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "================================================================"
echo "  vLLM 特性性能实验 — 一键运行"
echo "================================================================"
echo ""
echo "  注意: 每次运行一个配置，总共约 14 组实验"
echo "  预计总耗时: ~60-90 分钟 (取决于模型下载速度)"
echo ""

cd "$PROJECT_DIR"

run_experiment() {
    local label="$1"
    local config="vllm_benchmark/configs/${2}.json"
    local prompts_flag="${3:-}"
    local wait_time="${4:-30}"

    echo "================================================================"
    echo "  [$label]"
    echo "================================================================"

    # 启动 vLLM 服务（后台）
    $PYTHON vllm_benchmark/vllm_server.py --config "$config" --port 8000 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID, 等待 ${wait_time}s 启动..."
    sleep "$wait_time"

    # 运行 benchmark（有 warmup 重试逻辑，5次尝试）
    $PYTHON vllm_benchmark/benchmark.py \
        --config "$config" \
        $BENCHMARK_ARGS \
        --api-base http://127.0.0.1:8000 \
        $prompts_flag

    # 停止 vLLM
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 5
    echo ""
}

# ── 实验 1: 主基线 ──
run_experiment "1/14 主基线" "baseline"

# ── 实验 2: 投机解码专属基线 ──
run_experiment "2/14 投机解码专属基线" "baseline_spec"

# ── 实验 3: 投机推理 (draft_model) ⭐ ──
run_experiment "3/14 投机推理 (draft_model)" "speculative" "" 60

# ── 实验 4: N-gram 投机解码 + 标准 Prompt ──
run_experiment "4/14 N-gram + 标准 Prompt" "speculative_ngram" \
    "--prompts-file vllm_benchmark/prompts/test_prompts.jsonl"

# ── 实验 5: N-gram + 简单 Prompt (代码补全) ──
run_experiment "5/14 N-gram + 代码补全" "speculative_ngram_simple" \
    "--prompts-file vllm_benchmark/prompts/simple_prompts.jsonl"

# ── 实验 6: draft_model + 简单 Prompt ──
run_experiment "6/14 draft_model + 代码补全" "speculative_simple" \
    "--prompts-file vllm_benchmark/prompts/simple_prompts.jsonl" 60

# ── 实验 7: N-gram + 标准答案 Prompt ──
run_experiment "7/14 N-gram + 标准答案" "speculative_ngram_correct" \
    "--prompts-file vllm_benchmark/prompts/correct_prompts.jsonl"

# ── 实验 8: draft_model + 标准答案 Prompt ──
run_experiment "8/14 draft_model + 标准答案" "speculative_correct" \
    "--prompts-file vllm_benchmark/prompts/correct_prompts.jsonl" 60

# ── 实验 9: Draft 独立测试 ──
run_experiment "9/14 Draft Model 独立测试" "draft_standalone"

# ── 实验 10: 前缀缓存 ──
run_experiment "10/14 前缀缓存" "prefix_caching"

# ── 实验 11: 分块预填充 ──
run_experiment "11/14 分块预填充" "chunked_prefill"

# ── 实验 12: 并发度 (低, 8) ──
run_experiment "12/14 并发度 seqs=8" "max_seqs_low"

# ── 实验 13: 并发度 (高, 64) ──
run_experiment "13/14 并发度 seqs=64" "max_seqs_high"

# ── 实验 14: 全特性组合 ──
run_experiment "14/14 全特性组合" "all_features" "" 60

# ── 生成总对比报告 ──
echo "================================================================"
echo "  生成总对比报告..."
echo "================================================================"
$PYTHON vllm_benchmark/analyze.py --compare-dirs \
    "$OUTPUT_BASE/baseline" \
    "$OUTPUT_BASE/baseline_spec" \
    "$OUTPUT_BASE/speculative" \
    "$OUTPUT_BASE/speculative_ngram" \
    "$OUTPUT_BASE/speculative_ngram_simple" \
    "$OUTPUT_BASE/speculative_simple" \
    "$OUTPUT_BASE/speculative_ngram_correct" \
    "$OUTPUT_BASE/speculative_correct" \
    "$OUTPUT_BASE/draft_standalone" \
    "$OUTPUT_BASE/prefix_caching" \
    "$OUTPUT_BASE/chunked_prefill" \
    "$OUTPUT_BASE/max_seqs_low" \
    "$OUTPUT_BASE/max_seqs_high" \
    "$OUTPUT_BASE/all_features" \
    --output "$OUTPUT_BASE/comparison_report.md"

echo ""
echo "================================================================"
echo "  全部实验完成!"
echo "  总对比报告: $OUTPUT_BASE/comparison_report.md"
echo "================================================================"
