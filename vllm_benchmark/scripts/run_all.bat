@echo off
REM run_all.bat — vLLM 特性性能实验一键运行脚本
REM ================================================
REM 逐个启动 vLLM 配置 → 运行 Benchmark → 分析结果
REM 由于 RTX 4050 6GB 显存限制，实验需串行执行

setlocal enabledelayedexpansion

set PYTHON=%CONDA_PREFIX%\python.exe
set BENCHMARK_ARGS=--num-prompts 50 --max-tokens 128
set OUTPUT_BASE=results

echo ================================================================
echo   vLLM 特性性能实验 — 一键运行
echo ================================================================
echo.
echo   注意: 每次运行一个配置，总共约 7 组实验
echo   预计总耗时: ~30-45 分钟 (取决于模型下载速度)
echo.

REM ── 实验 1: 基线 ──
echo [1/7] 基线配置...
start "vLLM-Baseline" %PYTHON% vllm_server.py --config configs/baseline.json --port 8000
echo   等待 vLLM 启动 (约30秒)...
timeout /t 30 /nobreak >nul
%PYTHON% benchmark.py --config configs/baseline.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
%PYTHON% analyze.py --results-dir %OUTPUT_BASE%/baseline --output %OUTPUT_BASE%/baseline_report.md
taskkill /FI "WINDOWTITLE eq vLLM-Baseline*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 2: 前缀缓存 ──
echo [2/7] 前缀缓存...
start "vLLM-Prefix" %PYTHON% vllm_server.py --config configs/prefix_caching.json --port 8000
timeout /t 30 /nobreak >nul
%PYTHON% benchmark.py --config configs/prefix_caching.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
%PYTHON% analyze.py --results-dir %OUTPUT_BASE%/prefix_caching --output %OUTPUT_BASE%/prefix_caching_report.md
taskkill /FI "WINDOWTITLE eq vLLM-Prefix*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 3: 分块预填充 ──
echo [3/7] 分块预填充...
start "vLLM-Chunked" %PYTHON% vllm_server.py --config configs/chunked_prefill.json --port 8000
timeout /t 30 /nobreak >nul
%PYTHON% benchmark.py --config configs/chunked_prefill.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
%PYTHON% analyze.py --results-dir %OUTPUT_BASE%/chunked_prefill --output %OUTPUT_BASE%/chunked_prefill_report.md
taskkill /FI "WINDOWTITLE eq vLLM-Chunked*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 4: 最大并发序列数 (低) ──
echo [4/7] 并发度 (低, 8)...
start "vLLM-SeqLow" %PYTHON% vllm_server.py --config configs/max_seqs_low.json --port 8000
timeout /t 30 /nobreak >nul
%PYTHON% benchmark.py --config configs/max_seqs_low.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
taskkill /FI "WINDOWTITLE eq vLLM-SeqLow*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 5: 最大并发序列数 (高) ──
echo [5/7] 并发度 (高, 64)...
start "vLLM-SeqHigh" %PYTHON% vllm_server.py --config configs/max_seqs_high.json --port 8000
timeout /t 30 /nobreak >nul
%PYTHON% benchmark.py --config configs/max_seqs_high.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
taskkill /FI "WINDOWTITLE eq vLLM-SeqHigh*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 6: 投机推理 ⭐ ──
echo [6/7] 投机推理 (核心实验)...
start "vLLM-Speculative" %PYTHON% vllm_server.py --config configs/speculative.json --port 8000
echo   加载两个模型，等待较长时间 (约60秒)...
timeout /t 60 /nobreak >nul
%PYTHON% benchmark.py --config configs/speculative.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
%PYTHON% analyze.py --results-dir %OUTPUT_BASE%/speculative --output %OUTPUT_BASE%/speculative_report.md
taskkill /FI "WINDOWTITLE eq vLLM-Speculative*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 实验 7: 全特性组合 ──
echo [7/7] 全特性组合...
start "vLLM-AllFeatures" %PYTHON% vllm_server.py --config configs/all_features.json --port 8000
echo   加载所有特性，等待较长时间 (约60秒)...
timeout /t 60 /nobreak >nul
%PYTHON% benchmark.py --config configs/all_features.json %BENCHMARK_ARGS% --api-base http://127.0.0.1:8000
%PYTHON% analyze.py --results-dir %OUTPUT_BASE%/all_features --output %OUTPUT_BASE%/all_features_report.md
taskkill /FI "WINDOWTITLE eq vLLM-AllFeatures*" /F 2>nul
timeout /t 5 /nobreak >nul
echo.

REM ── 生成总对比报告 ──
echo ================================================================
echo   生成总对比报告...
echo ================================================================
%PYTHON% analyze.py --compare-dirs ^
    %OUTPUT_BASE%/baseline ^
    %OUTPUT_BASE%/prefix_caching ^
    %OUTPUT_BASE%/chunked_prefill ^
    %OUTPUT_BASE%/max_seqs_low ^
    %OUTPUT_BASE%/max_seqs_high ^
    %OUTPUT_BASE%/speculative ^
    %OUTPUT_BASE%/all_features ^
    --output %OUTPUT_BASE%/comparison_report.md

echo.
echo ================================================================
echo   全部实验完成!
echo   总对比报告: %OUTPUT_BASE%/comparison_report.md
echo ================================================================

endlocal
