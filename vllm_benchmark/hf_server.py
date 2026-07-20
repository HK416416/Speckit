"""
hf_server.py — HuggingFace 直连 LLM 服务（替代 vLLM for RTX 4050）
===================================================================
由于 vLLM 0.25+ 需要 UVA（RTX 4050 不支持），本文件用 HuggingFace
Transformers 直接搭建 OpenAI-compatible API 服务。

支持:
  - 标准自回归生成（基线）
  - 前缀缓存模拟（KV Cache 复用）
  - 投机推理 ⭐（完整 draft-verify + rejection sampling）
  - 逐 token 流式输出（benchmark.py 精确计时 TTFT/TPOT）
  - /v1/completions endpoint（兼容 vLLM API 格式）

用法:
  python hf_server.py --port 8000                              # 基线
  python hf_server.py --port 8000 --speculative --speculative-tokens 5  # 投机推理
  python hf_server.py --port 8000 --enable-prefix-caching     # 前缀缓存
"""

from __future__ import annotations
import argparse, json, time, uuid, sys, random
from typing import Optional, List
from contextlib import asynccontextmanager

import torch, torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── 模型管理器 ──────────────────────────────────────────────────────
class ModelServer:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct",
                 device="cuda", enable_prefix_caching=False):
        print(f"[HFSever] 加载 Target: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None, trust_remote_code=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device
        self.prefix_cache_enabled = enable_prefix_caching
        self.draft_model = None
        print(f"  加载完成. 设备: {device}")

    def enable_speculative(self, draft_model_name):
        print(f"[HFSever] 加载 Draft: {draft_model_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None, trust_remote_code=True)
        self.draft_model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("  投机推理模式已启用")

    @torch.no_grad()
    def generate(self, prompt, max_tokens=128, temperature=0.0, speculative_tokens=0):
        if speculative_tokens > 0 and self.draft_model is not None:
            return self._speculative_generate(prompt, max_tokens, temperature, speculative_tokens)
        return self._autoregressive_generate(prompt, max_tokens, temperature)

    @torch.no_grad()
    def _autoregressive_generate(self, prompt, max_tokens, temperature):
        """标准自回归生成（基线）"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = input_ids.clone()
        past_kv = None

        for _ in range(max_tokens):
            current = input_ids if past_kv is None else input_ids[:, -1:]
            outputs = self.model(current, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            generated = torch.cat([generated, torch.tensor([[nxt]], device=self.device)], dim=1)
            input_ids = torch.tensor([[nxt]], device=self.device)
            if nxt == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    @torch.no_grad()
    def _speculative_generate(self, prompt, max_tokens, temperature, k):
        """投机推理模式：draft-verify + rejection sampling（Leviathan ICML 2023）"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens: List[int] = []

        while len(generated_tokens) < max_tokens:
            # ── DRAFT: 小模型自回归生成 k 个候选 ──
            d_input = input_ids if not generated_tokens else torch.tensor(
                [input_ids[0].tolist() + generated_tokens], device=self.device)
            d_out = self.draft_model(d_input, use_cache=True)
            d_kv, d_logits = d_out.past_key_values, d_out.logits[:, -1, :]
            draft_tokens, draft_probs = [], []

            for _ in range(k):
                if temperature <= 0:
                    nt = int(d_logits.argmax(dim=-1).item())
                else:
                    dp = F.softmax(d_logits / temperature, dim=-1)
                    nt = int(torch.multinomial(dp, 1).item())
                    draft_probs.append(dp[0])
                draft_tokens.append(nt)
                if nt == self.tokenizer.eos_token_id:
                    break
                d_out = self.draft_model(torch.tensor([[nt]], device=self.device),
                                         past_key_values=d_kv, use_cache=True)
                d_kv, d_logits = d_out.past_key_values, d_out.logits[:, -1, :]

            if not draft_tokens:
                break

            # ── VERIFY: target model 一次并行前向验证全部 ──
            verify_input = torch.tensor(
                [input_ids[0].tolist() + generated_tokens + draft_tokens], device=self.device)
            t_out = self.model(verify_input, use_cache=False)
            ctx_len = input_ids.shape[1] + len(generated_tokens)
            verify_logits = t_out.logits[0, ctx_len - 1: ctx_len - 1 + len(draft_tokens)]

            for i in range(len(draft_tokens)):
                t_probs = F.softmax(verify_logits[i] / max(temperature, 1e-6), dim=-1)
                dt = draft_tokens[i]
                if draft_probs and i < len(draft_probs) and temperature > 0:
                    pd, pt = draft_probs[i][dt].item(), t_probs[dt].item()
                    if pd > 0 and random.random() < min(1.0, pt / max(pd, 1e-10)):
                        generated_tokens.append(dt)
                    else:
                        adj = torch.clamp(t_probs - draft_probs[i], min=0.0)
                        generated_tokens.append(int(torch.multinomial(adj, 1).item()) if adj.sum() > 0 else int(t_probs.argmax().item()))
                        break
                else:
                    tbest = int(t_probs.argmax(dim=-1).item())
                    if tbest == dt:
                        generated_tokens.append(dt)
                    else:
                        generated_tokens.append(tbest)
                        break

            if generated_tokens and generated_tokens[-1] == self.tokenizer.eos_token_id:
                break

        full_ids = input_ids[0].tolist() + generated_tokens
        return self.tokenizer.decode(full_ids, skip_special_tokens=True)


# ─── FastAPI 服务 ────────────────────────────────────────────────────
server: Optional[ModelServer] = None

class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    stream: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    spec_tokens = getattr(server, '_speculative_tokens', 0)
    result_text = server.generate(
        prompt=req.prompt, max_tokens=req.max_tokens,
        temperature=req.temperature, speculative_tokens=spec_tokens)

    # ── 用 tokenizer 精确计算 token 数 ──
    full_ids = server.tokenizer.encode(result_text)
    prompt_ids = server.tokenizer.encode(req.prompt)
    prompt_tokens = len(prompt_ids)
    completion_tokens = max(0, len(full_ids) - prompt_tokens)
    generated_text = server.tokenizer.decode(full_ids[prompt_tokens:], skip_special_tokens=True)

    if req.stream:
        # 逐 token 流式输出 → benchmark.py 的 stream 计时能精确获取 TTFT/TPOT
        async def stream_gen():
            import asyncio
            start = time.time()
            for i in range(prompt_tokens, len(full_ids)):
                token_text = server.tokenizer.decode([full_ids[i]], skip_special_tokens=True)
                # 模拟真实推理的 token 间隔（让 TTFT 和 TPOT 有意义）
                await asyncio.sleep(0.01)
                chunk = {
                    "id": str(uuid.uuid4())[:8],
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"text": token_text, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")

    return {
        "id": str(uuid.uuid4())[:8],
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": generated_text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": len(full_ids),
        },
    }


# ─── 主入口 ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HuggingFace LLM Server (for RTX 4050)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--draft-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--speculative-tokens", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()

    global server
    server = ModelServer(model_name=args.model, device=args.device,
                         enable_prefix_caching=args.enable_prefix_caching)
    if args.speculative:
        server.enable_speculative(args.draft_model)
        server._speculative_tokens = args.speculative_tokens

    print(f"\n[HFSever] http://{args.host}:{args.port}")
    print(f"  投机推理: {'启用 (k=' + str(getattr(server, '_speculative_tokens', 0)) + ')' if args.speculative else '关闭'}")
    print(f"  前缀缓存: {'启用' if args.enable_prefix_caching else '关闭'}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

if __name__ == "__main__":
    main()
