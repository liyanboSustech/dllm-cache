"""Single-sample LLaDA correctness/smoke run; timings are NOT clean benchmarks."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModel, AutoTokenizer
from accelerate import init_empty_weights
from vendor.sparse_llada.configuration_llada import LLaDAConfig
from vendor.sparse_llada.modeling_llada import CustomCache, LLaDAModelLM
from vendor.sparse_llada.llada_generate import get_num_transfer_tokens, generate as upstream_generate

PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs "
    "6 kilometers per hour. How many kilometers can she run in 8 hours? "
    "Show your calculation and end with 'Final answer: <number>'."
)
RATIOS = {"D": 1.0, "C100": 1.0, "S50": 0.5, "S25": 0.25}


def emit(event, **kwargs):
    print(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                      "event": event, **kwargs}, ensure_ascii=False), flush=True)


def gpu_snapshot():
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True).strip()
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "gpus": query(["--query-gpu=index,uuid,name,memory.used,utilization.gpu"]),
            "processes": query(["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"])}


def wait_idle(index, seconds, out):
    records = []
    start = time.monotonic()
    while True:
        snap = gpu_snapshot()
        records.append(snap)
        rows = [r.split(", ") for r in snap["gpus"].splitlines()]
        row = next(r for r in rows if int(r[0]) == index)
        uuid = row[1]
        idle = int(row[3]) <= 1000 and int(row[4]) <= 5 and uuid not in snap["processes"]
        (out / "idle_check.json").write_text(json.dumps(records, indent=2))
        if not idle:
            raise RuntimeError(f"GPU {index} is occupied; no other process will be interrupted")
        elapsed = time.monotonic() - start
        emit("idle_check", gpu=index, elapsed_seconds=round(elapsed, 1))
        if elapsed >= seconds:
            return uuid
        time.sleep(min(10, seconds - elapsed))


class Cache100(CustomCache):
    """Keep all external positions in original order, without scoring/top-k."""
    def filter_cache(self, layer_id, q_block, cur_filtered_len, block_len):
        cached = self.get_cache(layer_id)
        self.cache[layer_id] = {
            name: torch.cat((value[:, :, :cur_filtered_len],
                             value[:, :, cur_filtered_len + block_len:]), dim=2)
            for name, value in cached.items()
        }


@torch.inference_mode()
def generate(model, prompt, mode, gen_length, steps, block_length=32, original=False):
    assert prompt.shape[0] == 1
    assert gen_length % block_length == 0
    blocks = gen_length // block_length
    assert steps % blocks == 0
    local_steps = steps // blocks
    assert local_steps >= 3
    mask_id = model.config.mask_token_id
    plen = prompt.shape[1]
    x = torch.full((1, plen + gen_length), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :plen] = prompt
    finite = torch.ones((), dtype=torch.bool, device=prompt.device)
    stats = {"full_forwards": 0, "block_forwards": 0, "anchor_forwards": 0,
             "max_external_cache_bytes": 0, "anchor_external_lengths": []}
    for block in range(blocks):
        start = plen + block * block_length
        end = start + block_length
        cache_cls = Cache100 if mode == "C100" else CustomCache
        cache = cache_cls(model.config.n_layers, prompt.device, kernel_size=3, keep_ratio=RATIOS[mode])
        transfer = get_num_transfer_tokens(x[:, start:end] == mask_id, local_steps)
        for step in range(local_steps):
            state = 0 if mode == "D" else min(step, 2)
            inp = x if state != 2 else x[:, start:end]
            masked = inp == mask_id
            logits = (model(inp).logits if original else model(inp, start, state, cache).logits)
            # Smoke-only instrumentation, intentionally included in these timings.
            finite &= torch.isfinite(logits).all()
            predicted = logits.argmax(dim=-1)
            confidence = F.softmax(logits, dim=-1).gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            if state != 2:
                confidence[:, end:] = -torch.inf
            predicted = torch.where(masked, predicted, inp)
            confidence = torch.where(masked, confidence, -torch.inf)
            selected = torch.zeros_like(predicted, dtype=torch.bool)
            _, indices = torch.topk(confidence[0], k=transfer[0, step])
            selected[0, indices] = True
            if state == 2:
                x[:, start:end][selected] = predicted[selected]
                stats["block_forwards"] += 1
            else:
                x[selected] = predicted[selected]
                stats["full_forwards"] += 1
            if state == 1:
                stats["anchor_forwards"] += 1
                expected = int((plen + gen_length - block_length) * RATIOS[mode])
                assert len(cache.cache) == model.config.n_layers
                assert all(v["k"].shape[-2] == expected for v in cache.cache.values())
                stats["anchor_external_lengths"].append(expected)
                nbytes = sum(t.numel() * t.element_size() for v in cache.cache.values() for t in v.values())
                stats["max_external_cache_bytes"] = max(stats["max_external_cache_bytes"], nbytes)
            del logits, confidence, predicted, selected
        cache.clear()
    return x, finite, stats


def describe_output(tokenizer, ids):
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    stops = [i for i, token in enumerate(ids) if token in stop_ids]
    stop = min(stops) if stops else len(ids)
    answer = tokenizer.decode(ids[:stop], skip_special_tokens=True)
    matches = re.findall(r"Final answer:\s*\**\s*([-+]?\d+(?:\.\d+)?)", answer, flags=re.I)
    parsed = matches[-1] if matches else None
    return {"raw_output": tokenizer.decode(ids, skip_special_tokens=False),
            "answer_before_stop": answer, "first_stop_position": stops[0] if stops else None,
            "no_stop_within_budget": not stops, "parsed_answer": parsed,
            "expected_answer": "72", "correct": parsed is not None and float(parsed) == 72.0,
            "remaining_mask_tokens": ids.count(126336), "generated_token_ids": ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/dataset/money-shredder-group/tv/LLaDA-8B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "command.json").write_text(json.dumps({"argv": sys.argv, "cwd": os.getcwd()}, indent=2))
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.gpu)
    uuid = wait_idle(args.gpu, args.idle_seconds, args.output)
    torch.set_num_threads(4)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.get_device_capability(0) == (7, 0)
    backend_checks = {}
    for name, backend in [("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION), ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                for qlen, klen in [(320, 320), (32, 320), (32, 104)]:
                    q = torch.randn(1, 32, qlen, 128, device="cuda", dtype=torch.float16)
                    k = torch.randn(1, 32, klen, 128, device="cuda", dtype=torch.float16)
                    y = F.scaled_dot_product_attention(q, k, k, is_causal=False)
                    assert torch.isfinite(y).all().item()
            backend_checks[name] = "passed"
        except Exception as exc:
            backend_checks[name] = repr(exc)
    del q, k, y
    backend_name = "EFFICIENT_ATTENTION" if backend_checks["EFFICIENT_ATTENTION"] == "passed" else "MATH"
    assert backend_checks[backend_name] == "passed"
    backend = getattr(SDPBackend, backend_name)
    env = {"python": sys.version, "python_executable": sys.executable,
           "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate", "safetensors", "einops"]},
           "torch_file": torch.__file__, "cuda_build": torch.version.cuda,
           "compiled_archs": torch.cuda.get_arch_list(), "gpu_uuid": uuid,
           "dtype": "float16", "attention_backend": backend_name, "backend_checks": backend_checks,
           "model_path": args.model, "model_revision_from_download_metadata": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
           "source": json.loads((ROOT / "vendor/sparse_llada/UPSTREAM.json").read_text()),
           "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "protocol": {"gen_length": args.gen_length, "steps": args.steps, "block_length": 32,
                        "temperature": 0, "cfg_scale": 0, "seed": 2025, "early_stop": False},
           "timing_note": "One sample, one repetition, finite-logit checks enabled; smoke timings only."}
    (args.output / "environment.json").write_text(json.dumps(env, indent=2))
    emit("backend_ready", **backend_checks)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    assert tokenizer.convert_tokens_to_ids("<|mdm_mask|>") == 126336
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True)
    prompt = torch.tensor([tokenizer(rendered)["input_ids"]], device="cuda")
    (args.output / "sample.json").write_text(json.dumps({"id": "smoke_lily_72", "source": "adapted upstream demonstration; not a benchmark dataset", "prompt": PROMPT, "rendered_prompt": rendered,
        "prompt_token_ids": prompt[0].tolist(), "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(), "expected_answer": "72"}, indent=2))
    emit("loading_model", model=args.model, prompt_length=prompt.shape[1])
    cfg = LLaDAConfig.from_pretrained(args.model, local_files_only=True)
    cfg.block_len = 32
    cfg.flash_attention = False
    cfg.use_cache = False
    model, loading = LLaDAModelLM.from_pretrained(args.model, config=cfg, torch_dtype=torch.float16,
        device_map={"": "cuda:0"}, local_files_only=True, output_loading_info=True)
    model.eval()
    (args.output / "loading_info.json").write_text(json.dumps(loading, indent=2))
    assert not any(loading[k] for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
    emit("model_loaded", weight_bytes=sum(p.numel() * p.element_size() for p in model.parameters()))
    checks = {}
    # Both upstream constructors change global SDPA flags. Construct everything
    # BEFORE entering the common backend context, which must be authoritative.
    ref_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    with init_empty_weights():
        reference = AutoModel.from_config(ref_cfg, trust_remote_code=True)
    reference.load_state_dict(model.state_dict(), strict=True, assign=True)
    reference.eval()
    assert all(p.device.type == "cuda" for p in reference.parameters())
    with torch.inference_mode(), sdpa_kernel(backend):
        # Instantiate the original model on meta, sharing the SAME loaded tensors.
        # This validates the dense path without loading a second weight copy.
        checks["backend_flags"] = {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        }
        assert checks["backend_flags"]["efficient"] == (backend_name == "EFFICIENT_ATTENTION")
        assert checks["backend_flags"]["math"] == (backend_name == "MATH")
        assert not checks["backend_flags"]["flash"] and not checks["backend_flags"]["cudnn"]
        emit("checking_original_dense")
        original_out, original_finite, _ = generate(reference, prompt, "D", 64, 64, original=True)
        dense_out, dense_finite, _ = generate(model, prompt, "D", 64, 64)
        checks["dense_matches_original_64_tokens"] = torch.equal(original_out, dense_out)
        checks["original_and_dense_finite"] = bool((original_finite & dense_finite).item())
        assert checks["dense_matches_original_64_tokens"] and checks["original_and_dense_finite"]
        del reference, original_out, dense_out
        for mode in ["S50", "S25"]:
            model.config.keep_ratio = RATIOS[mode]
            actual, finite, _ = generate(model, prompt, mode, 64, 64)
            expected = upstream_generate(model, prompt, steps=64, gen_length=64, block_length=32, temperature=0, cfg_scale=0)
            checks[f"{mode}_matches_upstream_64_tokens"] = torch.equal(actual, expected)
            assert checks[f"{mode}_matches_upstream_64_tokens"] and finite.item()
            del actual, expected
        (args.output / "validation.json").write_text(json.dumps(checks, indent=2))
        emit("implementation_checks_passed", **checks)
        results = []
        for mode in RATIOS:
            model.config.keep_ratio = RATIOS[mode]
            emit("running", mode=mode)
            warm, finite, _ = generate(model, prompt, mode, 32, 32)
            assert finite.item()
            del warm, finite
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = gpu_snapshot()
            started = time.perf_counter()
            output, finite, stats = generate(model, prompt, mode, args.gen_length, args.steps)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            after = gpu_snapshot()
            ids = output[0, prompt.shape[1]:].tolist()
            record = {"sample_id": "smoke_lily_72", "mode": mode, "keep_ratio": RATIOS[mode],
                "gpu_uuid": uuid, "dtype": "float16", "attention_backend": backend_name,
                "prompt_length": prompt.shape[1], **env["protocol"],
                "generation_seconds": elapsed, "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": reserved, "all_logits_finite": bool(finite.item()),
                **stats, **describe_output(tokenizer, ids), "gpu_before": before, "gpu_after": after}
            results.append(record)
            with (args.output / "results.jsonl").open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            emit("result", **{k: record[k] for k in ["mode", "generation_seconds", "peak_allocated_bytes", "all_logits_finite", "correct", "answer_before_stop", "remaining_mask_tokens"]})
            assert record["all_logits_finite"] and record["remaining_mask_tokens"] == 0
            assert stats["full_forwards"] == (args.steps if mode == "D" else args.gen_length // 32 * 2)
            del output, finite
        summary = {"status": "passed", "scope": "single-sample smoke test, not benchmark accuracy or stable speedup", "validation": checks,
            "results": [{k: r[k] for k in ["mode", "correct", "generation_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "answer_before_stop", "all_logits_finite", "full_forwards", "block_forwards", "max_external_cache_bytes"]} for r in results]}
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        emit("completed", output=str(args.output))


if __name__ == "__main__":
    main()
