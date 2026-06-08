"""
t2_4_inference_throughput.py
Inference throughput benchmark for BitNet YaRN 8K context extension.

Measures tokens/second and VRAM usage at multiple context lengths
for both the original and fine-tuned model.

Run:
    python t2_4_inference_throughput.py

Outputs:
    /bitnet_output/benchmark_results/throughput_results.json
    /bitnet_output/benchmark_results/plots/throughput.png
"""

import json, time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import BitNetForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_MODEL  = "microsoft/bitnet-b1.58-2B-4T-bf16"
FINETUNED_MODEL = "/bitnet_output/bitnet_yarn_output/final_model"
OUTPUT_DIR      = Path("/bitnet_output/benchmark_results")
PLOTS_DIR       = OUTPUT_DIR / "plots"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 6144, 8192]
NEW_TOKENS      = 100   # tokens to generate per run
N_WARMUP        = 2     # warmup runs (not counted)
N_RUNS          = 5     # measured runs per context length

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path):
    log(f"Loading: {path}")
    model = BitNetForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, device_map="auto", low_cpu_mem_usage=True
    )
    model = model.to(DEVICE)
    model.config.use_cache = True   # use KV cache for generation speed
    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark one context length
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_ctx(model, tokenizer, ctx_len, n_new=NEW_TOKENS):
    # Build a prompt of exactly ctx_len tokens
    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx_len),
                               dtype=torch.long).to(DEVICE)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for i in range(N_WARMUP + N_RUNS):
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = model.generate(
                prompt_ids,
                max_new_tokens=n_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    avg_time   = sum(times) / len(times)
    tokens_sec = n_new / avg_time

    return {
        "ctx_len":    ctx_len,
        "avg_time_s": avg_time,
        "tokens_sec": tokens_sec,
        "peak_vram_gb": peak_vram,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, model_name):
    log(f"\n{'='*60}")
    log(f"Throughput benchmark: {model_name}")
    log(f"Generating {NEW_TOKENS} tokens per run, {N_RUNS} runs each")

    results = {}
    for ctx_len in CONTEXT_LENGTHS:
        r = benchmark_ctx(model, tokenizer, ctx_len)
        results[ctx_len] = r
        log(f"  ctx={ctx_len:>5} | {r['tokens_sec']:>6.1f} tok/s | "
            f"vram={r['peak_vram_gb']:.1f}GB | "
            f"time={r['avg_time_s']:.2f}s")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_throughput(results_orig, results_ft, save_path):
    ctx       = CONTEXT_LENGTHS
    tps_orig  = [results_orig[c]["tokens_sec"]   for c in ctx]
    tps_ft    = [results_ft[c]["tokens_sec"]     for c in ctx]
    vram_orig = [results_orig[c]["peak_vram_gb"] for c in ctx]
    vram_ft   = [results_ft[c]["peak_vram_gb"]   for c in ctx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput
    ax1.plot(ctx, tps_orig, "r-o", linewidth=2, markersize=7,
             label="Original")
    ax1.plot(ctx, tps_ft,   "b-o", linewidth=2, markersize=7,
             label="Fine-tuned YaRN")
    ax1.axvline(x=4096, color="gray", linestyle="--", alpha=0.7,
                label="4K boundary")
    ax1.set_xlabel("Context Length (tokens)", fontsize=12)
    ax1.set_ylabel("Tokens / Second", fontsize=12)
    ax1.set_title("Inference Throughput", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # VRAM
    ax2.plot(ctx, vram_orig, "r-o", linewidth=2, markersize=7,
             label="Original")
    ax2.plot(ctx, vram_ft,   "b-o", linewidth=2, markersize=7,
             label="Fine-tuned YaRN")
    ax2.axvline(x=4096, color="gray", linestyle="--", alpha=0.7,
                label="4K boundary")
    ax2.set_xlabel("Context Length (tokens)", fontsize=12)
    ax2.set_ylabel("Peak VRAM (GB)", fontsize=12)
    ax2.set_title("Peak VRAM Usage", fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("BitNet b1.58 Inference Throughput — Original vs YaRN Fine-tuned",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("T2.4 Inference Throughput Benchmark")
    log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_orig = load_model(ORIGINAL_MODEL)
    results_orig = evaluate_model(model_orig, tokenizer, "Original")
    del model_orig
    torch.cuda.empty_cache()

    model_ft = load_model(FINETUNED_MODEL)
    results_ft = evaluate_model(model_ft, tokenizer, "Fine-tuned YaRN")
    del model_ft
    torch.cuda.empty_cache()

    plot_throughput(results_orig, results_ft, PLOTS_DIR / "throughput.png")

    log("\n" + "=" * 60)
    log("THROUGHPUT RESULTS")
    log(f"{'CTX':>6} | {'ORIG tok/s':>12} | {'FT tok/s':>12} | "
        f"{'ORIG VRAM':>10} | {'FT VRAM':>10}")
    log("-" * 60)
    for ctx in CONTEXT_LENGTHS:
        o = results_orig[ctx]
        f = results_ft[ctx]
        marker = " ← ext" if ctx > 4096 else ""
        log(f"{ctx:>6} | {o['tokens_sec']:>11.1f} | {f['tokens_sec']:>11.1f} | "
            f"{o['peak_vram_gb']:>9.1f}G | {f['peak_vram_gb']:>9.1f}G{marker}")

    # Overhead of extension
    overhead = ((results_ft[8192]["avg_time_s"] /
                 results_ft[4096]["avg_time_s"]) - 1) * 100
    log(f"\nGeneration overhead at 8K vs 4K: {overhead:+.1f}%")

    all_results = {
        "original": results_orig,
        "finetuned": results_ft,
        "summary": {
            "overhead_8k_vs_4k_pct": overhead,
            "ft_throughput_8k": results_ft[8192]["tokens_sec"],
            "ft_vram_8k_gb": results_ft[8192]["peak_vram_gb"],
        }
    }
    out_path = OUTPUT_DIR / "throughput_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved: {out_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
