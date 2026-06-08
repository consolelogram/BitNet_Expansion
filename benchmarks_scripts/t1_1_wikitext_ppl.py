"""
t1_1_wikitext_ppl.py
WikiText-103 perplexity curve for BitNet YaRN 8K context extension.

Measures PPL at multiple context lengths on both the original and fine-tuned
model. Produces a plot with a vertical line at 4096 (original training limit).
This is the standard evaluation used by all context extension papers.

Run:
    python t1_1_wikitext_ppl.py

Outputs:
    /bitnet_output/benchmark_results/wikitext_ppl_results.json
    /bitnet_output/benchmark_results/plots/wikitext_ppl_curve.png
"""

import json, math, time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
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

CONTEXT_LENGTHS = [512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]
STRIDE          = 512    # sliding window stride
MAX_TOKENS      = 100_000  # how many tokens of WikiText to use for eval
ORIGINAL_LIMIT  = 4096   # vertical line on plot

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path):
    log(f"Loading model: {path}")
    model = BitNetForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.config.use_cache = False
    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Load WikiText-103
# ─────────────────────────────────────────────────────────────────────────────

def load_wikitext(tokenizer):
    log("Loading WikiText-103 test set...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Concatenate all text
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    ids  = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids  = ids[:MAX_TOKENS]
    log(f"WikiText-103: {len(ids):,} tokens loaded")
    return torch.tensor(ids, dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# Sliding window PPL
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl_sliding(model, token_ids, context_len, stride=None):
    """
    Sliding window perplexity evaluation.
    For each window of context_len tokens, compute loss only on the
    tokens not seen in the previous window (stride tokens from end).
    This is the standard method for evaluating long-context models.
    """
    if stride is None:
        stride = min(STRIDE, context_len // 2)

    seq_len    = len(token_ids)
    total_loss = 0.0
    total_toks = 0

    prev_end = 0
    for begin in range(0, seq_len - context_len + 1, stride):
        end      = begin + context_len
        ids      = token_ids[begin:end].unsqueeze(0).to(DEVICE)

        # Only compute loss on tokens from prev_end onward (new tokens in window)
        target_len = end - max(begin, prev_end)
        if target_len <= 0:
            prev_end = end
            continue

        labels = ids.clone()
        # Mask out tokens we've already counted
        mask_len = context_len - target_len
        if mask_len > 0:
            labels[:, :mask_len] = -100

        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = model(ids, labels=labels)

        # out.loss is mean over non-masked tokens
        n_valid = (labels != -100).sum().item() - 1  # -1 for shift
        if n_valid > 0:
            total_loss += out.loss.item() * n_valid
            total_toks += n_valid

        prev_end = end
        if end >= seq_len:
            break

    if total_toks == 0:
        return float("inf"), float("inf")

    avg_loss = total_loss / total_toks
    return math.exp(avg_loss), avg_loss

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one model across all context lengths
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, token_ids, model_name):
    log(f"\n{'='*60}")
    log(f"Evaluating: {model_name}")
    log(f"{'='*60}")

    results = {}
    for ctx_len in CONTEXT_LENGTHS:
        # Skip lengths beyond model's config if it's the original
        t0   = time.time()
        ppl, loss = compute_ppl_sliding(model, token_ids, ctx_len)
        elapsed = time.time() - t0

        results[ctx_len] = {"ppl": ppl, "loss": loss}
        marker = " ← BEYOND ORIGINAL LIMIT" if ctx_len > ORIGINAL_LIMIT else ""
        log(f"  ctx={ctx_len:>5} | PPL={ppl:>8.3f} | loss={loss:.4f} | "
            f"{elapsed:.1f}s{marker}")

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_ppl_curve(results_orig, results_ft, save_path):
    ctx_lens  = CONTEXT_LENGTHS
    ppl_orig  = [results_orig[c]["ppl"] for c in ctx_lens]
    ppl_ft    = [results_ft[c]["ppl"]   for c in ctx_lens]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(ctx_lens, ppl_orig, "r-o", linewidth=2, markersize=7,
            label="Original BitNet b1.58 (4K limit)")
    ax.plot(ctx_lens, ppl_ft,   "b-o", linewidth=2, markersize=7,
            label="Fine-tuned YaRN BitNet b1.58 (8K)")

    ax.axvline(x=ORIGINAL_LIMIT, color="gray", linestyle="--",
               linewidth=1.5, label=f"Original training limit ({ORIGINAL_LIMIT:,} tokens)")

    ax.fill_betweenx(
        [min(ppl_orig + ppl_ft) * 0.98, max(ppl_orig + ppl_ft) * 1.02],
        ORIGINAL_LIMIT, max(ctx_lens),
        alpha=0.05, color="blue", label="Extended context region"
    )

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity (WikiText-103)", fontsize=12)
    ax.set_title("Perplexity vs Context Length — BitNet b1.58 YaRN 4K→8K", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xticks(ctx_lens)
    ax.set_xticklabels([str(c) for c in ctx_lens], rotation=30)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved PPL curve: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("T1.1 WikiText-103 Perplexity Curve")
    log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = load_wikitext(tokenizer)
    all_results = {}

    # ── Original model ─────────────────────────────────────────────────────
    model_orig = load_model(ORIGINAL_MODEL)
    results_orig = evaluate_model(model_orig, token_ids, "Original")
    all_results["original"] = results_orig
    del model_orig
    torch.cuda.empty_cache()

    # ── Fine-tuned model ───────────────────────────────────────────────────
    model_ft = load_model(FINETUNED_MODEL)
    results_ft = evaluate_model(model_ft, token_ids, "Fine-tuned YaRN")
    all_results["finetuned"] = results_ft
    del model_ft
    torch.cuda.empty_cache()

    # ── Plot ───────────────────────────────────────────────────────────────
    plot_ppl_curve(results_orig, results_ft, PLOTS_DIR / "wikitext_ppl_curve.png")

    # ── Summary ────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("WIKITEXT-103 PPL RESULTS")
    log(f"{'CTX':>6} | {'ORIGINAL':>10} | {'FINETUNED':>10} | {'DELTA':>8}")
    log("-" * 45)
    for ctx in CONTEXT_LENGTHS:
        po   = results_orig[ctx]["ppl"]
        pf   = results_ft[ctx]["ppl"]
        marker = " ← ext" if ctx > ORIGINAL_LIMIT else ""
        log(f"{ctx:>6} | {po:>10.3f} | {pf:>10.3f} | {pf-po:>+8.3f}{marker}")

    # Key metrics for paper
    ppl_orig_4k = results_orig[4096]["ppl"]
    ppl_ft_4k   = results_ft[4096]["ppl"]
    ppl_orig_8k = results_orig[8192]["ppl"]
    ppl_ft_8k   = results_ft[8192]["ppl"]

    log(f"\n4K: {ppl_orig_4k:.3f} → {ppl_ft_4k:.3f}  (Δ {ppl_ft_4k-ppl_orig_4k:+.3f})")
    log(f"8K: {ppl_orig_8k:.3f} → {ppl_ft_8k:.3f}  (Δ {ppl_ft_8k-ppl_orig_8k:+.3f})")

    if ppl_ft_8k < ppl_orig_8k and ppl_ft_4k <= ppl_orig_4k + 0.5:
        verdict = "PASS — 8K PPL improved, 4K PPL preserved"
    elif ppl_ft_8k < ppl_orig_8k:
        verdict = "PARTIAL — 8K improved but 4K degraded"
    else:
        verdict = "FAIL — No PPL improvement at 8K"
    log(f"VERDICT: {verdict}")
    log("=" * 60)

    # Save
    all_results["summary"] = {
        "ppl_original_4k":  ppl_orig_4k,
        "ppl_finetuned_4k": ppl_ft_4k,
        "ppl_original_8k":  ppl_orig_8k,
        "ppl_finetuned_8k": ppl_ft_8k,
        "delta_4k": ppl_ft_4k - ppl_orig_4k,
        "delta_8k": ppl_ft_8k - ppl_orig_8k,
        "verdict": verdict,
    }
    out_path = OUTPUT_DIR / "wikitext_ppl_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
