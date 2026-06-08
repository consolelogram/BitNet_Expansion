"""
t1_4_context_boundary.py
PPL degradation at the 4096 token boundary.

Measures perplexity at fine-grained context lengths around the 4096 boundary.
The original model should show a sharp cliff at 4097+.
The fine-tuned model should show smooth continuation.

Run:
    python t1_4_context_boundary.py

Outputs:
    /bitnet_output/benchmark_results/boundary_results.json
    /bitnet_output/benchmark_results/plots/context_boundary.png
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

# Fine-grained around the 4096 boundary
CONTEXT_LENGTHS = [
    3584, 3840, 3968, 4032, 4064, 4080, 4096,   # approaching limit
    4097, 4112, 4128, 4160, 4224, 4352, 4608,   # just beyond limit
    5120, 6144, 7168, 8192                        # extended range
]
N_SAMPLES   = 10   # number of documents to average over
ORIGINAL_LIMIT = 4096

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
    model.config.use_cache = False
    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
    model.eval()
    return model

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_chunks(tokenizer, n_chunks=N_SAMPLES, chunk_size=8192):
    log("Loading WikiText-103...")
    dataset  = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text     = "\n\n".join([t for t in dataset["text"] if t.strip()])
    all_ids  = tokenizer(text, add_special_tokens=False)["input_ids"]

    chunks = []
    for i in range(0, len(all_ids) - chunk_size, chunk_size):
        chunks.append(torch.tensor(all_ids[i:i+chunk_size], dtype=torch.long))
        if len(chunks) >= n_chunks:
            break
    log(f"Loaded {len(chunks)} chunks of {chunk_size} tokens")
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# Compute PPL at exact length
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl_exact(model, chunks, ctx_len):
    """PPL using exactly the last ctx_len tokens of each chunk."""
    total_loss, total_toks = 0.0, 0
    for chunk in chunks:
        # Take the last ctx_len tokens (tests positions near ctx_len)
        if len(chunk) >= ctx_len:
            ids = chunk[:ctx_len].unsqueeze(0).to(DEVICE)
        else:
            continue

        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = model(ids, labels=ids)

        n_toks = ids.shape[1] - 1
        total_loss += out.loss.item() * n_toks
        total_toks += n_toks

    if total_toks == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_toks
    return math.exp(avg_loss), avg_loss

# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, chunks, model_name):
    log(f"\n{'='*60}")
    log(f"Evaluating: {model_name}")
    results = {}
    for ctx_len in CONTEXT_LENGTHS:
        ppl, loss = compute_ppl_exact(model, chunks, ctx_len)
        results[ctx_len] = {"ppl": ppl, "loss": loss}
        marker = " ← BEYOND LIMIT" if ctx_len > ORIGINAL_LIMIT else ""
        log(f"  ctx={ctx_len:>5} | PPL={ppl:>8.3f} | loss={loss:.4f}{marker}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_boundary(results_orig, results_ft, save_path):
    ctx   = CONTEXT_LENGTHS
    p_orig = [results_orig[c]["ppl"] for c in ctx]
    p_ft   = [results_ft[c]["ppl"]   for c in ctx]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(ctx, p_orig, "r-o", linewidth=2, markersize=6,
            label="Original BitNet b1.58 (4K limit)")
    ax.plot(ctx, p_ft,   "b-o", linewidth=2, markersize=6,
            label="Fine-tuned YaRN BitNet b1.58 (8K)")

    ax.axvline(x=ORIGINAL_LIMIT, color="gray", linestyle="--",
               linewidth=2, label=f"4096 training boundary")
    ax.axvspan(ORIGINAL_LIMIT, max(ctx), alpha=0.05, color="blue",
               label="Extended region")

    ax.set_xlabel("Context Length (tokens)", fontsize=12)
    ax.set_ylabel("Perplexity", fontsize=12)
    ax.set_title("PPL at the 4096-Token Boundary — BitNet b1.58 YaRN Extension",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Mark the cliff if it exists
    orig_at_limit = results_orig[4096]["ppl"]
    orig_beyond   = results_orig[4097]["ppl"] if 4097 in results_orig else None
    if orig_beyond and orig_beyond > orig_at_limit * 1.05:
        ax.annotate("PPL cliff →",
                    xy=(4097, orig_beyond),
                    xytext=(4200, orig_beyond + 0.5),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontsize=10)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("T1.4 Context Boundary PPL Test")
    log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chunks = load_chunks(tokenizer)

    model_orig = load_model(ORIGINAL_MODEL)
    results_orig = evaluate_model(model_orig, chunks, "Original")
    del model_orig
    torch.cuda.empty_cache()

    model_ft = load_model(FINETUNED_MODEL)
    results_ft = evaluate_model(model_ft, chunks, "Fine-tuned YaRN")
    del model_ft
    torch.cuda.empty_cache()

    plot_boundary(results_orig, results_ft, PLOTS_DIR / "context_boundary.png")

    # ── Key metric: cliff size at 4096 boundary ────────────────────────────
    log("\n" + "=" * 60)
    log("BOUNDARY ANALYSIS")
    ppl_orig_4096 = results_orig[4096]["ppl"]
    ppl_orig_4097 = results_orig.get(4097, {}).get("ppl", None)
    ppl_ft_4096   = results_ft[4096]["ppl"]
    ppl_ft_8192   = results_ft[8192]["ppl"]

    if ppl_orig_4097:
        cliff = ppl_orig_4097 - ppl_orig_4096
        log(f"Original PPL cliff at 4096→4097: {ppl_orig_4096:.3f} → {ppl_orig_4097:.3f}  (+{cliff:.3f})")
    log(f"Fine-tuned PPL at 4096: {ppl_ft_4096:.3f}")
    log(f"Fine-tuned PPL at 8192: {ppl_ft_8192:.3f}")
    log(f"Fine-tuned degradation 4096→8192: {ppl_ft_8192 - ppl_ft_4096:+.3f}")

    ft_smooth = (ppl_ft_8192 - ppl_ft_4096) < 1.0
    verdict = "PASS — Fine-tuned model smooth through boundary" if ft_smooth \
              else "PARTIAL — Some degradation at boundary"
    log(f"VERDICT: {verdict}")
    log("=" * 60)

    all_results = {
        "original": results_orig,
        "finetuned": results_ft,
        "summary": {
            "ppl_orig_4096": ppl_orig_4096,
            "ppl_ft_4096": ppl_ft_4096,
            "ppl_ft_8192": ppl_ft_8192,
            "ft_boundary_degradation": ppl_ft_8192 - ppl_ft_4096,
            "verdict": verdict,
        }
    }
    out_path = OUTPUT_DIR / "boundary_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
