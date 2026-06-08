"""
t1_2_needle_haystack.py
Needle-in-a-haystack evaluation for BitNet YaRN 8K context extension.

Tests whether the model can retrieve a specific fact inserted at various
positions across various document lengths. Produces a heatmap comparison
between the original model and the fine-tuned model.

Run:
    python t1_2_needle_haystack.py

Outputs:
    /bitnet_output/benchmark_results/needle_results.json
    /bitnet_output/benchmark_results/plots/needle_heatmap_original.png
    /bitnet_output/benchmark_results/plots/needle_heatmap_finetuned.png
    /bitnet_output/benchmark_results/plots/needle_heatmap_comparison.png
"""

import json, time, re
from pathlib import Path

import torch
import numpy as np
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

# Test matrix
DOC_LENGTHS      = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
NEEDLE_POSITIONS = [0.10, 0.25, 0.50, 0.75, 0.90]  # fraction of doc length

# Needle config — change PASSWORD to make it unique
NEEDLE_TEMPLATE  = "The secret authentication code is {password}."
QUESTION         = "What is the secret authentication code?"
PASSWORDS        = [
    "ALPHA-7734", "BETA-2291", "GAMMA-5518", "DELTA-8847",
    "EPSILON-3362", "ZETA-9901", "ETA-4456", "THETA-6623",
]

# Filler text — long enough to fill 8K tokens when repeated
FILLER = """
The development of large language models has transformed how we think about
artificial intelligence and its applications in the modern world. These systems
are trained on vast corpora of text data and learn to predict the next token
given a context window of preceding tokens. The attention mechanism allows
models to selectively focus on relevant parts of the input sequence regardless
of distance. Researchers have proposed numerous improvements to the original
transformer architecture including sparse attention rotary position embeddings
and mixture of experts routing. The scaling laws for neural language models
suggest that performance improves predictably with increases in model size
training data and compute budget. Fine-tuning pretrained models on downstream
tasks allows practitioners to adapt general capabilities to specific domains
with relatively small amounts of labeled data. The emergence of instruction
following abilities in sufficiently large models has enabled new interaction
paradigms where users can specify tasks in natural language without explicit
programming. Context length remains a fundamental constraint in transformer
architectures due to the quadratic scaling of attention with sequence length
though many techniques have been developed to extend effective context windows.
""" * 200  # repeat to ensure enough tokens

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(path):
    log(f"Loading model: {path}")
    model = BitNetForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.config.use_cache = True   # use cache for generation
    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
    model.eval()
    log(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Build test document
# ─────────────────────────────────────────────────────────────────────────────

def build_document(tokenizer, target_tokens, needle_text, needle_position_frac):
    """
    Build a document of exactly target_tokens tokens with needle_text
    inserted at needle_position_frac of the way through.
    Returns the full prompt string ready for tokenization.
    """
    filler_ids = tokenizer(FILLER, add_special_tokens=False)["input_ids"]

    # Calculate insert position in tokens
    insert_pos = int(target_tokens * needle_position_frac)
    insert_pos = max(50, min(insert_pos, target_tokens - 50))

    # Build: filler[:insert_pos] + needle + filler[insert_pos:target_tokens]
    before_ids = filler_ids[:insert_pos]
    after_ids  = filler_ids[insert_pos : target_tokens]

    before_text = tokenizer.decode(before_ids, skip_special_tokens=True)
    after_text  = tokenizer.decode(after_ids,  skip_special_tokens=True)

    document = before_text + " " + needle_text + " " + after_text
    prompt   = document + f"\n\nQuestion: {QUESTION}\nAnswer:"

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# Generate answer
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_answer(model, tokenizer, prompt, max_new_tokens=30):
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).input_ids.to(DEVICE)

    with torch.autocast(device_type="cuda", dtype=DTYPE):
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer


def score_answer(answer, password):
    """1 if password appears in answer, 0 otherwise."""
    return 1 if password.lower() in answer.lower() else 0


# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation for one model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, model_name):
    log(f"\n{'='*60}")
    log(f"Evaluating: {model_name}")
    log(f"{'='*60}")

    # Results matrix: doc_lengths × needle_positions
    scores  = np.zeros((len(DOC_LENGTHS), len(NEEDLE_POSITIONS)))
    details = []

    total = len(DOC_LENGTHS) * len(NEEDLE_POSITIONS)
    done  = 0

    for i, doc_len in enumerate(DOC_LENGTHS):
        for j, pos_frac in enumerate(NEEDLE_POSITIONS):
            # Use a different password for each test to avoid memorisation
            password = PASSWORDS[(i * len(NEEDLE_POSITIONS) + j) % len(PASSWORDS)]
            needle   = NEEDLE_TEMPLATE.format(password=password)

            prompt   = build_document(tokenizer, doc_len, needle, pos_frac)
            # Verify actual token count
            actual_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

            answer = get_answer(model, tokenizer, prompt)
            s      = score_answer(answer, password)
            scores[i, j] = s

            done += 1
            status = "✓" if s == 1 else "✗"
            log(f"  [{done:>2}/{total}] doc={doc_len:>4}tok pos={pos_frac:.0%} "
                f"actual={actual_tokens:>4}tok | {status} | "
                f"password={password} | answer='{answer[:40]}'")

            details.append({
                "doc_length": doc_len,
                "needle_position": pos_frac,
                "actual_tokens": actual_tokens,
                "password": password,
                "answer": answer,
                "score": s,
            })

    overall = scores.mean()
    log(f"\n{model_name} overall accuracy: {overall:.1%}")
    log(f"Scores by doc length:")
    for i, dl in enumerate(DOC_LENGTHS):
        row_score = scores[i].mean()
        bar = "█" * int(row_score * 10)
        log(f"  {dl:>4} tokens: {row_score:.1%}  {bar}")

    return scores, details, overall


# ─────────────────────────────────────────────────────────────────────────────
# Plot heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(scores, title, save_path, vmin=0, vmax=1):
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(scores, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(NEEDLE_POSITIONS)))
    ax.set_xticklabels([f"{p:.0%}" for p in NEEDLE_POSITIONS])
    ax.set_yticks(range(len(DOC_LENGTHS)))
    ax.set_yticklabels([f"{d:,}" for d in DOC_LENGTHS])

    ax.set_xlabel("Needle Position (% through document)")
    ax.set_ylabel("Document Length (tokens)")
    ax.set_title(title)

    # Add score text in each cell
    for i in range(len(DOC_LENGTHS)):
        for j in range(len(NEEDLE_POSITIONS)):
            val = scores[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Retrieval Accuracy")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved heatmap: {save_path}")


def plot_comparison(scores_orig, scores_ft, save_path):
    """Side-by-side heatmap comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, scores, title in zip(
        axes,
        [scores_orig, scores_ft],
        ["Original (4K limit)", "Fine-tuned YaRN (8K)"]
    ):
        im = ax.imshow(scores, cmap="RdYlGn", vmin=0, vmax=1,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(NEEDLE_POSITIONS)))
        ax.set_xticklabels([f"{p:.0%}" for p in NEEDLE_POSITIONS])
        ax.set_yticks(range(len(DOC_LENGTHS)))
        ax.set_yticklabels([f"{d:,}" for d in DOC_LENGTHS])
        ax.set_xlabel("Needle Position (% through document)")
        ax.set_ylabel("Document Length (tokens)")
        ax.set_title(title)
        ax.axhline(y=3.5, color="yellow", linewidth=2, linestyle="--",
                   label="4K boundary")

        for i in range(len(DOC_LENGTHS)):
            for j in range(len(NEEDLE_POSITIONS)):
                val = scores[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Retrieval Accuracy")

    plt.suptitle("Needle-in-a-Haystack: Original vs YaRN Fine-tuned BitNet b1.58",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved comparison: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("T1.2 Needle-in-a-Haystack Benchmark")
    log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ── Original model ────────────────────────────────────────────────────────
    model_orig = load_model(ORIGINAL_MODEL)
    scores_orig, details_orig, acc_orig = evaluate_model(
        model_orig, tokenizer, "Original"
    )
    results["original"] = {
        "accuracy": float(acc_orig),
        "scores_matrix": scores_orig.tolist(),
        "details": details_orig,
    }
    plot_heatmap(
        scores_orig,
        f"Original BitNet b1.58 — Needle Retrieval (overall {acc_orig:.1%})",
        PLOTS_DIR / "needle_heatmap_original.png"
    )
    del model_orig
    torch.cuda.empty_cache()

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    model_ft = load_model(FINETUNED_MODEL)
    scores_ft, details_ft, acc_ft = evaluate_model(
        model_ft, tokenizer, "Fine-tuned YaRN"
    )
    results["finetuned"] = {
        "accuracy": float(acc_ft),
        "scores_matrix": scores_ft.tolist(),
        "details": details_ft,
    }
    plot_heatmap(
        scores_ft,
        f"Fine-tuned YaRN BitNet b1.58 — Needle Retrieval (overall {acc_ft:.1%})",
        PLOTS_DIR / "needle_heatmap_finetuned.png"
    )

    # ── Comparison plot ────────────────────────────────────────────────────────
    plot_comparison(scores_orig, scores_ft, PLOTS_DIR / "needle_heatmap_comparison.png")

    # ── Save results ──────────────────────────────────────────────────────────
    results["summary"] = {
        "original_accuracy":  float(acc_orig),
        "finetuned_accuracy": float(acc_ft),
        "improvement":        float(acc_ft - acc_orig),
        "doc_lengths":        DOC_LENGTHS,
        "needle_positions":   NEEDLE_POSITIONS,
    }
    out_path = OUTPUT_DIR / "needle_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Verdict ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("NEEDLE-IN-A-HAYSTACK RESULTS")
    log(f"  Original  accuracy: {acc_orig:.1%}")
    log(f"  Fine-tuned accuracy: {acc_ft:.1%}")
    log(f"  Improvement: {acc_ft - acc_orig:+.1%}")

    # Check 4K+ performance specifically
    orig_4k_plus  = scores_orig[4:].mean()   # doc lengths >= 5000
    ft_4k_plus    = scores_ft[4:].mean()
    log(f"  >4K docs — Original: {orig_4k_plus:.1%}  Fine-tuned: {ft_4k_plus:.1%}")

    if acc_ft > acc_orig + 0.1:
        verdict = "PASS — Fine-tuned model significantly better at needle retrieval"
    elif acc_ft > acc_orig:
        verdict = "PARTIAL — Fine-tuned model slightly better"
    else:
        verdict = "FAIL — No improvement in needle retrieval"
    log(f"  VERDICT: {verdict}")
    log("=" * 60)
    log(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
