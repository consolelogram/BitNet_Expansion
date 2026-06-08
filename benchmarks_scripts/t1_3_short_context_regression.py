"""
t1_3_short_context_regression.py
Short-context regression test for BitNet YaRN 8K context extension.

Tests HellaSwag and ARC-Easy on both models to confirm fine-tuning
did not degrade short-context reasoning abilities.

Run:
    pip install lm-eval
    python t1_3_short_context_regression.py

Outputs:
    /bitnet_output/benchmark_results/short_context_regression_results.json
    /bitnet_output/benchmark_results/plots/short_context_regression.png
"""

import json, time, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_MODEL  = "microsoft/bitnet-b1.58-2B-4T-bf16"
FINETUNED_MODEL = "/bitnet_output/bitnet_yarn_output/final_model"
OUTPUT_DIR      = Path("/bitnet_output/benchmark_results")
PLOTS_DIR       = OUTPUT_DIR / "plots"

TASKS       = ["hellaswag", "arc_easy"]
NUM_FEWSHOT = 0
LIMIT       = 200   # number of examples per task — set to None for full eval

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Install lm-eval if needed
# ─────────────────────────────────────────────────────────────────────────────

def ensure_lm_eval():
    try:
        import lm_eval
        log("lm-eval already installed")
    except ImportError:
        log("Installing lm-eval...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "lm-eval"],
            check=True
        )
        log("lm-eval installed")

# ─────────────────────────────────────────────────────────────────────────────
# Run lm-eval for one model
# ─────────────────────────────────────────────────────────────────────────────

def run_lm_eval(model_path, model_name):
    log(f"\nRunning lm-eval on: {model_name}")
    log(f"  Tasks: {TASKS}  Few-shot: {NUM_FEWSHOT}  Limit: {LIMIT}")

    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    # Build model args string
    model_args = f"pretrained={model_path},dtype=bfloat16"

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=TASKS,
        num_fewshot=NUM_FEWSHOT,
        limit=LIMIT,
        device="cuda",
        batch_size=1,
    )

    extracted = {}
    for task in TASKS:
        task_results = results["results"].get(task, {})
        # lm-eval uses acc or acc_norm depending on task
        acc = task_results.get("acc_norm,none",
              task_results.get("acc,none",
              task_results.get("acc_norm",
              task_results.get("acc", None))))
        if acc is not None:
            extracted[task] = float(acc)
            log(f"  {task}: {acc:.4f} ({acc*100:.1f}%)")
        else:
            log(f"  {task}: could not extract accuracy. Keys: {list(task_results.keys())}")
            extracted[task] = None

    return extracted, results

# ─────────────────────────────────────────────────────────────────────────────
# Plot comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results_orig, results_ft, save_path):
    tasks       = [t for t in TASKS if results_orig.get(t) and results_ft.get(t)]
    orig_scores = [results_orig[t] * 100 for t in tasks]
    ft_scores   = [results_ft[t]   * 100 for t in tasks]

    x     = range(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar([xi - width/2 for xi in x], orig_scores, width,
                   label="Original BitNet b1.58", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar([xi + width/2 for xi in x], ft_scores,   width,
                   label="Fine-tuned YaRN",       color="#3498db", alpha=0.85)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", " ").title() for t in tasks], fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Short-Context Regression Test\nOriginal vs YaRN Fine-tuned BitNet b1.58",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    log(f"Saved plot: {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("T1.3 Short-Context Regression Test")
    log("=" * 60)

    ensure_lm_eval()

    all_results = {}

    # ── Original ──────────────────────────────────────────────────────────────
    scores_orig, raw_orig = run_lm_eval(ORIGINAL_MODEL, "Original")
    all_results["original"] = {"scores": scores_orig}

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    scores_ft, raw_ft = run_lm_eval(FINETUNED_MODEL, "Fine-tuned YaRN")
    all_results["finetuned"] = {"scores": scores_ft}

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_comparison(scores_orig, scores_ft,
                    PLOTS_DIR / "short_context_regression.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    log("\n" + "=" * 60)
    log("SHORT CONTEXT REGRESSION RESULTS")
    log(f"{'Task':<15} | {'Original':>10} | {'Fine-tuned':>10} | {'Delta':>8}")
    log("-" * 50)

    any_degraded = False
    for task in TASKS:
        so = scores_orig.get(task)
        sf = scores_ft.get(task)
        if so is not None and sf is not None:
            delta = sf - so
            if delta < -0.02:   # >2% degradation
                any_degraded = True
            log(f"{task:<15} | {so*100:>9.1f}% | {sf*100:>9.1f}% | {delta*100:>+7.1f}%")
        else:
            log(f"{task:<15} | {'N/A':>10} | {'N/A':>10} | {'N/A':>8}")

    if not any_degraded:
        verdict = "PASS — No significant degradation on short-context tasks"
    else:
        verdict = "PARTIAL — Some degradation detected, check individual tasks"
    log(f"\nVERDICT: {verdict}")
    log("=" * 60)

    all_results["summary"] = {
        "verdict": verdict,
        "scores_original":  scores_orig,
        "scores_finetuned": scores_ft,
    }

    out_path = OUTPUT_DIR / "short_context_regression_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
