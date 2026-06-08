"""
setup_and_run_benchmarks.py
Run this on a fresh JarvisLabs instance after uploading:
  - All 6 benchmark scripts (t1_1, t1_2, t1_3, t1_4, t2_2, t2_4)
  - run_all_benchmarks.py
  - The final model folder (uploaded or pulled from HF/GitHub)

This script:
  1. Fixes all paths in benchmark scripts to point to correct locations
  2. Verifies the fine-tuned model exists and is loadable
  3. Runs all benchmarks

Run:
    python setup_and_run_benchmarks.py
"""

import os, sys, subprocess, time, json
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# EDIT THESE TWO PATHS IF NEEDED
# ─────────────────────────────────────────────────────────────────────────────
FINETUNED_MODEL_PATH = "/bitnet_output/bitnet_8k_sft/final_model"
SUBNORM_PROFILES_PATH = "/bitnet_output/bitnet_8k_sft/subnorm_profiles"
OUTPUT_DIR = Path("/bitnet_output/benchmark_results")
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_MODEL = "microsoft/bitnet-b1.58-2B-4T-bf16"

BENCHMARK_SCRIPTS = [
    "t1_1_wikitext_ppl.py",
    "t1_2_needle_haystack.py",
    "t1_3_short_context_regression.py",
    "t1_4_context_boundary.py",
    "t2_2_subnorm_analysis.py",
    "t2_4_inference_throughput.py",
    "run_all_benchmarks.py",
]

# All possible stale paths that might be in the scripts
STALE_FINETUNED_PATHS = [
    "/bitnet_output/bitnet_yarn_output/final_model",
    "/bitnet_output/bitnet_yarn_8k/final_model",
    "./bitnet_yarn_output/final_model",
]
STALE_SUBNORM_PATHS = [
    "/bitnet_output/bitnet_yarn_output/subnorm_profiles",
    "/bitnet_output/bitnet_yarn_8k/subnorm_profiles",
    "./bitnet_yarn_output/subnorm_profiles",
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Install dependencies
# ─────────────────────────────────────────────────────────────────────────────

def install_deps():
    log("Installing dependencies...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "transformers", "accelerate", "datasets",
        "safetensors", "matplotlib", "lm-eval"
    ], check=True)
    log("Dependencies installed ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fix paths in all benchmark scripts
# ─────────────────────────────────────────────────────────────────────────────

def fix_paths():
    log("Fixing paths in benchmark scripts...")
    fixed_count = 0

    for script_name in BENCHMARK_SCRIPTS:
        script = Path(script_name)
        if not script.exists():
            log(f"  SKIP (not found): {script_name}")
            continue

        content = script.read_text()
        original = content

        # Fix fine-tuned model path
        for stale in STALE_FINETUNED_PATHS:
            content = content.replace(stale, FINETUNED_MODEL_PATH)

        # Fix subnorm path
        for stale in STALE_SUBNORM_PATHS:
            content = content.replace(stale, SUBNORM_PROFILES_PATH)

        if content != original:
            script.write_text(content)
            log(f"  Fixed paths: {script_name}")
            fixed_count += 1
        else:
            log(f"  No changes: {script_name}")

    log(f"Path fixing complete: {fixed_count} files updated ✓")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Verify fine-tuned model is loadable
# ─────────────────────────────────────────────────────────────────────────────

def verify_model():
    log(f"Verifying fine-tuned model at: {FINETUNED_MODEL_PATH}")

    model_path = Path(FINETUNED_MODEL_PATH)
    if not model_path.exists():
        log(f"  ERROR: Path does not exist: {FINETUNED_MODEL_PATH}")
        log(f"  You need to upload the final_model folder to this path")
        log(f"  Or change FINETUNED_MODEL_PATH at the top of this script")
        return False

    # Check for required files
    required = ["config.json"]
    safetensors = list(model_path.glob("*.safetensors"))
    pytorch_bins = list(model_path.glob("*.bin"))

    missing = [f for f in required if not (model_path / f).exists()]
    if missing:
        log(f"  ERROR: Missing files: {missing}")
        return False

    if not safetensors and not pytorch_bins:
        log(f"  ERROR: No model weights found (no .safetensors or .bin files)")
        return False

    log(f"  config.json: found ✓")
    log(f"  weights: {len(safetensors)} safetensors, {len(pytorch_bins)} bin files ✓")

    # Quick load test
    log(f"  Running quick load test...")
    try:
        import torch
        from transformers import BitNetForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
        model = BitNetForCausalLM.from_pretrained(
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if hasattr(model, "set_use_kernels"):
            model.set_use_kernels(False)

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        sub_norm  = sum(1 for n, _ in model.named_parameters() if "sub_norm" in n)
        rope_ok   = hasattr(model.model, "rotary_emb")

        log(f"  Params: {n_params:.0f}M ✓")
        log(f"  Sub_norm tensors: {sub_norm} (expect 60) {'✓' if sub_norm==60 else '✗'}")
        log(f"  rotary_emb: {'found ✓' if rope_ok else 'NOT FOUND ✗'}")

        # Check max_position_embeddings — should be 8192 for fine-tuned
        max_pos = model.config.max_position_embeddings
        log(f"  max_position_embeddings: {max_pos} "
            f"({'✓ extended to 8192' if max_pos >= 8192 else '⚠ still at 4096 — YaRN patch needed'})")

        del model
        import gc; gc.collect()
        import torch; torch.cuda.empty_cache()
        log(f"  Load test passed ✓")
        return True

    except Exception as e:
        log(f"  Load test FAILED: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Verify subnorm profiles exist
# ─────────────────────────────────────────────────────────────────────────────

def verify_subnorm():
    log(f"Checking subnorm profiles at: {SUBNORM_PROFILES_PATH}")
    path = Path(SUBNORM_PROFILES_PATH)

    if not path.exists():
        log(f"  WARNING: subnorm_profiles not found — T2.2 will be skipped")
        log(f"  Upload subnorm_profiles/ folder if you have it")
        return False

    csvs = list(path.glob("*.csv"))
    txts = list(path.glob("*.txt"))
    log(f"  Found {len(csvs)} CSV files, {len(txts)} TXT files")

    if len(csvs) < 2:
        log(f"  WARNING: Need at least 2 snapshot files for T2.2 comparison")
        return False

    log(f"  Subnorm profiles OK ✓")
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Print benchmark summary before running
# ─────────────────────────────────────────────────────────────────────────────

def print_plan():
    log("\n" + "=" * 65)
    log("BENCHMARK PLAN")
    log(f"  Original model : {ORIGINAL_MODEL}")
    log(f"  Fine-tuned model: {FINETUNED_MODEL_PATH}")
    log(f"  Output dir     : {OUTPUT_DIR}")
    log("")
    log("  T2.2  Sub-norm analysis      ~1 min   (no GPU needed)")
    log("  T2.4  Throughput             ~15 min")
    log("  T1.4  Context boundary PPL   ~20 min")
    log("  T1.1  WikiText PPL curve     ~40 min")
    log("  T1.2  Needle in haystack     ~90 min")
    log("  T1.3  Short-context (lm-eval)~30 min")
    log("")
    log("  Total estimated: ~3.5 hours")
    log("=" * 65 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("=" * 65)
    log("BitNet YaRN 8K Benchmark Setup")
    log("=" * 65)

    # Install deps
    install_deps()

    # Fix all paths
    fix_paths()

    # Verify fine-tuned model
    model_ok = verify_model()
    if not model_ok:
        log("\nERROR: Fine-tuned model not found or not loadable.")
        log("Fix FINETUNED_MODEL_PATH at the top of this script and try again.")
        log(f"Current path: {FINETUNED_MODEL_PATH}")
        sys.exit(1)

    # Verify subnorm profiles
    verify_subnorm()

    # Print plan
    print_plan()

    # Create output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)

    # Run benchmarks
    if not Path("run_all_benchmarks.py").exists():
        log("ERROR: run_all_benchmarks.py not found in current directory")
        sys.exit(1)

    log("Starting benchmark suite...")
    subprocess.run([sys.executable, "run_all_benchmarks.py"], check=False)

    log("\nSetup and benchmarks complete.")
    log("Start HTTP server to download results:")
    log("  python -m http.server 8080")
    log(f"  Then download: {OUTPUT_DIR}/all_benchmark_results.zip")


if __name__ == "__main__":
    main()
