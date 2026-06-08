"""
run_all_benchmarks.py
Master script — runs all benchmarks in the correct order.

Run:
    python run_all_benchmarks.py

Runs in this order (safest to most expensive):
    1. t2_2  — Sub-norm analysis (no model needed, just CSVs, ~30 seconds)
    2. t2_4  — Throughput benchmark (~10 minutes)
    3. t1_4  — Context boundary PPL (~15 minutes)
    4. t1_1  — WikiText-103 PPL curve (~30 minutes)
    5. t1_2  — Needle-in-a-haystack (~60 minutes)
    6. t1_3  — Short-context regression with lm-eval (~30 minutes)

Total estimated time: ~2.5 hours on RTX 6000 Ada.

Final outputs zipped to: /bitnet_output/all_benchmark_results.zip
"""

import subprocess, sys, time, json
from pathlib import Path

OUTPUT_DIR = Path("/bitnet_output/benchmark_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("T2.2 Sub-norm Analysis",          "t2_2_subnorm_analysis.py"),
    ("T2.4 Throughput",                 "t2_4_inference_throughput.py"),
    ("T1.4 Context Boundary",           "t1_4_context_boundary.py"),
    ("T1.1 WikiText PPL Curve",         "t1_1_wikitext_ppl.py"),
    ("T1.2 Needle in a Haystack",       "t1_2_needle_haystack.py"),
    ("T1.3 Short-Context Regression",   "t1_3_short_context_regression.py"),
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def run_script(name, script):
    log(f"\n{'='*65}")
    log(f"STARTING: {name}")
    log(f"{'='*65}")
    t0 = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,  # let output stream live
    )

    elapsed = (time.time() - t0) / 60
    if result.returncode == 0:
        log(f"COMPLETED: {name} in {elapsed:.1f} min ✓")
        return True
    else:
        log(f"FAILED: {name} after {elapsed:.1f} min ✗  (returncode={result.returncode})")
        return False


def main():
    log("=" * 65)
    log("BitNet YaRN 8K — Full Benchmark Suite")
    log(f"Scripts to run: {len(SCRIPTS)}")
    log("=" * 65)

    results_summary = {}
    t_total_start = time.time()

    for name, script in SCRIPTS:
        if not Path(script).exists():
            log(f"SKIP: {script} not found")
            results_summary[name] = "skipped"
            continue

        success = run_script(name, script)
        results_summary[name] = "passed" if success else "failed"

    total_time = (time.time() - t_total_start) / 60

    log("\n" + "=" * 65)
    log("BENCHMARK SUITE COMPLETE")
    log(f"Total time: {total_time:.1f} minutes")
    log("-" * 65)
    for name, status in results_summary.items():
        icon = "✓" if status == "passed" else ("⚠" if status == "skipped" else "✗")
        log(f"  {icon}  {name}: {status}")

    # Zip all results
    log("\nZipping all results...")
    import shutil
    zip_path = "/bitnet_output/all_benchmark_results"
    shutil.make_archive(zip_path, "zip", "/bitnet_output", "benchmark_results")
    log(f"Results zipped: {zip_path}.zip")

    # Save summary
    with open(OUTPUT_DIR / "benchmark_summary.json", "w") as f:
        json.dump({
            "total_time_min": total_time,
            "results": results_summary,
        }, f, indent=2)

    log("=" * 65)
    log("Download: python -m http.server 8080")
    log("Then open http://YOUR_IP:8080 and download all_benchmark_results.zip")
    log("=" * 65)


if __name__ == "__main__":
    main()
