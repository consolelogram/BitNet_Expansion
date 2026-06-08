from pathlib import Path
import subprocess
import sys
import time

FILES = [
    "t1_1_wikitext_ppl.py",
    "t1_2_needle_haystack.py",
    "t1_3_short_context_regression.py",
    "t1_4_context_boundary.py",
    "t2_2_subnorm_analysis.py",
    "t2_4_inference_throughput.py",
    "setup_and_run_benchmarks.py",
    "run_all_benchmarks.py",
]

MODEL_PATH = "/home/bitnet_output/bitnet_8k_sft/final_model"
SUBNORM_PATH = "/home/bitnet_output/bitnet_8k_sft/subnorm_profiles"
BENCH_OUT = "/home/bitnet_output/benchmark_results"

REPLS = {
    "/bitnet_output/bitnet_yarn_output/final_model": MODEL_PATH,
    "/bitnet_output/bitnet_yarn_8k/final_model": MODEL_PATH,
    "./bitnet_yarn_output/final_model": MODEL_PATH,
    "/bitnet_output/bitnet_8k_sft/final_model": MODEL_PATH,
    "bitnet_output/bitnet_8k_sft/final_model": MODEL_PATH,

    "/bitnet_output/bitnet_yarn_output/subnorm_profiles": SUBNORM_PATH,
    "/bitnet_output/bitnet_yarn_8k/subnorm_profiles": SUBNORM_PATH,
    "./bitnet_yarn_output/subnorm_profiles": SUBNORM_PATH,
    "/bitnet_output/bitnet_8k_sft/subnorm_profiles": SUBNORM_PATH,
    "bitnet_output/bitnet_8k_sft/subnorm_profiles": SUBNORM_PATH,

    'Path("/bitnet_output/benchmark_results")': 'Path("/home/bitnet_output/benchmark_results")',
    'Path("bitnet_output/benchmark_results")': 'Path("/home/bitnet_output/benchmark_results")',
}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def patch_file(name: str):
    p = Path(name)
    if not p.exists():
        log(f"SKIP missing: {name}")
        return

    text = p.read_text()
    orig = text

    for a, b in REPLS.items():
        text = text.replace(a, b)

    if name in {
        "t1_1_wikitext_ppl.py",
        "t1_2_needle_haystack.py",
        "t1_4_context_boundary.py",
        "t2_4_inference_throughput.py",
    }:
        old = """    model = BitNetForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )"""
        new = """    model = BitNetForCausalLM.from_pretrained(
        path,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=Path(path).exists(),
    )"""
        text = text.replace(old, new)

    if name == "t1_3_short_context_regression.py":
        if "from pathlib import Path" not in text:
            text = text.replace(
                "import json, time, subprocess, sys",
                "import json, time, subprocess, sys\nfrom pathlib import Path",
            )
        text = text.replace(
            'model_args = f"pretrained={model_path},dtype=bfloat16"',
            'model_args = f"pretrained={Path(model_path).resolve()},dtype=bfloat16"',
        )
        text = text.replace(
            'OUTPUT_DIR      = Path("/bitnet_output/benchmark_results")',
            'OUTPUT_DIR      = Path("/home/bitnet_output/benchmark_results")',
        )

    if name == "t2_2_subnorm_analysis.py":
        text = text.replace(
            "STEPS = [0, 1000, 2500, 5000]",
            "STEPS = [0, 1000, 2500, 5000, 6000, 8000]",
        )

    if name == "setup_and_run_benchmarks.py":
        text = text.replace(
            'FINETUNED_MODEL_PATH = "/bitnet_output/bitnet_8k_sft/final_model"',
            f'FINETUNED_MODEL_PATH = "{MODEL_PATH}"',
        )
        text = text.replace(
            'SUBNORM_PROFILES_PATH = "/bitnet_output/bitnet_8k_sft/subnorm_profiles"',
            f'SUBNORM_PROFILES_PATH = "{SUBNORM_PATH}"',
        )
        text = text.replace(
            'OUTPUT_DIR = Path("/bitnet_output/benchmark_results")',
            f'OUTPUT_DIR = Path("{BENCH_OUT}")',
        )

    if text != orig:
        p.write_text(text)
        log(f"patched: {name}")
    else:
        log(f"no changes: {name}")

def main():
    log("checking required paths")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"missing model path: {MODEL_PATH}")
    if not Path(SUBNORM_PATH).exists():
        raise FileNotFoundError(f"missing subnorm path: {SUBNORM_PATH}")

    for name in FILES:
        patch_file(name)

    log("starting setup_and_run_benchmarks.py")
    rc = subprocess.call([sys.executable, "setup_and_run_benchmarks.py"])
    log(f"setup_and_run_benchmarks.py finished with rc={rc}")
    sys.exit(rc)

if __name__ == "__main__":
    main()
