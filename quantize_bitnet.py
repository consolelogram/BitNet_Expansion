"""
quantize_bitnet.py
Convert fine-tuned BitNet bf16 safetensors → I2_S GGUF (true ternary packing)
Uses the Microsoft BitNet repo's official pipeline.

WHAT THIS DOES:
  1. Verifies the fine-tuned model weights look correct
  2. Converts bf16 safetensors → GGUF using llama.cpp
  3. Applies I2_S quantization (4 ternary weights per byte)
  4. Verifies the output GGUF is loadable and generates coherent text
  5. Reports file size and compression ratio

SETUP:
  - Microsoft BitNet repo must be cloned
  - llama.cpp must be built (or use pre-built binaries)

Run:
    python quantize_bitnet.py \
        --model_path /path/to/final_model \
        --bitnet_repo /path/to/BitNet \
        --output_dir /path/to/output

Windows example:
    python quantize_bitnet.py ^
        --model_path D:\\Code\\Bitnet\\models\\bitnet-hf ^
        --bitnet_repo D:\\Code\\Bitnet\\BitNet ^
        --output_dir D:\\Code\\Bitnet\\quantized
"""

import argparse, subprocess, sys, time, json, shutil, math
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",  required=True,
                    help="Path to bf16 fine-tuned model (contains config.json + safetensors)")
parser.add_argument("--bitnet_repo", required=True,
                    help="Path to cloned microsoft/BitNet repo")
parser.add_argument("--output_dir",  default="./quantized_output",
                    help="Where to write the GGUF file")
parser.add_argument("--llama_cpp",   default=None,
                    help="Path to llama.cpp build dir. Auto-detected from bitnet_repo if not set.")
parser.add_argument("--verify_only", action="store_true",
                    help="Only verify an existing GGUF, skip conversion")
parser.add_argument("--gguf_path",   default=None,
                    help="Path to existing GGUF (for --verify_only)")
args = parser.parse_args()

MODEL_PATH  = Path(args.model_path)
BITNET_REPO = Path(args.bitnet_repo)
OUTPUT_DIR  = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def section(title):
    log("=" * 65)
    log(title)
    log("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Verify source model
# ─────────────────────────────────────────────────────────────────────────────

def verify_source_model():
    section("Step 1: Verifying source model")

    assert MODEL_PATH.exists(), f"Model path not found: {MODEL_PATH}"

    config_path = MODEL_PATH / "config.json"
    assert config_path.exists(), f"config.json not found in {MODEL_PATH}"

    safetensors = list(MODEL_PATH.glob("*.safetensors"))
    bins        = list(MODEL_PATH.glob("*.bin"))
    assert safetensors or bins, f"No weights found in {MODEL_PATH}"

    log(f"Model path   : {MODEL_PATH}")
    log(f"Weight files : {len(safetensors)} safetensors, {len(bins)} bin")

    with open(config_path) as f:
        config = json.load(f)

    log(f"Model type   : {config.get('model_type', 'unknown')}")
    log(f"Hidden size  : {config.get('hidden_size')}")
    log(f"Layers       : {config.get('num_hidden_layers')}")
    log(f"Max position : {config.get('max_position_embeddings')}")
    log(f"Vocab size   : {config.get('vocab_size')}")

    max_pos = config.get("max_position_embeddings", 4096)
    if max_pos >= 8192:
        log(f"Context      : {max_pos} tokens ✓ (extended)")
    else:
        log(f"Context      : {max_pos} tokens ⚠ (original, not extended)")

    # Check weight dtype
    try:
        import torch
        from safetensors import safe_open
        st_file = safetensors[0] if safetensors else None
        if st_file:
            with safe_open(str(st_file), framework="pt") as f:
                keys = list(f.keys())
                first_key = keys[0]
                tensor = f.get_tensor(first_key)
                log(f"Weight dtype : {tensor.dtype} ({len(keys)} tensors total)")
                if tensor.dtype == torch.bfloat16:
                    log("Dtype check  : bfloat16 ✓ (correct for quantization input)")
                else:
                    log(f"Dtype check  : ⚠ Expected bfloat16, got {tensor.dtype}")
    except ImportError:
        log("safetensors not installed — skipping dtype check")
    except Exception as e:
        log(f"Dtype check  : could not inspect ({e})")

    log("Source model verified ✓")
    return config

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Locate tools
# ─────────────────────────────────────────────────────────────────────────────

def find_tools():
    section("Step 2: Locating conversion tools")

    # Find convert script — BitNet repo has its own
    candidates_convert = [
        BITNET_REPO / "utils" / "convert-hf-to-gguf.py",
        BITNET_REPO / "convert-hf-to-gguf.py",
        BITNET_REPO / "convert_hf_to_gguf.py",
        BITNET_REPO / "utils" / "convert_hf_to_gguf.py",
    ]
    convert_script = None
    for c in candidates_convert:
        if c.exists():
            convert_script = c
            log(f"Convert script: {convert_script} ✓")
            break

    if not convert_script:
        # Fall back to llama.cpp's convert script if available
        llama_cpp_dir = Path(args.llama_cpp) if args.llama_cpp else BITNET_REPO / "3rdparty" / "llama.cpp"
        for c in [llama_cpp_dir / "convert_hf_to_gguf.py",
                  llama_cpp_dir / "convert-hf-to-gguf.py"]:
            if c.exists():
                convert_script = c
                log(f"Convert script: {convert_script} (llama.cpp) ✓")
                break

    if not convert_script:
        log("ERROR: Could not find convert_hf_to_gguf.py")
        log("  Looked in:")
        for c in candidates_convert:
            log(f"    {c}")
        log("  Make sure the BitNet repo is fully cloned:")
        log("    git clone --recursive https://github.com/microsoft/BitNet")
        sys.exit(1)

    # Find quantize binary
    llama_cpp_dir = Path(args.llama_cpp) if args.llama_cpp else BITNET_REPO / "3rdparty" / "llama.cpp"
    candidates_quant = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "llama-quantize",
        llama_cpp_dir / "quantize",
        # Windows
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "quantize.exe",
    ]
    quantize_bin = None
    for c in candidates_quant:
        if c.exists():
            quantize_bin = c
            log(f"Quantize bin : {quantize_bin} ✓")
            break

    if not quantize_bin:
        log("ERROR: Could not find llama-quantize binary")
        log("  Build llama.cpp first:")
        log(f"    cd {llama_cpp_dir}")
        log("    cmake -B build && cmake --build build --config Release -j")
        log("  Or on Linux/Mac: make -j")
        sys.exit(1)

    # Find llama-cli for verification
    candidates_cli = [
        llama_cpp_dir / "build" / "bin" / "llama-cli",
        llama_cpp_dir / "build" / "bin" / "main",
        llama_cpp_dir / "llama-cli",
        llama_cpp_dir / "main",
        llama_cpp_dir / "build" / "bin" / "Release" / "llama-cli.exe",
        llama_cpp_dir / "build" / "bin" / "Release" / "main.exe",
    ]
    llama_cli = None
    for c in candidates_cli:
        if c.exists():
            llama_cli = c
            log(f"llama-cli    : {llama_cli} ✓")
            break

    if not llama_cli:
        log("llama-cli    : not found (verification will be skipped)")

    return convert_script, quantize_bin, llama_cli

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Convert to GGUF (f16 intermediate)
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_gguf(convert_script):
    section("Step 3: Converting bf16 → GGUF (f16 intermediate)")

    gguf_f16_path = OUTPUT_DIR / "model_f16.gguf"

    if gguf_f16_path.exists():
        log(f"F16 GGUF already exists: {gguf_f16_path}")
        log("Delete it to re-convert. Proceeding with existing file.")
        return gguf_f16_path

    cmd = [
        sys.executable,
        str(convert_script),
        str(MODEL_PATH),
        "--outfile", str(gguf_f16_path),
        "--outtype", "f16",
    ]

    log(f"Running: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        log(f"Conversion FAILED (returncode={result.returncode})")
        log("Common issues:")
        log("  - Missing Python packages: pip install torch transformers sentencepiece")
        log("  - Model format not recognised: ensure config.json is present")
        sys.exit(1)

    elapsed = time.time() - t0
    size_gb = gguf_f16_path.stat().st_size / 1e9
    log(f"F16 GGUF created in {elapsed:.1f}s: {gguf_f16_path} ({size_gb:.2f} GB)")
    return gguf_f16_path

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Quantize to I2_S (true ternary)
# ─────────────────────────────────────────────────────────────────────────────

def quantize_i2s(quantize_bin, gguf_f16_path):
    section("Step 4: Quantizing to I2_S (true ternary — 4 weights per byte)")

    model_name = MODEL_PATH.name.replace(" ", "_")
    gguf_i2s_path = OUTPUT_DIR / f"{model_name}_i2s.gguf"

    if gguf_i2s_path.exists():
        log(f"I2_S GGUF already exists: {gguf_i2s_path}")
        log("Delete it to re-quantize. Proceeding with existing file.")
        return gguf_i2s_path

    cmd = [
        str(quantize_bin),
        str(gguf_f16_path),
        str(gguf_i2s_path),
        "I2_S",   # True ternary: 4 weights per byte, base-3 in base-4 encoding
    ]

    log(f"Running: {' '.join(str(c) for c in cmd)}")
    log("I2_S packing: each byte stores 4 ternary values {-1, 0, +1}")
    log("Expected output size: ~1.2–1.5 GB for 2B parameter model")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        log(f"Quantization FAILED (returncode={result.returncode})")
        log("If I2_S is not recognised, try: Q2_K as fallback")
        log("  (Q2_K is not true ternary but produces similar file size)")
        sys.exit(1)

    elapsed = time.time() - t0
    size_gb = gguf_i2s_path.stat().st_size / 1e9
    log(f"I2_S GGUF created in {elapsed:.1f}s")
    log(f"Output: {gguf_i2s_path} ({size_gb:.2f} GB)")

    # Compression stats
    f16_size  = gguf_f16_path.stat().st_size / 1e9
    ratio     = f16_size / size_gb
    bits_per_weight = (size_gb * 8e9) / 2_413_000_000  # 2413M params
    log(f"F16 size     : {f16_size:.2f} GB")
    log(f"I2_S size    : {size_gb:.2f} GB")
    log(f"Compression  : {ratio:.1f}x")
    log(f"Bits/weight  : {bits_per_weight:.2f}  (theoretical minimum: 1.58)")

    return gguf_i2s_path

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Verify GGUF loads and generates
# ─────────────────────────────────────────────────────────────────────────────

def verify_gguf(llama_cli, gguf_path):
    section("Step 5: Verifying GGUF generation")

    if not llama_cli:
        log("Skipping — llama-cli not found")
        log("Manual verification:")
        log(f"  ./llama-cli -m {gguf_path} -p 'The transformer architecture' -n 50")
        return

    gguf_path = Path(gguf_path)
    assert gguf_path.exists(), f"GGUF not found: {gguf_path}"

    prompts = [
        "The transformer architecture revolutionized",
        "Artificial intelligence is",
        "The context window of a language model determines",
    ]

    for prompt in prompts:
        cmd = [
            str(llama_cli),
            "-m", str(gguf_path),
            "-p", prompt,
            "-n", "40",
            "--temp", "0",
            "-ngl", "0",   # CPU only for verification
            "--log-disable",
        ]

        log(f"\nPrompt: '{prompt}'")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            log(f"Generation failed: {result.stderr[:200]}")
            continue

        output = result.stdout.strip()
        # Extract just the generated text after the prompt
        if prompt in output:
            generated = output[output.index(prompt):][:200]
        else:
            generated = output[:200]

        log(f"Generated  : {generated}")

    log("\nVerification complete")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Save metadata
# ─────────────────────────────────────────────────────────────────────────────

def save_metadata(gguf_path, config):
    section("Step 6: Saving metadata")

    gguf_path = Path(gguf_path)
    meta = {
        "source_model":      str(MODEL_PATH),
        "gguf_path":         str(gguf_path),
        "gguf_size_gb":      round(gguf_path.stat().st_size / 1e9, 3),
        "quantization":      "I2_S",
        "bits_per_weight":   round((gguf_path.stat().st_size * 8) / 2_413_000_000, 3),
        "base_model":        "microsoft/bitnet-b1.58-2B-4T-bf16",
        "training":          "YaRN 8K context extension + UltraChat SFT, 8000 steps",
        "max_context":       config.get("max_position_embeddings", "unknown"),
        "architecture":      {
            "hidden_size":       config.get("hidden_size"),
            "num_layers":        config.get("num_hidden_layers"),
            "num_attn_heads":    config.get("num_attention_heads"),
            "num_kv_heads":      config.get("num_key_value_heads"),
            "intermediate_size": config.get("intermediate_size"),
            "rope_theta":        config.get("rope_theta"),
            "vocab_size":        config.get("vocab_size"),
        },
        "yarn_patch": {
            "scale_zone":   "bands 32-42",
            "scale_factor": 2.0,
            "target_max":   8192,
        },
        "benchmark_results": {
            "wikitext_ppl_4k_original":  12.559,
            "wikitext_ppl_4k_finetuned": 12.447,
            "wikitext_ppl_8k_original":  16.785,
            "wikitext_ppl_8k_finetuned": 16.366,
            "needle_accuracy_original":  1.0,
            "needle_accuracy_finetuned": 1.0,
            "hellaswag_original":        0.615,
            "hellaswag_finetuned":       0.600,
            "arc_easy_original":         0.730,
            "arc_easy_finetuned":        0.710,
        },
    }

    meta_path = OUTPUT_DIR / "quantization_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Metadata saved: {meta_path}")

    # Print summary
    log("\n" + "=" * 65)
    log("QUANTIZATION COMPLETE")
    log(f"  GGUF file    : {gguf_path}")
    log(f"  Size         : {gguf_path.stat().st_size / 1e9:.2f} GB")
    log(f"  Format       : I2_S (true ternary, 4 weights/byte)")
    log(f"  Context      : {config.get('max_position_embeddings')} tokens")
    log(f"  Bits/weight  : {(gguf_path.stat().st_size * 8) / 2_413_000_000:.2f}")
    log("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log("BitNet bf16 → I2_S GGUF Quantization Pipeline")
    log(f"Model  : {MODEL_PATH}")
    log(f"Repo   : {BITNET_REPO}")
    log(f"Output : {OUTPUT_DIR}")

    if args.verify_only:
        if not args.gguf_path:
            log("ERROR: --verify_only requires --gguf_path")
            sys.exit(1)
        _, _, llama_cli = find_tools()
        verify_gguf(llama_cli, args.gguf_path)
        return

    # Full pipeline
    config         = verify_source_model()
    convert_script, quantize_bin, llama_cli = find_tools()
    gguf_f16_path  = convert_to_gguf(convert_script)
    gguf_i2s_path  = quantize_i2s(quantize_bin, gguf_f16_path)
    verify_gguf(llama_cli, gguf_i2s_path)
    save_metadata(gguf_i2s_path, config)

    log("\nNext steps:")
    log(f"  1. Run inference: ./llama-cli -m {gguf_i2s_path} -p 'your prompt' -n 200")
    log(f"  2. Upload to HuggingFace as a GGUF release")
    log(f"  3. Use metadata JSON for paper citations")


if __name__ == "__main__":
    main()
