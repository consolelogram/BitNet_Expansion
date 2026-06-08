#!/usr/bin/env python3
"""
BitNet b1.58 4K -> 8K context-extension recipe with safer defaults.

What this script changes versus a plain YaRN patch:
  1) Keeps your audited RoPE decision by default: scale only bands 32-42 by 2.0x.
  2) Uses overlapping long-document windows instead of only disjoint 8K chunks.
  3) Uses a short->long curriculum to preserve 4K while improving 8K.
  4) Uses a 3-phase unfreeze schedule for high-gain sub_norm layers.
  5) Slightly upweights the loss beyond 4K during later phases.
  6) Can start from the fresh 4K HF model OR resume from an existing checkpoint.

Recommended use:
  - For best results, start from the fresh 4K HF model for this stage.
  - After this stage finishes, run your SFT on the resulting 8K checkpoint.
  - Only resume from an existing checkpoint if it is a clean long-context checkpoint,
    not a heavily instruction-tuned model that already drifted.

Install:
  pip install -U transformers==4.46.3 datasets accelerate safetensors matplotlib

Run:
  nohup python bitnet_yarn_8k_recipe_v2.py > train.log 2>&1 &
  tail -f train.log
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import BitNetForCausalLM, AutoTokenizer

# ============================================================================
# EDIT ONLY THIS BLOCK
# ============================================================================

REPO_ID = os.environ.get("MODEL_NAME_OR_PATH", "microsoft/bitnet-b1.58-2B-4T-bf16")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./bitnet_yarn_recipe_output"))

# RoPE / YaRN: keep the audited setting by default.
ROPE_THETA = 500_000
HEAD_DIM = 128
ORIGINAL_MAX = 4096
TARGET_MAX = 8192
SCALE_ZONE = (32, 42)
SCALE_FACTOR = 2.0
USE_EDGE_TAPER = False  # keep False for the safest first run
TAPER_RATIO = 0.85      # used only when USE_EDGE_TAPER=True

# Training
TOTAL_STEPS = 6000
GRAD_ACCUM = 8
BATCH_SIZE = 1
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
LOG_EVERY = 50
EVAL_EVERY = 100
SAVE_EVERY = 500
EVAL_SAMPLES = 20

# Data
N_WINDOWS = 3000
MIN_DOC_TOKENS = 10_000
WINDOW_STRIDE = 4096  # overlap improves coverage near boundaries
DATASET_NAME = "togethercomputer/RedPajama-Data-V2"
DATASET_CONFIG = "sample"   # first test this; later switch to "default"
DATASET_SPLIT = "train"


# Sub_norm / phase schedule
HIGH_GAIN_LAYERS = [25, 26, 27, 28, 29]
PHASES = [
    # end_step, lr, frozen_sub_norm_layers, long_tail_weight
    (1500, 1.0e-5, [25, 26, 27, 28, 29], 1.05),
    (3500, 7.0e-6, [25, 26],             1.15),
    (6000, 5.0e-6, [],                   1.25),
]

# Eval lengths
EVAL_LENGTHS = [2048, 4096, 8192]

# Runtime
SEED = 1337
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WindowStore:
    windows: List[torch.Tensor]
    train_windows: List[torch.Tensor]
    eval_windows: List[torch.Tensor]


# ----------------------------------------------------------------------------
# YaRN helpers
# ----------------------------------------------------------------------------

def compute_base_inv_freq(theta: float, head_dim: int) -> np.ndarray:
    i = np.arange(0, head_dim, 2, dtype=np.float64)
    return 1.0 / (theta ** (i / head_dim))


def compute_yarn_inv_freq(
    theta: float,
    head_dim: int,
    scale_zone: Tuple[int, int],
    scale_factor: float,
    use_edge_taper: bool = False,
    taper_ratio: float = 0.85,
) -> np.ndarray:
    base = compute_base_inv_freq(theta, head_dim)
    scale = np.ones(head_dim // 2, dtype=np.float64)
    lo, hi = scale_zone
    scale[lo : hi + 1] = 1.0 / scale_factor

    # Optional smoother boundary. Keep off by default for the safest first run.
    if use_edge_taper:
        scale[lo] = 1.0 / (1.0 + (scale_factor - 1.0) * taper_ratio)
        scale[hi] = 1.0 / (1.0 + (scale_factor - 1.0) * taper_ratio)

    return (base * scale).astype(np.float32)


def save_yarn_array(yarn_inv_freq: np.ndarray, output_dir: Path) -> Path:
    path = output_dir / "yarn_inv_freq.npy"
    np.save(path, yarn_inv_freq)
    return path


def patch_rope(model: torch.nn.Module, yarn_inv_freq: torch.Tensor) -> int:
    if not hasattr(model, "model") or not hasattr(model.model, "rotary_emb"):
        raise RuntimeError("Expected shared model.model.rotary_emb on BitNet model.")
    model.model.rotary_emb.register_buffer("inv_freq", yarn_inv_freq.clone(), persistent=True)
    model.config.max_position_embeddings = TARGET_MAX
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = TARGET_MAX
    return 1


def verify_yarn() -> None:
    base = compute_base_inv_freq(ROPE_THETA, HEAD_DIM)
    yarn = compute_yarn_inv_freq(
        ROPE_THETA,
        HEAD_DIM,
        SCALE_ZONE,
        SCALE_FACTOR,
        use_edge_taper=USE_EDGE_TAPER,
        taper_ratio=TAPER_RATIO,
    )
    base_wl = (2 * np.pi) / base
    yarn_wl = (2 * np.pi) / yarn
    lo, hi = SCALE_ZONE

    # Preserve your audited rationale.
    if not base_wl[31] < ORIGINAL_MAX:
        raise RuntimeError("Band 31 is no longer saturated; rerun the audit before changing the zone.")
    if not yarn_wl[32] > TARGET_MAX:
        raise RuntimeError("Band 32 does not safely clear 8K after scaling.")
    if not np.isclose(yarn_wl[32] / base_wl[32], 2.0 if not USE_EDGE_TAPER else (1.0 / (1.0 / (1.0 + (SCALE_FACTOR - 1.0) * TAPER_RATIO))), rtol=1e-3):
        raise RuntimeError("Unexpected ratio at band 32.")
    for b in range(0, lo):
        if not np.isclose(yarn[b], base[b], rtol=1e-6):
            raise RuntimeError(f"Band {b} changed unexpectedly.")
    for b in range(hi + 1, HEAD_DIM // 2):
        if not np.isclose(yarn[b], base[b], rtol=1e-6):
            raise RuntimeError(f"Band {b} changed unexpectedly.")

    log(
        f"YaRN verified | band31 base_wl={base_wl[31]:.1f} | band32 base_wl={base_wl[32]:.1f} -> yarn_wl={yarn_wl[32]:.1f}"
    )
    log(
        f"Zone {lo}-{hi} | factor={SCALE_FACTOR} | edge_taper={USE_EDGE_TAPER} | bands 43+ untouched"
    )


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def build_windows(tokenizer: AutoTokenizer) -> WindowStore:
    log(
        f"Streaming {DATASET_NAME}:{DATASET_CONFIG}:{DATASET_SPLIT} and building "
        f"{N_WINDOWS} overlapping {TARGET_MAX}-token windows (stride={WINDOW_STRIDE})..."
    )

    if DATASET_CONFIG == "sample":
        ds = load_dataset(
            DATASET_NAME,
            name="sample",
            split=DATASET_SPLIT,
            streaming=True,
            trust_remote_code=True,
        )
    else:
        ds = load_dataset(
            DATASET_NAME,
            name="default",
            split=DATASET_SPLIT,
            streaming=True,
            trust_remote_code=True,
            partition=RP_PARTITION,
            snapshots=RP_SNAPSHOTS,
            languages=RP_LANGUAGES,
        )

    windows: List[torch.Tensor] = []

    for doc in ds:
        if len(windows) >= N_WINDOWS:
            break

        text = doc.get("raw_content")
        if not text:
            continue

        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) < MIN_DOC_TOKENS or len(ids) < TARGET_MAX:
            continue

        for start in range(0, max(1, len(ids) - TARGET_MAX + 1), WINDOW_STRIDE):
            end = start + TARGET_MAX
            if end > len(ids):
                break
            windows.append(torch.tensor(ids[start:end], dtype=torch.long))
            if len(windows) >= N_WINDOWS:
                break

    if len(windows) < 256:
        raise RuntimeError(
            f"Too few windows built: {len(windows)}. "
            "Try DATASET_CONFIG='sample' first, or reduce MIN_DOC_TOKENS."
        )

    random.shuffle(windows)
    eval_count = max(64, int(0.05 * len(windows)))
    eval_windows = windows[:eval_count]
    train_windows = windows[eval_count:]
    log(f"Windows ready | total={len(windows)} | train={len(train_windows)} | eval={len(eval_windows)}")
    return WindowStore(windows=windows, train_windows=train_windows, eval_windows=eval_windows)

# ----------------------------------------------------------------------------
# Sampling / curriculum
# ----------------------------------------------------------------------------

def sample_seq_len(step: int) -> int:
    progress = step / max(TOTAL_STEPS - 1, 1)
    if progress < 0.20:
        choices = [2048, 4096]
        probs = [0.35, 0.65]
    elif progress < 0.60:
        choices = [2048, 4096, 6144, 8192]
        probs = [0.10, 0.35, 0.25, 0.30]
    else:
        choices = [4096, 6144, 8192]
        probs = [0.15, 0.25, 0.60]
    return random.choices(choices, weights=probs, k=1)[0]


def sample_batch(windows: Sequence[torch.Tensor], seq_len: int) -> torch.Tensor:
    chunk = random.choice(windows)
    if seq_len == TARGET_MAX:
        return chunk.clone()
    max_start = TARGET_MAX - seq_len
    start = 0 if max_start <= 0 else random.randint(0, max_start)
    return chunk[start : start + seq_len].clone()


# ----------------------------------------------------------------------------
# Loss / eval
# ----------------------------------------------------------------------------

def weighted_causal_lm_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    long_tail_weight: float,
) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float().contiguous()
    labels = input_ids[:, 1:].contiguous()

    losses = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="none",
    ).view_as(labels)

    weights = torch.ones_like(losses, dtype=losses.dtype)
    if input_ids.shape[1] > ORIGINAL_MAX and long_tail_weight != 1.0:
        tail_start = max(ORIGINAL_MAX - 1, 0)
        weights[:, tail_start:] = long_tail_weight

    return (losses * weights).sum() / weights.sum()


@torch.no_grad()
def eval_ppl(model: torch.nn.Module, windows: Sequence[torch.Tensor], max_len: int, n_samples: int) -> Tuple[float, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for idx in range(min(n_samples, len(windows))):
        ids = windows[idx][:max_len].unsqueeze(0).to(DEVICE)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        n_tokens = ids.shape[1] - 1
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens
    avg_loss = total_nll / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    return ppl, avg_loss


# ----------------------------------------------------------------------------
# Sub_norm management
# ----------------------------------------------------------------------------

def is_high_gain_subnorm(name: str) -> bool:
    return ("sub_norm" in name) and any(f"layers.{i}." in name for i in HIGH_GAIN_LAYERS)


def apply_freeze_mask(model: torch.nn.Module, frozen_sub_norm_layers: Sequence[int]) -> None:
    frozen_set = set(frozen_sub_norm_layers)
    for name, param in model.named_parameters():
        if "sub_norm" in name and any(f"layers.{i}." in name for i in frozen_set):
            param.requires_grad = False
        else:
            param.requires_grad = True

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Freeze mask applied | frozen={frozen/1e6:.2f}M | trainable={trainable/1e6:.2f}M")


def make_optimizer(model: torch.nn.Module, lr: float) -> AdamW:
    base_params = []
    top_params = []
    top_subnorm = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_high_gain_subnorm(name):
            top_subnorm.append(param)
        elif any(f"layers.{i}." in name for i in range(24, 30)):
            top_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": lr * 0.80, "weight_decay": WEIGHT_DECAY})
    if top_params:
        groups.append({"params": top_params, "lr": lr, "weight_decay": WEIGHT_DECAY})
    if top_subnorm:
        groups.append({"params": top_subnorm, "lr": lr * 1.15, "weight_decay": 0.0})

    fused_ok = bool(torch.cuda.is_available())
    try:
        opt = AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=fused_ok)
    except TypeError:
        opt = AdamW(groups, betas=(0.9, 0.95), eps=1e-8)
    return opt


def make_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int = 200, min_ratio: float = 0.10) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------------
# Logging / checkpointing
# ----------------------------------------------------------------------------

def init_csv(path: Path) -> None:
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step", "phase", "seq_len", "train_loss", "lr_max_group",
            "ppl_2048", "ppl_4096", "ppl_8192", "elapsed_min"
        ])


def append_csv(path: Path, row: list) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def detect_phase(step: int) -> Tuple[int, float, List[int], float]:
    for idx, (end_step, lr, frozen_layers, tail_weight) in enumerate(PHASES, start=1):
        if step < end_step:
            return idx, lr, list(frozen_layers), tail_weight
    idx, lr, frozen_layers, tail_weight = len(PHASES), PHASES[-1][1], list(PHASES[-1][2]), PHASES[-1][3]
    return idx, lr, frozen_layers, tail_weight


def save_checkpoint(model: torch.nn.Module, tokenizer: AutoTokenizer, output_dir: Path, step: int, csv_path: Path, yarn_path: Path) -> None:
    ckpt = output_dir / "checkpoints" / f"step_{step:05d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt))
    tokenizer.save_pretrained(str(ckpt))
    shutil.copy2(yarn_path, ckpt / "yarn_inv_freq.npy")
    shutil.copy2(csv_path, ckpt / "train_log.csv")
    log(f"Checkpoint saved: {ckpt}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)

    if DEVICE != "cuda":
        raise RuntimeError("CUDA is required for this script.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    log("=" * 72)
    log("BitNet b1.58 | audited YaRN 8K recipe")
    log("=" * 72)
    log(f"Model init: {REPO_ID}")
    log(f"GPU       : {torch.cuda.get_device_name(0)}")
    log(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"PyTorch   : {torch.__version__}")

    verify_yarn()
    yarn_np = compute_yarn_inv_freq(
        ROPE_THETA,
        HEAD_DIM,
        SCALE_ZONE,
        SCALE_FACTOR,
        use_edge_taper=USE_EDGE_TAPER,
        taper_ratio=TAPER_RATIO,
    )
    yarn_path = save_yarn_array(yarn_np, OUTPUT_DIR)
    yarn_t = torch.tensor(yarn_np, dtype=torch.float32, device=DEVICE)

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading model...")
    model = BitNetForCausalLM.from_pretrained(
        load_path,
        torch_dtype=DTYPE,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    model.config.use_cache = False
    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
        log("Kernels: disabled (pure PyTorch) ✓")
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()

    patched = patch_rope(model, yarn_t)
    if patched != 1:
        raise RuntimeError(f"Expected to patch shared rotary_emb once, patched {patched}.")
    log("RoPE patched in shared model.model.rotary_emb.")

    store = build_windows(tokenizer)

    baseline = {}
    log("Computing baseline PPL...")
    for eval_len in EVAL_LENGTHS:
        ppl, loss = eval_ppl(model, store.eval_windows, eval_len, EVAL_SAMPLES)
        baseline[str(eval_len)] = {"ppl": ppl, "loss": loss}
        log(f"Baseline @ {eval_len:>4}: ppl={ppl:.3f} | loss={loss:.4f}")

    csv_path = OUTPUT_DIR / "train_log.csv"
    init_csv(csv_path)

    current_phase_id = None
    optimizer = None
    scheduler = None
    long_tail_weight = 1.0

    step_start_time = time.time()
    accum_count = 0
    running_loss = 0.0

    for step in range(TOTAL_STEPS):
        phase_id, phase_lr, frozen_layers, long_tail_weight = detect_phase(step)
        if phase_id != current_phase_id:
            current_phase_id = phase_id
            log(f"\\nEntering phase {phase_id} | lr={phase_lr:.2e} | frozen_sub_norm={frozen_layers} | tail_weight={long_tail_weight:.2f}")
            apply_freeze_mask(model, frozen_layers)
            optimizer = make_optimizer(model, phase_lr)
            phase_total_steps = PHASES[phase_id - 1][0] - (0 if phase_id == 1 else PHASES[phase_id - 2][0])
            scheduler = make_scheduler(optimizer, total_steps=phase_total_steps, warmup_steps=min(200, max(phase_total_steps // 10, 50)))
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        seq_len = sample_seq_len(step)
        ids = sample_batch(store.train_windows, seq_len).unsqueeze(0).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=DTYPE):
            loss = weighted_causal_lm_loss(model, ids, long_tail_weight=long_tail_weight)
            loss = loss / GRAD_ACCUM

        loss.backward()
        running_loss += loss.item()
        accum_count += 1

        if accum_count == GRAD_ACCUM:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

        if step % LOG_EVERY == 0:
            elapsed_min = (time.time() - step_start_time) / 60.0
            lr_max = max(group["lr"] for group in optimizer.param_groups)
            log(
                f"step={step:>5} | phase={phase_id} | seq={seq_len:>4} | train_loss={running_loss * GRAD_ACCUM / max(LOG_EVERY,1):.4f} | lr_max={lr_max:.2e} | elapsed={elapsed_min:.1f}m"
            )
            running_loss = 0.0

        if step % EVAL_EVERY == 0:
            eval_row = {}
            model.eval()
            for eval_len in EVAL_LENGTHS:
                ppl, loss_eval = eval_ppl(model, store.eval_windows, eval_len, EVAL_SAMPLES)
                eval_row[eval_len] = ppl
                log(f"  eval @ {eval_len:>4}: ppl={ppl:.3f} | loss={loss_eval:.4f}")
            model.train()
            append_csv(
                csv_path,
                [
                    step,
                    phase_id,
                    seq_len,
                    float(loss.item() * GRAD_ACCUM),
                    max(group["lr"] for group in optimizer.param_groups),
                    eval_row[2048],
                    eval_row[4096],
                    eval_row[8192],
                    round((time.time() - step_start_time) / 60.0, 2),
                ],
            )

        if (step + 1) % SAVE_EVERY == 0:
            model.eval()
            save_checkpoint(model, tokenizer, OUTPUT_DIR, step + 1, csv_path, yarn_path)
            model.train()

    log("\\nFinal evaluation...")
    model.eval()
    final = {}
    for eval_len in EVAL_LENGTHS:
        ppl, loss = eval_ppl(model, store.eval_windows, eval_len, EVAL_SAMPLES)
        final[str(eval_len)] = {"ppl": ppl, "loss": loss}
        log(f"Final @ {eval_len:>4}: ppl={ppl:.3f} | loss={loss:.4f}")

    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    shutil.copy2(yarn_path, final_dir / "yarn_inv_freq.npy")
    shutil.copy2(csv_path, final_dir / "train_log.csv")

    summary = {
        "model_init": REPO_ID,
        "rope_theta": ROPE_THETA,
        "head_dim": HEAD_DIM,
        "original_max": ORIGINAL_MAX,
        "target_max": TARGET_MAX,
        "scale_zone": list(SCALE_ZONE),
        "scale_factor": SCALE_FACTOR,
        "use_edge_taper": USE_EDGE_TAPER,
        "total_steps": TOTAL_STEPS,
        "baseline": baseline,
        "final": final,
        "recommendation": "Run your SFT on OUTPUT_DIR/final, not on the original 4K model.",
    }
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log(f"Final model  : {final_dir}")
    log(f"Results JSON : {OUTPUT_DIR / 'results_summary.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
