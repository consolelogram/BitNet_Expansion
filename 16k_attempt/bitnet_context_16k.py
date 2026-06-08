"""
train_bitnet_16k_sft.py
BitNet b1.58 — YaRN 16K Context Extension + Instruction Finetuning
Single unified training run on JarvisLabs RTX6000Ada (48GB VRAM)

THREE PHASES:
  Phase 1 (steps    0–2000): Context extension, layers 25-29 sub_norm FROZEN, lr=1e-5
  Phase 2 (steps 2000–7000): Context extension, fully unfrozen, lr=5e-6
  Phase 3 (steps 7000–10000): Instruction finetuning on UltraChat, lr=1e-6
                              Loss computed ONLY on assistant turns

UPLOAD THESE FILES TO JARVIS:
  train_bitnet_16k_sft.py    ← this file
  yarn_inv_freq.npy          ← regenerated on server if missing (see below)

SETUP:
  pip install -U transformers accelerate datasets safetensors matplotlib
  # Regenerate yarn_inv_freq.npy if you don't have it:
  python -c "
import numpy as np, math
theta,hdim,lo,hi,sf = 500000,128,32,52,4.0
i = np.arange(0,hdim,2,dtype=np.float64)
inv = 1.0/(theta**(i/hdim))
sc  = np.ones(hdim//2); sc[lo:hi+1] = 1.0/sf
np.save('yarn_inv_freq.npy',(inv*sc).astype(np.float32))
print('band32 ratio:', (2*math.pi)/(1.0/(theta**(64/hdim))) / ((2*math.pi)/((inv*sc).astype(np.float32)[32])))
"

RUN:
  python train_bitnet_16k_sft.py --smoke    # 30-step sanity check first
  nohup python train_bitnet_16k_sft.py > /bitnet_output/train.log 2>&1 &
  tail -f /bitnet_output/train.log

OUTPUTS (/bitnet_output/bitnet_16k_sft/):
  checkpoints/step_XXXXX/    saved every 500 steps, keeps only last 2
  subnorm_profiles/          snapshots at steps 0,1000,2500,4000,6000,8000,10000
  final_model/               final weights after all 3 phases
  loss_curve.png             updated at every checkpoint
  train_log.csv              full numeric log with phase labels
  results_summary.json       final PPL + SFT info
  gpu_health.log             VRAM + util throughout
"""

import os, sys, math, time, random, csv, json, shutil, argparse, gc, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--smoke',  action='store_true',
                    help='30-step smoke test')
parser.add_argument('--resume', type=str, default=None,
                    help='Path to checkpoint to resume from')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

REPO_ID    = "microsoft/bitnet-b1.58-2B-4T-bf16"
OUTPUT_DIR = Path("/bitnet_output/bitnet_16k_sft")
YARN_NPY   = Path("./yarn_inv_freq.npy")

# RoPE / YaRN — 16K target, scale factor 4.0 over bands 32-52
ROPE_THETA   = 500_000
HEAD_DIM     = 128
SCALE_ZONE   = (32, 52)
SCALE_FACTOR = 4.0
TARGET_MAX   = 16384
ORIGINAL_MAX = 4096

# ── Phase boundaries ─────────────────────────────────────────────────────────
if args.smoke:
    TOTAL_STEPS  = 30
    PHASE2_START = 10    # unfreeze sub_norm
    PHASE3_START = 20    # switch to SFT
    SAVE_EVERY   = 10
    LOG_EVERY    = 2
    EVAL_SAMPLES = 3
    N_CHUNKS     = 30
    N_SFT_ITEMS  = 20
    WARMUP_STEPS = 3
    SUBNORM_STEPS = {0, 10, 20, 30}
else:
    TOTAL_STEPS  = 10_000
    PHASE2_START = 2_000
    PHASE3_START = 7_000
    SAVE_EVERY   = 500
    LOG_EVERY    = 50
    EVAL_SAMPLES = 20
    N_CHUNKS     = 2_000
    N_SFT_ITEMS  = 5_000
    WARMUP_STEPS = 200
    SUBNORM_STEPS = {0, 1_000, 2_500, 4_000, 6_000, 8_000, 10_000}

# ── Learning rates ────────────────────────────────────────────────────────────
LR_PHASE1    = 1e-5    # context extension, frozen sub_norm
LR_PHASE2    = 5e-6    # context extension, unfrozen
LR_PHASE3    = 1e-6    # instruction finetuning
LR_END       = 5e-7
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
GRAD_ACCUM   = 8
MAX_SEQ_LEN  = 16384
MIN_DOC_TOK  = 12_000   # ensure long enough for 16K chunks

HIGH_GAIN_LAYERS = list(range(25, 30))

# Resume
RESUME_CKPT = Path(args.resume) if args.resume else None
RESUME_STEP = 0
if RESUME_CKPT is not None:
    m = re.search(r'step_(\d+)', RESUME_CKPT.name)
    assert m, f"Cannot parse step from: {RESUME_CKPT.name}"
    RESUME_STEP = int(m.group(1))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# GPU health
# ─────────────────────────────────────────────────────────────────────────────

def gpu_health(label=""):
    if not torch.cuda.is_available():
        return 0, 0
    free, total = torch.cuda.mem_get_info(0)
    alloc       = torch.cuda.memory_allocated(0)
    used_gb     = (total - free) / 1e9
    total_gb    = total / 1e9

    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        smi = r.stdout.strip()
    except Exception:
        smi = "n/a"

    line = (f"[GPU {label}] used={used_gb:.1f}GB free={free/1e9:.1f}GB "
            f"alloc={alloc/1e9:.1f}GB total={total_gb:.1f}GB | smi={smi}")
    log(line)

    health_path = OUTPUT_DIR / "gpu_health.log"
    with open(health_path, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {label} | {line}\n")

    if used_gb / total_gb > 0.92:
        log("⚠  VRAM above 92%")

    return used_gb, total_gb

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight
# ─────────────────────────────────────────────────────────────────────────────

def preflight():
    log("=" * 70)
    mode = "SMOKE TEST" if args.smoke else f"FULL RUN ({TOTAL_STEPS} steps)"
    if RESUME_STEP > 0:
        mode += f" — RESUMING FROM STEP {RESUME_STEP}"
    log(f"BitNet b1.58 — YaRN 16K + Instruction Finetuning — {mode}")
    log("=" * 70)
    log(f"Python  : {sys.version.split()[0]}")
    log(f"PyTorch : {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA not found"
    log(f"GPU     : {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info(0)
    log(f"VRAM    : {total/1e9:.1f}GB total  {free/1e9:.1f}GB free")
    assert total / 1e9 >= 20, f"Only {total/1e9:.1f}GB VRAM — need 20GB+"

    import transformers
    log(f"transformers: {transformers.__version__}")
    major, minor = [int(x) for x in transformers.__version__.split(".")[:2]]
    assert (major, minor) >= (4, 48), \
        f"transformers {transformers.__version__} too old — pip install -U transformers"

    # yarn_inv_freq.npy — generate if missing
    if not YARN_NPY.exists():
        log("yarn_inv_freq.npy not found — generating now...")
        i       = np.arange(0, HEAD_DIM, 2, dtype=np.float64)
        inv     = 1.0 / (ROPE_THETA ** (i / HEAD_DIM))
        sc      = np.ones(HEAD_DIM // 2, dtype=np.float64)
        lo, hi  = SCALE_ZONE
        sc[lo:hi+1] = 1.0 / SCALE_FACTOR
        yarn    = (inv * sc).astype(np.float32)
        np.save(str(YARN_NPY), yarn)
        log(f"yarn_inv_freq.npy generated and saved ✓")
    else:
        log("yarn_inv_freq.npy: found ✓")

    for d in [OUTPUT_DIR, OUTPUT_DIR/"checkpoints", OUTPUT_DIR/"subnorm_profiles"]:
        d.mkdir(parents=True, exist_ok=True)

    log(f"Output dir: {OUTPUT_DIR.resolve()}")
    log(f"Phase 1: steps 0–{PHASE2_START-1}  (context ext, sub_norm frozen, lr={LR_PHASE1:.0e})")
    log(f"Phase 2: steps {PHASE2_START}–{PHASE3_START-1}  (context ext, unfrozen, lr={LR_PHASE2:.0e})")
    log(f"Phase 3: steps {PHASE3_START}–{TOTAL_STEPS-1}  (SFT instruction tuning, lr={LR_PHASE3:.0e})")
    log("Pre-flight: ALL PASSED ✓")
    log("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# YaRN
# ─────────────────────────────────────────────────────────────────────────────

def load_and_verify_yarn():
    yarn_np = np.load(str(YARN_NPY))
    yarn_t  = torch.tensor(yarn_np, dtype=torch.float32).to(DEVICE)

    wl_base = (2 * math.pi) / (1.0 / (ROPE_THETA ** (64 / HEAD_DIM)))
    wl_yarn = (2 * math.pi) / yarn_t[32].item()
    ratio   = wl_yarn / wl_base
    # With scale factor 4.0 and bands 32-52, ratio should be ~4.0
    assert abs(ratio - SCALE_FACTOR) < 0.01, f"Band 32 ratio={ratio:.4f} (expected {SCALE_FACTOR})"

    base_0  = 1.0 / (ROPE_THETA ** (0   / HEAD_DIM))
    base_63 = 1.0 / (ROPE_THETA ** (126 / HEAD_DIM))
    assert abs(yarn_t[0].item()  - base_0)  < 1e-5
    assert abs(yarn_t[63].item() - base_63) < 1e-9

    log(f"YaRN verified: band32_ratio={ratio:.4f} (target {SCALE_FACTOR}) ✓  bands 0,63 untouched ✓")
    return yarn_t


def patch_rope(model, yarn_t):
    assert hasattr(model.model, 'rotary_emb'), "model.model.rotary_emb not found"
    model.model.rotary_emb.register_buffer('inv_freq', yarn_t.clone(), persistent=True)
    model.config.max_position_embeddings = TARGET_MAX
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_length = TARGET_MAX

    ratio = ((2*math.pi) / model.model.rotary_emb.inv_freq[32].item()) / \
            ((2*math.pi) / (1.0 / (ROPE_THETA**(64/HEAD_DIM))))
    assert abs(ratio - SCALE_FACTOR) < 0.01
    log(f"RoPE patched ✓  post-patch band32_ratio={ratio:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Freeze strategy
# ─────────────────────────────────────────────────────────────────────────────

def set_frozen(model, freeze_high_gain: bool):
    for name, param in model.named_parameters():
        if any(f"layers.{i}." in name for i in HIGH_GAIN_LAYERS) and "sub_norm" in name:
            param.requires_grad = not freeze_high_gain
        else:
            param.requires_grad = True
    frozen    = sum(p.numel() for _, p in model.named_parameters() if not p.requires_grad)
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    log(f"  Frozen={frozen/1e6:.2f}M  Trainable={trainable/1e6:.2f}M")

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler — cosine with warmup
# ─────────────────────────────────────────────────────────────────────────────

def make_scheduler(optimizer, warmup, total, lr_end, lr_peak):
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress  = (step - warmup) / max(total - warmup, 1)
        cosine    = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = lr_end / lr_peak
        return min_ratio + (1.0 - min_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)

# ─────────────────────────────────────────────────────────────────────────────
# Context extension dataset — Gutenberg long docs
# ─────────────────────────────────────────────────────────────────────────────

def build_context_chunks(tokenizer):
    from datasets import load_dataset
    log(f"Loading Gutenberg dataset (target {N_CHUNKS} chunks)...")

    try:
        dataset = load_dataset(
            "sedthh/gutenberg_english", split="train", streaming=True
        )
        chunks, buffer = [], []
        for doc in dataset:
            if len(chunks) >= N_CHUNKS:
                break
            text = doc.get("TEXT", doc.get("text", ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < MIN_DOC_TOK:
                continue
            buffer.extend(ids)
            while len(buffer) >= MAX_SEQ_LEN:
                chunks.append(torch.tensor(buffer[:MAX_SEQ_LEN], dtype=torch.long))
                buffer = buffer[MAX_SEQ_LEN:]

        if len(chunks) >= 20:
            log(f"Context dataset: {len(chunks)} chunks from gutenberg_english ✓")
            return chunks
        raise RuntimeError(f"Only {len(chunks)} chunks")

    except Exception as e:
        log(f"Gutenberg failed ({e}) — using synthetic fallback")
        return build_synthetic_chunks(tokenizer)


def build_synthetic_chunks(tokenizer):
    seed = (
        "The transformer architecture introduced the self-attention mechanism which allows "
        "models to relate tokens at any distance in the sequence without regard to their "
        "sequential distance. Residual connections and layer normalisation stabilise training "
        "in deep networks. The feed-forward sublayer applies a pointwise transformation to "
        "each position independently using two linear projections with a nonlinear activation. "
        "Rotary position embeddings encode position information directly into the query and key "
        "vectors rather than adding absolute position embeddings to the input representations. "
    ) * 1000

    ids = tokenizer(seed, add_special_tokens=False)["input_ids"]
    while len(ids) < MAX_SEQ_LEN * N_CHUNKS:
        ids = ids + ids

    chunks = []
    for i in range(0, len(ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
        chunks.append(torch.tensor(ids[i:i+MAX_SEQ_LEN], dtype=torch.long))
        if len(chunks) >= N_CHUNKS:
            break

    log(f"Synthetic context dataset: {len(chunks)} chunks")
    return chunks

# ─────────────────────────────────────────────────────────────────────────────
# SFT dataset — UltraChat instruction pairs
# ─────────────────────────────────────────────────────────────────────────────

# Chat template tokens for BitNet (uses Llama tokenizer)
B_INST = "<|begin_of_text|>"
E_INST = "<|eot_id|>"
USER_TAG  = "<|start_header_id|>user<|end_header_id|>\n\n"
ASST_TAG  = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def format_sft_example(tokenizer, user_text, assistant_text):
    """
    Format a single instruction pair and return:
      input_ids: full sequence
      labels:    -100 for user tokens (masked), token ids for assistant tokens
    """
    # Build full conversation
    prompt = (
        f"{B_INST}"
        f"{USER_TAG}{user_text.strip()}{E_INST}"
        f"{ASST_TAG}{assistant_text.strip()}{E_INST}"
    )

    full_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )["input_ids"]

    # Find where assistant response starts — mask everything before it
    prefix = (
        f"{B_INST}"
        f"{USER_TAG}{user_text.strip()}{E_INST}"
        f"{ASST_TAG}"
    )
    prefix_ids = tokenizer(
        prefix, add_special_tokens=False
    )["input_ids"]

    labels = [-100] * len(full_ids)
    # Only compute loss on assistant tokens
    asst_start = len(prefix_ids)
    for i in range(asst_start, len(full_ids)):
        labels[i] = full_ids[i]

    return (
        torch.tensor(full_ids, dtype=torch.long),
        torch.tensor(labels,   dtype=torch.long),
    )


def build_sft_dataset(tokenizer):
    from datasets import load_dataset
    log(f"Loading UltraChat SFT dataset (target {N_SFT_ITEMS} items)...")

    try:
        dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k",
            split="train_sft",
            streaming=True,
        )
        items = []
        for example in dataset:
            if len(items) >= N_SFT_ITEMS:
                break
            messages = example.get("messages", [])
            if len(messages) < 2:
                continue

            # Get first user/assistant turn
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
            asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)

            if not user_msg or not asst_msg:
                continue
            if len(user_msg) < 20 or len(asst_msg) < 20:
                continue

            ids, labels = format_sft_example(tokenizer, user_msg, asst_msg)
            if ids.shape[0] < 10:
                continue

            # Verify at least some assistant tokens are unmasked
            n_asst_tokens = (labels != -100).sum().item()
            if n_asst_tokens < 5:
                continue

            items.append((ids, labels))

        if len(items) >= 10:
            log(f"SFT dataset: {len(items)} examples from UltraChat ✓")
            return items
        raise RuntimeError(f"Only {len(items)} SFT items")

    except Exception as e:
        log(f"UltraChat failed ({e}) — using synthetic SFT fallback")
        return build_synthetic_sft(tokenizer)


def build_synthetic_sft(tokenizer):
    """Synthetic instruction pairs for smoke test / fallback."""
    pairs = [
        ("What is machine learning?",
         "Machine learning is a branch of artificial intelligence that enables systems "
         "to learn and improve from experience without being explicitly programmed. "
         "It focuses on developing algorithms that can access data and use it to learn for themselves."),
        ("Explain the attention mechanism in transformers.",
         "The attention mechanism allows a model to focus on different parts of the input "
         "sequence when producing each output token. It computes a weighted sum of value "
         "vectors, where the weights are determined by the compatibility between query and key vectors."),
        ("What is the difference between supervised and unsupervised learning?",
         "Supervised learning uses labelled training data where the correct output is known "
         "for each input. Unsupervised learning finds patterns in data without labelled examples, "
         "discovering structure through clustering, dimensionality reduction, or generative modelling."),
        ("How does backpropagation work?",
         "Backpropagation computes gradients of the loss function with respect to each parameter "
         "by applying the chain rule of calculus backwards through the network layers. "
         "These gradients are then used by an optimiser like SGD or Adam to update the weights."),
        ("What are transformers used for?",
         "Transformers are used for a wide range of tasks including natural language processing, "
         "computer vision, speech recognition, and protein structure prediction. "
         "Their ability to model long-range dependencies makes them highly effective for sequence data."),
    ] * max(1, N_SFT_ITEMS // 5)

    items = []
    for user, asst in pairs[:N_SFT_ITEMS]:
        ids, labels = format_sft_example(tokenizer, user, asst)
        items.append((ids, labels))

    log(f"Synthetic SFT dataset: {len(items)} examples")
    return items

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, chunks, max_len, n_samples=None):
    if n_samples is None:
        n_samples = EVAL_SAMPLES
    model.eval()
    total_loss, total_toks = 0.0, 0
    for chunk in chunks[:n_samples]:
        ids = chunk[:max_len].unsqueeze(0).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            out = model(ids, labels=ids)
        total_loss += out.loss.item() * (ids.shape[1] - 1)
        total_toks += ids.shape[1] - 1
    avg = total_loss / total_toks
    return math.exp(avg), avg

# ─────────────────────────────────────────────────────────────────────────────
# Sub_norm snapshot
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_subnorm(model, step):
    rows   = []
    header = ["layer","ffn_mean","ffn_max","ffn_var","attn_mean","attn_max","attn_var","status"]

    for i, layer in enumerate(model.model.layers):
        aw = layer.self_attn.attn_sub_norm.weight.float().abs().detach().cpu()
        fw = layer.mlp.ffn_sub_norm.weight.float().abs().detach().cpu()
        r  = [i,
              fw.mean().item(), fw.max().item(), fw.var().item(),
              aw.mean().item(), aw.max().item(), aw.var().item()]
        r.append("HIGH GAIN" if (r[1] > 2.0 or r[4] > 2.0) else "Standard")
        rows.append(r)

    csv_p = OUTPUT_DIR / "subnorm_profiles" / f"subnorm_step_{step:05d}.csv"
    with open(csv_p, "w", newline="") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(rows)

    txt_p = OUTPUT_DIR / "subnorm_profiles" / f"subnorm_step_{step:05d}.txt"
    with open(txt_p, "w") as f:
        f.write(f"Sub_norm gain profile — step {step}\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Layer':>5} | {'ffn_mean':>8} {'ffn_max':>8} {'ffn_var':>8} | "
                f"{'attn_mean':>9} {'attn_max':>8} {'attn_var':>8} | Status\n")
        f.write("-" * 90 + "\n")
        for r in rows:
            f.write(f"{r[0]:>5} | {r[1]:>8.2f} {r[2]:>8.2f} {r[3]:>8.2f} | "
                    f"{r[4]:>9.2f} {r[5]:>8.2f} {r[6]:>8.2f} | {r[7]}\n")

    threshold = next((r[0] for r in rows if r[7] == "HIGH GAIN"), None)
    log(f"Sub_norm snapshot step {step} → {txt_p.name}")
    log(f"  First HIGH GAIN: layer {threshold}  (baseline=14)")
    log(f"  Layer 29 attn_var: {rows[29][6]:.2f}  (baseline=48.35)")

# ─────────────────────────────────────────────────────────────────────────────
# Loss curve
# ─────────────────────────────────────────────────────────────────────────────

def save_loss_curve(log_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps   = log_data["step"]
        loss_4k = log_data["loss_4k"]
        loss_8k = log_data["loss_8k"]
        phases  = log_data["phase"]

        fig, ax = plt.subplots(figsize=(16, 5))

        # Shade phase regions
        p2 = next((s for s, p in zip(steps, phases) if p == 2), None)
        p3 = next((s for s, p in zip(steps, phases) if p == 3), None)
        if p2:
            p3_or_end = p3 if p3 else steps[-1]
            ax.axvspan(p2, p3_or_end, alpha=0.06, color="orange",
                       label="Phase 2 (unfrozen)")
        if p3:
            ax.axvspan(p3, steps[-1], alpha=0.06, color="green",
                       label="Phase 3 (SFT)")

        ax.plot(steps, loss_4k, "b-", linewidth=1.5, label="Loss @ 4K")
        ax.plot(steps, loss_8k, "r-", linewidth=1.5, label="Loss @ 8K")
        ax.axhline(log_data["baseline_4k"], color="b", linestyle="--",
                   alpha=0.4, label=f"Baseline 4K ({log_data['baseline_4k']:.4f})")
        ax.axhline(log_data["baseline_8k"], color="r", linestyle="--",
                   alpha=0.4, label=f"Baseline 8K ({log_data['baseline_8k']:.4f})")

        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title("BitNet b1.58 — YaRN 16K Context Extension + Instruction Finetuning")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out = OUTPUT_DIR / "loss_curve.png"
        plt.savefig(str(out), dpi=150)
        plt.close()
        log(f"Loss curve saved: {out}")
    except Exception as e:
        log(f"Loss curve save failed (non-fatal): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

import csv as csv_mod

def init_csv(path):
    with open(path, "w", newline="") as f:
        csv_mod.writer(f).writerow(
            ["step","loss_4k","loss_8k","gap","lr","phase","elapsed_min","vram_gb"]
        )

def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv_mod.writer(f).writerow(row)

# ─────────────────────────────────────────────────────────────────────────────
# Build optimizer for a given phase
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer_scheduler(model, lr_peak, total_phase_steps,
                               warmup=0, lr_end=LR_END):
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_peak, weight_decay=WEIGHT_DECAY,
    )
    sched = make_scheduler(opt, warmup, total_phase_steps, lr_end, lr_peak)
    return opt, sched

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    preflight()

    from transformers import BitNetForCausalLM, AutoTokenizer

    # ── YaRN ─────────────────────────────────────────────────────────────────
    yarn_t = load_and_verify_yarn()

    # ── Tokenizer ────────────────────────────────────────────────────────────
    log(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model ─────────────────────────────────────────────────────────────────
    load_path = str(RESUME_CKPT) if RESUME_CKPT else REPO_ID
    log(f"Loading model from: {load_path}")
    gpu_health("before_load")
    t0 = time.time()

    model = BitNetForCausalLM.from_pretrained(
        load_path,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = model.to(DEVICE)
    model.config.use_cache = False

    if hasattr(model, "set_use_kernels"):
        model.set_use_kernels(False)
        log("Kernels: disabled (pure PyTorch) ✓")

    log(f"Model loaded in {time.time()-t0:.1f}s  "
        f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
    gpu_health("after_load")

    # Verify structure
    sub_norm_count = sum(1 for n, _ in model.named_parameters() if "sub_norm" in n)
    assert sub_norm_count == 60, f"Expected 60 sub_norm, got {sub_norm_count}"
    assert len(model.model.layers) == 30
    assert hasattr(model.model, "rotary_emb")
    log(f"Structure verified: sub_norm=60 ✓  layers=30 ✓  rotary_emb ✓")

    # ── Patch RoPE ────────────────────────────────────────────────────────────
    patch_rope(model, yarn_t)

    # ── Datasets ──────────────────────────────────────────────────────────────
    context_chunks = build_context_chunks(tokenizer)
    assert len(context_chunks) >= 10

    sft_items = build_sft_dataset(tokenizer)
    assert len(sft_items) >= 5
    log(f"Datasets ready: {len(context_chunks)} context chunks, "
        f"{len(sft_items)} SFT items")

    # ── Baseline PPL ──────────────────────────────────────────────────────────
    if RESUME_STEP == 0:
        log("Computing baseline PPL...")
        ppl_4k_base, loss_4k_base = compute_ppl(model, context_chunks, 4096)
        ppl_8k_base, loss_8k_base = compute_ppl(model, context_chunks, 8192)
        log(f"  Baseline PPL@4K={ppl_4k_base:.3f}  PPL@8K={ppl_8k_base:.3f}")
        log(f"  8K degradation: {ppl_8k_base - ppl_4k_base:+.3f} PPL points")
    else:
        log(f"Resuming from step {RESUME_STEP} — skipping baseline PPL")
        ppl_4k_base = loss_4k_base = 0.0
        ppl_8k_base = loss_8k_base = 0.0

    # ── Step 0 sub_norm snapshot ───────────────────────────────────────────────
    if RESUME_STEP == 0:
        snapshot_subnorm(model, 0)

    # ── Initial freeze state ───────────────────────────────────────────────────
    if RESUME_STEP < PHASE2_START:
        log("Phase 1: freezing layers 25-29 sub_norm")
        set_frozen(model, freeze_high_gain=True)
    else:
        log("Resuming: all params unfrozen")
        set_frozen(model, freeze_high_gain=False)

    model.gradient_checkpointing_enable()
    log("Gradient checkpointing: ON")

    # ── Initial optimizer ─────────────────────────────────────────────────────
    if RESUME_STEP < PHASE2_START:
        optimizer, scheduler = build_optimizer_scheduler(
            model, LR_PHASE1, TOTAL_STEPS, WARMUP_STEPS, LR_END
        )
        for _ in range(RESUME_STEP):
            scheduler.step()
    elif RESUME_STEP < PHASE3_START:
        optimizer, scheduler = build_optimizer_scheduler(
            model, LR_PHASE2, PHASE3_START - PHASE2_START, 0, LR_END
        )
        for _ in range(RESUME_STEP - PHASE2_START):
            scheduler.step()
    else:
        optimizer, scheduler = build_optimizer_scheduler(
            model, LR_PHASE3, TOTAL_STEPS - PHASE3_START, 0, LR_END
        )
        for _ in range(RESUME_STEP - PHASE3_START):
            scheduler.step()

    log(f"Optimizer ready. lr={scheduler.get_last_lr()[0]:.2e}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "train_log.csv"
    if RESUME_STEP == 0:
        init_csv(csv_path)

    log_data = {
        "step": [], "loss_4k": [], "loss_8k": [], "phase": [],
        "baseline_4k": loss_4k_base, "baseline_8k": loss_8k_base,
    }

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    t_start = time.time()

    log("=" * 70)
    log(f"TRAINING — steps {RESUME_STEP}–{TOTAL_STEPS-1}")
    log("=" * 70)

    for step in range(RESUME_STEP, TOTAL_STEPS):

        current_phase = (1 if step < PHASE2_START
                         else 2 if step < PHASE3_START
                         else 3)

        # ── Phase 2 transition ────────────────────────────────────────────────
        if step == PHASE2_START and RESUME_STEP < PHASE2_START:
            log(f"\n→ Step {step}: Phase 2 — unfreezing layers 25-29 sub_norm")
            set_frozen(model, freeze_high_gain=False)
            optimizer, scheduler = build_optimizer_scheduler(
                model, LR_PHASE2, PHASE3_START - PHASE2_START, 0, LR_END
            )
            optimizer.zero_grad()
            gpu_health("phase2_start")

        # ── Phase 3 transition ────────────────────────────────────────────────
        if step == PHASE3_START:
            log(f"\n→ Step {step}: Phase 3 — Instruction finetuning (SFT)")
            log(f"  Loss will be computed only on assistant turns")
            set_frozen(model, freeze_high_gain=False)
            optimizer, scheduler = build_optimizer_scheduler(
                model, LR_PHASE3, TOTAL_STEPS - PHASE3_START, 0, LR_END
            )
            optimizer.zero_grad()
            gpu_health("phase3_start")

        # ── Sub_norm snapshot ─────────────────────────────────────────────────
        if step in SUBNORM_STEPS and step > 0:
            model.eval()
            snapshot_subnorm(model, step)
            model.train()

        # ── Forward pass ──────────────────────────────────────────────────────
        if current_phase < 3:
            # Context extension — train on long document chunks
            chunk = random.choice(context_chunks).to(DEVICE)
            ids   = chunk.unsqueeze(0)

            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model(ids, labels=ids)
            loss = out.loss / GRAD_ACCUM

        else:
            # SFT — train only on assistant tokens
            ids_raw, labels_raw = random.choice(sft_items)
            ids    = ids_raw.unsqueeze(0).to(DEVICE)
            labels = labels_raw.unsqueeze(0).to(DEVICE)

            with torch.autocast(device_type="cuda", dtype=DTYPE):
                out = model(ids, labels=labels)
            loss = out.loss / GRAD_ACCUM

        loss.backward()

        # ── Grad step ─────────────────────────────────────────────────────────
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ───────────────────────────────────────────────────────────
        if step % LOG_EVERY == 0:
            elapsed  = (time.time() - t_start) / 60
            eta_min  = (elapsed / max(step - RESUME_STEP, 1)) * \
                       (TOTAL_STEPS - step)
            cur_lr   = scheduler.get_last_lr()[0]
            vram, _  = gpu_health(f"step_{step}")

            model.eval()
            with torch.no_grad():
                _, l4k = compute_ppl(model, context_chunks, 4096,  n_samples=5)
                _, l8k = compute_ppl(model, context_chunks, 8192, n_samples=5)
            model.train()

            log_data["step"].append(step)
            log_data["loss_4k"].append(l4k)
            log_data["loss_8k"].append(l8k)
            log_data["phase"].append(current_phase)

            append_csv(csv_path, [step, l4k, l8k, l8k-l4k,
                                   cur_lr, current_phase,
                                   f"{elapsed:.1f}", f"{vram:.1f}"])

            log(f"Step {step:>5} | loss@4K={l4k:.4f} | loss@8K={l8k:.4f} | "
                f"gap={l8k-l4k:+.4f} | lr={cur_lr:.2e} | "
                f"phase={current_phase} | ETA={eta_min:.0f}min")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            ckpt = OUTPUT_DIR / "checkpoints" / f"step_{step+1:05d}"
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            shutil.copy(str(YARN_NPY), str(ckpt / "yarn_inv_freq.npy"))
            log(f"  → Checkpoint: {ckpt}")
            save_loss_curve(log_data)
            gpu_health(f"ckpt_{step+1}")

            # Keep only 2 most recent checkpoints
            all_ckpts = sorted((OUTPUT_DIR / "checkpoints").iterdir())
            for old in all_ckpts[:-2]:
                shutil.rmtree(str(old))
                log(f"  → Removed: {old.name}")

    # ── Final ─────────────────────────────────────────────────────────────────
    model.eval()
    snapshot_subnorm(model, TOTAL_STEPS)

    log("\nFinal evaluation...")
    ppl_4k_after, _ = compute_ppl(model, context_chunks, 4096, n_samples=20)
    ppl_8k_after, _ = compute_ppl(model, context_chunks, 8192, n_samples=20)

    log("=" * 70)
    log("RESULTS")
    log(f"  PPL@4K: {ppl_4k_base:.3f} → {ppl_4k_after:.3f}  "
        f"(Δ {ppl_4k_after - ppl_4k_base:+.3f})")
    log(f"  PPL@8K: {ppl_8k_base:.3f} → {ppl_8k_after:.3f}  "
        f"(Δ {ppl_8k_after - ppl_8k_base:+.3f})")

    delta_8k = ppl_8k_after - ppl_8k_base
    delta_4k = ppl_4k_after - ppl_4k_base
    if delta_8k < -1.0 and delta_4k < 1.0:
        verdict = "✓ SUCCESS"
    elif delta_8k < 0:
        verdict = "~ PARTIAL"
    else:
        verdict = "? INCONCLUSIVE"
    log(verdict)
    log("=" * 70)

    final = OUTPUT_DIR / "final_model"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    shutil.copy(str(YARN_NPY), str(final / "yarn_inv_freq.npy"))

    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump({
            "ppl_4k_base": ppl_4k_base,   "ppl_4k_after": ppl_4k_after,
            "ppl_8k_base": ppl_8k_base,   "ppl_8k_after": ppl_8k_after,
            "delta_4k": delta_4k,          "delta_8k": delta_8k,
            "verdict": verdict,
            "total_steps": TOTAL_STEPS,
            "phase1_end": PHASE2_START,
            "phase2_end": PHASE3_START,
            "phase3_end": TOTAL_STEPS,
            "sft_dataset": "HuggingFaceH4/ultrachat_200k",
            "context_dataset": "sedthh/gutenberg_english",
            "yarn_bands": f"{SCALE_ZONE[0]}-{SCALE_ZONE[1]}",
            "scale_factor": SCALE_FACTOR,
        }, f, indent=2)

    save_loss_curve(log_data)
    log(f"Final model: {final}")
    log(f"Results:     {OUTPUT_DIR / 'results_summary.json'}")
    log("Done.")


if __name__ == "__main__":
    main()