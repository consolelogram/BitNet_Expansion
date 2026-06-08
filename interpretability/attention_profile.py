# attention_profile.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

model_path = r"D:\Code\Bitnet\models\bitnet-hf"

print("Loading tokenizer and model (this takes ~2-3 min on CPU)...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    attn_implementation="eager"
)
model.eval()
print("Model loaded.")

# test at these sequence lengths
# keep short for CPU — 512/1024/2048 are fast, 4096 will take ~15 min
test_lengths = [128, 256, 512, 1024, 2048]
# add 4096 only if you have time — it will be slow
# test_lengths.append(4096)

results = {}

for seq_len in test_lengths:
    print(f"\nTesting seq_len={seq_len}...")

    # synthetic sequence — repeating token pattern
    input_ids = torch.randint(100, 32000, (1, seq_len))

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            return_dict=True
        )

    attentions = outputs.attentions  # tuple of (batch, heads, seq, seq) per layer

    layer_stats = []
    for layer_idx, attn in enumerate(attentions):
        attn_np = attn.squeeze(0).numpy()  # (heads, seq, seq)

        # entropy of attention distribution per head
        # high entropy = diffuse attention (bad, model is lost)
        # low entropy = focused attention (good, model knows what to attend to)
        eps = 1e-9
        entropy = -np.sum(attn_np * np.log(attn_np + eps), axis=-1)  # (heads, seq)
        mean_entropy = entropy.mean()

        # max attention weight — drops when model is confused
        max_attn = attn_np.max(axis=-1).mean()

        # sink token ratio — how much attention goes to token 0
        # excessive sink = attention collapse
        sink_ratio = attn_np[:, :, 0].mean()

        layer_stats.append({
            'layer': layer_idx,
            'mean_entropy': float(mean_entropy),
            'max_attn': float(max_attn),
            'sink_ratio': float(sink_ratio)
        })

    results[seq_len] = layer_stats
    print(f"  Done. Layer 14 entropy: {layer_stats[14]['mean_entropy']:.4f}, "
          f"Layer 29 entropy: {layer_stats[29]['mean_entropy']:.4f}")
    print(f"  Layer 29 sink ratio: {layer_stats[29]['sink_ratio']:.4f}")

# save raw results
with open('attention_profile_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to attention_profile_results.json")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
layers = list(range(30))
colors = plt.cm.viridis(np.linspace(0, 1, len(test_lengths)))

for i, seq_len in enumerate(test_lengths):
    stats = results[seq_len]
    entropies  = [s['mean_entropy'] for s in stats]
    max_attns  = [s['max_attn']     for s in stats]
    sink_ratios= [s['sink_ratio']   for s in stats]

    axes[0].plot(layers, entropies,   color=colors[i], label=f'{seq_len}', linewidth=1.8)
    axes[1].plot(layers, max_attns,   color=colors[i], label=f'{seq_len}', linewidth=1.8)
    axes[2].plot(layers, sink_ratios, color=colors[i], label=f'{seq_len}', linewidth=1.8)

for ax, title, ylabel in zip(axes,
    ['Attention entropy per layer', 'Max attention weight per layer', 'Sink token ratio per layer'],
    ['Mean entropy (nats)', 'Mean max weight', 'Fraction to token 0']):
    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title='seq len', fontsize=9)
    ax.axvline(x=14, color='red',    linestyle='--', alpha=0.5, linewidth=1, label='layer 14')
    ax.axvline(x=29, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='layer 29')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('attention_profile.png', dpi=150)
plt.show()
print("Plot saved to attention_profile.png")