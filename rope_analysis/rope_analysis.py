import numpy as np
import matplotlib.pyplot as plt
import json

cfg = json.load(open(r"D:\Code\Bitnet\models\bitnet-hf\config.json"))

theta = cfg["rope_theta"]                  # 500000.0
head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
max_seq = cfg["max_position_embeddings"]   # 4096
target_4k = 4096
target_8k = 8192
target_12k = 12288

print(f"theta: {theta}")
print(f"head_dim: {head_dim}")
print(f"num_heads: {cfg['num_attention_heads']}")

dims = np.arange(0, head_dim, 2)
freqs = 1.0 / (theta ** (dims / head_dim))
wavelengths = 2 * np.pi / freqs

sat_4k  = target_4k  / wavelengths
sat_8k  = target_8k  / wavelengths
sat_12k = target_12k / wavelengths

print(f"\nTotal frequency bands: {len(freqs)}")
print(f"\n--- Saturation at 4K ---")
print(f"Bands >100% saturated (wrapping): {(sat_4k > 1.0).sum()}")
print(f"Bands >80% saturated (at risk):   {(sat_4k > 0.8).sum()}")
print(f"Bands <10% saturated (very safe): {(sat_4k < 0.1).sum()}")

print(f"\n--- Saturation at 8K ---")
print(f"Bands >100% saturated (wrapping): {(sat_8k > 1.0).sum()}")
print(f"Bands >80% saturated (at risk):   {(sat_8k > 0.8).sum()}")

print(f"\n--- Saturation at 12K ---")
print(f"Bands >100% saturated (wrapping): {(sat_12k > 1.0).sum()}")
print(f"Bands >80% saturated (at risk):   {(sat_12k > 0.8).sum()}")

print(f"\n--- Interpolation scale factors ---")
print(f"Linear PI for 8K:  {target_8k / max_seq:.2f}x")
print(f"Linear PI for 12K: {target_12k / max_seq:.2f}x")
print(f"\nLowest freq wavelength:  {wavelengths[-1]:.1f} tokens")
print(f"Highest freq wavelength: {wavelengths[0]:.1f} tokens")

print(f"\n--- Per-band detail (dangerous bands only) ---")
print(f"{'Band':>5} {'Wavelength':>12} {'Sat@4K':>8} {'Sat@8K':>8} {'Sat@12K':>9} {'Status'}")
for i, (w, s4, s8, s12) in enumerate(zip(wavelengths, sat_4k, sat_8k, sat_12k)):
    if s8 > 0.5:
        status = "WRAPPING" if s4 > 1.0 else "AT RISK" if s4 > 0.8 else "WATCH"
        print(f"{i:>5} {w:>12.1f} {s4:>8.2%} {s8:>8.2%} {s12:>9.2%}  {status}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].semilogy(range(len(wavelengths)), wavelengths, color='steelblue', linewidth=2)
axes[0].axhline(y=target_4k,  color='green',  linestyle='--', linewidth=1.2, label='4K (current)')
axes[0].axhline(y=target_8k,  color='orange', linestyle='--', linewidth=1.2, label='8K (target)')
axes[0].axhline(y=target_12k, color='red',    linestyle='--', linewidth=1.2, label='12K (stretch)')
axes[0].set_xlabel('Dimension pair index')
axes[0].set_ylabel('Wavelength (tokens, log scale)')
axes[0].set_title('RoPE wavelengths per band')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

x = range(len(wavelengths))
axes[1].bar(x, np.minimum(sat_4k, 1.5),  alpha=0.7, label='4K',  color='steelblue')
axes[1].bar(x, np.minimum(sat_8k, 1.5),  alpha=0.4, label='8K',  color='orange')
axes[1].bar(x, np.minimum(sat_12k, 1.5), alpha=0.3, label='12K', color='red')
axes[1].axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Wrap threshold')
axes[1].set_xlabel('Dimension pair index')
axes[1].set_ylabel('Saturation ratio')
axes[1].set_title('Band saturation across target lengths')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rope_analysis.png', dpi=150)
plt.show()
print("\nPlot saved to rope_analysis.png")