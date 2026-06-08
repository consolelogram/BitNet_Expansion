"""
yarn_rope.py
YaRN selective RoPE interpolation for BitNet b1.58 context extension (4K -> 8K).

Forensic audit findings used here:
  - rope_theta    = 500000
  - head_dim      = 128  -> 64 bands total
  - original_max  = 4096
  - target_max    = 8192
  - Scale zone    = bands 32-42  (wavelength 4K-15K tokens)
  - Scale factor  = 2.0x  (wavelength doubled = frequency halved)
  - Bands 0-31    = SATURATED  (wavelength < 4K) -> DO NOT TOUCH
  - Bands 43-63   = SAFE       (wavelength > 15K) -> DO NOT TOUCH

Output: yarn_inv_freq.npy  (64 float64 values, ready to inject into model)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (from forensic audit — do not change without re-running audit)
# ─────────────────────────────────────────────────────────────────────────────
ROPE_THETA   = 500_000
HEAD_DIM     = 128
ORIGINAL_MAX = 4_096
TARGET_MAX   = 8_192
SCALE_FACTOR = 2.0          # wavelength multiplier for bands in SCALE_ZONE
SCALE_ZONE   = (32, 42)     # inclusive band indices to scale


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Standard baseline inv_freq
# ─────────────────────────────────────────────────────────────────────────────
def compute_base_inv_freq(theta: float, head_dim: int) -> np.ndarray:
    """
    inv_freq[i] = 1 / (theta ^ (2i / head_dim))   for i in 0 .. head_dim//2 - 1
    Returns shape (head_dim // 2,)  ->  64 values for head_dim=128
    """
    i = np.arange(0, head_dim, 2, dtype=np.float64)          # [0, 2, 4, ..., 126]
    inv_freq = 1.0 / (theta ** (i / head_dim))
    return inv_freq                                            # shape (64,)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Wavelength per band
# ─────────────────────────────────────────────────────────────────────────────
def compute_wavelengths(inv_freq: np.ndarray) -> np.ndarray:
    """
    wavelength[i] = 2π / inv_freq[i]
    Tells you how many tokens band i needs to complete one full rotation.
    """
    return (2 * np.pi) / inv_freq                             # shape (64,)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Build per-band scale multiplier array
# ─────────────────────────────────────────────────────────────────────────────
def build_scale_array(n_bands: int, scale_zone: tuple, scale_factor: float) -> np.ndarray:
    """
    All bands = 1.0 except SCALE_ZONE bands which get 1/scale_factor.
    We divide inv_freq (not multiply) because:
        lower frequency -> longer wavelength -> safe at more tokens.
    """
    scale_array = np.ones(n_bands, dtype=np.float64)
    lo, hi = scale_zone
    scale_array[lo : hi + 1] = 1.0 / scale_factor            # frequency halved = wavelength doubled
    return scale_array


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Apply multipliers -> YaRN inv_freq
# ─────────────────────────────────────────────────────────────────────────────
def compute_yarn_inv_freq(
    theta: float,
    head_dim: int,
    scale_zone: tuple,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (base_inv_freq, yarn_inv_freq) both shape (head_dim//2,).
    """
    base_inv_freq = compute_base_inv_freq(theta, head_dim)
    n_bands       = head_dim // 2
    scale_array   = build_scale_array(n_bands, scale_zone, scale_factor)
    yarn_inv_freq = base_inv_freq * scale_array
    return base_inv_freq, yarn_inv_freq


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Validate
# ─────────────────────────────────────────────────────────────────────────────
def validate(
    base_inv_freq: np.ndarray,
    yarn_inv_freq: np.ndarray,
    original_max: int,
    target_max: int,
    scale_zone: tuple,
    scale_factor: float,
) -> bool:
    base_wl = compute_wavelengths(base_inv_freq)
    yarn_wl = compute_wavelengths(yarn_inv_freq)
    lo, hi  = scale_zone
    n_bands = len(base_inv_freq)

    print("\n" + "═" * 90)
    print(f"{'Band':>5}  {'Base λ (tokens)':>16}  {'YaRN λ (tokens)':>16}  "
          f"{'Ratio':>7}  {'Status at 4K':>14}  {'Status at 8K':>14}  {'Zone':>8}")
    print("─" * 90)

    all_ok = True
    for band in range(n_bands):
        bwl  = base_wl[band]
        ywl  = yarn_wl[band]
        ratio = ywl / bwl

        # classify base status at original_max
        if bwl < original_max:
            base_status = "SATURATED"
        elif bwl < target_max * 3:
            base_status = "TRANSITION"
        else:
            base_status = "SAFE"

        # classify yarn status at target_max
        if ywl < target_max:
            yarn_status = "SATURATED ⚠"
        elif ywl < target_max * 3:
            yarn_status = "TRANSITION"
        else:
            yarn_status = "SAFE ✓"

        in_zone = lo <= band <= hi
        zone_tag = "← SCALED" if in_zone else ""

        print(f"{band:>5}  {bwl:>16.1f}  {ywl:>16.1f}  {ratio:>7.3f}  "
              f"{base_status:>14}  {yarn_status:>14}  {zone_tag}")

        # assertions
        if in_zone:
            expected_ratio = scale_factor   # wavelength should have doubled
            if not np.isclose(ratio, expected_ratio, rtol=1e-6):
                print(f"  !! FAIL band {band}: ratio={ratio:.6f}, expected {expected_ratio}")
                all_ok = False
        else:
            if not np.isclose(ratio, 1.0, rtol=1e-6):
                print(f"  !! FAIL band {band}: ratio={ratio:.6f}, expected 1.0 (untouched)")
                all_ok = False

    print("═" * 90)

    # summary stats
    scaled_bands  = list(range(lo, hi + 1))
    touched_wl    = yarn_wl[scaled_bands]
    print(f"\nScale zone  : bands {lo}–{hi}  ({len(scaled_bands)} bands)")
    print(f"Scale factor: {scale_factor}x wavelength  (freq × {1/scale_factor})")
    print(f"Scaled λ range: {touched_wl.min():.1f} – {touched_wl.max():.1f} tokens")
    print(f"Original max: {original_max}   Target max: {target_max}")
    print(f"\nValidation  : {'✓ ALL ASSERTIONS PASSED' if all_ok else '✗ FAILURES DETECTED'}")

    # sanity: no band in scale zone should still be saturated at target_max
    still_saturated = [b for b in scaled_bands if yarn_wl[b] < target_max]
    if still_saturated:
        print(f"  !! Bands still saturated at {target_max}: {still_saturated}")
        all_ok = False

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Save
# ─────────────────────────────────────────────────────────────────────────────
def save(yarn_inv_freq: np.ndarray, path: str = "yarn_inv_freq.npy") -> None:
    np.save(path, yarn_inv_freq)
    loaded = np.load(path)
    assert np.array_equal(loaded, yarn_inv_freq), "Save/load roundtrip failed"
    print(f"\nSaved  : {path}")
    print(f"Shape  : {loaded.shape}")
    print(f"dtype  : {loaded.dtype}")
    print(f"min    : {loaded.min():.6e}")
    print(f"max    : {loaded.max():.6e}")
    print(f"\nLoad in Colab with:")
    print(f"  import numpy as np, torch")
    print(f"  yarn_inv_freq = torch.tensor(np.load('yarn_inv_freq.npy'), dtype=torch.float32)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("YaRN RoPE Selective Interpolation")
    print(f"  theta={ROPE_THETA}  head_dim={HEAD_DIM}  "
          f"{ORIGINAL_MAX} -> {TARGET_MAX} tokens  "
          f"scale={SCALE_FACTOR}x on bands {SCALE_ZONE[0]}-{SCALE_ZONE[1]}")

    base_inv_freq, yarn_inv_freq = compute_yarn_inv_freq(
        theta        = ROPE_THETA,
        head_dim     = HEAD_DIM,
        scale_zone   = SCALE_ZONE,
        scale_factor = SCALE_FACTOR,
    )

    ok = validate(
        base_inv_freq = base_inv_freq,
        yarn_inv_freq = yarn_inv_freq,
        original_max  = ORIGINAL_MAX,
        target_max    = TARGET_MAX,
        scale_zone    = SCALE_ZONE,
        scale_factor  = SCALE_FACTOR,
    )

    if ok:
        save(yarn_inv_freq, path="yarn_inv_freq.npy")
    else:
        print("\nNot saving — fix validation failures first.")