"""
CSE 220 — Online 9: Reconstruction Sanity Check + Band-Pass Filter
=================================================================
1. Validates that InverseCFT2D.reconstruct() recovers the original image
   when no filter is applied.
2. Implements a band-pass filter (annulus) and describes what it isolates.
"""
import numpy as np
import matplotlib.pyplot as plt
from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D


def band_pass_filter(real, imag, r1, r2):
    """
    Apply a band-pass filter: keep only frequencies in annulus r1 <= r <= r2.
    Zeroes everything inside r1 and outside r2.
    """
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    
    real_f = real.copy()
    imag_f = imag.copy()
    
    # Create coordinate grids
    i_idx = np.arange(rows)
    j_idx = np.arange(cols)
    I_grid, J_grid = np.meshgrid(i_idx, j_idx, indexing='ij')
    dist = np.sqrt((I_grid - cx) ** 2 + (J_grid - cy) ** 2)
    
    # Zero everything NOT in the annulus
    mask = (dist < r1) | (dist > r2)
    real_f[mask] = 0
    imag_f[mask] = 0
    
    return real_f, imag_f


def reconstruction_sanity_and_bandpass(image_path, r1=10, r2=30):
    """
    Task 1: Reconstruction sanity check (no filter).
    Task 2: Band-pass filter reconstruction.
    
    Parameters
    ----------
    image_path : str
        Path to input image
    r1, r2 : float
        Inner and outer radii for band-pass filter
    """
    # Step 1: Compute CFT
    img = ContinuousImage(image_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    
    I_orig = img.image
    u = cft2d.u
    v = cft2d.v
    
    # ============================================================
    # TASK 1: Reconstruction Sanity Check (no filter)
    # ============================================================
    icft_sanity = InverseCFT2D(real, imag, u, v, img.x, img.y)
    I_reconstructed = icft_sanity.reconstruct()
    
    # Compute MSE between reconstructed and original
    mse = np.mean((I_reconstructed - I_orig) ** 2)
    max_error = np.max(np.abs(I_reconstructed - I_orig))
    
    print("=" * 60)
    print("TASK 1: Reconstruction Sanity Check")
    print("=" * 60)
    print(f"  MSE (reconstructed vs original):     {mse:.2e}")
    print(f"  Max absolute error:                  {max_error:.2e}")
    print(f"  Sanity check: {'PASSED ✓' if mse < 1e-6 else 'FAILED ✗'}")
    print(f"  (Small MSE confirms compute_cft and reconstruct are correct inverses)")
    
    # ============================================================
    # TASK 2: Band-Pass Filter
    # ============================================================
    real_bp, imag_bp = band_pass_filter(real, imag, r1, r2)
    
    icft_bp = InverseCFT2D(real_bp, imag_bp, u, v, img.x, img.y)
    I_bandpass = icft_bp.reconstruct()
    
    # Also compute high-pass for comparison
    from cft_edge_detector import FrequencyFilter
    filt = FrequencyFilter()
    real_hp, imag_hp = filt.high_pass(real, imag, cutoff=r1)
    icft_hp = InverseCFT2D(real_hp, imag_hp, u, v, img.x, img.y)
    I_highpass = icft_hp.reconstruct()
    
    print("\n" + "=" * 60)
    print("TASK 2: Band-Pass vs High-Pass Comparison")
    print("=" * 60)
    print(f"  Band-pass radii: r1={r1}, r2={r2}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Spatial domain images
    ax = axes[0, 0]
    ax.imshow(I_orig, cmap='gray')
    ax.set_title('Original')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(np.real(I_bandpass), cmap='gray')
    ax.set_title(f'Band-Pass\n(r1={r1}, r2={r2})')
    ax.axis('off')
    
    ax = axes[0, 2]
    edge_map = np.abs(np.real(I_highpass))
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max()
    ax.imshow(1 - edge_map, cmap='gray')
    ax.set_title(f'High-Pass\n(cutoff={r1})')
    ax.axis('off')
    
    # Row 2: Frequency domain spectra
    F_orig_sq = real**2 + imag**2
    F_bp_sq = real_bp**2 + imag_bp**2
    F_hp_sq = real_hp**2 + imag_hp**2
    
    ax = axes[1, 0]
    ax.imshow(np.log(1 + np.sqrt(F_orig_sq)), cmap='gray')
    ax.set_title('Original Spectrum')
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(np.log(1 + np.sqrt(F_bp_sq)), cmap='gray')
    ax.set_title('Band-Pass Spectrum\n(annulus mask)')
    ax.axis('off')
    
    ax = axes[1, 2]
    ax.imshow(np.log(1 + np.sqrt(F_hp_sq)), cmap='gray')
    ax.set_title('High-Pass Spectrum\n(disk mask)')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('online9_sanity_bandpass.png', dpi=150)
    plt.show()
    
    # Description
    print("\n--- What Band-Pass Isolates ---")
    print("A band-pass filter (annulus r1 <= r <= r2) isolates MID-SCALE")
    print("spatial features — neither the very coarse (low-frequency) structure")
    print("nor the very fine (high-frequency) details like sharp edges.")
    print("")
    print("Comparison:")
    print("  High-pass (cutoff=r1):  Keeps ONLY edges and fine texture.")
    print("                          Removes all smooth variation.")
    print("  Band-pass (r1, r2):    Keeps a specific SCALE of texture.")
    print("                          Shows mid-frequency patterns, contours,")
    print("                          and medium-detail features.")
    print("  Low-pass (cutoff=r2):   Keeps ONLY smooth, coarse structure.")
    print("                          Blurs away all fine detail.")
    print("")
    print("Think of it like a 'Goldilocks' filter — not too coarse, not too fine,")
    print("but just the right spatial scale for the features you want to see.")
    
    return {
        'mse_sanity': mse,
        'max_error_sanity': max_error,
        'I_reconstructed': I_reconstructed,
        'I_bandpass': I_bandpass
    }


if __name__ == "__main__":
    result = reconstruction_sanity_and_bandpass("D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task2\\pikachu.png", r1=10, r2=30)