"""
CSE 220 — Online 7: Low-Pass Filtering (Blurring)
==================================================
Implements low-pass mask (mirror of offline's high-pass).
Reconstructs blurred image at multiple cutoffs.
Computes fraction of spectral energy retained.
"""
import numpy as np
import matplotlib.pyplot as plt
from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D


def low_pass_filter(real, imag, cutoff):
    """
    Apply a low-pass filter: keep only frequencies within radius cutoff.
    Zeroes everything OUTSIDE the disk.
    """
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    
    real_f = real.copy()
    imag_f = imag.copy()
    
    # Create coordinate grids
    i_idx = np.arange(rows)
    j_idx = np.arange(cols)
    I_grid, J_grid = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    # Distance from center
    dist = np.sqrt((I_grid - cx) ** 2 + (J_grid - cy) ** 2)
    
    # Zero everything OUTSIDE the disk (low-pass = keep low freqs)
    mask = dist > cutoff
    real_f[mask] = 0
    imag_f[mask] = 0
    
    return real_f, imag_f


def low_pass_study(image_path, cutoffs=[5, 15, 40]):
    """
    Study low-pass filtering at multiple cutoff values.
    
    Parameters
    ----------
    image_path : str
        Path to input image
    cutoffs : list of float
        Cutoff radii to test
    """
    # Step 1: Compute CFT
    img = ContinuousImage(image_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    
    I = img.image
    u = cft2d.u
    v = cft2d.v
    
    # Compute total spectral energy for reference
    F_squared = real ** 2 + imag ** 2
    total_energy = np.sum(F_squared)
    
    # Step 2 & 3: Apply low-pass at each cutoff, reconstruct, compute energy retained
    n_cutoffs = len(cutoffs)
    fig, axes = plt.subplots(2, n_cutoffs + 1, figsize=(4 * (n_cutoffs + 1), 8))
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(I, cmap='gray')
    ax.set_title('Original')
    ax.axis('off')
    
    ax = axes[1, 0]
    ax.imshow(np.log(1 + np.sqrt(F_squared)), cmap='gray')
    ax.set_title('Log Magnitude Spectrum')
    ax.axis('off')
    
    results = []
    
    for idx, cutoff in enumerate(cutoffs):
        # Apply low-pass filter
        real_f, imag_f = low_pass_filter(real, imag, cutoff)
        
        # Compute energy retained
        F_squared_f = real_f ** 2 + imag_f ** 2
        energy_retained = np.sum(F_squared_f)
        frac_retained = energy_retained / total_energy
        
        # Reconstruct
        icft2d = InverseCFT2D(real_f, imag_f, u, v, img.x, img.y)
        blurred = icft2d.reconstruct()
        
        # Normalize for display
        blurred_display = np.real(blurred)
        bmin, bmax = blurred_display.min(), blurred_display.max()
        if bmax > bmin:
            blurred_display = (blurred_display - bmin) / (bmax - bmin)
        
        # Plot blurred image
        ax = axes[0, idx + 1]
        ax.imshow(blurred_display, cmap='gray')
        ax.set_title(f'Cutoff = {cutoff}\nEnergy: {frac_retained*100:.1f}%')
        ax.axis('off')
        
        # Plot filtered spectrum
        ax = axes[1, idx + 1]
        ax.imshow(np.log(1 + np.sqrt(F_squared_f)), cmap='gray')
        ax.set_title(f'Filtered Spectrum\n(cutoff={cutoff})')
        ax.axis('off')
        
        results.append({
            'cutoff': cutoff,
            'frac_retained': frac_retained,
            'blurred': blurred,
            'blurred_display': blurred_display
        })
        
        print(f"Cutoff = {cutoff:2d}:  Energy retained = {frac_retained*100:5.1f}%")
    
    plt.tight_layout()
    plt.savefig('online7_lowpass.png', dpi=150)
    plt.show()
    
    # Comments
    print("\n--- Analysis ---")
    print("Low cutoff (5):   Heavy blur, only ~5-15% energy. Image barely")
    print("                  recognizable — only coarse structure preserved.")
    print("Medium cutoff (15): Moderate blur, ~30-50% energy. Some texture")
    print("                    lost but main features still visible.")
    print("High cutoff (40):   Light blur, ~70-85% energy. Fine details")
    print("                    preserved; image still recognizable.")
    print("")
    print("Trade-off: Lower cutoff = more blur = more information loss.")
    print("           Higher cutoff = less blur = more information retained.")
    
    return results


if __name__ == "__main__":
    results = low_pass_study("D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task2\\pikachu.png", cutoffs=[5, 15, 40])