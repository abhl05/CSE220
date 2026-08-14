"""
CSE 220 — Online 6: Parseval's Theorem in 2D (Image Energy Conservation)
=========================================================================
Verifies that Integral Integral |I(x,y)|² dx dy = Integral Integral |F(u,v)|² du dv.
Uses separable trapezoidal integration only — no FFT.
"""
import numpy as np
import matplotlib.pyplot as plt
from cft_edge_detector import ContinuousImage, CFT2D


def parseval_2d(image_path):
    """
    Verify 2D Parseval's theorem for a given image.
    
    Parameters
    ----------
    image_path : str
        Path to input image (e.g., 'pikachu.png')
    
    Returns
    -------
    dict with LHS, RHS, relative_error
    """
    # Step 1: Load image and compute CFT
    img = ContinuousImage(image_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    
    I = img.image
    x = img.x
    y = img.y
    u = cft2d.u
    v = cft2d.v
    
    # Step 2: Compute time-domain (spatial) energy
    # Integral over x first, then y
    # np.trapezoid(I**2, x, axis=1) gives integral over x for each y
    # Then np.trapezoid(..., y) integrates over y
    energy_spatial = np.trapezoid(
        np.trapezoid(I ** 2, x, axis=1),
        y
    )
    
    # Step 3: Compute frequency-domain energy
    # |F(u,v)|² = real² + imag²
    F_squared = real ** 2 + imag ** 2
    energy_freq = np.trapezoid(
        np.trapezoid(F_squared, u, axis=1),
        v
    )
    
    # Step 4: Report relative error
    rel_error = np.abs(energy_spatial - energy_freq) / np.abs(energy_spatial)
    
    print(f"Image: {image_path}")
    print(f"  Spatial energy:   {energy_spatial:.10f}")
    print(f"  Frequency energy: {energy_freq:.10f}")
    print(f"  Relative error:   {rel_error:.2e}")
    print(f"  Grid size:        {I.shape[0]} x {I.shape[1]}")
    
    # Comment on why error should be small
    print("\\n--- Why the error is small ---")
    print("Parseval's theorem is exact for the continuous Fourier transform.")
    print("The small error here comes from:")
    print("  1. Finite grid discretization (trapezoidal rule approximation)")
    print("  2. Finite frequency range (Nyquist-limited u,v axes)")
    print("  3. On a fine grid, the trapezoidal rule converges rapidly,")
    print("     so the error should be O(dx²) or smaller.")
    
    # Optional: visualize the energy distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    im = ax.imshow(I ** 2, extent=[x[0], x[-1], y[0], y[-1]], cmap='hot')
    ax.set_title('|I(x,y)|² (Spatial Energy Density)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    im = ax.imshow(np.log(1 + F_squared), 
                   extent=[u[0], u[-1], v[0], v[-1]], cmap='hot', origin='lower')
    ax.set_title('log(1+|F(u,v)|²) (Frequency Energy Density)')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('online6_parseval_2d.png', dpi=150)
    plt.show()
    
    return {
        'energy_spatial': energy_spatial,
        'energy_freq': energy_freq,
        'rel_error': rel_error,
        'shape': I.shape
    }


if __name__ == "__main__":
    result = parseval_2d("D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task2\\pikachu.png")