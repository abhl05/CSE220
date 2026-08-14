"""
CSE 220 — Online 10: Frequency-Domain Denoising Using Your Own CFT2D
======================================================================
Mission-style: Identify and remove periodic noise peaks in the spectrum.
Uses trapezoidal CFT2D/InverseCFT2D (no np.fft anywhere).
"""
import numpy as np
import matplotlib.pyplot as plt
from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D


def denoise_image(image_path, noise_peaks=None):
    """
    Denoise an image by removing specific frequency peaks from its spectrum.
    
    Parameters
    ----------
    image_path : str
        Path to noisy image (should be small for O(N^3) runtime)
    noise_peaks : list of tuples, optional
        Each tuple is ((u_idx, v_idx), radius) defining a region to zero.
        If None, user must inspect plot_magnitude() output and specify manually.
    
    Returns
    -------
    dict with denoised image and metrics
    """
    # Step 1: Load noisy image and compute CFT
    img = ContinuousImage(image_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    
    u = cft2d.u
    v = cft2d.v
    
    # Step 2: Visualize magnitude spectrum to identify noise peaks
    magnitude = np.sqrt(real**2 + imag**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original noisy image
    ax = axes[0]
    ax.imshow(img.image, cmap='gray')
    ax.set_title('Noisy Input')
    ax.axis('off')
    
    # Log-scaled magnitude spectrum
    ax = axes[1]
    im = ax.imshow(np.log(1 + magnitude), cmap='gray', origin='lower')
    ax.set_title('Log Magnitude Spectrum — Inspect for Noise Peaks')
    ax.set_xlabel('u index')
    ax.set_ylabel('v index')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('online10_noise_inspection.png', dpi=150)
    plt.show()
    
    print("Inspect the spectrum plot above. Noise peaks appear as bright")
    print("dots or lines that are off-center (not at DC).")
    print("Note their (u_idx, v_idx) coordinates and pass them as noise_peaks.")
    
    # If no peaks specified, return early for inspection
    if noise_peaks is None:
        print("\nNo noise_peaks provided. Call again with noise_peaks=[((u1,v1), r1), ...]")
        return {'magnitude_spectrum': magnitude}
    
    # Step 3: Build custom mask that zeros only noise peak regions
    rows, cols = real.shape
    mask = np.ones((rows, cols), dtype=bool)  # True = keep, False = zero
    
    for (u_idx, v_idx), radius in noise_peaks:
        # Create circular mask around each noise peak
        i_idx = np.arange(rows)
        j_idx = np.arange(cols)
        I_grid, J_grid = np.meshgrid(i_idx, j_idx, indexing='ij')
        dist = np.sqrt((I_grid - u_idx)**2 + (J_grid - v_idx)**2)
        mask[dist <= radius] = False  # Zero this region
    
    # Apply mask
    real_denoised = real.copy()
    imag_denoised = imag.copy()
    real_denoised[~mask] = 0
    imag_denoised[~mask] = 0
    
    # Step 4: Reconstruct with InverseCFT2D
    icft2d = InverseCFT2D(real_denoised, imag_denoised, u, v, img.x, img.y)
    denoised = icft2d.reconstruct()
    denoised_real = np.real(denoised)
    
    # Normalize for display
    dmin, dmax = denoised_real.min(), denoised_real.max()
    if dmax > dmin:
        denoised_display = (denoised_real - dmin) / (dmax - dmin)
    else:
        denoised_display = denoised_real
    
    # Step 5: Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    ax = axes[0, 0]
    ax.imshow(img.image, cmap='gray')
    ax.set_title('Noisy Input')
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(denoised_display, cmap='gray')
    ax.set_title('Denoised Output')
    ax.axis('off')
    
    ax = axes[0, 2]
    ax.imshow(np.abs(img.image - denoised_display), cmap='hot')
    ax.set_title('Difference (Noise Removed)')
    ax.axis('off')
    
    # Row 2: Spectra
    ax = axes[1, 0]
    ax.imshow(np.log(1 + magnitude), cmap='gray', origin='lower')
    ax.set_title('Original Spectrum')
    ax.set_xlabel('u'); ax.set_ylabel('v')
    
    ax = axes[1, 1]
    denoised_mag = np.sqrt(real_denoised**2 + imag_denoised**2)
    ax.imshow(np.log(1 + denoised_mag), cmap='gray', origin='lower')
    ax.set_title('Denoised Spectrum\\n(peaks removed)')
    ax.set_xlabel('u'); ax.set_ylabel('v')
    
    ax = axes[1, 2]
    # Show mask overlay
    mask_overlay = np.log(1 + magnitude).copy()
    mask_overlay[~mask] = np.nan  # Highlight removed regions
    ax.imshow(mask_overlay, cmap='gray', origin='lower')
    ax.set_title('Removed Regions Highlighted')
    ax.set_xlabel('u'); ax.set_ylabel('v')
    
    plt.tight_layout()
    plt.savefig('online10_denoising_result.png', dpi=150)
    plt.show()
    
    # Compute noise reduction metric
    noise_estimate = np.std(img.image - denoised_display)
    print(f"\nEstimated residual noise std: {noise_estimate:.4f}")
    
    return {
        'denoised_image': denoised_real,
        'denoised_display': denoised_display,
        'mask': mask,
        'noise_estimate': noise_estimate
    }


if __name__ == "__main__":
    # First run: inspect spectrum to find noise peaks
    result = denoise_image("D:\\BUET\\CSE220\\practice\\Online3\\21\\noisy_image.png")
    
    # Second run: after inspecting, provide peak locations
    # Example: noise at (u=50, v=30) and (u=70, v=80), each with radius 3
    result = denoise_image("D:\\BUET\\CSE220\\practice\\Online3\\21\\noisy_image.png", 
                           noise_peaks=[((50, 30), 3), ((70, 80), 3)])
    pass