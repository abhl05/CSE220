"""
CSE 220 - Predicted Online 10 ("Mission"-style): Frequency-Domain Denoising
                                                   Using Your Own CFT2D
Time: 40 minutes

Sequel to last year's "Prove Your Relevancy" secret-agent mission, but this
year everything must go through your OWN trapezoidal CFT2D / InverseCFT2D
(imported unmodified from cft_edge_detector.py) - no np.fft anywhere, and no
new classes/methods (the custom notch mask lives in this script, exactly
like the offline's own high_pass is just a disk mask you're allowed to
mimic, not subclass).
"""

import numpy as np
import matplotlib.pyplot as plt

from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def notch_mask(real, imag, peak_coords, notch_radius=2):
    """
    Zero out small disks around each given (row, col) index into the
    spectrum - and their point-mirror partner, since a real-valued image
    has conjugate-symmetric spectrum, so periodic noise shows up as a PAIR
    of symmetric peaks.
    """
    rows, cols = real.shape
    real, imag = real.copy(), imag.copy()
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    mask = np.ones((rows, cols), dtype=bool)
    for (pr, pc) in peak_coords:
        d = np.sqrt((yy - pr) ** 2 + (xx - pc) ** 2)
        mask &= d > notch_radius
        mr, mc = rows - 1 - pr, cols - 1 - pc  # mirror about the spectrum center
        d2 = np.sqrt((yy - mr) ** 2 + (xx - mc) ** 2)
        mask &= d2 > notch_radius
    real[~mask] = 0
    imag[~mask] = 0
    return real, imag


if __name__ == "__main__":
    IMAGE_PATH = "noisy_letter_small.png"   # keep images SMALL - trapz CFT is O(N^3)

    img = ContinuousImage(IMAGE_PATH)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    mag = np.sqrt(real ** 2 + imag ** 2)

    # ---- Step 1: visualize the spectrum and locate the noise peak(s) -----------
    plt.figure(figsize=(5, 5))
    plt.imshow(np.log1p(mag), cmap="viridis",
               extent=[cft2d.u[0], cft2d.u[-1], cft2d.v[0], cft2d.v[-1]], origin="lower")
    plt.title("Log-magnitude spectrum - look for bright off-center dots")
    plt.colorbar(); plt.tight_layout(); plt.show()

    # ---- Step 2: automatically find the strongest non-DC peak(s) for convenience
    # (on exam day you may just eyeball this from the plot instead)
    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2
    mag_copy = mag.copy()
    mag_copy[cy - 1:cy + 2, cx - 1:cx + 2] = 0  # exclude near-DC
    top_idx = np.dstack(np.unravel_index(np.argsort(-mag_copy.ravel())[:4], mag_copy.shape))[0]
    peak_coords = [tuple(p) for p in top_idx]
    print("Detected candidate noise peaks (row, col):", peak_coords)

    # ---- Step 3: notch them out and reconstruct ---------------------------------
    real_f, imag_f = notch_mask(real, imag, peak_coords, notch_radius=2)
    icft2d = InverseCFT2D(real_f, imag_f, cft2d.u, cft2d.v, img.x, img.y)
    denoised = np.real(icft2d.reconstruct())

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img.image, cmap="gray"); axes[0].set_title("Noisy input"); axes[0].axis("off")
    axes[1].imshow(denoised, cmap="gray"); axes[1].set_title("Denoised (CFT2D notch)"); axes[1].axis("off")
    plt.tight_layout(); plt.show()

    print("\nWorkflow reminder for the real exam image:")
    print("1. Keep the image SMALL (e.g. <= 64x64) - your CFT2D is O(N^3) trapz, not FFT.")
    print("2. Use plot_magnitude()/this log-spectrum plot to find the noise peak(s) by eye")
    print("   if the automatic peak-finder above doesn't isolate them cleanly.")
    print("3. Reuse notch_mask(...) with the coordinates you found - do NOT add new")
    print("   methods to CFT2D/InverseCFT2D/FrequencyFilter, keep the custom logic here.")
    print("4. If the noise is a full row/column of energy in the spectrum (pure")
    print("   horizontal- or vertical-only stripes), zero the whole row/column instead")
    print("   of individual point peaks - it's faster and more thorough.")
