"""
CSE 220 - Predicted Online 9: Reconstruction Sanity Check + Band-Pass Filter
Time: 30-40 minutes

Part A: skipping the filter entirely, InverseCFT2D.reconstruct() should
        recover the original image almost exactly - a direct validation of
        compute_cft()/reconstruct() being true inverses of each other.
Part B: a band-pass mask (an annulus r1 <= r <= r2) isolates a specific
        SCALE of texture/edges, unlike a plain high-pass which keeps
        "everything above one cutoff."
"""

import numpy as np
import matplotlib.pyplot as plt

from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def band_pass(real, imag, r1, r2):
    """Keep only frequency components with r1 <= radius <= r2; zero elsewhere."""
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    real, imag = real.copy(), imag.copy()
    for i in range(rows):
        for j in range(cols):
            r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
            if not (r1 <= r <= r2):
                real[i, j] = 0
                imag[i, j] = 0
    return real, imag


if __name__ == "__main__":
    IMAGE_PATH = "test_face.png"

    img = ContinuousImage(IMAGE_PATH)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    # ---- Part A: no-filter round trip -------------------------------------------
    icft2d = InverseCFT2D(real, imag, cft2d.u, cft2d.v, img.x, img.y)
    recon = icft2d.reconstruct()
    recon_mse = float(np.mean((np.real(recon) - img.image) ** 2))
    print(f"[Part A] Reconstruction MSE (no filtering)             = {recon_mse:.3e}")
    print(f"[Part A] max |orig - recon| pixel-wise                 = {np.max(np.abs(np.real(recon)-img.image)):.3e}")

    # controlled comparison on a smooth signal (no sharp edges) for contrast
    xx, yy = np.meshgrid(img.x, img.y)
    smooth = np.exp(-(xx ** 2 + yy ** 2) / 0.3)

    class _SynthImage(ContinuousImage):
        def __init__(self, arr, x, y):
            self.image, self.x, self.y = arr, x, y

    smooth_img = _SynthImage(smooth, img.x, img.y)
    cft_s = CFT2D(smooth_img)
    rs, is_ = cft_s.compute_cft()
    recon_s = InverseCFT2D(rs, is_, cft_s.u, cft_s.v, img.x, img.y).reconstruct()
    mse_smooth = float(np.mean((np.real(recon_s) - smooth) ** 2))
    print(f"[Part A, smooth control] Reconstruction MSE (Gaussian) = {mse_smooth:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(img.image, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(np.real(recon), cmap="gray"); axes[1].set_title("Reconstructed (unfiltered)"); axes[1].axis("off")
    plt.tight_layout(); plt.show()

    # ---- Part B: band-pass filter -----------------------------------------------
    bands = [(0, 5), (5, 15), (15, 40)]
    fig, axes = plt.subplots(1, len(bands) + 1, figsize=(4 * (len(bands) + 1), 4.5))
    axes[0].imshow(img.image, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")

    for ax, (r1, r2) in zip(axes[1:], bands):
        real_bp, imag_bp = band_pass(real, imag, r1, r2)
        icft_bp = InverseCFT2D(real_bp, imag_bp, cft2d.u, cft2d.v, img.x, img.y)
        band_result = icft_bp.reconstruct()
        ax.imshow(np.real(band_result), cmap="gray")
        ax.set_title(f"band [{r1},{r2}]")
        ax.axis("off")

    plt.tight_layout(); plt.show()

    print("\nComment [Part A]: the round-trip MSE is small relative to the image's full\n"
          "[0,1] range but not machine-precision zero, and drops sharply on the smooth\n"
          "Gaussian control - because compute_cft/reconstruct use trapezoidal quadrature\n"
          "over a FINITE Nyquist band rather than an exact orthogonal DFT/IDFT pair, a\n"
          "real image's sharp edges (broadband content right up to and past the Nyquist\n"
          "limit) reconstruct with visible ringing, while a smooth, band-limited signal\n"
          "round-trips almost exactly. This is still the right sanity check to run before\n"
          "trusting any filtered result.")
    print("\nComment [Part B]: the low band [0,5] isolates coarse brightness (near-DC)\n"
          "content, the highest band [15,40] behaves like the offline's own high_pass\n"
          "and isolates fine edges/texture, and the middle band [5,15] isolates a\n"
          "specific intermediate SCALE - broad shape outlines without fine texture -\n"
          "something neither a pure low-pass nor a pure high-pass alone can show.")
