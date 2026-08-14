"""
CSE 220 - Predicted Online 7: Low-Pass Filtering (Blurring)
Time: 30 minutes

The offline's FrequencyFilter.high_pass zeroes everything WITHIN radius
`cutoff` of the spectrum center (keeping edges). Here we flip the mask -
zero everything OUTSIDE `cutoff` (keeping only slow/low-frequency content) -
to produce a blur, without touching any class definitions (per the offline's
"no new classes/methods" constraint - the mask logic lives in this script).
"""

import numpy as np
import matplotlib.pyplot as plt

from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def low_pass(real, imag, cutoff):
    """Mirror image of FrequencyFilter.high_pass: keep only the disk of
    radius `cutoff` around the spectrum center, zero everything outside it."""
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    real, imag = real.copy(), imag.copy()
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) > cutoff:
                real[i, j] = 0
                imag[i, j] = 0
    return real, imag


def energy_fraction_within(real, imag, cutoff):
    """Fraction of total spectral energy retained inside radius `cutoff`."""
    rows, cols = real.shape
    cx, cy = rows // 2, cols // 2
    mag2 = real ** 2 + imag ** 2
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    inside = np.sqrt((yy - cx) ** 2 + (xx - cy) ** 2) <= cutoff
    return float(mag2[inside].sum() / mag2.sum())


if __name__ == "__main__":
    IMAGE_PATH = "test_face.png"

    img = ContinuousImage(IMAGE_PATH)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    cutoffs = [5, 15, 40]
    fig, axes = plt.subplots(1, len(cutoffs) + 1, figsize=(4 * (len(cutoffs) + 1), 4.5))
    axes[0].imshow(img.image, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")

    for ax, cutoff in zip(axes[1:], cutoffs):
        real_f, imag_f = low_pass(real, imag, cutoff)
        icft2d = InverseCFT2D(real_f, imag_f, cft2d.u, cft2d.v, img.x, img.y)
        blurred = icft2d.reconstruct()

        frac = energy_fraction_within(real, imag, cutoff)
        print(f"cutoff={cutoff:>3}  energy retained = {frac*100:5.1f}%")

        ax.imshow(np.real(blurred), cmap="gray")
        ax.set_title(f"cutoff={cutoff}\n({frac*100:.1f}% energy)")
        ax.axis("off")

    plt.tight_layout(); plt.show()

    print("\nComment: as cutoff shrinks, more of the fine spatial detail (edges, texture)\n"
          "gets discarded and the image blurs more heavily, while the retained-energy\n"
          "fraction drops. At very low cutoff only the coarse brightness pattern (near-DC\n"
          "content) survives, matching the intuition that most of a natural image's\n"
          "energy sits at low spatial frequency - exactly the opposite behavior of the\n"
          "offline's high_pass, which discards that same low-frequency content instead.")
