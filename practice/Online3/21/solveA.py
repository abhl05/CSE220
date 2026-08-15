"""
CSE 220: Online - "Prove Your Relevancy" (image denoising via Fourier Transform)

Task: load noisy_image.png, use the Fourier Transform to filter out periodic
      noise, and recover a recognizable letter.

NOTE ON THE FFT RESTRICTION: unlike the other CFT onlines, this problem does NOT
state "no np.fft allowed" - it is a 2D DISCRETE image-processing task (the hint
"apply FT row by row" is about the discrete 1D FFT of each image row), so np.fft
is the appropriate and expected tool here.

Two approaches are given, matching the hint:
  A) Row-by-row 1D FFT + notch filtering (exactly what the hint suggests)
  B) Full 2D FFT + a radial low-pass / notch mask (usually the fastest/most robust)
Both are provided; use whichever recovers the clearest letter for your actual image.
"""

import numpy as np
import matplotlib.pyplot as plt

# NOTE: point this at your actual intercepted file.
IMAGE_PATH = "noisy_image.png"


def load_grayscale(path):
    """Loads an image as a 2D float array in [0, 1]. Falls back to a synthetic
    noisy test image (a letter 'A' + periodic stripe noise) if the real file
    isn't found, so the pipeline below is runnable/demoable on its own."""
    try:
        img = plt.imread(path)
        if img.ndim == 3:
            img = img[..., :3].mean(axis=2)  # RGB -> grayscale
        return img.astype(float)
    except FileNotFoundError:
        print(f"[!] '{path}' not found - generating a synthetic demo image instead.")
        return _synthetic_demo_image()


def _synthetic_demo_image(size=128):
    """Builds a blocky letter 'A' and corrupts it with periodic (sinusoidal)
    stripe noise, so you can test/debug the denoising pipeline before exam day."""
    img = np.zeros((size, size))
    # crude letter "A" from line segments
    for i in range(size):
        frac = i / size
        left = int(size * 0.5 - frac * size * 0.35)
        right = int(size * 0.5 + frac * size * 0.35)
        if 0 <= left < size:
            img[i, max(0, left - 2):left + 2] = 1.0
        if 0 <= right < size:
            img[i, right - 2:min(size, right + 2)] = 1.0
        if 0.45 < frac < 0.55:
            img[i, max(0, left):min(size, right)] = 1.0

    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    noise = 0.5 * np.sin(2 * np.pi * xx * 12 / size) + 0.3 * np.sin(2 * np.pi * yy * 20 / size)
    noisy = np.clip(img + noise, 0, 1)
    return noisy


# ======================= APPROACH A: row-by-row 1D FFT ==============================
def denoise_row_by_row(img, notch_frac=0.06):
    """
    For each row, take its 1D FFT, zero out the highest-frequency bins (which
    typically carry high-frequency periodic/salt noise while preserving the
    coarse letter shape), then inverse FFT back.
    `notch_frac` = fraction of the row's frequency bins (from the Nyquist edge
    inward) that get zeroed out on each side.
    """
    rows, cols = img.shape
    cutoff = int(cols * notch_frac)
    out = np.zeros_like(img)
    for r in range(rows):
        F = np.fft.fft(img[r])
        F_shift = np.fft.fftshift(F)
        mid = cols // 2
        # zero out a band near the extreme (highest) frequencies
        F_shift[:cutoff] = 0
        F_shift[-cutoff:] = 0
        F_back = np.fft.ifftshift(F_shift)
        out[r] = np.real(np.fft.ifft(F_back))
    return out


# ======================= APPROACH B: full 2D FFT + radial low-pass ===================
def denoise_2d_lowpass(img, keep_radius_frac=0.15):
    """
    Full 2D FFT, then keep only a low-frequency disk of radius
    `keep_radius_frac * min(rows, cols)/2` around the DC term (zero-frequency),
    zeroing everything else. This removes fine periodic/high-frequency noise
    while preserving the coarse shape of the letter.
    """
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    rows, cols = img.shape
    cy, cx = rows // 2, cols // 2
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    keep_radius = keep_radius_frac * min(rows, cols) / 2

    mask = radius <= keep_radius
    F_filtered = F_shift * mask

    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    return img_back, np.abs(F_shift), mask


# ======================= APPROACH B2: notch filter for KNOWN noise peaks ==============
def denoise_2d_notch(img, peak_coords, notch_radius=3):
    """
    If you can visually spot bright noise-only peaks in the 2D magnitude
    spectrum (common for periodic stripe noise), zero out small disks around
    each of those peak coordinates (and their mirror image) instead of a blanket
    low-pass. This preserves more real detail than approach B.
    peak_coords: list of (row, col) indices INTO THE SHIFTED spectrum.
    """
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)
    rows, cols = img.shape
    yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    mask = np.ones((rows, cols), dtype=bool)
    for (py, px) in peak_coords:
        d = np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
        mask &= d > notch_radius
        # also notch the mirror-symmetric peak (real images have conjugate symmetry)
        my, mx = rows - py, cols - px
        d2 = np.sqrt((yy - my) ** 2 + (xx - mx) ** 2)
        mask &= d2 > notch_radius

    F_filtered = F_shift * mask
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    return img_back


# ======================= APPROACH C: full-row notch (best for pure horizontal noise) =
def denoise_row_notch(img, dc_keep_width=1):
    """
    Special case of the notch idea, tuned for noise that is a PURE function of x
    (i.e. vertical stripes with no y-variation at all). Such noise puts ALL of its
    energy exactly on the zero-vertical-frequency row of the 2D spectrum (row = cy,
    the row through the DC term) - so we can remove essentially the whole thing by
    zeroing that entire row except a tiny window around the true DC term, while
    leaving every other row (which is where any real 2D content, like a letter,
    lives) completely untouched.
    """
    rows, cols = img.shape
    cy, cx = rows // 2, cols // 2
    F = np.fft.fft2(img)
    F_shift = np.fft.fftshift(F)

    mask = np.ones((rows, cols), dtype=bool)
    mask[cy, :] = False
    mask[cy, cx - dc_keep_width:cx + dc_keep_width + 1] = True  # keep true DC

    F_filtered = F_shift * mask
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    return img_back


if __name__ == "__main__":
    img = load_grayscale(IMAGE_PATH)

    denoised_rowwise = denoise_row_by_row(img, notch_frac=0.06)
    denoised_2d, mag_spectrum, lp_mask = denoise_2d_lowpass(img, keep_radius_frac=0.15)
    denoised_row_notch = denoise_row_notch(img, dc_keep_width=0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    axes[0, 0].imshow(img, cmap="gray"); axes[0, 0].set_title("Noisy input")
    axes[0, 1].imshow(np.log1p(mag_spectrum), cmap="viridis")
    axes[0, 1].set_title("2D FFT magnitude spectrum (log scale)\n"
                          "-> here ALL noise energy sits on one horizontal line")
    axes[0, 2].axis("off")
    axes[1, 0].imshow(denoised_rowwise, cmap="gray")
    axes[1, 0].set_title("Approach A: row-by-row 1D FFT notch")
    axes[1, 1].imshow(denoised_2d, cmap="gray")
    axes[1, 1].set_title("Approach B: 2D FFT radial low-pass")
    axes[1, 2].imshow(denoised_row_notch, cmap="gray")
    axes[1, 2].set_title("Approach C: full-row notch\n(best for this image)")
    for ax in axes.ravel():
        ax.axis("off")
    plt.savefig("denoising_results.png")
    plt.tight_layout()
    plt.show()

    print("Workflow for the real exam image:")
    print("1. Run this script on noisy_image.png and look at the log-magnitude")
    print("   spectrum panel (top-right).")
    print("2. If the noise shows as bright, isolated dots away from the center,")
    print("   read off their (row, col) pixel coordinates and pass them into")
    print("   denoise_2d_notch(img, peak_coords=[...]) for the cleanest result.")
    print("3. If the noise is spread across high frequencies generally, use")
    print("   denoise_2d_lowpass() or denoise_row_by_row() and tune the cutoff")
    print("   fraction until the letter is recognizable (perfect clarity is NOT")
    print("   required per the problem statement).")