"""
CSE 220 - Predicted Online 6: Parseval's Theorem in 2D (Image Energy Conservation)
Time: 30-40 minutes

Property: Integral Integral |I(x,y)|^2 dx dy  =  Integral Integral |F(u,v)|^2 du dv

Built directly on the offline's ContinuousImage / CFT2D classes (imported
unmodified).
"""

import numpy as np
import matplotlib.pyplot as plt

from cft_edge_detector import ContinuousImage, CFT2D

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


if __name__ == "__main__":
    IMAGE_PATH = "D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task2\\pikachu.png"   # replace with the exam's actual input image

    img = ContinuousImage(IMAGE_PATH)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()
    magnitude_sq = real ** 2 + imag ** 2

    # ---- time-domain energy: separable double trapz over y then x --------------
    E_time = trapz(trapz(img.image ** 2, img.x, axis=1), img.y)

    # ---- frequency-domain energy: separable double trapz over v then u ---------
    E_freq = trapz(trapz(magnitude_sq, cft2d.u, axis=1), cft2d.v)

    rel_err = abs(E_time - E_freq) / E_time

    print(f"Time-domain energy   Integral Integral |I(x,y)|^2 dx dy = {E_time:.6f}")
    print(f"Freq-domain energy   Integral Integral |F(u,v)|^2 du dv = {E_freq:.6f}")
    print(f"Relative error                                          = {rel_err:.3e}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img.image, cmap="gray"); axes[0].set_title("I(x,y)"); axes[0].axis("off")
    im = axes[1].imshow(np.log1p(np.sqrt(magnitude_sq)), cmap="viridis",
                         extent=[cft2d.u[0], cft2d.u[-1], cft2d.v[0], cft2d.v[-1]],
                         origin="lower")
    axes[1].set_title("log(1+|F(u,v)|)")
    plt.colorbar(im, ax=axes[1])
    plt.savefig("parseval_2d.png")
    plt.tight_layout(); plt.show()

    print("\nComment: some relative error here is expected and instructive - a real image\n"
          "has sharp edges (jump discontinuities), so trapezoidal quadrature of the\n"
          "underlying continuous integral converges only like O(1/gridsize) rather than\n"
          "the much faster convergence seen for smooth signals (e.g. the earlier Gaussian\n"
          "online). Try re-running on a higher-resolution version of the same image -\n"
          "the relative error should shrink roughly proportionally to 1/(image width),\n"
          "confirming it is a discretization effect rather than a bug in compute_cft.")
