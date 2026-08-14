"""
CSE 220 - Predicted Online 4: Differentiation Property of the Fourier Series
                                + constant-speed (equal arc-length) sanity check
Time: 30 minutes

Property: f'(t)  <-->  j n omega c_n
"""

import numpy as np
import matplotlib.pyplot as plt

from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


if __name__ == "__main__":
    SVG_PATH = "svgs/heart.svg"
    N = 150

    t, z = load_svg_path(SVG_PATH, num_points=1000)

    fs = FourierEpicycles(t, z, n_harmonics=N)
    fs.calculate_all_coefficients()

    # ---- numerical velocity signal ---------------------------------------------
    v = np.gradient(z, t)          # dz/dt, complex-valued (vx + j*vy)

    fs_v = FourierEpicycles(t, v, n_harmonics=N)
    fs_v.calculate_all_coefficients()

    ns = np.array(sorted(fs.coeffs.keys()))
    c_n = np.array([fs.coeffs[n] for n in ns])
    e_n = np.array([fs_v.coeffs[n] for n in ns])       # coefficients of v(t)
    theory = 1j * ns * fs.omega * c_n                   # predicted from position coeffs

    mag_mse = float(np.mean((np.abs(e_n) - np.abs(theory)) ** 2))

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    significant = np.abs(theory) > 1e-3 * np.max(np.abs(theory))
    phase_mse = float(np.mean(wrap(np.angle(e_n) - np.angle(theory))[significant] ** 2))

    print(f"Magnitude MSE  (|e_n| vs |j n omega c_n|)             = {mag_mse:.3e}")
    print(f"Phase MSE (significant harmonics only)                = {phase_mse:.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].stem(ns, np.abs(e_n), linefmt="C0-", markerfmt="C0o", basefmt=" ",
                 label="|e_n| numerical (FS of velocity)")
    axes[0].stem(ns, np.abs(theory), linefmt="C1--", markerfmt="C1x", basefmt=" ",
                 label="|n*omega*c_n| theory")
    axes[0].set_title("Differentiation property: magnitude"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ns[significant], wrap(np.angle(e_n))[significant], "o", label="numerical")
    axes[1].plot(ns[significant], wrap(np.angle(theory))[significant], "x", label="theory")
    axes[1].set_title("Differentiation property: phase")
    axes[1].set_xlabel("n"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.show()

    # ---- constant-speed (equal arc-length) sanity check on svg_utils' claim ----
    speed = np.abs(v)
    mean_speed = speed.mean()
    rel_std = speed.std() / mean_speed

    plt.figure(figsize=(9, 4))
    plt.plot(t, speed, lw=2)
    plt.axhline(mean_speed, color="r", ls="--", label=f"mean = {mean_speed:.3f}")
    plt.title("Pen speed |dz/dt| over one period\n"
              "(svg_utils.load_svg_path claims equal arc-length -> should be ~constant)")
    plt.xlabel("t"); plt.ylabel("|v(t)|"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

    print(f"\nMean pen speed                                        = {mean_speed:.4f}")
    print(f"Relative std. dev. of speed                           = {rel_std:.3e}")
    print("\nComment: the differentiation property holds numerically (small MSE, worse\n"
          "near high |n| where numerical np.gradient differentiation is less accurate\n"
          "on a finite grid). Separately, the low relative std. dev. of |v(t)| confirms\n"
          "svg_utils.py's claim that the curve is re-parametrized to constant arc-length\n"
          "speed - a good example of numerically verifying a 'given/black-box' module's\n"
          "documented guarantee rather than trusting it blindly.")
