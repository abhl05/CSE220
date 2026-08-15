"""
CSE 220 - Predicted Online 1: Parseval's Theorem for the Fourier Series
Time: 30 minutes

Property:
    (1/T) * Integral_0^T |f(t)|^2 dt   =   Sum_{n=-inf}^{inf} |c_n|^2

Built directly on top of your offline's FourierEpicycles class (imported
unmodified) and svg_utils.load_svg_path (given).
"""

import numpy as np
import matplotlib.pyplot as plt

from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def parseval_check(fs: FourierEpicycles):
    """Returns (E_time, E_freq, relative_error)."""
    E_time = trapz(np.abs(fs.signal) ** 2, fs.t) / fs.T
    E_freq = sum(abs(c) ** 2 for c in fs.coeffs.values())
    rel_err = abs(E_time - E_freq) / E_time
    return E_time, E_freq, rel_err


if __name__ == "__main__":
    SVG_PATH = "D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task1\\svgs\\heart.svg"     # change to whichever shape the online gives you
    N = 150

    t, z = load_svg_path(SVG_PATH, num_points=1000)
    fs = FourierEpicycles(t, z, n_harmonics=N)
    fs.calculate_all_coefficients()

    E_time, E_freq, rel_err = parseval_check(fs)
    print(f"Time-domain average power   (1/T)*Integral|f(t)|^2 dt = {E_time:.6f}")
    print(f"Freq-domain total power     Sum |c_n|^2                = {E_freq:.6f}")
    print(f"Relative error                                         = {rel_err:.3e}")

    # ---- power spectrum plot -------------------------------------------------
    ns = np.array(sorted(fs.coeffs.keys()))
    power = np.array([abs(fs.coeffs[n]) ** 2 for n in ns])

    plt.figure(figsize=(9, 4))
    plt.stem(ns, power)
    plt.title("Power spectrum |c_n|^2 vs harmonic n")
    plt.xlabel("n"); plt.ylabel("|c_n|^2"); plt.grid(True)
    plt.savefig("power_spectrum.png")
    plt.tight_layout(); plt.show()

    # ---- cumulative energy captured vs number of harmonics kept ---------------
    order = np.argsort(-power)  # largest-power harmonics first
    cum_energy = np.cumsum(power[order])
    frac_energy = cum_energy / E_freq

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, len(frac_energy) + 1), frac_energy, lw=2)
    plt.axhline(0.99, color="gray", ls="--", label="99% energy")
    plt.title("Cumulative fraction of signal energy captured\n"
              "(harmonics added in order of decreasing power)")
    plt.xlabel("number of harmonics included"); plt.ylabel("fraction of total energy")
    plt.legend(); plt.grid(True)
    plt.savefig("cumulative_energy.png")
    plt.tight_layout(); plt.show()

    n99 = int(np.searchsorted(frac_energy, 0.99) + 1)
    print(f"Harmonics needed to capture 99% of energy: {n99} (out of {2*N+1} available)")
    print("\nComment: the relative error should be extremely small (<1e-6) since Parseval\n"
          "is an algebraic identity of the same numerical integration used to compute\n"
          "the coefficients in the first place - any residual error is pure floating\n"
          "point roundoff, not a modeling approximation.")
