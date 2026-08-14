"""
CSE 220 - Predicted Online 2: Time-Shift Property of the Fourier Series
Time: 30 minutes

Property: g(t) = f(t - t0)  (same period T)  <-->  d_n = c_n * e^{-j n omega t0}

Built on the offline's FourierEpicycles class (imported unmodified).
"""

import numpy as np
import matplotlib.pyplot as plt

from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def time_shift_periodic(t, signal, T, t0):
    """
    g(t) = f(t - t0), implemented OOP-style via periodic interpolation
    (NOT a manual np.roll / array slice) so it works for any t0, not just
    multiples of the sample spacing.
    """
    # shift and wrap into [0, T)
    t_shifted = (t - t0) % T
    # signal is defined on the closed grid t in [0, T] with signal[-1]==signal[0],
    # so np.interp with periodic wrapping reconstructs g(t) correctly.
    re = np.interp(t_shifted, t, signal.real, period=T)
    im = np.interp(t_shifted, t, signal.imag, period=T)
    return re + 1j * im


if __name__ == "__main__":
    SVG_PATH = "svgs/heart.svg"
    N = 150

    t, z = load_svg_path(SVG_PATH, num_points=1000)
    T = t[-1] - t[0]
    t0 = T / 4.0

    g = time_shift_periodic(t, z, T, t0)

    fs_orig = FourierEpicycles(t, z, n_harmonics=N)
    fs_orig.calculate_all_coefficients()

    fs_shift = FourierEpicycles(t, g, n_harmonics=N)
    fs_shift.calculate_all_coefficients()

    ns = np.array(sorted(fs_orig.coeffs.keys()))
    c_n = np.array([fs_orig.coeffs[n] for n in ns])
    d_n = np.array([fs_shift.coeffs[n] for n in ns])
    omega = fs_orig.omega
    theory = c_n * np.exp(-1j * ns * omega * t0)

    mag_mse = float(np.mean((np.abs(d_n) - np.abs(theory)) ** 2))

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    significant = np.abs(c_n) > 1e-3 * np.max(np.abs(c_n))
    phase_diff = wrap(np.angle(d_n) - np.angle(theory))
    phase_mse = float(np.mean(phase_diff[significant] ** 2))

    print(f"Magnitude MSE  (|d_n| vs |c_n|)                        = {mag_mse:.3e}")
    print(f"Phase MSE      (only where |c_n| significant)          = {phase_mse:.3e}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].stem(ns, np.abs(c_n), linefmt="C0-", markerfmt="C0o", basefmt=" ", label="|c_n|")
    axes[0].stem(ns, np.abs(d_n), linefmt="C1--", markerfmt="C1x", basefmt=" ", label="|d_n|")
    axes[0].set_title("Magnitude spectrum: unchanged by a pure time shift")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(ns[significant], wrap(np.angle(d_n))[significant], "o", label="angle(d_n)")
    axes[1].plot(ns[significant], wrap(np.angle(theory))[significant], "x",
                 label="angle(c_n) - n*omega*t0 (theory)")
    axes[1].set_title("Phase: linear-in-n shift introduced by t0")
    axes[1].set_xlabel("n"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.show()

    # ---- does the traced SHAPE change? -----------------------------------------
    t_dense = np.linspace(0, T, 2000, endpoint=False)
    z_hat_orig = fs_orig.approximate(t_dense)
    z_hat_shift = fs_shift.approximate(t_dense)

    plt.figure(figsize=(5, 5))
    plt.plot(z_hat_orig.real, z_hat_orig.imag, lw=2, label="original reconstruction")
    plt.plot(z_hat_shift.real, z_hat_shift.imag, "--", lw=2, label="shifted reconstruction")
    plt.gca().set_aspect("equal"); plt.legend(); plt.title("Shape comparison")
    plt.tight_layout(); plt.show()

    print("\nComment: |d_n| == |c_n| confirms a pure time shift leaves the magnitude\n"
          "spectrum unchanged, and the phase MSE (restricted to significant harmonics)\n"
          "confirms the predicted linear phase term -n*omega*t0. The two reconstructed\n"
          "curves overlap exactly as CLOSED SHAPES - time-shifting only changes WHERE\n"
          "along the curve the pen starts (the phase reference), not the geometry drawn.")
