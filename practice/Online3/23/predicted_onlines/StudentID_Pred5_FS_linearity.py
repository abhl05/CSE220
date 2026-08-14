"""
CSE 220 - Predicted Online 5: Linearity Property of the Fourier Series
Time: 30 minutes

Property: if h(t) = f(t) + g(t) (same T, same sampling grid), then
          c_n(h) = c_n(f) + c_n(g)   for every n.
"""

import numpy as np
import matplotlib.pyplot as plt

from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles


if __name__ == "__main__":
    N = 150

    # Both shapes must share the exact same t grid (both come from
    # load_svg_path with the same num_points, which always returns t in [0,2*pi]).
    t, z_heart = load_svg_path("svgs/heart.svg", num_points=1000)
    _, z_circle = load_svg_path("svgs/circle.svg", num_points=1000)

    # combined signal (simple sum - not a geometrically meaningful "drawing",
    # but a valid periodic complex signal to test linearity on)
    z_sum = z_heart + z_circle

    fs_f = FourierEpicycles(t, z_heart, n_harmonics=N); fs_f.calculate_all_coefficients()
    fs_g = FourierEpicycles(t, z_circle, n_harmonics=N); fs_g.calculate_all_coefficients()
    fs_h = FourierEpicycles(t, z_sum, n_harmonics=N); fs_h.calculate_all_coefficients()

    ns = np.array(sorted(fs_f.coeffs.keys()))
    c_f = np.array([fs_f.coeffs[n] for n in ns])
    c_g = np.array([fs_g.coeffs[n] for n in ns])
    c_h = np.array([fs_h.coeffs[n] for n in ns])

    predicted_h = c_f + c_g
    mse = float(np.mean(np.abs(c_h - predicted_h) ** 2))
    print(f"MSE between c_n(f+g) and c_n(f)+c_n(g)  = {mse:.3e}")

    plt.figure(figsize=(9, 4))
    plt.stem(ns, np.abs(c_h), linefmt="C0-", markerfmt="C0o", basefmt=" ",
             label="|c_n(h)| numerical")
    plt.stem(ns, np.abs(predicted_h), linefmt="C1--", markerfmt="C1x", basefmt=" ",
             label="|c_n(f) + c_n(g)| theory")
    plt.title("Linearity property of the Fourier Series"); plt.xlabel("n")
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

    # ---- visualize the three reconstructed curves -------------------------------
    t_dense = np.linspace(0, fs_f.T, 2000, endpoint=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fs_f.approximate(t_dense).real, fs_f.approximate(t_dense).imag,
            label="heart f(t)")
    ax.plot(fs_g.approximate(t_dense).real, fs_g.approximate(t_dense).imag,
            label="circle g(t)")
    ax.plot(fs_h.approximate(t_dense).real, fs_h.approximate(t_dense).imag,
            label="sum h(t) = f(t)+g(t)")
    ax.set_aspect("equal"); ax.legend(); ax.set_title("Individual shapes vs. their sum")
    plt.tight_layout(); plt.show()

    print("\nComment: the MSE is essentially machine-precision zero, confirming linearity -\n"
          "the Fourier Series coefficient operator is linear because the defining integral\n"
          "Integral f(t)*exp(-jnwt)dt is itself linear in f. Note the combined curve h(t)\n"
          "is just a pointwise vector sum and is not a 'meaningful drawing' on its own -\n"
          "linearity holds for any two periodic signals sharing a period, not only for\n"
          "geometrically sensible combinations.")
