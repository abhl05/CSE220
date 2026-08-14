"""
CSE 220 - Predicted Online 3: Convergence / Gibbs Phenomenon vs Number of Harmonics
Time: 30-40 minutes

Compares reconstruction quality vs N for a shape with sharp corners (star.svg)
against a perfectly smooth shape (circle.svg), using the offline's
FourierEpicycles class unmodified.
"""

import numpy as np
import matplotlib.pyplot as plt

from svg_utils import load_svg_path
from fs_redrawer import FourierEpicycles


def reconstruction_mse(t, z_true, N):
    fs = FourierEpicycles(t, z_true, n_harmonics=N)
    fs.calculate_all_coefficients()
    t_dense = np.linspace(0, fs.T, 2000, endpoint=False)
    z_hat = fs.approximate(t_dense)
    z_true_dense_re = np.interp(t_dense, t, z_true.real, period=fs.T)
    z_true_dense_im = np.interp(t_dense, t, z_true.imag, period=fs.T)
    z_true_dense = z_true_dense_re + 1j * z_true_dense_im
    mse = float(np.mean(np.abs(z_hat - z_true_dense) ** 2))
    return mse, fs, t_dense, z_hat, z_true_dense


if __name__ == "__main__":
    N_values = [5, 10, 20, 50, 100, 150, 300]
    shapes = {"star (sharp corners)": "svgs/star.svg",
              "circle (smooth)": "svgs/circle.svg"}

    results = {}
    for label, path in shapes.items():
        t, z = load_svg_path(path, num_points=1000)
        mses = []
        for N in N_values:
            mse, fs, t_dense, z_hat, z_true_dense = reconstruction_mse(t, z, N)
            mses.append(mse)
        results[label] = (mses, t, z)
        print(f"{label}: MSE per N -> " +
              ", ".join(f"N={n}:{m:.2e}" for n, m in zip(N_values, mses)))

    plt.figure(figsize=(8, 5))
    for label, (mses, _, _) in results.items():
        plt.plot(N_values, mses, "o-", label=label)
    plt.yscale("log")
    plt.xlabel("Number of harmonics N")
    plt.ylabel("Reconstruction MSE (log scale)")
    plt.title("Convergence rate: smooth vs cornered shape")
    plt.legend(); plt.grid(True, which="both")
    plt.tight_layout(); plt.show()

    # ---- zoom on a star corner at low N to show Gibbs overshoot ----------------
    t_star, z_star = load_svg_path("svgs/star.svg", num_points=1000)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, N in zip(axes, [10, 50, 150]):
        fs = FourierEpicycles(t_star, z_star, n_harmonics=N)
        fs.calculate_all_coefficients()
        t_dense = np.linspace(0, fs.T, 3000, endpoint=False)
        z_hat = fs.approximate(t_dense)
        ax.plot(z_star.real, z_star.imag, color="0.6", lw=3, alpha=0.6, label="Original")
        ax.plot(z_hat.real, z_hat.imag, color="crimson", lw=1.2, label=f"N={N}")
        ax.set_aspect("equal"); ax.legend(); ax.set_title(f"Star reconstruction, N={N}")
    plt.tight_layout(); plt.show()

    print("\nComment: the star's MSE decays much more slowly with N than the circle's -\n"
          "because the star's sharp corners are points where the derivative of the\n"
          "traced path is discontinuous, the Fourier coefficients decay only like\n"
          "O(1/n^2) (vs. exponentially fast for the perfectly smooth circle). At low N\n"
          "you can see visible ringing/overshoot ('Gibbs phenomenon') right at the\n"
          "star's points, which shrinks in width (but not fully in height) as N grows.")
