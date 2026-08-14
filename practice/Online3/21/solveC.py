"""
CSE 220: Online C1-C2
Topic   : Decomposing a product-of-sines signal into a sum of unit-amplitude,
          zero-phase sinusoids, and confirming it using the CFT.

Given:
    f(t) = 2 sin(14*pi*t) - sin(2*pi*t) * (4 sin(2*pi*t) sin(14*pi*t) - 1)

--------------------------- ANALYTIC DERIVATION ------------------------------------
Expand:
    f(t) = 2 sin(14πt) - 4 sin^2(2πt) sin(14πt) + sin(2πt)

Use sin^2(x) = (1 - cos(2x)) / 2  with x = 2πt:
    4 sin^2(2πt) = 2 (1 - cos(4πt)) = 2 - 2 cos(4πt)

Substitute:
    f(t) = 2 sin(14πt) - [2 - 2 cos(4πt)] sin(14πt) + sin(2πt)
         = 2 sin(14πt) - 2 sin(14πt) + 2 cos(4πt) sin(14πt) + sin(2πt)
         = 2 cos(4πt) sin(14πt) + sin(2πt)

Use product-to-sum: 2 cos(A) sin(B) = sin(B+A) + sin(B-A), A = 4πt, B = 14πt:
    2 cos(4πt) sin(14πt) = sin(18πt) + sin(10πt)

Therefore:
    f(t) = sin(18πt) + sin(10πt) + sin(2πt)
         = sin(2π*9*t) + sin(2π*5*t) + sin(2π*1*t)

=> f(t) is the sum of THREE unit-amplitude, zero-phase sine waves at
   f1 = 1 Hz, f2 = 5 Hz, f3 = 9 Hz.
--------------------------------------------------------------------------------------

This script verifies the derivation two ways:
  1. Directly: plot f(t) against sin(2π*1t)+sin(2π*5t)+sin(2π*9t) and show MSE ~ 0.
  2. Via CFT: locate the three magnitude peaks of F(f) and read off their frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


class SignalGenerator:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def f_original(self):
        t = self.t
        return 2 * np.sin(14 * np.pi * t) - np.sin(2 * np.pi * t) * (
            4 * np.sin(2 * np.pi * t) * np.sin(14 * np.pi * t) - 1
        )

    def f_decomposed(self, freqs):
        """Sum of unit-amplitude, zero-phase sine components at the given freqs (Hz)."""
        t = self.t
        return sum(np.sin(2 * np.pi * fk * t) for fk in freqs)


class CFTAnalyzer:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def cft(self, x, f):
        x = np.asarray(x)
        f = np.asarray(f, dtype=float)
        X = np.empty(len(f), dtype=complex)
        for i, fi in enumerate(f):
            X[i] = trapz(x * np.exp(-1j * 2 * np.pi * fi * self.t), self.t)
        return X

    @staticmethod
    def mse(a, b):
        return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def find_peak_frequencies(f, mag, n_peaks=3, min_sep=1.0):
    """Simple greedy peak-picker on the (positive-frequency half of the) spectrum."""
    pos = f > 0.05
    f_pos, mag_pos = f[pos], mag[pos]
    order = np.argsort(mag_pos)[::-1]
    found = []
    for idx in order:
        fc = f_pos[idx]
        if all(abs(fc - fp) > min_sep for fp in found):
            found.append(fc)
        if len(found) == n_peaks:
            break
    return sorted(found)


if __name__ == "__main__":
    t = np.linspace(-5, 5, 4000)
    sg = SignalGenerator(t)

    f_orig = sg.f_original()

    # ---- direct verification of the algebraic decomposition ----------------------
    derived_freqs = [1.0, 5.0, 9.0]
    f_sum = sg.f_decomposed(derived_freqs)
    direct_mse = float(np.mean((f_orig - f_sum) ** 2))
    print(f"Direct-domain MSE  f(t)  vs  sin(2π·1t)+sin(2π·5t)+sin(2π·9t)  = {direct_mse:.3e}")

    plt.figure(figsize=(9, 4))
    plt.plot(t, f_orig, label="f(t) original", lw=2)
    plt.plot(t, f_sum, "--", label="sin(2π·1t)+sin(2π·5t)+sin(2π·9t)", lw=2)
    plt.title("Original signal vs. derived 3-tone decomposition")
    plt.xlabel("t"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

    # ---- CFT-based frequency identification --------------------------------------
    f = np.linspace(-15, 15, 3000)
    an = CFTAnalyzer(t)
    F = an.cft(f_orig, f)

    plt.figure(figsize=(9, 4))
    plt.plot(f, np.abs(F), lw=2)
    plt.title("Magnitude spectrum |F(f)| - expect 3 peak pairs at ±1, ±5, ±9 Hz")
    plt.xlabel("f (Hz)"); plt.ylabel("|F(f)|"); plt.grid(True)
    plt.tight_layout(); plt.show()

    peaks = find_peak_frequencies(f, np.abs(F), n_peaks=3, min_sep=1.5)
    print(f"Frequencies located from CFT magnitude peaks (Hz): {[round(p,2) for p in peaks]}")
    print(f"Analytically derived frequencies (Hz)            : {derived_freqs}")

    f_from_cft = sg.f_decomposed(peaks)
    cft_route_mse = an.mse(f_orig, f_from_cft)
    print(f"MSE using CFT-located frequencies                 = {cft_route_mse:.3e}")

    print("\nConclusion: f(t) = sin(2π·1·t) + sin(2π·5·t) + sin(2π·9·t) - three unit-\n"
          "amplitude, zero-phase sinusoids at 1 Hz, 5 Hz and 9 Hz, confirmed both by\n"
          "direct algebraic substitution and by locating the CFT magnitude peaks.")