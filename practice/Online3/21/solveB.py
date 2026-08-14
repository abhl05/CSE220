"""
CSE 220: Online B1-B2
Topic   : Parseval's theorem for a piecewise-parabolic signal
          Integral |x(t)|^2 dt  =  Integral |X(f)|^2 df

Signal (from the figure): a symmetric bump on t in [-3,3], peak value 5 at t=0,
built from two parabola pieces, zero for |t|>3:
    - |t| <= 1 :  x(t) = 5 - 2 t^2                 (peak 5 at 0, value 3 at |t|=1)
    - 1<=|t|<=3:  x(t) = 3 * ((3-|t|)/2)^2          (continuous with the above,
                                                       reaches 0 at |t|=3)
    - |t|  > 3 :  x(t) = 0

*** IMPORTANT ASSUMPTION ***
The exact coefficients of the two parabola pieces were not fully legible from the
scanned figure (only "peak 5 at t=0" and "reaches 0 by |t|=3" are certain). The two
formulas above were chosen to (a) be genuine parabolas, (b) match those two visible
facts, and (c) be continuous at the t=+-1 junction. If your copy of the figure shows
different numbers, just edit `left_piece` / `right_piece` below - the rest of the
pipeline (FT/IFT/Parseval) does not change.
"""

import numpy as np
import matplotlib.pyplot as plt

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ----------------------------- OOP framework (from your offline) ------------------
class SignalGenerator:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def piecewise_parabola(self):
        """Builds the bump described above. Edit the two parabola formulas here
        if your exact figure differs."""
        t = self.t
        x = np.zeros_like(t)

        inner = np.abs(t) <= 1
        outer = (np.abs(t) > 1) & (np.abs(t) <= 3)

        left_piece = 5 - 2 * t[inner] ** 2
        right_piece = 3 * ((3 - np.abs(t[outer])) / 2) ** 2

        x[inner] = left_piece
        x[outer] = right_piece
        return x


class CFTAnalyzer:
    """Forward/Inverse CFT via trapz only (mirrors your offline FT/IFT code)."""

    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def cft(self, x, f):
        x = np.asarray(x)
        f = np.asarray(f, dtype=float)
        X = np.empty(len(f), dtype=complex)
        for i, fi in enumerate(f):
            X[i] = trapz(x * np.exp(-1j * 2 * np.pi * fi * self.t), self.t)
        return X

    def icft(self, X, f, t_out=None):
        if t_out is None:
            t_out = self.t
        X = np.asarray(X)
        f = np.asarray(f, dtype=float)
        x = np.empty(len(t_out), dtype=complex)
        for i, ti in enumerate(t_out):
            x[i] = trapz(X * np.exp(1j * 2 * np.pi * f * ti), f)
        return x

    @staticmethod
    def energy_time(t, x):
        return float(trapz(np.abs(x) ** 2, t))

    @staticmethod
    def energy_freq(f, X):
        return float(trapz(np.abs(X) ** 2, f))


# ----------------------------- main experiment -------------------------------------
if __name__ == "__main__":
    t = np.linspace(-10, 10, 4000)     # wide enough to safely include the [-3,3] support
    sg = SignalGenerator(t)
    x = sg.piecewise_parabola()

    plt.figure(figsize=(7, 4))
    plt.plot(t, x, lw=2)
    plt.title("Piecewise parabolic input signal x(t)")
    plt.xlabel("t"); plt.ylabel("x(t)"); plt.grid(True)
    plt.tight_layout(); plt.show()

    # frequency axis wide/fine enough to capture essentially all the signal energy
    f = np.linspace(-5, 5, 4000)
    an = CFTAnalyzer(t)
    X = an.cft(x, f)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))
    axes[0].plot(f, np.abs(X), lw=2)
    axes[0].set_title("Magnitude spectrum |X(f)|")
    axes[0].set_xlabel("f (Hz)"); axes[0].grid(True)
    axes[1].plot(f, np.angle(X), lw=2)
    axes[1].set_title("Phase spectrum angle(X(f))")
    axes[1].set_xlabel("f (Hz)"); axes[1].grid(True)
    plt.tight_layout(); plt.show()

    # ---- Parseval's theorem check --------------------------------------------
    E_time = an.energy_time(t, x)
    E_freq = an.energy_freq(f, X)
    rel_err = abs(E_time - E_freq) / E_time

    print(f"Energy in time domain   Integral |x(t)|^2 dt  = {E_time:.6f}")
    print(f"Energy in freq domain   Integral |X(f)|^2 df  = {E_freq:.6f}")
    print(f"Relative error                                = {rel_err:.3e}")

    # ---- sanity check: IFT should reconstruct x(t) -----------------------------
    x_rec = an.icft(X, f, t)
    recon_mse = float(np.mean((np.abs(x_rec) - x) ** 2))
    print(f"Reconstruction MSE (IFT(CFT(x)) vs x)          = {recon_mse:.3e}")

    print("\nComment: the relative error between time- and frequency-domain energy is\n"
          "expected to be small (well under 1%) with a fine grid and a frequency axis\n"
          "wide enough to capture the spectrum's significant energy, confirming\n"
          "Parseval's theorem for this signal.")