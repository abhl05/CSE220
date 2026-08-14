"""
CSE 220 - Online on CFT | Subsection C1+C2 (July 2025)
Topic   : Combined time-scaling + frequency-shift (modulation) property
Property: y(t) = x(a t) * e^{j 2*pi*f0*t}   <-->   Y(f) = (1/|a|) * X( (f-f0)/a )

x(t) = Square(t) + Triangle(t)   (fundamental frequency assumed 1 Hz - not stated
       explicitly in the problem, so this is a documented, easily-changed assumption)

Operations requested:
  i.  phase-shift (modulate) by exp(j*2*pi*f0*t), f0 = 10
  ii. compress the time axis by a = 10   -> x(a t)
  y(t) is the combination of both: y(t) = x(a t) * e^{j*2*pi*f0*t}
"""

import numpy as np
import matplotlib.pyplot as plt

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


# ----------------------------- OOP framework --------------------------------------
class SignalGenerator:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)

    def square_wave(self, freq, amplitude=1.0):
        return amplitude * np.sign(np.sin(2 * np.pi * freq * self.t))

    def triangle_wave(self, freq, amplitude=1.0):
        return (2 * amplitude / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * self.t))

    def time_scale(self, x, a):
        """y(t) = x(a t) via interpolation (OOP - not a manual array reshuffle)."""
        return np.interp(a * self.t, self.t, x, left=0.0, right=0.0)

    def modulate(self, x, f0):
        """y(t) = x(t) * exp(j*2*pi*f0*t)"""
        return x * np.exp(1j * 2 * np.pi * f0 * self.t)

    def scale_then_modulate(self, x, a, f0):
        """y(t) = x(a t) * exp(j*2*pi*f0*t)"""
        return self.modulate(self.time_scale(x, a), f0)


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

    @staticmethod
    def wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def interp_complex(f_target, f_src, X_src):
    """Interpolate a complex spectrum X_src(f_src) onto new points f_target,
    needed to evaluate X((f-f0)/a) at points that fall between the original grid."""
    re = np.interp(f_target, f_src, X_src.real, left=0.0, right=0.0)
    im = np.interp(f_target, f_src, X_src.imag, left=0.0, right=0.0)
    return re + 1j * im


# ----------------------------- main experiment -------------------------------------
if __name__ == "__main__":
    t = np.linspace(-5, 5, 2000)
    sg = SignalGenerator(t)

    fund_freq = 1.0  # ASSUMPTION: fundamental frequency of Square(t)/Triangle(t) = 1 Hz
    x = sg.square_wave(fund_freq) + sg.triangle_wave(fund_freq)

    a, f0 = 10.0, 10.0
    y = sg.scale_then_modulate(x, a, f0)      # y(t) = x(a t) * exp(j*2*pi*f0*t)

    f = np.linspace(-10, 10, 1000)
    an = CFTAnalyzer(t)
    X = an.cft(x, f)
    Y = an.cft(y, f)

    # Theory: Y(f) = (1/|a|) * X( (f-f0)/a )
    f_shifted_scaled = (f - f0) / a
    X_theory_shifted = interp_complex(f_shifted_scaled, f, X) / abs(a)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(f, np.abs(Y), label="|Y(f)| numerical", lw=2)
    axes[0].plot(f, np.abs(X_theory_shifted), "--",
                 label="(1/|a|)|X((f-f0)/a)| theory", lw=2)
    axes[0].set_title("Magnitude: scaling shrinks bandwidth by a and modulation "
                       "re-centers it at f0")
    axes[0].set_xlabel("f (Hz)"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(f, an.wrap(np.angle(Y)), label="angle(Y(f)) numerical", lw=2)
    axes[1].plot(f, an.wrap(np.angle(X_theory_shifted)), "--",
                 label="angle(X((f-f0)/a)) theory", lw=2)
    axes[1].set_title("Phase comparison")
    axes[1].set_xlabel("f (Hz)"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    mse_mag = an.mse(np.abs(Y), np.abs(X_theory_shifted))
    significant = np.abs(X_theory_shifted) > 1e-3 * np.max(np.abs(X_theory_shifted))
    phase_diff = an.wrap(np.angle(Y) - np.angle(X_theory_shifted))
    mse_phase = float(np.mean(phase_diff[significant] ** 2))

    print(f"MSE magnitude  (|Y(f)|  vs  (1/|a|)|X((f-f0)/a)|)      = {mse_mag:.6e}")
    print(f"MSE phase, where spectrum significant                 = {mse_phase:.6e}")
    print("\nEffect of (i) modulation: multiplying by exp(j*2*pi*f0*t) shifts the entire\n"
          "spectrum to be centered at f0 (frequency-shift property).\n"
          "Effect of (ii) compression: replacing t by a*t (a>10) SPREADS the spectrum\n"
          "out by factor a and scales its height down by 1/|a| (time-scaling property:\n"
          "compressing in time broadens in frequency).")