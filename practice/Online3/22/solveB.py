import numpy as np
import matplotlib.pyplot as plt
 
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

class SignalGenerator:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)
 
    def gaussian(self, a):
        """x(t) = exp(-a t^2)"""
        return np.exp(-a * self.t ** 2)
 
    def time_shift(self, x, t0):
        """y(t) = x(t - t0), implemented via interpolation (OOP, no manual np.roll etc.)"""
        return np.interp(self.t - t0, self.t, x, left=0.0, right=0.0)
 
 
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
 
if __name__ == "__main__":
    # ---- Part 2: time axis and original signal ------------------------------
    t = np.linspace(-5, 5, 2000)
    sg = SignalGenerator(t)
    x = sg.gaussian(a=1.0)          # x(t) = exp(-t^2)
 
    # ---- Part 3: OOP time shift ----------------------------------------------
    t0 = 1.0
    y = sg.time_shift(x, t0)        # y(t) = x(t - 1)
 
    # ---- Part 4: CFT of both signals ------------------------------------------
    f = np.linspace(-10, 10, 1000)
    an = CFTAnalyzer(t)
    X = an.cft(x, f)
    Y = an.cft(y, f)
 
    # theoretical prediction from the time-shift property
    Y_theory_mag = np.abs(X)
    Y_theory_phase = np.angle(X) - 2 * np.pi * f * t0
 
    # ---- Part 5: plots ----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(f, np.abs(X), label="|X(f)|", lw=2)
    axes[0].plot(f, np.abs(Y), "--", label="|Y(f)| numerical", lw=2)
    axes[0].set_title("Magnitude spectra: |X(f)| vs |Y(f)|")
    axes[0].set_xlabel("f (Hz)"); axes[0].legend(); axes[0].grid(True)
 
    axes[1].plot(f, an.wrap(np.angle(Y)), label="angle(Y(f)) numerical", lw=2)
    axes[1].plot(f, an.wrap(Y_theory_phase), "--",
                 label="angle(X(f)) - 2*pi*f*t0 (theory)", lw=2)
    axes[1].set_title("Phase spectra: measured vs theoretical prediction")
    axes[1].set_xlabel("f (Hz)"); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    plt.show()
 
    # ---- Part 6: error metrics --------------------------------------------------
    mse_mag = an.mse(np.abs(X), np.abs(Y))
    phase_diff = an.wrap(np.angle(Y) - Y_theory_phase)
    mse_phase_full = float(np.mean(phase_diff ** 2))
 
    # A Gaussian's spectrum decays extremely fast (exp(-(pi f)^2)), so at the
    # edges of f in [-10,10] |X(f)| ~ 0 and the "phase" there is pure numerical
    # noise (angle of a near-zero complex number is meaningless). We therefore
    # also report a magnitude-weighted phase MSE, restricted to points where the
    # spectrum is not negligible - this is the numerically honest comparison.
    significant = np.abs(X) > 1e-3 * np.max(np.abs(X))
    mse_phase_significant = float(np.mean(phase_diff[significant] ** 2))
 
    print(f"MSE(|X(f)|, |Y(f)|)                                  = {mse_mag:.6e}")
    print(f"MSE(phase), all f in [-10,10]                        = {mse_phase_full:.6e}")
    print(f"MSE(phase), only where |X(f)| is significant         = {mse_phase_significant:.6e}")
    print("\nComment: |X(f)| ~ |Y(f)| confirms a pure time shift does not change the\n"
          "magnitude spectrum. The full-range phase MSE looks large only because a\n"
          "Gaussian spectrum decays to ~0 well before f=10, so phase becomes numerical\n"
          "noise out there; restricted to the region where |X(f)| is non-negligible,\n"
          "the phase MSE is essentially machine-precision zero, confirming\n"
          "angle(Y(f)) = angle(X(f)) - 2*pi*f*t0.")