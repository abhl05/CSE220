import numpy as np
import matplotlib.pyplot as plt

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

class SignalGenerator:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)
 
    def x_original(self):
        """x(t) = 0.5 cos(4t) + 0.5 sin(6t)"""
        return 0.5 * np.cos(4 * self.t) + 0.5 * np.sin(6 * self.t)
 
    def dx_dt(self, order):
        """
        Analytic derivatives of x(t), computed symbolically by hand and evaluated
        numerically (exact - not a finite-difference approximation):
            x   =  0.5 cos(4t) + 0.5 sin(6t)
            x'  = -2   sin(4t) + 3   cos(6t)
            x'' = -8   cos(4t) - 18  sin(6t)
            x'''=  32  sin(4t) - 108 cos(6t)
        """
        t = self.t
        if order == 1:
            return -2 * np.sin(4 * t) + 3 * np.cos(6 * t)
        elif order == 2:
            return -8 * np.cos(4 * t) - 18 * np.sin(6 * t)
        elif order == 3:
            return 32 * np.sin(4 * t) - 108 * np.cos(6 * t)
        raise ValueError("order must be 1, 2 or 3")

class CFTAnalyzer:
    def __init__(self, t):
        self.t = np.asarray(t, dtype=float)
 
    def cft(self, x, f):
        x = np.asarray(x)
        X = np.empty(len(f), dtype=complex)
        for i, fi in enumerate(f):
            X[i] = trapz(x * np.exp(-1j * 2 * np.pi * fi * self.t), self.t)
        return X
 
    @staticmethod
    def mse(a, b):
        return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
 
    @staticmethod
    def phase_mse(pa, pb):
        d = (np.asarray(pa) - np.asarray(pb) + np.pi) % (2 * np.pi) - np.pi
        return float(np.mean(d ** 2))
    
if __name__ == "__main__":
    # x(t) is a pure (non-decaying) sum of sinusoids, so truncating it to a finite
    # window is itself an approximation: the CFT peaks become tall, narrow sinc
    # lobes rather than true Dirac deltas. A WIDER window -> narrower/taller lobes
    # -> smaller relative leakage error, so we use a long window here.
    T = 200
    t = np.linspace(-T, T, int(T * 500))
    f = np.linspace(-2, 2, 2000)   # tones sit near f ~ 0.64 Hz and 0.95 Hz
 
    sg = SignalGenerator(t)
    an = CFTAnalyzer(t)
 
    x = sg.x_original()
    X = an.cft(x, f)
 
    print(f"{'order':>5} | {'mag MSE':>12} | {'mag MSE (rel.)':>15} | {'phase MSE':>12}")
    for n in (1, 2, 3):
        y_n = sg.dx_dt(n)                       # analytic derivative signal
        Y_n = an.cft(y_n, f)                    # its numerical CFT
        theory_n = (1j * 2 * np.pi * f) ** n * X  # property applied to X(f)
 
        mag_mse = an.mse(np.abs(Y_n), np.abs(theory_n))
        # Because the peaks scale like (2*pi*f)^n, raw MSE grows with n even when
        # the FIT is excellent - so also report MSE normalized by peak power.
        rel_mse = mag_mse / np.max(np.abs(theory_n) ** 2)
        ph_mse = an.phase_mse(np.angle(Y_n), np.angle(theory_n))
        print(f"{n:>5} | {mag_mse:12.3e} | {rel_mse:15.3e} | {ph_mse:12.3e}")
 
        fig, axes = plt.subplots(2, 1, figsize=(9, 6))
        axes[0].plot(f, np.abs(Y_n), label=f"|Y{n}(f)| numerical", lw=2)
        axes[0].plot(f, np.abs(theory_n), "--", label=f"|(j2πf)^{n} X(f)| theory", lw=2)
        axes[0].set_title(f"Order {n}: Magnitude overlap"); axes[0].legend(); axes[0].grid(True)
 
        axes[1].plot(f, np.angle(Y_n), label="numerical phase", lw=2)
        axes[1].plot(f, np.angle(theory_n), "--", label="theoretical phase", lw=2)
        axes[1].set_title(f"Order {n}: Phase overlap"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout()
        plt.show()