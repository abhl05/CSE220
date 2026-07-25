import numpy as np
import matplotlib.pyplot as plt

def sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float) -> np.ndarray:
    """Computes the discrete-time sinusoid x[n] = A cos(Omega0*n + phi)."""
    return A * np.cos(Omega0 * n + phi)


def time_shift_sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float, n0: int) -> np.ndarray:
    """Computes the time-shifted sinusoid x[n - n0]."""
    return A * np.cos(Omega0 * (n - n0) + phi)


def phase_change_sinusoid(n: np.ndarray, A: float, Omega0: float, phi: float, phi0: float) -> np.ndarray:
    """Computes the phase-changed sinusoid with an additional phase phi0."""
    return A * np.cos(Omega0 * n + phi + phi0)


# -----------------------------
# 2) Utility functions
# -----------------------------
def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two sequences of equal length."""
    return float(np.mean((a - b) ** 2))


def stem_plot(ax, n, x, label):
    """A nicer stem plot for discrete-time sequences."""
    markerline, stemlines, baseline = ax.stem(n, x, label=label)
    baseline.set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("n")
    ax.set_ylabel("Amplitude")


# -----------------------------
# 3) Main experiment
# -----------------------------
def main():
    # Base sinusoid parameters (you may change these to experiment)
    A = 1.0
    Omega0 = np.pi / 4
    phi = 0.0

    # Index range
    n = np.arange(-20, 21)  # -20, -19, ..., 20

    # Original signal
    x = sinusoid(n, A, Omega0, phi)

    # ---------------------------------------------------------
    # PART A: Time Shift -> Phase Change (Task 4)
    # ---------------------------------------------------------
    n0 = 2  # integer time shift (delay of 2 samples)
    x_time = time_shift_sinusoid(n, A, Omega0, phi, n0)

    # Compute the phase shift phi0_equiv that makes x_phase match x_time
    phi0_equiv = -Omega0 * n0 

    x_phase_equiv = phase_change_sinusoid(n, A, Omega0, phi, phi0_equiv)

    err_A = mse(x_time, x_phase_equiv)
    print(f"[Part A] MSE between time-shifted and equivalent phase-changed: {err_A:.10f}")

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    stem_plot(ax1, n, x, "original x[n]")
    stem_plot(ax1, n, x_time, f"time shift by n0={n0}")
    
    # We plot this with a different marker/color just to show it perfectly overlaps
    markerline, stemlines, baseline = ax1.stem(n, x_phase_equiv, linefmt='r--', markerfmt='rx', label=f"phase change by phi0={phi0_equiv:.3f}")
    baseline.set_visible(False)
    
    ax1.legend()
    ax1.set_title("Task 4: Time Shift corresponds to an EXACT Phase Change")
    fig1.tight_layout()

    # ---------------------------------------------------------
    # PART B: Phase Change -> Time Shift (Task 5)
    # ---------------------------------------------------------
    # Choose an arbitrary phase change that doesn't perfectly align with a discrete shift
    phi0 = 1.0  
    x_phase = phase_change_sinusoid(n, A, Omega0, phi, phi0)

    # Search over integer shifts to see if any time shift matches this phase change
    k_min, k_max = -12, 12
    best_k = None
    best_err = None

    for k in range(k_min, k_max + 1):
        x_time_k = time_shift_sinusoid(n, A, Omega0, phi, k)
        e = mse(x_time_k, x_phase)
        if (best_err is None) or (e < best_err):
            best_err = e
            best_k = k

    print(f"[Part B] Best matching integer shift in [{k_min},{k_max}] is k={best_k} with MSE={best_err:.5f}")

    x_time_best = time_shift_sinusoid(n, A, Omega0, phi, best_k)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    stem_plot(ax2, n, x_phase, f"phase change by phi0={phi0:.3f}")
    
    markerline, stemlines, baseline = ax2.stem(n, x_time_best, linefmt='r--', markerfmt='rx', label=f"best time shift k={best_k}")
    baseline.set_visible(False)
    
    ax2.legend()
    ax2.set_title("Task 5: Arbitrary Phase Change does NOT always have an exact Integer Time Shift")
    fig2.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    main()