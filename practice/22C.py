"""
    July 2025 CSE 220

    Online on Signals and Their Properties

    Subsections: C1, C2
    Total Marks: 10
    Total Time: 30 Minutes

    January 6, 2026

    In this assignment, you need to implement time reversal of a sampled continuous-time
    signal in Python. Then using this function, you need to decompose a given signal x(t) into
    even and odd components. You also need to plot the graphs of each of these signals.
    Let the base signal be x(t) (provided in the template file). You will compute and plot:

    x(t), x(−t), xe(t), xo(t).

    Tasks
    1. Generate the time axis t and compute x(t).
    2. Implement a function time reverse(...) that produces x(−t).
    3. Using only your time reverse(...) function, compute xe(t) and xo(t).
    4. Plot (with proper labels and legend) on the same figure: x(t), xe(t), and xo(t). Also
    make a separate plot of x(t) and x(−t).
    Marks Distribution
    • Plotting the graph of x(t): 1 Mark
    • Implementing time reversal: 2 Marks
    • Implementation of Even and Odd Decomposition: 4 Marks
    • Plotting the graph of x(−t), xe(t) and xo(t): 3 Marks
"""

import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX, N = -4.0, 4.0, 4001

def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t).
    """
    # Combination of components (can be replaced)
    # 1) Triangular pulse centered at 0
    tri0 = np.zeros_like(t, dtype=float)
    m0 = np.abs(t) <= 1.0
    tri0[m0] = 1.0 - np.abs(t[m0])

    # 2) Windowed ramp (odd-ish component)
    ramp = np.zeros_like(t, dtype=float)
    m1 = np.abs(t) <= 1.0
    ramp[m1] = t[m1]

    # 3) Shifted triangular pulse (breaks symmetry)
    tri_shift = np.zeros_like(t, dtype=float)
    u = t - 1.2
    m2 = np.abs(u) <= 1.0
    tri_shift[m2] = 1.0 - np.abs(u[m2])

    return tri0 + 0.6 * ramp + 0.4 * tri_shift


def time_reverse(x: np.ndarray) -> np.ndarray:
    """
    Given samples x(t), return samples of x(-t)
    """
    x_rev = np.flip(x)  # Flip the array to reverse time
    return x_rev

def even_odd_decompose(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Must call time_reverse(...) inside this function.
    """
    x_rev = time_reverse(x)
    xe = 0.5 * (x + x_rev)
    xo = 0.5 * (x - x_rev)
    return xe, xo


# ----------------------------
# Provided plotting (do not modify)
# ----------------------------
def plot_three(t: np.ndarray, x: np.ndarray, xe: np.ndarray, xo: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xe, label="xe(t)")
    plt.plot(t, xo, label="xo(t)")
    plt.title("Even–Odd Decomposition")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


def plot_pair(t: np.ndarray, x: np.ndarray, xr: np.ndarray):
    plt.figure()
    plt.plot(t, x, label="x(t)")
    plt.plot(t, xr, label="x(-t)")
    plt.title("Time Reversal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()


# ----------------------------
# Main (provided)
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    # Compute time reverse and even odd components
    xr = time_reverse(x)
    xe, xo = even_odd_decompose(x)

    # Plot x(t), x(-t), xe(t) and xo(t) using the previously defined functions
    plot_three(t, x, xe, xo)
    plot_pair(t, x, xr)

    plt.show()


if __name__ == "__main__":
    main()