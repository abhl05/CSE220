"""
signal_operations.py
=====================

A small, generic toolkit implementing the signal concepts covered in
"Lecture 1: A Little Introduction to Signals" (CSE 219 - Signals and
Linear Systems).

Covers:
    - Continuous signals x(t)            -> represented as Python callables
    - Discrete signals   x[n]            -> represented as (n, x) numpy arrays
    - Energy and average power           (both continuous & discrete)
    - Time shifting (delay / advance)
    - Time reversal
    - Time scaling (compression / stretching)
    - The general transform x(alpha*t + beta)  (shift -> reverse -> scale)
    - Even / odd decomposition and even/odd checks
    - Simple plotting helpers (matplotlib)

Design notes
------------
Continuous signals are represented as ordinary Python functions/lambdas
``x(t)`` that accept a numpy array (or scalar) ``t`` and return the same
shape (i.e. they should be numpy-vectorized, e.g. built with ``np.sin``,
``np.where`` etc.). All continuous-signal operations here return *new*
callables, so you can freely chain them:

    >>> x = lambda t: np.where((t >= 0) & (t <= np.pi), np.sin(t), 0.0)
    >>> y = time_scale(time_shift(x, t0=2), alpha=3)   # x(3t + 2) family
    >>> y(np.linspace(-5, 5, 11))

Discrete signals are represented as a ``DiscreteSignal`` dataclass holding
an index array ``n`` (integers) and a value array ``x`` of the same length.
This keeps track of *where* each sample lives (important since discrete
shifting/scaling changes which integer index a value sits at).

Two convenience classes, ``ContinuousSignal`` and ``DiscreteSignal``, wrap
the underlying functions so operations can be chained with methods, e.g.
``x.shift(2).reverse().scale(3)``. The plain functions are also exposed
standalone for anyone who prefers a functional style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# numpy >= 2.0 renamed trapz -> trapezoid; keep this module working on both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")

ArrayLike = Union[np.ndarray, float, int]
ContinuousFunc = Callable[[ArrayLike], ArrayLike]


# ---------------------------------------------------------------------------
# 1. ENERGY AND POWER  (slides 7, 8, 9, 10, 11, 12)
# ---------------------------------------------------------------------------

def energy_continuous(
    x_func: ContinuousFunc,
    t1: float,
    t2: float,
    num_points: int = 200_000,
) -> float:
    """
    Compute the energy of a continuous signal over [t1, t2]:

        E = integral_{t1}^{t2} |x(t)|^2 dt

    Parameters
    ----------
    x_func : callable
        The signal x(t). Must accept a numpy array and return a numpy array
        (real or complex).
    t1, t2 : float
        Integration limits.
    num_points : int
        Number of samples used for the numerical (trapezoidal) integration.
        Increase this for signals with fine detail or discontinuities.

    Returns
    -------
    float
        The energy E[t1, t2].
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")
    t = np.linspace(t1, t2, num_points)
    x = np.asarray(x_func(t))
    integrand = np.abs(x) ** 2
    return float(_trapz(integrand, t))


def power_continuous(
    x_func: ContinuousFunc,
    t1: float,
    t2: float,
    num_points: int = 200_000,
) -> float:
    """
    Compute the average power of a continuous signal over [t1, t2]:

        P = E[t1, t2] / (t2 - t1)
    """
    E = energy_continuous(x_func, t1, t2, num_points=num_points)
    return E / (t2 - t1)


def energy_discrete(
    x: np.ndarray,
    n: Optional[np.ndarray] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
) -> float:
    """
    Compute the energy of a discrete-time signal over the index range
    [n1, n2] (inclusive):

        E = sum_{n=n1}^{n2} |x[n]|^2

    Parameters
    ----------
    x : array_like
        The full array of sample values.
    n : array_like, optional
        The integer index for each sample in x (same length as x).
        If omitted, indices 0..len(x)-1 are assumed.
    n1, n2 : int, optional
        Sub-range of indices to sum over (inclusive). Defaults to the
        full extent of ``n``.

    Returns
    -------
    float
        The energy E[n1, n2].
    """
    x = np.asarray(x)
    if n is None:
        n = np.arange(len(x))
    else:
        n = np.asarray(n)

    if n1 is None:
        n1 = n.min()
    if n2 is None:
        n2 = n.max()

    mask = (n >= n1) & (n <= n2)
    return float(np.sum(np.abs(x[mask]) ** 2))


def power_discrete(
    x: np.ndarray,
    n: Optional[np.ndarray] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
) -> float:
    """
    Compute the average power of a discrete-time signal over [n1, n2]:

        P = (1 / (n2 - n1 + 1)) * sum_{n=n1}^{n2} |x[n]|^2
    """
    x = np.asarray(x)
    if n is None:
        n = np.arange(len(x))
    else:
        n = np.asarray(n)

    if n1 is None:
        n1 = n.min()
    if n2 is None:
        n2 = n.max()

    E = energy_discrete(x, n, n1, n2)
    num_samples = n2 - n1 + 1
    return E / num_samples


# ---------------------------------------------------------------------------
# 2. TRANSFORMING CONTINUOUS SIGNALS  (slides 14-27)
# ---------------------------------------------------------------------------

def amplitude_scale(x_func: ContinuousFunc, gain: float) -> ContinuousFunc:
    """Return the signal gain * x(t)  (vertical scaling, slide 14)."""
    return lambda t: gain * np.asarray(x_func(t))


def time_shift(x_func: ContinuousFunc, t0: float) -> ContinuousFunc:
    """
    Return x(t - t0).

    t0 > 0 -> delay (shifts the graph to the RIGHT)
    t0 < 0 -> advance (shifts the graph to the LEFT)

    (slides 15, 16)
    """
    return lambda t: x_func(np.asarray(t) - t0)


def time_reverse(x_func: ContinuousFunc) -> ContinuousFunc:
    """Return x(-t): mirror the signal about the vertical axis t = 0 (slide 18-19)."""
    return lambda t: x_func(-np.asarray(t))


def time_scale(x_func: ContinuousFunc, alpha: float) -> ContinuousFunc:
    """
    Return x(alpha * t).

    |alpha| > 1  -> time compression (signal looks "faster"/narrower)
    0<|alpha|<1  -> time stretching  (signal looks "slower"/wider)
    alpha < 0    -> also flips the signal in time

    (slides 20-21)
    """
    if alpha == 0:
        raise ValueError("alpha cannot be zero (division/scale by zero).")
    return lambda t: x_func(alpha * np.asarray(t))


def transform_signal(x_func: ContinuousFunc, alpha: float, beta: float) -> ContinuousFunc:
    """
    Return x(alpha * t + beta) using the 3-step recipe from the lecture
    (slides 24-27):

        Step 1: g1(t) = x(t + beta)                 -- shift
        Step 2: g2(t) = g1(t)  if alpha > 0
                        g1(-t) if alpha < 0          -- reverse (only if needed)
        Step 3: g3(t) = g2(|alpha| * t) = x(alpha*t + beta)   -- scale

    This mirrors the exam-style intermediate steps taught in the slides,
    rather than just directly composing x(alpha*t + beta) in one shot.

    Parameters
    ----------
    x_func : callable
        Original signal x(t).
    alpha : float
        Scaling factor (can be negative -> reversal + scaling).
    beta : float
        Shift amount added inside the argument (x(t + beta) in step 1).

    Returns
    -------
    callable
        The fully transformed signal g3(t) = x(alpha * t + beta).
    """
    if alpha == 0:
        raise ValueError("alpha cannot be zero.")

    g1 = time_shift(x_func, -beta)          # x(t + beta) == time_shift(x, t0=-beta)
    g2 = g1 if alpha > 0 else time_reverse(g1)
    g3 = time_scale(g2, abs(alpha))
    return g3


def time_shift_discrete(
    x: np.ndarray,
    n: np.ndarray,
    n0: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift a discrete signal: y[n] = x[n - n0].

    n0 > 0 -> delay (samples occur later / shift right)
    n0 < 0 -> advance (samples occur earlier / shift left)

    Returns
    -------
    (n_new, x) : tuple of arrays
        The same values x, but living at shifted indices n_new = n + n0.
    """
    n = np.asarray(n)
    x = np.asarray(x)
    return n + n0, x.copy()


def time_reverse_discrete(
    x: np.ndarray,
    n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reverse a discrete signal: y[n] = x[-n].

    Returns
    -------
    (n_new, x_new) : tuple of arrays
        Indices are negated and values are reordered to stay sorted by
        increasing index.
    """
    n = np.asarray(n)
    x = np.asarray(x)
    n_new = -n
    order = np.argsort(n_new)
    return n_new[order], x[order]


# ---------------------------------------------------------------------------
# 3. EVEN / ODD DECOMPOSITION  (slides 36-45)
# ---------------------------------------------------------------------------

def even_part_continuous(x_func: ContinuousFunc) -> ContinuousFunc:
    """Return x_e(t) = 1/2 [x(t) + x(-t)]  (slide 39, 43)."""
    return lambda t: 0.5 * (np.asarray(x_func(t)) + np.asarray(x_func(-np.asarray(t))))


def odd_part_continuous(x_func: ContinuousFunc) -> ContinuousFunc:
    """Return x_o(t) = 1/2 [x(t) - x(-t)]  (slide 39, 44)."""
    return lambda t: 0.5 * (np.asarray(x_func(t)) - np.asarray(x_func(-np.asarray(t))))


def is_even_continuous(x_func: ContinuousFunc, t_test: np.ndarray, atol: float = 1e-8) -> bool:
    """Check x(-t) == x(t) numerically over a set of test points t_test."""
    t_test = np.asarray(t_test)
    return bool(np.allclose(x_func(-t_test), x_func(t_test), atol=atol))


def is_odd_continuous(x_func: ContinuousFunc, t_test: np.ndarray, atol: float = 1e-8) -> bool:
    """Check x(-t) == -x(t) numerically over a set of test points t_test."""
    t_test = np.asarray(t_test)
    return bool(np.allclose(x_func(-t_test), -np.asarray(x_func(t_test)), atol=atol))


def even_odd_parts_discrete(
    x: np.ndarray,
    n: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the even and odd parts of a discrete signal defined on an
    index array n (need not be symmetric -- values outside the original
    support are treated as 0, matching the "otherwise: 0" convention used
    throughout the slides).

    Returns
    -------
    (n_common, x_even, x_odd)
        A common index array (the union of n and -n) together with the
        even and odd parts evaluated at those indices.
    """
    n = np.asarray(n)
    x = np.asarray(x, dtype=float)

    n_common = np.union1d(n, -n).astype(int)

    # Build a lookup so we can evaluate x[k] for any integer k (0 if absent)
    lookup = dict(zip(n.tolist(), x.tolist()))

    def x_at(k: int) -> float:
        return lookup.get(int(k), 0.0)

    x_full = np.array([x_at(k) for k in n_common])
    x_flip = np.array([x_at(-k) for k in n_common])

    x_even = 0.5 * (x_full + x_flip)
    x_odd = 0.5 * (x_full - x_flip)
    return n_common, x_even, x_odd


def is_even_discrete_simple(x: np.ndarray, n: np.ndarray, atol: float = 1e-8) -> bool:
    """Directly check x[n] == x[-n] at every sample where both are defined."""
    n = np.asarray(n)
    x = np.asarray(x, dtype=float)
    lookup = dict(zip(n.tolist(), x.tolist()))
    ok = True
    for k, val in zip(n.tolist(), x.tolist()):
        other = lookup.get(int(-k), 0.0)
        if not np.isclose(val, other, atol=atol):
            ok = False
            break
    return ok


def is_odd_discrete_simple(x: np.ndarray, n: np.ndarray, atol: float = 1e-8) -> bool:
    """Directly check x[n] == -x[-n] at every sample where both are defined."""
    n = np.asarray(n)
    x = np.asarray(x, dtype=float)
    lookup = dict(zip(n.tolist(), x.tolist()))
    ok = True
    for k, val in zip(n.tolist(), x.tolist()):
        other = lookup.get(int(-k), 0.0)
        if not np.isclose(val, -other, atol=atol):
            ok = False
            break
    return ok


# ---------------------------------------------------------------------------
# 4. PLOTTING HELPERS
# ---------------------------------------------------------------------------

def plot_continuous_signal(
    x_func: ContinuousFunc,
    t1: float,
    t2: float,
    num_points: int = 2000,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot a continuous signal x(t) over [t1, t2] as a smooth curve."""
    t = np.linspace(t1, t2, num_points)
    x = np.asarray(x_func(t))

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    ax.plot(t, x, **plot_kwargs)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_discrete_signal(
    x: np.ndarray,
    n: Optional[np.ndarray] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a discrete signal x[n] as a stem plot."""
    x = np.asarray(x)
    if n is None:
        n = np.arange(len(x))
    else:
        n = np.asarray(n)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    ax.stem(n, x, basefmt=" ")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("n")
    ax.set_ylabel("x[n]")
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# 5. OBJECT-ORIENTED WRAPPERS (optional, convenience for chaining operations)
# ---------------------------------------------------------------------------

@dataclass
class ContinuousSignal:
    """
    Thin OO wrapper around a continuous-time signal function, so operations
    can be chained fluently:

        x = ContinuousSignal(lambda t: np.sin(t))
        y = x.shift(2).reverse().scale(3)   # x(-3t + ... ) family
    """
    func: ContinuousFunc

    def __call__(self, t: ArrayLike) -> ArrayLike:
        return self.func(t)

    def shift(self, t0: float) -> "ContinuousSignal":
        return ContinuousSignal(time_shift(self.func, t0))

    def reverse(self) -> "ContinuousSignal":
        return ContinuousSignal(time_reverse(self.func))

    def scale(self, alpha: float) -> "ContinuousSignal":
        return ContinuousSignal(time_scale(self.func, alpha))

    def amplitude_scale(self, gain: float) -> "ContinuousSignal":
        return ContinuousSignal(amplitude_scale(self.func, gain))

    def transform(self, alpha: float, beta: float) -> "ContinuousSignal":
        """Return x(alpha * t + beta) directly (shift -> reverse -> scale)."""
        return ContinuousSignal(transform_signal(self.func, alpha, beta))

    def even_part(self) -> "ContinuousSignal":
        return ContinuousSignal(even_part_continuous(self.func))

    def odd_part(self) -> "ContinuousSignal":
        return ContinuousSignal(odd_part_continuous(self.func))

    def is_even(self, t_test: np.ndarray, atol: float = 1e-8) -> bool:
        return is_even_continuous(self.func, t_test, atol=atol)

    def is_odd(self, t_test: np.ndarray, atol: float = 1e-8) -> bool:
        return is_odd_continuous(self.func, t_test, atol=atol)

    def energy(self, t1: float, t2: float, num_points: int = 200_000) -> float:
        return energy_continuous(self.func, t1, t2, num_points=num_points)

    def power(self, t1: float, t2: float, num_points: int = 200_000) -> float:
        return power_continuous(self.func, t1, t2, num_points=num_points)

    def plot(self, t1: float, t2: float, num_points: int = 2000, title: str = "", ax=None, **kwargs):
        return plot_continuous_signal(self.func, t1, t2, num_points=num_points, title=title, ax=ax, **kwargs)


@dataclass
class DiscreteSignal:
    """
    Thin OO wrapper around a discrete-time signal (n, x pair), so operations
    can be chained fluently:

        x = DiscreteSignal(n=np.array([0,1,2,3]), x=np.array([1,-2,2,-1]))
        y = x.shift(4).reverse()
    """
    n: np.ndarray
    x: np.ndarray

    def __post_init__(self):
        self.n = np.asarray(self.n)
        self.x = np.asarray(self.x, dtype=float)
        if self.n.shape != self.x.shape:
            raise ValueError("n and x must have the same shape.")

    def shift(self, n0: int) -> "DiscreteSignal":
        n_new, x_new = time_shift_discrete(self.x, self.n, n0)
        return DiscreteSignal(n_new, x_new)

    def reverse(self) -> "DiscreteSignal":
        n_new, x_new = time_reverse_discrete(self.x, self.n)
        return DiscreteSignal(n_new, x_new)

    def amplitude_scale(self, gain: float) -> "DiscreteSignal":
        return DiscreteSignal(self.n.copy(), gain * self.x)

    def energy(self, n1: Optional[int] = None, n2: Optional[int] = None) -> float:
        return energy_discrete(self.x, self.n, n1, n2)

    def power(self, n1: Optional[int] = None, n2: Optional[int] = None) -> float:
        return power_discrete(self.x, self.n, n1, n2)

    def even_odd_parts(self) -> Tuple["DiscreteSignal", "DiscreteSignal"]:
        n_common, x_even, x_odd = even_odd_parts_discrete(self.x, self.n)
        return DiscreteSignal(n_common, x_even), DiscreteSignal(n_common, x_odd)

    def is_even(self, atol: float = 1e-8) -> bool:
        return is_even_discrete_simple(self.x, self.n, atol=atol)

    def is_odd(self, atol: float = 1e-8) -> bool:
        return is_odd_discrete_simple(self.x, self.n, atol=atol)

    def plot(self, title: str = "", ax=None):
        return plot_discrete_signal(self.x, self.n, title=title, ax=ax)


# ---------------------------------------------------------------------------
# 6. DEMO  (reproduces the worked examples from the slides)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Slide 9-10: half-sine energy/power example ---
    half_sine = lambda t: np.where((t >= 0) & (t <= np.pi), np.sin(t), 0.0)
    E = energy_continuous(half_sine, 0, np.pi)
    P = power_continuous(half_sine, 0, np.pi)
    print(f"Half-sine energy  E = {E:.4f}  (expected pi/2 = {np.pi/2:.4f})")
    print(f"Half-sine power   P = {P:.4f}  (expected 0.5)")

    # --- Slide 11-12: discrete energy/power example ---
    n = np.array([0, 1, 2, 3])
    x = np.array([1, -2, 2, -1])
    print(f"Discrete energy E = {energy_discrete(x, n)}  (expected 10)")
    print(f"Discrete power  P = {power_discrete(x, n)}  (expected 2.5)")

    # --- Slide 27: x(-3t + 2) via the 3-step recipe ---
    x_signal = ContinuousSignal(half_sine)
    y_signal = x_signal.transform(alpha=-3, beta=2)
    t_vals = np.linspace(-2, 2, 5)
    print("x(-3t + 2) sample values:", y_signal(t_vals))

    # --- Slide 41-44: even/odd decomposition example ---
    tri = lambda t: np.where((t >= -1) & (t <= 1), t + 1, 0.0)
    tri_signal = ContinuousSignal(tri)
    even_sig = tri_signal.even_part()
    odd_sig = tri_signal.odd_part()
    t_check = np.linspace(-1, 1, 5)
    print("Even part samples:", even_sig(t_check), " (expected all 1)")
    print("Odd part samples :", odd_sig(t_check), " (expected == t)")

    # --- Uncomment to visualize ---
    # fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    # plot_continuous_signal(half_sine, -1, 4, title="x(t) = half sine", ax=axes[0, 0])
    # plot_continuous_signal(y_signal.func, -2, 2, title="x(-3t + 2)", ax=axes[0, 1])
    # plot_discrete_signal(x, n, title="x[n]", ax=axes[1, 0])
    # n_e, x_e, x_o = even_odd_parts_discrete(x, n)
    # plot_discrete_signal(x_e, n_e, title="Even part of x[n]", ax=axes[1, 1])
    # plt.tight_layout()
    # plt.show()
