"""
CSE220 Online on Convolution - A1, A2 (Step Response)

Given the step response s[n] of an LTI system, recover h[n] and compute the
output for any x[n] two ways:
  1. using ONLY the step response:  y_s[n] = (delta_x * s)[n]
  2. using the recovered impulse response: y_h[n] = (x * h)[n]
and verify they match.

Identities used:
  h[n]      = s[n] - s[n-1]        (s[-1] = 0)
  delta_x[n] = x[n] - x[n-1]        (x[-1] = 0)
  y[n]      = (delta_x * s)[n]
"""

import numpy as np
import matplotlib.pyplot as plt


# ---- Offline 1 implementation (pasted in, self-contained) ----

class Signal:
    def __init__(self, INF):
        self.INF = int(INF)
        self.values = [0.0] * (2 * self.INF + 1)

    def _index(self, t):
        idx = t + self.INF
        if idx < 0 or idx >= len(self.values):
            raise IndexError(f"time index {t} outside [-{self.INF}, {self.INF}]")
        return idx

    def set_value_at_time(self, t, value):
        self.values[self._index(t)] = float(value)

    def get_value_at_time(self, t):
        if t < -self.INF or t > self.INF:
            return 0.0
        return self.values[self._index(t)]

    def shift(self, k):
        k = int(k)
        shifted = Signal(self.INF)
        for n in range(-self.INF, self.INF + 1):
            src = n - k
            if -self.INF <= src <= self.INF:
                shifted.set_value_at_time(n, self.get_value_at_time(src))
        return shifted

    def add(self, other):
        INF = max(self.INF, other.INF)
        result = Signal(INF)
        for n in range(-INF, INF + 1):
            result.set_value_at_time(n, self.get_value_at_time(n) + other.get_value_at_time(n))
        return result

    def multiply(self, scalar):
        result = Signal(self.INF)
        result.values = [v * float(scalar) for v in self.values]
        return result

    def plot(self, title="Discrete Signal"):
        n_values = list(range(-self.INF, self.INF + 1))
        plt.figure()
        plt.stem(n_values, self.values)
        plt.title(title)
        plt.xlabel("n")
        plt.ylabel("value")
        plt.grid(True, alpha=0.3)
        plt.show()


class LTI_System:
    def __init__(self, impulse_response: Signal):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal: Signal):
        impulses = []
        coefficients = []
        INF = input_signal.INF
        for k in range(-INF, INF + 1):
            x_k = input_signal.get_value_at_time(k)
            if x_k != 0.0:
                delta = Signal(INF)
                delta.set_value_at_time(k, 1.0)
                delta.impulse_time = k
                impulses.append(delta)
                coefficients.append(x_k)
        return impulses, coefficients

    def output(self, input_signal: Signal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        h = self.impulse_response
        INF = max(input_signal.INF, h.INF)
        y = Signal(INF)
        for delta, x_k in zip(impulses, coefficients):
            k = delta.impulse_time
            shifted_h = h.shift(k)
            y = y.add(shifted_h.multiply(x_k))
        return y


def read_signal_from_file(filename: str, INF: int) -> Signal:
    sig = Signal(INF)
    with open(filename, "r", encoding="utf-8") as f:
        nstart, nend = map(int, f.readline().strip().split())
        vals = list(map(float, f.readline().strip().split()))
    assert len(vals) == (nend - nstart + 1)
    for i, v in enumerate(vals):
        n = nstart + i
        if -INF <= n <= INF:
            sig.set_value_at_time(n, v)
    return sig


def first_difference(sig: Signal) -> Signal:
    """Delta[n] = sig[n] - sig[n-1], using only Signal.shift/add/multiply."""
    return sig.add(sig.shift(1).multiply(-1))


def impulse_from_step_response(step_response: Signal) -> Signal:
    """h[n] = s[n] - s[n-1] (s[-1] = 0) -- same formula as first_difference."""
    return first_difference(step_response)


def output_using_step_response(x: Signal, step_response: Signal) -> Signal:
    """
    y[n] = (delta_x * s)[n], computed ONLY from the step response:
    build delta_x, then reuse Offline 1's LTI_System machinery with the
    step response itself acting as the impulse response of a helper system.
    """
    dx = first_difference(x)
    sys_s = LTI_System(step_response)
    return sys_s.output(dx)


if __name__ == "__main__":
    INF = 50

    # ---- Load provided files ----
    s = read_signal_from_file("step_response.txt", INF)
    x = read_signal_from_file("input_signal.txt", INF)

    # ---- Part 1: recover impulse response ----
    h = impulse_from_step_response(s)

    s.plot("Step Response s[n]")
    h.plot("Recovered Impulse Response h[n] = s[n] - s[n-1]")

    # ---- Part 2: output using only step response ----
    dx = first_difference(x)
    y_s = output_using_step_response(x, s)

    x.plot("Input x[n]")
    dx.plot("First Difference Delta_x[n]")
    y_s.plot("Output y_s[n] computed via step response")

    # ---- Part 3: verify with impulse-response method ----
    sys_h = LTI_System(h)
    y_h = sys_h.output(x)
    y_h.plot("Output y_h[n] computed via impulse response")

    # Check if outputs match closely
    if np.allclose(y_s.values, y_h.values, atol=1e-6):
        print("Outputs match closely!")
    else:
        print("Outputs differ!")