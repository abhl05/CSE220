"""
CSE220 Online on Convolution - C1, C2 (Superposition of Multiple Signals)

A SuperSignal bundles several (coefficient, Signal) components representing
x(n) = sum_i c_i * x_i(n). LTI_System.output_super builds that composite
signal (using only Signal.multiply/add) and then reuses the ordinary
convolution machinery (output) on it -- this works because an LTI system
obeys linearity, so convolving the composite is equivalent to combining
each x_i's output with the same coefficients.
"""

import numpy as np
import matplotlib.pyplot as plt


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


class SuperSignal:
    def __init__(self):
        self.components = []

    def add(self, signal: Signal, coefficient=1.0):
        self.components.append((coefficient, signal))


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

    def output_super(self, super_signal: SuperSignal):
        """Build x(n) = sum_i c_i * x_i(n), then convolve it with h(n)."""
        combined_input = None
        for coeff, sig in super_signal.components:
            scaled = sig.multiply(coeff)
            combined_input = scaled if combined_input is None else combined_input.add(scaled)
        return self.output(combined_input)


if __name__ == "__main__":
    INF = 10

    # Component signals
    x1 = Signal(INF)
    x1.set_value_at_time(0, 1)

    x2 = Signal(INF)
    x2.set_value_at_time(2, 1)

    # SuperSignal: x(n) = 2*x1(n) - x2(n)
    x_super = SuperSignal()
    x_super.add(x1, 2.0)
    x_super.add(x2, -1.0)

    # Impulse response
    h = Signal(INF)
    h.set_value_at_time(0, 1)
    h.set_value_at_time(1, 0.5)

    system = LTI_System(h)

    # Output using superposition
    y = system.output_super(x_super)
    y.plot("Output y(n) via superposition of 2*x1(n) - x2(n)")

    print("y(n) samples:", [y.get_value_at_time(n) for n in range(-3, 6)])