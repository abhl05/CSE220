"""
CSE220 Online on Convolution - B1, B2 (Combined LTI Systems)

Topology (as given in the figure):
    x --> h1 --\
                +--> h3 --> y      (h1 and h2 in parallel, summed, then
    x --> h2 --/                    cascaded into h3)

    x --> h_combined --> y          (a single equivalent LTI system)

h_combined = h3 * (h1 + h2)   [convolved, since parallel branches add their
impulse responses, and cascading with h3 convolves that sum with h3]
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


if __name__ == "__main__":
    INF = 10

    x = Signal(INF)
    x.set_value_at_time(0, 1)
    x.set_value_at_time(2, -1)
    x.plot("Input x(n)")

    h1 = Signal(INF)
    h1.set_value_at_time(0, 1)

    h2 = Signal(INF)
    h2.set_value_at_time(1, 0.5)

    h3 = Signal(INF)
    h3.set_value_at_time(0, 1)
    h3.set_value_at_time(1, 1)

    sys1 = LTI_System(h1)
    sys2 = LTI_System(h2)
    sys3 = LTI_System(h3)

    # ---- Output block by block: h1 and h2 in parallel (summed), then h3 ----
    parallel_output = sys1.output(x).add(sys2.output(x))
    y_final_1 = sys3.output(parallel_output)
    y_final_1.plot("Output via block-by-block system")

    # ---- h_combined: convolve h3 with (h1 + h2) ----
    h1_plus_h2 = h1.add(h2)
    h_combined = LTI_System(h1_plus_h2).output(h3)   # (h1+h2) * h3

    sys_combined = LTI_System(h_combined)
    y_final_2 = sys_combined.output(x)
    y_final_2.plot("Output via combined impulse response")

    print("Outputs are equal:",
          np.allclose(y_final_1.values, y_final_2.values))