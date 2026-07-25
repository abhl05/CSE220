"""
CSE220 Online (Jan 2024) - Section A1, A2 (Exponential Smoothing)

Exponential smoothing is convolution: y[n] = sum_k x[k]*h[n-k], where
  x[k]  = the k-th stock price (k = 0 .. m-1, chronological order)
  h[k]  = alpha * (1-alpha)^k   for k = 0 .. window-1   (0 elsewhere)

h[0] carries the most weight, so pairing h[n-k] with x[k] naturally puts the
largest weight on the most recent price in each window (k = n, the last
index of that window) -- exactly matching the "most recent value gets
weight alpha" rule in the spec.

The valid outputs are y[n] for n = window-1 .. m-1 (that's m-window+1
values, matching the spec: "print out total m-n+1 outputs").

Uses the DiscreteSignal / LTI_Discrete classes from Offline 1
(pasted in below so this file is self-contained).
"""

import numpy as np


# ---- Offline 1 classes (pasted in) ----

class DiscreteSignal:
    def __init__(self, INF):
        self.INF = int(INF)
        self.values = np.zeros(2 * self.INF + 1)

    def _idx(self, t):
        return t + self.INF

    def set_value_at_time(self, time, value):
        self.values[self._idx(time)] = value

    def get_value_at_time(self, time):
        if time < -self.INF or time > self.INF:
            return 0.0
        return self.values[self._idx(time)]

    def shift_signal(self, shift):
        new_sig = DiscreteSignal(self.INF)
        for n in range(-self.INF, self.INF + 1):
            src = n - shift
            if -self.INF <= src <= self.INF:
                new_sig.set_value_at_time(n, self.get_value_at_time(src))
        return new_sig

    def add(self, other):
        INF = max(self.INF, other.INF)
        result = DiscreteSignal(INF)
        for n in range(-INF, INF + 1):
            result.set_value_at_time(n, self.get_value_at_time(n) + other.get_value_at_time(n))
        return result

    def multiply(self, other):
        INF = max(self.INF, other.INF)
        result = DiscreteSignal(INF)
        for n in range(-INF, INF + 1):
            result.set_value_at_time(n, self.get_value_at_time(n) * other.get_value_at_time(n))
        return result

    def multiply_const_factor(self, scaler):
        result = DiscreteSignal(self.INF)
        result.values = self.values * scaler
        return result


class LTI_Discrete:
    def __init__(self, impulse_response):
        self.impulse_response = impulse_response

    def linear_combination_of_impulses(self, input_signal):
        INF = input_signal.INF
        impulses, coefficients = [], []
        for n in range(-INF, INF + 1):
            val = input_signal.get_value_at_time(n)
            if val != 0:
                delta = DiscreteSignal(INF)
                delta.set_value_at_time(n, 1.0)
                impulses.append(delta)
                coefficients.append(val)
        return impulses, coefficients

    def output(self, input_signal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        h = self.impulse_response
        INF = max(input_signal.INF, h.INF)
        responses = []
        output_signal = DiscreteSignal(INF)
        for delta, coeff in zip(impulses, coefficients):
            k = int(np.nonzero(delta.values)[0][0]) - delta.INF
            shifted_h = h.shift_signal(k)
            scaled_response = shifted_h.multiply_const_factor(coeff)
            responses.append(scaled_response)
            output_signal = output_signal.add(scaled_response)
        return output_signal, responses, coefficients


# ---- Sample test case (hard-coded, as requested) ----
price_list = [10, 11, 12, 9, 10, 13, 15, 16, 17, 18]
n = 3          # window size
alpha = 0.8

# ---- Solve using convolution ----
m = len(price_list)
INF = m + n + 2  # comfortable margin so no edge truncation occurs

x = DiscreteSignal(INF)
for k, price in enumerate(price_list):
    x.set_value_at_time(k, price)

h = DiscreteSignal(INF)
for k in range(n):
    h.set_value_at_time(k, alpha * (1 - alpha) ** k)

lti = LTI_Discrete(h)
y, responses, coefficients = lti.output(x)

exsm = [y.get_value_at_time(idx) for idx in range(n - 1, m)]

print("Exponential Smoothing: " + ", ".join(f"{num:.2f}" for num in exsm))
# Expected: 11.68, 9.47, 9.82, 12.29, 14.40, 15.62, 16.64, 17.63