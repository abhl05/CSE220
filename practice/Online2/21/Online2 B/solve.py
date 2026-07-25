"""
CSE220 Online (Jan 2024) - Section B1, B2 (Moving Averages)

Both moving averages are convolution y[n] = sum_k x[k]*h[n-k], with
  x[k] = the k-th stock price (k = 0 .. m-1, chronological order)

Unweighted Moving Average (UMA):
  h[j] = 1/window   for j = 0 .. window-1   (equal weight on every day)

Weighted Moving Average (WMA):
  spec says: weight of the LAST (most recent) day = window,
             weight of the day before that = window-1, ..., oldest day = 1,
             then normalize so the weights sum to 1.
  In convolution, h[n-k] pairs with x[k]; h[0] pairs with the most recent
  sample in the window (k=n), so:
      h[j] = (window - j) / (window*(window+1)/2)   for j = 0 .. window-1

Valid outputs are y[n] for n = window-1 .. m-1 (m-window+1 values).

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


def moving_averages(price_list, n):
    """Return (uma, wma) lists using convolution via LTI_Discrete."""
    m = len(price_list)
    INF = m + n + 2

    x = DiscreteSignal(INF)
    for k, price in enumerate(price_list):
        x.set_value_at_time(k, price)

    h_uma = DiscreteSignal(INF)
    for j in range(n):
        h_uma.set_value_at_time(j, 1.0 / n)

    norm = n * (n + 1) / 2
    h_wma = DiscreteSignal(INF)
    for j in range(n):
        h_wma.set_value_at_time(j, (n - j) / norm)

    y_uma, _, _ = LTI_Discrete(h_uma).output(x)
    y_wma, _, _ = LTI_Discrete(h_wma).output(x)

    uma = [y_uma.get_value_at_time(idx) for idx in range(n - 1, m)]
    wma = [y_wma.get_value_at_time(idx) for idx in range(n - 1, m)]
    return uma, wma


# ---- Sample test case (hard-coded, as requested) ----
price_list = [1, 2, 3, 4, 5, 6, 7, 8]
n = 4

uma, wma = moving_averages(price_list, n)

print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma))
# Expected UMA: 2.50, 3.50, 4.50, 5.50, 6.50
# Expected WMA: 3.00, 4.00, 5.00, 6.00, 7.00

print()
print("-- Case 2 --")
price_list2 = [5, -2, 3, 1, 0, -6, 4, -2, 1]
n2 = 3
uma2, wma2 = moving_averages(price_list2, n2)
print("Unweighted Moving Averages: " + ", ".join(f"{num:.2f}" for num in uma2))
print("Weighted Moving Averages:   " + ", ".join(f"{num:.2f}" for num in wma2))
# Expected UMA: 2.00, 0.67, 1.33, -1.67, -0.67, -1.33, 1.00
# Expected WMA: 1.67, 1.17, 0.83, -2.83, 0.00, -0.67, 0.50