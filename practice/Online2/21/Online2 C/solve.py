"""
CSE220 Online (Jan 2024) - Section C (Polynomial Multiplication via Convolution)

Multiplying two polynomials is exactly discrete convolution of their
coefficient sequences, indexed by increasing power of x:
  poly1(x) = sum_k a_k x^k   <->   signal a[k]  (k = 0 .. d1)
  poly2(x) = sum_k b_k x^k   <->   signal b[k]  (k = 0 .. d2)
  product coefficient of x^n = sum_k a[k]*b[n-k] = (a * b)[n]

The input is given as coefficients from the HIGHEST exponent down to x^0,
so we reverse each input list to build the "increasing power" signal, run
it through convolution (via LTI_Discrete, treating poly2 as the impulse
response), then reverse the output back to highest-exponent-first for
printing.

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


def multiply_polynomials(d1, poly1, d2, poly2):
    """
    poly1, poly2: coefficients from highest exponent to x^0.
    Returns (result_degree, coefficients_highest_to_lowest).
    """
    INF = d1 + d2 + 2

    x1 = DiscreteSignal(INF)
    for k in range(d1 + 1):
        x1.set_value_at_time(k, poly1[d1 - k])   # reverse -> coeff of x^k

    x2 = DiscreteSignal(INF)
    for k in range(d2 + 1):
        x2.set_value_at_time(k, poly2[d2 - k])

    lti = LTI_Discrete(x2)          # poly2 acts as the "impulse response"
    y, responses, coefficients = lti.output(x1)

    result_degree = d1 + d2
    coeffs_desc = [int(round(y.get_value_at_time(k))) for k in range(result_degree, -1, -1)]
    return result_degree, coeffs_desc


# ---- Sample test case (hard-coded, as requested) ----
d1 = 2
poly1 = [3, -2, 1]          # 3x^2 - 2x + 1
d2 = 3
poly2 = [2, 0, -3, 1]       # 2x^3 - 3x + 1

degree, coefficients = multiply_polynomials(d1, poly1, d2, poly2)

print(f"Degree of the Polynomial: {degree}")
print(f"Coefficients: {' '.join(map(str, coefficients))}")
# Expected: Degree 5, Coefficients: 6 -4 -7 9 -5 1