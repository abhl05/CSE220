"""
Instructions:
- x1, x2, a, b, k, and h are already given below.
- Complete the TODOs.
- Do NOT use numpy.convolve / scipy.signal / any built-in convolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_lti import DiscreteSignal, LTISystem


def make_signal(start_time, end_time, values):
    """Helper: build a DiscreteSignal from a list of values."""
    signal = DiscreteSignal(start_time, end_time)
    for offset, value in enumerate(values):
        signal.set_value_at_time(start_time + offset, value)
    return signal


def max_absolute_difference(first_signal, second_signal):
    """Helper: largest |difference| between two signals over their combined range."""
    # TODO: reuse your offline implementation of this function.
    start = min(first_signal.start_time, second_signal.start_time)
    end = max(first_signal.end_time, second_signal.end_time)

    max_diff = 0.0
    for n in range(start, end + 1):
        diff = abs(first_signal.get_value_at_time(n) - second_signal.get_value_at_time(n))
        if diff > max_diff:
            max_diff = diff
    return max_diff


# ---- Generic property testers ----
# These must work for ANY apply_system callable
# method such as system_a.output. Do not assume apply_system is an LTISystem.

# You shoulb be able to use this function like: test_linearity(sys_a.output, x1, x2, a, b) 
# or test_linearity(system_b, x1, x2, a, b)

def test_linearity(apply_system, x1, x2, a, b):

    #TODO: Return max| apply_system(a*x1 + b*x2)  -  (a*apply_system(x1) + b*apply_system(x2)) |
    return max_absolute_difference(
        apply_system(x1.multiply(a).add(x2.multiply(b))),
        apply_system(x1).multiply(a).add(apply_system(x2).multiply(b))
    )


def test_time_invariance(apply_system, x, k):

    #TODO: Return max| apply_system(x shifted by k)  -  (apply_system(x) shifted by k) |
    return max_absolute_difference(
            apply_system(x.shift(k)), 
            apply_system(x).shift(k)
        )


# ---- System B: y[n] = n * x[n] ----

def system_b(input_signal):
    # TODO: build and return a DiscreteSignal where output[n] = n * input_signal[n]
    return make_signal(input_signal.start_time, input_signal.end_time, [n * input_signal.get_value_at_time(n) for n in input_signal.times()])


def main():
    tolerance = 1e-9

    # ---- Given signals and scalars (do not change) ----
    x1 = make_signal(-2, 2, [1, 0, 2, -1, 3])
    x2 = make_signal(-1, 3, [2, -3, 0, 1, 1])
    a, b = 2.0, -3.0
    k = 3

    h = make_signal(0, 2, [1.0, 0.5, 0.25])
    #TODO: Test both properties for system A
    
    print("=== System A: genuine LTI system (LTISystem.output) ===")
    T = LTISystem(h)
    diff_linear_a = test_linearity(T.output, x1, x2, a, b)
    diff_ti_a = test_time_invariance(T.output, x1, k)
    
    print(f"Linearity max diff:        {diff_linear_a}")
    print(f"Time-invariance max diff:  {diff_ti_a}")

    print()

    #TODO: Test both properties for system B
    print("=== System B: y[n] = n * x[n] ===")
    diff_linear_b = test_linearity(system_b, x1, x2, a, b)
    diff_ti_b = test_time_invariance(system_b, x1, k)
    print(f"Linearity max diff:        {diff_linear_b}")
    print(f"Time-invariance max diff:  {diff_ti_b}")

    print()
    # TODO: print a short conclusion stating which property System B fails
    # (linearity or time-invariance).
    if diff_linear_b > tolerance:
        print("System B fails linearity.")
    elif diff_ti_b > tolerance:
        print("System B fails time-invariance.")
    else:
        print("System B passes both properties.")


if __name__ == "__main__":
    main()
