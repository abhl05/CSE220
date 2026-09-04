import math
import cmath
import numpy as np


def next_power_of_two(n):
    k = 1
    while k < n:
        k *= 2
    return k

def fft(a):
    a = np.asarray(a, dtype=np.complex128)
    N = a.shape[0]
    if N & (N - 1) != 0:
        raise ValueError("Input length must be a power of two.")
    if N <= 1:
        return a
    arr = a.copy()
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    length = 2
    while length <= N:
        half_length = length // 2
        twiddle_factor = np.exp(-2j * np.pi * np.arange(half_length) / length)
        for start in range(0, N, length):
            for i in range(half_length):
                g = arr[start + i]
                h = arr[start + i + half_length] * twiddle_factor[i]
                arr[start + i] = g + h
                arr[start + i + half_length] = g - h
        length *= 2
    return arr

def ifft(a):
    a = np.asarray(a, dtype=np.complex128)
    N = len(a)
    if N == 0:
        return a
    if N & (N - 1):
        raise ValueError("FFT length must be a power of two")
    return np.conjugate(fft(np.conjugate(a))) / N


def weighted_polynomial_multiply(P, Q, W):
    p_asc = P[::-1]           # descending -> ascending (p_asc[i] = p_i)
    q_asc = Q[::-1]
    w_asc = W[::-1]
    wp = [w_asc[i] * p_asc[i] for i in range(len(p_asc))]

    m1, n1 = len(wp), len(q_asc)
    L = m1 + n1 - 1
    N = next_power_of_two(L)
    A = np.zeros(N, dtype=complex); A[:m1] = wp
    B = np.zeros(N, dtype=complex); B[:n1] = q_asc
    C = ifft(fft(A) * fft(B)).real
    R_asc = [round(v) for v in C[:L]]
    return R_asc[::-1]        # back to descending, matching P/Q/W's own convention
    

if __name__ == "__main__":
    P = [1, 3, 2, 6, 7]
    Q = [4,1]
    W = [3, 2, 1, 5, 6]
 

    R = weighted_polynomial_multiply(P, Q, W)

    print("Result:", R)