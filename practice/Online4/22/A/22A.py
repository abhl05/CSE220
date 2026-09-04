# ===== 2022 Section A =====
import numpy as np

# === PORTED FROM transforms.py: next_power_of_two, FFTTransformer.transform/.inverse ===
def next_power_of_two(n):
    k = 1
    while k < n:
        k *= 2
    return k

def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N & (N - 1) != 0:
        raise ValueError("Input length must be a power of two.")
    if N <= 1:
        return x
    a = x.copy()
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= N:
        half_length = length // 2
        twiddle_factor = np.exp(-2j * np.pi * np.arange(half_length) / length)
        for start in range(0, N, length):
            for i in range(half_length):
                g = a[start + i]
                h = a[start + i + half_length] * twiddle_factor[i]
                a[start + i] = g + h
                a[start + i + half_length] = g - h
        length *= 2
    return a

def ifft(spectrum):
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    N = len(spectrum)
    if N == 0:
        return spectrum
    if N & (N - 1):
        raise ValueError("FFT length must be a power of two")
    return np.conjugate(fft(np.conjugate(spectrum))) / N
# === END PORTED BLOCK ===

# === NEW CODE: LSD-first multiply -- identical convention to your offline bigmul.py limbs ===
def multiply_lsd_first(a_digits, b_digits):
    m, n = len(a_digits), len(b_digits)
    L = m + n - 1
    N = next_power_of_two(L)
    A = np.zeros(N, dtype=complex); A[:m] = a_digits
    B = np.zeros(N, dtype=complex); B[:n] = b_digits
    C = ifft(fft(A) * fft(B)).real
    C = np.round(C[:L]).astype(np.int64)

    result, carry = [], 0
    for k in range(L):
        total = int(C[k]) + carry
        result.append(total % 10)
        carry = total // 10
    while carry:
        result.append(carry % 10)
        carry //= 10
    return result

if __name__ == "__main__":
    print(multiply_lsd_first([3, 2, 1], [5, 4]))     # -> [5, 3, 5, 5]   (123*45=5535)
    print(multiply_lsd_first([9, 9, 9], [9, 9]))     # -> [1, 0, 9, 8, 9] (999*99=98901)