import numpy as np
import matplotlib.pyplot as plt

#implement the necessary functions here

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

def circular_cross_correlation(x, y):
    """Peaks at lag d such that y[n] == x[(n-d) mod N]. Needs len(x)==len(y),
    a power of two -- true here since both images are 256x256."""
    X = fft(x)
    Y = fft(y)
    R = ifft(X * np.conj(Y))
    return R.real

image = plt.imread("image.png")
shifted_image = plt.imread("shifted_image.png")

# === NEW CODE: projection-based shift detection (see theory note above) ===
# Column-sum profile is invariant to the row (vertical) shift -> isolates dc.
col_profile_orig = image.sum(axis=0)
col_profile_shift = shifted_image.sum(axis=0)
dc = int(np.argmax(circular_cross_correlation(col_profile_orig, col_profile_shift)))


# Row-sum profile is invariant to the column (horizontal) shift -> isolates dr.
row_profile_orig = image.sum(axis=1)
row_profile_shift = shifted_image.sum(axis=1)
dr = int(np.argmax(circular_cross_correlation(row_profile_orig, row_profile_shift)))



print(f"Detected shift: vertical dr={dr}, horizontal dc={dc}")

# Reverse the shift (roll, not DFT -- the spec explicitly allows this)
reversed_shifted_image = np.roll(np.roll(shifted_image, dr, axis=0), dc, axis=1)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1); plt.imshow(image, cmap='gray'); plt.title("Original Image"); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(shifted_image, cmap='gray'); plt.title("Shifted Image"); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(reversed_shifted_image, cmap='gray'); plt.title("Reversed Shifted Image"); plt.axis('off')
plt.tight_layout()
plt.show()
