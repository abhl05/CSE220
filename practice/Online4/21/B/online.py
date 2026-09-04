import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === PORTED FROM transforms.py: FFTTransformer.transform / .inverse ===
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

# Convert the image to a NumPy array
image = Image.open("D:\\BUET\\CSE220\\practice\\Online4\\21\\B\\encrypted_image.tiff")
encrypted_image = np.array(image)

# === NEW CODE: locate the key row and invert the circular convolution ===
H, W = encrypted_image.shape
key_row_idx = int(np.argmin(encrypted_image[:, 0]))   # any column works; verified with several
key_row = encrypted_image[key_row_idx, :]
KEY = fft(key_row)
decrypted_image = np.zeros_like(encrypted_image) 
decrypted_image[key_row_idx, :] = key_row

for r in range(H) :
    if r == key_row_idx:
        continue
    ENC = fft(encrypted_image[r, :])
    decrypted_image[r, :] = ifft(ENC / KEY).real        # convolution theorem, inverted
    
decrypted_image = np.clip(np.round(decrypted_image), 0, 255).astype(np.uint8)

plt.figure(figsize=(8, 6))

# Encrypted image
plt.subplot(1, 2, 1)
plt.imshow(encrypted_image, cmap='gray')
plt.title("Encrypted Image")
plt.axis('off')

# Decrypted image
plt.subplot(1, 2, 2)
plt.imshow(decrypted_image, cmap='gray')
plt.title("Decrypted Image")
plt.axis('off')

plt.show()