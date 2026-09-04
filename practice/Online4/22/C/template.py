import cv2
import numpy as np
import math

def fft(x):
    """
    Compute 1D FFT
    """
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

def ifft(X):
    """
    Compute 1D inverse FFT using the FFT function
    """
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    if N == 0:
        return X
    if N & (N - 1):
        raise ValueError("FFT length must be a power of two")
    return np.conjugate(fft(np.conjugate(X))) / N

def circular_cross_correlation(x, y):
    """Peaks at lag d such that y[n] == x[(n-d) mod N]. Needs len(x)==len(y),
    a power of two -- true here since both images are 256x256."""
    X = fft(x)
    Y = fft(y)
    R = ifft(X * np.conj(Y))
    return R.real

def reconstruct_image_using_fft(original_path, shifted_path, output_path):
    
    original_img = cv2.imread(original_path)
    shifted_img = cv2.imread(shifted_path)

    if original_img is None or shifted_img is None:
        print("Error: Could not load images.")
        return

    if original_img.shape != shifted_img.shape:
        print("Error: Image dimensions do not match.")
        return
    
    # Convert the original and shifted color images to grayscale.
    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    shift_gray = cv2.cvtColor(shifted_img, cv2.COLOR_BGR2GRAY)
    
    reconstructed_img = None #implement

    print("Reconstructing image using manual FFT...")

    #implement the rest
    H, W = orig_gray.shape
    reconstructed_img = np.zeros_like(orig_gray)
    
    for r in range(H) :
        X = fft(orig_gray[r, :])
        Y = fft(shift_gray[r, :])
        C = ifft(X * np.conj(Y)).real
        shift = int(np.argmax(C))
        reconstructed_img[r, :] = np.roll(shift_gray[r, :], shift)
    reconstructed_img = np.clip(np.round(reconstructed_img), 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, reconstructed_img)
    

    cv2.imwrite(output_path, reconstructed_img)
    
if __name__ == "__main__":
    reconstruct_image_using_fft("original_image.png", "shifted_image.jpg", "reconstructed_image_fft.jpg")