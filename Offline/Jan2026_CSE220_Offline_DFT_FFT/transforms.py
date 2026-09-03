"""
transforms.py  --  YOUR CODE GOES HERE.

The shared transform core used by BOTH tasks. Write it once; bigmul.py
(Task A) and image_conv.py (Task B) import it.

Nothing in this file may call numpy.fft, scipy.fft, numpy.convolve,
scipy.signal, or any other library routine that performs a Fourier
transform, a convolution or a correlation for you. NumPy is for array
arithmetic only.

A quick self-test you should run before touching either application:

    import numpy as np
    from transforms import DFTAnalyzer, FFTTransformer
    x = np.random.randn(64) + 1j * np.random.randn(64)
    d, f = DFTAnalyzer(), FFTTransformer()
    assert np.max(np.abs(d.transform(x) - f.transform(x))) < 1e-9
    assert np.max(np.abs(d.inverse(d.transform(x)) - x)) < 1e-9
"""

import numpy as np


def next_power_of_two(n):
    """
    Return the smallest power of two that is >= ``n`` (and at least 1).

    Both tasks need this to choose a transform length for the radix-2 FFT.
    """
    # TODO: implement this function
    # raise NotImplementedError("Implement next_power_of_two")
    k = 1
    
    while k < n: 
        k *=2
    return k


class DFTAnalyzer:
    """
    The Discrete Fourier Transform, computed straight from its definition.

        Analysis:   X[k] = sum_{n=0}^{N-1} x[n] * exp(-2j*pi*k*n/N)
        Synthesis:  x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * exp(+2j*pi*k*n/N)

    How you write it is up to you -- a literal double loop, a precomputed
    table of twiddle factors indexed by (k*n) % N, or a NumPy expression --
    as long as it computes these sums directly and is not secretly an FFT.
    """

    name = "dft"

    def transform(self, x):
        """
        Forward DFT.

        Parameters
        ----------
        x : 1D array_like, length N (real or complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
        """
        # TODO: implement this method
        # raise NotImplementedError("Implement DFTAnalyzer.transform")
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        n = np.arange(N) # create an array of indices from 0 to N-1. arange is used to create an array of evenly spaced values within a given range. In this case, it generates an array of integers from 0 to N-1, which represent the indices of the input signal x.        
        k = n.reshape((N, 1)) # reshape k to be a column vector
        M = np.exp(-2j * np.pi * k * n / N) # compute the DFT matrix using broadcasting. The expression k * n creates a 2D array where each element is the product of the corresponding elements in k and n. The division by N normalizes the values, and the exponential function computes the complex exponentials for each element in the resulting array. dimension of M is (N, N), where each row corresponds to a frequency component and each column corresponds to a time sample.
        return np.dot(M, x)  # compute the DFT by taking the dot product of the DFT matrix M and the input signal x. The result is a 1D array of complex numbers representing the frequency components of the input signal. dimension of the output is (N,), where each element corresponds to a frequency component. the product is calculated by summing the products of the corresponding elements in each row of M and the input signal x.

    def inverse(self, spectrum):
        """
        Inverse DFT, including the 1/N factor.

        Parameters
        ----------
        spectrum : 1D array_like, length N (complex)

        Returns
        -------
        numpy.ndarray of complex128, shape (N,)
            Do NOT discard the imaginary part here -- the caller decides when
            it is safe to take .real.
        """
        # TODO: implement this method
        # raise NotImplementedError("Implement DFTAnalyzer.inverse")
        spectrum = np.asarray(spectrum, dtype=np.complex128)
        N = spectrum.shape[0] # shape[0] gives the number of elements in the first dimension of the array, which corresponds to the length of the input spectrum.
        n = np.arange(N) # create an array of indices from 0 to N-1. arange is used to create an array of evenly spaced values within a given range. In this case, it generates an array of integers from 0 to N-1, which represent the indices of the output signal.
        k = n.reshape((N, 1)) # reshape k to be a column vector
        M = np.exp(2j * np.pi * k * n / N) # compute the inverse DFT matrix using broadcasting. The expression k * n creates a 2D array where each element is the product of the corresponding elements in k and n. The division by N normalizes the values, and the exponential function computes the complex exponentials for each element in the resulting array. dimension of M is (N, N), where each row corresponds to a time sample and each column corresponds to a frequency component.
        return np.dot(M, spectrum) / N  # compute the inverse DFT by taking the dot product of the inverse DFT matrix M and the input spectrum, and then dividing by N to include the 1/N factor. The result is a 1D array of complex numbers representing the time-domain signal. dimension of the output is (N,), where each element corresponds to a time sample. the product is calculated by summing the products of the corresponding elements in each row of M and the input spectrum, and then dividing by N to normalize the result.

class FFTTransformer(DFTAnalyzer):
    """
    Radix-2 decimation-in-time (Cooley-Tukey) FFT, in O(N log N).

    It inherits from DFTAnalyzer so that both applications can treat the two
    interchangeably: they call ``engine.transform(...)`` and
    ``engine.inverse(...)`` without caring which engine they hold.

    Requirements:
      * Recursive or iterative (with bit-reversal permutation) -- your choice.
      * N must be a power of two; raise ValueError for any other length.
        The caller is responsible for zero-padding up to next_power_of_two.
      * The inverse must reuse the same butterfly machinery (conjugated
        twiddles, or conjugate-transform-conjugate), not a second copy of it.
      * Twiddle factors for a stage are computed once per stage, never once
        per butterfly.
    """

    name = "fft"

    def transform(self, x):
        """Forward FFT. Same contract as DFTAnalyzer.transform."""
        # TODO: implement this method
        # raise NotImplementedError("Implement FFTTransformer.transform")
        x = np.asarray(x, dtype=np.complex128)
        N = x.shape[0]
        if N & (N - 1) != 0:  
            # check if N is a power of two using bitwise operation. If N is not a power of two, raise a ValueError.
                raise ValueError("Input length must be a power of two.")
        if N <= 1:
            return x  # base case: if the input length is 1 or less, return the input as is.
        
        a = x.copy()  # perform bit-reversal permutation on the input array x. This rearranges the elements of x in a specific order that is required for the Cooley-Tukey FFT algorithm. The _bit_reverse_permute method takes the input array x and returns a new array with the elements rearranged according to their bit-reversed indices.
        
        # bit reversal permutation
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

    def inverse(self, spectrum):
        """Inverse FFT, including the 1/N factor."""
        # TODO: implement this method
        # raise NotImplementedError("Implement FFTTransformer.inverse")
        spectrum = np.asarray(spectrum, dtype=np.complex128)

        N = len(spectrum)

        if N == 0:
            return spectrum

        if N & (N - 1):
            raise ValueError("FFT length must be a power of two")

        return np.conjugate(
            self.transform(np.conjugate(spectrum))
        ) / N


# ---------------------------------------------------------------------------
# BONUS (optional) -- arbitrary-length FFT.
#
# Delete this class if you are not attempting the bonus. If you do attempt it,
# run both tasks with --engine arbitrary and leave those output directories in
# your submission as the evidence.
# ---------------------------------------------------------------------------
class ArbitraryLengthFFT(FFTTransformer):
    """
    Bonus: an O(N log N) transform for ANY length N, not just powers of two.

    Bluestein's chirp-z algorithm is the usual route: rewrite the DFT as a
    convolution of two chirp sequences, and evaluate that convolution with a
    radix-2 FFT of length >= 2N-1. A mixed-radix Cooley-Tukey that factorises
    N is equally acceptable.

    With this engine, Task A no longer has to pad the digit arrays up to a
    power of two, and Task B no longer has to pad the image up to one.
    """

    name = "arbitrary"

    def transform(self, x):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.transform")
    def inverse(self, spectrum):
        # TODO (bonus): implement this method
        raise NotImplementedError("Bonus: implement ArbitraryLengthFFT.inverse")
