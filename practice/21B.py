import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

INF = 8

def plot(
        signal, 
        title=None, 
        y_range=(-1, 3), 
        figsize = (8, 3),
        x_label='n (Time Index)',
        y_label='x[n]',
        saveTo=None
    ):
    plt.figure(figsize=figsize)
    plt.xticks(np.arange(-INF, INF + 1, 1))
    
    y_range = (y_range[0], max(np.max(signal), y_range[1]) + 1)
    # set y range
    plt.ylim(*y_range)
    plt.stem(np.arange(-INF, INF + 1, 1), signal)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    if saveTo is not None:
        plt.savefig(saveTo)
    plt.show() # Uncommented so you can see the graphs when you run the script

def init_signal():
    return np.zeros(2 * INF + 1)


# ==========================================
# Task 1: Time Scale (Expansion)
# ==========================================
def time_scale_signal(x: np.ndarray, k: int) -> np.ndarray:
    """
    Time scaling (expansion): y[n] = x[n/k].
    Intermediate samples that are not perfectly divisible by k are set to 0.
    Implemented without loops (Bonus).
    """
    y = np.zeros_like(x)
    
    # 1. Create an array of actual time indices 'n' from -8 to 8
    n = np.arange(-INF, INF + 1)
    
    # 2. Find indices where n is perfectly divisible by k
    valid_mask = (n % k == 0)
    
    # 3. Calculate the corresponding source indices in the original signal
    src_n = (n[valid_mask] // k)
    
    # 4. Convert mathematical time (src_n) to Python array indices by adding INF
    src_idx = src_n + INF
    
    # 5. Assign the values simultaneously
    y[valid_mask] = x[src_idx]
    
    return y

# ==========================================
# Task 2: Time Scale with Interpolation
# ==========================================
def time_scale_signal_interpolate(x: np.ndarray, k: int) -> np.ndarray:
    """
    Time scaling with interpolation.
    Missing samples are set to the unweighted average of the 
    nearest left and right original samples.
    Implemented without loops (Bonus).
    """
    # 1. Create target mathematical time index array
    n = np.arange(-INF, INF + 1)
    
    # 2. Calculate exact floating-point source index mapping (e.g., n/k)
    p = n / k 
    
    # 3. Find the nearest left (floor) and right (ceil) mathematical integer times
    left_n = np.floor(p).astype(int)
    right_n = np.ceil(p).astype(int)
    
    # 4. Convert mathematical times to valid Python array indices
    left_idx = left_n + INF
    right_idx = right_n + INF
    
    # 5. Compute the average. 
    # (Note: If perfectly divisible, left_idx == right_idx, so it averages with itself).
    y = (x[left_idx] + x[right_idx]) / 2.0
    
    return y


def main():
    img_root = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root}/x[n].png')
    plot(time_scale_signal(signal, 3), title='x[n/3]', saveTo=f'{img_root}/x[n divided by 3].png')
    plot(time_scale_signal(signal, 1), title='x[n/1]', saveTo=f'{img_root}/x[n divided by 1].png')
    plot(time_scale_signal_interpolate(signal, 3), title='x[n/3] with interpolation', saveTo=f'{img_root}/x[n divided by 3]_with_interpolation.png')
    plot(time_scale_signal_interpolate(signal, 1), title='x[n/1] with interpolation', saveTo=f'{img_root}/x[n divided by 1]_with_interpolation.png')

if __name__ == '__main__':
    main()