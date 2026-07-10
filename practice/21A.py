import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()  # Uncommented so you can see the graphs when you run the script

def init_signal():
    return np.zeros(2 * INF + 1)


# ==========================================
# Task 1: Time Shift
# ==========================================
def time_shift_signal(x: np.ndarray, k: int) -> np.ndarray:
    """
    Shifts the signal by k units: y[n] = x[n-k].
    Uses NumPy slicing to avoid explicit loops (Bonus) and avoids 
    circular wrap-around caused by np.roll().
    """
    y = np.zeros_like(x)
    if k > 0:
        # Shift right: skip the first k elements
        y[k:] = x[:-k]
    elif k < 0:
        # Shift left: skip the last k elements
        y[:k] = x[-k:]
    else:
        # No shift
        y[:] = x
    return y


# ==========================================
# Task 2: Time Scale
# ==========================================
def time_scale_signal(x: np.ndarray, k: int, *args) -> np.ndarray:
    """
    Scales the signal by k units: y[n] = x[k*n].
    Uses NumPy array indexing to completely avoid loops (Bonus).
    
    *Note: *args is added to the parameters because the provided main() 
    function buggily passes a 3rd argument 'True' when calling this function.
    """
    y = np.zeros_like(x)
    
    # 1. Create an array of actual time indices 'n' from -8 to 8
    n = np.arange(-INF, INF + 1)
    
    # 2. Calculate k * n for every index simultaneously
    kn = k * n
    
    # 3. Create a boolean mask of valid indices (where k*n is between -8 and 8)
    valid_mask = (kn >= -INF) & (kn <= INF)
    
    # 4. Map the valid time indices back to the 0-indexed NumPy arrays (by adding INF)
    # and assign the values all at once
    y[valid_mask] = x[kn[valid_mask] + INF]
    
    return y


def main():
    img_root_path = '.'
    signal = init_signal()
    signal[INF] = 1
    signal[INF+1] = .5
    signal[INF-1] = 2
    signal[INF + 2] = 1
    signal[INF - 2] = .5

    plot(signal, title='Original Signal(x[n])', saveTo=f'{img_root_path}/x[n].png')

    plot(time_shift_signal(signal, 2), title='x[n-2]', saveTo=f'{img_root_path}/x[n-2].png')
    
    plot(time_shift_signal(signal, -2), title='x[n+2]', saveTo=f'{img_root_path}/x[n+2].png')
    
    plot(time_shift_signal(signal, 0), title='x[n+0]', saveTo=f'{img_root_path}/x[n+0].png')
    
    plot(time_scale_signal(signal, 2, True), title='x[2n]', saveTo=f'{img_root_path}/x[2n].png')
    
    plot(time_scale_signal(signal, 1, True), title='x[1n]', saveTo=f'{img_root_path}/x[1n].png')
    

if __name__ == '__main__':
    main()
    
    
    
    """ 
    Summary Example
    Let's look at one specific point in time: $n = 3$.
    We are calculating y at $n = 3$.The formula is $y[3] = x[2 \cdot 3] = x[6]$.
    Is math time 6 valid? Yes (valid_mask is True here).
    Where is math time 6 stored in Python? $6 + 8 = 14$.
    NumPy grabs the value at x[14] and assigns it into the y array exactly where the mask tells it to.
    """