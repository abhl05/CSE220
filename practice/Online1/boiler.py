import numpy as np
import matplotlib.pyplot as plt

def unit_impulse(n: np.ndarray) -> np.ndarray:
    """ Dirac Delta / Unit Impulse: delta[n] = 1 if n == 0, else 0 """
    return np.where(n == 0, 1.0, 0.0)

def unit_step(n: np.ndarray) -> np.ndarray:
    """ Unit Step: u[n] = 1 if n >= 0, else 0 """
    return np.where(n >= 0, 1.0, 0.0)

def unit_ramp(n: np.ndarray) -> np.ndarray:
    """ Unit Ramp: r[n] = n if n >= 0, else 0 """
    return np.maximum(0.0, n)
    
def exponential_decay(n: np.ndarray, alpha: float) -> np.ndarray:
    """ Causal Exponential: x[n] = (alpha^n) * u[n] """
    u_n = unit_step(n)
    return (alpha ** n) * u_n


def time_reverse(n: np.ndarray, x: np.ndarray):
    """ y[n] = x[-n] """
    # Negate and flip time, just flip amplitude
    return -n[::-1], x[::-1].copy()

def time_shift(n: np.ndarray, x: np.ndarray, k: int):
    """ y[n] = x[n - k] (Delay by k) """
    # Do NOT change the amplitude array. Just shift the time axis!
    # If k > 0, the signal shifts right (delayed). 
    return n + k, x.copy()

def even_odd_decompose(n: np.ndarray, x: np.ndarray):
    """ 
    Even: 0.5 * (x[n] + x[-n])
    Odd: 0.5 * (x[n] - x[-n]) 
    *Assumes n is symmetric around 0*
    """
    n_rev, x_rev = time_reverse(n, x)
    x_even = 0.5 * (x + x_rev)
    x_odd = 0.5 * (x - x_rev)
    return x_even, x_odd


def signal_energy(x: np.ndarray) -> float:
    """ E = sum(|x[n]|^2) """
    return np.sum(np.abs(x)**2)

def signal_power(x: np.ndarray) -> float:
    """ P = limit as N->inf (1/2N) * sum(|x[n]|^2) """
    # For a finite discrete signal, it's just the mean of the squared magnitudes
    return np.mean(np.abs(x)**2)

def discrete_convolution(n_x: np.ndarray, x: np.ndarray, n_h: np.ndarray, h: np.ndarray):
    """ 
    y[n] = x[n] * h[n] 
    Students often forget to calculate the NEW time axis!
    """
    # 1. Convolve the amplitudes using numpy
    y = np.convolve(x, h, mode='full')
    
    # 2. Calculate the new time axis boundaries
    # The new start time is the sum of the original start times
    start_n = n_x[0] + n_h[0]
    
    # The new end time is the sum of the original end times
    end_n = n_x[-1] + n_h[-1]
    
    # 3. Generate the new time array
    n_y = np.arange(start_n, end_n + 1)
    
    return n_y, y


def plot_signals(n, x, continuous=False, title="Signal", color='blue'):
    plt.figure(figsize=(8, 4))
    
    # Use stem() for discrete, plot() for continuous
    if continuous:
        plt.plot(n, x, color=color, linewidth=2, label="x(t)")
    else:
        # baseline.set_visible(False) removes the ugly default red line at y=0
        markerline, stemlines, baseline = plt.stem(n, x, linefmt=f'{color}-', markerfmt=f'{color}o', label="x[n]")
        baseline.set_visible(False) 
    
    # The "Always Include These" formatting
    plt.axhline(0, color='black', linewidth=1) # True Zero line
    plt.axvline(0, color='black', linewidth=1) # True Y-axis
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Time index (n)" if not continuous else "Time (t)", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6) # here alpha makes the grid lines lighter
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()