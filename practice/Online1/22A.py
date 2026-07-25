"""
July 2025 CSE 220

Online on Signals and Their Properties

Subsections: A1, A2
Total Marks: 10
Total Time: 30 Minutes

January 7, 2026

In this assignment, you need to implement time sub-scaling of a sampled continuous-time
signal in Python. You also need to plot the graphs of each of these signals.
Let the base signal be x(t) (provided in the template file). You will compute and plot:

x(t), y(t) = x( Where is a positive integer

t
k
); k

Tasks
● Generate the time axis t and compute x(t). (given)
● Implement a function time_scale(...) that produces y(t) = x(
t
k
)

● As the program is using a sampled continuous-time signal, time sub-scaling requires
new samples between original values. For example y(0) = x(0), y(2) = x(1) but
y(1) = x(0. 5) which is not available. So, you have to interpolate the value. You will
implement a function interpolate_signal(...) to interpolate the missing sample values.
Missing values will be calculated from the average of the nearest left and right values.
y(1) = 0. 5 * (x(0) + x(1)).
● Plot (with proper labels and legend) on the same figure: x(t), y(t). Ignore the values that
go beyond the specified time range.
Marks Distribution
● Implementing interpolation: 4
● Implementing time sub-scaling: 4
● Plotting: 2
"""


import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray: 
    """                                  
    Base signal x(t): sinusoidal signal
    takes a time array t and returns the corresponding signal values.
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate missing values using the weighted average 
    of the nearest left and right samples (Linear Interpolation).
    1. Calculate the uniform time step size (dt)
    2. Find the exact fractional index for each requested time
    3. Determine the integer indices for the nearest left and right samples
    4. Clip indices to prevent out-of-bounds errors at the very edges
    """
    # 1. Calculate the uniform time step size (dt)
    dt = (t_original[-1] - t_original[0]) / (len(t_original) - 1)
    
    # 2. Find the exact fractional index for each requested time
    fractional_idx = (t_query - t_original[0]) / dt # this 
    
    # 3. Determine the integer indices for the nearest left and right samples
    idx_left = np.floor(fractional_idx).astype(int)
    idx_right = np.ceil(fractional_idx).astype(int)
    
    # 4. Clip indices to prevent out-of-bounds errors at the very edges
    idx_left = np.clip(idx_left, 0, len(t_original) - 1)
    idx_right = np.clip(idx_right, 0, len(t_original) - 1)
    
    # 5. Calculate weights for the average based on proximity
    # For k=2, the point is exactly halfway, making weight_right and weight_left both 0.5
    weight_right = fractional_idx - idx_left
    weight_left = 1.0 - weight_right
    
    # 6. Compute the interpolated values
    x_interpolated = (x_original[idx_left] * weight_left) + (x_original[idx_right] * weight_right)
    
    return x_interpolated


# np.ndarray is used for type hinting to indicate that the function expects and returns numpy arrays.

def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    # Calculate the new time coordinates we need to query
    t_query = t / k
    
    # Interpolate the signal at these new coordinates
    y = interpolate_signal(t, x, t_query)
    
    # Ignore values that go beyond the original time range by setting them to 0
    out_of_bounds = (t_query < t[0]) | (t_query > t[-1])
    y[out_of_bounds] = 0.0
    
    return y


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs side-by-side or on the same figure with proper labels.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot original signal
    plt.plot(t, x, label='Original Signal: x(t)', color='blue', alpha=0.7)
    
    # Plot sub-scaled signal
    plt.plot(t, y, label='Time Sub-scaled: y(t)', color='red', linestyle='--', linewidth=2)
    
    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim(t[0], t[-1])
    plt.axhline(0, color='black', linewidth=0.5, linestyle='-') # Adds a center zero-line
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()