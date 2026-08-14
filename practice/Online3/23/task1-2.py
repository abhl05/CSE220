import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fs_redrawer import FourierEpicycles
from svg_utils import load_svg_path

def compute_phase_mse(actual, theoretical, magnitudes, threshold=1e-4):
    """
    Computes phase MSE only for coefficients with significant magnitude.
    Phase near zero-magnitude is pure computational noise and will artificially inflate MSE.
    """
    mask = magnitudes > threshold
    phase_diff = np.angle(actual[mask]) - np.angle(theoretical[mask])
    # Wrap phase difference to the range [-pi, pi]
    phase_diff_wrapped = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    
    if len(phase_diff_wrapped) == 0:
        return 0.0
    return np.mean(phase_diff_wrapped**2)

if __name__ == "__main__":
    N_HARMONICS = 150
    # Adjust this path based on your directory structure
    svg_path = Path("D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/heart.svg")
    
    # 1. Build f from an SVG as usual
    t, f_t = load_svg_path(svg_path, num_points=1000)
    T = t[-1] - t[0]
    omega = 2 * np.pi / T
    
    # 2. Construct the shifted signal g(t) = f(t - t0)
    t0 = T / 4  # Shift by a quarter of a period
    
    # Calculate the wrapped sampling times for interpolation
    # Modulo T ensures periodic wrapping, adding t[0] keeps it in the [t[0], t[-1]] grid
    t_sample = (t - t[0] - t0) % T + t[0]
    
    # Interpolate real and imaginary parts separately (OOP style, no manual slicing/rolling)
    g_t_real = np.interp(t_sample, t, np.real(f_t))
    g_t_imag = np.interp(t_sample, t, np.imag(f_t))
    g_t = g_t_real + 1j * g_t_imag
    
    # 3. Build FourierEpicycles instances for both signals
    fs_f = FourierEpicycles(t, f_t, n_harmonics=N_HARMONICS)
    fs_f.calculate_all_coefficients()
    
    fs_g = FourierEpicycles(t, g_t, n_harmonics=N_HARMONICS)
    fs_g.calculate_all_coefficients()
    
    # 4. Compare d_n against the theoretical c_n * e^{-j n omega t0}
    n_values = np.arange(-N_HARMONICS, N_HARMONICS + 1)
    
    c_n = np.array([fs_f.coeffs[n] for n in n_values])
    d_n_actual = np.array([fs_g.coeffs[n] for n in n_values])
    
    # Theoretical prediction: d_n = c_n * e^{-j n omega t0}
    shift_multiplier = np.exp(-1j * n_values * omega * t0)
    d_n_theoretical = c_n * shift_multiplier
    
    # Calculate Mean Squared Errors
    mag_mse = np.mean((np.abs(d_n_actual) - np.abs(d_n_theoretical))**2)
    phase_mse = compute_phase_mse(d_n_actual, d_n_theoretical, np.abs(c_n))
    
    print("--- Time-Shift Property Verification ---")
    print(f"Shift t0: {t0:.4f}s (T = {T:.4f}s)")
    print(f"Magnitude MSE: {mag_mse:.6e}")
    print(f"Phase MSE (wrapped, thresholded): {phase_mse:.6e}")
    
    # 5. Plotting 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot: |c_n| vs |d_n| ---
    ax1.plot(n_values, np.abs(c_n), 'b-', label='$|c_n|$ (Original)', alpha=0.8, linewidth=3)
    ax1.plot(n_values, np.abs(d_n_actual), 'r--', label='$|d_n|$ (Shifted)', alpha=0.8, linewidth=2)
    ax1.set_title("Magnitude Comparison (Should Overlap perfectly)")
    ax1.set_xlabel("$n$ (Harmonic Index)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Plot: angle(d_n) vs angle(c_n) - n*omega*t0 ---
    # Only plot phase for coefficients where magnitude is significant
    sig_mask = np.abs(c_n) > 1e-3
    n_sig = n_values[sig_mask]
    
    actual_phase = np.angle(d_n_actual[sig_mask])
    # Compute theoretical phase and wrap it to [-pi, pi]
    theoretical_phase = np.angle(c_n[sig_mask]) - (n_sig * omega * t0)
    theoretical_phase_wrapped = (theoretical_phase + np.pi) % (2 * np.pi) - np.pi
    
    ax2.scatter(n_sig, actual_phase, color='red', marker='o', label='Actual $\\angle d_n$', alpha=0.6)
    ax2.scatter(n_sig, theoretical_phase_wrapped, color='blue', marker='x', label='Theoretical Phase', alpha=0.9)
    ax2.set_title("Phase Comparison (Significant Harmonics)")
    ax2.set_xlabel("$n$ (Harmonic Index)")
    ax2.set_ylabel("Phase (radians)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("time_shift_property.png", bbox_inches='tight')
    print("\nPlots successfully saved as 'time_shift_property.png'.")