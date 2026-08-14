"""
CSE 220 — Online 4: Differentiation Property / Constant-Speed Check
=====================================================================
Verifies f'(t) <--> j*n*omega*c_n.
Also checks that |v(t)| is nearly constant (equal arc-length reparametrization).
"""
import numpy as np
import matplotlib.pyplot as plt
from fs_redrawer import FourierEpicycles
from svg_utils import load_svg_path


def differentiation_property(svg_path, N=150):
    """
    Verify the differentiation property and check constant-speed claim.
    
    Parameters
    ----------
    svg_path : str
        Path to SVG file
    N : int
        Number of harmonics
    """
    # Step 1: Load signal and compute velocity numerically
    t, z = load_svg_path(svg_path, num_points=1000)
    T = t[-1] - t[0]
    omega = 2 * np.pi / T
    
    # Numerical derivative: v(t) = dz/dt
    v = np.gradient(z, t)
    
    # Step 2: Build FourierEpicycles for original signal f(t)
    fs_f = FourierEpicycles(t, z, n_harmonics=N)
    fs_f.calculate_all_coefficients()
    c_n = fs_f.coeffs
    
    # Step 3: Build FourierEpicycles for velocity signal v(t)
    fs_v = FourierEpicycles(t, v, n_harmonics=N)
    fs_v.calculate_all_coefficients()
    e_n = fs_v.coeffs  # coefficients of derivative
    
    # Step 4: Compare e_n against theoretical j*n*omega*c_n
    n_vals = np.arange(-N, N + 1)
    e_numeric = np.array([e_n[n] for n in n_vals])
    e_theoretical = np.array([1j * n * omega * c_n[n] for n in n_vals])
    
    # Magnitude comparison
    mag_mse = np.mean(np.abs(np.abs(e_numeric) - np.abs(e_theoretical)) ** 2)
    
    # Phase comparison (wrapped, significant coeffs only)
    phase_numeric = np.angle(e_numeric)
    phase_theoretical = np.angle(e_theoretical)
    phase_diff = np.angle(np.exp(1j * (phase_numeric - phase_theoretical)))
    significant = np.abs(e_numeric) > 1e-10 * np.max(np.abs(e_numeric))
    phase_mse = np.mean(phase_diff[significant] ** 2) if np.any(significant) else 0.0
    
    print(f"SVG: {svg_path}")
    print(f"  Magnitude MSE:  {mag_mse:.2e}")
    print(f"  Phase MSE:      {phase_mse:.2e}")
    
    # Step 5: Check constant speed claim
    speed = np.abs(v)
    speed_mean = np.mean(speed)
    speed_std = np.std(speed)
    speed_cv = speed_std / speed_mean  # coefficient of variation
    
    print(f"\\n--- Constant Speed Check ---")
    print(f"  Mean |v(t)|:    {speed_mean:.6f}")
    print(f"  Std |v(t)|:     {speed_std:.6e}")
    print(f"  CV (std/mean):  {speed_cv:.4f} ({speed_cv*100:.2f}%)")
    
    if speed_cv < 0.05:
        print(f"  CONCLUSION: Speed is nearly constant — equal arc-length")
        print(f"  reparametrization claim is VALIDATED.")
    else:
        print(f"  CONCLUSION: Speed varies significantly — equal arc-length")
        print(f"  reparametrization may not be perfectly achieved.")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: |e_n| comparison
    ax = axes[0, 0]
    ax.plot(n_vals, np.abs(e_numeric), 'b-o', markersize=3, label='|e_n| (numeric)')
    ax.plot(n_vals, np.abs(e_theoretical), 'r--s', markersize=3, label='|j*n*ω*c_n| (theory)')
    ax.set_xlabel('Harmonic n')
    ax.set_ylabel('|Coefficient|')
    ax.set_title('Derivative Coefficients: Numeric vs Theory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Phase comparison
    ax = axes[0, 1]
    ax.plot(n_vals[significant], phase_numeric[significant], 'b-o', markersize=3, label='angle(e_n)')
    ax.plot(n_vals[significant], phase_theoretical[significant], 'r--s', markersize=3, label='angle(j*n*ω*c_n)')
    ax.set_xlabel('Harmonic n')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Phase: Numeric vs Theory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speed over time
    ax = axes[1, 0]
    ax.plot(t, speed, 'b-', linewidth=1)
    ax.axhline(y=speed_mean, color='r', linestyle='--', label=f'Mean = {speed_mean:.4f}')
    ax.fill_between(t, speed_mean - speed_std, speed_mean + speed_std, alpha=0.2, color='red')
    ax.set_xlabel('t')
    ax.set_ylabel('|v(t)| = |dz/dt|')
    ax.set_title('Tracing Speed vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Speed histogram
    ax = axes[1, 1]
    ax.hist(speed, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=speed_mean, color='r', linestyle='--', linewidth=2, label=f'Mean = {speed_mean:.4f}')
    ax.set_xlabel('|v(t)|')
    ax.set_ylabel('Frequency')
    ax.set_title('Speed Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('online4_differentiation.png', dpi=150)
    plt.show()
    
    return {
        'mag_mse': mag_mse,
        'phase_mse': phase_mse,
        'speed_mean': speed_mean,
        'speed_std': speed_std,
        'speed_cv': speed_cv
    }


if __name__ == "__main__":
    result = differentiation_property("D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/heart.svg", N=150)