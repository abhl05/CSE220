import numpy as np
import matplotlib
matplotlib.use('TkAgg') # <-- Add this line before importing pyplot
import matplotlib.pyplot as plt

from fs_redrawer import FourierEpicycles
from svg_utils import load_svg_path
from epicycle_animation import save_outputs

if __name__ == "__main__":
    import sys
    from pathlib import Path
    N_HARMONICS = 150
    svg_path = Path("D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/heart.svg")
    t, signal = load_svg_path(svg_path, num_points=1000)
    fs = FourierEpicycles(t, signal, n_harmonics=N_HARMONICS)
    fs.calculate_all_coefficients()
    
    # Compute LHS and RHS
    LHS = np.trapezoid(np.abs(signal)**2, t) / fs.T
    RHS = np.sum(np.abs(list(fs.coeffs.values()))**2)
    
    ## relative error
    rel_error = np.abs(LHS - RHS) / LHS
    print(f"LHS: {LHS:.6f}, RHS: {RHS:.6f}")
    print(f"Relative error: {rel_error:.6e}")
    
    n_values = np.arange(-N_HARMONICS, N_HARMONICS + 1)
    c_n_values = np.array([fs.coeffs[n] for n in n_values])
    power_spectrum = np.abs(c_n_values)**2
    
    plt.figure(figsize=(10, 4))
    plt.stem(n_values, power_spectrum)
    plt.xlabel("n (Harmonic Index)")
    plt.ylabel("$|c_n|^2$ (Power)")
    plt.title("Power Spectrum")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig("power_spectrum.png", bbox_inches='tight') # <-- Replaces plt.show()
    plt.clf() # Clears the figure so the next plot starts fresh
    
    ## power spectrum |c_n|^2 vs n (stem plot), for n = -N ... N.
    k_values = np.arange(0, N_HARMONICS + 1)
    energy_at_k = np.zeros_like(k_values, dtype=float)
    
    for i, k in enumerate(k_values):
        if k == 0:
            energy_at_k[i] = np.abs(fs.coeffs[0])**2
        else:
            energy_at_k[i] = np.abs(fs.coeffs[k])**2 + np.abs(fs.coeffs[-k])**2
            
    cumulative_energy = np.cumsum(energy_at_k)
    
    plt.figure(figsize=(10, 4))
    plt.plot(k_values, cumulative_energy, marker='o', markersize=3)
    plt.xlabel("k (Number of Harmonic Pairs)")
    plt.ylabel("Cumulative Energy")
    plt.title("Cumulative Energy Captured vs $k$")
    plt.grid(True, alpha=0.3)
    plt.savefig("power_spectrum.png", bbox_inches='tight') # <-- Replaces plt.show()
    plt.clf() # Clears the figure so the next plot starts fresh