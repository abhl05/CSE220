"""
CSE 220 — Online 1: Parseval's Theorem for Fourier Series
============================================================
Verifies that average power in time domain equals sum of |c_n|^2.
Also plots power spectrum and cumulative energy vs harmonic count.
"""
import numpy as np
import matplotlib.pyplot as plt
from fs_redrawer import FourierEpicycles
from svg_utils import load_svg_path


def parseval_check(svg_path, N=150):
    """
    Compute and verify Parseval's theorem for a given SVG shape.
    
    Parameters
    ----------
    svg_path : str
        Path to the SVG file (e.g., 'svgs/heart.svg')
    N : int
        Number of harmonics (default 150)
    
    Returns
    -------
    dict with LHS, RHS, relative_error, and plot data
    """
    # Step 1: Load SVG and build FourierEpicycles instance
    t, z = load_svg_path(svg_path, num_points=1000)
    fs = FourierEpicycles(t, z, n_harmonics=N)
    fs.calculate_all_coefficients()
    
    T = fs.T
    
    # Step 2: Compute LHS — average power in time domain
    lhs = np.trapezoid(np.abs(fs.signal) ** 2, fs.t) / T
    
    # Step 3: Compute RHS — sum of |c_n|^2 over all stored coefficients
    cn_values = np.array([fs.coeffs[n] for n in range(-N, N + 1)])
    rhs = np.sum(np.abs(cn_values) ** 2)
    
    # Step 4: Report relative error
    rel_error = np.abs(lhs - rhs) / np.abs(lhs)
    
    print(f"SVG: {svg_path}")
    print(f"  LHS (time-domain power): {lhs:.10f}")
    print(f"  RHS (sum |c_n|^2):       {rhs:.10f}")
    print(f"  Relative error:          {rel_error:.2e}")
    
    # Step 5: Plot power spectrum |c_n|^2 vs n
    n_range = np.arange(-N, N + 1)
    power = np.abs(cn_values) ** 2
    
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Power spectrum (stem plot)
    plt.subplot(1, 2, 1)
    plt.stem(n_range, power, basefmt=' ')
    plt.xlabel('Harmonic n')
    plt.ylabel('|c_n|²')
    plt.title(f'Power Spectrum — {svg_path.split("/")[-1]}')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative energy captured
    plt.subplot(1, 2, 2)
    # Sort by |n| so we add harmonics symmetrically around 0
    k_vals = np.arange(0, N + 1)
    cum_energy = np.zeros(N + 1)
    for k in k_vals:
        # Sum |c_n|^2 for |n| <= k
        mask = np.abs(n_range) <= k
        cum_energy[k] = np.sum(power[mask])
    
    plt.plot(k_vals, cum_energy, 'b-', linewidth=2)
    plt.axhline(y=rhs, color='r', linestyle='--', label=f'Total = {rhs:.4f}')
    plt.xlabel('Max harmonic |n|')
    plt.ylabel('Cumulative Energy')
    plt.title('Energy Captured vs Harmonic Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('online1_parseval.png', dpi=150)
    plt.show()
    
    return {
        'lhs': lhs,
        'rhs': rhs,
        'rel_error': rel_error,
        'n_range': n_range,
        'power': power,
        'cum_energy': cum_energy
    }


def compare_shapes(svg_paths, N=150):
    """
    Compare convergence rates between smooth (circle) and cornered (star) shapes.
    """
    plt.figure(figsize=(10, 6))
    
    for svg_path in svg_paths:
        t, z = load_svg_path(svg_path, num_points=1000)
        fs = FourierEpicycles(t, z, n_harmonics=N)
        fs.calculate_all_coefficients()
        
        cn_values = np.array([fs.coeffs[n] for n in range(-N, N + 1)])
        power = np.abs(cn_values) ** 2
        n_range = np.arange(-N, N + 1)
        total_energy = np.sum(power)
        
        k_vals = np.arange(0, N + 1)
        frac_captured = np.zeros(N + 1)
        for k in k_vals:
            mask = np.abs(n_range) <= k
            frac_captured[k] = np.sum(power[mask]) / total_energy
        
        label = svg_path.split('/')[-1].replace('.svg', '')
        plt.plot(k_vals, frac_captured, linewidth=2, label=label)
    
    plt.xlabel('Max harmonic |n|')
    plt.ylabel('Fraction of Total Energy Captured')
    plt.title('Energy Convergence: Smooth vs Cornered Shapes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('online1_convergence_comparison.png', dpi=150)
    plt.show()
    
    print("\\nComment: Smooth shapes (circle) converge faster because their")
    print("Fourier coefficients decay as O(1/n²) or faster, while shapes with")
    print("sharp corners (star) have discontinuous derivatives, leading to")
    print("O(1/n) decay and slower convergence (Gibbs phenomenon).")


if __name__ == "__main__":
    # Main task: Parseval check on heart.svg
    result = parseval_check("D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/heart.svg", N=150)
    
    # Extra: Compare convergence for multiple shapes
    # compare_shapes(["svgs/circle.svg", "svgs/star.svg"], N=150)