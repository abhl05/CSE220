"""
CSE 220 — Online 5: Linearity Property (Combining Two Shapes)
=================================================================
Verifies c_n(h) = c_n(f) + c_n(g) for h(t) = f(t) + g(t).
"""
import numpy as np
import matplotlib.pyplot as plt
from fs_redrawer import FourierEpicycles
from svg_utils import load_svg_path


def linearity_property(svg_path1, svg_path2, N=150):
    """
    Verify linearity: h = f + g  =>  c_n(h) = c_n(f) + c_n(g).
    
    Parameters
    ----------
    svg_path1, svg_path2 : str
        Paths to two SVG files (must share same t grid)
    N : int
        Number of harmonics
    """
    # Step 1: Load both SVGs
    t1, z1 = load_svg_path(svg_path1, num_points=1000)
    t2, z2 = load_svg_path(svg_path2, num_points=1000)
    
    # Ensure same t grid (if different, interpolate to common grid)
    if not np.allclose(t1, t2):
        T = t1[-1] - t1[0]
        t_common = np.linspace(0, T, 1000)
        z1 = np.interp(t_common, t1, z1)
        z2 = np.interp(t_common, t2, z2)
        t = t_common
    else:
        t = t1
    
    # Step 2: Compute coefficients for each individually
    fs_f = FourierEpicycles(t, z1, n_harmonics=N)
    fs_f.calculate_all_coefficients()
    c_f = fs_f.coeffs
    
    fs_g = FourierEpicycles(t, z2, n_harmonics=N)
    fs_g.calculate_all_coefficients()
    c_g = fs_g.coeffs
    
    # Step 3: Compute coefficients for sum signal h = f + g
    z_h = z1 + z2
    fs_h = FourierEpicycles(t, z_h, n_harmonics=N)
    fs_h.calculate_all_coefficients()
    c_h = fs_h.coeffs
    
    # Step 4: Verify c_n(h) == c_n(f) + c_n(g)
    n_vals = np.arange(-N, N + 1)
    c_h_numeric = np.array([c_h[n] for n in n_vals])
    c_h_theoretical = np.array([c_f[n] + c_g[n] for n in n_vals])
    
    mse = np.mean(np.abs(c_h_numeric - c_h_theoretical) ** 2)
    max_error = np.max(np.abs(c_h_numeric - c_h_theoretical))
    
    print(f"Shape 1: {svg_path1}")
    print(f"Shape 2: {svg_path2}")
    print(f"  MSE between c_n(h) and c_n(f)+c_n(g):  {mse:.2e}")
    print(f"  Max error:                               {max_error:.2e}")
    print(f"  Linearity property: {'VERIFIED ✓' if mse < 1e-10 else 'FAILED ✗'}")
    
    # Step 5: Plot reconstructed shapes overlaid
    T = t[-1] - t[0]
    t_plot = np.linspace(0, T, 2000)
    
    z1_recon = fs_f.approximate(t_plot)
    z2_recon = fs_g.approximate(t_plot)
    z_h_recon = fs_h.approximate(t_plot)
    z_h_sum_recon = z1_recon + z2_recon
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Individual shapes
    ax = axes[0]
    ax.plot(z1_recon.real, z1_recon.imag, 'b-', linewidth=1.5, label='f(t)')
    ax.plot(z2_recon.real, z2_recon.imag, 'r-', linewidth=1.5, label='g(t)')
    ax.set_aspect('equal')
    ax.set_title('Individual Shapes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Combined shape
    ax = axes[1]
    ax.plot(z_h_recon.real, z_h_recon.imag, 'g-', linewidth=2, label='h(t) = f(t)+g(t)')
    ax.plot(z_h_sum_recon.real, z_h_sum_recon.imag, 'm--', linewidth=1, alpha=0.7, label='fs_f + fs_g')
    ax.set_aspect('equal')
    ax.set_title('Combined Shape (Linearity Verification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('online5_linearity.png', dpi=150)
    plt.show()
    
    # Plot 3: Coefficient comparison (scatter)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(c_h_numeric.real, c_h_numeric.imag, c='blue', s=20, alpha=0.6, label='c_n(h) numeric')
    ax.scatter(c_h_theoretical.real, c_h_theoretical.imag, c='red', s=10, alpha=0.6, label='c_n(f)+c_n(g)')
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title('Coefficient Comparison in Complex Plane')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('online5_linearity_complex.png', dpi=150)
    plt.show()
    
    return {
        'mse': mse,
        'max_error': max_error,
        'c_h_numeric': c_h_numeric,
        'c_h_theoretical': c_h_theoretical
    }


if __name__ == "__main__":
    result = linearity_property("D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/heart.svg", "D:/BUET/CSE220/Offline/Jan2026_CSE220_Offline_FS_CFT/task1/svgs/circle.svg", N=150)