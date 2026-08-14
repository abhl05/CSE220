"""
CSE 220 — Online 8: Spatial Shift (Translation) Property of the 2D-CFT
=========================================================================
Verifies that I(x-x0, y-y0) <--> F(u,v) * e^{-j*2*pi*(u*x0 + v*y0)}.
Uses interpolation-based OOP shift (no np.roll).
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.interpolate import RegularGridInterpolator
from cft_edge_detector import ContinuousImage, CFT2D


def shift_image_interpolated(image_obj, x0, y0):
    """
    Shift an image by (x0, y0) using interpolation (OOP style, no np.roll).
    
    Parameters
    ----------
    image_obj : ContinuousImage
        Original image object
    x0, y0 : float
        Shift amounts in continuous coordinates
    
    Returns
    -------
    ContinuousImage-like object with shifted image
    """
    x = image_obj.x
    y = image_obj.y
    I = image_obj.image
    
    # Create interpolator
    # Note: image is I[y_idx, x_idx], so we pass (y, x) coordinates
    interp = RegularGridInterpolator(
        (y, x), I,
        bounds_error=False, fill_value=0.0, method='linear'
    )
    
    # Create shifted coordinate grids
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Query at (x - x0, y - y0) to get I(x - x0, y - y0)
    # Because we want g(x,y) = f(x - x0, y - y0)
    points = np.stack([Y.flatten() - y0, X.flatten() - x0], axis=-1)
    I_shifted = interp(points).reshape(I.shape)
    
    # Create a new ContinuousImage-like object
    class ShiftedImage:
        def __init__(self, image, x, y):
            self.image = image
            self.x = x
            self.y = y
    
    return ShiftedImage(I_shifted, x, y)


def spatial_shift_property(image_path, x0=0.1, y0=0.1):
    """
    Verify the 2D spatial shift property.
    
    Parameters
    ----------
    image_path : str
        Path to input image
    x0, y0 : float
        Shift amounts in the [-1, 1] coordinate system
    """
    # Step 1: Load original image
    img_orig = ContinuousImage(image_path)
    cft_orig = CFT2D(img_orig)
    real_orig, imag_orig = cft_orig.compute_cft()
    F_orig = real_orig + 1j * imag_orig
    
    # Step 2: Shift image by interpolation
    img_shifted = shift_image_interpolated(img_orig, x0, y0)
    cft_shifted = CFT2D(img_shifted)
    real_shifted, imag_shifted = cft_shifted.compute_cft()
    F_shifted = real_shifted + 1j * imag_shifted
    
    u = cft_orig.u
    v = cft_orig.v
    
    # Step 3: Verify magnitude unchanged
    # |F_shifted(u,v)| should equal |F_orig(u,v)|
    mag_orig = np.abs(F_orig)
    mag_shifted = np.abs(F_shifted)
    
    # Only compare where magnitude is significant (avoid noise in tails)
    threshold = 1e-6 * np.max(mag_orig)
    significant = mag_orig > threshold
    
    mag_mse = np.mean((mag_shifted[significant] - mag_orig[significant]) ** 2)
    
    # Step 4: Verify phase relationship
    # angle(F_shifted) should equal angle(F_orig) - 2*pi*(u*x0 + v*y0)
    U, V = np.meshgrid(u, v, indexing='xy')
    phase_theoretical = np.angle(F_orig) - 2 * np.pi * (U * x0 + V * y0)
    phase_numeric = np.angle(F_shifted)
    
    # Wrap phase difference
    phase_diff = np.angle(np.exp(1j * (phase_numeric - phase_theoretical)))
    phase_mse = np.mean(phase_diff[significant] ** 2)
    
    print(f"Image: {image_path}")
    print(f"  Shift: (x0, y0) = ({x0}, {y0})")
    print(f"  Magnitude MSE:  {mag_mse:.2e}")
    print(f"  Phase MSE:      {phase_mse:.2e}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    ax = axes[0, 0]
    ax.imshow(img_orig.image, cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')
    
    # Shifted image
    ax = axes[0, 1]
    ax.imshow(img_shifted.image, cmap='gray')
    ax.set_title(f'Shifted by ({x0}, {y0})')
    ax.axis('off')
    
    # Difference
    ax = axes[0, 2]
    diff = np.abs(img_shifted.image - img_orig.image)
    ax.imshow(diff, cmap='hot')
    ax.set_title('Absolute Difference')
    ax.axis('off')
    
    # Magnitude comparison
    ax = axes[1, 0]
    ax.plot(mag_orig[significant].flatten(), mag_shifted[significant].flatten(), 
            'b.', markersize=1, alpha=0.3)
    ax.plot([0, np.max(mag_orig)], [0, np.max(mag_orig)], 'r--', linewidth=1)
    ax.set_xlabel('|F_orig|')
    ax.set_ylabel('|F_shifted|')
    ax.set_title('Magnitude: Orig vs Shifted')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Phase comparison (sample a few points)
    ax = axes[1, 1]
    # Sample every 10th point for clarity
    sample = significant[::10, ::10]
    ax.plot(phase_theoretical[::10, ::10].flatten(), 
            phase_numeric[::10, ::10].flatten(), 
            'g.', markersize=1, alpha=0.3)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], 'r--', linewidth=1)
    ax.set_xlabel('Theoretical Phase')
    ax.set_ylabel('Numeric Phase')
    ax.set_title('Phase: Theory vs Numeric')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Phase difference histogram
    ax = axes[1, 2]
    ax.hist(phase_diff[significant].flatten(), bins=100, color='purple', 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Phase Difference (rad)')
    ax.set_ylabel('Count')
    ax.set_title('Phase Difference Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('online8_spatial_shift.png', dpi=150)
    plt.show()
    
    return {
        'mag_mse': mag_mse,
        'phase_mse': phase_mse,
        'x0': x0,
        'y0': y0
    }


if __name__ == "__main__":
    result = spatial_shift_property("D:\\BUET\\CSE220\\Offline\\Jan2026_CSE220_Offline_FS_CFT\\task2\\pikachu.png", x0=0.1, y0=0.1)