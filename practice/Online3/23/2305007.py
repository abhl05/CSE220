import numpy as np

from svg_utils import load_svg_path
from epicycle_animation import save_outputs


class FourierEpicycles:
    def __init__(self, t, signal, n_harmonics):
        self.t = t
        self.T = float(t[-1] - t[0])
        self.omega = float(2 * np.pi / self.T)
        self.signal = signal
        self.N = n_harmonics
        self.coeffs = {}

    def calculate_cn(self, n):
        cn = (1 / self.T) * np.trapezoid(
            self.signal * np.exp(-1j * n * self.omega * self.t), self.t
        )
        return cn

    def calculate_all_coefficients(self):
        for n in range(-self.N, self.N + 1):
            self.coeffs[n] = self.calculate_cn(n)

    def approximate(self, t):
        f_hat = np.zeros_like(t, dtype=complex)
        for n in range(-self.N, self.N + 1):
            f_hat += self.coeffs[n] * np.exp(1j * n * self.omega * t)
        return f_hat.item() if np.isscalar(t) else f_hat

    def prune_harmonics_by_energy(self, r: float):
        """
        Greedy version (no np.cumsum) - repeatedly grab the largest-energy
        remaining harmonic and accumulate, exactly like your original idea.
        Three changes from your version, each fixing a concrete bug:

          1. `ns`/`energies` are plain parallel lists indexed 0..2N, so
             there's no reliance on Python's negative-index wraparound to
             connect an array position back to a harmonic n.
          2. The loop is a bounded `for _ in range(len(ns))`, not an
             open-ended `while run_sum < target`. A while-loop keyed on an
             exact floating-point inequality can spin forever at r=1.00,
             because the running sum (accumulated in argmax-order) and
             E_t (accumulated via Python's sum() in dict-order) round
             differently in their last bit - see the diagnostic above.
             A bounded loop cannot hang, full stop, regardless of that.
          3. Discarded harmonics get self.coeffs[n] ACTUALLY set to zero
             (not just flagged in a side array), so approximate() - and
             therefore plot_comparison()/save_outputs() - correctly shows
             the pruned reconstruction with no extra bookkeeping needed.
        """
        ns = sorted(self.coeffs.keys())
        energies = np.array([abs(self.coeffs[n]) ** 2 for n in ns])
        E_total = energies.sum()
        target_energy = r * E_total

        remaining = energies.copy()
        run_sum = 0.0
        keep_positions = []

        for _ in range(len(ns)):
            i = int(np.argmax(remaining))
            if remaining[i] == 0:
                break  # nothing left to add (only possible if r > 1, guarded below)
            run_sum += remaining[i]
            keep_positions.append(i)
            remaining[i] = 0.0
            if run_sum >= target_energy:   # check AFTER adding -> no overshoot below r
                break

        
        for i, n in enumerate(ns):
            if i not in keep_positions:
                self.coeffs[n] = 0.0 + 0.0j   # actually discard it

        retained = len(keep_positions)
        actual_ratio = float(run_sum / E_total)
        return retained, actual_ratio

    def evaluate_reconstruction_error(self):
        """Uses the CURRENT (possibly pruned) self.coeffs via approximate(),
        so it automatically reflects whatever prune_harmonics_by_energy did."""
        f_hat = self.approximate(self.t)
        return float(np.mean(np.abs(self.signal - f_hat) ** 2))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from epicycle_animation import plot_comparison

    svg_path = sys.argv[1] if len(sys.argv) > 1 else "heart.svg"
    N_HARMONICS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stem = Path(svg_path).stem

    ratios = [0.96, 0.98, 0.99, 1.00]

    print(f"{'Target Ratio':>12} | {'Harmonics Retained':>18} | "
          f"{'Actual Energy Ratio':>20} | {'MSE':>10}")
    print("-" * 70)

    for r in ratios:
        t, z = load_svg_path(svg_path, num_points=1000)
        fs = FourierEpicycles(t, z, n_harmonics=N_HARMONICS)
        fs.calculate_all_coefficients()

        retained, actual_ratio = fs.prune_harmonics_by_energy(r)
        mse = fs.evaluate_reconstruction_error()

        print(f"{r:>12.2f} | {retained:>18d} | {actual_ratio:>20.4f} | {mse:>10.3e}")

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_comparison(fs, z, ax=ax)
        ax.set_title(f"{stem}, r={r:.2f} ({retained} harmonics, MSE={mse:.2e})")
        fig.savefig(f"{stem}_pruned_{r:.2f}.png", dpi=120)
        plt.close(fig)

    print("\nSaved: " + ", ".join(f"{stem}_pruned_{r:.2f}.png" for r in ratios))