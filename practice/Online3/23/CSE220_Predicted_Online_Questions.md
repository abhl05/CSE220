# CSE 220 — Predicted "Online" Questions
### Based on: Jan'26 Offline — Fourier Series Epicycles (`fs_redrawer.py`) & 2D-CFT Edge Detection (`cft_edge_detector.py`)

---

## How I derived these

Every past online you've shown me follows the same recipe: **take the offline's OOP
skeleton, keep the class structure fixed, and bolt on ONE untested property or
theorem**, timed 30–40 min, always with "no `np.fft` anywhere, trapz only."
Nothing in the online ever requires new algorithms — it always reuses
`calculate_cn` / `approximate` or `compute_cft` / `reconstruct` exactly as you
already wrote them, just fed different inputs or checked against a different
theoretical prediction.

So the question bank below is organized by **which property gets bolted onto
which class**, mirroring last year's A1A2 (derivative), B1B2 (time-shift),
C1C2 (scale+modulate), B1B2 (Parseval), C1C2 (trig decomposition), and the
"mission" (image denoising) formats.

---

# TRACK 1 — built on `FourierEpicycles` (Task 1: FS)

## Predicted Online 1: Parseval's Theorem for Fourier Series
**Time: 30 minutes**

**Background.** For a periodic signal with Fourier series coefficients `c_n`,
Parseval's theorem states
```
(1/T) * Integral_0^T |f(t)|^2 dt   =   Sum_{n=-inf}^{inf} |c_n|^2
```
i.e. total average power in the time domain equals the total power carried by
all harmonics.

**Task.**
1. Load any provided SVG (e.g. `heart.svg`) with `load_svg_path`, build a
   `FourierEpicycles` instance with `N = 150`, call
   `calculate_all_coefficients()`.
2. Compute the LHS using `np.trapezoid(np.abs(signal)**2, t) / T`.
3. Compute the RHS by summing `abs(c_n)**2` over all stored coefficients.
4. Report the relative error between LHS and RHS.
5. Plot the **power spectrum**: `|c_n|^2` vs `n` (stem plot), for
   `n = -N ... N`.
6. Extra (very likely add-on): plot **cumulative energy captured** —
   `sum_{|n|<=k} |c_n|^2` vs `k` — to show how quickly the shape's energy is
   captured as you add harmonics, and comment on which shapes (smooth
   `circle.svg` vs cornered `star.svg`) converge faster.

**Constraints:** reuse your existing `FourierEpicycles` methods unmodified;
`np.trapezoid` only; no `np.fft`.

*Why likely: this is the exact same "Parseval" online format from last year's
B1-B2, just moved onto this year's FS class — very high confidence.*

---

## Predicted Online 2: Time-Shift Property of the Fourier Series
**Time: 30 minutes**

**Background.** If `g(t) = f(t - t0)` (same period `T`), then
```
d_n = c_n * e^{-j n omega t0}
```
i.e. shifting in time only rotates the phase of each coefficient — magnitude
is unchanged.

**Task.**
1. Build `f` from an SVG as usual.
2. Construct the shifted signal `g(t) = f(t - t0)` for some `t0` (e.g.
   `t0 = T/4`) via interpolation on the closed `[0, T]` grid (`np.interp`,
   wrapping periodically — do **not** just slice/roll the array by hand,
   mirror the "OOP, no manual shifting" instruction from last year's
   Gaussian time-shift online).
3. Build a second `FourierEpicycles` instance on `g`, get `d_n`.
4. Compare `d_n` against the theoretical `c_n * e^{-j n omega t0}` for every
   `n`: magnitude MSE and (wrapped) phase MSE.
5. Plot `|c_n|` vs `|d_n|` (should overlap) and `angle(d_n)` vs
   `angle(c_n) - n*omega*t0`.
6. Conceptual question likely appended: "Does the traced *shape* change under
   a pure time shift? Why or why not?" (Answer: no — it only changes where in
   the drawing the animation starts / the phase reference, since it's the same
   closed curve retraced from a different starting point.)

**Constraints:** trapz only, no fft, OOP shift (no manual array slicing).

*Why likely: directly parallels last year's Online-03 (Gaussian, t0=1) —
same property, same verification structure, just complex-signal/FS flavored.*

---

## Predicted Online 3: Convergence / Gibbs Phenomenon vs. Number of Harmonics
**Time: 30–40 minutes**

**Background.** Truncating a Fourier series to `N` harmonics causes visible
ringing ("Gibbs phenomenon") near sharp corners/discontinuities in the
derivative of the traced path, and converges more slowly there than for
smooth curves.

**Task.**
1. Using `star.svg` (has sharp corners) and `circle.svg` (perfectly smooth),
   compute reconstructions at several harmonic counts, e.g.
   `N in {5, 10, 20, 50, 100, 150, 300}`.
2. For each `N`, compute MSE between `fs.approximate(t_dense)` and the true
   `z_true` over one period.
3. Plot MSE vs `N` (log-log or semilog-y) for both shapes on the same axes.
4. Zoom in on a corner of the star reconstruction at low N and comment on the
   visible overshoot (Gibbs).
5. Comment on why the smooth shape's MSE decays much faster with `N`.

**Constraints:** reuse `FourierEpicycles` unmodified across all `N` values
(new instance per `N`); trapz only.

*Why likely: this is a completely standard, obvious pedagogical follow-up the
offline sets up perfectly (four shapes given, "must work on all of them"
already hints the professor wants you comparing them) but never actually
tests numerically — a natural online.*

---

## Predicted Online 4: Differentiation Property / Constant-Speed Check
**Time: 30 minutes**

**Background.** If `f(t)` has coefficients `c_n`, then
```
f'(t)  <-->  j n omega c_n
```

**Task.**
1. Numerically differentiate the sampled complex signal:
   `v = np.gradient(signal, t)` (this is the pen's *velocity* vector).
2. Build a **separate** `FourierEpicycles` instance treating `v` as the
   "signal" (same `t`, same `N`), get its coefficients `e_n`.
3. Compare `e_n` against the theoretical `1j * n * omega * c_n` (using the
   *original* signal's coefficients) — magnitude & phase MSE per harmonic.
4. **Bonus twist specific to this offline:** because `load_svg_path`
   re-parametrizes the curve to *equal arc length* (constant tracing speed),
   `|v(t)|` should be nearly constant across `t`. Plot `|v(t)|` and comment
   on whether this matches the equal-arc-length claim in `svg_utils.py`'s
   docstring — a nice numerical sanity-check on code you were told to treat
   as a black box.

**Constraints:** trapz only, no fft.

*Why likely: mirrors last year's A1-A2 (derivative property), and the
"constant speed" detail is a strong, on-brand exam hook since it's explicitly
called out in the provided `svg_utils.py` docstring you were told not to read
closely — professors love testing whether you actually understood a "black
box" module's stated guarantee.*

---

## Predicted Online 5: Linearity Property (Combining Two Shapes)
**Time: 30 minutes**

**Background.** If `h(t) = f(t) + g(t)` (same `T`, same sampling), then
`h`'s Fourier coefficients are simply `c_n(h) = c_n(f) + c_n(g)`.

**Task.**
1. Load two SVGs sharing the same `t` grid (e.g. `heart.svg` and a scaled/
   translated copy, or `heart.svg` + `circle.svg`).
2. Compute `FourierEpicycles` coefficients for each individually, and for
   their sum signal `h = f + g`.
3. Verify `c_n(h) == c_n(f) + c_n(g)` for all `n` (MSE near machine
   precision).
4. Plot the three reconstructed shapes overlaid to visually show the
   combined drawing.

**Constraints:** trapz only, no fft; must not hardcode coefficients.

*Why likely: linearity is the cheapest, fastest property to test and a common
"warm-up" online — lower confidence than 1–4 but plausible as a short 20–25
min variant.*

---

# TRACK 2 — built on `CFT2D` / `InverseCFT2D` (Task 2: 2D-CFT)

## Predicted Online 6: Parseval's Theorem in 2D (Image Energy Conservation)
**Time: 30–40 minutes**

**Background.**
```
Integral Integral |I(x,y)|^2 dx dy   =   Integral Integral |F(u,v)|^2 du dv
```

**Task.**
1. Using your completed `ContinuousImage` + `CFT2D` classes on `pikachu.png`
   (or any provided image), compute `real, imag = cft2d.compute_cft()`.
2. Compute time-domain energy: `np.trapezoid(np.trapezoid(I**2, x, axis=1), y)`.
3. Compute frequency-domain energy the same separable way over
   `real**2 + imag**2` on the `u, v` grid.
4. Report relative error; comment on why it should be small on a fine grid.

**Constraints:** trapz only, separable integration only (no direct O(N^4)
double loop), no fft — identical constraints to the offline.

*Why likely: this is the single most predictable online in the whole set —
it's the exact 2D generalization of last year's B1-B2 Parseval problem, and
your offline already built every piece needed (`compute_cft`) except the
energy check itself.*

---

## Predicted Online 7: Low-Pass Filtering (Blurring) — the Mirror of the Offline's High-Pass
**Time: 30 minutes**

**Background.** The offline's `FrequencyFilter.high_pass` zeroes everything
*within* radius `cutoff` (keeping edges). Flipping the mask — zeroing
everything *outside* `cutoff` — keeps only the slowly-varying low-frequency
content, i.e. a **blur**.

**Task.**
1. Using the same pipeline (`CFT2D.compute_cft` → filter → `InverseCFT2D.
   reconstruct`), implement (in your submission script — no need to add a
   class method, exactly like the offline forbids new class members) a
   low-pass mask keeping only `sqrt((i-cx)^2+(j-cy)^2) <= cutoff`.
2. Reconstruct and display the blurred image at 2–3 cutoff values (e.g. 5,
   15, 40) and comment on the trade-off between blur strength and
   information loss.
3. Compute the **fraction of total spectral energy retained** at each
   cutoff (`sum |F|^2` inside the disk `/` total `sum |F|^2`) and relate it
   to how recognizable the blurred image is.

**Constraints:** reuse `compute_cft`/`reconstruct` exactly; no new classes;
trapz only.

*Why likely: an almost-too-obvious twist on the offline's own
`FrequencyFilter.high_pass` — inverting one boolean condition — which makes
it a very cheap, very likely online (low implementation risk for the grader
too, since it reuses 95% of your offline code).*

---

## Predicted Online 8: Spatial Shift (Translation) Property of the 2D-CFT
**Time: 30 minutes**

**Background.**
```
I(x - x0, y - y0)   <-->   F(u,v) * e^{-j2*pi*(u*x0 + v*y0)}
```

**Task.**
1. Shift `pikachu.png` by `(x0, y0)` in continuous coordinates (interpolate
   onto the shifted `x, y` grid — OOP style, no `np.roll`).
2. Run both the original and shifted images through `CFT2D.compute_cft`.
3. Verify `|F_shifted(u,v)| ≈ |F(u,v)|` (magnitude unchanged) and
   `angle(F_shifted) ≈ angle(F) - 2*pi*(u*x0+v*y0)` (restricted to
   frequencies where `|F|` is non-negligible, exactly like the tail-noise
   caveat from your Gaussian time-shift online last year).
4. Report magnitude & phase MSE.

**Constraints:** separable trapz integration, no fft, reuse `CFT2D` as-is.

*Why likely: direct 2D generalization of last year's Online-03 (Gaussian
time-shift) — very plausible, moderate implementation cost (mainly the
interpolation-based shift).*

---

## Predicted Online 9: Reconstruction Sanity Check + Band-Pass Filter
**Time: 30–40 minutes**

**Background.** Skipping the filter entirely should let
`InverseCFT2D.reconstruct()` recover the original image almost exactly —
this is the most basic possible correctness check on your `compute_cft` /
`reconstruct` pair. A **band-pass** filter (an annulus, not a disk) isolates
a specific *scale* of texture rather than "everything above/below one
cutoff."

**Task.**
1. Run `real, imag = compute_cft()` on the unfiltered spectrum straight into
   `reconstruct()`. Report MSE between the reconstructed and original image
   (should be tiny) — a direct validation of your Task 2 implementation.
2. Implement a band-pass mask keeping only
   `r1 <= sqrt((i-cx)^2+(j-cy)^2) <= r2` for two radii `r1 < r2`, reconstruct,
   and describe what the result isolates (mid-scale texture/edges) versus a
   plain high-pass.

**Constraints:** no new classes; trapz only; no fft.

*Why likely: the "does your inverse actually invert your forward transform"
check is a natural, cheap opening sub-task any professor would want verified
before trusting a more elaborate property question — moderate-high
confidence as part of a longer online, lower confidence as a standalone one.*

---

## Predicted Online 10 ("Mission"-style): Frequency-Domain Denoising Using Your Own CFT2D
**Time: 40 minutes**

**Background.** This is the natural sequel to last year's "Prove Your
Relevancy" secret-agent mission — but this year's offline explicitly bans
`np.fft` everywhere, so the *same kind* of periodic-stripe-noise denoising
task would have to be done with your own trapezoidal `CFT2D`/`InverseCFT2D`
instead. Expect a **small** noisy image (large images would make an O(N^3)
trapz transform too slow for a 40-minute exam — watch for a hint like
"image has been downsampled to keep runtime reasonable").

**Task.**
1. Load a noisy image with `ContinuousImage`, run `compute_cft()`.
2. Call `plot_magnitude()` and visually identify the noise peak(s) in the
   spectrum (e.g. bright dots or a bright line off-center, as in the
   stripe-noise pattern from last year's mission).
3. Build a custom mask in your submission script (not a new class method)
   that zeros only those specific frequency-index regions, leaving the rest
   of the spectrum untouched.
4. Reconstruct with `InverseCFT2D.reconstruct()` and display the recovered
   image/letter.

**Constraints:** trapz-only 2D CFT throughout (no `np.fft` — this is the key
difference from last year's mission, which allowed it); no new classes; must
exploit separability (an O(N^4) direct approach will time out on even a
small image).

*Why likely but riskier: thematically very on-brand given last year's exact
"secret letter in noisy image" mission, but the O(N^3) trapz cost is heavy
for a timed online unless the image is quite small — moderate confidence.*

---

## General patterns to prepare for regardless of exact wording

- **Always trapz-only, always no FFT** — every single past online and this
  offline enforce it; assume it's non-negotiable.
- **Reuse your offline classes completely unmodified** — every past online
  built directly on the offline's exact class/method signatures; expect the
  same here (`FourierEpicycles.calculate_cn/approximate`,
  `CFT2D.compute_cft`, `InverseCFT2D.reconstruct`).
- **MSE-based verification is mandatory** — every property question ends in
  "compute the MSE between numerical and theoretical," usually magnitude +
  phase separately, sometimes with a "restrict to where magnitude is
  significant" caveat (both your Gaussian-shift and mission examples from
  last year hit exactly this issue with near-zero-magnitude phase noise).
- **Plot pairs**: magnitude-vs-magnitude and phase-vs-phase overlays for 1D
  properties; before/after images (or magnitude-spectrum panels) for the 2D
  CFT track.
- **Time budget**: 30 min for a single clean property, 40 min when a
  visualization/mission element is added.
- **Submission format**: single `StudentID.py`, following past convention.

I'm happy to draft full mock-exam PDFs (matching the past papers' exact
formatting) or complete worked solutions for any of the ten above if you want
to rehearse a specific one before the real online.
