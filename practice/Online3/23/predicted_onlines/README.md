# Predicted Online Solutions — How to Use This Bundle

## Folder layout
```
predicted_onlines/
├── StudentID_Pred1_FS_Parseval.py           <- Track 1 (Fourier Series)
├── StudentID_Pred2_FS_timeshift.py
├── StudentID_Pred3_FS_convergence_gibbs.py
├── StudentID_Pred4_FS_derivative_speed.py
├── StudentID_Pred5_FS_linearity.py
├── StudentID_Pred6_CFT2D_parseval.py         <- Track 2 (2D CFT / edge detection)
├── StudentID_Pred7_CFT2D_lowpass_blur.py
├── StudentID_Pred8_CFT2D_translation.py
├── StudentID_Pred9_CFT2D_sanity_bandpass.py
├── StudentID_Pred10_CFT2D_denoise_mission.py
├── offline_modules/     <- your own offline files, copied in unmodified
│   ├── fs_redrawer.py
│   ├── svg_utils.py
│   ├── epicycle_animation.py
│   └── cft_edge_detector.py
└── test_assets/         <- synthetic test data I generated to verify everything runs
    ├── svgs/circle.svg, heart.svg, star.svg   (simple hand-built SVGs)
    ├── test_face.png                          (synthetic face-like image, 48x48)
    └── noisy_letter_small.png                 (synthetic noisy "L", 32x32)
```

## To run any of the 10 scripts

1. Put the script in the **same folder** as `offline_modules/*.py` (i.e. copy
   `fs_redrawer.py`, `svg_utils.py`, `epicycle_animation.py`,
   `cft_edge_detector.py` next to it) — every script does
   `from fs_redrawer import FourierEpicycles` or
   `from cft_edge_detector import ContinuousImage, CFT2D, InverseCFT2D`
   exactly as an online submission would.
2. Either use the bundled test assets (paths already point at
   `svgs/heart.svg`, `test_face.png`, `noisy_letter_small.png` — copy the
   `test_assets/` contents next to the script too), or point the `SVG_PATH`
   / `IMAGE_PATH` variable near the top of each script at the real exam's
   provided SVG/image instead.
3. Run normally: `python3 StudentID_PredN_....py`

## Which offline method each script exercises

| # | Property tested | Offline class(es) used |
|---|---|---|
| 1 | Parseval (FS) | `FourierEpicycles` |
| 2 | Time-shift (FS) | `FourierEpicycles` |
| 3 | Convergence / Gibbs | `FourierEpicycles` |
| 4 | Differentiation + constant speed | `FourierEpicycles` |
| 5 | Linearity | `FourierEpicycles` |
| 6 | Parseval (2D CFT) | `ContinuousImage`, `CFT2D` |
| 7 | Low-pass / blur | `ContinuousImage`, `CFT2D`, `InverseCFT2D` |
| 8 | Translation (shift) | `ContinuousImage`, `CFT2D` |
| 9 | Reconstruction sanity + band-pass | `ContinuousImage`, `CFT2D`, `InverseCFT2D` |
| 10 | CFT2D-based denoising | `ContinuousImage`, `CFT2D`, `InverseCFT2D` |

## Honest caveats already documented inside the scripts

- **#6 (2D Parseval)** and **#9 (reconstruction sanity check)**: on a
  *real* sharp-edged image, expect a few-percent relative error, not
  near-zero — this is genuine trapezoidal-quadrature discretization error on
  discontinuities, not a bug. Both scripts include a smooth "control" signal
  alongside the real image to prove the underlying math is correct.
- **#8 (translation property)**: a hard-edged test image shifted inside a
  finite window picks up boundary artifacts that inflate the phase error.
  The script includes a clean synthetic Gaussian-blob control that confirms
  the property holds almost exactly (errors ~1e-6–1e-8) when there's no
  edge-of-window effect.
- **#10 (denoising)**: keep exam images small — the offline's CFT2D is
  O(N³) via `np.trapezoid`, not FFT, so anything much bigger than ~64×64
  will be too slow for a 30–40 minute online.

All ten were run against the bundled synthetic assets before delivery — see
the numeric print statements inside each script for the actual verified
MSE/error values.
