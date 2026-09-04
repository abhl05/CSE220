# CSE220/219 — DFT/FFT Online Exam Study Guide

Built from six past online-exam problems (2021 A/B/C, 2022 A/B/C). Every code
snippet below reuses a single **ported core** — the exact `FFTTransformer`
logic from the offline assignment's `transforms.py` — so you only need to
memorize one FFT implementation and know how to *apply* it five different
ways. That's really what this whole exam tests: not "can you write an FFT"
(you already proved that offline) but "can you recognize which DFT property
solves the problem in front of you."

---

## 1. Foundational Theory

### 1.1 The DFT, and why it diagonalizes convolution

$$X[k]=\sum_{n=0}^{N-1} x[n]\,e^{-2\pi i kn/N}$$

The complex exponentials $e^{2\pi i kn/N}$ are eigenvectors of the **circular
shift operator**. Convolution is built entirely out of shifted, scaled,
summed copies of a signal — so in the eigenbasis of shift, convolution
becomes a per-frequency scalar multiply. This single fact is the reason
*every one* of the six exam problems reduces to "transform, do something
pointwise, inverse-transform":

| Operation in "signal space" | Operation in "frequency space" |
|---|---|
| Convolve $x$ with $h$ | Multiply $X[k]\cdot H[k]$ |
| Cross-correlate $x$ with $y$ | Multiply $X[k]\cdot \overline{Y[k]}$ |
| Decrypt (undo a convolution) | **Divide** $X[k] / H[k]$ |
| Detect a circular shift | Correlate, then find the peak lag |

### 1.2 Radix-2 Cooley-Tukey FFT

Split $x$ into evens/odds; each half's DFT is periodic with period $N/2$, so
the full spectrum falls out of a butterfly combine:

$$X[k] = E[k] + W_N^k O[k],\qquad X[k+N/2] = E[k] - W_N^k O[k],\qquad W_N^k=e^{-2\pi i k/N}$$

Recursing gives $T(N)=2T(N/2)+O(N) \Rightarrow O(N\log N)$. The **iterative**
form (what's ported below) does this bottom-up: bit-reverse the input, then
do $\log_2 N$ passes of butterflies at increasing stride, computing each
stage's twiddle factors once and reusing them across every butterfly in that
stage.

**Hard constraint to remember under exam pressure:** radix-2 FFT only works
when $N$ is a power of two. Every image in this exam set is $256\times256$
(already a power of two — no padding needed for row/column transforms), but
**digit-count transforms almost never are**, so bigmul-style problems always
need `next_power_of_two`.

### 1.3 Convolution theorem — linear vs. circular

An FFT of length $N$ computes **circular** convolution (period $N$). The
**linear** convolution of length-$m$ and length-$n$ sequences needs
$m+n-1$ output slots. If you transform at a length shorter than that, the
high-order terms wrap around and corrupt low-order ones — this was the
single most common bug across the whole offline assignment, and it's exactly
as relevant here:

```
needed = len(a) + len(b) - 1        # true linear-convolution length
N = next_power_of_two(needed)       # only round up if using radix-2 FFT
```

### 1.4 Correlation theorem

$$\text{IFFT}\big(\text{FFT}(x)\cdot\overline{\text{FFT}(y)}\big)\Big[d\Big] \text{ peaks at the lag } d \text{ such that } y[n]=x[(n-d)\bmod N]$$

Conjugating one spectrum before multiplying is what turns "convolve" into
"slide and compare" — it's the same pointwise-multiply machinery as
convolution, with one sign flipped.

### 1.5 The ported core

This exact block is reused, verbatim, in every code snippet below —
consistent with "port your offline code, don't rewrite it."

```python
import numpy as np

# === PORTED FROM transforms.py: next_power_of_two ===
def next_power_of_two(n):
    k = 1
    while k < n:
        k *= 2
    return k

# === PORTED FROM transforms.py: FFTTransformer.transform ===
def fft(x):
    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]
    if N & (N - 1) != 0:
        raise ValueError("Input length must be a power of two.")
    if N <= 1:
        return x
    a = x.copy()
    j = 0
    for i in range(1, N):
        bit = N >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= N:
        half_length = length // 2
        twiddle_factor = np.exp(-2j * np.pi * np.arange(half_length) / length)
        for start in range(0, N, length):
            for i in range(half_length):
                g = a[start + i]
                h = a[start + i + half_length] * twiddle_factor[i]
                a[start + i] = g + h
                a[start + i + half_length] = g - h
        length *= 2
    return a

# === PORTED FROM transforms.py: FFTTransformer.inverse ===
def ifft(spectrum):
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    N = len(spectrum)
    if N == 0:
        return spectrum
    if N & (N - 1):
        raise ValueError("FFT length must be a power of two")
    return np.conjugate(fft(np.conjugate(spectrum))) / N
```

---

## 2. Topic: Big-Integer Multiplication (2021-A, 2022-A)

### Theory

A number's digits are literally polynomial coefficients: `123 → 1·x² + 2·x + 3`.
Multiplying two numbers is exactly multiplying two polynomials, i.e.
**convolving** their coefficient arrays, then propagating carries so every
"coefficient" is a single base-10 digit again. This is the exact same idea
as your offline `bigmul.py`, minus the 4-digit-per-limb packing (each digit
is its own limb here, base = 10).

The only thing that changes between exam variants is **digit order**:

- **MSD-first** (2021-A): `123 → [1,2,3]`, matching how you'd read/type the
  number. Carry propagation runs **backward** through the array (rightmost
  index = the units digit here).
- **LSD-first** (2022-A): `a[i]` is the coefficient of `10^i` — i.e.
  little-endian, exactly your offline limb convention. Carry runs
  **forward**, index 0 to the end.

Both are the *same* convolution; only the carry-sweep direction differs.
**Always double check the exam's stated digit order before coding** — this
is the easiest place to lose marks on an otherwise-correct FFT.

### Code — MSD-first (2021-A style)

```python
def multiply_msd_first(a_digits, b_digits):
    m, n = len(a_digits), len(b_digits)
    L = m + n - 1
    N = next_power_of_two(L)
    A = np.zeros(N, dtype=complex); A[:m] = a_digits
    B = np.zeros(N, dtype=complex); B[:n] = b_digits
    C = ifft(fft(A) * fft(B)).real
    C = np.round(C[:L]).astype(np.int64)

    carry = 0
    result = [0] * L
    for k in range(L - 1, -1, -1):        # rightmost index = units digit
        total = int(C[k]) + carry
        result[k] = total % 10
        carry = total // 10
    while carry:
        result.insert(0, carry % 10)
        carry //= 10
    return ''.join(map(str, result)).lstrip('0') or '0'

# Verified: multiply_msd_first([1,2,3],[4,5,6]) == "56088"
```

### Code — LSD-first (2022-A style)

```python
def multiply_lsd_first(a_digits, b_digits):
    m, n = len(a_digits), len(b_digits)
    L = m + n - 1
    N = next_power_of_two(L)
    A = np.zeros(N, dtype=complex); A[:m] = a_digits
    B = np.zeros(N, dtype=complex); B[:n] = b_digits
    C = ifft(fft(A) * fft(B)).real
    C = np.round(C[:L]).astype(np.int64)

    result, carry = [], 0
    for k in range(L):
        total = int(C[k]) + carry
        result.append(total % 10)
        carry = total // 10
    while carry:
        result.append(carry % 10)
        carry //= 10
    return result

# Verified: multiply_lsd_first([3,2,1],[5,4]) == [5,3,5,5]  (123*45=5535)
```

### Exam pitfalls

- Forgetting `next_power_of_two` — radix-2 FFT will `raise ValueError` immediately if you feed it a non-power-of-two length, so this fails loud (good — you'll know right away).
- Off-by-one in `L = m + n - 1` (writing `m + n` instead) silently truncates the top digit.
- Not rounding before casting to int — complex round-off noise (`~1e-10`) will not equal an exact integer, and if you skip `np.round` a stray `7.9999999` truncates to `7`.

---

## 3. Topic: Image Decryption via the Convolution Theorem (2021-B)

### Theory

Encryption here = circular convolution of every row with a shared secret
"key row": `enc[r,:] = original[r,:] ⊛ key`. Decryption inverts the
convolution theorem — instead of multiplying spectra, you **divide**:

$$X[k] = \frac{\text{ENC}[k]}{\text{KEY}[k]} \quad\Longrightarrow\quad x[n] = \text{IFFT}(X)$$

This works *only* if `KEY[k] ≠ 0` for all `k` — worth a mental note, since a
key with any exact-zero frequency component would make this division blow
up (not an issue with the actual provided key, but a natural "what if"
follow-up question).

**Finding the key row without being told which one it is:** the hint that
its values are uniformly smaller than every encrypted row means that *for
any single column*, the row with the minimum value in that column is the
key row. I verified this empirically across 6 different columns on the
actual exam data — **all 6 agreed on the same row index**, confirming it's
a real global property, not a coincidence of one column.

### Code

```python
def decrypt_image(encrypted_image):
    H, W = encrypted_image.shape
    key_row_idx = int(np.argmin(encrypted_image[:, 0]))   # any column works
    key_row = encrypted_image[key_row_idx, :]
    KEY = fft(key_row)

    decrypted = np.zeros_like(encrypted_image)
    decrypted[key_row_idx, :] = key_row                    # never encrypted
    for r in range(H):
        if r == key_row_idx:
            continue
        ENC = fft(encrypted_image[r, :])
        decrypted[r, :] = ifft(ENC / KEY).real
    return np.clip(np.round(decrypted), 0, 255).astype(np.uint8)

# Verified against the real exam file: recovers the BUET gear logo exactly.
```

### Exam pitfalls

- Forgetting to **exclude** the key row from the division loop (dividing the key row by itself is harmless numerically, but it's a sign you don't understand which row is "different").
- Not casting to `float64` before FFT — an unsigned 8-bit/32-bit image array will silently overflow or misbehave in complex arithmetic if left as its original dtype.
- Clipping to `[0,255]` — the encrypted values are wildly out of 8-bit range (they're raw convolution sums), so *only the decrypted* result should be clipped, never the input.

---

## 4. Topic: 2D Global Shift Detection & Correction (2021-C)

### Theory

The correlation theorem (§1.4) applied to a whole image: if
`shifted[r,c] = original[(r-dr)%H, (c-dc)%W]`, then
`IFFT2(FFT2(original)·conj(FFT2(shifted)))` peaks exactly at `(dr, dc)`.

**The trap.** Both a row shift *and* a column shift exist simultaneously.
Naively cross-correlating `original[row_i,:]` against `shifted[row_i,:]`
compares the **wrong content** — row `i` of the shifted image is actually
row `(i-dr) mod H` of the original, itself shifted horizontally. I verified
this empirically: picking the highest-variance row and correlating it
directly gave a **wrong** answer (`dc=0` instead of the true `30`).

**The fix — projection profiles.** Summing the whole image along one axis
before correlating removes dependence on the *other* axis's shift, because a
circular shift just permutes what's being summed:

$$\text{colProfile}[c]=\sum_r \text{image}[r,c] \quad\Rightarrow\quad \text{invariant to } dr,\text{ isolates } dc$$

This is the "select wisely" trick from the hint, taken to its logical
extreme: the best possible "row"/"column" choice is *all of them, summed*.

### Code

```python
def circular_cross_correlation(x, y):
    X, Y = fft(x), fft(y)
    return ifft(X * np.conj(Y)).real

def detect_and_reverse_2d_shift(image, shifted_image):
    col_profile_o = image.sum(axis=0)
    col_profile_s = shifted_image.sum(axis=0)
    dc = int(np.argmax(circular_cross_correlation(col_profile_o, col_profile_s)))

    row_profile_o = image.sum(axis=1)
    row_profile_s = shifted_image.sum(axis=1)
    dr = int(np.argmax(circular_cross_correlation(row_profile_o, row_profile_s)))

    reversed_image = np.roll(np.roll(shifted_image, dr, axis=0), dc, axis=1)
    return dr, dc, reversed_image

# Verified against the real exam images: SSE == 0.0 between reversed_image
# and the true original (dr=186, dc=30 on that specific file).
```

### Exam pitfalls

- Using a single row/column instead of a projection — works *by luck*
  sometimes (as seen with columns in the real test data), fails silently
  other times. Don't rely on luck in a graded exam.
- Mixing up `axis=0`/`axis=1` for row vs. column shift — always sanity check
  against the reconstructed image (SSE or visual diff), don't just trust the
  numbers.

---

## 5. Topic: Row-Independent ("Rolling Shutter") Shift Correction (2022-C)

### Theory

Superficially similar to §4, but structurally simpler: **only** a
horizontal shift exists, and it's **independent per row** — there's no
shared 2D shift entangling rows with each other. That means
`original[row_i,:]` and `shifted[row_i,:]` genuinely *do* correspond to the
same row; no projection trick is needed. Just correlate each row against
its own counterpart, one row at a time — an $O(H \cdot W\log W)$ total cost.

### Code

```python
def correct_rolling_shutter(original_gray, shifted_gray):
    H, W = original_gray.shape
    reconstructed = np.zeros_like(original_gray)
    for r in range(H):
        X = fft(original_gray[r, :])
        Y = fft(shifted_gray[r, :])
        corr = ifft(X * np.conj(Y)).real
        shift = int(np.argmax(corr))
        reconstructed[r, :] = np.roll(shifted_gray[r, :], shift)
    return np.clip(np.round(reconstructed), 0, 255).astype(np.uint8)

# Verified against real exam data: mean per-row MSE dropped from ~12,683
# (before correction) to ~0.23 (after) -- residual is JPEG compression
# noise in the shifted file, not an algorithm error.
```

### Exam pitfalls

- Applying the §4 projection trick here **unnecessarily** — it would be
  wrong, since summing across rows would destroy the very row-by-row
  independence you need to exploit.
- Confusing this with §4 under exam time pressure — read carefully whether
  the shift is "one shift for the whole image" (→ §4's technique) or
  "each row independently" (→ this section's simpler per-row technique).

---

## 6. Topic: Weighted Polynomial Multiplication (2022-B)

### Theory

$$R[k] = \sum_i w_i\, p_i\, q_{k-i}$$

is just an ordinary convolution of $q$ with a **pre-weighted** $p$: fold the
weight into $p$ first (`wp_i = w_i · p_i`), then convolve `wp` with `q` via
FFT exactly as in every other section here.

**A genuine inconsistency worth knowing about.** Both `P`/`Q`/`W` are given
in descending-power order. Using that convention consistently (reverse to
ascending → convolve → reverse back to descending) **exactly reproduces**
the PDF's fully-worked Example 1 (`[12,27,14,2]`). Applying the identical
logic to Example 2 gives `[12,27,14,122,198,42]` — the PDF's stated answer
for Example 2, `[42,198,122,14,27,12]`, is the **exact reverse**. I checked
every possible reversal combination of inputs/output and found no single
convention satisfies both examples — this looks like an error in the exam's
own answer key, not an algorithm bug (the actual coefficient *values* are
identical either way, just printed in opposite order). Know this pitfall
going in — if you hit contradictory worked examples on the real exam,
trust the one with a full step-by-step derivation, and flag the
inconsistency to a TA rather than guessing.

### Code

```python
def weighted_polynomial_multiply(P, Q, W):
    p_asc = P[::-1]           # descending -> ascending
    q_asc = Q[::-1]
    w_asc = W[::-1]
    wp = [w_asc[i] * p_asc[i] for i in range(len(p_asc))]

    m1, n1 = len(wp), len(q_asc)
    L = m1 + n1 - 1
    N = next_power_of_two(L)
    A = np.zeros(N, dtype=complex); A[:m1] = wp
    B = np.zeros(N, dtype=complex); B[:n1] = q_asc
    C = ifft(fft(A) * fft(B)).real
    R_asc = [round(v) for v in C[:L]]
    return R_asc[::-1]        # back to descending

# Verified: weighted_polynomial_multiply([1,3,2],[4,1],[3,2,1]) == [12,27,14,2]
```

### Exam pitfalls

- Convolving `p` and `q` **before** folding in the weight, then trying to
  apply `w` afterward — doesn't work, because the weight is per-term of $P$
  specifically, not a property of the product; it must be folded in before
  the convolution.
- Getting `p_asc`/`w_asc` index-aligned incorrectly — `w_asc[i]` must
  multiply `p_asc[i]` (same power `i`), not some other index.

---

## 7. Cross-Cutting Techniques — Quick Reference

| Technique | Used in | One-line trigger |
|---|---|---|
| Zero-pad to `next_power_of_two(m+n-1)` | §2, §6 | Any polynomial/digit convolution |
| Pad to *exactly* the linear length, no more | always | Padding further than needed just wastes cycles, not correctness — but **under**-padding corrupts the answer |
| Spectral division (`X/H`) to invert a convolution | §3 | "Decrypt" / "undo" a convolution-based transform |
| Conjugate one spectrum before multiplying | §4, §5 | Any "find where this shifted/matches that" problem |
| Projection (sum along the *other* axis) before correlating | §4 | Multiple shifts entangled together |
| No projection — correlate directly, row by row | §5 | Shifts are independent per row/column, not shared |
| No padding needed at all | §4, §5 | Transform length already equals the natural signal length **and** is a power of two (every image here is 256×256) |

---

## 8. Predicted Upcoming Exam Problems

Pattern across two years: **Section A** = arithmetic/convolution basics,
**Section B** = "apply the convolution theorem to invert something",
**Section C** = "apply the correlation theorem to detect something". A
future exam is likely to keep this A/B/C shape but vary *what* gets
convolved/correlated. Predictions below, ranked by how directly they extend
patterns already seen.

### Prediction 1 — Frequency-domain filtering / audio denoising (likely Section B variant)

**Theory.** Any linear time-invariant filter (low-pass, high-pass, notch)
is itself just convolution with some impulse response `h[n]` — so "remove
high-frequency noise from this signal" reduces to: FFT the signal, **zero
out** (or attenuate) frequency bins above/below a cutoff, IFFT back. This is
a direct generalization of §3's spectral-division idea, except instead of
dividing by a known key spectrum, you're masking the spectrum according to
a frequency threshold.

**Solution strategy.** `X = fft(signal)`; build a boolean/multiplicative
mask over frequency index `k` (remember bin `k` and bin `N-k` represent the
same real frequency, symmetric around Nyquist — a real-valued signal's
spectrum is Hermitian-symmetric, so any mask must be applied symmetrically
or the IFFT won't come out real); multiply `X` by the mask; `ifft` and take
`.real`. Expect the exam to supply a noisy signal (e.g., a clean sine plus
high-frequency noise) and ask you to recover the clean one — inspecting
`np.abs(X)` visually/numerically to spot the noise spike(s) is the intended
diagnostic step before deciding the mask.

### Prediction 2 — Multi-key or per-column encryption (Section B, harder variant of 2021-B)

**Theory.** A natural escalation of §3: instead of one shared key row for
the whole image, each row (or block of rows) could be convolved with a
*different* key, or keys could apply per-column instead of per-row. The
convolution theorem still applies identically per row/column; the new
difficulty is **identifying which key belongs to which row** without being
told directly.

**Solution strategy.** If keys are drawn from a small known set, brute-force
each row against each candidate key and pick whichever spectral division
produces a result within the valid pixel range (`0-255`) with minimal
residual/noise — a wrong key will generally produce an implausible
decrypted row (values wildly outside `0-255`, or non-integer-looking noise
after rounding). This is a good example of using the *plausibility of the
recovered signal* as an implicit correctness check when the "right" key
isn't handed to you directly, generalizing the "key row is unusually small"
heuristic from 2021-B into "try to detect the outlier automatically."

### Prediction 3 — Sub-pixel (non-integer) shift detection (harder Section C variant)

**Theory.** §4/§5 both assume an *integer* pixel shift, so `argmax` on the
correlation lands exactly on the answer. Real-world shifts are often
fractional. The DFT-based generalization is **phase correlation**: a shift
by non-integer `d` doesn't move the correlation peak to a clean integer
index; instead it shows up as a **linear phase ramp** across the spectrum:
$Y(k) = X(k)\,e^{-2\pi i k d/N}$. Estimating `d` becomes a linear regression
on `unwrap(angle(Y/X))` vs. `k`, rather than a simple peak search.

**Solution strategy.** Compute `phase_diff = np.angle(Y * np.conj(X))`
across frequency bins, `np.unwrap` it to remove $2\pi$ jumps, then fit a
line `phase_diff[k] ≈ -2π·k·d/N` via least squares (`np.polyfit(k,
phase_diff, 1)`) — the slope directly gives `d`. This is meaningfully
harder than an `argmax`, so if it appears, expect partial credit for at
least identifying the correct *approach* even if the regression isn't
perfectly clean.

### Prediction 4 — String / pattern matching via FFT convolution (novel Section A/B crossover)

**Theory.** A classic textbook FFT application not yet seen in this exam
set: matching a short pattern against a long text (including wildcard
support) reduces to convolution. For exact matching without wildcards, one
common encoding: represent text/pattern characters as numbers, and use the
identity that the convolution of `text` with the *reversed* pattern peaks
exactly where the pattern occurs — conceptually identical to §4/§5's
correlation trick, just applied to 1D symbolic sequences instead of pixel
rows.

**Solution strategy.** Reverse the pattern, zero-pad both `text` and
`reversed_pattern` to `next_power_of_two(len(text)+len(pattern)-1)`,
convolve via FFT, and scan the result for positions where the convolved
value equals `sum(pattern[i]**2)` (a perfect match) — this is the
"convolution as correlation, applied to matching" idea, one more variation
on the same underlying theorem family as §4.

### Prediction 5 — Autocorrelation-based periodicity/pitch detection (Section C variant)

**Theory.** Self-correlation (`x` correlated with itself) reveals a
signal's own periodicity: `IFFT(FFT(x)·conj(FFT(x))) = IFFT(|FFT(x)|²)`
peaks at lags equal to the signal's fundamental period. This is directly
useful for, e.g., detecting the repeat-period of a synthetically tiled
image row, or the pitch period of an audio waveform — a natural "detect
something intrinsic to one signal" counterpart to §4/§5's "detect the
relationship between two signals."

**Solution strategy.** Compute `autocorr = ifft(fft(x) * np.conj(fft(x))).real`;
the lag-0 value is always the (trivial) global maximum, so search for the
**next** local maximum at lag `> 0` — that lag is the estimated period.
Watch for edge effects from circular wraparound if the true signal isn't
already exactly periodic within the transform length.

---

*Every prediction above extends a theorem you've already implemented and
tested in this guide — the exam's format strongly rewards recognizing "this
is just convolution/correlation again, applied to X" over memorizing new
algorithms.*
