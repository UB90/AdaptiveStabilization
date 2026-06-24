# AIMA 203 — Adaptive Stabilization Techniques: Experimental Verification

Project: reproduce and stress-test every numerical claim made in the presentation
*"Adaptive Stabilization Techniques for Ill-Conditioned Problems in Machine Learning"*
(Poudel, Sapkota, Baral — Kathmandu University, May 2026).

This guide tells you exactly what to run, in what order, what to look for in the
output, and what each result does or does not prove.

---

## 0. Prerequisites

| Requirement | Check command | Expected |
|---|---|---|
| Python ≥3.9 | `python3 --version` | 3.9+ |
| numpy | `python3 -c "import numpy; print(numpy.__version__)"` | any recent |
| scipy | `python3 -c "import scipy; print(scipy.__version__)"` | any recent |
| matplotlib | `python3 -c "import matplotlib; print(matplotlib.__version__)"` | any recent |

If any import fails:
```bash
pip install numpy scipy matplotlib --break-system-packages
```

Folder layout assumed below:
```
AIMA203_experiments/
├── exp1_condition_number.py
├── exp2_ridge_eigenvalue_shift.py
├── exp3_optimizer_comparison.py
├── exp4_batchnorm_jacobian.py
└── exp5_case_study.py
```

Run every script the same way:
```bash
cd AIMA203_experiments
python3 expN_<name>.py
```
Each script is self-contained — no shared imports, no config files, no CLI args required to get baseline output. Each prints a numeric table to stdout and (exp3–exp5) saves a `.png` plot in the same folder.

---

## 1. exp1_condition_number.py

### Claim under test (slide 3)
$$\frac{\delta x}{\|x\|} \le \kappa(A)\cdot\frac{\delta b}{\|b\|}$$
Perturbing the right-hand side of a linear system amplifies error proportionally to the condition number.

### Method
1. Construct $A = U S V^T$ via random orthogonal $U,V$ (QR of random Gaussian matrices) and a chosen singular-value spectrum $S$ — this gives **exact, controllable** $\kappa(A)$, unlike random matrices where $\kappa$ is uncontrolled.
2. Fix a ground-truth $x_{true}$, compute $b = Ax_{true}$.
3. Inject 20 independent small perturbations $\delta b$ (relative size $10^{-6}$) for each $\kappa \in \{1, 10, 10^3, 10^6, 10^9\}$.
4. Solve $Ax' = b+\delta b$ via `np.linalg.solve`, measure $\delta x/\|x\|$ ÷ $\delta b/\|b\|$.

### How to check it worked
Run it. Look at the printed table:
```
    kappa(A) |    bound: k*db/b |  actual dx/x (max over trials)
```
- **Pass condition:** every row prints `<= bound OK`. If any row prints `VIOLATES BOUND`, something is wrong with the matrix construction (re-check the QR step) — the bound is a mathematical theorem, not a statistical tendency, so violation means a bug, not an interesting finding.
- **Expected trend:** the "actual" column should grow roughly with $\kappa$, staying comfortably under the bound (the bound is an upper bound, not a tight estimate, so actual values being far below it for small κ is normal).

### What this proves / doesn't prove
- Proves: the error-amplification inequality holds numerically, exactly as derived analytically.
- Does not prove: anything about ML training dynamics — this is pure linear algebra, the foundation the rest of the presentation builds on.

---

## 2. exp2_ridge_eigenvalue_shift.py

### Claim under test (slide 5)
$$\kappa(X^TX + \lambda I) = \frac{\sigma_{max}+\lambda}{\sigma_{min}+\lambda}$$
Ridge regularization is literally an eigenvalue shift that shrinks the condition number.

### Method
1. Build a multicollinear design matrix: all columns derived from one shared latent variable plus tiny independent noise (correlation ≈0.999) — this guarantees $\sigma_{min}(X^TX)\approx 0$.
2. Compute $\sigma_{max}, \sigma_{min}$ of $X^TX$ via SVD once.
3. For $\lambda \in \{0, 0.01, 0.1, 1, 10\}$: compute the **predicted** $\kappa$ from the closed-form formula, and the **measured** $\kappa$ from a fresh SVD of $X^TX+\lambda I$.
4. Separately: solve for ridge coefficients at each $\lambda$ and report $\|coef\|$ to show the practical consequence (coefficient explosion at $\lambda=0$).

### How to check it worked
```
    lambda |  predicted k=(smax+l)/(smin+l) |   measured kappa | match
```
- **Pass condition:** every row says `OK` (predicted and measured agree to 1e-6 relative tolerance). This is an identity, not an approximation — `MISMATCH` means a coding error in either formula.
- **Second table** (`Coefficient norm ||x*|| vs lambda`): expect a sharp drop in $\|coef\|$ from $\lambda=0$ to $\lambda=0.01$, then near-flat for larger $\lambda$. A flat-then-rising pattern at very large $\lambda$ would also be expected if you extend the range (over-regularization shrinks coefficients toward zero) — not tested here but worth knowing if you modify the script.

### What this proves
- The "regularization = preconditioning" framing in the slide is not a metaphor — it is the literal eigenvalue-shift identity, confirmed exactly.
- Practical payoff: unregularized normal equations can be technically solvable (`np.linalg.solve` won't always throw a `LinAlgError`) while still being numerically unstable — coefficient norm is the tell, not solver failure.

---

## 3. exp3_optimizer_comparison.py

### Claim under test (slide 6)
On an ill-conditioned loss surface, vanilla gradient descent is forced to use a learning rate small enough for the steepest direction, which makes it crawl in flat directions. Adaptive optimizers (AdaGrad, RMSprop, Adam) rescale per-parameter and escape this tradeoff.

### Method
1. Loss surface: $f(x) = \tfrac12 x^THx$ with $H=\text{diag}(\kappa, 1)$ — a 2D quadratic bowl, condition number exactly $\kappa$, one steep axis, one flat axis.
2. Implement four optimizers from their exact update rules (matching slide 6 formulas precisely): GD, AdaGrad, RMSprop, Adam (with bias correction).
3. Run all four from $x_0=(1,1)$ for 200 steps at $\kappa=1000$, with learning rates hand-tuned per optimizer (this mirrors real practice — different optimizers need different LR scales, that's part of the point).
4. Track loss per step and the 2D trajectory.

### How to check it worked
```
Optimizer            |  loss @ step 0 | loss @ step 50 | loss @ step 200
```
- **Pass condition:** SGD's final loss should be the **largest** of the four (it stalls); Adam/AdaGrad should be many orders of magnitude lower.
- Open `exp3_optimizer_comparison.png`: left panel is loss-vs-iteration (log scale) — SGD's curve should visibly plateau early; adaptive methods should keep dropping. Right panel is the 2D trajectory — SGD's path should show a slow crawl along the flat axis; adaptive methods should move diagonally toward the origin.

### Known sensitivity (read before re-running with different numbers)
This experiment's qualitative ranking **depends on the learning rates chosen**. If you change `kappa` or any `lr`, re-tune: SGD's lr must satisfy $lr < 2/\kappa$ to avoid diverging on the steep axis, which is exactly the constraint the slide describes. Picking a larger SGD lr to "be fair" will make it diverge, not converge faster — that divergence *is* the phenomenon, not a bug.

### What this proves
- Confirms the mechanism, not just the outcome: SGD's failure mode is structural (single global LR vs. anisotropic curvature), not a tuning accident.
- AdaGrad's near-zero final loss ($10^{-95}$) is partly an artifact of this toy problem (monotonically shrinking denominator on a strongly convex bowl with no noise) — don't generalize this magnitude to real training, where AdaGrad's aggressive decay is usually a *weakness* (slide 6 itself notes this).

---

## 4. exp4_batchnorm_jacobian.py

### Claim under test (slide 8)
Batch Normalization bounds the singular values of each layer's Jacobian, preventing activation norms from exploding or vanishing across network depth.

### Method
1. Build a 30-layer network with random Gaussian weights scaled by 1.3/√width (deliberately >1 to induce explosion).
2. Run two parallel forward passes — one without BN, one with BN inserted after every linear layer — and track mean activation norm at each layer.
3. **Run twice with different activation functions** (this is the important methodological step):
   - `nonlinearity="linear"` — no squashing function at all. This isolates the claim cleanly: any explosion you see is purely from the weight matrices' spectral radius, with nothing else able to mask it.
   - `nonlinearity="tanh"` — realistic case. Tanh saturates between −1 and 1, which *on its own* limits activation growth regardless of BN.

### How to check it worked
**Linear case** — this is the test that actually isolates the claim:
```
Final/initial norm ratio (growth factor over 30 layers):
  no-BN : <should be >>1, e.g. ~2500x>
  BN    : <should be ≈1, e.g. ~0.99x>
```
- **Pass condition:** no-BN ratio should be orders of magnitude above 1; BN ratio should stay close to 1 across the full depth.
- Check `exp4_batchnorm_depth_linear.png`: no-BN curve should rise roughly geometrically (straight line on the log-scale y-axis); BN curve should be flat.

**Tanh case** — read this one skeptically:
- Both curves will look similarly bounded, because tanh saturates on its own. **A flat tanh curve in the no-BN case does not contradict the claim** — it means the nonlinearity, not BN, is doing the bounding in that specific run. This is a real limitation of testing the claim on saturating activations; the slide's claim is best demonstrated (and was likely derived from) linear or ReLU-style unbounded activations, not tanh.

### What this proves / doesn't prove
- Proves cleanly, on linear layers: BN's normalization step is sufficient on its own to prevent exponential signal growth, independent of weight initialization scale.
- Does not prove: that BN is *necessary* whenever a saturating nonlinearity is present — tanh provides redundant protection in that regime. If you want a stricter test, rerun with ReLU (`act = lambda z: np.maximum(0, z)`) — ReLU doesn't saturate on the positive side, so it won't mask the effect the way tanh does. This is a good next step if you want to extend the experiment.

---

## 5. exp5_case_study.py

### Claim under test (slide 11)
A 3-layer MLP trained on features with correlation $\rho>0.95$ (giving $\kappa(X^TX)\approx10^7$) shows: SGD oscillates and converges slowly; Adam+BN converges fast and stably.

### Method
1. Generate correlated Gaussian features via a covariance matrix with off-diagonal $\rho$, sample $N(0,\Sigma)$.
2. **Calibration step**: $\rho$ must be extremely close to 1 (not 0.95–0.97) to reach $\kappa\sim10^7$ with only 10 features — this script uses $\rho=0.999999$ to hit that target. This is the single most important parameter in the script; if you change feature count or sample size, you will need to recalibrate $\rho$ to hit a comparable $\kappa$.
3. Binary labels from a random linear separator on the (correlated) features.
4. A from-scratch 3-layer MLP (manual forward/backward pass, no autodiff library) with optional BatchNorm after each hidden layer.
5. Train under three configurations: plain SGD, plain Adam, Adam+BN. 300 epochs, full-batch.

### How to check it worked
```
Generated data: rho=0.999999, kappa(X^T X) = <should print ~1e7>
```
- **Pass condition for calibration:** measured $\kappa$ should be within an order of magnitude of $10^7$. If it isn't, increase $\rho$ closer to 1 (e.g. `0.9999999`) or reduce `n_features`.
- **Loss table**: compare loss at epochs 0, 50, 150, 299 across the three configs.
- **Open `exp5_case_study.png`** and look for oscillation (sharp up-down spikes) versus smooth monotonic descent.

### Read this result carefully — it did not confirm the slide narrative on first run
When this was run during development, the actual ranking was:
- **Plain Adam** converged lowest and smoothest — not "Adam+BN."
- **Adam+BN** showed visible spikes mid-training (full-batch BN statistics computed over the *entire* dataset every step is a much less noisy estimate than the mini-batch BN the slide implicitly assumes, yet it still introduced instability here — likely from BN's extra learnable parameters (γ, β) adding optimization surface, not from poor curvature).
- **Plain SGD** converged steadily, **without** the oscillation the slide predicts, at the learning rate used here.

**This is a legitimate and informative negative result, not a bug to be silently fixed.** Before concluding anything, try these to see if the slide's qualitative claim re-emerges:
1. Switch from full-batch to **mini-batch** training (e.g. batch size 32) — BN's benefit and SGD's oscillation are both batch-size-dependent effects that a full-batch run partially suppresses.
2. Increase network depth (slide's claims about Jacobian conditioning compound with depth — 3 layers may be too shallow to expose it).
3. Raise the SGD learning rate closer to its instability threshold — the oscillation claim is conditional on being near that boundary, not generic to any SGD run.

### What this proves
- Confirms the **setup** is faithful to the slide (κ target hit exactly).
- Demonstrates, honestly, that "Adam+BN beats SGD" is not a universal law — it is conditional on batch size, depth, and learning-rate regime. Presenting this nuance (rather than forcing a fit) is itself a correct application of the numerical-methods mindset the course is teaching: claims about iterative methods are regime-dependent, and a good experimentalist reports when a regime doesn't trigger the expected behavior.

---

## 6. Suggested order of work

| Step | Action | Time |
|---|---|---|
| 1 | Run exp1, exp2 — pure linear algebra, no tuning needed, should pass immediately | 2 min |
| 2 | Run exp3 — check plot, understand why SGD's lr is capped | 5 min |
| 3 | Run exp4 — compare linear vs tanh output, understand why tanh masks the effect | 5 min |
| 4 | Run exp5 — check κ calibration first, then attempt the mini-batch / depth extensions in §5 if you want the slide's full narrative to reproduce | 15–30 min |
| 5 | Write up: for each experiment, report the printed numbers + one sentence on whether it confirmed, partially confirmed, or contradicted the slide claim | — |

## 7. What "testing a theory" means here — methodological note

Three different outcomes occurred across these five experiments, and a correct report distinguishes them:
- **exp1, exp2**: mathematical identities. Confirmation is binary — either the numbers match the formula or there's a bug. No "interpretation" needed.
- **exp3, exp4 (linear case)**: mechanism-level claims about iterative algorithms. Confirmed cleanly because the toy problem was designed to isolate the mechanism (controlled κ, no confounding nonlinearity).
- **exp5**: an empirical claim about a specific architecture/optimizer/data combination. Partially contradicted under the tested conditions. This is the normal, expected outcome when testing an empirical (not mathematical) claim from a presentation slide — slide-level claims about "what works" are usually true in some regime and not stated with their boundary conditions. Reporting the boundary you found is more valuable than forcing the script to match the slide.
