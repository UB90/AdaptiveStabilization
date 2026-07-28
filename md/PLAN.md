# Adaptive Stabilization Techniques — Implementation Plan

## Goal

Build a complete experimental pipeline that:

1. Generates **3 synthetic datasets** with controlled condition numbers (well-conditioned, moderate, severe)
2. Trains **5 optimization methods** on each dataset
3. Repeats each training **5 times** (different seeds) for statistical robustness
4. Saves all training histories
5. Produces a **single comprehensive comparison plot** for research paper use

## Project Structure

```
Adaptive Stabilization Techniques/
├── readme.md                  (existing)
├── data_generator.py          [NEW] — SVD-based synthetic data generation
├── models.py                  [NEW] — Linear + shallow MLP model definitions
├── optimizers.py              [NEW] — Training loops for all 5 methods
├── train.py                   [NEW] — Main training orchestrator
├── plot_results.py            [NEW] — Comprehensive comparison plotting
├── config.py                  [NEW] — All hyperparameters and experiment config
├── results/                   [NEW] — Saved training histories (JSON/pickle)
└── plots/                     [NEW] — Generated comparison figures
```

---

## Proposed Changes

### Configuration — `config.py`

Centralized experiment configuration:

- **3 Datasets**:
  - Dataset 1: κ = 10 (well-conditioned, control)
  - Dataset 2: κ = 10⁴ (moderate ill-conditioning)
  - Dataset 3: κ = 10⁷ (severe, near float32 breakdown)
- Each dataset: n=1000 samples, d=50 features
- **5 Methods**: Vanilla SGD, Momentum SGD, Adam, Ridge L2 (SGD), BatchNorm MLP
- **5 Seeds per method-dataset pair** → 75 total training runs
- Training: 2000 epochs per run
- All hyperparameters tuned per method

---

### Data Generation — `data_generator.py`

Following README Section 2 (SVD-based construction):

1. Build X from its SVD directly for exact κ control
2. Singular values log-spaced from σ_max to σ_min = σ_max/√κ
3. Random orthogonal U, V matrices via QR of Gaussian matrices
4. Fixed true θ* (same across all datasets)
5. y = Xθ* + ε (small noise for realism)
6. Sanity-check: verify actual κ(X^TX) matches target
7. Generate both scale-based and correlation-based variants

---

### Models — `models.py`

Two model classes (PyTorch):

- **LinearModel**: Simple linear regression (for SGD, Momentum, Adam, Ridge)
- **MLPWithBatchNorm**: One hidden layer with BatchNorm (for the BatchNorm method, as README Section 4 explains BN on pure linear is just input standardization)

---

### Training — `optimizers.py`

5 training methods, each with its own function:

1. **Vanilla SGD**: Plain gradient descent, η tuned per κ
2. **Momentum SGD**: β=0.9, testing √κ vs κ scaling
3. **Adam**: lr=0.001, β1=0.9, β2=0.999 (diagonal preconditioner)
4. **Ridge/L2**: SGD with weight_decay (L2 penalty)
5. **BatchNorm MLP**: Adam on the MLP with BatchNorm layer

Each function returns a history dict with:

- Loss per epoch
- Gradient norm per epoch
- Parameter error ‖θ_t - θ*‖ per epoch

---

### Orchestrator — `train.py`

Main script that:

1. Generates all 3 datasets
2. For each dataset × method × seed: runs training, collects history
3. Saves all results to `results/` as pickle files
4. Prints progress and summary statistics

---

### Plotting — `plot_results.py`

Single comprehensive figure with:

- **Grid layout**: 3 rows (one per dataset/κ) × 3 columns (loss, gradient norm, parameter error)
- Each subplot: 5 methods overlaid, mean ± std band across 5 seeds
- Log-scale y-axis for loss
- Color-coded by method with clear legend
- Title showing κ value for each row
- Publication-quality styling (large fonts, tight layout)

---

## Verification Plan

### Automated Tests

- Run `python config.py` to verify configuration loads
- Run `python data_generator.py` to verify datasets generate with correct κ
- Run `python train.py` to execute all 75 training runs
- Run `python plot_results.py` to generate the comparison figure

### Manual Verification

- Check that κ sanity checks pass for all 3 datasets
- Verify training loss decreases for well-conditioned case
- Verify the plot is generated and visually shows the expected patterns (Vanilla SGD struggles at high κ, adaptive methods help)




# Issue in the above Implementation Plan

```
Plan Assessment
├── SOUND
│   ├── SVD construction: σ_min=σ_max/√κ ⇒ κ(X)=√κ_target, κ(XᵀX)=κ(X)²=κ_target ✓
│   ├── QR-of-Gaussian → Haar-orthogonal U,V ✓
│   └── Run count: 3 datasets × 5 methods × 5 seeds = 75 ✓
├── SELF-CONTRADICTS
│   └── "scale-based AND correlation-based variants" (data_generator.py)
│        vs "3 datasets / 75 runs" (everywhere else)
├── CONFOUNDS (invalidate the comparison)
│   ├── κ=10⁷ target vs float32 machine epsilon
│   ├── Momentum β=0.9 fixed vs stated "√κ vs κ scaling" goal
│   ├── BatchNorm-MLP vs "parameter error to θ*" metric
│   └── Batch size unspecified (full-batch vs stochastic)
└── UNDERSPECIFIED
    ├── Data-seed vs train-seed scope
    ├── MLP hidden width
    └── 2000 epochs not justified across 3-decade κ range
```

- **Contradiction — dataset count.** "Generate both scale-based and correlation-based variants" (data_generator.py) implies 2 variants × 3 κ levels = 6 datasets → 6×5×5=150 runs. This contradicts "3 synthetic datasets" and "75 total training runs" stated in Goal/config/train.py. Resolve before coding: pick one variant, or explicitly double the run count.
- **κ=10⁷ vs float32.** float32 machine epsilon ≈1.19×10⁻⁷, so 1/ε≈8.4×10⁶≈10⁷. At κ(XᵀX)=10⁷, gradient components along the small-eigenvalue direction are at the edge of representable precision — you cannot tell "optimizer struggles with conditioning" apart from "float32 rounding noise dominates." Must state explicitly: either force float64 for the severe dataset (isolates algorithmic effect) or keep float32 and reframe Dataset 3 as testing numerical breakdown, not just optimizer sensitivity.
- **Momentum β fixed at 0.9.** Optimal heavy-ball momentum for condition number κ is β* = ((√κ−1)/(√κ+1))². At κ=10: β*≈0.27. At κ=10⁴: β*≈0.96. At κ=10⁷: β*≈0.999. A fixed β=0.9 is far too high for Dataset 1 (oscillation/overshoot risk) and too low for Dataset 3 (acceleration not realized). This directly undermines the stated goal of demonstrating √κ vs κ convergence scaling — β must be a function of κ, or drop that claim from the goal.
- **BatchNorm-MLP metric mismatch.** "Parameter error ‖θ_t−θ*‖" requires a well-defined θ in the same space as the other 4 linear methods. An MLP with a hidden layer has no unique linear θ (nonlinearity + hidden-unit permutation symmetry), so this metric is undefined for that method as specified. Fix: replace with "BatchNorm-on-Linear" (BN layer directly before a single linear output, no hidden layer) — this keeps θ comparable and directly tests the README's own claim that BN on pure-linear reduces to input standardization. Otherwise, drop the parameter-error column for that one method in the plot grid.
- **Batch size absent.** Not specified anywhere (config.py, optimizers.py, models.py). Conditioning experiments require full-batch gradient descent to isolate curvature effects from stochastic gradient noise — with n=1000, d=50 this is computationally trivial. State full-batch explicitly, or the 5-method comparison is confounded by an uncontrolled stochasticity variable.
- **Seed scope ambiguous.** "5 seeds per method-dataset pair" doesn't say whether the seed regenerates X,y or only controls init/shuffling. If (X,y) regenerate per seed, the std bands mix data-generation variance with optimizer variance — two different things. Fix: one fixed data-generation seed per dataset (X,y,ε constant across all 5 method-seeds), seed varies only model init.
- **Minor gaps.** MLP hidden width unstated in config. 2000-epoch budget is fixed across a 10⁶-fold κ range — vanilla SGD at κ=10⁷ may not converge in-budget while κ=10 converges in the first few hundred epochs; either confirm this asymmetry is the intended demonstration or use κ-dependent epoch counts.
