
This is a genuinely well-posed project — one idea (the condition number of G) explains why five very different-looking fixes all work, and that idea is worth stating up front because it makes everything else click: **every technique on your list is a cheap approximation to the same expensive operation — using G⁻¹ (true curvature / Newton preconditioning) to correct the step direction — without ever forming or inverting G, because forming/inverting G is exactly the operation that breaks down at high κ.** Momentum approximates it temporally (averaging gradient history), Adam approximates it diagonally (per-coordinate second-moment scaling), Ridge approximates it by regularizing the very matrix you'd otherwise invert, and BatchNorm approximates it by reshaping the landscape so a bad κ matters less. Keep that thread in mind — it's a good line for your discussion section, and it's what ties the rest of this together.

## 1. What you should know (theory)

**Condition number, geometrically.** κ(G) = λ_max/λ_min measures the eccentricity of the loss bowl. For quadratic loss L(θ) = ½(θ−θ*)ᵀG(θ−θ*), the eigenvector for λ_max is steep (small steps overshoot and bounce); the eigenvector for λ_min is nearly flat (gradient signal is tiny, progress crawls). Plain GD's stability forces step size η < 2/λ_max; with η pinned that small, progress along λ_min is governed by (1 − 2/κ). One sentence captures the whole mechanism: **the same step size has to serve the steepest and flattest directions at once, and κ measures how badly those demands conflict.**

The classical result for quadratics with an optimally-tuned fixed step size:
||θ_k − θ*|| ≤ ((κ−1)/(κ+1))^k · ||θ_0 − θ*||
So iterations-to-ε scale like O(κ). This is the number your Vanilla SGD control should trace out empirically.

**Why direct inversion breaks.** Two compounding facts:

- Forming G = XᵀX squares the condition number: κ(G) = κ(X)². A merely annoying κ(X)=10⁴ becomes κ(G)=10⁸ the instant you form the Gram matrix — this is exactly why numerical analysts solve least squares via QR/SVD on X directly instead of the "normal equations": forming XᵀX throws away accuracy before you've solved anything.
- For any linear solve Gx=b, perturbation theory gives ||δx||/||x|| ≲ κ(G)·||δb||/||b||. Floating-point rounding behaves like a perturbation of relative size ≈ machine epsilon (≈1.2×10⁻⁷ for float32, ≈2.2×10⁻¹⁶ for float64). Once κ(G) approaches 1/epsilon, relative error in the solved θ approaches 100% — you're reading noise, not a slightly-off answer. That's κ ≈ 10⁷ for float32, κ ≈ 10¹⁵ for float64. These give you concrete numbers to design your κ sweep around.

**Why each method is expected to help — mechanism, not folklore:**

- **Momentum (heavy-ball):** accumulates gradient history, so consistent gradients (flat directions) build speed while alternating-sign gradients (steep, oscillating directions) cancel. Optimally tuned on a quadratic, the rate improves from (κ−1)/(κ+1) to (√κ−1)/(√κ+1) — κ-dependence drops from linear to square-root. **Checking whether your empirical momentum curve actually shows √κ scaling instead of κ scaling is the single most valuable analysis in this whole project.** Caveat: this clean rate is a quadratic-loss result (Polyak, 1964); for general non-quadratic strongly-convex losses, heavy-ball doesn't carry the same guarantee and has known non-convergent pathological cases. Your setup (linear regression → exactly quadratic loss) gets the clean theory for free — worth stating explicitly as a reason the testbed is the right one.
- **Ridge/L2:** literally shifts every eigenvalue of G by λ, since (G+λI)v = (λᵢ+λ)v whenever Gv=λᵢv. So κ_ridge = (λ_max+λ)/(λ_min+λ) → 1 as λ→∞. This is Tikhonov regularization / Hoerl & Kennard's ridge regression (1970) — originally a *numerical* fix for ill-conditioned XᵀX, with the statistical bias-variance story riding along. The cost: you're now solving a biased problem, so conditioning is traded for bias as λ grows.
- **Adam:** keeps a running per-coordinate second-moment estimate and divides the step by its square root. This is an approximate *diagonal* preconditioner — the cheap, gradient-statistics version of what a real Newton step would do exactly. Important, testable limitation: **if the ill-conditioning comes from correlated/collinear features (off-diagonal structure, eigenvectors not aligned with coordinate axes), Adam's diagonal preconditioner can't fully fix it** — it shines mainly when bad conditioning is closer to axis-aligned (heterogeneous per-feature scale). This is a genuinely strong result to go measure rather than assume.
- **BatchNorm:** normalizes activations to zero-mean/unit-variance per batch. The originally-claimed mechanism ("reduces internal covariate shift") was directly challenged by Santurkar et al. (2018), who showed BN's real effect is smoothing the loss landscape (more Lipschitz loss and gradients), which is what permits larger, more stable learning rates. Design implication: **a pure linear-regression model has no hidden activation to normalize** — "BN before a single linear layer" degenerates into input standardization, which only fixes scale-based conditioning, not correlation-based conditioning. To test BN in the sense the literature means, you need at least one hidden layer (see Part 4).

**One conceptual split that will sharpen your whole study.** There are two different flavors of "ill-conditioned," and your five methods respond to them differently:

- *Scale-based*: feature variances differ wildly but aren't correlated (diagonal-ish G, eigenvectors ≈ coordinate axes). Fixed by standardization, BN-style normalization, and largely by Adam.
- *Correlation-based*: features are nearly collinear (off-diagonal G, eigenvectors rotated away from the axes). NOT fixed by per-feature standardization (rescaling a feature doesn't touch its correlation with another). Needs ridge, momentum, or something that accounts for the rotated geometry.
  Generate both separately (Part 2) and show that Adam/BN fix one but not the other — a much sharper finding than "Adam beats SGD."

**References worth having on hand:** Polyak (1964, heavy-ball); Nesterov's convex optimization lecture notes (general κ-rate theory, accelerated gradient); Hoerl & Kennard (1970, ridge regression); Kingma & Ba (Adam); Ioffe & Szegedy (BatchNorm); Santurkar, Tsipras, Ilyas & Madry ("How Does Batch Normalization Help Optimization?", 2018); Loshchilov & Hutter (AdamW / decoupled weight decay — matters below). For flavor on what real (non-toy) Hessians look like, the empirical neural-network-Hessian-spectrum literature (a bulk of small eigenvalues plus a few large outliers) is worth skimming — it can inform how you shape your synthetic spectra beyond a simple two-value toy.

## 2. Generating the synthetic data

Core principle: **don't hope for a target κ by adding correlated noise — construct the spectrum directly, so κ is exact, not approximate.**

**Recommended method: build X from its SVD directly.** This gives exact control with no asymptotics:

1. Pick n (samples), d (features), and a target κ.
2. Since κ(XᵀX) = κ(X)², choose X's singular values running from σ_max down to σ_min = σ_max/√κ.
3. Space the intermediate singular values — log-spaced (geometric) between the extremes is the standard, clean default (alternatives below).
4. Generate two random orthogonal matrices (one n×n or n×d, one d×d) by orthogonalizing a matrix of i.i.d. Gaussian entries — these are U and V.
5. Assemble X = UΣVᵀ, Σ being the rectangular diagonal matrix of your chosen singular values.
6. Sanity-check by computing the actual eigenvalues of XᵀX and confirming κ matches target exactly (up to floating point).

**Alternative: generate from a covariance model**, if you want data that looks like "natural" correlated Gaussian features rather than an engineered X:

1. Build target covariance Σ = QΛQᵀ (Λ = your eigenvalue spectrum, Q = random orthogonal eigenvector directions).
2. Take Σ's Cholesky factor L.
3. Sample rows as z·Lᵀ, z ~ N(0, I).
   Caveat: with finite n, the *empirical* Gram matrix's κ won't exactly match Σ's — sample-covariance eigenvalues spread out relative to the population ones, more so when n isn't much larger than d (a Marchenko–Pastur / random-matrix-theory effect). Use the SVD method if you need exact κ, or make sure n≫d here.

**Controlling where the ill-conditioning comes from** — treat this as its own experimental factor:

- *Scale-based version*: keep Q ≈ identity, so eigenvectors ≈ coordinate axes, and just make individual feature variances span many orders of magnitude with no cross-correlation.
- *Correlation-based version*: keep individual feature variances similar, but make Q a genuinely random rotation, so the bad conditioning comes from cross-feature correlation — closer to classic regression multicollinearity.

**Shape of the spectrum:** log-spaced/geometric between the extremes (clean default); two-cluster/bimodal — most eigenvalues near λ_max, a few near λ_min or vice versa — closer to real neural-net Hessian spectra, worth trying once the basic story is established; or a simple two-value toy (half at λ_max, half at λ_min) for the cleanest possible visualizations and the tightest match to the (κ−1)/(κ+1)-style formulas.

**Labels and ground truth.** Fix a true θ* once (e.g., drawn from N(0,I), reused across all κ so it isn't a confound), then y = Xθ* + ε, ε ~ N(0, σ²I). Keep two variants: **noiseless** (σ=0), which isolates pure optimization/numerical behavior for your headline convergence plots, and **noisy** at a controlled SNR, which lets you see the interaction between conditioning and estimation variance — this is where ridge's bias/variance trade-off actually shows up (in the noiseless case ridge just looks like pure bias with no upside). Keeping θ* known is what lets you measure parameter recovery error ||θ_t − θ*|| directly, which is more informative than loss alone.

**Concrete severity levels to sweep** — treat κ as your primary independent variable, log-spaced across many orders of magnitude: κ≈1–10 (well-conditioned control), κ≈10²–10⁴ (moderate, "normal bad" regression territory), κ≈10⁵–10⁷ (severe, approaching float32's breakdown), κ≈10⁸–10¹⁵ (pathological — float32 already garbage, float64 failing near the top). A grid like κ ∈ {1, 10, 10², 10³, 10⁴, 10⁵, 10⁶, 10⁷, 10⁸, 10¹⁰}, run in both float32 and float64, turns "numerically lossy" from an assertion in your intro into a measured crossover point.

**Secondary factors, if you extend later:** n/d ratio, noise level/SNR, and float precision (32 vs 64 — genuinely worth treating as a core factor rather than an afterthought, given your own framing's emphasis on numerical lossiness).

## 3. Performance metrics

Group these into five buckets so your report has real structure.

**Convergence / optimization dynamics:** training loss vs iteration on a log scale (essential — linear scale hides everything but the first few steps); gradient norm ||∇L(θ_t)|| vs iteration (distinguishes "still approaching a minimum" from "plateaued"); iterations-to-threshold (steps to reach loss < ε or ||θ_t−θ*|| < ε) — your main quantitative axis, plotted against κ to test the O(κ) vs O(√κ) prediction. **Per-eigendirection error decomposition** is the standout metric here: since you control G's eigenvectors, project e_t = Qᵀ(θ_t−θ*) and plot the top-eigenvector component separately from the bottom-eigenvector component over iterations. This directly shows "oscillate in steep directions, crawl in flat ones" from your own project description, and shows concretely whether each method fixes the oscillation, the crawl, or both. Also worth having: an oscillation count (sign changes of the steep-direction gradient component across iterations), or total path length vs net displacement (a large ratio = wasted zig-zag).

**Numerical stability:** divergence/failure rate (fraction of runs hitting NaN/Inf, and at what iteration); a direct-solve comparison — compute θ̂ via naive (XᵀX)⁻¹Xᵀy and via a stable solve (SVD/pseudoinverse or QR), then plot relative error against κ for both. This is what turns "matrix inversion becomes lossy" into an actual measured result: the naive solve should blow up well before the stable one, and the blow-up point should roughly track 1/machine-epsilon. Repeat in float32 vs float64 to show the precision-dependent crossover directly.

**Statistical / estimation quality:** bias and variance of θ̂ across repeated noisy trials, decomposing MSE = Bias² + Variance — this is where ridge's real story shows up (Var(θ̂) = σ²(XᵀX+λI)⁻¹XᵀX(XᵀX+λI)⁻¹ shrinks with λ while bias grows); held-out test error/R² on a separate test set, connecting optimization behavior to generalization, since small-eigenvalue directions are exactly where OLS variance is highest (Var(θ̂)=σ²(XᵀX)⁻¹ has eigenvalues 1/λᵢ — the classical multicollinearity-inflates-variance result); optionally, ridge's effective degrees of freedom, df(λ) = Σᵢ λᵢ/(λᵢ+λ), a standard way to quantify how much shrinkage a given λ applies.

**Robustness & hyperparameter sensitivity:** maximum stable learning rate vs κ per method (theory says plain GD's boundary is ~2/λ_max — track where each method empirically starts to diverge, and how that shifts with κ); hyperparameter grids as heatmaps, not single numbers (LR×momentum, LR×Adam's β's, LR×ridge λ) — this also protects you from the single biggest threat to a fair comparison, an undertuned baseline; multi-seed variance — every headline number should be mean ± std across at least 10–20 seeds (varying data draw and initialization), since single curves are anecdotes, especially this close to numerical failure thresholds; wall-clock time per method, not just iteration count, since Adam and BatchNorm both do more work per step than vanilla SGD.

**Metrics that double as figures:** sorted eigenvalue spectrum of G before/after ridge (shows the λᵢ→λᵢ+λ shift directly) and before/after standardization (shows what standardization does and doesn't fix, tying back to the scale-vs-correlation split); a 2D loss-landscape contour plot with optimizer trajectories overlaid, for a small toy case (d=2) — the classic narrow-valley/zig-zag-vs-smooth-path figure, worth producing once from your own real trajectories as a centerpiece; a heatmap grid of final loss (or iterations-to-converge) over κ×learning-rate, one panel per method — a single figure summarizing the whole study.

## 4. Methodology — how to run the study

**Decide your model class first.** Run this as two tiers: a **primary linear-regression testbed** (matches your Gram-matrix framing exactly, gives closed-form ground truth, keeps the loss exactly quadratic so Part 1's theory applies cleanly), and a **secondary shallow MLP** (one hidden layer), specifically so BatchNorm has an actual hidden activation to normalize in the sense the literature means it — otherwise "BatchNorm" quietly becomes "input standardization," a real but weaker experiment.

**Implement each method with its real nuances:**

- Test **ridge** two ways — the closed-form solve (XᵀX+λI)⁻¹Xᵀy, isolating the pure numerical-conditioning benefit, and as an L2 penalty optimized iteratively with SGD, testing it as a training-time stabilizer. These answer different questions.
- Be careful combining **ridge/L2 with Adam** — this is where your own phrasing ("Ridge / L2 weight decay") hides a real subtlety. L2-in-the-loss and decoupled weight decay are mathematically identical for vanilla SGD, but *not* once Adam's per-parameter adaptive scaling is in the loop (this is the entire point of the AdamW paper): L2-in-the-loss gets divided by Adam's second-moment estimate, silently weakening regularization for parameters with large historical gradients. If you're testing "Adam + regularization" as a stabilizer, test both variants and report the difference — a genuinely under-appreciated result in exactly this ill-conditioned regime.
- Standard heavy-ball **momentum** (single β) matches Part 1's theory directly; Nesterov's variant is a natural bonus, not necessary for the core study.

**Make the comparison actually fair** — the most common failure mode in these studies, so treat it as a first-class step: tune each method's hyperparameters separately (small grid) rather than reusing one learning rate across all five; fix a consistent stopping rule (iteration budget or wall-clock budget — pick one, be explicit, since rankings can differ between them); use matched initial conditions within a trial (same θ₀, same data draw, same κ, vary only the optimizer), repeated across many seeds; report mean ± spread, never a single curve.

**Run the core sweep:** for each κ, for each of the five methods (tuned per above), for each seed, run to your stopping rule and collect every metric from Part 3. A grid like 10 κ values × 5 methods × 15 seeds × 2 precisions is ~1,500 runs — trivial for linear regression, so don't skimp on seeds to save compute.

**Check against theory — this is the capstone analysis.** Plot iterations-to-threshold against κ on a log-log scale and fit a power law (the slope is the exponent). Theory predicts slope ≈1 for Vanilla SGD and ≈0.5 for Momentum SGD. This one analysis turns your project from "we tried some optimizers and one was better" into "we measured the κ vs √κ scaling directly" — the strongest possible result for this kind of study.

**Report:** a summary table (method × κ level × key metrics), the log-log scaling plot, the eigenvalue-spectrum figures, the per-eigendirection decomposition plot, and a discussion tying back to the unifying frame from Part 1 (each method as a cheap approximation to Newton preconditioning).

**Optional extensions:** a **whitening/decorrelation** baseline (transform features so G becomes exactly I, κ=1) as an oracle upper bound on what any preconditioning-based method could achieve; **Newton's method** as the opposite-axis oracle — for an exact quadratic it converges in one step regardless of κ, precisely because it uses G⁻¹ directly, which makes the central tension explicit (the most powerful step direction is also the most numerically dangerous one to compute); and a real (non-synthetic) collinear regression dataset, to check whether your synthetic-spectrum conclusions transfer.

---

**Practical order of operations:** (1) pick model classes — linear core + optional MLP for BN; (2) build the SVD-based generator, verify exact target κ; (3) build both scale-based and correlation-based variants; (4) implement the five methods, plus the L2-vs-decoupled-decay variant for Adam; (5) tune hyperparameters per method; (6) run the full κ×method×seed×precision sweep; (7) compute all metrics, build the eigen-decomposition and spectrum figures; (8) fit the log-log scaling law against theory; (9) write up using the "cheap approximations to G⁻¹" frame to unify the discussion.

If it'd help make the geometry concrete, I can sketch the narrow-valley/eigenvalue-spectrum concept as a diagram — separate from the real figure you'll generate from your own trajectories.
