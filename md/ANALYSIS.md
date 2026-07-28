# Empirical Optimization vs. Structural Parameter Reconstruction Under Data Ill-Conditioning

This document presents a comprehensive comparative analysis of optimization algorithms evaluated across varying data condition numbers ($\kappa$). The core focus of this investigation is the crucial divergence between **loss minimization** (empirical convergence) and **parameter identification** (recovering the true underlying system vector $\theta^*$).

---

## 1. Global Performance Overview (Across All $\kappa$)

The global tracking plots illustrate the trajectories of all target optimizers across three conditioning regimes: well-conditioned ($\kappa = 10$), moderately ill-conditioned ($\kappa = 10^4$), and severely ill-conditioned ($\kappa = 10^7$).

### Global Trajectory Plots

### Key Global Insights

- **The Optimization Mirage:** For almost all unregularized architectures (Adam, AdamW, Momentum), train and test losses fall rapidly to the noise floor ($\sim 10^{-6}$) regardless of the condition number. However, their corresponding parameter errors ($\Vert{}\theta - \theta^*\Vert{}_2$) reveal complete structural divergence under high ill-conditioning.
- **The Regularization Exception:** `SGD + L2 (Ridge)` stands out as the only method where parameter error tracking mirrors loss minimization across all phases, achieving near-perfect reconstruction ($< 10^{-2}$) almost instantaneously.
- **The Batch Normalization Flaw:** `BN-on-Linear` completely decouples parameter error from the conditioning matrix. It achieves low loss values but leaves parameter error entirely unresolved at a static plateau of $\approx 0.98$.

---

## 2. Regime-Specific Micro-Analysis

---

### Regime A: Well-Conditioned Space ($\kappa = 10$)

In this ideal geometric landscape, the loss surface features symmetric, isotropic curvature. All directions provide clean gradient signals.

#### Optimization Trajectories ($\kappa = 10$)

- **Behavior Analysis:** Every standard optimizer (Vanilla SGD, Momentum, Adam, AdamW) exhibits textbook convergence. Both loss and parameter error decline rapidly in tandem. The parameter vectors easily track the true $\theta^*$ within the first 100 to 200 epochs.
- **Eigenvalue Resolution:** The `Eig Error` panels demonstrate that errors along both the top (solid line) and bottom (dashed line) singular vectors drop below $10^{-3}$ simultaneously, confirming uniform, unhindered identification.

---

### Regime B: Moderately Ill-Conditioned Space ($\kappa = 10^4$)

As curvature stretches into an elongated valley, structural differences between adaptive scaling, momentum, and vanilla gradients emerge.

#### Optimization Trajectories ($\kappa = 10^4$)

- **Adaptive vs. Momentum Settling:** Adam, AdamW, and Momentum quickly find stable parameter vectors. However, notice that their final parameter errors stabilize around $\approx 0.28$ rather than falling to zero. The eigenvalue plots show that while the top eigenvalue error drops cleanly, the bottom eigenvalue error begins to fluctuate alongside the injected noise floor ($\sim 10^{-3}$).
- **The Vanilla SGD Transient Valley:** Vanilla SGD shows a distinct parabolic trajectory in its parameter error panel. It gently tracks downward to a minimum error of $\approx 0.15$ around epoch 2,500 before entering a slow, upward drift. This highlights SGD's late-stage vulnerability to cumulative noise accumulation along flat coordinates, even when the loss remains perfectly flat.

---

### Regime C: Severely Ill-Conditioned Space ($\kappa = 10^7$)

Under extreme ill-conditioning, the landscape transforms into an ultra-flat valley floor bordered by exceptionally steep walls. True structural signals along the flat directions are buried beneath the stochastic noise floor.

#### Optimization Trajectories ($\kappa = 10^7$)

- **Catastrophic Parameter Explosion:** For Adam, AdamW, and Momentum, parameter error does not merely stall—it **explodes**. The error climbs from an initial baseline of $1.0$ up to nearly $6.0$.
- **Mechanistic Proof via Eig Error:** The `Eig Error` panels for Adam and Momentum expose the exact source of this failure. The solid line (top eigenvalue direction) drops rapidly to $10^{-3}$, meaning the steep directions are optimized perfectly. However, the dashed line (bottom eigenvalue direction) shoots upward, crossing $1.0$ and stabilizing near $4.0$. The optimizers are actively driving weights to extreme magnitudes along the flat, noise-dominated vectors to capture microscopic training loss reductions.
- **Vanilla SGD's Geometric Safety:** Remarkably, Vanilla SGD acts as the only unregularized method resistant to this explosion. Because its step updates are directly proportional to gradient magnitude, it naturally ignores flat directions where the true gradient is negligible. Its parameter error steadily trends downward, safely approaching $\approx 0.75$ by epoch 50,000.

---

## 3. Deep-Dive Architecture & Optimizer Breakdown

### Adaptive Optimizers (Adam & AdamW)

Adaptive algorithms scale updates by the inverse square root of the running second moment:

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} m_t$$

Along flat singular directions where the true gradient component approaches zero, the second moment accumulator $v_t$ is dominated almost entirely by the injected variance ($\sigma = 10^{-3}$). Division by $\sqrt{v_t}$ artificially amplifies the step sizes along these uninformative directions. This forces the parameter weights to expand drastically, chasing random noise realizations to minimize empirical loss at the cost of structural destruction.

### Momentum SGD

Momentum continuously integrates historical gradient noise over time via its autoregressive update rule. Lacking a strong restoring gradient along the minimum eigenspaces, this velocity accumulation turns the optimization process into an unconstrained random walk along the valley floor. As a result, its parameter error rises continuously in the severe regime, matching the failure profile of the adaptive methods.

### Batch Normalization (BN-on-Linear)

The structural failure of `BN-on-Linear` represents a fundamental mathematical limitation:

$$\hat{X} = (X - \mu)\Sigma^{-1/2}$$

By shifting and scaling the feature matrix to ensure zero mean and unit variance prior to the linear layer, BN removes the absolute scale and structural information inherent to the original data matrix. Because the true label generator relies on the unnormalized space ($f(X) = X\theta^*$), the network easily minimizes empirical loss by adapting its weights to the normalized distributions. However, it can never recover the true absolute coordinates of $\theta^*$. Consequently, its relative parameter error remains locked at $\approx 0.98$ across all condition numbers and epochs.

### Ridge Regularization (SGD + L2)

Ridge regression introduces a spherical quadratic penalty that alters the underlying geometry of the optimization space, directly modifying the data Hessian:

$$H_{\text{reg}} = X^T X + \lambda I$$

This shifts the entire spectrum of eigenvalues upward by $\lambda$, bound-protecting the minimum curvature. Even under severe ill-conditioning ($\kappa = 10^7$), the optimizer encounters a continuous, reliable restoring force toward the origin along flat directions. This effectively cuts off both adaptive step explosion and momentum drift, enabling rapid and accurate parameter recovery.

---

## 4. Architectural Summary

| Optimizer / Architecture | Loss Convergence Performance | Parameter Identification Accuracy ($\kappa \le 10^4$) | Parameter Identification Accuracy ($\kappa = 10^7$) | Primary Failure Mode under Extreme Curvature                  |
| ------------------------ | ---------------------------- | ----------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| **Adam / AdamW**         | Exceptional ($\sim 10^{-6}$) | Moderate ($\approx 0.28$ Error)                       | Catastrophic ($\approx 6.0$ Error)                  | Inverse-variance noise amplification along flat directions    |
| **Momentum SGD**         | Exceptional ($\sim 10^{-6}$) | Moderate ($\approx 0.28$ Error)                       | Catastrophic ($\approx 6.0$ Error)                  | Autoregressive noise drift along the valley floor             |
| **Vanilla SGD**          | Slow / Moderate              | Good ($\approx 0.15$ Error Trajectory)                | Robust Baseline ($\approx 0.75$ and falling)        | Susceptible to slow, late-stage stochastic noise accumulation |
| **BN-on-Linear**         | Volatile but Low             | Complete Failure ($\approx 0.98$ Fixed Error)         | Complete Failure ($\approx 0.98$ Fixed Error)       | Scale elimination via feature variance normalization          |
| **SGD + L2 (Ridge)**     | Instantaneous                | Perfect ($< 10^{-2}$ Error)                           | Perfect ($< 10^{-2}$ Error)                         | None (Successfully restores minimum curvature spectrum)       |
