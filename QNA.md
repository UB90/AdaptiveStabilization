### [1] Matrix Ill-Conditioning

- **Definition & Mechanics (What is it?)**: A structural deformity in the data matrix $X$, quantified by the condition number $\kappa$. It is the ratio of the maximum to minimum eigenvalues of the Hessian matrix $H = X^T X$, expressed as $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$. An ill-conditioned matrix ($\kappa \gg 1$, e.g., $10^7$) creates a loss landscape characterized by steep ravines along $\lambda_{\max}$ and nearly flat, infinitely long valleys along $\lambda_{\min}$.

- **Causation (Why does it occur?)**: It originates from severe multicollinearity among input features, extreme disparities in feature scaling, or a fundamental lack of variance across specific singular vectors in the dataset.

### [2] Stabilization Regularization

- **Core Definition (What is it?)**: Mathematical modifications applied to the objective function or update rule to prevent parameter divergence and enforce stable geometry.

- **Operational Purpose (Why is it used?)**: To counteract the noise amplification and parameter explosion that occur when optimizers attempt to navigate the flat singular spaces ($\lambda_{\min}$) of an ill-conditioned landscape, where stochastic noise overpowers the true gradient signal.

- **Functional Mechanism (How does it help? / What are the techniques?)**:
- **$L_2$ Regularization (Ridge)**: Modifies the objective to include $\frac{\lambda}{2} \Vert \theta \Vert_2^2$. This shifts the Hessian eigenvalues to $\lambda_i + \lambda$, bounding the maximum condition number to $\frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$. This mathematically guarantees strict convexity and provides a permanent restoring force toward the origin along flat dimensions.

- **Gradient Clipping**: Truncates gradient magnitudes to prevent numerical overflow during steep ravine traversal.

### [3] Optimizer Dynamics

- **Evaluated Algorithms (What techniques are used?)**: Vanilla SGD, Momentum SGD, Adam, AdamW, and Batch-Norm injected layers.

- **Execution Rationale (Why were they used?)**: To empirically stress-test the structural limits of primary optimization paradigms (first-order, autoregressive velocity, adaptive variance, and scale-invariant projections) against adverse mathematical conditioning. The objective is to map their distinct failure modes (e.g., Adam's step inflation, Momentum's random drift) when the signal-to-noise ratio approaches zero.

### [4] Experimental Architecture

- **Model Selection (Why Linear Regression?)**: Linear regression provides a purely quadratic loss landscape with a mathematically guaranteed constant Hessian ($H = X^T X$) and a closed-form global minimum ($\theta^* = (X^T X)^{-1} X^T y$). This isolates the optimizer's dynamic behavior from the compounding variables of deep neural networks (local minima, saddle points, dynamic Hessian evolution), allowing exact computation of parameter error $\Vert \theta_t - \theta^* \Vert_2$.

- **Epoch Horizons (Why is the epoch that number?)**: Time scales are dictated by the condition number. Vanilla SGD requires extreme horizons (e.g., 50,000+ epochs) because the learning rate is restricted by $\lambda_{\max}$, resulting in infinitesimal progress along the flat $\lambda_{\min}$ directions. Adaptive optimizers (Adam) require far fewer epochs (e.g., 1,000) because they artificially accelerate movement along flat directions, rapidly triggering parameter explosion and failure.

- **Learning Rate Boundaries (Why is the LR like that?)**: The maximum permissible learning rate for convergence is strictly bound by theoretical stability limits: $\alpha < \frac{2}{\lambda_{\max}}$. Exceeding this limit causes instant divergent oscillation. Therefore, step sizes must be infinitesimally small in highly conditioned scenarios unless adaptive scaling is applied.

- **Strategic Objective (Why a project like this?)**: To dismantle the assumption that empirical training loss minimization equates to successful system identification. The project proves mathematically and empirically that modern optimizers can achieve perfect loss ($\approx 10^{-6}$) while the internal parameters completely diverge from reality, exposing a critical vulnerability in applied machine learning pipelines.

### [5] Decoupled Regularization (Adam vs. AdamW)

- **Mechanism Definition (What is the difference?)**: Standard Adam implements $L_2$ regularization by appending the derivative of the penalty directly to the gradient ($g_t = \nabla \mathcal{L} + \lambda \theta_t$) before calculating the adaptive second-moment accumulator. AdamW decouples this operation, applying weight decay directly to the final parameter update step: $\theta_{t+1} = \theta_t - \eta \hat{m}_t - \eta \lambda \theta_t$.

- **Execution Necessity (Why use AdamW?)**: In standard Adam, the adaptive denominator scales the $L_2$ penalty inversely to historical gradient variance. Parameters with large gradients receive suppressed regularization, destroying the uniform spherical geometry of the $L_2$ penalty. AdamW isolates the decay step, forcing a uniform contraction toward the origin regardless of gradient history.

### [6] Scale Invariance (Batch Normalization)

- **Mathematical Cause (How does BN cause scale invariance?)**: Batch Normalization enforces an output standard deviation of $1$. For a linear projection $y = \theta x$, scaling the weights by any scalar $c$ results in $y = c\theta x$. The normalization operator strictly eliminates this constant via its variance denominator: $\text{BN}(c \theta x) = \text{BN}(\theta x)$.

- **Parametric Consequence (Why does this break parameter recovery?)**: The gradient of the loss with respect to the radial scale of the weight vector becomes exactly zero ($\nabla_{\theta} \mathcal{L}^T \theta = 0$). The optimizer can iteratively correct the angular direction of the parameter vector but is structurally blinded to its absolute magnitude. Consequently, true parameter identification ($\Vert \theta - \theta^* \Vert_2 \to 0$) is mathematically impossible.

### [7] Subspace Tracking (Eigenvalue Error)

- **Metric Definition (What is Eig Error?)**: The projection of the current parameter error vector onto the orthogonal eigenvectors ($V$) of the data Hessian matrix: $E = V^T(\theta_t - \theta^*)$. It dissects total error into independent components aligned with the steepest ($\lambda_{\max}$) and flattest ($\lambda_{\min}$) curvature directions.
- **Diagnostic Value (Why track it?)**: Total parameter error ($\Vert \theta - \theta^* \Vert_2$) is a scalar that obscures spatial dynamics. Eig Error tracking isolates the specific axis of failure. It provides absolute proof that runaway parameter divergence in Adam and Momentum occurs exclusively along the $\lambda_{\min}$ subspace, validating the noise-amplification hypothesis.

### [8] The Loss-Parameter Disconnect (Convergence Illusion)

- **The Phenomenon (What happens?)**: A catastrophic monitoring failure where empirical training loss achieves an optimal terminal state ($\mathcal{L} \approx 10^{-6}$) while the Euclidean distance between the active parameters and the true generative parameters explodes (e.g., $\Vert \theta - \theta^* \Vert_2 > 5.0$).
- **The Structural Cause (Why does this happen?)**: Extreme ill-conditioning ($\kappa = 10^7$) creates infinitely long, perfectly flat valleys in the loss topology. Movement of thousands of units along these specific singular vectors produces a change in the loss value that is smaller than standard floating-point precision. The optimizer satisfies the objective function without ever solving the underlying linear system.

### [9] Gradient Noise Floor (Stochastic Injection)

- **Mechanism Definition (What is it?)**: The variance ($\sigma^2$) introduced into the optimization trajectory by calculating gradients on uniformly sampled mini-batches of data rather than the full deterministic dataset.
- **Failure Trigger (How does it cause divergence?)**: Along flat eigenvectors ($\lambda_{\min}$), the true structural gradient magnitude approaches zero. When this structural signal drops below the stochastic noise floor, the signal-to-noise ratio inverts. Adaptive optimizers amplify this noise via division by small moment accumulators, and momentum algorithms integrate this noise over time, driving the unconstrained random walks responsible for optimization failure.
