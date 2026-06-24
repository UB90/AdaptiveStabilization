"""
EXP 3 — Adaptive Preconditioning in Gradient Descent
Claim (slide 6): vanilla GD oscillates/crawls on ill-conditioned H;
Adam/AdaGrad/RMSprop adapt per-dimension and converge faster.
Test: f(x) = 0.5 * x^T H x, H diagonal with controlled kappa.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(1)

def make_H(kappa, n=2):
    # H = diag(kappa, 1) generalizes to n dims via geomspace
    return np.diag(np.geomspace(kappa, 1.0, n))

def grad(H, x):
    return H @ x

def loss(H, x):
    return 0.5 * x @ H @ x

def run_gd(H, x0, lr, steps):
    x = x0.copy()
    hist = [x.copy()]
    for _ in range(steps):
        x = x - lr * grad(H, x)
        hist.append(x.copy())
    return np.array(hist)

def run_adagrad(H, x0, lr, steps, eps=1e-8):
    x = x0.copy()
    G = np.zeros_like(x)
    hist = [x.copy()]
    for _ in range(steps):
        g = grad(H, x)
        G += g ** 2
        x = x - lr * g / (np.sqrt(G) + eps)
        hist.append(x.copy())
    return np.array(hist)

def run_rmsprop(H, x0, lr, steps, beta=0.9, eps=1e-8):
    x = x0.copy()
    v = np.zeros_like(x)
    hist = [x.copy()]
    for _ in range(steps):
        g = grad(H, x)
        v = beta * v + (1 - beta) * g ** 2
        x = x - lr * g / (np.sqrt(v) + eps)
        hist.append(x.copy())
    return np.array(hist)

def run_adam(H, x0, lr, steps, b1=0.9, b2=0.999, eps=1e-8):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    hist = [x.copy()]
    for t in range(1, steps + 1):
        g = grad(H, x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        hist.append(x.copy())
    return np.array(hist)

def run():
    kappa = 1000.0
    H = make_H(kappa, n=2)
    x0 = np.array([1.0, 1.0])
    steps = 200

    results = {
        "SGD (lr=1e-3)": run_gd(H, x0, lr=1e-3, steps=steps),
        "AdaGrad (lr=0.5)": run_adagrad(H, x0, lr=0.5, steps=steps),
        "RMSprop (lr=0.1)": run_rmsprop(H, x0, lr=0.1, steps=steps),
        "Adam (lr=0.1)": run_adam(H, x0, lr=0.1, steps=steps),
    }

    print(f"kappa(H) = {kappa:.1e}")
    print(f"{'Optimizer':<20} | {'loss @ step 0':>14} | {'loss @ step 50':>14} | {'loss @ step 200':>15}")
    print("-" * 70)
    for name, hist in results.items():
        losses = np.array([loss(H, x) for x in hist])
        print(f"{name:<20} | {losses[0]:14.4e} | {losses[50]:14.4e} | {losses[-1]:15.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, hist in results.items():
        losses = np.array([loss(H, x) for x in hist])
        axes[0].plot(losses, label=name)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title(f"Convergence, kappa(H)={kappa:.0e}")
    axes[0].legend()

    # trajectory in parameter space (steep dim vs flat dim)
    for name, hist in results.items():
        axes[1].plot(hist[:, 0], hist[:, 1], marker=".", markersize=2, label=name)
    axes[1].set_xlabel("x0 (steep direction, eigval=kappa)")
    axes[1].set_ylabel("x1 (flat direction, eigval=1)")
    axes[1].set_title("Trajectory: steep vs flat dimension")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("/home/claude/experiments/exp3_optimizer_comparison.png", dpi=130)
    print("\nSaved plot: exp3_optimizer_comparison.png")

if __name__ == "__main__":
    run()
