"""
EXP 4 — Batch Normalization as Structural Preconditioning
Claim (slide 8): BN bounds singular values of each layer's Jacobian,
preventing exponential growth/decay of activations and gradients across depth.
Test: deep linear network (random weights, scale > 1) with vs without BN,
track ||activation|| and gradient norm by layer depth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(7)

def batchnorm(x, eps=1e-5):
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def run(depth=30, width=64, batch=256, weight_scale=1.3, nonlinearity="linear"):
    """nonlinearity='linear' isolates the Jacobian-scaling claim (no squashing to mask it).
    nonlinearity='tanh' shows the more realistic, partially self-limiting case."""
    act = (lambda z: z) if nonlinearity == "linear" else np.tanh

    x_no_bn = np.random.randn(batch, width)
    x_bn = x_no_bn.copy()

    norms_no_bn, norms_bn = [np.linalg.norm(x_no_bn, axis=1).mean()], [np.linalg.norm(x_bn, axis=1).mean()]
    sv_no_bn, sv_bn = [], []

    for layer in range(depth):
        W = np.random.randn(width, width) * weight_scale / np.sqrt(width)

        x_no_bn = act(x_no_bn @ W)
        x_bn_pre = x_bn @ W
        x_bn = act(batchnorm(x_bn_pre))

        norms_no_bn.append(np.linalg.norm(x_no_bn, axis=1).mean())
        norms_bn.append(np.linalg.norm(x_bn, axis=1).mean())

        # effective per-layer Jacobian singular values (via W, ignoring tanh saturation for the linear part)
        s = np.linalg.svd(W, compute_uv=False)
        sv_no_bn.append(s.max() / s.min())  # condition number of W alone, same both cases (BN acts on activations not W)

    print(f"{'layer':>6} | {'||act|| no-BN':>15} | {'||act|| with-BN':>16}")
    print("-" * 45)
    for i in [0, 5, 10, 15, 20, 25, 30]:
        print(f"{i:6d} | {norms_no_bn[i]:15.4e} | {norms_bn[i]:16.4e}")

    print(f"\nFinal/initial norm ratio (growth factor over {depth} layers):")
    print(f"  no-BN : {norms_no_bn[-1] / norms_no_bn[0]:.4e}")
    print(f"  BN    : {norms_bn[-1] / norms_bn[0]:.4e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(norms_no_bn, label="No BN")
    ax.plot(norms_bn, label="With BN")
    ax.set_yscale("log")
    ax.set_xlabel("layer depth")
    ax.set_ylabel("mean activation norm (log scale)")
    ax.set_title(f"Activation norm vs depth | weight_scale={weight_scale}, act={nonlinearity}")
    ax.legend()
    plt.tight_layout()
    fname = f"/home/claude/experiments/exp4_batchnorm_depth_{nonlinearity}.png"
    plt.savefig(fname, dpi=130)
    print(f"\nSaved plot: {fname}")

if __name__ == "__main__":
    print("=== LINEAR network (isolates Jacobian-scaling effect, no squashing) ===")
    run(nonlinearity="linear")
    print("\n=== TANH network (realistic case, nonlinearity partially self-limits) ===")
    run(nonlinearity="tanh")
