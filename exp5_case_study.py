"""
EXP 5 — Case Study Reproduction
Claim (slide 11): 3-layer MLP, features with corr rho>0.95 -> kappa(X^T X)~10^7.
SGD oscillates/slow; Adam+BN fast and stable.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(3)

def make_correlated_data(n_samples=2000, n_features=10, rho=0.999999):
    cov = np.full((n_features, n_features), rho)
    np.fill_diagonal(cov, 1.0)
    X = np.random.multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    true_w = np.random.randn(n_features)
    y = (X @ true_w > 0).astype(float)  # binary classification
    return X, y.reshape(-1, 1)

def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)
def sigmoid(z): return 1 / (1 + np.exp(-z))

class MLP:
    """3-layer MLP, manual forward/backward, optional BatchNorm after each hidden layer."""
    def __init__(self, sizes, use_bn=False, seed=0):
        rng = np.random.RandomState(seed)
        self.sizes = sizes
        self.use_bn = use_bn
        self.W = [rng.randn(sizes[i], sizes[i+1]) / np.sqrt(sizes[i]) for i in range(len(sizes)-1)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]
        if use_bn:
            self.gamma = [np.ones((1, sizes[i+1])) for i in range(len(sizes)-2)]
            self.beta = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-2)]

    def forward(self, X):
        cache = {"a": [X]}
        if self.use_bn:
            cache["bn"] = []
        a = X
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            if self.use_bn:
                mu, var = z.mean(0, keepdims=True), z.var(0, keepdims=True)
                z_hat = (z - mu) / np.sqrt(var + 1e-5)
                z_out = self.gamma[i] * z_hat + self.beta[i]
                cache["bn"].append((z, mu, var, z_hat))
            else:
                z_out = z
            a = relu(z_out)
            cache["a"].append(a)
            cache.setdefault("z_pre", []).append(z)
        z_final = a @ self.W[-1] + self.b[-1]
        y_hat = sigmoid(z_final)
        cache["a"].append(y_hat)
        cache["z_final"] = z_final
        return y_hat, cache

    def backward(self, y, cache):
        m = y.shape[0]
        grads = {"W": [None]*len(self.W), "b": [None]*len(self.b)}
        if self.use_bn:
            grads["gamma"] = [None]*len(self.gamma)
            grads["beta"] = [None]*len(self.beta)

        dz = (cache["a"][-1] - y) / m  # dL/dz_final, BCE+sigmoid combined grad
        grads["W"][-1] = cache["a"][-2].T @ dz
        grads["b"][-1] = dz.sum(0, keepdims=True)
        da_prev = dz @ self.W[-1].T

        for i in reversed(range(len(self.W) - 1)):
            dz_out = da_prev * relu_grad(cache["z_pre"][i] if not self.use_bn else cache["bn"][i][3])
            if self.use_bn:
                z, mu, var, z_hat = cache["bn"][i]
                dgamma = (dz_out * z_hat).sum(0, keepdims=True)
                dbeta = dz_out.sum(0, keepdims=True)
                dz_hat = dz_out * self.gamma[i]
                std_inv = 1.0 / np.sqrt(var + 1e-5)
                dz_pre = (1.0/m) * std_inv * (m*dz_hat - dz_hat.sum(0, keepdims=True)
                          - z_hat * (dz_hat * z_hat).sum(0, keepdims=True))
                grads["gamma"][i] = dgamma
                grads["beta"][i] = dbeta
            else:
                dz_pre = dz_out
            grads["W"][i] = cache["a"][i].T @ dz_pre
            grads["b"][i] = dz_pre.sum(0, keepdims=True)
            da_prev = dz_pre @ self.W[i].T
        return grads

    def loss(self, y, y_hat, eps=1e-9):
        return -np.mean(y*np.log(y_hat+eps) + (1-y)*np.log(1-y_hat+eps))


def train(model, X, y, optimizer="sgd", lr=0.1, epochs=300):
    params = model.W + model.b + (model.gamma + model.beta if model.use_bn else [])
    if optimizer == "adam":
        m_t = [np.zeros_like(p) for p in params]
        v_t = [np.zeros_like(p) for p in params]
        b1, b2, eps = 0.9, 0.999, 1e-8

    losses = []
    for epoch in range(1, epochs+1):
        y_hat, cache = model.forward(X)
        loss = model.loss(y, y_hat)
        losses.append(loss)
        grads = model.backward(y, cache)

        grad_list = grads["W"] + grads["b"] + (grads["gamma"] + grads["beta"] if model.use_bn else [])
        param_list = model.W + model.b + (model.gamma + model.beta if model.use_bn else [])

        if optimizer == "sgd":
            for p, g in zip(param_list, grad_list):
                p -= lr * g
        elif optimizer == "adam":
            for idx, (p, g) in enumerate(zip(param_list, grad_list)):
                m_t[idx] = b1*m_t[idx] + (1-b1)*g
                v_t[idx] = b2*v_t[idx] + (1-b2)*(g**2)
                m_hat = m_t[idx] / (1 - b1**epoch)
                v_hat = v_t[idx] / (1 - b2**epoch)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return losses


def run():
    X, y = make_correlated_data(rho=0.999999)
    XtX = X.T @ X
    s = np.linalg.svd(XtX, compute_uv=False)
    kappa = s.max() / s.min()
    print(f"Generated data: rho=0.999999, kappa(X^T X) = {kappa:.3e}  (slide target: ~1e7)")

    configs = [
        ("SGD",        dict(use_bn=False), "sgd",  0.5),
        ("Adam",       dict(use_bn=False), "adam", 0.01),
        ("Adam + BN",  dict(use_bn=True),  "adam", 0.01),
    ]

    results = {}
    print(f"\n{'Config':<12} | {'loss@0':>10} | {'loss@50':>10} | {'loss@150':>10} | {'loss@299':>10}")
    print("-" * 65)
    for name, model_kwargs, opt, lr in configs:
        model = MLP([X.shape[1], 16, 16, 1], **model_kwargs, seed=0)
        losses = train(model, X, y, optimizer=opt, lr=lr, epochs=300)
        results[name] = losses
        print(f"{name:<12} | {losses[0]:10.4f} | {losses[50]:10.4f} | {losses[150]:10.4f} | {losses[-1]:10.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in results.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title(f"3-layer MLP on correlated data (kappa~{kappa:.0e})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/experiments/exp5_case_study.png", dpi=130)
    print("\nSaved plot: exp5_case_study.png")

if __name__ == "__main__":
    run()
