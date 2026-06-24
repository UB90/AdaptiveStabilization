"""
EXP 1 — Condition Number → Error Amplification
Claim (slide 3): delta_x/||x|| <= kappa(A) * delta_b/||b||
Test: build A with controlled kappa, perturb b, measure actual ratio vs bound.
"""
import numpy as np

np.random.seed(0)

def build_matrix_with_kappa(n, kappa):
    """Construct A with exact condition number via SVD: A = U S V^T."""
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))
    s = np.geomspace(kappa, 1.0, n)   # singular values from kappa down to 1
    A = U @ np.diag(s) @ V.T
    return A, s

def run(n=50, kappas=(1, 10, 1e3, 1e6, 1e9), n_trials=20, perturb_scale=1e-6):
    print(f"{'kappa(A)':>12} | {'bound: k*db/b':>16} | {'actual dx/x (max over trials)':>30}")
    print("-" * 65)
    for kappa in kappas:
        A, s = build_matrix_with_kappa(n, kappa)
        kappa_actual = s.max() / s.min()
        x_true = np.random.randn(n)
        b = A @ x_true

        ratios = []
        for _ in range(n_trials):
            db = np.random.randn(n)
            db = db / np.linalg.norm(db) * perturb_scale * np.linalg.norm(b)
            x_perturbed = np.linalg.solve(A, b + db)
            dx_rel = np.linalg.norm(x_perturbed - x_true) / np.linalg.norm(x_true)
            db_rel = np.linalg.norm(db) / np.linalg.norm(b)
            ratios.append(dx_rel / db_rel)

        bound = kappa_actual
        actual_max = max(ratios)
        print(f"{kappa_actual:12.3e} | {bound:16.3e} | {actual_max:30.3e}  "
              f"{'<= bound OK' if actual_max <= bound * 1.0001 else 'VIOLATES BOUND'}")

if __name__ == "__main__":
    run()
