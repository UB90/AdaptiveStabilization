"""
EXP 2 — Ridge Regression as Eigenvalue Shift Preconditioning
Claim (slide 5): kappa(X^T X + lambda*I) = (sigma_max + lambda) / (sigma_min + lambda)
Test: build multicollinear X, sweep lambda, reproduce the kappa-vs-lambda table.
"""
import numpy as np

np.random.seed(42)

def build_multicollinear_X(n_samples=500, n_features=20, n_redundant_corr=0.999):
    """Features highly correlated -> near-singular X^T X."""
    base = np.random.randn(n_samples, 1)
    X = base @ np.ones((1, n_features))
    X += np.random.randn(n_samples, n_features) * (1 - n_redundant_corr)
    X -= X.mean(axis=0)
    return X

def kappa_of(M):
    s = np.linalg.svd(M, compute_uv=False)
    return s.max() / s.min()

def run():
    X = build_multicollinear_X()
    XtX = X.T @ X
    I = np.eye(XtX.shape[0])
    s_XtX = np.linalg.svd(XtX, compute_uv=False)
    sigma_max, sigma_min = s_XtX.max(), s_XtX.min()

    print(f"sigma_max(X^T X) = {sigma_max:.4e}   sigma_min(X^T X) = {sigma_min:.4e}")
    print(f"{'lambda':>10} | {'predicted k=(smax+l)/(smin+l)':>30} | {'measured kappa':>16} | match")
    print("-" * 75)
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        predicted = (sigma_max + lam) / (sigma_min + lam)
        measured = kappa_of(XtX + lam * I)
        match = "OK" if abs(predicted - measured) / predicted < 1e-6 else "MISMATCH"
        print(f"{lam:10.3f} | {predicted:30.4e} | {measured:16.4e} | {match}")

    # Coefficient stability check: does ridge actually stabilize the solution?
    print("\nCoefficient norm ||x*|| vs lambda (explosion check):")
    y = X[:, 0] * 3.0 + np.random.randn(X.shape[0]) * 0.01
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        try:
            coef = np.linalg.solve(XtX + lam * I, X.T @ y)
            print(f"  lambda={lam:6.3f} -> ||coef|| = {np.linalg.norm(coef):.4e}")
        except np.linalg.LinAlgError:
            print(f"  lambda={lam:6.3f} -> SINGULAR, no solution")

if __name__ == "__main__":
    run()
