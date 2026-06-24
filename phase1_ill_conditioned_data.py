import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# -------------------------------------------
# 1. Generate features with multicollinearity
# -------------------------------------------
n_samples = 200
n_features = 10

true_weights = np.zeros(n_features)
true_weights[:3] = [1.5, -2.0, 0.8]

features = np.random.randn(n_samples, n_features)

# Make two pairs of features highly correlated
features[:, 1] = features[:, 0] * 0.95 + np.random.normal(0, 0.02, n_samples)
features[:, 3] = features[:, 2] * 0.97 + np.random.normal(0, 0.015, n_samples)

target = features @ true_weights + np.random.normal(0, 0.1, n_samples)

# -------------------------------------------
# 2. Compute condition number of X^T X
# -------------------------------------------
gram = features.T @ features
eigenvalues = np.linalg.eigvalsh(gram)
eigenvalues = np.sort(eigenvalues)
condition_number = eigenvalues[-1] / eigenvalues[0]

print(f"Condition number κ = {condition_number:.2e}")
print(f"Smallest eigenvalue: {eigenvalues[0]:.4e}")
print(f"Largest eigenvalue : {eigenvalues[-1]:.4f}")

# Show that direct inversion fails, but lstsq (SVD) works
try:
    coef_direct = np.linalg.inv(gram) @ features.T @ target
    print("\nDirect inversion (first 5):", coef_direct[:5].round(3))
except np.linalg.LinAlgError:
    print("\nDirect inversion failed – matrix is numerically singular.")

coef_stable, _, rank, _ = np.linalg.lstsq(features, target, rcond=None)
print("Stable lstsq (first 5):", coef_stable[:5].round(3))
print(f"Matrix rank: {rank} (should be {n_features})")

# -------------------------------------------
# 3. Visualise eigenvalue spread
# -------------------------------------------
plt.figure(figsize=(8,4))
plt.bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
plt.yscale('log')
plt.axhline(y=eigenvalues[0], color='red', linestyle='--', label=f'min = {eigenvalues[0]:.2e}')
plt.axhline(y=eigenvalues[-1], color='green', linestyle='--', label=f'max = {eigenvalues[-1]:.2f}')
plt.title(f'Eigenvalue spectrum of XᵀX  (κ = {condition_number:.2e})')
plt.xlabel('Eigenvalue index')
plt.ylabel('Eigenvalue (log scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('eigenvalue_spectrum.png', dpi=150)
plt.show()

# -------------------------------------------
# 4. Save for Phase 2
# -------------------------------------------
np.savez('data_ill_conditioned.npz', X=features, y=target, true_w=true_weights,
         eigvals=eigenvalues, kappa=condition_number)
print("\nData saved to 'data_ill_conditioned.npz'")