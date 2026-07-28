"""
Synthetic data generation with exact condition numbers. Saves data to disk.
"""
import numpy as np
import torch
import os
import config

def _haar_orthogonal(rows, cols, rng):
    # FIX: Added assertion to prevent silent shape errors
    assert rows >= cols, "Haar orthogonal generation requires rows >= cols"
    A = rng.standard_normal((rows, cols))
    Q, _ = np.linalg.qr(A)
    signs = np.sign(np.diag(_))
    signs[signs == 0] = 1.0
    return Q * signs[None, :]

def make_dataset(kappa, variant="scale", data_seed=None):
    n, d = config.N_SAMPLES, config.N_FEATURES
    n_test = config.N_TEST_SAMPLES
    sigma_max = config.SIGMA_MAX
    sigma_min = sigma_max / np.sqrt(kappa)
    sigma = np.geomspace(sigma_max, sigma_min, num=d)

    if variant == "scale":
        rng = np.random.default_rng(data_seed + 1)
        V = np.eye(d) + 1e-3 * rng.standard_normal((d, d))
        V, _ = np.linalg.qr(V)
    elif variant == "correlation":
        rng = np.random.default_rng(data_seed + 1)
        V = _haar_orthogonal(d, d, rng)

    rng_u = np.random.default_rng(data_seed + 2)
    U = _haar_orthogonal(n, d, rng_u)
    X_np = U @ np.diag(sigma) @ V.T

    theta_rng = np.random.default_rng(config.DATA_SEED_BASE + 999)
    theta_star_np = theta_rng.standard_normal(d)
    theta_star_np /= np.linalg.norm(theta_star_np)

    eps_rng = np.random.default_rng(data_seed + 3)
    noise = eps_rng.standard_normal(n) * config.NOISE_STD
    y_np = X_np @ theta_star_np + noise

    rng_test = np.random.default_rng(data_seed + 4)
    U_test = _haar_orthogonal(n_test, d, rng_test)
    X_test_np = U_test @ np.diag(sigma) @ V.T
    y_test_np = X_test_np @ theta_star_np + eps_rng.standard_normal(n_test) * config.NOISE_STD

    G = X_np.T @ X_np
    kappa_actual = float(np.linalg.cond(G))
    if abs(kappa_actual - kappa) / kappa > 1e-6:
        raise RuntimeError("κ sanity check failed.")

    return {
        "X_train": torch.from_numpy(X_np),
        "y_train": torch.from_numpy(y_np),
        "X_test": torch.from_numpy(X_test_np),
        "y_test": torch.from_numpy(y_test_np),
        "theta_star": torch.from_numpy(theta_star_np),
        "V": torch.from_numpy(V),
        "sigma": torch.from_numpy(sigma),
        "kappa_target": float(kappa),
        "kappa_actual": kappa_actual,
    }

def generate_all_datasets():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    datasets = {}
    for i, spec in enumerate(config.DATASETS):
        ds = make_dataset(spec["kappa"], variant=config.DATA_VARIANT, data_seed=config.DATA_SEED_BASE + i)
        ds["name"] = spec["name"]
        ds["label"] = spec["label"]
        datasets[spec["name"]] = ds
        
        torch.save(ds, os.path.join(config.DATA_DIR, f"{spec['name']}.pt"))
        print(f"  [data] {spec['name']:>16s}  κ={ds['kappa_actual']:.4e}  (saved to data/{spec['name']}.pt)")
    return datasets