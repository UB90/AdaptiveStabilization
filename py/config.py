"""
Adaptive Stabilization Techniques — Experiment Configuration.
"""
import torch
import os

torch.set_default_dtype(torch.float64)
SEED = 0
torch.manual_seed(SEED)

DATASETS = [
    {"name": "well_conditioned", "kappa": 1e1,  "label": r"$\kappa=10$"},
    {"name": "moderate",         "kappa": 1e4,  "label": r"$\kappa=10^4$"},
    {"name": "severe",           "kappa": 1e7,  "label": r"$\kappa=10^7$"},
]

N_SAMPLES      = 1000
N_TEST_SAMPLES = 200
N_FEATURES     = 50
SIGMA_MAX      = 1.0
NOISE_STD      = 1e-3
DATA_SEED_BASE = 10_000
DATA_VARIANT   = "scale"

METHODS = [
    "vanilla_sgd", "momentum_sgd", "adam", "ridge_l2", "batchnorm_linear", "adamw"
]

EPOCHS_PER_KAPPA = 0.5
EPOCHS_FLOOR     = 1000
EPOCHS_CAP       = 50_000  # Reduced to prevent multi-day runtimes

def epochs_for_kappa(kappa: float) -> int:
    return int(min(EPOCHS_CAP, EPOCHS_PER_KAPPA * kappa + EPOCHS_FLOOR))

INIT_SEEDS = [0, 1, 2, 3, 4]

def lr_for_method(method: str, kappa: float) -> float:
    # FIX: L_max = (1/n) * sigma_max^2, so lr_max = 2 / L_max = 2n / sigma_max^2
    lr_max = 2.0 * N_SAMPLES / (SIGMA_MAX ** 2)
    if method == "vanilla_sgd":       return 0.9 * lr_max
    if method == "momentum_sgd":      return 0.95 * lr_max
    if method == "adam":              return 0.1  # Adam step is scale-invariant, 0.1 is fast
    if method == "ridge_l2":          return 0.9 * lr_max
    if method == "batchnorm_linear":  return 0.1
    if method == "adamw":             return 0.1
    raise ValueError(f"Unknown method: {method}")

def momentum_beta_for_kappa(kappa: float) -> float:
    sq = kappa ** 0.5
    return ((sq - 1.0) / (sq + 1.0)) ** 2

def ridge_lambda_for_kappa(kappa: float) -> float:
    lam_max = SIGMA_MAX ** 2
    lam_min = lam_max / kappa
    target_kappa_ridge = max(kappa ** 0.5, 10.0)
    lam = (lam_max - target_kappa_ridge * lam_min) / (target_kappa_ridge - 1.0)
    return max(lam, 1e-6)

def weight_decay_for_kappa(kappa: float) -> float:
    return ridge_lambda_for_kappa(kappa)

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS   = 1e-8

# Fallback for interactive environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
DATA_DIR    = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

METHOD_COLORS = {
    "vanilla_sgd": "#1f77b4", "momentum_sgd": "#ff7f0e", "adam": "#2ca02c",
    "ridge_l2": "#d62728", "batchnorm_linear": "#9467bd", "adamw": "#8c564b",
}
METHOD_LABELS = {
    "vanilla_sgd": "Vanilla SGD", "momentum_sgd": "Momentum (β*=f(κ))",
    "adam": "Adam", "ridge_l2": "SGD + L2 (Ridge)", "batchnorm_linear": "BN-on-Linear",
    "adamw": "AdamW (decoupled WD)",
}