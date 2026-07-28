"""
Main orchestrator. Runs training, saves data/weights, tracks resources, 
computes bias/variance and numerical solve stability.
"""
import os
import pickle
import time
import torch
import psutil
import numpy as np
import config
import data_generator
from optimizers import train_method
import models


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.WEIGHTS_DIR, exist_ok=True)
    process = psutil.Process(os.getpid())

    print("=" * 78); print("TRAINING PIPELINE"); print("=" * 78)

    print("\n[1/4] Generating & saving datasets…")
    datasets = data_generator.generate_all_datasets()

    print("\n[2/4] Computing numerical stability baselines (Naive vs Stable)…")
    for ds_name, ds in datasets.items():
        X, y = ds["X_train"], ds["y_train"]
        theta_star = ds["theta_star"]
        try:
            theta_naive = torch.linalg.inv(X.T @ X) @ X.T @ y
            ds["err_naive"] = (theta_naive - theta_star).norm().item() / theta_star.norm().item()
        except Exception:
            ds["err_naive"] = float('nan')
            
        theta_pinv = torch.linalg.pinv(X) @ y
        ds["err_pinv"] = (theta_pinv - theta_star).norm().item() / theta_star.norm().item()
        print(f"  {ds_name}: Naive err={ds['err_naive']:.4e}, Stable err={ds['err_pinv']:.4e}")

    print("\n[3/4] Running training…")
    results = []
    t_start = time.time()
    n_runs = len(config.DATASETS) * len(config.METHODS) * len(config.INIT_SEEDS)
    done = 0

    for ds_spec in config.DATASETS:
        ds_name = ds_spec["name"]
        ds = datasets[ds_name]
        kappa = ds["kappa_target"]
        epochs = config.epochs_for_kappa(kappa)

        for method in config.METHODS:
            for init_seed in config.INIT_SEEDS:
                process.cpu_percent(interval=None)
                mem_before = process.memory_info().rss / (1024 * 1024)
                
                t0 = time.time()
                history, model = train_method(method, ds, init_seed)
                elapsed = time.time() - t0
                
                cpu_usage = process.cpu_percent(interval=None)
                mem_after = process.memory_info().rss / (1024 * 1024)
                
                final_theta = model.get_theta().cpu().numpy()
                
                w_path = os.path.join(config.WEIGHTS_DIR, ds_name, f"{method}_seed{init_seed}.pth")
                models.save_model(model, w_path)  # creates subdirectory automatically
                
                results.append({
                    "method": method, "dataset_name": ds_name, "kappa": kappa,
                    "init_seed": init_seed, "epochs": epochs,
                    "history": history, "final_theta": final_theta,
                    "wall_clock_s": elapsed, "cpu_percent": cpu_usage,
                    "ram_mb": max(mem_before, mem_after), "weights_path": w_path
                })
                done += 1
                print(f"    [{done}/{n_runs}] {ds_name} | {method} | seed={init_seed}  done ({elapsed:.1f}s)")

    print(f"\n  Total training time: {time.time()-t_start:.1f}s")

    print("\n[4/4] Aggregating Bias/Variance and saving results…")
    bias_variance = {}
    for ds_spec in config.DATASETS:
        ds_name = ds_spec["name"]
        theta_star = datasets[ds_name]["theta_star"].cpu().numpy()
        bias_variance[ds_name] = {}
        for method in config.METHODS:
            # ---------- FIX: filter out diverged runs and non-finite values ----------
            final_thetas = []
            for r in results:
                if (r["dataset_name"] == ds_name and r["method"] == method
                        and not r["history"]["diverged"]
                        and np.all(np.isfinite(r["final_theta"]))):
                    final_thetas.append(r["final_theta"])
            # -----------------------------------------------------------------------
            if final_thetas:
                thetas_np = np.array(final_thetas)
                mean_theta = thetas_np.mean(axis=0)
                bias = np.linalg.norm(mean_theta - theta_star)
                variance = np.trace(np.cov(thetas_np, rowvar=False))
                bias_variance[ds_name][method] = {"bias_sq": bias**2, "variance": variance}
            else:
                bias_variance[ds_name][method] = {"bias_sq": np.nan, "variance": np.nan}
                print(f"  Warning: No valid runs for {ds_name} / {method}")

    out_path = os.path.join(config.RESULTS_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({
            "config": {
                "datasets": config.DATASETS, "methods": config.METHODS,
                "init_seeds": config.INIT_SEEDS
            },
            "dataset_meta": {
                name: {
                    "kappa_target": ds["kappa_target"], "kappa_actual": ds["kappa_actual"],
                    "err_naive": ds["err_naive"], "err_pinv": ds["err_pinv"]
                } for name, ds in datasets.items()
            },
            "bias_variance": bias_variance,
            "results": results,
        }, f)
    print(f"  → {out_path}")

if __name__ == "__main__":
    main()