"""
Advanced multi-level plotting:
- Per dataset × per method
- Per dataset (all methods merged)
- Per method (all datasets merged)
- Global averages (bar charts)
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config

# ----------------------------------------------------------------------
#  Helper: load results & aggregation (now matches plot_results.py exactly)
# ----------------------------------------------------------------------
def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _agg(values_per_seed):
    """Pad with NaN; replace inf with NaN before aggregation."""
    max_len = max(len(v) for v in values_per_seed) if values_per_seed else 0
    arr = np.full((len(values_per_seed), max_len), np.nan)
    for i, v in enumerate(values_per_seed):
        v_clean = np.array(v, dtype=float)
        v_clean[~np.isfinite(v_clean)] = np.nan
        arr[i, :len(v_clean)] = v_clean
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), max_len

# ----------------------------------------------------------------------
#  1. Per Dataset × Per Method  (one technique, one κ)
# ----------------------------------------------------------------------
def plot_per_dataset_method(data, methods, datasets):
    base = os.path.join(config.PLOTS_DIR, "per_dataset_method")
    for ds_spec in datasets:
        ds_name = ds_spec["name"]
        for method in methods:
            runs = [r for r in data["results"] 
                    if r["dataset_name"] == ds_name and r["method"] == method
                    and not r["history"]["diverged"]]
            if not runs:
                continue

            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            metric_titles = ["Train Loss", "Test Loss", "Param Error", "Eig Error"]
            keys = ["loss", "test_loss", "param_error", "err_top_eig"]

            for col, (title, key) in enumerate(zip(metric_titles, keys)):
                ax = axes[col]
                per_seed = [r["history"][key] for r in runs]
                mean, std, n = _agg(per_seed)
                x = np.arange(n)
                color = config.METHOD_COLORS.get(method, "gray")
                ax.plot(x, mean, color=color, lw=1.6)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
                
                if key == "err_top_eig":  # add bottom eig dashed
                    per_seed_bot = [r["history"]["err_bot_eig"] for r in runs]
                    mean_bot, _, _ = _agg(per_seed_bot)
                    ax.plot(x, mean_bot, color=color, lw=1.2, ls="--")
                    ax.text(0.02, 0.98, "solid=top, dashed=bot", 
                            transform=ax.transAxes, fontsize=8, va="top")

                ax.set_yscale("log")
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.grid(True, which="both", ls=":", alpha=0.4)

            label = config.METHOD_LABELS.get(method, method)
            fig.suptitle(f"{label}  —  {ds_spec['label']}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            out_dir = os.path.join(base, ds_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{method}.png")
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"  → Per-dataset×method plots saved to {base}/")

# ----------------------------------------------------------------------
#  2. Per Dataset – All Methods Merged  (compare techniques on one κ)
# ----------------------------------------------------------------------
def plot_per_dataset_merged(data, methods, datasets):
    base = os.path.join(config.PLOTS_DIR, "per_dataset_merged")
    for ds_spec in datasets:
        ds_name = ds_spec["name"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        metric_titles = ["Train Loss", "Test Loss", "Param Error", "Eig Error (top)"]
        keys = ["loss", "test_loss", "param_error", "err_top_eig"]

        for col, (title, key) in enumerate(zip(metric_titles, keys)):
            ax = axes[col]
            for method in methods:
                runs = [r for r in data["results"] 
                        if r["dataset_name"] == ds_name and r["method"] == method
                        and not r["history"]["diverged"]]
                if not runs:
                    continue
                per_seed = [r["history"][key] for r in runs]
                mean, std, n = _agg(per_seed)
                x = np.arange(n)
                color = config.METHOD_COLORS.get(method, "gray")
                label = config.METHOD_LABELS.get(method, method)
                ax.plot(x, mean, color=color, label=label, lw=1.6)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
            
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if col == 0:
                ax.legend(loc="upper right", fontsize=8)

        fig.suptitle(f"All Methods — {ds_spec['label']}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(base, f"{ds_name}.png")
        os.makedirs(base, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  → Per-dataset merged plots saved to {base}/")

# ----------------------------------------------------------------------
#  3. Per Method – All Datasets Merged  (one technique across κ)
# ----------------------------------------------------------------------
def plot_per_method_merged(data, methods, datasets):
    base = os.path.join(config.PLOTS_DIR, "per_method_merged")
    # Define dataset colours (if not already in config)
    if not hasattr(config, "DATASET_COLORS"):
        config.DATASET_COLORS = {
            "well_conditioned": "#e41a1c",
            "moderate": "#377eb8",
            "severe": "#4daf4a",
        }
    for method in methods:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        metric_titles = ["Train Loss", "Test Loss", "Param Error", "Eig Error (top)"]
        keys = ["loss", "test_loss", "param_error", "err_top_eig"]

        for col, (title, key) in enumerate(zip(metric_titles, keys)):
            ax = axes[col]
            for ds_spec in datasets:
                ds_name = ds_spec["name"]
                runs = [r for r in data["results"] 
                        if r["dataset_name"] == ds_name and r["method"] == method
                        and not r["history"]["diverged"]]
                if not runs:
                    continue
                per_seed = [r["history"][key] for r in runs]
                mean, std, n = _agg(per_seed)
                x = np.arange(n)
                color = config.DATASET_COLORS.get(ds_name, "gray")
                label = ds_spec["label"]
                ax.plot(x, mean, color=color, label=label, lw=1.6)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            
            ax.set_yscale("log")
            ax.set_title(title)
            ax.set_xlabel("epoch")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if col == 0:
                ax.legend(loc="upper right", fontsize=8)

        label = config.METHOD_LABELS.get(method, method)
        fig.suptitle(f"{label}  —  across all κ", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = os.path.join(base, f"{method}.png")
        os.makedirs(base, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  → Per-method merged plots saved to {base}/")

# ----------------------------------------------------------------------
#  4. Global Averages – Bar Charts (final performance averaged over seeds & datasets)
# ----------------------------------------------------------------------
def plot_global_averages(data, methods, datasets):
    base = os.path.join(config.PLOTS_DIR, "global_averages")
    os.makedirs(base, exist_ok=True)

    metrics = {
        "Final Train Loss": "loss",
        "Final Test Loss": "test_loss",
        "Final Param Error": "param_error",
    }

    for metric_name, key in metrics.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(methods))
        width = 0.6

        values = []
        stds = []
        for method in methods:
            all_final = []
            for ds_spec in datasets:
                ds_name = ds_spec["name"]
                runs = [r for r in data["results"] 
                        if r["dataset_name"] == ds_name and r["method"] == method
                        and not r["history"]["diverged"]]
                for r in runs:
                    if r["history"][key]:  # non-empty list
                        val = r["history"][key][-1]
                        if np.isfinite(val):
                            all_final.append(val)
            values.append(np.nanmean(all_final) if all_final else np.nan)
            stds.append(np.nanstd(all_final) if all_final else np.nan)

        colors = [config.METHOD_COLORS.get(m, "gray") for m in methods]
        ax.bar(x, values, width, yerr=stds, color=colors, capsize=5, 
               error_kw={"ecolor": "black", "linewidth": 1})
        ax.set_xticks(x)
        ax.set_xticklabels([config.METHOD_LABELS.get(m, m) for m in methods], rotation=45, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel(metric_name)
        ax.set_title(f"Global Average over all κ and seeds – {metric_name}")
        ax.grid(True, axis="y", ls=":", alpha=0.4)

        fig.tight_layout()
        out_path = os.path.join(base, f"{key}_global.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  → Global average bar charts saved to {base}/")

# ----------------------------------------------------------------------
#  Main runner
# ----------------------------------------------------------------------
def main():
    results_path = os.path.join(config.RESULTS_DIR, "results.pkl")
    data = load_results(results_path)
    methods = data["config"]["methods"]
    datasets = data["config"]["datasets"]

    print("\n[Advanced Plotting] Generating multi-level PNGs...")
    plot_per_dataset_method(data, methods, datasets)
    plot_per_dataset_merged(data, methods, datasets)
    plot_per_method_merged(data, methods, datasets)
    plot_global_averages(data, methods, datasets)
    print("Done.")

if __name__ == "__main__":
    main()