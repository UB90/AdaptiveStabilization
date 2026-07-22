"""
Comprehensive plotting. Generates convergence curves, resource usage, 
numerical stability, and bias-variance plots.
Excludes diverged runs automatically.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config

def load_results(path):
    with open(path, "rb") as f: return pickle.load(f)

def _agg(values_per_seed):
    # Pad with NaN; also filter out infs if any (just in case)
    max_len = max(len(v) for v in values_per_seed) if values_per_seed else 0
    arr = np.full((len(values_per_seed), max_len), np.nan)
    for i, v in enumerate(values_per_seed):
        # Replace any inf with nan to avoid overflow warnings
        v_clean = np.array(v)
        v_clean[~np.isfinite(v_clean)] = np.nan
        arr[i, :len(v_clean)] = v_clean
    # nanmean and nanstd ignore nans
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0), max_len

def main():
    results_path = os.path.join(config.RESULTS_DIR, "results.pkl")
    data = load_results(results_path)
    results = data["results"]
    methods = data["config"]["methods"]
    datasets = data["config"]["datasets"]

    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    # 1. Main Convergence Grid
    fig, axes = plt.subplots(nrows=len(datasets), ncols=4, figsize=(22, 4.5 * len(datasets)))
    if len(datasets) == 1: axes = axes[np.newaxis, :]
    
    metric_titles = ["Train Loss (MSE)", "Test Loss (MSE)", "Parameter Error ‖θ−θ*‖", "Eigen-Direction Error"]

    for row, ds_spec in enumerate(datasets):
        ds_name = ds_spec["name"]
        for col, title in enumerate(metric_titles):
            ax = axes[row, col]
            for method in methods:
                # ---- FILTER: only use runs that did NOT diverge ----
                runs = [r for r in results 
                        if r["dataset_name"] == ds_name and r["method"] == method
                        and not r["history"]["diverged"]]
                if not runs:
                    continue   # skip this method if no valid runs
                
                if col == 0: key = "loss"
                elif col == 1: key = "test_loss"
                elif col == 2: key = "param_error"
                elif col == 3: key = "err_top_eig"
                
                per_seed = [r["history"][key] for r in runs]
                mean, std, n = _agg(per_seed)
                x = np.arange(n)
                color = config.METHOD_COLORS.get(method, "gray")
                label = config.METHOD_LABELS.get(method, method)
                ax.plot(x, mean, color=color, label=label, lw=1.6)
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                
                if col == 3:
                    per_seed_bot = [r["history"]["err_bot_eig"] for r in runs]
                    mean_bot, _, _ = _agg(per_seed_bot)
                    ax.plot(x, mean_bot, color=color, lw=1.2, ls="--") 

            ax.set_yscale("log")
            ax.set_title(f"{title}  —  {ds_spec['label']}", fontsize=11)
            ax.set_xlabel("epoch")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            if row == 0 and col == 3: ax.legend(loc="upper right", fontsize=8)
            if col == 3:
                ax.text(0.5, 0.9, "Solid: Top Eig (Steep)\nDashed: Bot Eig (Flat)", transform=ax.transAxes, fontsize=8, verticalalignment='top')

    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "1_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Numerical Stability Plot
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    kappas = [d["kappa"] for d in datasets]
    err_naive = [data["dataset_meta"][d["name"]]["err_naive"] for d in datasets]
    err_pinv = [data["dataset_meta"][d["name"]]["err_pinv"] for d in datasets]
    ax2.plot(kappas, err_naive, "ro-", label="Naive Solve (Normal Eqns)")
    ax2.plot(kappas, err_pinv, "go-", label="Stable Solve (Pseudoinverse)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Condition number κ"); ax2.set_ylabel("Relative parameter error")
    ax2.set_title("Numerical Stability: Direct Solves")
    ax2.grid(True, which="both", ls=":", alpha=0.5); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(config.PLOTS_DIR, "2_numerical_stability.png"), dpi=200)
    plt.close(fig2)

    # 3. Bias-Variance Plot
    fig3, axes3 = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5))
    if len(datasets) == 1: axes3 = [axes3]
    for ax, ds_spec in zip(axes3, datasets):
        ds_name = ds_spec["name"]
        # bias_variance already contains NaNs for methods with no valid runs
        biases = [data["bias_variance"][ds_name][m]["bias_sq"] for m in methods]
        variances = [data["bias_variance"][ds_name][m]["variance"] for m in methods]
        x = np.arange(len(methods)); width = 0.35
        ax.bar(x - width/2, biases, width, label="Bias²")
        ax.bar(x + width/2, variances, width, label="Variance")
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([config.METHOD_LABELS[m].split(" ")[0] for m in methods], rotation=45, ha="right")
        ax.set_title(f"Bias-Variance ({ds_spec['label']})"); ax.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(config.PLOTS_DIR, "3_bias_variance.png"), dpi=200)
    plt.close(fig3)

    # 4. Resource Usage Plot
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
    for method in methods:
        # Also filter diverged runs for time/memory averages (though they should be finite)
        valid_runs = [r for r in results if r["method"] == method and not r["history"]["diverged"]]
        times = []
        rams = []
        for d in datasets:
            ds_name = d["name"]
            runs_ds = [r for r in valid_runs if r["dataset_name"] == ds_name]
            if runs_ds:
                times.append(np.mean([r["wall_clock_s"] for r in runs_ds]))
                rams.append(np.mean([r["ram_mb"] for r in runs_ds]))
            else:
                times.append(np.nan)
                rams.append(np.nan)
        # Plot only if there is at least one point
        if any(np.isfinite(times)):
            axes4[0].plot(kappas, times, "o-", label=config.METHOD_LABELS[method])
        if any(np.isfinite(rams)):
            axes4[1].plot(kappas, rams, "o-", label=config.METHOD_LABELS[method])
    axes4[0].set_xscale("log"); axes4[0].set_xlabel("κ"); axes4[0].set_ylabel("Time (s)"); axes4[0].set_title("Wall Clock Time")
    axes4[1].set_xscale("log"); axes4[1].set_xlabel("κ"); axes4[1].set_ylabel("RAM (MB)"); axes4[1].set_title("Memory Usage")
    axes4[0].legend(fontsize=8)
    fig4.tight_layout()
    fig4.savefig(os.path.join(config.PLOTS_DIR, "4_resource_usage.png"), dpi=200)
    plt.close(fig4)

    print("All core plots saved to plots/ directory.")

if __name__ == "__main__":
    main()