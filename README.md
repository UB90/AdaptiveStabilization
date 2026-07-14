
# 🧪 Adaptive Stabilization Techniques for Ill-Conditioned Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Code Style](<https://img.shields.io/badge/Code%20Style-Black-black.svg>)](https://github.com/psf/black)

A comprehensive experimental framework for studying how different optimization algorithms behave on ill-conditioned linear regression problems. This project systematically evaluates convergence properties, numerical stability, and bias-variance tradeoffs across multiple condition numbers.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Output &amp; Artifacts](#output--artifacts)
- [Experimental Design](#experimental-design)
- [Methods Compared](#methods-compared)
- [Results Visualization](#results-visualization)
- [Technical Implementation](#technical-implementation)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project investigates how different optimization algorithms cope with **ill-conditioned** linear regression problems, where the condition number κ (ratio of largest to smallest singular value of the design matrix) varies from well-conditioned (κ=10) to severely ill-conditioned (κ=10⁷). The framework includes:

- **Synthetic data generation** with exact control over condition numbers
- **Six optimization methods** including vanilla SGD, momentum, Adam, ridge regression, batch normalization, and AdamW
- **Comprehensive tracking** of convergence, resource usage, and numerical stability
- **Bias-variance decomposition** across different algorithms
- **Automated visualization** of all experimental results

## ✨ Key Features

- **Exact condition number control**: Synthetic data generation ensures precise condition numbers 【turn0search0】【turn0search3】
- **Multiple stabilization techniques**: Comparison of classical and modern optimization approaches
- **Resource monitoring**: Real-time tracking of CPU and memory usage during training
- **Numerical stability analysis**: Direct comparison of naive vs. stable linear system solves
- **Comprehensive metrics**: Training loss, test loss, parameter error, eigen-direction errors
- **Reproducible experiments**: Fixed seeds for data generation and model initialization

## 🏗️ Project Architecture

```mermaid
flowchart TD
    classDef largeBox fill:#f9f9f9,stroke:#333,stroke-width:2px,padding:20px,font-size:14px;
    classDef phaseBox fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,padding:15px,font-size:13px;
    classDef saveBox fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,padding:15px,font-size:13px;

    Start([<b>START EXPERIMENT</b><br>Run: python train.py]):::largeBox

    subgraph Phase1 [Phase 1: Configuration & Data Generation]
        direction TB
        Config[<b>config.py</b><br>Define datasets: κ = 10, 10^4, 10^7<br>Set N_SAMPLES = 1000, N_FEATURES = 50<br>Define methods and hyperparameters]:::phaseBox
        DataGen[<b>data_generator.py</b><br>Generate synthetic data via SVD/QR<br>Ensure exact condition numbers<br>Compute ground truth theta_star]:::phaseBox
        SaveData[<b>Save to data/*.pt</b><br>Persist X_train, y_train, X_test, y_test<br>Save V, sigma, theta_star]:::saveBox
        Config --> DataGen --> SaveData
    end

    subgraph Phase2 [Phase 2: Training & Resource Tracking]
        direction TB
        LoopInit[<b>Loop over Datasets</b><br>well_conditioned, moderate, severe]:::phaseBox
        LoopMethod[<b>Loop over 6 Methods</b><br>vanilla_sgd, momentum, adam,<br>ridge_l2, batchnorm_linear, adamw]:::phaseBox
        LoopSeed[<b>Loop over 5 Init Seeds</b><br>Ensures statistical robustness<br>for Bias/Variance computation]:::phaseBox
        Train[<b>optimizers.py</b><br>Execute training loop<br>Track Train Loss, Test Loss, Param Error<br>Track Eigen-direction Errors]:::phaseBox
        Track[<b>psutil Monitoring</b><br>Record Wall-Clock Time<br>Track CPU Percentage<br>Track RAM Usage in MB]:::phaseBox
        SaveWeights[<b>Save to weights/</b><br>torch.save(model.state_dict())<br>Format: dataset/method_seedN.pth]:::saveBox
        LoopInit --> LoopMethod --> LoopSeed --> Train --> Track --> SaveWeights
    end

    subgraph Phase3 [Phase 3: Analytics & Aggregation]
        direction TB
        BiasVar[<b>Compute Bias & Variance</b><br>Aggregate final theta across 5 seeds<br>Calculate Bias² and Variance]:::phaseBox
        NumStab[<b>Numerical Stability Baseline</b><br>Naive Solve: Inverse of Normal Equations<br>Stable Solve: Pseudoinverse]:::phaseBox
        SaveResults[<b>Save to results/results.pkl</b><br>Pickle dump all histories, metrics,<br>and resource usage data]:::saveBox
        BiasVar --> SaveResults
        NumStab --> SaveResults
    end

    subgraph Phase4 [Phase 4: Visualization & Plotting]
        direction TB
        Plot[<b>plot_results.py</b><br>Load results.pkl<br>Generate 4 comprehensive plot sets]:::phaseBox
        Out1[<b>1_convergence.png</b><br>4x3 Grid: Train/Test Loss,<br>Param Err, Eigen Err vs Epochs]:::phaseBox
        Out2[<b>2_numerical_stability.png</b><br>Direct Solves Error vs κ]:::phaseBox
        Out3[<b>3_bias_variance.png</b><br>Method Comparison across κ]:::phaseBox
        Out4[<b>4_resource_usage.png</b><br>Time and RAM vs κ]:::phaseBox
        Plot --> Out1 & Out2 & Out3 & Out4
    end

    Start --> Phase1
    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    
    End([<b>EXPERIMENT COMPLETE</b><br>Check ./plots directory]):::largeBox
    Phase4 --> End
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib
- psutil (for resource monitoring)

### Setup

```bash
# Clone the repository
git clone https://github.com/UB90/AdaptiveStabilization.git
cd AdaptiveStabilization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
psutil>=5.8.0
pickle-mixin>=1.0.0  # For saving/loading results
```

## 🚀 Usage

### Quick Start

```bash
# Run the complete experiment
python train.py

# Generate visualizations
python plot_results.py
```

### Detailed Execution

<details>
<summary>🔧 Advanced Configuration</summary>

Edit `config.py` to customize experiments:

```python
# Modify condition numbers
DATASETS = [
    {"name": "well_conditioned", "kappa": 1e1,  "label": r"$\kappa=10$"},
    {"name": "moderate",         "kappa": 1e4,  "label": r"$\kappa=10^4$"},
    {"name": "severe",           "kappa": 1e7,  "label": r"$\kappa=10^7$"},
]

# Adjust sample size and dimensions
N_SAMPLES      = 1000  # Training samples
N_TEST_SAMPLES = 200   # Test samples
N_FEATURES     = 50    # Problem dimensionality

# Change optimization methods
METHODS = [
    "vanilla_sgd", "momentum_sgd", "adam", 
    "ridge_l2", "batchnorm_linear", "adamw"
]

# Modify training duration
EPOCHS_PER_KAPPA = 0.5  # Scales with condition number
EPOCHS_FLOOR     = 1000
EPOCHS_CAP       = 50_000  # Prevent multi-day runs
```

</details>

### Running Specific Components

```bash
# Generate only datasets
python -c "import data_generator; data_generator.generate_all_datasets()"

# Train specific method
python -c "
import config
import data_generator
from optimizers import train_method

datasets = data_generator.generate_all_datasets()
for ds_name, ds in datasets.items():
    history, model = train_method('adam', ds, init_seed=0)
    print(f'{ds_name}: Final loss = {history[\"loss\"][-1]:.4e}')
"

# Load and inspect saved results
python -c "
import pickle
import config

with open(f'{config.RESULTS_DIR}/results.pkl', 'rb') as f:
    data = pickle.load(f)
  
print(f'Methods: {data[\"config\"][\"methods\"]}')
print(f'Datasets: {[d[\"name\"] for d in data[\"config\"][\"datasets\"]]}')
"
```

## 💾 Output & Artifacts

### ✅ Yes, this project saves everything!

The project automatically saves the following artifacts:

| Artifact Type           | Location                    | Format         | Description                                     |
| ----------------------- | --------------------------- | -------------- | ----------------------------------------------- |
| **Datasets**      | `data/`                   | `.pt` files  | Synthetic datasets with exact condition numbers |
| **Model Weights** | `weights/{dataset_name}/` | `.pth` files | Trained model parameters for each run           |
| **Results**       | `results/results.pkl`     | Pickle file    | Complete experimental results                   |
| **Plots**         | `plots/`                  | PNG files      | All visualizations                              |

### Directory Structure After Execution

```
adaptive-stabilization/
├── data/
│   ├── well_conditioned.pt
│   ├── moderate.pt
│   └── severe.pt
├── weights/
│   ├── well_conditioned/
│   │   ├── vanilla_sgd_seed0.pth
│   │   ├── vanilla_sgd_seed1.pth
│   │   └── ...
│   ├── moderate/
│   │   └── ...
│   └── severe/
│       └── ...
├── results/
│   └── results.pkl
├── plots/
│   ├── 1_convergence.png
│   ├── 2_numerical_stability.png
│   ├── 3_bias_variance.png
│   └── 4_resource_usage.png
└── (source code files)
```

### What Gets Saved?

<details>
<summary>📖 Detailed Save Mechanisms</summary>

1. **Datasets** (`data_generator.py`):

   ```python
   torch.save(ds, os.path.join(config.DATA_DIR, f"{spec['name']}.pt"))
   ```

   Saves complete dataset including: X_train, y_train, X_test, y_test, theta_star, V, sigma, condition numbers
2. **Model Weights** (`models.py`):

   ```python
   def save_model(model, path):
       os.makedirs(os.path.dirname(path), exist_ok=True)
       torch.save(model.state_dict(), path)
   ```

   Saves model parameters using PyTorch's recommended `state_dict` approach 【turn0search5】【turn0search7】
3. **Results** (`train.py`):

   ```python
   with open(out_path, "wb") as f:
       pickle.dump({
           "config": {...},
           "dataset_meta": {...},
           "bias_variance": {...},
           "results": results,
       }, f)
   ```

   Saves all experimental results including training histories, final parameters, resource usage

</details>

## 🧪 Experimental Design

### Condition Numbers Tested

| Name             | κ   | Description                                           |
| ---------------- | ---- | ----------------------------------------------------- |
| Well-conditioned | 10   | Easy problem, fast convergence expected               |
| Moderate         | 10⁴ | Moderately ill-conditioned, some methods may struggle |
| Severe           | 10⁷ | Severely ill-conditioned, tests numerical stability   |

### Metrics Tracked

- **Training Metrics**: Loss, gradient norm, parameter error
- **Test Metrics**: Test loss, generalization gap
- **Eigen-direction Errors**: Errors in top (steep) and bottom (flat) eigen-directions
- **Resource Usage**: CPU time, memory consumption, wall clock time
- **Bias-Variance Decomposition**: For each method across datasets

## 📊 Methods Compared

| Method                        | Description                                 | Key Hyperparameter                      |
| ----------------------------- | ------------------------------------------- | --------------------------------------- |
| **Vanilla SGD**         | Standard stochastic gradient descent        | Learning rate:`0.9 * 2n/σ²_max`     |
| **Momentum SGD**        | SGD with momentum                           | β:`(√κ - 1)/(√κ + 1))²`         |
| **Adam**                | Adaptive moment estimation                  | Learning rate: 0.1 (scale-invariant)    |
| **Ridge L2**            | SGD with L2 regularization                  | λ:`σ²_max(1 - √κ/κ)/(√κ - 1)` |
| **BatchNorm-on-Linear** | Batch normalization applied to linear model | Learning rate: 0.1                      |
| **AdamW**               | Adam with decoupled weight decay            | Weight decay: Same as Ridge λ          |

## 📈 Results Visualization

The project generates four comprehensive visualization sets:

### 1. Convergence Grid (`1_convergence.png`)

- 4×3 grid of plots (4 metrics × 3 condition numbers)
- Shows training/test loss, parameter error, eigen-direction errors
- Includes standard deviation bands across initialization seeds

### 2. Numerical Stability (`2_numerical_stability.png`)

- Compares naive inverse vs. pseudoinverse solutions
- Shows relative error as function of condition number
- Demonstrates importance of stable numerical methods 【turn0search4】

### 3. Bias-Variance Decomposition (`3_bias_variance.png`)

- Bar plots showing bias² and variance for each method
- Separate panels for each condition number
- Logarithmic scale for better visualization

### 4. Resource Usage (`4_resource_usage.png`)

- Wall clock time and memory usage vs. condition number
- Separate lines for each optimization method
- Helps identify computational efficiency tradeoffs

## 🔧 Technical Implementation

### Data Generation Strategy

The synthetic data generation ensures exact condition numbers through:

1. **Singular value specification**: `σ_max = 1.0`, `σ_min = σ_max/√κ`
2. **Orthogonal matrices**: Using QR decomposition for random orthogonal matrices
3. **Verification**: Actual condition number checked against target

```python
G = X_np.T @ X_np
kappa_actual = float(np.linalg.cond(G))
if abs(kappa_actual - kappa) / kappa > 1e-6:
    raise RuntimeError("κ sanity check failed.")
```

### Numerical Stability Considerations

- **Naive solution**: `θ_naive = (XᵀX)⁻¹Xᵀy` - unstable for high κ
- **Stable solution**: `θ_pinv = X⁺y` (pseudoinverse) - numerically stable
- **Regularization**: Ridge regression adds λI to improve conditioning 【turn0search4】

### Resource Monitoring

The project uses `psutil` for accurate resource tracking 【turn0search12】【turn0search14】:

```python
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)
# ... training code ...
cpu_usage = process.cpu_percent(interval=None)
mem_after = process.memory_info().rss / (1024 * 1024)
```

## ❓ Troubleshooting

<details>
<summary>🔧 Common Issues</summary>

### Q: Training takes too long for severe condition numbers

**A**: Reduce `EPOCHS_CAP` in `config.py` or use fewer initialization seeds.

### Q: Out of memory errors

**A**: Reduce `N_SAMPLES` or `N_FEATURES` in `config.py`.

### Q: Divergence errors

**A**: This is expected for some methods at high condition numbers. The framework automatically detects and handles divergence.

### Q: How to resume interrupted training?

**A**: The framework doesn't currently support resuming. However, you can modify `train.py` to check for existing results and skip completed runs.

### Q: Different Python/PyTorch versions

**A**: The code is tested with Python 3.8+ and PyTorch 2.0+. Minor adjustments may be needed for older versions.

</details>

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{adaptive_stabilization_2026,
  title={Adaptive Stabilization Techniques for Ill-Conditioned Linear Regression},
  author={Utsab Baral, Aman Sapkota, Kushagra Poudel},
  year={2026},
  publisher={GitHub},
  url={https://github.com/UB90/AdaptiveStabilization}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- psutil developers for cross-platform system monitoring
- Numerical linear algebra community for insights on condition numbers and stability

---

**Status**: Active development
**Last Updated**: 2026-07-14
**Version**: 2.0.0

> **Note**: This README provides comprehensive documentation for the adaptive stabilization project. For specific implementation details, refer to the source code files. The project saves all datasets, model weights, and results automatically as described in the Output & Artifacts section.
