# Project Progress Update: Adaptive Stabilization (Phases 3–5)

**Course**: AIMA 203 – Numerical Methods  
**Institution**: Kathmandu University, Nepal  
**Date**: May 2026  
**Authors**: Kushagra Poudel, Aman Sapkota, Utsab Baral

---

## 📌 Overview of Updates

We have implemented the remaining phases (Phases 3, 4, and 5) of the **Adaptive Stabilization Techniques for Ill-Conditioned Problems in ML** project. 

The baseline experiments established that Vanilla Stochastic Gradient Descent (SGD) fails to converge efficiently when features are highly multicollinear (condition number $\kappa \approx 2.04 \times 10^4$), showing severe loss oscillations and slow parameter updates. To resolve this, we benchmarked three stabilization techniques:
1. **L2 (Ridge) Regularization** (Phase 3)
2. **Adam Optimizer** (Phase 4)
3. **Batch Normalization** (Phase 5)

---

## 🛠 New Files Added

| File Path | Description |
| :--- | :--- |
| **[`generate_results.py`](generate_results.py)** | Script that runs all training benchmarks, sets global seeds, saves results, and generates the comparison loss plot. |
| **[`generate_slides.py`](generate_slides.py)** | Programmatically generates the 7-slide progress report PowerPoint deck (`adaptive_stabilization_progress.pptx`). |
| **[`generate_pdf_report.py`](generate_pdf_report.py)** | Programmatically compiles the publication-quality, 6-page project progress report (`adaptive_stabilization_report.pdf`). |
| **[`comparison_loss_curves.png`](comparison_loss_curves.png)** | High-resolution plot overlaying loss curves on a log-scale to compare all experimental runs. |
| **[`experimental_results.npz`](experimental_results.npz)** | Compressed NumPy array data containing loss histories for future analysis. |

---

## 📊 Experimental Results & Benchmarks

The MLP network (10 → 64 → 64 → 1) was trained for 300 epochs under identical weight initializations for all setups. The results are summarized below:

| Configuration | Learning Rate | Final MSE Loss | Convergence Behavior |
| :--- | :---: | :---: | :--- |
| **Vanilla SGD (Control)** | 0.001 | 0.5198 | Slow and oscillating heavily due to steep ravine walls. |
| **Ridge Regularization** | 0.001 | 0.4812 | Shifts eigenvalues to stabilize, but convergence remains restricted. |
| **Batch Normalization (Low)** | 0.001 | 0.4501 | Stable updates, but slow step progression. |
| **Batch Normalization (High)** | 0.010 | 0.2815 | Centered activations bound the Jacobian, permitting 10x higher stable learning rate. |
| **Adam Optimizer** | 0.005 | **0.0048** | **Extremely rapid convergence (<20 epochs)** due to coordinate-wise diagonal preconditioning. |

### Key Takeaways
1. **Adam is the clear winner**: By dividing updates by the square root of the running average of squared gradients, Adam acts as a dynamic diagonal Hessian preconditioner, neutralizing ravine scaling and converging in under 20 epochs.
2. **Batch Normalization enables high learning rates**: Standardizing activations bounds the singular values of the Jacobian, smoothing the loss surface. This allows us to scale up the learning rate to `0.01` safely (which would normally explode under Vanilla SGD), achieving faster convergence.
3. **Ridge Regularization stabilizes computation**: Adding L2 weight decay shifts eigenvalues ($\kappa' = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$), smoothing loss contours but not correcting step scale disparities.

---

## 🚀 How to Run the Scripts

Ensure you have the required dependencies installed:
```bash
pip install torch matplotlib python-pptx reportlab numpy
```

### 1. Re-run Training Benchmarks
To run the training loops and generate the comparison loss plot:
```bash
python generate_results.py
```
*This will output `experimental_results.npz` and `comparison_loss_curves.png` in the local directory.*

### 2. Generate Presentation Slides
To compile the widescreen slide deck:
```bash
python generate_slides.py
```
*This saves `adaptive_stabilization_progress.pptx` in the parent directory (`../`).*

### 3. Generate PDF Report
To compile the multi-page formal report:
```bash
python generate_pdf_report.py
```
*This saves `adaptive_stabilization_report.pdf` in the parent directory (`../`).*
