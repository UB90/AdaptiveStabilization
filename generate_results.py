import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # If CUDA is available, set seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Load the ill‑conditioned data
data = np.load('data_ill_conditioned.npz')
X = data['X']
y = data['y']
kappa = data['kappa']

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Standard MLP Architecture
class MLP(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# MLP Architecture with Batch Normalization
class MLPWithBN(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

# Helper function to train a model
def train_model(model_class, optimizer_class, optimizer_kwargs, epochs=300, is_bn=False):
    set_seed(42)
    model = model_class(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    
    # Instantiate optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    
    losses = []
    for epoch in range(epochs):
        if is_bn:
            model.train()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return losses

# ----------------------------------------------------
# Run all experimental configurations
# ----------------------------------------------------
print("Running Vanilla SGD...")
sgd_losses = train_model(
    MLP, 
    optim.SGD, 
    {"lr": 0.001, "momentum": 0.0}
)

print("Running Ridge Regularization (SGD + L2 weight decay)...")
ridge_losses = train_model(
    MLP, 
    optim.SGD, 
    {"lr": 0.001, "momentum": 0.0, "weight_decay": 0.05}
)

print("Running Adam Optimizer...")
adam_losses = train_model(
    MLP, 
    optim.Adam, 
    {"lr": 0.005}  # Standard Adam learning rate
)

print("Running Batch Normalization (SGD, lr=0.001)...")
bn_losses_low = train_model(
    MLPWithBN, 
    optim.SGD, 
    {"lr": 0.001, "momentum": 0.0},
    is_bn=True
)

print("Running Batch Normalization (SGD, high lr=0.01)...")
bn_losses_high = train_model(
    MLPWithBN, 
    optim.SGD, 
    {"lr": 0.01, "momentum": 0.0},
    is_bn=True
)

# Save results for references
np.savez('experimental_results.npz', 
         sgd=sgd_losses, 
         ridge=ridge_losses, 
         adam=adam_losses, 
         bn_low=bn_losses_low, 
         bn_high=bn_losses_high)

print("Saved experimental results to 'experimental_results.npz'")

# ----------------------------------------------------
# Plot the comparison
# ----------------------------------------------------
plt.figure(figsize=(10, 6))

# Custom styling for rich aesthetics
colors = {
    "sgd": "#a61e1e",      # dark red
    "ridge": "#d97706",    # amber/orange
    "adam": "#2563eb",     # royal blue
    "bn_low": "#0d9488",   # teal
    "bn_high": "#10b981"   # emerald
}

plt.plot(sgd_losses, label="Vanilla SGD (lr=0.001)", color=colors["sgd"], linewidth=2)
plt.plot(ridge_losses, label="Ridge Regularization (L2, lr=0.001)", color=colors["ridge"], linewidth=2)
plt.plot(bn_losses_low, label="Batch Norm (SGD, lr=0.001)", color=colors["bn_low"], linewidth=2)
plt.plot(bn_losses_high, label="Batch Norm (SGD, lr=0.01)", color=colors["bn_high"], linewidth=2)
plt.plot(adam_losses, label="Adam Optimizer (lr=0.005)", color=colors["adam"], linewidth=2.5)

plt.title("Numerical Stabilization Comparison on Ill-Conditioned Data (κ ≈ 2.04×10⁴)", 
          fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Epochs", fontsize=11, labelpad=8)
plt.ylabel("Mean Squared Error (Loss)", fontsize=11, labelpad=8)
plt.yscale('log') # Log scale is much better for ill-conditioned data convergence visualization
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True, fontsize=10)

plt.savefig('comparison_loss_curves.png', dpi=300)
plt.close()

print("Saved final comparison plot to 'comparison_loss_curves.png'")
