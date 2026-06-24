import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------
# Load the ill‑conditioned data
# -------------------------------------------
data = np.load('data_ill_conditioned.npz')
X = data['X']
y = data['y']
kappa = data['kappa']

print(f"Loaded data: X shape {X.shape}, κ = {kappa:.2e}")

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# -------------------------------------------
# Define a simple 3‑layer MLP
# -------------------------------------------
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

model = MLP(input_dim=X.shape[1])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)  # pure Vanilla

# -------------------------------------------
# Train and record loss
# -------------------------------------------
epochs = 300
losses = []

for epoch in range(epochs):
    pred = model(X_t)
    loss = criterion(pred, y_t)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | Loss = {loss.item():.4f}")

# -------------------------------------------
# Plot the loss curve – oscillations expected
# -------------------------------------------
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(losses, color='darkred', linewidth=1.5)
plt.title(f'Vanilla SGD on Ill‑Conditioned Data\n(κ = {kappa:.2e})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.plot(losses[:50], color='darkred', linewidth=2)
plt.title('First 50 epochs – oscillating')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vanilla_sgd_loss.png', dpi=150)
plt.show()

print("\nControl done. Vanilla SGD struggles – oscillations and slow convergence.")