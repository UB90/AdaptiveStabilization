"""
Model definitions and weight saving utilities.
"""
import torch
import torch.nn as nn
import os

class LinearModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(d))
    def forward(self, X): return X @ self.theta
    def get_theta(self): return self.theta.detach().clone()

class LinearWithBatchNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        # FIX: track_running_stats=True so eval() behaves correctly
        self.bn = nn.BatchNorm1d(d, affine=False, track_running_stats=True)
        self.theta = nn.Parameter(torch.zeros(d))
    def forward(self, X): return self.bn(X) @ self.theta
    def get_theta(self): return self.theta.detach().clone()

# Removed ShallowMLP as it was unused

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)