"""
Training loops. Tracks train/test loss, param error, and eigen-direction errors.
"""
import math
import numpy as np
import torch
import config
from models import LinearModel, LinearWithBatchNorm

def mse_loss(model, X, y):
    res = model(X) - y
    return 0.5 * (res @ res) / X.shape[0]

def ridge_loss(model, X, y, lam):
    return mse_loss(model, X, y) + 0.5 * lam * (model.theta @ model.theta)

def new_history():
    return {
        "loss": [], "test_loss": [], "grad_norm": [], "param_error": [],
        "err_top_eig": [], "err_bot_eig": [], "diverged": False
    }

def record(history, model, X, y, X_test, y_test, theta_star, V):
    with torch.no_grad():
        # Training metrics
        loss = mse_loss(model, X, y).item()
        theta = model.get_theta()
        err = theta - theta_star
        pe = err.norm().item()
        gn = model.theta.grad.norm().item() if model.theta.grad is not None else 0.0
        
        # Test metrics (FIX: switch to eval mode for proper BN behavior)
        model.eval()
        test_loss = mse_loss(model, X_test, y_test).item()
        model.train()
        
        # Eigen-direction error
        err_proj = V.T @ err
        err_top = err_proj[0].abs().item()
        err_bot = err_proj[-1].abs().item()
        
        if not np.isfinite(loss) or not np.isfinite(pe):
            history["diverged"] = True

    history["loss"].append(loss)
    history["test_loss"].append(test_loss)
    history["grad_norm"].append(gn)
    history["param_error"].append(pe)
    history["err_top_eig"].append(err_top)
    history["err_bot_eig"].append(err_bot)

def train_vanilla_sgd(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearModel(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("vanilla_sgd", kappa)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad()
        loss = mse_loss(model, X, y)
        loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        with torch.no_grad(): model.theta -= lr * model.theta.grad
    return history, model

def train_momentum_sgd(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearModel(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("momentum_sgd", kappa)
    beta = config.momentum_beta_for_kappa(kappa)
    velocity = torch.zeros_like(model.theta)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad()
        loss = mse_loss(model, X, y)
        loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        with torch.no_grad():
            velocity = beta * velocity + model.theta.grad
            model.theta -= lr * velocity
    return history, model

def train_adam(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearModel(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("adam", kappa)
    opt = torch.optim.Adam([model.theta], lr=lr, betas=(config.ADAM_BETA1, config.ADAM_BETA2), eps=config.ADAM_EPS)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad(); loss = mse_loss(model, X, y); loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        opt.step()
    return history, model

def train_ridge_l2(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearModel(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("ridge_l2", kappa)
    lam = config.ridge_lambda_for_kappa(kappa)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad()
        loss = ridge_loss(model, X, y, lam)
        loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        with torch.no_grad(): model.theta -= lr * model.theta.grad
    return history, model

def train_batchnorm_linear(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearWithBatchNorm(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("batchnorm_linear", kappa)
    opt = torch.optim.Adam([model.theta], lr=lr, betas=(config.ADAM_BETA1, config.ADAM_BETA2), eps=config.ADAM_EPS)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad(); loss = mse_loss(model, X, y); loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        opt.step()
    return history, model

def train_adamw(X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs):
    d = X.shape[1]
    model = LinearModel(d)
    torch.manual_seed(init_seed); model.theta.data.normal_(0.0, 0.1)
    lr = config.lr_for_method("adamw", kappa)
    wd = config.weight_decay_for_kappa(kappa)
    opt = torch.optim.AdamW([model.theta], lr=lr, weight_decay=wd, betas=(config.ADAM_BETA1, config.ADAM_BETA2), eps=config.ADAM_EPS)
    history = new_history()
    for ep in range(epochs):
        model.zero_grad(); loss = mse_loss(model, X, y); loss.backward()
        record(history, model, X, y, X_test, y_test, theta_star, V)
        if history["diverged"]: break
        opt.step()
    return history, model

TRAINERS = {
    "vanilla_sgd": train_vanilla_sgd, "momentum_sgd": train_momentum_sgd,
    "adam": train_adam, "ridge_l2": train_ridge_l2,
    "batchnorm_linear": train_batchnorm_linear, "adamw": train_adamw
}

def train_method(method, dataset, init_seed):
    X, y = dataset["X_train"], dataset["y_train"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    theta_star, V = dataset["theta_star"], dataset["V"]
    kappa = dataset["kappa_target"]
    epochs = config.epochs_for_kappa(kappa)
    return TRAINERS[method](X, y, X_test, y_test, theta_star, V, init_seed, kappa, epochs)