"""MLP and KAN autoencoders, baseline detectors and shared training."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

if TYPE_CHECKING:
    from pipeline import PipelineConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------
class MLPAutoencoder(nn.Module):
    def __init__(self, dim, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.dec(self.enc(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(1).cpu().numpy()


class BSplineEdge(nn.Module):
    """Learned spline connections in one Kolmogorov-Arnold layer:
    phi(x) = w_b * silu(x) + sum_k c_k B_k(x).

    Weights scale with the number of inputs. Before training, fit_grid sets
    each input's knots from its 1st and 99th percentiles, with a margin.
    Grids stay fixed during training to preserve the spline representation.
    """

    def __init__(self, in_f, out_f, G=5, k=3, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_f, self.out_f, self.G, self.k = in_f, out_f, G, k
        n = G + k
        fan_in = max(1, in_f)
        # fan-in scaling keeps output variance roughly constant across
        # layers of very different width (e.g. dim=26 -> 64 -> latent=12)
        self.sp_w = nn.Parameter(torch.randn(out_f, in_f, n) / np.sqrt(fan_in) * 0.1)
        self.base_w = nn.Parameter(torch.randn(out_f, in_f) / np.sqrt(fan_in))
        self._build_grid(grid_range)

    def _build_grid(self, grid_range):
        device = self.sp_w.device
        lo = torch.as_tensor(grid_range[0], dtype=torch.float32, device=device).flatten()
        hi = torch.as_tensor(grid_range[1], dtype=torch.float32, device=device).flatten()
        if lo.numel() == 1:
            lo = lo.repeat(self.in_f)
        if hi.numel() == 1:
            hi = hi.repeat(self.in_f)
        span = (hi - lo).clamp_min(1e-3)
        hi = lo + span
        h = span / self.G
        offsets = torch.arange(-self.k, self.G + self.k + 1,
                               dtype=torch.float32, device=device)
        grid = lo[:, None] + h[:, None] * offsets[None, :]
        if hasattr(self, "grid"):
            self.grid.copy_(grid)
        else:
            self.register_buffer("grid", grid)

    @torch.no_grad()
    def fit_grid(self, x, margin=0.05):
        """Re-fit the spline grid to the 1st/99th percentile of a real batch
        of inputs to this edge (with a small margin), instead of assuming
        [-1, 1]. Call once, on a representative batch, before training."""
        x = x.detach().float()
        lo = torch.quantile(x, 0.01, dim=0)
        hi = torch.quantile(x, 0.99, dim=0)
        span = (hi - lo).clamp_min(1e-3)
        lo -= margin * span
        hi += margin * span
        self._build_grid((lo, hi))

    def bspline(self, x):
        x = x.unsqueeze(-1)
        g = self.grid.unsqueeze(0)
        b = ((x >= g[:, :, :-1]) & (x < g[:, :, 1:])).float()
        for order in range(1, self.k + 1):
            gl, gr = g[:, :, :-(order + 1)], g[:, :, order:-1]
            gl2, gr2 = g[:, :, 1:-order], g[:, :, (order + 1):]
            d1, d2 = gr - gl, gr2 - gl2
            b1 = torch.where(d1 > 1e-8, (x - gl) / d1 * b[:, :, :-1],
                              torch.zeros_like(b[:, :, :-1]))
            b2 = torch.where(d2 > 1e-8, (gr2 - x) / d2 * b[:, :, 1:],
                              torch.zeros_like(b[:, :, 1:]))
            b = b1 + b2
        return b

    def forward(self, x):
        lo = self.grid[:, self.k]
        hi = self.grid[:, -(self.k + 1)]
        # Keep the upper endpoint inside the final half-open knot interval.
        hi_open = torch.nextafter(hi, lo)
        xc = torch.minimum(torch.maximum(x, lo), hi_open)
        sp = torch.einsum("oin,bin->bo", self.sp_w, self.bspline(xc))
        return sp + nn.functional.silu(x) @ self.base_w.T

    def spline_l1(self):
        return self.sp_w.abs().mean()


class KANAutoencoder(nn.Module):
    def __init__(self, dim, hidden=8, latent=3, G=3, k=3):
        super().__init__()
        self.enc_edges = nn.ModuleList([
            BSplineEdge(dim, hidden, G=G, k=k),
            BSplineEdge(hidden, latent, G=G, k=k),
        ])
        self.dec_edges = nn.ModuleList([
            BSplineEdge(latent, hidden, G=G, k=k),
            BSplineEdge(hidden, dim, G=G, k=k),
        ])

    def encode(self, x):
        for e in self.enc_edges:
            x = e(x)
        return x

    def decode(self, z):
        for e in self.dec_edges:
            z = e(z)
        return z

    def forward(self, x):
        return self.decode(self.encode(x))

    def recon_error(self, x):
        with torch.no_grad():
            return ((self(x) - x) ** 2).mean(1).cpu().numpy()

    def spline_l1(self):
        edges = list(self.enc_edges) + list(self.dec_edges)
        return sum(e.spline_l1() for e in edges) / len(edges)

    @torch.no_grad()
    def fit_grids(self, x):
        """Re-fit every edge's spline grid to the real data it will see,
        layer by layer (each edge is fit on the *output* of the previous
        one), using a single warm-up batch. Call once before training."""
        h = x
        for e in self.enc_edges:
            e.fit_grid(h)
            h = e(h)
        for e in self.dec_edges:
            e.fit_grid(h)
            h = e(h)


def build_traditional_models(n_train):
    lof_k = max(2, min(20, n_train - 1))
    return {
        "OC-SVM": OneClassSVM(kernel="rbf", nu=0.1, gamma="scale"),
        "Isolation Forest": IsolationForest(n_estimators=200, contamination=0.1,
                                             random_state=42),
        "LOF": LocalOutlierFactor(n_neighbors=lof_k, novelty=True,
                                   contamination=0.1),
        "Elliptic Envelope": EllipticEnvelope(contamination=0.1, random_state=42),
    }


def make_autoencoder(name, dim, cfg: PipelineConfig):
    """Use the same architecture for training and parameter-count reporting."""
    if name == "MLP Autoencoder":
        return MLPAutoencoder(dim, hidden=cfg.mlp_hidden_width, latent=cfg.mlp_latent)
    if name == "KAN Autoencoder":
        return KANAutoencoder(dim, hidden=cfg.kan_hidden_width, latent=cfg.kan_latent,
                              G=cfg.kan_grid_size, k=cfg.kan_spline_degree)
    raise ValueError(f"Unknown autoencoder: {name}")


# --------------------------------------------------------------------------
# Shared training
# --------------------------------------------------------------------------
def train_autoencoder(model, X, max_epochs, patience, lr, weight_decay,
                       batch_size=32, l1_lambda=0.0, val_frac=0.2,
                       min_val=3, seed=0, use_cosine=True):
    """Trains `model` to reconstruct X with early stopping on a held-out
    validation slice. Falls back to training-loss-based stopping when X is
    too small to split (a real constraint here: some subjects only have
    ~10-15 training beats). Returns (model, info_dict).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)

    n_val = int(round(n * val_frac))
    can_split = n_val >= min_val and (n - n_val) >= 4
    if can_split:
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
    else:
        val_idx, tr_idx = np.array([], dtype=int), idx

    Xtr = torch.tensor(X[tr_idx].astype(np.float32)).to(DEVICE)
    Xval = (torch.tensor(X[val_idx].astype(np.float32)).to(DEVICE)
            if len(val_idx) else None)

    if hasattr(model, "fit_grids"):
        model.fit_grids(Xtr)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs) if use_cosine else None
    loader = DataLoader(TensorDataset(Xtr), batch_size=min(batch_size, len(Xtr)),
                         shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            out = model(xb)
            loss = nn.functional.mse_loss(out, xb)
            if l1_lambda > 0 and hasattr(model, "spline_l1"):
                loss = loss + l1_lambda * model.spline_l1()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched is not None:
            sched.step()
        epochs_run = epoch + 1

        model.eval()
        with torch.no_grad():
            probe = Xval if Xval is not None else Xtr
            val_loss = nn.functional.mse_loss(model(probe), probe).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs_run": epochs_run, "best_val_loss": best_val,
                    "used_val_split": bool(Xval is not None)}

