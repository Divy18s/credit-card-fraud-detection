"""PyTorch models and training utilities for the deep-learning tier.

All classifiers output raw logits; loss is BCEWithLogitsLoss.
The AutoEncoder is trained on legitimate transactions only and scores
fraud by reconstruction error (higher = more anomalous).
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

from src import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DNN(nn.Module):
    """Deeper/wider variant to test whether depth helps on this task."""

    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class AutoEncoder(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout,
                            bidirectional=bidirectional)
        head_in = hidden * 2 if bidirectional else hidden
        self.head = nn.Linear(head_in, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class FTTransformer(nn.Module):
    """Compact FT-Transformer (Gorishniy et al.) for all-numeric tabular data."""

    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, ffn: int = 128, dropout: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.b = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.W + self.b
        cls = self.cls.expand(len(x), -1, -1)
        out = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(self.norm(out[:, 0])).squeeze(-1)


def _loader(X: np.ndarray, y: np.ndarray | None, batch: int, shuffle: bool) -> DataLoader:
    xt = torch.as_tensor(X, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(xt)
    else:
        ds = TensorDataset(xt, torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def reconstruction_error(model: nn.Module, X: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    errs = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=DEVICE)
        rec = model(xb)
        errs.append(((rec - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(errs)


def train_classifier(model, X_tr, y_tr, X_va, y_va, *, pos_weight=None,
                     epochs=40, patience=6, lr=1e-3, batch=2048,
                     verbose=False, seed=config.RANDOM_STATE):
    """Train with BCE + early stopping on validation PR-AUC; returns best model."""
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    pw = torch.tensor([pos_weight], device=DEVICE) if pos_weight else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = _loader(X_tr, y_tr, batch, shuffle=True)
    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        val_ap = average_precision_score(y_va, _sigmoid(predict_logits(model, X_va)))
        if verbose:
            print(f"    epoch {epoch + 1:>2}  val PR-AUC={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def train_autoencoder(model, X_legit, X_va, y_va, *, epochs=50, patience=5,
                      lr=1e-3, batch=2048, verbose=False):
    """Reconstruction training on legitimate transactions only."""
    torch.manual_seed(config.RANDOM_STATE)
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = _loader(X_legit, None, batch, shuffle=True)

    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            opt.step()
        val_err = reconstruction_error(model, X_va)
        val_ap = average_precision_score(y_va, val_err)
        if verbose:
            print(f"    epoch {epoch + 1:>2}  val PR-AUC={val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, best_state, bad = val_ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def make_sequences(X: np.ndarray, y: np.ndarray | None, seq_len: int):
    """Sliding windows over time-sorted rows; label = class of last step."""
    xs = [X[i:i + seq_len] for i in range(len(X) - seq_len + 1)]
    X_seq = np.stack(xs).astype(np.float32)
    if y is None:
        return X_seq, None
    return X_seq, y[seq_len - 1:].astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
