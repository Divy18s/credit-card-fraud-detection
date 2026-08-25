"""Compact tabular GAN for minority-class oversampling.

Trains a vanilla GAN on scaled minority samples, then generates synthetic
fraud examples to rebalance the training set (Transformer-GAN style,
arXiv:2509.19032, simplified to a dense architecture).
"""

import hashlib

import numpy as np
import torch
import torch.nn as nn

from src import config


class Generator(nn.Module):
    def __init__(self, latent: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_gan(X_minority: np.ndarray, *, latent: int = 24, hidden: int = 256,
              steps: int = 8000, batch: int = 128,
              seed: int = config.RANDOM_STATE, verbose: bool = False) -> Generator:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    X = torch.as_tensor(X_minority, dtype=torch.float32)
    dim = X.shape[1]
    G, D = Generator(latent, dim, hidden).to(device), Discriminator(dim, hidden).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    n = len(X)

    g_updates = 0
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        real = X[idx].to(device)

        opt_d.zero_grad()
        z = torch.randn(len(real), latent, device=device)
        fake = G(z).detach()
        d_real = criterion(D(real), torch.full((len(real), 1), 0.9, device=device))
        d_fake = criterion(D(fake), torch.zeros(len(real), 1, device=device))
        (d_real + d_fake).backward()
        opt_d.step()

        if step % 2 == 0:
            opt_g.zero_grad()
            z = torch.randn(batch, latent, device=device)
            g_loss = criterion(D(G(z)), torch.ones(batch, 1, device=device))
            g_loss.backward()
            opt_g.step()
            g_updates += 1
        if verbose and (step + 1) % 2000 == 0:
            print(f"    gan step {step + 1}/{steps}  D={float(d_real + d_fake):.3f}  G={float(g_loss):.3f}")
    return G


@torch.no_grad()
def generate(G: Generator, n: int, latent: int = 24) -> np.ndarray:
    device = next(G.parameters()).device
    outs = []
    for i in range(0, n, 4096):
        z = torch.randn(min(4096, n - i), latent, device=device)
        outs.append(G(z).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


_CACHE = {}


def oversample_with_gan(X: np.ndarray, y: np.ndarray, *, seq_mode: bool = False):
    """Return augmented (X, y) balanced to 1:1 with GAN-generated minorities.

    seq_mode treats rows as flattened (seq_len*n_feat) windows: 3D input
    (n, seq_len, n_feat) is flattened before GAN training and the synthetic
    samples are reshaped back afterwards.
    The trained generator is cached per (mode, dimension, data fingerprint):
    the fingerprint is a hash of the actual minority rows, so a generator is
    NEVER reused across different training sets (e.g. across split protocols).
    """
    y = np.asarray(y)
    orig_shape = X.shape[1:]
    flat = X.reshape(len(X), -1)
    X_min = flat[y == 1]
    n_min, n_maj = int((y == 1).sum()), int((y == 0).sum())
    latent = 48 if seq_mode else 24
    fp = hashlib.sha1(np.ascontiguousarray(X_min, dtype=np.float64).tobytes()).hexdigest()[:16]
    key = ("seq" if seq_mode else "rows", flat.shape[1], len(X_min), fp)
    if key not in _CACHE:
        _CACHE[key] = train_gan(X_min.astype(np.float32), latent=latent,
                                hidden=512 if seq_mode else 256)
    G = _CACHE[key]
    X_syn = generate(G, n_maj - n_min, latent=latent).reshape(-1, *orig_shape)
    X_aug = np.vstack([X, X_syn]).astype(np.float32)
    y_aug = np.concatenate([y, np.ones(len(X_syn), dtype=y.dtype)])
    return X_aug, y_aug
