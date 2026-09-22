"""Graph tier: k-NN transaction graphs + GraphSAGE / GATv2 classifiers.

ULB provides no entity IDs, so the graph is a FEATURE-SIMILARITY graph:
nodes are transactions, edges join each node to its k nearest neighbours
in standardized feature space. Graphs are built independently per split,
so no information crosses the time-based boundary (inductive evaluation).
"""

import copy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GATv2Conv, SAGEConv

from src import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_knn_graph(X: np.ndarray, k: int = 10) -> torch.Tensor:
    """Return undirected edge_index (2, E) of a k-NN graph over rows of X."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    src = np.repeat(np.arange(len(X)), k)
    dst = idx.reshape(-1)
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    return torch.cat([ei, ei.flip(0)], dim=1)


class SageNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.convs = torch.nn.ModuleList(
            SAGEConv(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x).squeeze(-1)


class GatNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=0.3)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        return self.head(h).squeeze(-1)


def train_graph_model(model, x_tr, ei_tr, y_tr, x_va, ei_va, y_va, *,
                      pos_weight=None, epochs=30, patience=6, lr=1e-3,
                      seed=config.RANDOM_STATE, verbose=False):
    """Full-batch training with BCE + early stopping on validation PR-AUC."""
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    pw = torch.tensor([pos_weight], device=DEVICE) if pos_weight else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    x_tr, ei_tr = x_tr.to(DEVICE), ei_tr.to(DEVICE)
    x_va_d, ei_va_d = x_va.to(DEVICE), ei_va.to(DEVICE)
    y_tr_d = y_tr.to(DEVICE)

    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = criterion(model(x_tr, ei_tr), y_tr_d)
        loss.backward()
        opt.step()
        val_ap = average_precision_score(
            y_va.numpy(), _sigmoid(predict_logits(model, x_va_d, ei_va_d)))
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


class _Adjacency:
    """CSR-style undirected adjacency over edge_index for fast neighbor lookup."""

    def __init__(self, edge_index: torch.Tensor, n: int):
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        order = np.argsort(dst, kind="stable")
        self.src = src[order]
        counts = np.bincount(dst, minlength=n)
        self.ptr = np.zeros(n + 1, dtype=np.int64)
        self.ptr[1:] = np.cumsum(counts)

    def neighbors_of(self, nodes: np.ndarray) -> np.ndarray:
        starts = self.ptr[nodes]
        ends = self.ptr[nodes + 1]
        lengths = ends - starts
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        offsets = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        return self.src[np.repeat(starts, lengths) + offsets]


def _subgraph_batch(adj, x, y, ei, seeds, rng, hop_cap=30000):
    """Given seed nodes -> 2-hop neighborhood -> induced subgraph tensors."""
    n = x.shape[0]
    hop1 = adj.neighbors_of(seeds)
    cand = np.unique(np.concatenate([seeds, hop1]))
    hop2 = adj.neighbors_of(cand)
    if len(hop2) > hop_cap:
        hop2 = rng.choice(hop2, size=hop_cap, replace=False)
    nodes = np.unique(np.concatenate([cand, hop2]))

    lut = np.full(n, -1, dtype=np.int64)
    lut[nodes] = np.arange(len(nodes))
    mask = (lut[ei[0]] >= 0) & (lut[ei[1]] >= 0)
    sub_ei = lut[ei[:, mask]]

    return (x[nodes], sub_ei, y[seeds], lut[seeds], len(seeds))


def train_gat_minibatch(model, x, ei, y, *, pos_weight=None, epochs=30,
                        patience=6, lr=1e-3, batch_size=1024, hop_cap=20000,
                        seed=config.RANDOM_STATE, verbose=False):
    """Manual neighbor-sampled mini-batch training (NeighborLoader's compiled
    backends are unavailable on this build; full-batch GAT OOMs at 2M edges)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = model.to(DEVICE)
    pw = torch.tensor([pos_weight], device=DEVICE) if pos_weight else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    x_np, ei_np, y_np = x.numpy(), ei.numpy(), y.numpy()
    adj = _Adjacency(ei, x.shape[0])

    best_ap, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(x_np))
        for start in range(0, len(perm), batch_size):
            seed_nodes = perm[start:start + batch_size]
            xs, eis, ys, seed_ids, n_seed = _subgraph_batch(
                adj, x_np, y_np, ei_np, seed_nodes, rng, hop_cap=hop_cap)
            xs = torch.as_tensor(xs, device=DEVICE)
            eis = torch.as_tensor(eis, device=DEVICE)
            ys = torch.as_tensor(ys, device=DEVICE)
            seed_ids_t = torch.as_tensor(seed_ids, device=DEVICE)
            opt.zero_grad()
            out = model(xs, eis)[seed_ids_t]
            loss = criterion(out, ys)
            loss.backward()
            opt.step()
        val_pred = predict_minibatch(model, x, ei, y)
        val_ap = average_precision_score(y.numpy(), _sigmoid(val_pred))
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


@torch.no_grad()
def predict_minibatch(model, x, ei, y=None, batch_size=16384) -> np.ndarray:
    """Neighborhood-preserving chunked inference: each chunk of nodes is
    evaluated on its own induced subgraph augmented with all their global
    neighbors, so no edges are lost; only the chunk's rows are recorded."""
    model.eval()
    x_np, ei_np = x.numpy(), ei.numpy()
    adj = _Adjacency(ei, len(x_np))
    out = np.empty(len(x_np), dtype=np.float64)
    for start in range(0, len(x_np), batch_size):
        seeds = np.arange(start, min(start + batch_size, len(x_np)))
        nbrs = adj.neighbors_of(seeds)
        nodes = np.unique(np.concatenate([seeds, nbrs]))
        lut = np.full(len(x_np), -1, dtype=np.int64)
        lut[nodes] = np.arange(len(nodes))
        mask = (lut[ei_np[0]] >= 0) & (lut[ei_np[1]] >= 0)
        xs = torch.as_tensor(x_np[nodes], device=DEVICE)
        es = torch.as_tensor(lut[ei_np[:, mask]], device=DEVICE)
        logits = model(xs, es)[torch.as_tensor(lut[seeds], device=DEVICE)]
        out[seeds] = logits.cpu().numpy()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


@torch.no_grad()
def predict_logits(model, x, edge_index, batch=None) -> np.ndarray:
    model.eval()
    return model(x, edge_index).cpu().numpy()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
