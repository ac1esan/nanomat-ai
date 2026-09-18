"""CGCNN models (PyTorch Geometric), byte-compatible with the shipped checkpoints.

Do not rename modules/attributes: state_dict keys are part of the checkpoint
contract (emb, convs.*, head.*, centers / body.*, reg.*, cls.*).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool

from .graph import DEFAULT_CUTOFF, DEFAULT_N_RBF


class _Body(nn.Module):
    """Shared trunk: Z-embedding -> n_conv CGConv layers with RBF edge features.

    `ang_dim > 0` adds a per-atom angular descriptor (nanomat.graph.angle_features)
    to the atom embedding. An edge carries a distance and nothing else, so without
    it two polymorphs with the same bonds and different coordination geometry are
    nearly indistinguishable - measured on 1H against 1T-MX2, where the model
    compresses a 0.44 eV gap difference into 0.12 eV.

    At ang_dim = 0 no parameter is created and the state_dict is byte-identical to
    the shipped checkpoints, which is the contract this class has to keep.
    """

    def __init__(self, h: int, n_conv: int, cutoff: float, n_rbf: int, ang_dim: int = 0):
        super().__init__()
        self.emb = nn.Embedding(100, h)
        self.ang_dim = int(ang_dim)
        if self.ang_dim:
            self.ang = nn.Linear(self.ang_dim, h)
        self.convs = nn.ModuleList(
            [CGConv(h, dim=n_rbf, batch_norm=True) for _ in range(n_conv)]
        )
        self.register_buffer("centers", torch.linspace(0, cutoff, n_rbf))

    def encode(self, data) -> torch.Tensor:
        # Gaussian RBF expansion of distances, computed on the fly (memory-friendly)
        ea = torch.exp(-0.5 * (data.edge_weight.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2)
        x = self.emb(data.z)
        if self.ang_dim:
            x = x + self.ang(data.ang)
        for conv in self.convs:
            x = conv(x, data.edge_index, ea)
        return global_mean_pool(x, data.batch)


class CGCNN(_Body):
    """Regressor: structure -> normalised band gap (one scalar)."""

    def __init__(self, h: int = 128, n_conv: int = 4, p: float = 0.2,
                 cutoff: float = DEFAULT_CUTOFF, n_rbf: int = DEFAULT_N_RBF,
                 ang_dim: int = 0):
        super().__init__(h, n_conv, cutoff, n_rbf, ang_dim)
        self.head = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, 1))
        self.drop = nn.Dropout(p)

    def forward(self, data) -> torch.Tensor:
        x = self.drop(self.encode(data))
        return self.head(x).squeeze(-1)


class CGCNNcls(_Body):
    """Two heads: regression (normalised gap) + binary logit.

    Used for the direct/indirect gap classifier (`cls="type"`, positive class =
    indirect) and for the metal/semiconductor gate (`cls="metal"`, positive =
    metal).
    """

    def __init__(self, h: int = 128, n_conv: int = 4,
                 cutoff: float = DEFAULT_CUTOFF, n_rbf: int = DEFAULT_N_RBF,
                 ang_dim: int = 0):
        super().__init__(h, n_conv, cutoff, n_rbf, ang_dim)
        self.body = nn.Sequential(nn.Linear(h, h), nn.Softplus())
        self.reg = nn.Linear(h, 1)
        self.cls = nn.Linear(h, 1)

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.body(self.encode(data))
        return self.reg(x).squeeze(-1), self.cls(x).squeeze(-1)


def load_trunk_from(model: nn.Module, ckpt_state: dict) -> int:
    """Copy every parameter whose name and shape match (transfer learning /
    3D pre-training). Returns the number of tensors copied."""
    own = model.state_dict()
    copied = 0
    for k, v in ckpt_state.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            copied += 1
    model.load_state_dict(own)
    return copied
