from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Ensure project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.custom_network import GAT, ObservationEncoder


class QNetwork(nn.Module):
    """Observation encoder + GAT encoder + MLP head for per-node Q-values.

    Dueling optional: split into value + advantage and recombine.
    """

    def __init__(
        self,
        occupancy_dim: int,
        phase_dim: int,
        hidden_dim: int,
        action_dim: int,
        heads: int = 4,
        tau: float = 1.0,
        dropout: float = 0.1,
        concat: bool = False,
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.encoder = ObservationEncoder(
            occupancy_dim=occupancy_dim,
            phase_dim=phase_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="gelu",
            use_residual=True,
        )
        gat_out_dim = hidden_dim
        self.gat = GAT(
            in_dim=hidden_dim,
            out_dim=gat_out_dim,
            heads=heads,
            tau=tau,
            dropout=dropout,
            add_self_loops=False,
            use_residual=True,
            concat=concat,
            activation="relu",
        )
        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(gat_out_dim, gat_out_dim),
                nn.ReLU(),
                nn.Linear(gat_out_dim, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(gat_out_dim, gat_out_dim),
                nn.ReLU(),
                nn.Linear(gat_out_dim, action_dim),
            )
        else:
            self.q_head = nn.Linear(gat_out_dim, action_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.gat(h, edge_index)
        if not self.dueling:
            return self.q_head(h)
        value = self.value_head(h)  # [N,1]
        adv = self.adv_head(h)      # [N,A]
        adv_mean = adv.mean(dim=-1, keepdim=True)
        q = value + adv - adv_mean
        return q
