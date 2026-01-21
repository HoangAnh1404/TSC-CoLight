from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter


def phase_to_feature(
    cur_phase: Optional[Union[int, torch.Tensor, Sequence[int]]],
    phase_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert a phase value to a fixed-length one-hot/binary tensor."""
    if phase_dim <= 0:
        return torch.zeros(0, device=device, dtype=dtype)
    if cur_phase is None:
        return torch.zeros(phase_dim, device=device, dtype=dtype)

    if isinstance(cur_phase, torch.Tensor):
        phase_tensor = cur_phase.detach().to(device=device)
    else:
        phase_tensor = torch.as_tensor(cur_phase, device=device)

    if phase_tensor.numel() == 1:
        idx = int(phase_tensor.item())
        if idx < 0 or idx >= phase_dim:
            return torch.zeros(phase_dim, device=device, dtype=dtype)
        return F.one_hot(torch.tensor(idx, device=device), num_classes=phase_dim).to(dtype)

    flat = phase_tensor.view(-1).to(dtype=dtype)
    if flat.numel() != phase_dim:
        raise ValueError(f"Phase feature length {flat.numel()} does not match phase_dim={phase_dim}.")
    return flat


def pack_state_dict(
    state_dict: Dict[str, Any],
    tls_id_to_idx: Dict[str, int],
    phase_dim: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack per-tls occupancy + phase into x of shape [N, F].

    Args:
        state_dict: Mapping tls_id -> occupancy vector or dict with keys
            {"occupancy"/"state"/"obs", optional "phase"/"cur_phase"}.
        tls_id_to_idx: Mapping from tls id to row index.
        phase_dim: Length of the phase one-hot/binary expansion.
    """
    if not state_dict:
        raise ValueError("state_dict is empty.")
    sample_entry = next(iter(state_dict.values()))

    def _extract(entry: Any, keys: Tuple[str, ...]) -> Any:
        if not isinstance(entry, dict):
            return None
        for k in keys:
            if k in entry and entry[k] is not None:
                return entry[k]
        return None

    occ_sample = _extract(sample_entry, ("occupancy", "state", "obs")) if isinstance(sample_entry, dict) else sample_entry
    if occ_sample is None:
        raise ValueError("Cannot infer occupancy from state_dict entries.")
    occupancy_dim = int(torch.as_tensor(occ_sample).numel())

    num_nodes = len(tls_id_to_idx)
    x = torch.zeros((num_nodes, occupancy_dim + phase_dim), device=device, dtype=dtype)

    for tls_id, idx in tls_id_to_idx.items():
        entry = state_dict.get(tls_id)
        if entry is None:
            continue
        if isinstance(entry, dict):
            occ = _extract(entry, ("occupancy", "state", "obs"))
            phase_val = _extract(entry, ("phase", "cur_phase", "phase_index"))
        else:
            occ = entry
            phase_val = None

        if occ is None:
            occ_tensor = torch.zeros(occupancy_dim, device=device, dtype=dtype)
        else:
            occ_tensor = torch.as_tensor(occ, device=device, dtype=dtype).view(-1)
            if occ_tensor.numel() != occupancy_dim:
                raise ValueError(
                    f"Occupancy length {occ_tensor.numel()} for tls {tls_id} "
                    f"does not match inferred dim {occupancy_dim}."
                )

        phase_tensor = phase_to_feature(phase_val, phase_dim, device=device, dtype=dtype)
        x[idx, :occupancy_dim] = occ_tensor
        if phase_dim > 0:
            x[idx, occupancy_dim:] = phase_tensor
    return x


class ObservationEncoder(nn.Module):
    """Encode per-node occupancy histories into fixed-size embeddings.

    Args:
        occupancy_dim: Length of the occupancy/count history (default 60).
        phase_dim: One-hot phase length to concatenate to occupancy (default 0).
        hidden_dim: Output/hidden embedding size (e.g., 64 or 128).
        dropout: Dropout probability applied after normalization.
        activation: Non-linearity to use between linear layers.
        use_residual: Whether to add a skip projection from the input.
    """

    def __init__(
        self,
        occupancy_dim: int = 60,
        phase_dim: int = 0,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        activation: Literal["relu", "gelu"] = "relu",
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.occupancy_dim = occupancy_dim
        self.phase_dim = phase_dim
        self.input_dim = occupancy_dim + phase_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        if activation == "gelu":
            act: nn.Module = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = act

        self.skip_proj: Optional[nn.Linear] = (
            nn.Linear(self.input_dim, hidden_dim) if use_residual else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features shaped (num_nodes, occupancy_dim + phase_dim) or
                (batch_num_nodes, occupancy_dim + phase_dim).

        Returns:
            Tensor of shape (num_nodes, hidden_dim) encoded per node.
        """
        x_norm = self.input_norm(x)
        h = self.act(self.fc1(x_norm))
        h = self.act(self.fc2(h))
        if self.skip_proj is not None:
            h = h + self.skip_proj(x_norm)
        h = self.output_norm(h)
        h = self.dropout(h)

        return h


class TemperatureScaledGATConv(MessagePassing):
    """GAT convolution following the provided four-step interaction diagram.

    1) Interaction score: e_ij = (h_i W_t) · (h_j W_s)^T
    2) Temperature softmax over neighbors of i: α_ij = softmax_j(e_ij / τ)
    3) Aggregate: h_tilde_i = Σ_j α_ij (h_j W_c)
    4) Project + activation: h'_i = σ(W_q h_tilde_i + b_q)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        tau: float = 1.0,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        concat: bool = True,
        activation: Optional[Literal["relu", "gelu"]] = "relu",
    ) -> None:
        super().__init__(node_dim=0, aggr="add")
        if concat and out_dim % heads != 0:
            raise ValueError("out_dim must be divisible by heads when concat=True.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.tau = tau
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.concat = concat
        self.head_dim = out_dim // heads if concat else out_dim

        # Learnable projections W_t, W_s, W_c per head
        self.lin_t = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.lin_s = nn.Linear(in_dim, heads * self.head_dim, bias=False)
        self.lin_c = nn.Linear(in_dim, heads * self.head_dim, bias=False)

        out_in_dim = self.head_dim * heads if concat else self.head_dim
        self.out_proj = nn.Linear(out_in_dim, out_dim, bias=True)
        if activation == "gelu":
            self.out_act: nn.Module = nn.GELU()
        elif activation == "relu":
            self.out_act = nn.ReLU()
        else:
            self.out_act = nn.Identity()

        self._alpha: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_t.weight)
        nn.init.xavier_uniform_(self.lin_s.weight)
        nn.init.xavier_uniform_(self.lin_c.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        q = self.lin_t(x).view(-1, self.heads, self.head_dim)
        k = self.lin_s(x).view(-1, self.heads, self.head_dim)
        v = self.lin_c(x).view(-1, self.heads, self.head_dim)
        self._alpha = None

        out = self.propagate(edge_index, x_t=q, x_s=k, x_c=v, size=None)
        if self.concat:
            out = out.view(-1, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)

        out = self.out_proj(out)
        out = self.out_act(out)

        if return_attention:
            alpha = (
                self._alpha
                if self._alpha is not None
                else torch.empty((0,), device=x.device)
            )
            return out, alpha
        return out

    def message(
        self,
        x_t_i: torch.Tensor,
        x_s_j: torch.Tensor,
        x_c_j: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        # Step 1 & 2: interaction + temperature softmax over neighbors of i
        scores = (x_t_i * x_s_j).sum(dim=-1)  # [E, heads]
        scores = scores / self.tau
        alpha = softmax(scores, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        self._alpha = alpha

        # Step 3: weighted message
        return x_c_j * alpha.unsqueeze(-1)

    def aggregate(
        self,
        inputs: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> torch.Tensor:
        # Scatter-add over destination index (node-wise aggregation).
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")


class GAT(nn.Module):
    """Single-layer GAT block for junction-level cooperation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        tau: float = 1.0,
        dropout: float = 0.0,
        add_self_loops: bool = False,
        use_residual: bool = True,
        concat: bool = True,
        activation: Optional[Literal["relu", "gelu"]] = "relu",
    ) -> None:
        super().__init__()
        self.feature_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv = TemperatureScaledGATConv(
            in_dim=in_dim,
            out_dim=out_dim,
            heads=heads,
            tau=tau,
            dropout=dropout,
            add_self_loops=add_self_loops,
            concat=concat,
            activation=activation,
        )
        self.use_residual = use_residual
        if use_residual:
            self.res_proj = (
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
        else:
            self.res_proj = None

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute node embeddings.

        Args:
            x: Node features shaped (num_nodes, in_dim).
            edge_index: Edge index with shape (2, E), direction j -> i.
            return_attention: If True, also return attention weights (E, heads).

        Returns:
            Node embeddings of shape (num_nodes, out_dim) and optionally attention.
        """
        x_in = self.feature_dropout(x)
        if return_attention:
            h, attn = self.conv(x_in, edge_index, return_attention=True)
        else:
            h = self.conv(x_in, edge_index, return_attention=False)
            attn = None

        if self.res_proj is not None:
            h = h + self.res_proj(x)

        if return_attention:
            return h, attn
        return h
