from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.custom_network import pack_state_dict


def reorder_state(
    state_dict: Dict[str, Any],
    tls_id_to_idx: Dict[str, int],
    node_id_list: List[str],
    phase_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack and reorder occupancy+phase into x[N,F] using node_id_list/tls_id_to_idx mapping."""
    x = pack_state_dict(state_dict, tls_id_to_idx=tls_id_to_idx, phase_dim=phase_dim, device=device)
    order = torch.tensor([tls_id_to_idx[tls_id] for tls_id in node_id_list], device=device, dtype=torch.long)
    return x[order]


def build_action_dict(actions: torch.Tensor, node_id_list: List[str]) -> Dict[str, int]:
    """Synchronize actions across TLS using node_id_list ordering."""
    actions = actions.view(-1)
    if actions.numel() != len(node_id_list):
        raise ValueError(f"Expected {len(node_id_list)} actions, got {actions.numel()}.")
    return {tls_id: int(actions[i].item()) for i, tls_id in enumerate(node_id_list)}
