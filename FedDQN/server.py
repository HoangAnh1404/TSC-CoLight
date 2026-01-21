from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


def _assert_same_keys(reference: Iterable[str], candidate: Dict[str, torch.Tensor], name: str) -> None:
    ref_keys = set(reference)
    cand_keys = set(candidate.keys())
    if ref_keys != cand_keys:
        missing = ref_keys - cand_keys
        extra = cand_keys - ref_keys
        raise ValueError(f"State dict keys mismatch for {name}: missing={sorted(missing)}, extra={sorted(extra)}")


def _check_finite(state_dict: Dict[str, torch.Tensor], name: str) -> None:
    for k, v in state_dict.items():
        if not v.is_floating_point():
            raise ValueError(f"Non-float tensor detected in state '{k}' from {name} (dtype={v.dtype}).")
        if not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values detected in state '{k}' from {name}.")


def fed_avg_weighted(state_weight_pairs: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    if not state_weight_pairs:
        raise ValueError("No client updates provided.")
    ref_keys = set(state_weight_pairs[0][0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        stacked = torch.stack([sd[k].float() * w for sd, w in state_weight_pairs], dim=0)
        out[k] = stacked.sum(dim=0)
    # assume weights already normalized before
    return out


@dataclass
class ClientUpdate:
    tls_id: str
    state_dict: Dict[str, torch.Tensor]
    weight: float
    metrics: Dict[str, float]


class FedAvgServer:
    def __init__(self, alpha: float = 0.0, weight_fn: Optional[Callable[[ClientUpdate], float]] = None) -> None:
        self.alpha = alpha
        self.weight_fn = weight_fn or (lambda u: float(max(u.weight, 1.0)))
        self._ref_keys: Optional[set[str]] = None

    def aggregate(self, updates: List[ClientUpdate], prev_global: Optional[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("No updates to aggregate.")
        valid_pairs: List[Tuple[Dict[str, torch.Tensor], float]] = []
        for u in updates:
            try:
                state = u.state_dict
                if self._ref_keys is None:
                    self._ref_keys = set(state.keys())
                _assert_same_keys(self._ref_keys, state, u.tls_id)
                _check_finite(state, u.tls_id)
                w = self.weight_fn(u)
                if not torch.isfinite(torch.tensor(w)):
                    raise ValueError(f"Non-finite weight from client {u.tls_id}: {w}")
                valid_pairs.append((state, float(w)))
            except Exception as exc:
                print(f"[WARN] Skipping client {u.tls_id} due to error: {exc}")
        if not valid_pairs:
            raise ValueError("All client updates were invalid; cannot aggregate.")

        weight_sum = sum(w for _, w in valid_pairs)
        if weight_sum <= 0:
            raise ValueError("Sum of aggregation weights is non-positive.")
        norm_pairs = [(sd, w / weight_sum) for sd, w in valid_pairs]
        local_fed = fed_avg_weighted(norm_pairs)

        if prev_global is None or self.alpha <= 0:
            return local_fed
        _assert_same_keys(local_fed.keys(), prev_global, "global")
        mixed: Dict[str, torch.Tensor] = {}
        for k in local_fed:
            mixed[k] = self.alpha * prev_global[k].to(local_fed[k].dtype) + (1 - self.alpha) * local_fed[k]
        return mixed

    @staticmethod
    def broadcast(clients, global_state: Dict[str, torch.Tensor]) -> None:
        for c in clients:
            c.load_policy_state(global_state)

    @staticmethod
    def save_global(path: Path, global_state: Dict[str, torch.Tensor]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_state, path)
