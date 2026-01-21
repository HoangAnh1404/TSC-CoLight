from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from stable_baselines3 import PPO

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------- FedAvg helper ----------------------------------------------------- #
def _assert_same_keys(reference: Iterable[str], candidate: Dict[str, torch.Tensor], tls_id: str) -> None:
    ref_keys = set(reference)
    cand_keys = set(candidate.keys())
    if ref_keys != cand_keys:
        missing = ref_keys - cand_keys
        extra = cand_keys - ref_keys
        raise ValueError(
            f"State dict keys mismatch for client {tls_id}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _check_finite(state_dict: Dict[str, torch.Tensor], tls_id: str) -> None:
    for k, v in state_dict.items():
        if not v.is_floating_point():
            raise ValueError(f"Non-float tensor detected in state '{k}' from client {tls_id} (dtype={v.dtype}).")
        if not torch.isfinite(v).all():
            raise ValueError(f"Non-finite values detected in state '{k}' from client {tls_id}.")


def fed_avg_weighted(state_weight_pairs: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
    """Standard FedAvg with float weights (e.g., samples, steps)."""
    if not state_weight_pairs:
        raise ValueError("No client updates provided.")

    ref_keys = set(state_weight_pairs[0][0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in ref_keys:
        stacked = torch.stack([sd[k].float() * w for sd, w in state_weight_pairs], dim=0)
        out[k] = stacked.sum(dim=0)
    return out


# -------- Client + server orchestration ------------------------------------ #
@dataclass
class ClientUpdate:
    tls_id: str
    state_dict: Dict[str, torch.Tensor]
    weight: float  # usually number of steps/samples
    metrics: Dict[str, float]


class SB3PPOClient:
    """
    Wrapper around SB3 PPO for FL.
    Boundary is policy-only: we sync `model.policy.state_dict()` (no optimizer/buffer).
    VecNormalize stats stay local and are not aggregated.
    """

    def __init__(
        self,
        tls_id: str,
        model: PPO,
    ) -> None:
        self.tls_id = tls_id
        self.model = model
        self._policy_keys = set(model.policy.state_dict().keys())

    def get_policy_state(self) -> Dict[str, torch.Tensor]:
        state = self.model.policy.state_dict()
        _check_finite(state, self.tls_id)
        return {k: v.detach().cpu() for k, v in state.items()}

    def load_policy_state(self, state: Dict[str, torch.Tensor]) -> None:
        _assert_same_keys(self._policy_keys, state, self.tls_id)
        _check_finite(state, self.tls_id)
        self.model.policy.load_state_dict(state, strict=True)

    def train_local(self, timesteps: int, callback=None) -> ClientUpdate:
        start_steps = self.model.num_timesteps
        # Keep num_timesteps monotonic across rounds so scheduler/logging stay consistent.
        self.model.learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=callback)
        trained_steps = max(int(self.model.num_timesteps - start_steps), timesteps)

        # SB3 stores last rollout stats in logger
        metrics_raw = self.model.logger.name_to_value if hasattr(self.model, "logger") else {}
        rew_mean = metrics_raw.get("rollout/ep_rew_mean", None)
        reward_mean = float(rew_mean) if rew_mean is not None and torch.isfinite(torch.tensor(rew_mean)) else None
        metrics = {
            "reward_mean": reward_mean,
            # TODO: add queue_mean/delay_mean if exposed by env infos or custom callback
        }
        state_cpu = self.get_policy_state()
        return ClientUpdate(
            tls_id=self.tls_id,
            state_dict=state_cpu,
            weight=float(max(trained_steps, 1)),
            metrics=metrics,
        )

    def save_local(self, save_dir: Path, tag: str) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_policy_state(), save_dir / f"policy_{tag}.pt")
        vec_env = self.model.get_vec_normalize_env()
        if vec_env is not None:
            vec_env.save(str(save_dir / f"vecnorm_{tag}.pkl"))

    def close(self) -> None:
        env = self.model.get_env()
        if env is not None:
            env.close()


class FederatedPPOCoordinator:
    """Central server: collect local PPO weights, run FedAvg, broadcast."""

    def __init__(
        self,
        weight_fn: Optional[Callable[[ClientUpdate], float]] = None,
        alpha: float = 0.0,  # FL rate mixing previous global with new FedAvg
    ) -> None:
        self.weight_fn = weight_fn or (lambda u: float(max(u.weight, 1.0)))
        self._ref_keys: Optional[set[str]] = None
        self.alpha = alpha

    def aggregate(
        self,
        updates: List[ClientUpdate],
        prev_global: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
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

        # Normalize weights to sum to 1 (pi in the formula); if all zero, raise.
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

    def broadcast(self, clients: List[SB3PPOClient], global_state: Dict[str, torch.Tensor]) -> None:
        for c in clients:
            c.load_policy_state(global_state)

    @staticmethod
    def save_global(path: Path, global_state: Dict[str, torch.Tensor]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_state, path)
