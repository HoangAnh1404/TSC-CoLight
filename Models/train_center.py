from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

# Ensure project root is on sys.path for absolute imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Env.make_env import make_multi_envs
from Env.road_graph_builder import GraphSpec, RoadGraphBuilder
from Models.custom_network import pack_state_dict
from Models.train import DQNConfig, GraphDQNAgent, QNetwork, ReplayBufferGraph, build_action_dict


@dataclass
class LocalEnvConfig:
    sumo_cfg: str
    net_file: str
    trip_info: Optional[str]
    num_seconds: int
    tls_action_type: str
    log_dir: Path


@dataclass
class LocalWorkerConfig:
    neighbor_hop_k: int = 1
    max_steps_per_ep: int = 300
    replay_capacity: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 5000


@dataclass
class LocalGraphView:
    node_id_list: List[str]
    tls_id_to_idx: Dict[str, int]
    edge_index: torch.Tensor


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


def build_local_graph_view(graph: GraphSpec, center_tls_id: str, hop_k: int, device: torch.device) -> LocalGraphView:
    """Extract a hop-k subgraph around center_tls_id with reverse edges and self-loops."""
    start_idx = graph.tls_id_to_idx[center_tls_id]
    visited = {start_idx}
    frontier = {start_idx}
    for _ in range(max(0, hop_k)):
        next_frontier = set()
        for idx in frontier:
            next_frontier.update(graph.neighbors_list[idx])
        next_frontier -= visited
        if not next_frontier:
            break
        visited |= next_frontier
        frontier = next_frontier

    keep_idx = sorted(visited)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    node_id_list = [graph.idx_to_tls_id[i] for i in keep_idx]
    edges = set()
    for old_src in keep_idx:
        src = old_to_new[old_src]
        for old_dst in graph.neighbors_list[old_src]:
            if old_dst not in old_to_new:
                continue
            dst = old_to_new[old_dst]
            edges.add((src, dst))
            edges.add((dst, src))
    for i in range(len(node_id_list)):
        edges.add((i, i))

    if edges:
        edge_index = torch.as_tensor(sorted(edges), dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    tls_id_to_idx = {tls_id: idx for idx, tls_id in enumerate(node_id_list)}
    return LocalGraphView(node_id_list=node_id_list, tls_id_to_idx=tls_id_to_idx, edge_index=edge_index)


def fed_avg(state_weight_pairs: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """Standard FedAvg aggregation across client state_dicts."""
    if not state_weight_pairs:
        raise ValueError("No client updates provided for aggregation.")
    total_weight = float(sum(weight for _, weight in state_weight_pairs))
    if total_weight <= 0:
        raise ValueError("Total aggregation weight must be positive.")

    avg_state: Dict[str, torch.Tensor] = {}
    for key in state_weight_pairs[0][0].keys():
        stacked = torch.stack(
            [sd[key].float() * weight for sd, weight in state_weight_pairs],
            dim=0,
        )
        avg_state[key] = stacked.sum(dim=0) / total_weight
    return avg_state


class LocalDQNWorker:
    """DQN trainer scoped to a single tls_id subgraph; used as an FL client."""

    def __init__(
        self,
        tls_id: str,
        graph: GraphSpec,
        env_cfg: LocalEnvConfig,
        dqn_config: DQNConfig,
        worker_cfg: LocalWorkerConfig,
        hidden_dim: int = 128,
        heads: int = 4,
        learning_rate: float = 3e-4,
        feature_spec: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.tls_id = tls_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_cfg = env_cfg
        self.worker_cfg = worker_cfg
        self.dqn_config = dqn_config

        self.graph_view = build_local_graph_view(
            graph=graph, center_tls_id=tls_id, hop_k=worker_cfg.neighbor_hop_k, device=self.device
        )
        log_path = env_cfg.log_dir / f"{tls_id}.monitor.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.env = make_multi_envs(
            tls_ids=self.graph_view.node_id_list,
            sumo_cfg=env_cfg.sumo_cfg,
            num_seconds=env_cfg.num_seconds,
            use_gui=False,
            net_file=env_cfg.net_file,
            trip_info=env_cfg.trip_info,
            tls_action_type=env_cfg.tls_action_type,
            log_path=str(log_path),
        )

        init_state, _ = self.env.reset()
        self.feature_spec = self._infer_feature_spec(init_state, override=feature_spec)
        self.action_dim = 2 if env_cfg.tls_action_type == "next_or_not" else max(1, self.feature_spec["phase_dim"] * 2)

        def optimizer_builder(model: nn.Module):
            return torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.q_net = QNetwork(
            occupancy_dim=self.feature_spec["occupancy_dim"],
            phase_dim=self.feature_spec["phase_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            heads=heads,
            tau=1.0,
            dropout=0.1,
        ).to(self.device)
        self.target_net = QNetwork(
            occupancy_dim=self.feature_spec["occupancy_dim"],
            phase_dim=self.feature_spec["phase_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            heads=heads,
            tau=1.0,
            dropout=0.1,
        ).to(self.device)
        self.agent = GraphDQNAgent(
            q_network=self.q_net,
            target_network=self.target_net,
            edge_index=self.graph_view.edge_index,
            action_dim=self.action_dim,
            config=dqn_config,
            optimizer_builder=optimizer_builder,
            device=self.device,
        )
        self.replay = ReplayBufferGraph(capacity=worker_cfg.replay_capacity, device=self.device)
        self.env_state = self._init_env_state(init_state)
        self.global_step = 0
        self.last_epsilon = worker_cfg.epsilon_start

    def _infer_feature_spec(self, state: Dict[str, Any], override: Optional[Dict[str, int]]) -> Dict[str, int]:
        sample_entry = next(iter(state.values()))
        occupancy_dim = len(sample_entry.get("occupancy", []))
        phase_dim = len(sample_entry.get("phase", []))
        if override is not None:
            if override["occupancy_dim"] != occupancy_dim or override["phase_dim"] != phase_dim:
                raise ValueError(
                    f"Feature mismatch for tls {self.tls_id}: "
                    f"expected (occ={override['occupancy_dim']}, phase={override['phase_dim']}), "
                    f"got (occ={occupancy_dim}, phase={phase_dim})."
                )
            return override
        return {"occupancy_dim": occupancy_dim, "phase_dim": phase_dim}

    def _init_env_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        x_tensor = reorder_state(
            state,
            tls_id_to_idx=self.graph_view.tls_id_to_idx,
            node_id_list=self.graph_view.node_id_list,
            phase_dim=self.feature_spec["phase_dim"],
            device=self.device,
        )
        num_nodes = len(self.graph_view.node_id_list)
        return {
            "x": x_tensor,
            "done_mask": torch.zeros(num_nodes, dtype=torch.bool, device=self.device),
            "action_mask": torch.ones((num_nodes, self.action_dim), device=self.device),
            "episode_step": 0,
        }

    def sync_from_global(self, global_state: Dict[str, torch.Tensor]) -> None:
        self.q_net.load_state_dict(global_state)
        self.agent.hard_update()

    def run_local_steps(self, num_steps: int) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
        steps = 0
        losses: List[float] = []
        while steps < num_steps:
            if self.env_state["episode_step"] >= self.worker_cfg.max_steps_per_ep or self.env_state["done_mask"].all():
                state, _ = self.env.reset()
                self.env_state = self._init_env_state(state)

            epsilon = max(
                self.worker_cfg.epsilon_end,
                self.worker_cfg.epsilon_start - self.global_step / float(self.worker_cfg.epsilon_decay),
            )
            actions = self.agent.act(
                self.env_state["x"], action_mask=self.env_state["action_mask"], epsilon=epsilon
            ).view(-1)
            actions_dict = build_action_dict(actions, self.graph_view.node_id_list)

            next_state, reward_dict, truncated_dict, done_dict, infos = self.env.step(actions_dict)
            rewards = torch.tensor(
                [reward_dict[tls] for tls in self.graph_view.node_id_list],
                device=self.device,
                dtype=torch.float32,
            )
            dones = torch.tensor(
                [done_dict[tls] or truncated_dict[tls] for tls in self.graph_view.node_id_list],
                device=self.device,
                dtype=torch.bool,
            )
            next_x = reorder_state(
                next_state,
                tls_id_to_idx=self.graph_view.tls_id_to_idx,
                node_id_list=self.graph_view.node_id_list,
                phase_dim=self.feature_spec["phase_dim"],
                device=self.device,
            )
            next_action_mask = torch.tensor(
                [infos.get(tls, {}).get("can_perform_action", True) for tls in self.graph_view.node_id_list],
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(-1).expand(-1, self.action_dim)

            self.replay.add(self.env_state["x"], actions, rewards, next_x, dones)
            loss_val = self.agent.optimize(self.replay, action_mask_next=None)
            if loss_val is not None:
                losses.append(loss_val)

            self.env_state["x"] = next_x
            self.env_state["done_mask"] = dones
            self.env_state["action_mask"] = next_action_mask
            self.env_state["episode_step"] += 1

            self.global_step += 1
            steps += 1
            self.last_epsilon = epsilon

        state_dict = {k: v.detach().cpu() for k, v in self.q_net.state_dict().items()}
        metrics = {
            "avg_loss": float(sum(losses) / len(losses)) if losses else None,
            "epsilon": self.last_epsilon,
            "steps": steps,
        }
        return state_dict, steps, metrics

    def close(self) -> None:
        self.env.close()


def train_federated(
    sumo_cfg: str,
    net_file: str,
    trip_info: Optional[str] = None,
    num_seconds: int = 500,
    tls_action_type: str = "next_or_not",
    num_rounds: int = 5,
    local_steps: int = 200,
    neighbor_hop_k: int = 1,
    hidden_dim: int = 128,
    heads: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 64,
    warmup_steps: int = 50,
    target_update_interval: int = 200,
    tau: float = 0.0,
    grad_clip_norm: float = 5.0,
    replay_capacity: int = 5000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5000,
    max_steps_per_ep: int = 300,
    log_dir: str = "logs/federated",
    save_dir: str = "Models/result/federated",
    device: Optional[str] = None,
    client_tls_ids: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run FedAvg with DQN+GAT clients (one per tls_id).
    Local training logic mirrors Models/train.py but is scoped to subgraphs.
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env_cfg = LocalEnvConfig(
        sumo_cfg=sumo_cfg,
        net_file=net_file,
        trip_info=trip_info,
        num_seconds=num_seconds,
        tls_action_type=tls_action_type,
        log_dir=Path(log_dir),
    )
    dqn_cfg = DQNConfig(
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        huber_delta=1.0,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
    )
    worker_cfg = LocalWorkerConfig(
        neighbor_hop_k=neighbor_hop_k,
        max_steps_per_ep=max_steps_per_ep,
        replay_capacity=replay_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    graph = RoadGraphBuilder.build_from_net_file(
        net_file=net_file,
        directed=True,
        make_bidirectional=False,
        include_self_loops=False,
        max_hops_between_tls=1,
        neighbor_strategy="hop",
        neighbor_hop_k=neighbor_hop_k,
        include_self_in_neighbor=True,
    )
    all_tls = graph.node_id_list or graph.idx_to_tls_id
    selected_tls = client_tls_ids or all_tls

    workers: List[LocalDQNWorker] = []
    shared_feature_spec: Optional[Dict[str, int]] = None
    for tls in selected_tls:
        worker = LocalDQNWorker(
            tls_id=tls,
            graph=graph,
            env_cfg=env_cfg,
            dqn_config=dqn_cfg,
            worker_cfg=worker_cfg,
            hidden_dim=hidden_dim,
            heads=heads,
            learning_rate=learning_rate,
            feature_spec=shared_feature_spec,
            device=device_t,
        )
        shared_feature_spec = shared_feature_spec or worker.feature_spec
        workers.append(worker)

    global_state = {k: v.detach().cpu() for k, v in workers[0].q_net.state_dict().items()}
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    for round_idx in range(num_rounds):
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
        round_losses: List[float] = []
        for worker in workers:
            worker.sync_from_global(global_state)
            state_dict, sample_count, metrics = worker.run_local_steps(local_steps)
            client_updates.append((state_dict, sample_count))
            if metrics["avg_loss"] is not None:
                round_losses.append(metrics["avg_loss"])
            print(
                f"[Round {round_idx + 1}] Client {worker.tls_id} "
                f"steps={sample_count} loss={metrics['avg_loss']} eps={metrics['epsilon']:.3f}"
            )

        global_state = fed_avg(client_updates)
        torch.save(global_state, save_dir_path / f"fed_round_{round_idx + 1}.pt")
        if round_losses:
            print(f"[Round {round_idx + 1}] Mean loss={sum(round_losses)/len(round_losses):.4f}")

    for worker in workers:
        worker.close()

    torch.save(global_state, save_dir_path / "fed_final.pt")
    return global_state


if __name__ == "__main__":
    env_name = "1node"
    tls_action_type = "next_or_not"
    train_federated(
        sumo_cfg=f"Scenario/{env_name}/env/vehicle.sumocfg",
        net_file=f"Scenario/{env_name}/env/{env_name}.net.xml",
        trip_info=None,
        num_seconds=500,
        tls_action_type=tls_action_type,
        num_rounds=3,
        local_steps=200,
        neighbor_hop_k=1,
        hidden_dim=128,
        heads=4,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        warmup_steps=50,
        target_update_interval=200,
        tau=0.0,
        grad_clip_norm=5.0,
        replay_capacity=5000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        max_steps_per_ep=300,
        log_dir=f"logs/{env_name}/fed",
        save_dir=f"Models/result/{env_name}/fed_{tls_action_type}",
        device=None,
    )
