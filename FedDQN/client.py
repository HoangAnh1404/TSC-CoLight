from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# Ensure project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Env.make_env import make_multi_envs
from Env.road_graph_builder import GraphSpec, RoadGraphBuilder
from FedDQN.buffer import ReplayBufferGraph
from FedDQN.networks import QNetwork
from FedDQN.utils import build_action_dict, reorder_state


@dataclass
class DQNConfig:
    gamma: float = 0.99
    tau: float = 0.005  # soft update rate; set to 0 to disable
    target_update_interval: int = 0  # set >0 for hard update cadence
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    warmup_steps: int = 1000
    batch_size: int = 64


class GraphDQNAgent:
    """Double DQN with GNN backbone, action masking, and target sync."""

    def __init__(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        edge_index: torch.Tensor,
        action_dim: int,
        config: DQNConfig,
        optimizer_builder: callable,
        device: Optional[torch.device] = None,
    ) -> None:
        self.q_network = q_network
        self.target_network = target_network
        self.edge_index = edge_index
        self.action_dim = action_dim
        self.config = config
        self.device = device or torch.device("cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)
        self.edge_index = self.edge_index.to(self.device)
        self.optimizer = optimizer_builder(self.q_network)
        self.train_steps = 0
        self.hard_update()

    def hard_update(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self) -> None:
        tau = self.config.tau
        for tgt, src in zip(self.target_network.parameters(), self.q_network.parameters()):
            tgt.data.copy_(tgt.data * (1.0 - tau) + src.data * tau)

    @staticmethod
    def _batch_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * num_nodes
        edge_index_b = edge_index.unsqueeze(0) + offsets  # [B, 2, E]
        return edge_index_b.view(2, -1)

    def act(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None, epsilon: float = 0.0) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 2:
            batch_size, num_nodes = 1, x.size(0)
            x_flat = x
            edge_index = self.edge_index
        elif x.dim() == 3:
            batch_size, num_nodes = x.shape[0], x.shape[1]
            x_flat = x.view(batch_size * num_nodes, -1)
            edge_index = self._batch_edge_index(self.edge_index, batch_size, num_nodes)
        else:
            raise ValueError("x must have shape [N, F] or [B, N, F].")

        if torch.rand(1).item() < epsilon:
            logits = torch.rand((batch_size, num_nodes, self.action_dim), device=self.device)
        else:
            with torch.no_grad():
                q_raw = self.q_network(x_flat, edge_index)
            logits = q_raw.view(batch_size, num_nodes, self.action_dim)

        if action_mask is not None:
            if action_mask.dim() == 2 and batch_size > 1:
                action_mask = action_mask.unsqueeze(0).expand(batch_size, -1, -1)
            logits = logits.masked_fill(action_mask.to(self.device) < 0.5, float("-inf"))

        actions = torch.argmax(logits, dim=-1)
        return actions.view(1, num_nodes) if batch_size == 1 else actions

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        action_mask_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x, actions, rewards, next_x, dones = [b.to(self.device) for b in batch]
        batch_size, num_nodes = x.shape[:2]
        edge_b = self._batch_edge_index(self.edge_index, batch_size, num_nodes)

        x_flat = x.view(batch_size * num_nodes, -1)
        next_x_flat = next_x.view(batch_size * num_nodes, -1)

        q_pred = self.q_network(x_flat, edge_b).view(batch_size, num_nodes, -1)
        q_taken = q_pred.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)

        if action_mask_next is not None:
            if action_mask_next.dim() == 2:
                action_mask_next = action_mask_next.unsqueeze(0).expand(batch_size, -1, -1)
            action_mask_next = action_mask_next.to(self.device)

        with torch.no_grad():
            q_next_online = self.q_network(next_x_flat, edge_b).view(batch_size, num_nodes, -1)
            if action_mask_next is not None:
                q_next_online = q_next_online.masked_fill(action_mask_next < 0.5, float("-inf"))
            next_actions = torch.argmax(q_next_online, dim=-1, keepdim=True)
            q_next_target = self.target_network(next_x_flat, edge_b).view(batch_size, num_nodes, -1)
            q_next_target = q_next_target.gather(-1, next_actions).squeeze(-1)
            target = rewards + (1 - dones.float()) * self.config.gamma * q_next_target

        loss = F.huber_loss(q_taken, target, delta=self.config.huber_delta)
        return loss

    def optimize(
        self,
        replay: ReplayBufferGraph,
        action_mask_next: Optional[torch.Tensor] = None,
    ) -> Optional[float]:
        if len(replay) < self.config.warmup_steps:
            return None
        batch = replay.sample(self.config.batch_size)
        loss = self.compute_loss(batch, action_mask_next)

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.grad_clip_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.config.target_update_interval and self.train_steps % self.config.target_update_interval == 0:
            self.hard_update()
        elif self.config.tau > 0:
            self.soft_update()
        return float(loss.item())


class DQNClient:
    """Local trainer for one TLS (or k-hop cluster)."""

    def __init__(
        self,
        tls_id: str,
        graph: GraphSpec,
        env_cfg: Dict[str, Any],
        dqn_config: DQNConfig,
        hidden_dim: int = 128,
        heads: int = 4,
        learning_rate: float = 3e-4,
        dueling: bool = False,
        neighbor_hop_k: int = 2,
        replay_capacity: int = 5000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        max_steps_per_ep: int = 300,
        device: Optional[str] = None,
    ) -> None:
        self.tls_id = tls_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.env_cfg = env_cfg
        self.dqn_config = dqn_config
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dueling = dueling
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_steps_per_ep = max_steps_per_ep

        self.graph_view = self._build_local_graph_view(graph=graph, hop_k=neighbor_hop_k)

        log_path = Path(env_cfg["log_dir"]) / f"{tls_id}.monitor.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.env = make_multi_envs(
            tls_ids=self.graph_view["node_id_list"],
            sumo_cfg=env_cfg["sumo_cfg"],
            num_seconds=env_cfg["num_seconds"],
            use_gui=env_cfg.get("use_gui", False),
            net_file=env_cfg["net_file"],
            trip_info=env_cfg.get("trip_info"),
            tls_action_type=env_cfg["tls_action_type"],
            log_path=str(log_path),
        )

        init_state, _ = self.env.reset()
        self.feature_spec = self._infer_feature_spec(init_state)
        self.action_dim = 2 if env_cfg["tls_action_type"] == "next_or_not" else max(1, self.feature_spec["phase_dim"] * 2)

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
            dueling=dueling,
        ).to(self.device)
        self.target_net = QNetwork(
            occupancy_dim=self.feature_spec["occupancy_dim"],
            phase_dim=self.feature_spec["phase_dim"],
            hidden_dim=hidden_dim,
            action_dim=self.action_dim,
            heads=heads,
            tau=1.0,
            dropout=0.1,
            dueling=dueling,
        ).to(self.device)
        self.agent = GraphDQNAgent(
            q_network=self.q_net,
            target_network=self.target_net,
            edge_index=self.graph_view["edge_index"],
            action_dim=self.action_dim,
            config=dqn_config,
            optimizer_builder=optimizer_builder,
            device=self.device,
        )
        self.replay = ReplayBufferGraph(capacity=replay_capacity, device=self.device)
        self.env_state = self._init_env_state(init_state)
        self.global_step = 0
        self.last_epsilon = epsilon_start

    def _build_local_graph_view(self, graph: GraphSpec, hop_k: int):
        start_idx = graph.tls_id_to_idx[self.tls_id]
        visited = {start_idx}
        frontier = {start_idx}
        for _ in range(max(0, hop_k)):
            nxt = set()
            for idx in frontier:
                nxt.update(graph.neighbors_list[idx])
            nxt -= visited
            if not nxt:
                break
            visited |= nxt
            frontier = nxt
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
            edge_index = torch.as_tensor(sorted(edges), dtype=torch.long, device=self.device).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        tls_id_to_idx = {tls_id: idx for idx, tls_id in enumerate(node_id_list)}
        return {"node_id_list": node_id_list, "tls_id_to_idx": tls_id_to_idx, "edge_index": edge_index}

    def _infer_feature_spec(self, state: Dict[str, Any]) -> Dict[str, int]:
        sample_entry = next(iter(state.values()))
        occupancy_dim = len(sample_entry.get("occupancy", []))
        phase_dim = len(sample_entry.get("phase", []))
        return {"occupancy_dim": occupancy_dim, "phase_dim": phase_dim}

    def _init_env_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        x_tensor = reorder_state(
            state,
            tls_id_to_idx=self.graph_view["tls_id_to_idx"],
            node_id_list=self.graph_view["node_id_list"],
            phase_dim=self.feature_spec["phase_dim"],
            device=self.device,
        )
        num_nodes = len(self.graph_view["node_id_list"])
        return {
            "x": x_tensor,
            "done_mask": torch.zeros(num_nodes, dtype=torch.bool, device=self.device),
            "action_mask": torch.ones((num_nodes, self.action_dim), device=self.device),
            "episode_step": 0,
        }

    def get_policy_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.q_net.state_dict().items()}

    def load_policy_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.q_net.load_state_dict(state, strict=True)
        self.agent.hard_update()

    def train_local(self, num_steps: int) -> Dict[str, Any]:
        steps = 0
        losses: List[float] = []
        while steps < num_steps:
            if self.env_state["episode_step"] >= self.max_steps_per_ep or self.env_state["done_mask"].all():
                state, _ = self.env.reset()
                self.env_state = self._init_env_state(state)

            epsilon = max(
                self.epsilon_end,
                self.epsilon_start - self.global_step / float(self.epsilon_decay),
            )
            actions = self.agent.act(
                self.env_state["x"], action_mask=self.env_state["action_mask"], epsilon=epsilon
            ).view(-1)
            actions_dict = build_action_dict(actions, self.graph_view["node_id_list"])

            next_state, reward_dict, truncated_dict, done_dict, infos = self.env.step(actions_dict)
            rewards = torch.tensor(
                [reward_dict[tls] for tls in self.graph_view["node_id_list"]],
                device=self.device,
                dtype=torch.float32,
            )
            dones = torch.tensor(
                [done_dict[tls] or truncated_dict[tls] for tls in self.graph_view["node_id_list"]],
                device=self.device,
                dtype=torch.bool,
            )
            next_x = reorder_state(
                next_state,
                tls_id_to_idx=self.graph_view["tls_id_to_idx"],
                node_id_list=self.graph_view["node_id_list"],
                phase_dim=self.feature_spec["phase_dim"],
                device=self.device,
            )
            next_action_mask = torch.tensor(
                [infos.get(tls, {}).get("can_perform_action", True) for tls in self.graph_view["node_id_list"]],
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

        return {
            "state_dict": self.get_policy_state(),
            "steps": steps,
            "metrics": {"loss_mean": float(sum(losses) / len(losses)) if losses else None, "epsilon": self.last_epsilon},
        }

    def save_local(self, save_dir: Path, tag: str) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.get_policy_state(), save_dir / f"policy_{tag}.pt")

    def close(self) -> None:
        self.env.close()
