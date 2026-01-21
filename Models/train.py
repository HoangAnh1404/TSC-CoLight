from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
import sys

# Ensure project root is on sys.path for absolute imports like Models/Env
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.custom_network import GAT, ObservationEncoder, pack_state_dict
from Env.road_graph_builder import RoadGraphBuilder
from Env.make_env import make_multi_envs


def reorder_state(
    state_dict: Dict[str, Any],
    tls_id_to_idx: Dict[str, int],
    node_id_list: List[str],
    phase_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Reorder occupancy+phase into x[N,F] using node_id_list/tls_id_to_idx mapping."""
    x = pack_state_dict(state_dict, tls_id_to_idx=tls_id_to_idx, phase_dim=phase_dim, device=device)
    # Ensure order strictly follows node_id_list
    order = torch.tensor([tls_id_to_idx[tls_id] for tls_id in node_id_list], device=device, dtype=torch.long)
    return x[order]


def build_action_dict(actions: torch.Tensor, node_id_list: List[str]) -> Dict[str, int]:
    """Synchronize actions across TLS using node_id_list ordering."""
    actions = actions.view(-1)
    if actions.numel() != len(node_id_list):
        raise ValueError(f"Expected {len(node_id_list)} actions, got {actions.numel()}.")
    return {tls_id: int(actions[i].item()) for i, tls_id in enumerate(node_id_list)}


def apply_action_mask(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Set invalid action Q to -inf before argmax. action_mask: 1/0 or bool."""
    mask = action_mask.to(dtype=q_values.dtype)
    invalid = mask < 0.5
    q_values = q_values.masked_fill(invalid, float("-inf"))
    return q_values


class ReplayBufferGraph:
    """Replay buffer storing synchronous TLS transitions without edge duplication."""

    def __init__(self, capacity: int, device: Optional[torch.device] = None) -> None:
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self.ptr = 0
        self.full = False

        self.states: List[torch.Tensor] = []
        self.next_states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_x: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Add a synchronous transition (all TLS decide together)."""
        x = x.detach().to(self.device)
        next_x = next_x.detach().to(self.device)
        action = action.detach().to(self.device).long().view(-1)
        reward = reward.detach().to(self.device).view(-1)
        done = done.detach().to(self.device).bool().view(-1)

        if len(self.states) < self.capacity:
            self.states.append(x)
            self.next_states.append(next_x)
            self.actions.append(action)
            self.rewards.append(reward)
            self.dones.append(done)
        else:
            self.states[self.ptr] = x
            self.next_states[self.ptr] = next_x
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        if len(self) == 0:
            raise ValueError("Replay buffer is empty.")
        batch_size = min(batch_size, len(self))
        indices = random.sample(range(len(self)), k=batch_size)

        x_batch = torch.stack([self.states[i] for i in indices])
        next_x_batch = torch.stack([self.next_states[i] for i in indices])
        actions_batch = torch.stack([self.actions[i] for i in indices])
        rewards_batch = torch.stack([self.rewards[i] for i in indices])
        dones_batch = torch.stack([self.dones[i] for i in indices])
        return x_batch, actions_batch, rewards_batch, next_x_batch, dones_batch


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
        optimizer_builder: Callable[[nn.Module], torch.optim.Optimizer],
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

    def act(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None, epsilon: float = 0.0) -> torch.Tensor:
        """Epsilon-greedy with action mask applied before argmax. Supports [N,F] or [B,N,F] inputs."""
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

        if random.random() < epsilon:
            logits = torch.rand((batch_size, num_nodes, self.action_dim), device=self.device)
        else:
            with torch.no_grad():
                q_raw = self.q_network(x_flat, edge_index)
            logits = q_raw.view(batch_size, num_nodes, self.action_dim)

        if action_mask is not None:
            if action_mask.dim() == 2 and batch_size > 1:
                action_mask = action_mask.unsqueeze(0).expand(batch_size, -1, -1)
            logits = apply_action_mask(logits, action_mask.to(self.device))

        actions = torch.argmax(logits, dim=-1)
        return actions.view(1, num_nodes) if batch_size == 1 else actions

    @staticmethod
    def _batch_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        offsets = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * num_nodes
        edge_index_b = edge_index.unsqueeze(0) + offsets  # [B, 2, E]
        return edge_index_b.view(2, -1)

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
                q_next_online = apply_action_mask(q_next_online, action_mask_next)
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


class QNetwork(nn.Module):
    """Observation encoder + GAT encoder + MLP head for per-node Q-values."""

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
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(
            occupancy_dim=occupancy_dim,
            phase_dim=phase_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation="gelu",
            use_residual=True,
        )
        gat_out_dim = hidden_dim  # keep attention space rich; project to Q later
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
        self.q_head = nn.Linear(gat_out_dim, action_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = self.gat(h, edge_index)
        q = self.q_head(h)
        return q


tls_action_type = 'next_or_not'
env_name = 'test'

def train(
    sumo_cfg: str = f"Scenario/{env_name}/env/vehicle.sumocfg",
    net_file: str = f"Scenario/{env_name}/env/{env_name}.net.xml",
    trip_info: str = None,
    num_seconds: int = 500,
    log_path: str = f"logs/{env_name}/{tls_action_type}",
    num_episodes: int = 2000,
    max_steps_per_ep: int = 500,
    total_train_steps: Optional[int] = None,
    hidden_dim: int = 128,
    heads: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 128,
    warmup_steps: int = 50,
    target_update_interval: int = 200,
    tau: float = 0.0,
    grad_clip_norm: float = 5.0,
    device: Optional[str] = None,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5000,
    save_dir: str = f"Models/result/{env_name}/{tls_action_type}",
    save_every_episodes: int = 100,
    parallel_envs: int = 5,
) -> None:
    """
    Minimal training loop for Scenario/{env_name}/env/vehicle.sumocfg.
    """
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log_dir = Path(log_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    num_envs = max(1, parallel_envs)

    # Build static graph and edge_index with reverse edges + self-loops
    graph = RoadGraphBuilder.build_from_net_file(
        net_file=net_file,
        directed=True,
        make_bidirectional=False,
        include_self_loops=False,
        max_hops_between_tls=1,
        neighbor_strategy="hop",
        neighbor_hop_k=1,
        neighbor_top_k=None,
        include_self_in_neighbor=True,
    )
    node_id_list = graph.node_id_list or graph.idx_to_tls_id
    edge_index = graph.neighbors_edge_index(add_reverse=True, add_self_loops=True)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.to(device_t)

    # Env pool with dedicated monitor logs per instance
    envs = []
    for env_idx in range(num_envs):
        log_file = log_dir / f"{env_idx}.monitor.csv"
        envs.append(
            make_multi_envs(
                tls_ids=node_id_list,
                sumo_cfg=sumo_cfg,
                num_seconds=num_seconds,
                use_gui=False,
                net_file=net_file,
                trip_info=trip_info,
                tls_action_type=tls_action_type,
                log_path=str(log_file),
            )
        )

    # Probe initial state to infer occupancy_dim and phase_dim
    init_state, _ = envs[0].reset()
    sample_entry = next(iter(init_state.values()))
    occupancy_dim = len(sample_entry.get("occupancy", []))
    phase_dim = len(sample_entry.get("phase", []))

    action_dim = 2 if tls_action_type=='next_or_not' else max(1, phase_dim * 2)  # +delta/-delta per phase

    def optimizer_builder(model: nn.Module):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)

    q_net = QNetwork(
        occupancy_dim=occupancy_dim,
        phase_dim=phase_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        heads=heads,
        tau=1.0,
        dropout=0.1,
    ).to(device_t)
    target_net = QNetwork(
        occupancy_dim=occupancy_dim,
        phase_dim=phase_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        heads=heads,
        tau=1.0,
        dropout=0.1,
    ).to(device_t)

    config = DQNConfig(
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        huber_delta=1.0,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
    )
    agent = GraphDQNAgent(
        q_network=q_net,
        target_network=target_net,
        edge_index=edge_index,
        action_dim=action_dim,
        config=config,
        optimizer_builder=optimizer_builder,
        device=device_t,
    )

    replay = ReplayBufferGraph(capacity=5000, device=device_t)
    num_nodes = len(node_id_list)

    def init_env_state(env, init_state=None, episode_idx: int = 0):
        state = init_state
        if state is None:
            state, _ = env.reset()
        x_tensor = reorder_state(
            state,
            tls_id_to_idx=graph.tls_id_to_idx,
            node_id_list=node_id_list,
            phase_dim=phase_dim,
            device=device_t,
        )
        return {
            "env": env,
            "x": x_tensor,
            "done_mask": torch.zeros(num_nodes, dtype=torch.bool, device=device_t),
            "action_mask": torch.ones((num_nodes, action_dim), device=device_t),
            "episode_step": 0,
            "episode_idx": episode_idx,
        }

    env_states = [init_env_state(envs[0], init_state=init_state, episode_idx=0)]
    for env in envs[1:]:
        env_states.append(init_env_state(env, episode_idx=0))

    global_step = 0
    episodes_finished = 0
    stop_training = False

    def save_checkpoint(tag: str) -> None:
        torch.save(q_net.state_dict(), save_path / f"q_net_{tag}.pt")
        torch.save(target_net.state_dict(), save_path / f"target_net_{tag}.pt")

    while not stop_training and episodes_finished < num_episodes:
        active_indices = []
        x_batch = []
        action_mask_batch = []

        for idx, env_state in enumerate(env_states):
            if env_state["episode_step"] >= max_steps_per_ep or env_state["done_mask"].all():
                episodes_finished += 1
                print(
                    f"Env {idx} episode {env_state['episode_idx'] + 1} finished at step {env_state['episode_step']}, "
                    f"global_step {global_step}, epsilon {max(epsilon_end, epsilon_start - global_step / epsilon_decay):.3f}"
                )
                if episodes_finished % save_every_episodes == 0:
                    save_checkpoint(f"ep{episodes_finished}")
                if episodes_finished >= num_episodes or (total_train_steps is not None and global_step >= total_train_steps):
                    stop_training = True
                    break
                env_states[idx] = init_env_state(env_state["env"], episode_idx=env_state["episode_idx"] + 1)
                env_state = env_states[idx]

            active_indices.append(idx)
            x_batch.append(env_state["x"])
            action_mask_batch.append(env_state["action_mask"])

        if stop_training or not active_indices:
            break

        x_batch_t = torch.stack(x_batch, dim=0)
        action_mask_t = torch.stack(action_mask_batch, dim=0)
        epsilon = max(epsilon_end, epsilon_start - global_step / epsilon_decay)
        actions_batch = agent.act(x_batch_t, action_mask=action_mask_t, epsilon=epsilon)

        for batch_idx, env_idx in enumerate(active_indices):
            env_state = env_states[env_idx]
            actions_tensor = actions_batch[batch_idx]
            actions_dict = build_action_dict(actions_tensor, node_id_list)

            next_state, reward_dict, truncated_dict, done_dict, infos = env_state["env"].step(actions_dict)
            rewards = torch.tensor(
                [reward_dict[tls] for tls in node_id_list],
                device=device_t,
                dtype=torch.float32,
            )
            dones = torch.tensor(
                [done_dict[tls] or truncated_dict[tls] for tls in node_id_list],
                device=device_t,
                dtype=torch.bool,
            )
            next_x = reorder_state(
                next_state,
                tls_id_to_idx=graph.tls_id_to_idx,
                node_id_list=node_id_list,
                phase_dim=phase_dim,
                device=device_t,
            )
            next_action_mask = torch.tensor(
                [infos.get(tls, {}).get("can_perform_action", True) for tls in node_id_list],
                device=device_t,
                dtype=torch.float32,
            ).unsqueeze(-1).expand(-1, action_dim)

            replay.add(env_state["x"], actions_tensor, rewards, next_x, dones)
            loss = agent.optimize(replay, action_mask_next=None)

            env_state["x"] = next_x
            env_state["done_mask"] = dones
            env_state["action_mask"] = next_action_mask
            env_state["episode_step"] += 1
            env_states[env_idx] = env_state
            global_step += 1

            if loss is not None and global_step % 100 == 0:
                print(
                    f"Env {env_idx} Ep {env_state['episode_idx'] + 1} Step {env_state['episode_step']} "
                    f"Global {global_step} Loss {loss:.4f} Epsilon {epsilon:.3f}"
                )

            if total_train_steps is not None and global_step >= total_train_steps:
                stop_training = True
                break

    for env in envs:
        env.close()
    save_checkpoint("final")


if __name__ == "__main__":
    train()
