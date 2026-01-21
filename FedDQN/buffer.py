from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch


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
