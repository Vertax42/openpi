"""Off-policy replay buffer for RLT.

Stores chunk-level transitions for TD3 training. Each transition represents
one action chunk execution (C timesteps) with the associated RL token state,
proprioceptive state, reference actions, and reward.
"""

import torch
from torch import Tensor


class ReplayBuffer:
    """Pre-allocated CPU replay buffer for chunk-level transitions."""

    def __init__(self, config, rl_token_dim: int, proprio_dim: int, action_dim: int, action_chunk: int):
        self.capacity = config.capacity
        self.rl_token_dim = rl_token_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_chunk = action_chunk

        # Pre-allocate CPU tensors
        self.z_rl = torch.zeros(self.capacity, rl_token_dim)
        self.proprio = torch.zeros(self.capacity, proprio_dim)
        self.actions = torch.zeros(self.capacity, action_chunk, action_dim)
        self.ref_actions = torch.zeros(self.capacity, action_chunk, action_dim)
        self.rewards = torch.zeros(self.capacity, 1)
        self.next_z_rl = torch.zeros(self.capacity, rl_token_dim)
        self.next_proprio = torch.zeros(self.capacity, proprio_dim)
        self.dones = torch.zeros(self.capacity, 1)

        self._size = 0
        self._ptr = 0

    def add(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        actions: Tensor,
        ref_actions: Tensor,
        rewards: Tensor,
        next_z_rl: Tensor,
        next_proprio: Tensor,
        dones: Tensor,
    ):
        """Add a single transition to the buffer.

        All inputs should be 1D/2D tensors (no batch dimension) or batched.
        """
        if z_rl.dim() == 1:
            # Single transition
            self.z_rl[self._ptr] = z_rl.cpu()
            self.proprio[self._ptr] = proprio.cpu()
            self.actions[self._ptr] = actions.cpu()
            self.ref_actions[self._ptr] = ref_actions.cpu()
            self.rewards[self._ptr] = rewards.cpu()
            self.next_z_rl[self._ptr] = next_z_rl.cpu()
            self.next_proprio[self._ptr] = next_proprio.cpu()
            self.dones[self._ptr] = dones.cpu()

            self._ptr = (self._ptr + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)
        else:
            # Batched transitions
            batch_size = z_rl.shape[0]
            for i in range(batch_size):
                self.z_rl[self._ptr] = z_rl[i].cpu()
                self.proprio[self._ptr] = proprio[i].cpu()
                self.actions[self._ptr] = actions[i].cpu()
                self.ref_actions[self._ptr] = ref_actions[i].cpu()
                self.rewards[self._ptr] = rewards[i].cpu()
                self.next_z_rl[self._ptr] = next_z_rl[i].cpu()
                self.next_proprio[self._ptr] = next_proprio[i].cpu()
                self.dones[self._ptr] = dones[i].cpu()

                self._ptr = (self._ptr + 1) % self.capacity
                self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, Tensor]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of transitions to sample
            device: Target device for tensors

        Returns:
            Dictionary of batched tensors on the target device
        """
        indices = torch.randint(0, self._size, (batch_size,))
        return {
            "z_rl": self.z_rl[indices].to(device),
            "proprio": self.proprio[indices].to(device),
            "actions": self.actions[indices].to(device),
            "ref_actions": self.ref_actions[indices].to(device),
            "rewards": self.rewards[indices].to(device),
            "next_z_rl": self.next_z_rl[indices].to(device),
            "next_proprio": self.next_proprio[indices].to(device),
            "dones": self.dones[indices].to(device),
        }

    def __len__(self) -> int:
        return self._size
