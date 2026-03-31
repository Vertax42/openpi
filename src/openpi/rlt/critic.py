"""RLT Critic network.

Twin Q-function Q_psi(x, a) for TD3-style training. Uses an ensemble of two
Q-networks to mitigate overestimation bias.
"""

import torch
from torch import Tensor
from torch import nn


class QNetwork(nn.Module):
    """Single Q-value network mapping (z_rl, proprio, actions) -> scalar Q."""

    def __init__(self, config):
        super().__init__()
        input_dim = config.rl_token_dim + config.proprio_dim + config.action_chunk * config.action_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> Tensor:
        """Compute Q-value.

        Args:
            z_rl: [B, D] RL token
            proprio: [B, p] proprioceptive state
            actions: [B, C, d] action chunk

        Returns:
            q_value: [B, 1] Q-value estimate
        """
        bsz = z_rl.shape[0]
        actions_flat = actions.reshape(bsz, -1)
        x = torch.cat([z_rl, proprio, actions_flat], dim=-1)
        return self.mlp(x)


class RLTCritic(nn.Module):
    """Twin Q-function for TD3 (clipped double-Q learning)."""

    def __init__(self, config):
        super().__init__()
        self.q1 = QNetwork(config)
        self.q2 = QNetwork(config)

    def forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Compute both Q-values.

        Returns:
            q1: [B, 1] Q1 value
            q2: [B, 1] Q2 value
        """
        return self.q1(z_rl, proprio, actions), self.q2(z_rl, proprio, actions)

    def q1_forward(self, z_rl: Tensor, proprio: Tensor, actions: Tensor) -> Tensor:
        """Compute only Q1 (used in actor loss to save compute)."""
        return self.q1(z_rl, proprio, actions)
