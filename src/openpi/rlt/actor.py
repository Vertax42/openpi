"""RLT Actor network.

Gaussian actor pi_theta that predicts action chunks conditioned on the RL token,
proprioceptive state, and reference VLA actions. Implements reference action dropout
to prevent the actor from simply copying the VLA's predictions.
"""

import torch
from torch import Tensor
from torch import nn


class RLTActor(nn.Module):
    """Gaussian actor pi_theta(.|x, a_tilde) for RLT.

    Predicts action chunks conditioned on:
    - z_rl: RL token embedding from the encoder
    - proprio: proprioceptive state (joint positions, velocities, etc.)
    - ref_actions: reference VLA action chunk (with dropout during training)

    Output: N(mu_theta(x, a_tilde), sigma^2 I)
    """

    def __init__(self, config):
        super().__init__()
        self.action_dim = config.action_dim
        self.action_chunk = config.action_chunk
        self.reference_dropout = config.reference_dropout
        self.beta = config.beta_regularization

        input_dim = config.rl_token_dim + config.proprio_dim + config.action_chunk * config.action_dim

        # Build MLP
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
        layers.append(nn.Linear(prev_dim, config.action_chunk * config.action_dim))
        self.mlp = nn.Sequential(*layers)

        # Fixed log standard deviation
        self.log_std = nn.Parameter(
            torch.full((config.action_chunk * config.action_dim,), torch.tensor(config.fixed_std).log()),
            requires_grad=False,
        )

    def forward(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_actions: Tensor,
        apply_ref_dropout: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Sample actions from the policy.

        Args:
            z_rl: [B, D] RL token embedding
            proprio: [B, p] proprioceptive state
            ref_actions: [B, C, d] reference VLA action chunk
            apply_ref_dropout: whether to apply reference action dropout

        Returns:
            actions: [B, C, d] sampled action chunk
            log_probs: [B] log probabilities
        """
        bsz = z_rl.shape[0]
        ref_flat = ref_actions.reshape(bsz, -1)  # [B, C*d]

        # Reference action dropout: zero out entire reference with probability p
        if self.training and apply_ref_dropout and self.reference_dropout > 0:
            dropout_mask = (torch.rand(bsz, 1, device=ref_flat.device) > self.reference_dropout).float()
            ref_flat = ref_flat * dropout_mask

        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mu = self.mlp(x)  # [B, C*d]

        # Sample from Gaussian
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        actions_flat = dist.rsample()  # [B, C*d]
        log_probs = dist.log_prob(actions_flat).sum(dim=-1)  # [B]

        actions = actions_flat.reshape(bsz, self.action_chunk, self.action_dim)
        return actions, log_probs

    def get_mean_action(self, z_rl: Tensor, proprio: Tensor, ref_actions: Tensor) -> Tensor:
        """Get deterministic mean action (for evaluation).

        Args:
            z_rl: [B, D] RL token
            proprio: [B, p] proprioceptive state
            ref_actions: [B, C, d] reference actions

        Returns:
            actions: [B, C, d] mean action chunk
        """
        bsz = z_rl.shape[0]
        ref_flat = ref_actions.reshape(bsz, -1)
        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mu = self.mlp(x)
        return mu.reshape(bsz, self.action_chunk, self.action_dim)
