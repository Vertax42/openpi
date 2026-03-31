"""PPO wrapper for RSL-RL integration.

Adapts the RLT actor and a value network to the RSL-RL ActorCritic interface.
RSL-RL is an optional dependency - import errors are caught with clear instructions.
"""

import torch
from torch import Tensor
from torch import nn


class RLTValueNetwork(nn.Module):
    """State value function V(x) for PPO."""

    def __init__(self, rl_token_dim: int, proprio_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        input_dim = rl_token_dim + proprio_dim

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
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

    def forward(self, z_rl: Tensor, proprio: Tensor) -> Tensor:
        x = torch.cat([z_rl, proprio], dim=-1)
        return self.mlp(x)


class RLTPPOActorCritic(nn.Module):
    """Adapts RLT actor + value network to RSL-RL's ActorCritic interface.

    Packs multi-part state (z_rl, proprio, ref_actions_flat) into a single
    flat observation vector for RSL-RL compatibility.
    """

    def __init__(self, actor, value_net, rl_token_dim: int, proprio_dim: int, action_dim: int, action_chunk: int):
        super().__init__()
        self.actor = actor
        self.value_net = value_net
        self.rl_token_dim = rl_token_dim
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.ref_dim = action_chunk * action_dim

    def _unpack(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Unpack flat observation vector into (z_rl, proprio, ref_actions).

        The observation is packed as [z_rl | proprio | ref_actions_flat].
        """
        idx1 = self.rl_token_dim
        idx2 = idx1 + self.proprio_dim
        z_rl = observations[:, :idx1]
        proprio = observations[:, idx1:idx2]
        ref_flat = observations[:, idx2:]
        ref_actions = ref_flat.reshape(-1, self.action_chunk, self.action_dim)
        return z_rl, proprio, ref_actions

    @staticmethod
    def pack_observation(z_rl: Tensor, proprio: Tensor, ref_actions: Tensor) -> Tensor:
        """Pack multi-part state into flat observation for RSL-RL.

        Args:
            z_rl: [B, D] RL token
            proprio: [B, p] proprioceptive state
            ref_actions: [B, C, d] reference actions

        Returns:
            observation: [B, D+p+C*d] flat observation
        """
        bsz = z_rl.shape[0]
        ref_flat = ref_actions.reshape(bsz, -1)
        return torch.cat([z_rl, proprio, ref_flat], dim=-1)

    def act(self, observations: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """RSL-RL rollout interface.

        Args:
            observations: [B, obs_dim] flat packed observations

        Returns:
            actions_flat: [B, C*d] flat actions
            log_probs: [B] log probabilities
            values: [B, 1] value estimates
            mu_flat: [B, C*d] mean actions
        """
        z_rl, proprio, ref_actions = self._unpack(observations)

        # Sample actions
        actions, log_probs = self.actor(z_rl, proprio, ref_actions, apply_ref_dropout=True)
        mu = self.actor.get_mean_action(z_rl, proprio, ref_actions)

        # Value estimate
        values = self.value_net(z_rl, proprio)

        bsz = actions.shape[0]
        return actions.reshape(bsz, -1), log_probs, values, mu.reshape(bsz, -1)

    def evaluate(self, observations: Tensor, actions_flat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """RSL-RL PPO update interface.

        Args:
            observations: [B, obs_dim] flat packed observations
            actions_flat: [B, C*d] flat actions taken

        Returns:
            log_probs: [B] log probabilities under current policy
            entropy: [B] entropy of the policy
            values: [B, 1] value estimates
        """
        z_rl, proprio, ref_actions = self._unpack(observations)

        # Reconstruct distribution
        bsz = actions_flat.shape[0]
        ref_flat = ref_actions.reshape(bsz, -1)

        # No dropout during evaluation
        x = torch.cat([z_rl, proprio, ref_flat], dim=-1)
        mu = self.actor.mlp(x)
        std = self.actor.log_std.exp()
        dist = torch.distributions.Normal(mu, std)

        log_probs = dist.log_prob(actions_flat).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value_net(z_rl, proprio)

        return log_probs, entropy, values


def create_ppo_algorithm(actor_critic: RLTPPOActorCritic, config):
    """Create RSL-RL PPO algorithm instance.

    Args:
        actor_critic: RLTPPOActorCritic instance
        config: PPOConfig with hyperparameters

    Returns:
        PPO algorithm instance

    Raises:
        ImportError: If rsl-rl-lib is not installed
    """
    try:
        from rsl_rl.algorithms import PPO  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "RSL-RL is required for PPO training. Install it with:\n"
            "  pip install rsl-rl-lib>=5.0.1\n"
            "or:\n"
            "  uv pip install rsl-rl-lib>=5.0.1"
        ) from None

    return PPO(
        actor_critic=actor_critic,
        num_learning_epochs=config.num_learning_epochs,
        num_mini_batches=config.num_mini_batches,
        clip_param=config.clip_param,
        gamma=config.gamma,
        lam=config.lam,
        value_loss_coef=config.value_loss_coef,
        entropy_coef=config.entropy_coef,
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
    )
