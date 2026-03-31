"""TD3 algorithm for RLT.

Implements Twin Delayed DDPG with paper-specific modifications:
- Reference action regularization: beta * ||a - a_tilde||^2
- Chunk-level discount: gamma^C for action chunk transitions
- Reference action pass-through and dropout in actor
"""

import copy

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812

from openpi.rlt.actor import RLTActor
from openpi.rlt.config import TD3Config
from openpi.rlt.critic import RLTCritic
from openpi.rlt.replay_buffer import ReplayBuffer


class TD3:
    """TD3 algorithm with RLT-specific modifications."""

    def __init__(
        self,
        actor: RLTActor,
        critic: RLTCritic,
        config: TD3Config,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.config = config
        self.device = device

        # Target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target.eval()
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self._update_count = 0
        # Chunk-level discount factor: gamma^C
        self._chunk_discount = config.discount**actor.action_chunk

    @torch.no_grad()
    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module):
        """Polyak averaging for target network update."""
        tau = self.config.tau
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> dict[str, float]:
        """Perform one TD3 update step.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Batch size for update

        Returns:
            Dictionary of training metrics
        """
        metrics = {}

        batch = replay_buffer.sample(batch_size, self.device)
        z_rl = batch["z_rl"]
        proprio = batch["proprio"]
        actions = batch["actions"]
        ref_actions = batch["ref_actions"]
        rewards = batch["rewards"]
        next_z_rl = batch["next_z_rl"]
        next_proprio = batch["next_proprio"]
        dones = batch["dones"]

        # --- Critic update ---
        with torch.no_grad():
            # Target action with smoothing noise
            next_actions, _ = self.actor_target(next_z_rl, next_proprio, ref_actions, apply_ref_dropout=False)
            # Add clipped noise for target policy smoothing
            noise = torch.randn_like(next_actions) * self.config.policy_noise
            noise = noise.clamp(-self.config.noise_clip, self.config.noise_clip)
            next_actions = next_actions + noise

            # Clipped double Q: take minimum of two target Q values
            target_q1, target_q2 = self.critic_target(next_z_rl, next_proprio, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # TD target with chunk-level discount
            td_target = rewards + (1 - dones) * self._chunk_discount * target_q

        # Current Q values
        q1, q2 = self.critic(z_rl, proprio, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics["critic_loss"] = critic_loss.item()
        metrics["q1_mean"] = q1.mean().item()
        metrics["q2_mean"] = q2.mean().item()

        # --- Actor update (delayed) ---
        self._update_count += 1
        if self._update_count % self.config.policy_delay == 0:
            # Actor loss: -Q + beta * ||a - a_tilde||^2
            pred_actions, _ = self.actor(z_rl, proprio, ref_actions, apply_ref_dropout=True)
            q_value = self.critic.q1_forward(z_rl, proprio, pred_actions)

            # Reference regularization
            bc_loss = F.mse_loss(pred_actions, ref_actions)
            actor_loss = -q_value.mean() + self.actor.beta * bc_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            metrics["actor_loss"] = actor_loss.item()
            metrics["bc_loss"] = bc_loss.item()
            metrics["q_actor"] = q_value.mean().item()

            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

        return metrics

    def select_action(
        self,
        z_rl: Tensor,
        proprio: Tensor,
        ref_actions: Tensor,
        exploration_noise: float = 0.1,
    ) -> Tensor:
        """Select action for environment interaction.

        Args:
            z_rl: [1, D] or [B, D] RL token
            proprio: [1, p] or [B, p] proprioceptive state
            ref_actions: [1, C, d] or [B, C, d] reference actions
            exploration_noise: std of exploration noise (0 for deterministic)

        Returns:
            actions: [B, C, d] action chunk
        """
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor.get_mean_action(z_rl, proprio, ref_actions)
            if exploration_noise > 0:
                noise = torch.randn_like(actions) * exploration_noise
                actions = actions + noise
        self.actor.train()
        return actions

    def save(self, path: str):
        """Save all networks and optimizer states."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str, device: torch.device | None = None):
        """Load all networks and optimizer states."""
        if device is None:
            device = self.device
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self._update_count = checkpoint["update_count"]
