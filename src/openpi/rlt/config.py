"""Configuration dataclasses for RLT training."""

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class RLTokenConfig:
    """Configuration for the RL token encoder and decoder transformers."""

    vla_embed_dim: int = 2048
    num_layers: int = 2
    num_heads: int = 8
    head_dim: int = 256
    mlp_dim: int = 4096
    dropout: float = 0.0
    max_seq_len: int = 1024


@dataclasses.dataclass(frozen=True)
class ActorConfig:
    """Configuration for the RLT actor network."""

    rl_token_dim: int = 2048
    proprio_dim: int = 32
    action_dim: int = 14
    action_chunk: int = 10
    hidden_dims: tuple[int, ...] = (256, 256)
    fixed_std: float = 0.1
    reference_dropout: float = 0.5
    beta_regularization: float = 1.0


@dataclasses.dataclass(frozen=True)
class CriticConfig:
    """Configuration for the RLT twin Q critic network."""

    rl_token_dim: int = 2048
    proprio_dim: int = 32
    action_dim: int = 14
    action_chunk: int = 10
    hidden_dims: tuple[int, ...] = (256, 256)


@dataclasses.dataclass(frozen=True)
class TD3Config:
    """Configuration for the TD3 algorithm."""

    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    utd_ratio: int = 5
    subsampling_stride: int = 2


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    """Configuration for PPO via RSL-RL."""

    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    learning_rate: float = 3e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0
    num_steps_per_env: int = 24


@dataclasses.dataclass(frozen=True)
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int = 1_000_000


@dataclasses.dataclass(frozen=True)
class RLTTrainConfig:
    """Top-level configuration for RLT training."""

    # Sub-configs
    rl_token: RLTokenConfig = dataclasses.field(default_factory=RLTokenConfig)
    actor: ActorConfig = dataclasses.field(default_factory=ActorConfig)
    critic: CriticConfig = dataclasses.field(default_factory=CriticConfig)
    td3: TD3Config = dataclasses.field(default_factory=TD3Config)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    replay_buffer: ReplayBufferConfig = dataclasses.field(default_factory=ReplayBufferConfig)

    # VLA model
    vla_checkpoint_path: str = ""
    vla_config_name: str = "pi0_droid"

    # RL algorithm selection
    rl_algorithm: str = "td3"  # "td3" or "ppo"

    # Phase 1: RL token training
    phase1_lr: float = 1e-4
    phase1_steps: int = 10000
    phase1_batch_size: int = 32
    vla_finetune_alpha: float = 0.0

    # Phase 2: RL training
    phase2_lr_actor: float = 3e-4
    phase2_lr_critic: float = 3e-4
    phase2_batch_size: int = 256
    phase2_warmup_steps: int = 1000

    # General training
    seed: int = 42
    exp_name: str = "rlt_default"
    checkpoint_dir: str = "checkpoints/rlt"
    save_interval: int = 1000
    log_interval: int = 100

    # Wandb
    wandb_enabled: bool = False
    project_name: str = "openpi-rlt"

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir) / self.exp_name
