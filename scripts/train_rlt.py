"""RLT training script.

Phase 1: Train RL token encoder-decoder on demonstration data.
Phase 2: Train actor-critic with TD3 or PPO on the RL token representation.

Usage:
  # Phase 1: Train RL token encoder-decoder
  uv run scripts/train_rlt.py --phase 1 --vla-checkpoint-path <path> --vla-config-name <config>

  # Phase 2: Train RL actor-critic with TD3 (requires user-provided env)
  uv run scripts/train_rlt.py --phase 2 --rl-algorithm td3 --phase1-ckpt <path>

  # Both phases sequentially
  uv run scripts/train_rlt.py --phase both --vla-checkpoint-path <path> --vla-config-name <config>
"""

import dataclasses
import logging
import os
import time
from typing import Literal

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm
import tyro
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.rlt.actor import RLTActor
from openpi.rlt.config import RLTTrainConfig
from openpi.rlt.critic import RLTCritic
from openpi.rlt.encoder_decoder import RLTokenEncoderDecoder
from openpi.rlt.replay_buffer import ReplayBuffer
from openpi.rlt.td3 import TD3
from openpi.rlt.vla_interface import VLAEmbeddingExtractor
import openpi.training.config as _train_config
import openpi.training.data_loader as _data


def init_logging():
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vla_model(
    checkpoint_path: str,
    config_name: str,
    device: torch.device,
) -> VLAEmbeddingExtractor:
    """Load a frozen VLA model and wrap it for embedding extraction."""
    # Get training config to extract model config
    train_config = _train_config.get_config(config_name)
    model_cfg = train_config.model

    if not isinstance(model_cfg, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype="bfloat16",
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            max_token_len=model_cfg.max_token_len,
            paligemma_variant=getattr(model_cfg, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(model_cfg, "action_expert_variant", "gemma_300m"),
            pi05=getattr(model_cfg, "pi05", False),
        )

    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # Load weights
    model_path = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_path):
        safetensors.torch.load_model(pi0_model, model_path)
        logging.info(f"Loaded VLA weights from {model_path}")
    else:
        logging.warning(f"No model.safetensors found at {checkpoint_path}, using random weights")

    return VLAEmbeddingExtractor(pi0_model)


def train_phase1(config: RLTTrainConfig, device: torch.device):
    """Phase 1: Train RL token encoder-decoder on demonstration data.

    Uses the existing openpi data pipeline to iterate over demonstrations,
    extracts VLA embeddings with the frozen model, and trains the encoder-decoder
    to compress and reconstruct those embeddings.
    """
    logging.info("=== Phase 1: RL Token Training ===")

    checkpoint_dir = config.checkpoint_path / "phase1"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build frozen VLA
    vla = build_vla_model(config.vla_checkpoint_path, config.vla_config_name, device)

    # Build encoder-decoder
    enc_dec = RLTokenEncoderDecoder(config.rl_token).to(device)
    optimizer = torch.optim.AdamW(enc_dec.parameters(), lr=config.phase1_lr)

    # Build data loader using existing pipeline
    train_config = _train_config.get_config(config.vla_config_name)
    # Override batch size
    train_config = dataclasses.replace(train_config, batch_size=config.phase1_batch_size)
    loader = _data.create_data_loader(train_config, framework="pytorch", shuffle=True)

    # Wandb
    if config.wandb_enabled:
        wandb.init(
            name=f"{config.exp_name}_phase1",
            config=dataclasses.asdict(config),
            project=config.project_name,
        )

    # Training loop
    enc_dec.train()
    global_step = 0
    start_time = time.time()

    logging.info(f"Training encoder-decoder for {config.phase1_steps} steps, lr={config.phase1_lr}")

    pbar = tqdm.tqdm(total=config.phase1_steps, desc="Phase 1")

    while global_step < config.phase1_steps:
        for observation, _actions in loader:
            if global_step >= config.phase1_steps:
                break

            # Move to device
            observation = jax.tree.map(lambda x: x.to(device), observation)

            # Extract VLA embeddings (frozen, no grad)
            vla_embeddings, pad_mask, _state = vla.extract_embeddings(observation)

            # Forward through encoder-decoder
            _z_rl, rto_loss = enc_dec(vla_embeddings.float(), pad_mask)

            # Backward
            optimizer.zero_grad()
            rto_loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_dec.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            if global_step % config.log_interval == 0:
                elapsed = time.time() - start_time
                logging.info(f"step={global_step} rto_loss={rto_loss.item():.6f} time={elapsed:.1f}s")
                if config.wandb_enabled:
                    wandb.log({"rto_loss": rto_loss.item(), "step": global_step}, step=global_step)
                start_time = time.time()

            # Save checkpoint
            if global_step > 0 and global_step % config.save_interval == 0:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "encoder": enc_dec.encoder.state_dict(),
                        "decoder": enc_dec.decoder.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                        "config": dataclasses.asdict(config),
                    },
                    ckpt_path,
                )
                logging.info(f"Saved Phase 1 checkpoint: {ckpt_path}")

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{rto_loss.item():.6f}"})

    pbar.close()

    # Save final checkpoint
    final_path = checkpoint_dir / "final.pt"
    torch.save(
        {
            "encoder": enc_dec.encoder.state_dict(),
            "decoder": enc_dec.decoder.state_dict(),
            "step": global_step,
            "config": dataclasses.asdict(config),
        },
        final_path,
    )
    logging.info(f"Phase 1 complete. Final checkpoint: {final_path}")

    if config.wandb_enabled:
        wandb.finish()

    return enc_dec.encoder


def setup_phase2_td3(config: RLTTrainConfig, encoder, device: torch.device):
    """Set up TD3 components for Phase 2 training.

    Returns the TD3 algorithm, replay buffer, and encoder.
    The environment interaction loop is left to the user.
    """
    logging.info("=== Phase 2 Setup: TD3 ===")

    # Create actor and critic
    actor = RLTActor(config.actor)
    critic = RLTCritic(config.critic)

    # Create TD3
    td3 = TD3(
        actor=actor,
        critic=critic,
        config=config.td3,
        lr_actor=config.phase2_lr_actor,
        lr_critic=config.phase2_lr_critic,
        device=device,
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        config=config.replay_buffer,
        rl_token_dim=config.actor.rl_token_dim,
        proprio_dim=config.actor.proprio_dim,
        action_dim=config.actor.action_dim,
        action_chunk=config.actor.action_chunk,
    )

    logging.info(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    logging.info(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")
    logging.info(f"Replay buffer capacity: {config.replay_buffer.capacity:,}")

    return td3, replay_buffer


def demo_phase2_td3(config: RLTTrainConfig, device: torch.device):
    """Demonstrate Phase 2 TD3 training with synthetic data.

    This function shows the complete training loop structure. In practice,
    replace the synthetic data with real environment interactions.
    """
    logging.info("=== Phase 2 Demo: TD3 with Synthetic Data ===")

    checkpoint_dir = config.checkpoint_path / "phase2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load encoder from Phase 1
    phase1_path = config.checkpoint_path / "phase1" / "final.pt"
    from openpi.rlt.encoder_decoder import RLTokenEncoder  # noqa: PLC0415

    encoder = RLTokenEncoder(config.rl_token).to(device)
    if phase1_path.exists():
        ckpt = torch.load(phase1_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        logging.info(f"Loaded Phase 1 encoder from {phase1_path}")
    else:
        logging.warning("No Phase 1 checkpoint found, using random encoder")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Setup TD3
    td3, replay_buffer = setup_phase2_td3(config, encoder, device)

    # Fill replay buffer with synthetic warmup data
    logging.info(f"Filling replay buffer with {config.phase2_warmup_steps} warmup transitions...")
    rl_dim = config.actor.rl_token_dim
    proprio_dim = config.actor.proprio_dim
    chunk_len = config.actor.action_chunk
    act_dim = config.actor.action_dim

    for _ in range(config.phase2_warmup_steps):
        replay_buffer.add(
            z_rl=torch.randn(rl_dim),
            proprio=torch.randn(proprio_dim),
            actions=torch.randn(chunk_len, act_dim),
            ref_actions=torch.randn(chunk_len, act_dim),
            rewards=torch.randn(1),
            next_z_rl=torch.randn(rl_dim),
            next_proprio=torch.randn(proprio_dim),
            dones=torch.zeros(1),
        )

    # Training loop (demonstrating UTD ratio)
    num_updates = 1000
    logging.info(f"Running {num_updates} TD3 updates with UTD={config.td3.utd_ratio}")

    if config.wandb_enabled:
        wandb.init(
            name=f"{config.exp_name}_phase2_td3",
            config=dataclasses.asdict(config),
            project=config.project_name,
        )

    pbar = tqdm.tqdm(total=num_updates, desc="Phase 2 TD3")
    for step in range(num_updates):
        # Multiple gradient updates per environment step (UTD ratio)
        all_metrics = {}
        for _ in range(config.td3.utd_ratio):
            metrics = td3.update(replay_buffer, config.phase2_batch_size)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)

        if step % config.log_interval == 0:
            avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
            log_str = " ".join(f"{k}={v:.4f}" for k, v in avg_metrics.items())
            logging.info(f"step={step} {log_str}")
            if config.wandb_enabled:
                wandb.log(avg_metrics, step=step)

        if step > 0 and step % config.save_interval == 0:
            td3.save(str(checkpoint_dir / f"td3_step_{step}.pt"))

        pbar.update(1)

    pbar.close()

    # Save final
    td3.save(str(checkpoint_dir / "td3_final.pt"))
    logging.info(f"Phase 2 TD3 complete. Saved to {checkpoint_dir / 'td3_final.pt'}")

    if config.wandb_enabled:
        wandb.finish()


def demo_phase2_ppo(config: RLTTrainConfig, device: torch.device):
    """Demonstrate Phase 2 PPO training setup.

    Shows how to create the RSL-RL PPO components. Actual training requires
    a VecEnv-compatible environment.
    """
    logging.info("=== Phase 2 Demo: PPO Setup ===")

    from openpi.rlt.ppo_wrapper import RLTPPOActorCritic  # noqa: PLC0415
    from openpi.rlt.ppo_wrapper import RLTValueNetwork  # noqa: PLC0415
    from openpi.rlt.ppo_wrapper import create_ppo_algorithm  # noqa: PLC0415

    # Create actor
    actor = RLTActor(config.actor).to(device)

    # Create value network
    value_net = RLTValueNetwork(
        rl_token_dim=config.actor.rl_token_dim,
        proprio_dim=config.actor.proprio_dim,
        hidden_dims=config.actor.hidden_dims,
    ).to(device)

    # Create PPO actor-critic wrapper
    actor_critic = RLTPPOActorCritic(
        actor=actor,
        value_net=value_net,
        rl_token_dim=config.actor.rl_token_dim,
        proprio_dim=config.actor.proprio_dim,
        action_dim=config.actor.action_dim,
        action_chunk=config.actor.action_chunk,
    ).to(device)

    # Create PPO algorithm
    ppo = create_ppo_algorithm(actor_critic, config.ppo)

    logging.info("PPO algorithm created successfully.")
    logging.info(f"Actor-Critic params: {sum(p.numel() for p in actor_critic.parameters()):,}")
    logging.info(
        "To train with PPO, provide a VecEnv-compatible environment and use "
        "rsl_rl.runners.OnPolicyRunner for the training loop."
    )
    logging.info("Pack observations with RLTPPOActorCritic.pack_observation(z_rl, proprio, ref_actions)")

    return ppo, actor_critic


_DEFAULT_CONFIG = RLTTrainConfig()


def main(
    phase: Literal["1", "2", "both"] = "both",
    config: RLTTrainConfig = _DEFAULT_CONFIG,
):
    """RLT training entrypoint.

    Args:
        phase: Which phase to run ("1", "2", or "both")
        config: RLT training configuration
    """
    init_logging()
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    logging.info(f"Phase: {phase}, RL algorithm: {config.rl_algorithm}")

    if phase in ("1", "both"):
        if not config.vla_checkpoint_path:
            logging.error("--config.vla-checkpoint-path is required for Phase 1")
            return
        train_phase1(config, device)

    if phase in ("2", "both"):
        if config.rl_algorithm == "td3":
            demo_phase2_td3(config, device)
        elif config.rl_algorithm == "ppo":
            demo_phase2_ppo(config, device)
        else:
            logging.error(f"Unknown RL algorithm: {config.rl_algorithm}")


if __name__ == "__main__":
    tyro.cli(main)
