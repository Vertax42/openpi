#!/usr/bin/env python3
"""Utility script to inspect training configurations from checkpoints."""

import argparse
import json
import pathlib
import sys
from typing import Any

import torch


def load_pytorch_checkpoint_metadata(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    """Load metadata from PyTorch checkpoint."""
    metadata_path = checkpoint_dir / "metadata.pt"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    return metadata


def is_jax_checkpoint(checkpoint_dir: pathlib.Path) -> bool:
    """Check if this is a JAX checkpoint."""
    return (checkpoint_dir / "_CHECKPOINT_METADATA").exists()


def load_jax_checkpoint_info(checkpoint_dir: pathlib.Path) -> dict[str, Any]:
    """Load basic info from JAX checkpoint."""
    info = {"checkpoint_type": "JAX"}

    # Read checkpoint metadata
    metadata_path = checkpoint_dir / "_CHECKPOINT_METADATA"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            info["init_timestamp"] = metadata.get("init_timestamp_nsecs")
            info["commit_timestamp"] = metadata.get("commit_timestamp_nsecs")
            info["item_handlers"] = metadata.get("item_handlers", {})

    # Try to get step number from directory name
    step = checkpoint_dir.name
    if step.isdigit():
        info["step"] = int(step)

    return info


def find_wandb_config(
    wandb_id: str, base_dir: pathlib.Path = None
) -> dict[str, Any] | None:
    """Find and load WandB config from local wandb directory."""
    if base_dir is None:
        base_dir = pathlib.Path.cwd()

    # Try to find wandb directory in multiple possible locations
    possible_wandb_dirs = [
        base_dir / "wandb",
        base_dir.parent / "wandb",
        pathlib.Path.cwd() / "wandb",
    ]

    for wandb_dir in possible_wandb_dirs:
        if not wandb_dir.exists():
            continue

        # Find wandb run directory
        run_pattern = f"run-*{wandb_id}*"
        matching_runs = list(wandb_dir.glob(run_pattern))

        if not matching_runs:
            continue

        # Try to load config.yaml
        config_file = matching_runs[0] / "files" / "config.yaml"
        if config_file.exists():
            try:
                import yaml

                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    return config_data
            except Exception as e:
                if base_dir is None or base_dir == pathlib.Path.cwd():
                    print(f"Warning: Could not parse config.yaml: {e}", file=sys.stderr)
                continue

    return None


def print_config(config_dict: dict[str, Any], indent: int = 0) -> None:
    """Print configuration dictionary in a readable format."""
    prefix = "  " * indent

    for key, value in sorted(config_dict.items()):
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        elif isinstance(value, list):
            if len(value) <= 3:
                print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: [{len(value)} items] {value[:2]}... {value[-1]}")
        else:
            print(f"{prefix}{key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect training configuration from a checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect a specific checkpoint
  python scripts/inspect_checkpoint.py checkpoints/pi05_base_arx5_full/xense_bi_arx5_pick_and_place_cube/20000

  # Inspect the latest checkpoint in a training run
  python scripts/inspect_checkpoint.py checkpoints/pi05_base_arx5_full/xense_bi_arx5_pick_and_place_cube/
        """,
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory (can be a specific step or the base directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output configuration as JSON",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Show only key training parameters (batch size, learning rate, etc.)",
    )

    args = parser.parse_args()
    checkpoint_path = pathlib.Path(args.checkpoint_path).resolve()

    # Check if it's a JAX checkpoint first
    is_jax = is_jax_checkpoint(checkpoint_path)
    is_pytorch = (checkpoint_path / "metadata.pt").exists()

    # If it's a base checkpoint directory, try to find the latest checkpoint
    if checkpoint_path.is_dir() and not is_jax and not is_pytorch:
        # Find numeric subdirectories
        checkpoints = sorted(
            [d for d in checkpoint_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name),
            reverse=True,
        )

        if not checkpoints:
            print(
                f"Error: No checkpoint directories found in {checkpoint_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        checkpoint_path = checkpoints[0]
        print(f"Using latest checkpoint: {checkpoint_path}\n")
        # Re-check JAX after selecting checkpoint
        is_jax = is_jax_checkpoint(checkpoint_path)

    # Check if it's a JAX checkpoint (do this before trying to load PyTorch)
    if is_jax:
        jax_info = load_jax_checkpoint_info(checkpoint_path)

        print("=" * 80)
        print("Checkpoint Information")
        print("=" * 80)
        print(f"Checkpoint directory: {checkpoint_path}")
        print(f"Checkpoint type: {jax_info['checkpoint_type']}")
        print(f"Step: {jax_info.get('step', 'unknown')}")

        # Check for wandb ID
        wandb_id = None
        wandb_id_file = checkpoint_path.parent / "wandb_id.txt"
        if wandb_id_file.exists():
            with open(wandb_id_file) as f:
                wandb_id = f.read().strip()
            print(f"WandB ID: {wandb_id}")
            print(f"WandB URL: https://wandb.ai/[ENTITY]/[PROJECT]/runs/{wandb_id}")

        # Try to load config from WandB
        wandb_config = None
        if wandb_id:
            wandb_config = find_wandb_config(wandb_id, checkpoint_path)

        print("\n" + "=" * 80)
        if wandb_config:
            print("Training Configuration (from WandB)")
            print("=" * 80)
            # Extract config values from wandb format
            config_values = {
                k: v.get("value") if isinstance(v, dict) and "value" in v else v
                for k, v in wandb_config.items()
                if not k.startswith("_")
            }

            if args.json:
                import yaml

                print(
                    yaml.dump(
                        config_values, default_flow_style=False, allow_unicode=True
                    )
                )
            elif args.brief:
                key_params = [
                    "exp_name",
                    "name",
                    "batch_size",
                    "num_train_steps",
                    "seed",
                    "lr_schedule",
                    "optimizer",
                    "model",
                ]
                for key in key_params:
                    if key in config_values:
                        value = config_values[key]
                        if isinstance(value, dict) and key == "lr_schedule":
                            print(f"{key}:")
                            print(f"  peak_lr: {value.get('peak_lr')}")
                            print(f"  warmup_steps: {value.get('warmup_steps')}")
                            print(f"  decay_steps: {value.get('decay_steps')}")
                        elif isinstance(value, dict):
                            print(f"{key}: {value}")
                        else:
                            print(f"{key}: {value}")
            else:
                print_config(config_values)
        else:
            print("Note: JAX checkpoint metadata")
            print("=" * 80)
            print(
                "JAX checkpoints don't store training configuration in the checkpoint."
            )
            if wandb_id:
                print(
                    "WandB config was not found locally. To find training parameters:"
                )
            else:
                print("To find training parameters:")
            print("  1. Check the WandB run (if enabled)")
            print("  2. Check the training script arguments/logs")
            print("  3. Look for config files in the checkpoint parent directory")

        if jax_info.get("init_timestamp"):
            print("\nTimestamps:")
            print(f"  Init: {jax_info['init_timestamp']}")
            print(f"  Commit: {jax_info['commit_timestamp']}")

        print("=" * 80)
        return

    # Try to load PyTorch checkpoint
    try:
        metadata = load_pytorch_checkpoint_metadata(checkpoint_path)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(
            "\nThis doesn't appear to be a PyTorch checkpoint.",
            file=sys.stderr,
        )
        if is_jax_checkpoint(checkpoint_path):
            print(
                "This is a JAX checkpoint. Rerunning with JAX support...",
                file=sys.stderr,
            )
        sys.exit(1)

    # Extract information
    global_step = metadata.get("global_step", "unknown")
    timestamp = metadata.get("timestamp", "unknown")
    config = metadata.get("config", {})

    print("=" * 80)
    print("Checkpoint Information")
    print("=" * 80)
    print(f"Checkpoint directory: {checkpoint_path}")
    print(f"Global step: {global_step}")
    print(f"Timestamp: {timestamp}")

    # Check for wandb ID
    wandb_id_file = checkpoint_path.parent / "wandb_id.txt"
    if wandb_id_file.exists():
        with open(wandb_id_file) as f:
            wandb_id = f.read().strip()
        print(f"WandB ID: {wandb_id}")
        print(f"WandB URL: https://wandb.ai/[ENTITY]/[PROJECT]/runs/{wandb_id}")

    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)

    if args.json:
        # Convert all values to JSON-serializable format
        def convert_to_json(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json(item) for item in obj]
            elif hasattr(obj, "__dict__"):
                return str(obj)
            else:
                return obj

        config_json = convert_to_json(config)
        print(json.dumps(config_json, indent=2))
    elif args.brief:
        # Show only key parameters
        key_params = [
            "exp_name",
            "name",
            "batch_size",
            "num_train_steps",
            "learning_rate",
            "warmup_steps",
            "weight_decay",
            "seed",
        ]

        for key in key_params:
            if key in config:
                print(f"{key}: {config[key]}")
    else:
        print_config(config)

    print("=" * 80)


if __name__ == "__main__":
    main()
