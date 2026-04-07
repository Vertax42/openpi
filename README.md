# openpi - Xense Robotics Fork

> **Note:** This is a fork of [Physical Intelligence's openpi repository](https://github.com/Physical-Intelligence/openpi), adapted and extended for **Xense Robotics platforms** (BiARX5 and Xense Flare dual-arm robots).

## 🎯 Our Contributions

This fork focuses on adapting OpenPI models to Xense Robotics platforms with the following key contributions:

- **Xense Platform Support**: Complete integration for BiARX5 and Xense Flare dual-arm robot platforms
- **Custom Training Configurations**: Fine-tuned configs for various manipulation tasks (tie shoes, pick-and-place, open lock, wipe vase, etc.)
- **Platform-Specific Policies**: `xense_flare_policy.py` and optimized data processing pipelines for Xense robots
- **Real-World Deployment**: Production-ready inference and training commands for Xense platforms
- **Streamlined Codebase**: Removed ALOHA and LIBERO dependencies to focus on DROID and Xense platforms

---

## About OpenPI

openpi holds open-source models and packages for robotics, originally published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

This repository contains three types of models:

- **[π₀ model](https://www.physicalintelligence.company/blog/pi0)**: A flow-based vision-language-action model (VLA)
- **[π₀-FAST model](https://www.physicalintelligence.company/research/fast)**: An autoregressive VLA based on the FAST action tokenizer
- **[π₀.₅ model](https://www.physicalintelligence.company/blog/pi05)**: An upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation)

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID.

## Requirements

**⚠️ IMPORTANT: This project requires NVIDIA GPU with CUDA support. CPU-only execution is not supported.**

### Hardware Requirements

**GPU:** NVIDIA GPU with CUDA 12 support and **minimum 24GB VRAM** is required for training and inference. The models use JAX with CUDA 12 and PyTorch with CUDA 12.8.

| Mode               | Minimum VRAM | Recommended GPU    |
| ------------------ | ------------ | ------------------ |
| Inference          | 24 GB        | RTX 4090 (24GB)    |
| Fine-Tuning (LoRA) | 24 GB        | RTX 4090 (24GB)    |
| Fine-Tuning (Full) | 70+ GB       | A100 (80GB) / H100 |

**Note:** These estimations assume a single GPU. You can use multiple GPUs with model parallelism (FSDP) to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Multi-node training is not currently supported.

**Operating System:** Ubuntu 22.04 or later (other Linux distributions may work but are not officially tested). Windows and macOS are not supported.

## Installation

We use the `lerobot-xense` mamba environment as the base, then install openpi into it:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:Vertax42/openpi.git
# Or if you already cloned:
git submodule update --init --recursive

# Activate the base environment and install openpi
mamba activate lerobot-xense
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Model Checkpoints

### Base Models

We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                          | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning                       | `gs://openpi-assets/checkpoints/pi05_base`     |

### Fine-Tuned Models

We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with the DROID Franka setup, they might not generalize to your particular setup, though we found the DROID checkpoint to generalize quite broadly in practice.

| Model              | Use Case                | Description                                                                                                                                                                                                                           | Checkpoint Path                                 |
| ------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| $\pi_0$-FAST-DROID | Inference               | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform                              | `gs://openpi-assets/checkpoints/pi0_fast_droid` |
| $\pi_0$-DROID      | Fine-Tuning             | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                                             | `gs://openpi-assets/checkpoints/pi0_droid`      |
| $\pi_{0.5}$-DROID  | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`     |

By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.

## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.

## Fine-Tuning Base Models on Your Own Data

We will explain how to fine-tune a base model on your own data using examples from the DROID dataset and Xense platform. We will explain three steps:

1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

For training, we use the LeRobot dataset format. You can convert your own data to this format by following the LeRobot documentation. We provide example conversion scripts for the DROID dataset in the [`examples/droid`](examples/droid) directory.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for DROID and Xense platforms in [`config.py`](src/openpi/training/config.py), which you can modify for your own dataset:

- Data transforms: Define the data mapping from your environment to the model (see [`droid_policy.py`](src/openpi/policies/droid_policy.py) or [`xense_flare_policy.py`](src/openpi/policies/xense_flare_policy.py) for examples)
- `DataConfig`: Defines how to process raw data from LeRobot dataset for training
- `TrainConfig`: Defines fine-tuning hyperparameters, data config, and weight loader

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config (e.g., for Xense):

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_base_arx5_lora
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_base_arx5_lora --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for _reloading_ normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from your robot runtime. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_base_arx5_lora --policy.dir=checkpoints/pi05_base_arx5_lora/my_experiment/19999
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For more detailed examples of running inference on specific platforms, see:

- [DROID README](examples/droid/README.md) for DROID platform
- [BiARX5 README](examples/bi_arx5_real/README.md) for Xense platform

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).

### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the DROID benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup

1. Make sure transformers 4.53.2 is installed: `pip show transformers`

2. Apply the transformers library patches (adjust the site-packages path for your Python version):
   ```bash
   # Find your site-packages path
   python -c "import transformers; print(transformers.__file__)"
   # Copy patches
   cp -r ./src/openpi/models_pytorch/transformers_replace/* $(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:

   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example using debug config:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**

1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**

1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo.

| Issue                                     | Resolution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency conflicts                      | Make sure you are in the `lerobot-xense` mamba environment, then run `GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` to install openpi.                                                                                                                                                                                                                                                                                                                                                                                 |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may way to consider disabling EMA. |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                                                                                                                                                                                                                                                                                                                                                   |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                                                                                                                                                                                                                                                                                                                                        |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try _uninstalling_ system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts.                                                                                                                                    |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                                                                                                                                                                                                                                                                                                                                                                                 |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                                                                                                                                                                                                                                                                                                                                         |
| Diverging training loss                   | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround.                                                                                                                                                                                                                   |

---

## 🤖 Xense Platform Training & Deployment

This section contains production-ready commands for training and deploying models on Xense Robotics platforms.

### Platform Overview

- **BiARX5**: Bi-manual ARX-5 robot setup with parallel grippers
- **Xense Flare**: UMI-style dual-arm robot with data collection grippers

### Environment Setup for Training

````bash
# Start a tmux session
tmux new -s training

# Configure offline mode for local datasets
export HF_HUB_OFFLINE=1 && export HF_DATASETS_OFFLINE=1 && echo "Offline mode enabled"
export HF_DATASETS_CACHE=/home/ubuntu/.cache/huggingface/datasets

# Configure NCCL for multi-GPU training
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_TOPO_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=1 scripts/train_pytorch.py pi05_base_arx5_full --exp-name=xense_bi_arx5_pick_and_place_cube_full --resume ; shutdown -h +5

torchrun --nproc_per_node=4 scripts/train_pytorch.py pi05_base_arx5_tie_shoes_full --exp-name=tie_shoes_full_100_episodes_torch --overwrite ; shutdown -h +5

python scripts/compute_norm_stats.py --config-name pi05_base_arx5_full
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_pytorch.py pi05_base_arx5_full --exp-name=xense_bi_arx5_pick_and_place_cube --overwrite / --resume

# 20251021_XenseRobotics_TieShoes
python scripts/compute_norm_stats.py --config-name pi05_base_arx5_tie_shoes_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_tie_shoes_lora --exp-name=tie_shoes_lora_100_episodes --overwrite / --resume

# 20251027_TieShoes_HighQuality_Lora
python scripts/compute_norm_stats.py --config-name pi05_base_arx5_tie_shoes_high_quality_lora_1027
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_tie_shoes_high_quality_lora_1027 --exp-name=tie_shoes_lora_50_episodes --overwrite / --resume

# 20251028_TieShoes_HighQuality_White_Lora
python scripts/compute_norm_stats.py --config-name pi05_base_arx5_tie_shoes_high_quality_white_lora_1028
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_tie_shoes_high_quality_white_lora_1028 --exp-name=tie_shoes_white_lora_50_episodes --overwrite / --resume

# 20251101_TIeShoes_25episodes_lora_no_adjust
python scripts/compute_norm_stats.py --config-name tie_shoes_white_lora_finetune_1030_25_episodes
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py tie_shoes_white_lora_finetune_1030_25_episodes --exp-name=tie_shoes_25_episodes_lora_no_adjust_1101 --overwrite / --resume

# 20251021_AutoDL_TieShoes
jax[cuda13]
orbax-checkpoint==0.11.20

# 20251103 TieShoes_50episodes_lora_no_adjust
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py tie_shoes_50_episodes_lora_no_adjust_1101 --exp-name tie_shoes_50_episodes_lora_no_adjust_1103_40000 --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py tie_shoes_50_episodes_lora_no_adjust_1101 --exp-name tie_shoes_50_episodes_lora_no_adjust_1103_40000 --resume

python scripts/compute_norm_stats.py --config-name pi05_base_arx5_tie_shoes_full
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_tie_shoes_full --exp-name=tie_shoes_full_100_episodes_gpu_test --overwrite / --resume

# test 20251111 lerobot040_test_bi_arx5
python scripts/compute_norm_stats.py --config-name pi05_base_full_test
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_full_test --exp-name=pi05_base_full_test --overwrite / --resume

# 20251204 pick and place chips train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_lora_pick_and_place_chips --exp-name=pi05_base_arx5_lora_pick_and_place_chips_20251204 --overwrite

# 20251209 training time rtc
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_lora_training_time_rtc --exp-name=training_time_rtc_20251209 --overwrite

# 20260108 xense flare open lock
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_open_lock
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_open_lock --exp-name=xense_flare_open_lock_20260108 --overwrite / --resume

# 20260113 xense flare wipe vase
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_wipe_vase
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_wipe_vase --exp-name=xense_flare_wipe_vase_20260113 --overwrite / --resume

# 20260115 xense flare pick and place cube
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_pick_and_place_cube
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_pick_and_place_cube --exp-name=xense_flare_pick_and_place_cube_20260115 --overwrite / --resume

# 20260202 tie shoes 50 episodes lora no adjust training time rtc
python scripts/compute_norm_stats.py --config-name tie_shoes_50_episodes_lora_no_adjust_training_time_rtc_0202
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py tie_shoes_50_episodes_lora_no_adjust_training_time_rtc_0202 --exp-name=tie_shoes_50_episodes_lora_no_adjust_training_time_rtc_0202 --overwrite / --resume

# 20260228 xense flare open lock training time rtc
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_open_lock_rtc_0228
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_open_lock_rtc_0228 --exp-name=xense_flare_open_lock_rtc_0228 --overwrite / --resume

# watch gpu temperature, power, fan speed, clocks, utilization
watch -n 1 'sensors | grep -E "Package|Core (0|4|8|12|16)" && echo "---" && nvidia-smi --query-gpu=temperature.gpu,power.draw,fan.speed,clocks.gr,clocks.mem,utilization.gpu --format=csv'

## inference time commands
copy checkpoints from autodl server to local server
```bash
scp -P 15443 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/openpi/checkpoints/pi05_base_arx5_tie_shoes_full/tie_shoes_full_100_episodes_torch/20000 .
````

```bash
# pick and place
python scripts/serve_policy.py --default-prompt="pick rgb cubes and place them into the blue box" policy:checkpoint --policy.config=pi05_base_arx5_lora --policy.dir=checkpoints/pi05_base_arx5_lora/xense_bi_arx5_pick_and_place_cube_arx5_assets/33000

# tie shoes
python scripts/serve_policy.py --default-prompt="tie shoelaces" policy:checkpoint --policy.config=pi05_base_arx5_tie_shoes_lora --policy.dir=checkpoints/pi05_base_arx5_tie_shoes_lora/tie_shoes_lora_50_episodes/33000

python scripts/serve_policy.py policy:checkpoint --policy.config=tie_shoes_50_episodes_lora_no_adjust_1101 --policy.dir=checkpoints/tie_shoes_50_episodes_lora_no_adjust_1101/tie_shoes_50_episodes_lora_no_adjust_1103_40000/16000

# pick and place chips
python scripts/serve_policy.py --default-prompt="pick up a potato chip and place it into the chips container" policy:checkpoint --policy.config=pi05_base_arx5_lora_pick_and_place_chips --policy.dir=checkpoints/pi05_base_arx5_lora_pick_and_place_chips/pi05_base_arx5_lora_pick_and_place_chips_20251204/19999

# training time RTC
python scripts/serve_policy.py --default-prompt="pick rgb cubes and place them into the blue box" policy:checkpoint --policy.config=pi05_base_arx5_lora_training_time_rtc --policy.dir=checkpoints/pi05_base_arx5_lora_training_time_rtc/training_time_rtc_20251209/39999

# open lock 20260108
python scripts/serve_policy.py --default-prompt="open the lock with the key" policy:checkpoint --policy.config=pi05_base_xense_flare_open_lock --policy.dir=checkpoints/pi05_base_xense_flare_open_lock/xense_flare_open_lock_20260108/19999

# wipe vase 20260115
python scripts/serve_policy.py --default-prompt="wipe the vase" policy:checkpoint --policy.config=pi05_base_xense_flare_wipe_vase --policy.dir=checkpoints/pi05_base_xense_flare_wipe_vase/xense_flare_wipe_vase_20260113/19999

# pick and place cube 20260115
python scripts/serve_policy.py --default-prompt="pick up cubes in rgb order from the table and place them in the blue box" policy:checkpoint --policy.config=pi05_base_xense_flare_pick_and_place_cube --policy.dir=checkpoints/pi05_base_xense_flare_pick_and_place_cube/xense_flare_pick_and_place_cube_20260115/39999

192.168.1.165:8000
vertax@Jarvis:~$ nc -zv 192.168.2.215 8000
Connection to 192.168.2.215 8000 port [tcp/*] succeeded!
python -m examples.bi_arx5_real.main     --args.host 192.168.2.215     --args.port 8000     --args.dry_run  --args.enable_tactile_sensors

```

### Training Commands

#### BiARX5 Platform

```bash
# Pick and place (full fine-tuning)
python scripts/compute_norm_stats.py --config-name pi05_base_arx5_full
torchrun --nproc_per_node=1 scripts/train_pytorch.py pi05_base_arx5_full \
    --exp-name=xense_bi_arx5_pick_and_place_cube_full --resume

# Tie shoes (LoRA fine-tuning)
python scripts/compute_norm_stats.py --config-name pi05_base_arx5_tie_shoes_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_tie_shoes_lora \
    --exp-name=tie_shoes_lora_100_episodes --overwrite

# Pick and place chips
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_arx5_lora_pick_and_place_chips \
    --exp-name=pi05_base_arx5_lora_pick_and_place_chips_20251204 --overwrite
```

#### Xense Flare Platform (UMI-style grippers)

```bash
# Open lock task
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_open_lock
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_open_lock \
    --exp-name=xense_flare_open_lock_20260108 --overwrite

# Wipe vase task
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_wipe_vase
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_wipe_vase \
    --exp-name=xense_flare_wipe_vase_20260113 --overwrite

# Pick and place cube task
python scripts/compute_norm_stats.py --config-name pi05_base_xense_flare_pick_and_place_cube
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_xense_flare_pick_and_place_cube \
    --exp-name=xense_flare_pick_and_place_cube_20260115 --overwrite
```

### BiFlexiv Platform

```bash
# Pack 6 cosmetic bottles into the carton
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora \
    --exp-name=bi_flexiv_pack_6_cosmetic_bottles_lora_20260329 --overwrite

# Assemble box with phone stand test
python scripts/compute_norm_stats.py --config-name pi05_base_bi_flexiv_assemble_box_with_phone_stand_test_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_base_bi_flexiv_assemble_box_with_phone_stand_test_lora \
    --exp-name=bi_flexiv_assemble_box_with_phone_stand_test_lora_20260403 --overwrite
```

### Deployment Commands

#### BiARX5 Platform Inference

```bash
# Pick and place RGB cubes
python scripts/serve_policy.py \
    --default-prompt="pick rgb cubes and place them into the blue box" \
    policy:checkpoint \
    --policy.config=pi05_base_arx5_lora \
    --policy.dir=checkpoints/pi05_base_arx5_lora/xense_bi_arx5_pick_and_place_cube_arx5_assets/33000

# Tie shoelaces
python scripts/serve_policy.py \
    --default-prompt="tie shoelaces" \
    policy:checkpoint \
    --policy.config=pi05_base_arx5_tie_shoes_lora \
    --policy.dir=checkpoints/pi05_base_arx5_tie_shoes_lora/tie_shoes_lora_50_episodes/33000

# Pick and place chips
python scripts/serve_policy.py \
    --default-prompt="pick up a potato chip and place it into the chips container" \
    policy:checkpoint \
    --policy.config=pi05_base_arx5_lora_pick_and_place_chips \
    --policy.dir=checkpoints/pi05_base_arx5_lora_pick_and_place_chips/pi05_base_arx5_lora_pick_and_place_chips_20251204/19999
```

#### BiFlexiv Platform Inference

```bash
# Pack 6 cosmetic bottles into the carton
python scripts/serve_policy.py \
    --default-prompt="pack 6 cosmetic bottles into the carton" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora/bi_flexiv_pack_6_cosmetic_bottles_lora_20260329/19999

# Assemble box with phone stand test
python scripts/serve_policy.py \
    --default-prompt="assemble the box with the phone stand" \
    policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_assemble_box_with_phone_stand_test_lora \
    --policy.dir=checkpoints/pi05_base_bi_flexiv_assemble_box_with_phone_stand_test_lora/bi_flexiv_assemble_box_with_phone_stand_test_lora_20260329/19999
```

### Utilities

#### Checkpoint Transfer

```bash
# Copy checkpoints from remote server to local
scp -P 15443 -r root@connect.westd.seetacloud.com:/root/autodl-tmp/openpi/checkpoints/pi05_base_arx5_tie_shoes_full/tie_shoes_full_100_episodes_torch/20000 .
```

#### Network Testing

```bash
# Test policy server connectivity
nc -zv 192.168.2.215 8000
nc -zv 192.168.142.158 8000
Connection to 192.168.142.158 8000 port [tcp/*] succeeded!

# Run BiARX5 client with tactile sensors
python -m examples.bi_arx5_real.main \
    --args.host 192.168.2.215 \
    --args.port 8000 \
    --args.dry_run \
    --args.enable_tactile_sensors
```

```bash
# Run BiFlexiv client
python -m examples.bi_flexiv_rizon4_rt.main \
    --args.host 192.168.142.158 \
    --args.port 8000 \
    --args.bi-mount-type side \
    --args.inner-control-hz 1000 \
    --args.interpolate-cmds \
    --args.runtime-hz 30 \
    --args.log-level INFO \
    --args.rtc-enabled \
    --args.dry-run
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository is based on the original [OpenPI](https://github.com/Physical-Intelligence/openpi) project by Physical Intelligence. We thank the Physical Intelligence team for open-sourcing their excellent work on vision-language-action models.
