# BiFlexiv Rizon4 RT — Real Robot Inference

Dual-arm Flexiv Rizon4 inference client using the OpenPI policy server.

## Prerequisites

- `lerobot-xense` conda environment with `lerobot2mcap` installed
- Flexiv Rizon4 RT arms reachable over Ethernet
- OpenPI policy server running (see below)

## Network Setup

Robot FastDDS communication and policy server inference must run on **separate physical links** to avoid network contention (FastDDS 1 kHz commands are latency-sensitive and will stall if competing with large inference payloads).

| Interface | IP | Connects to |
|---|---|---|
| enp8s0 (motherboard Ethernet) | 192.168.142.216 (DHCP) | Router → Robot arms (FastDDS) |
| enx6c1ff7618da5 (USB-C adapter) | **10.142.1.2**/24 (static) | Direct cable → Policy server |

**Policy server** (GPU machine) sets its corresponding port to **10.142.1.1**/24.

### Configure static IPs (one-time per boot)

On this machine (robot client):
```bash
sudo ip addr add 10.142.1.2/24 dev enx6c1ff7618da5
```

On the policy server:
```bash
sudo ip addr add 10.142.1.1/24 dev <interface_name>
```

Verify connectivity:
```bash
ping 10.142.1.1
```

Then use `--host 10.142.1.1` when launching the robot client.

## Usage

### Terminal 1: Start OpenPI Policy Server

```bash
cd ~/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_base_bi_flexiv_pack_6_cosmetic_bottles_lora \
    --policy.dir=<checkpoint_dir>
```

### Terminal 2: Run Robot Client

```bash
cd ~/openpi
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --host 10.142.1.1 \
    --port 8000
```

#### Common options

| Flag | Default | Description |
|---|---|---|
| `--host` | `localhost` | Policy server host |
| `--port` | `8000` | Policy server port |
| `--bi_mount_type` | `forward` | Robot mount: `forward` or `side` |
| `--stiffness_ratio` | `0.2` | Cartesian stiffness (0–1) |
| `--inner_control_hz` | `1000` | How often each 1 kHz RT loop consumes a new Python command |
| `--interpolate_cmds` | `True` | Enable linear interpolation between consumed commands |
| `--runtime_hz` | `20.0` | Policy inference Hz |
| `--action_horizon` | `50` | Steps per action chunk |
| `--num_episodes` | `1` | Episodes to run |
| `--max_episode_steps` | `100000` | Max steps per episode |
| `--dry_run` | `False` | Print actions, do not execute |

#### RTC (real-time correction) mode

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --host 10.142.1.1 --port 8000 \
    --rtc_enabled \
    --action_queue_size_to_get_new_actions 40 \
    --execution_horizon 50 \
    --blend_steps 3
```

#### Dry run (no robot motion)

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --host 10.142.1.1 --port 8000 \
    --dry_run
```

---

## Synchronized Recording

Record a new LeRobot-format dataset while running inference (raw 640×480 images,
absolute actions — same format as the training data).

```bash
mamba run -n lerobot-xense python -m examples.bi_flexiv_rizon4_rt.main \
    --host 10.142.1.1 --port 8000 \
    --record \
    --record_repo_id Xense/my_new_dataset \
    --task "pack 6 cosmetic bottles into the carton"
```

The dataset is saved locally to `~/.cache/huggingface/lerobot/<repo_id>` by default.
Use `--record_root /path/to/dir` to override the save location.

---

## Converting Recorded Dataset to MCAP

[MCAP](https://mcap.dev/) files can be opened in [Foxglove Studio](https://foxglove.dev/)
for visual inspection of observations, actions, and camera streams.

> Run in the `lerobot-xense` environment where `lerobot2mcap` is installed.

### Convert all episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset
```

### Convert specific episodes

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --episodes 0 1 2
```

### Parallel conversion (faster for large datasets)

```bash
mamba run -n lerobot-xense lerobot2mcap convert \
    ~/.cache/huggingface/lerobot/Xense/my_new_dataset \
    -o ~/mcap_output/my_new_dataset \
    --jobs 4
```

Each episode produces a separate `.mcap` file under the output directory.
Open any `.mcap` file directly in Foxglove Studio to inspect it.

---

## Action / State Space

20-dimensional Cartesian space:

| Index | Key |
|---|---|
| 0–2 | `left_tcp.x/y/z` |
| 3–8 | `left_tcp.r1–r6` (rotation) |
| 9–11 | `right_tcp.x/y/z` |
| 12–17 | `right_tcp.r1–r6` (rotation) |
| 18 | `left_gripper.pos` |
| 19 | `right_gripper.pos` |

TCP positions are **delta** actions; gripper positions are **absolute**.

---

## File Structure

```
examples/bi_flexiv_rizon4_rt/
├── main.py       # Entry point and CLI args
├── env.py        # OpenPI Environment adapter (image resize, obs format)
├── real_env.py   # BiFlexivRizon4RT robot control wrapper
└── recorder.py   # LeRobot-format episode recorder subscriber
```
