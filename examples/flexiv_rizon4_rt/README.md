# Flexiv Rizon4 RT Robot Inference

This example runs inference on a real Flexiv Rizon4 robot using the **RT driver** (`flexiv_rt`), which spawns a C++ RT thread at **1 kHz** (SCHED_FIFO) for deterministic Cartesian streaming control.

## Key Differences vs `flexiv_rizon4_real`

| | `flexiv_rizon4_real` (NRT) | `flexiv_rizon4_rt` (RT) |
|---|---|---|
| Backend | `flexivrdk` (NRT) | `flexiv_rt` (RT, 1 kHz C++ thread) |
| Control mode | joint impedance **or** Cartesian | **RT Cartesian only** |
| Action space | 8D (joint) or 10D (Cartesian) | **10D Cartesian always** |
| Reset | Blocking NRT MoveJ | Non-blocking RT trajectory |
| Latency | ~10 ms per command | < 1 ms (shared memory) |

## Hardware Requirements

- Flexiv Rizon4 7-DOF collaborative robot
- Flare Gripper with wrist camera (optional)
- Network connection to the robot

## Setup

```bash
# Install lerobot-xense with RT support
pip install -e /path/to/lerobot-xense

# Install flexiv_rt (libpyflexiv)
# Follow instructions at your internal repo
```

## Start Policy Server

```bash
uv run scripts/serve_policy.py \
    policy:checkpoint \
    --policy.config=pi05_base_xense_flare_pick_and_place_cube \
    --policy.dir=checkpoints/your_checkpoint
```

## Usage

### Basic Inference (non-RTC)

```bash
python -m examples.flexiv_rizon4_rt.main \
    --host 192.168.2.215 \
    --port 8000
```

### With RTC Enabled

```bash
python -m examples.flexiv_rizon4_rt.main \
    --host 192.168.2.215 \
    --port 8000 \
    --rtc_enabled \
    --execution_horizon 20 \
    --runtime_hz 25
```

### Dry Run (print actions without executing)

```bash
python -m examples.flexiv_rizon4_rt.main \
    --host 192.168.2.215 \
    --port 8000 \
    --dry_run
```

## Configuration Options

### Robot

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | localhost | Policy server IP |
| `port` | 8000 | Policy server port |
| `robot_sn` | Rizon4-063423 | Robot serial number |
| `use_gripper` | True | Enable Flare gripper |
| `use_force` | False | Enable force control axes |
| `go_to_start` | True | Move to start position on connect |
| `runtime_hz` | 25.0 | Control loop frequency (Hz) |
| `dry_run` | False | Print actions without executing |

### RT-specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stiffness_ratio` | 0.2 | Cartesian stiffness multiplier (×K_x_nom) |
| `inner_control_hz` | 1000 | How often the 1 kHz RT loop consumes a new Python command |
| `interpolate_cmds` | True | Enable linear interpolation between consumed commands |
| `start_position_degree` | `[-1.70, 4.48, ...]` | Start joint angles (deg) |
| `zero_ft_sensor_on_connect` | True | Zero FT sensor on startup |

### Gripper

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gripper_type` | flare_gripper | `flare_gripper` or `xense_gripper` |
| `gripper_mac_addr` | e2b26adbb104 | Gripper MAC address |
| `gripper_cam_size` | (640, 480) | Wrist camera resolution |
| `gripper_rectify_size` | (400, 700) | Tactile sensor rectified size |
| `gripper_max_pos` | 85.0 | Maximum gripper position (mm) |

### RTC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rtc_enabled` | False | Enable RTC mode |
| `execution_horizon` | 30 | Execution window size (< action_horizon) |
| `action_queue_size_to_get_new_actions` | 20 | Queue threshold for new inference |
| `blend_steps` | 5 | Steps for blending old/new actions |
| `default_delay` | 2 | Default inference delay (steps) |

## Control Architecture

```
Python main.py (25 Hz)
  └─ ActionChunkBroker / RTCActionChunkBroker
      └─ WebsocketClientPolicy  →  Policy Server (GPU)
          └─ FlexivRizon4RTEnvironment
              └─ FlexivRizon4RT.send_action()
                  └─ cc.set_target_pose()  [shared memory write]
                      └─ C++ RT thread (1 kHz, SCHED_FIFO)
                          └─ StreamCartesianMotionForce  →  Robot
```

## Safety Notes

1. Always ensure the robot workspace is clear before running
2. The robot will move to start position on connect (unless `--go_to_start False`)
3. Use `--dry_run` to verify action values without robot movement
4. Press Ctrl+C for graceful shutdown — RT thread stops, robot returns to home
5. `stiffness_ratio=0.2` (20% nominal) provides compliant behaviour; increase for stiffer tracking
