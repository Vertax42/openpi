# Flexiv Rizon4 Real Robot Inference

This example demonstrates how to run inference on a real Flexiv Rizon4 robot using OpenPI.

## Hardware Requirements

- Flexiv Rizon4 7-DOF collaborative robot
- Flare Gripper with wrist camera and tactile sensors (optional)
- Network connection to the robot

## Setup

1. Ensure the Flexiv robot is powered on and connected to the network
2. Install the required dependencies:

```bash
# Install lerobot with Flexiv support
pip install -e /path/to/lerobot-xense

# Install flexivrdk (Flexiv Robot Development Kit)
# Follow instructions at https://rdk.flexiv.com/
```

3. Start the policy server on the inference machine:

```bash
python scripts/serve_policy.py \
    --default-prompt="your task description" \
    policy:checkpoint \
    --policy.config=your_config \
    --policy.dir=checkpoints/your_checkpoint
```

## Usage

### Basic Inference (non-RTC mode)

```bash
python -m examples.flexiv_rizon4_real.main \
    --args.host <server_ip> \
    --args.port 8000 \
    --args.robot_sn Rizon4-063423
```

### With RTC (Real-Time Chunking) Enabled

```bash
python -m examples.flexiv_rizon4_real.main \
    --args.host <server_ip> \
    --args.port 8000 \
    --args.robot_sn Rizon4-063423 \
    --args.rtc_enabled
```

### Dry Run (No Robot Connection)

Test the setup without connecting to the robot:

```bash
python -m examples.flexiv_rizon4_real.main \
    --args.host <server_ip> \
    --args.port 8000 \
    --args.dry_run
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | localhost | Policy server IP address |
| `port` | 8000 | Policy server port |
| `robot_sn` | Rizon4-063423 | Robot serial number |
| `control_mode` | joint_impedance_control | Control mode: `joint_impedance_control` or `cartesian_motion_force_control` |
| `use_gripper` | True | Enable Flare gripper |
| `use_force` | False | Enable force control (only for cartesian mode) |
| `go_to_start` | True | Move to start position on connect |
| `runtime_hz` | 25.0 | Control frequency in Hz |
| `rtc_enabled` | False | Enable RTC mode |
| `dry_run` | False | Run without robot connection |

### Flare Gripper Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `flare_gripper_mac_addr` | e2b26adbb104 | Gripper MAC address |
| `flare_gripper_cam_size` | (640, 480) | Wrist camera resolution |
| `flare_gripper_rectify_size` | (400, 700) | Tactile sensor rectified size |
| `flare_gripper_max_pos` | 85.0 | Maximum gripper position (mm) |

### RTC Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `action_queue_size_to_get_new_actions` | 20 | Queue threshold for new inference |
| `execution_horizon` | 30 | Action horizon |
| `blend_steps` | 5 | Steps for blending actions |
| `default_delay` | 2 | Default inference delay |

## Control Modes

### Joint Impedance Control (`joint_impedance_control`)

- **Action space**: 8D (7 joint positions + 1 gripper)
- **Observation space**: 22D (7 positions + 7 velocities + 7 efforts + 1 gripper)
- Uses impedance control with configurable stiffness

### Cartesian Motion Force Control (`cartesian_motion_force_control`)

- **Action space**: 8D (7 TCP pose + 1 gripper) or 14D with force
- **Observation space**: 8D (7 TCP pose + 1 gripper) or 14D with force
- Supports pure motion control or motion + force control

## Safety Notes

1. Always ensure the robot workspace is clear before running
2. The robot will move to start position on connect (unless `go_to_start=False`)
3. Use `dry_run` mode to test configuration without robot movement
4. Press Ctrl+C for graceful shutdown - robot will return to home position

