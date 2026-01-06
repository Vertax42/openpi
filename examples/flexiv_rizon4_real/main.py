#!/usr/bin/env python
"""Main script for Flexiv Rizon4 real robot inference with OpenPI.

This script connects to a Flexiv Rizon4 robot and runs inference using a remote policy server.

Example usage:
    # Basic inference (non-RTC mode)
    python -m examples.flexiv_rizon4_real.main \\
        --args.host 192.168.2.215 \\
        --args.port 8000

    # With RTC enabled
    python -m examples.flexiv_rizon4_real.main \\
        --args.host 192.168.2.215 \\
        --args.port 8000 \\
        --args.rtc_enabled

    # Dry run (robot connected but actions not executed)
    python -m examples.flexiv_rizon4_real.main \\
        --args.host 192.168.2.215 \\
        --args.port 8000 \\
        --args.dry_run
"""

import signal
import sys
from dataclasses import dataclass
from typing import Optional  # noqa: F401

import tyro
from typing_extensions import override

import examples.flexiv_rizon4_real.env as _env
from lerobot.utils.robot_utils import get_logger
from openpi_client import rtc_action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import policy_agent as _policy_agent
from openpi_client.runtime import runtime as _runtime

logger = get_logger("FlexivRizon4Main")


class DryRunEnvironmentWrapper(_environment.Environment):
    """Wrapper: intercept and print policy action, but not actually execute."""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logger.info(f"\n{'='*80}")
        logger.info(
            f"🔄 Episode {self._episode_count} - environment reset (dry run mode)"
        )
        logger.info(f"{'='*80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        return self._wrapped_env.get_observation()

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1

        # Print policy output action
        actions = action.get("actions")
        if actions is not None:
            logger.info(f"\n{'─'*80}")
            logger.info(f"🎯 Step {self._step_count} - policy output action:")
            logger.info(f"{'─'*80}")

            # Print action info based on control mode
            logger.info(f"Action shape: {actions.shape}")
            logger.info(f"Action values: {actions}")

            logger.info(f"{'─'*80}")
            logger.info("⚠️  DRY RUN mode: action intercepted, NOT executed on robot")
            logger.info(f"{'─'*80}\n")

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for Flexiv Rizon4 inference."""

    # Policy server connection
    host: str = "localhost"
    port: int = 8000

    # Robot configuration
    robot_sn: str = "Rizon4-063423"
    control_mode: str = (
        "cartesian_motion_force"  # "joint_impedance" or "cartesian_motion_force"
    )
    use_gripper: bool = True
    use_force: bool = False
    go_to_start: bool = True
    log_level: str = "INFO"

    # Flare gripper settings
    flare_gripper_mac_addr: str = "e2b26adbb104"
    flare_gripper_cam_size: tuple[int, int] = (640, 480)
    flare_gripper_rectify_size: tuple[int, int] = (200, 350)
    flare_gripper_max_pos: float = 85.0

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = 25.0  # Control frequency
    num_episodes: int = 1
    max_episode_steps: int = 10000

    # Dry run mode (robot connected but actions not executed)
    dry_run: bool = False

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 20
    execution_horizon: int = 30
    blend_steps: int = 5
    default_delay: int = 2


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    metadata = ws_client_policy.get_server_metadata()
    logger.info(f"Server metadata: {metadata}")

    # Create base environment
    base_environment = _env.FlexivRizon4RealEnvironment(
        robot_sn=args.robot_sn,
        control_mode=args.control_mode,
        use_gripper=args.use_gripper,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        log_level=args.log_level,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
        flare_gripper_mac_addr=args.flare_gripper_mac_addr,
        flare_gripper_cam_size=args.flare_gripper_cam_size,
        flare_gripper_rectify_size=args.flare_gripper_rectify_size,
        flare_gripper_max_pos=args.flare_gripper_max_pos,
    )

    # If dry run mode, wrap the environment with DryRunEnvironmentWrapper
    if args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("🔍 DRY RUN mode enabled")
        logger.info("   - Policy action output will be printed")
        logger.info("   - Action will NOT be sent to robot")
        logger.info("   - Robot will stay in initial position")
        logger.info("=" * 80 + "\n")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        logger.info("✅ Normal mode: actions will be executed on robot")
        environment = base_environment

    if args.rtc_enabled:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=rtc_action_chunk_broker.RTCActionChunkBroker(
                    policy=ws_client_policy,
                    frequency_hz=args.runtime_hz,
                    action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
                    rtc_enabled=args.rtc_enabled,
                    execution_horizon=args.execution_horizon,
                    blend_steps=args.blend_steps,
                    default_delay=args.default_delay,
                )
            ),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(policy=ws_client_policy),
            subscribers=[],
            max_hz=args.runtime_hz,
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect():
        """Safe disconnect robot."""
        try:
            # Check if the environment is a wrapper
            actual_env = environment
            if isinstance(environment, DryRunEnvironmentWrapper):
                actual_env = environment._wrapped_env

            if hasattr(actual_env, "_env") and hasattr(actual_env._env, "robot"):
                if actual_env._env.robot.is_connected:
                    logger.info("Safe disconnect robot...")
                    actual_env._env.disconnect()
                    logger.info("✓ Robot disconnected")
                else:
                    logger.info("Robot not connected, no need to disconnect")
        except Exception as disconnect_error:
            logger.warn(f"Error disconnecting robot: {disconnect_error}")

    # Setup graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\n⚠️ Detected user interrupt (Ctrl+C)")
        logger.info("Program exited safely")
        safe_disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("\n⚠️ Detected user interrupt (Ctrl+C)")
    except Exception as e:
        logger.error(f"\n❌ Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Always safe disconnect robot
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
