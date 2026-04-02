#!/usr/bin/env python
"""Main script for BiFlexiv Rizon4 RT dual-arm robot inference with OpenPI.

Example usage:
    # Basic inference
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000

    # With RTC enabled
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --rtc_enabled

    # Side-mount configuration
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --bi_mount_type side

    # Dry run (robot connected but actions not sent)
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 --dry_run

    # Inference + simultaneous recording in LeRobot format
    python -m examples.bi_flexiv_rizon4_rt.main \\
        --host 192.168.2.100 --port 8000 \\
        --record \\
        --record_repo_id Xense/my_new_dataset \\
        --task "pack 6 cosmetic bottles into the carton"
"""

from dataclasses import dataclass
import signal
import sys

from lerobot.utils.robot_utils import get_logger
from openpi_client import action_chunk_broker
from openpi_client import rtc_action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from typing_extensions import override
import tyro

import examples.bi_flexiv_rizon4_rt.env as _env
import examples.bi_flexiv_rizon4_rt.recorder as _recorder

logger = get_logger("BiFlexivRizon4RTMain")

# Action dimension labels for dry-run logging
_ACTION_LABELS = [
    "left_tcp.x",
    "left_tcp.y",
    "left_tcp.z",
    "left_tcp.r1",
    "left_tcp.r2",
    "left_tcp.r3",
    "left_tcp.r4",
    "left_tcp.r5",
    "left_tcp.r6",
    "right_tcp.x",
    "right_tcp.y",
    "right_tcp.z",
    "right_tcp.r1",
    "right_tcp.r2",
    "right_tcp.r3",
    "right_tcp.r4",
    "right_tcp.r5",
    "right_tcp.r6",
    "left_gripper.pos",
    "right_gripper.pos",
]


class DryRunEnvironmentWrapper(_environment.Environment):
    """Intercepts policy actions and prints them without executing on robot."""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0
        self._last_rtc_inference_seq = -1

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        self._last_rtc_inference_seq = -1
        logger.info(f"\n{'='*80}")
        logger.info(f"Episode {self._episode_count} - reset (dry run)")
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
        actions = action.get("actions")
        if actions is not None:
            logger.info(f"\n{'─'*80}")
            logger.info(f"Step {self._step_count} - policy action (20D Cartesian):")
            logger.info(f"{'─'*80}")
            for i, (label, value) in enumerate(zip(_ACTION_LABELS, actions)):
                logger.info(f"  [{i:2d}] {label:18s}: {value:+.6f}")
            logger.info(f"{'─'*80}")
            rtc = action.get("rtc_metrics")
            if rtc is not None:
                seq = int(rtc.get("inference_seq", 0))
                if seq != self._last_rtc_inference_seq:
                    self._last_rtc_inference_seq = seq
                    next_d = rtc["delay_for_next_infer_steps"]
                    logger.info(
                        f"RTC [dry run] new chunk #{seq}: "
                        f"infer+server RTT={rtc['infer_round_trip_ms']:.1f} ms "
                        f"(model + WebSocket round-trip); "
                        f"delay est={rtc['estimated_delay_steps']} steps, "
                        f"real={rtc['real_delay_steps']} steps, "
                        f"time-based={rtc['inference_delay_steps']} steps; "
                        f"delay_for_next_infer={next_d}; "
                        f"merge={rtc['merge_ms']:.2f} ms; "
                        f"queue_after={rtc['queue_size_after_merge']}; "
                        f"infer RTT p95={rtc['latency_p95_ms']:.1f} ms"
                    )
            logger.info("DRY RUN: action NOT sent to robot")
            logger.info(f"{'─'*80}\n")

    def disconnect(self) -> None:
        self._wrapped_env.disconnect()


@dataclass
class Args:
    """Arguments for BiFlexiv Rizon4 RT inference."""

    # Policy server
    host: str = "localhost"
    port: int = 8000

    # Robot configuration
    bi_mount_type: str = "side"  # "forward" or "side"
    use_force: bool = False
    go_to_start: bool = True
    stiffness_ratio: float = 0.2
    inner_control_hz: int = 1000
    interpolate_cmds: bool = True
    enable_tactile_sensors: bool = False
    log_level: str = "DEBUG"

    # Image rendering
    render_height: int = 224
    render_width: int = 224

    # Runtime settings
    runtime_hz: float = 30.0
    num_episodes: int = 1
    max_episode_steps: int = 1000000

    # Dry run mode
    dry_run: bool = False

    # Non-RTC action chunking
    action_horizon: int = 50

    # RTC config
    rtc_enabled: bool = False
    action_queue_size_to_get_new_actions: int = 40
    execution_horizon: int = 50
    blend_steps: int = 3
    default_delay: int = 4

    # Recording (LeRobot format, raw 640x480 images + absolute actions)
    record: bool = False
    record_repo_id: str = "Xense/recorded_dataset"
    record_root: str | None = None  # local save path, defaults to ~/.cache/huggingface/lerobot
    task: str = "pack 6 cosmetic bottles into the carton"


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logger.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    base_environment = _env.BiFlexivRizon4RTEnvironment(
        bi_mount_type=args.bi_mount_type,
        use_force=args.use_force,
        go_to_start=args.go_to_start,
        stiffness_ratio=args.stiffness_ratio,
        inner_control_hz=args.inner_control_hz,
        interpolate_cmds=args.interpolate_cmds,
        enable_tactile_sensors=args.enable_tactile_sensors,
        log_level=args.log_level,
        render_height=args.render_height,
        render_width=args.render_width,
        setup_robot=True,
    )

    if args.dry_run:
        logger.info("DRY RUN mode: actions will be printed, not executed")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        environment = base_environment

    subscribers = []
    if args.record:
        if args.dry_run:
            logger.warning(
                "Recording is enabled in dry-run mode — state/action data will be from policy output only (no real robot motion)"
            )
        recorder = _recorder.make_recorder_subscriber(
            repo_id=args.record_repo_id,
            task=args.task,
            fps=int(args.runtime_hz),
            root=args.record_root,
        )
        subscribers.append(recorder)
        logger.info(f"Recording enabled: repo_id={args.record_repo_id}, task='{args.task}'")

    if args.rtc_enabled:
        policy = rtc_action_chunk_broker.RTCActionChunkBroker(
            policy=ws_client_policy,
            frequency_hz=args.runtime_hz,
            action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
            rtc_enabled=args.rtc_enabled,
            execution_horizon=args.execution_horizon,
            blend_steps=args.blend_steps,
            default_delay=args.default_delay,
            dry_run=args.dry_run,
        )
    else:
        policy = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
        )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=policy),
        subscribers=subscribers,
        max_hz=args.runtime_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    def safe_disconnect() -> None:
        try:
            actual_env = environment._wrapped_env if isinstance(environment, DryRunEnvironmentWrapper) else environment
            actual_env.disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

    def signal_handler(sig, frame):
        logger.info("Ctrl+C detected, disconnecting...")
        safe_disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        runtime.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt")
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        safe_disconnect()


if __name__ == "__main__":
    tyro.cli(main)
