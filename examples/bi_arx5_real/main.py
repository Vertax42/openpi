import dataclasses
import logging
import sys
from typing import Any

import numpy as np
import tyro
from openpi_client import action_chunk_broker
from openpi_client import rtc_action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime import environment as _environment
from typing_extensions import override

import examples.bi_arx5_real.env as _env


class DryRunEnvironmentWrapper(_environment.Environment):
    """Wrapper: intercept and print policy action, but not actually execute"""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logging.info(f"\n{'='*80}")
        logging.info(
            f"🔄 Episode {self._episode_count} - environment reset (dry run mode)"
        )
        logging.info(f"{'='*80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        obs = self._wrapped_env.get_observation()

        # prompt is automatically injected by the InjectDefaultPrompt transform of the policy server
        # no need to add it in the client

        # print observation summary (simplified version)
        # if self._step_count % 10 == 0:  # print observation summary every 10 steps
        #     state = obs.get("state")
        #     images = obs.get("images", {})
        #     logging.info(f"📊 step {self._step_count} - observation summary:")
        #     if state is not None:
        #         logging.info(
        #             f"   state dimension: {state.shape}, range: [{state.min():.3f}, {state.max():.3f}]"
        #         )
        #     logging.info(f"   image count: {len(images)}")

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1

        # print policy output action
        actions = action.get("actions")
        if actions is not None:
            logging.info(f"\n{'─'*80}")
            logging.info(f"🎯 step {self._step_count} - policy output action:")
            logging.info(f"{'─'*80}")

            # print detailed action information
            # logging.info(f"action dimension: {actions.shape}")
            # logging.info(f"action type: {actions.dtype}")
            # logging.info(f"action range: [{actions.min():.6f}, {actions.max():.6f}]")

            # print each joint action value
            # joint_names = [
            #     "left_joint_1",
            #     "left_joint_2",
            #     "left_joint_3",
            #     "left_joint_4",
            #     "left_joint_5",
            #     "left_joint_6",
            #     "left_gripper",
            #     "right_joint_1",
            #     "right_joint_2",
            #     "right_joint_3",
            #     "right_joint_4",
            #     "right_joint_5",
            #     "right_joint_6",
            #     "right_gripper",
            # ]

            # logging.info("\ndetailed action values:")
            # for i, (name, value) in enumerate(zip(joint_names, actions)):
            #     logging.info(f"  [{i:2d}] {name:12s}: {value:+.6f} rad")

            # logging.info("\ngripper action:")
            # logging.info(f"  left gripper (index 6):  {actions[6]:.6f}")
            # logging.info(f"  right gripper (index 13): {actions[13]:.6f}")

            # logging.info(f"{'─'*80}")
            # logging.info(
            #     "⚠️  dry run mode: action intercepted, not actually executed to robot"
            # )
            # logging.info(f"{'─'*80}\n")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    num_episodes: int = 1
    max_episode_steps: int = 10000

    # bi_arx5 specific configs
    left_arm_port: str = "can1"
    right_arm_port: str = "can3"
    log_level: str = "INFO"
    use_multithreading: bool = True

    action_horizon: int = 50  # action_horizon

    # lower controller config
    controller_dt: float = (
        0.002  # lower controller frequency, unit: second (0.002s = 2ms = 500Hz)
    )
    preview_time: float = 0.05  # preview time, unit: second (0.02s = 20ms)
    runtime_hz: int = 30  # runtime frequency, unit: Hz
    # dry run mode: only print policy output, not actually execute action
    dry_run: bool = False

    # RTC config
    rtc_enabled: bool = False
    # Threshold to request new actions, when action queue size is less than this value, new actions will be requested
    action_queue_size_to_get_new_actions: int = 10
    # Sample action with rtc horizon
    execution_horizon: int = 30


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    # create base environment
    base_environment = _env.BiARX5RealEnvironment(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        log_level=args.log_level,
        use_multithreading=args.use_multithreading,
        reset_position=metadata.get("reset_pose"),
        controller_dt=args.controller_dt,  # pass in lower controller frequency
        preview_time=args.preview_time,  # pass in preview time
    )

    # if dry run mode, wrap the environment with the wrapper
    if args.dry_run:
        logging.info("\n" + "=" * 80)
        logging.info("🔍 dry run mode enabled")
        logging.info("   - policy action output will be printed")
        logging.info("   - action will not be sent to robot")
        logging.info("   - robot will stay in initial position")
        logging.info("=" * 80 + "\n")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        logging.info("✅ normal mode: action will be executed to robot")
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
                )
            ),
            subscribers=[],
            max_hz=args.runtime_hz,  # runtime frequency, unit: Hz
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )
    else:
        runtime = _runtime.Runtime(
            environment=environment,
            agent=_policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=ws_client_policy,
                    action_horizon=args.action_horizon,
                )
            ),
            subscribers=[],
            max_hz=args.runtime_hz,  # runtime frequency, unit: Hz
            num_episodes=args.num_episodes,
            max_episode_steps=args.max_episode_steps,
        )

    def safe_disconnect():
        """safe disconnect robot"""
        try:
            # check if the environment is a wrapper
            actual_env = environment
            if isinstance(environment, DryRunEnvironmentWrapper):
                actual_env = environment._wrapped_env

            if hasattr(actual_env, "_env") and hasattr(actual_env._env, "robot"):
                if actual_env._env.robot.is_connected:
                    logging.info("safe disconnect robot...")
                    actual_env._env.disconnect()
                    logging.info("✓ robot disconnected")
                else:
                    logging.info("robot not connected, no need to disconnect")
        except Exception as disconnect_error:
            logging.warning(f"error disconnecting robot: {disconnect_error}")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n⚠️  detected user interrupt (Ctrl+C)")
        logging.info("program exited safely")
    except Exception as e:
        logging.error(f"\n❌ runtime error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # always safe disconnect robot
        safe_disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
