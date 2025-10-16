import dataclasses
import logging
import sys
from typing import Any

import numpy as np
import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime import environment as _environment
from typing_extensions import override

# # 添加 openpi 项目根目录到路径
# sys.path.insert(0, "/home/ubuntu/openpi")
# # 添加 lerobot-ARX5 路径
# sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

import examples.bi_arx5_real.env as _env


class DryRunEnvironmentWrapper(_environment.Environment):
    """包装器：拦截并打印 policy 动作，但不实际执行"""

    def __init__(self, wrapped_env: _environment.Environment):
        self._wrapped_env = wrapped_env
        self._step_count = 0
        self._episode_count = 0

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._step_count = 0
        logging.info(f"\n{'='*80}")
        logging.info(f"🔄 Episode {self._episode_count} - 环境重置 (干跑模式)")
        logging.info(f"{'='*80}\n")
        self._wrapped_env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return self._wrapped_env.is_episode_complete()

    @override
    def get_observation(self) -> dict:
        obs = self._wrapped_env.get_observation()

        # prompt 由 policy server 的 InjectDefaultPrompt transform 自动注入
        # 无需在客户端添加

        # 打印观测信息（简化版）
        if self._step_count % 10 == 0:  # 每10步打印一次观测摘要
            state = obs.get("state")
            images = obs.get("images", {})
            logging.info(f"📊 步骤 {self._step_count} - 观测摘要:")
            if state is not None:
                logging.info(
                    f"   状态维度: {state.shape}, 范围: [{state.min():.3f}, {state.max():.3f}]"
                )
            logging.info(f"   图像数量: {len(images)}")

        return obs

    @override
    def apply_action(self, action: dict) -> None:
        self._step_count += 1

        # 打印 policy 输出的动作
        actions = action.get("actions")
        if actions is not None:
            logging.info(f"\n{'─'*80}")
            logging.info(f"🎯 步骤 {self._step_count} - Policy 输出动作:")
            logging.info(f"{'─'*80}")

            # 打印动作的详细信息
            logging.info(f"动作维度: {actions.shape}")
            logging.info(f"动作类型: {actions.dtype}")
            logging.info(f"动作范围: [{actions.min():.6f}, {actions.max():.6f}]")

            # 打印每个关节的动作值
            joint_names = [
                "左臂关节1",
                "左臂关节2",
                "左臂关节3",
                "左臂关节4",
                "左臂关节5",
                "左臂关节6",
                "左夹爪",
                "右臂关节1",
                "右臂关节2",
                "右臂关节3",
                "右臂关节4",
                "右臂关节5",
                "右臂关节6",
                "右夹爪",
            ]

            logging.info(f"\n详细动作值:")
            for i, (name, value) in enumerate(zip(joint_names, actions)):
                logging.info(f"  [{i:2d}] {name:12s}: {value:+.6f} rad")

            logging.info(f"\n夹爪动作:")
            logging.info(f"  左夹爪 (索引6):  {actions[6]:.6f}")
            logging.info(f"  右夹爪 (索引13): {actions[13]:.6f}")

            logging.info(f"{'─'*80}")
            logging.info(f"⚠️  干跑模式：动作已拦截，未实际执行到机器人")
            logging.info(f"{'─'*80}\n")

        # 不调用 wrapped_env.apply_action()，这样就不会实际执行动作


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 50  # 匹配 Pi0.5 模型的 action_horizon

    num_episodes: int = 1
    max_episode_steps: int = 10000

    # bi_arx5 specific configs (基于 ARX5 SDK，无ROS)
    left_arm_port: str = "can1"
    right_arm_port: str = "can3"
    log_level: str = "INFO"
    use_multithreading: bool = True

    # 干跑模式：只打印 policy 输出，不实际执行动作
    dry_run: bool = False


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    # 创建基础环境
    base_environment = _env.BiARX5RealEnvironment(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        log_level=args.log_level,
        use_multithreading=args.use_multithreading,
        reset_position=metadata.get("reset_pose"),
    )

    # 如果是干跑模式，用包装器包装环境
    if args.dry_run:
        logging.info("\n" + "=" * 80)
        logging.info("🔍 干跑模式已启用")
        logging.info("   - Policy 的动作输出将被打印")
        logging.info("   - 动作不会实际发送到机器人")
        logging.info("   - 机器人将保持在初始位置")
        logging.info("=" * 80 + "\n")
        environment = DryRunEnvironmentWrapper(base_environment)
    else:
        logging.info("✅ 正常模式：动作将实际执行到机器人")
        environment = base_environment

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=50,  # 与 controller_dt=0.01 (100Hz) 兼容
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    def safe_disconnect():
        """安全断开机器人连接"""
        try:
            # 检查是否是包装器环境
            actual_env = environment
            if isinstance(environment, DryRunEnvironmentWrapper):
                actual_env = environment._wrapped_env

            if hasattr(actual_env, "_env") and hasattr(actual_env._env, "robot"):
                if actual_env._env.robot.is_connected:
                    logging.info("正在安全断开机器人连接...")
                    actual_env._env.disconnect()
                    logging.info("✓ 机器人已安全断开连接")
                else:
                    logging.info("机器人未连接，无需断开")
        except Exception as disconnect_error:
            logging.warning(f"断开连接时出现错误: {disconnect_error}")

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("\n⚠️  检测到用户中断 (Ctrl+C)")
        logging.info("程序已安全退出")
    except Exception as e:
        logging.error(f"\n❌ 运行时错误: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # 无论如何都要安全断开机器人
        safe_disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
