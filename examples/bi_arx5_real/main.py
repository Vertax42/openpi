import dataclasses
import logging
import sys

import tyro
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent

# 添加 lerobot-ARX5 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from examples.bi_arx5_real import env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    action_horizon: int = 25

    num_episodes: int = 1
    max_episode_steps: int = 1000

    # bi_arx5 specific configs (基于 ARX5 SDK，无ROS)
    left_arm_port: str = "can1"
    right_arm_port: str = "can3"
    log_level: str = "INFO"
    use_multithreading: bool = True


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()

    # 创建环境
    environment = _env.BiARX5RealEnvironment(
        left_arm_port=args.left_arm_port,
        right_arm_port=args.right_arm_port,
        log_level=args.log_level,
        use_multithreading=args.use_multithreading,
        reset_position=metadata.get("reset_pose"),
    )

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

    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("⚠️  检测到用户中断 (Ctrl+C)，正在安全断开机器人连接...")
        try:
            # 尝试安全断开机器人连接
            if hasattr(environment, "_env") and hasattr(environment._env, "robot"):
                if environment._env.robot.is_connected:
                    environment._env.disconnect()
                    logging.info("✓ 机器人已安全断开连接")
                else:
                    logging.info("机器人未连接，无需断开")
        except Exception as disconnect_error:
            logging.warning(f"断开连接时出现错误: {disconnect_error}")
        logging.info("程序已安全退出")
    except Exception as e:
        logging.error(f"运行时错误: {e}")
        # 尝试安全断开连接
        try:
            if hasattr(environment, "_env") and hasattr(environment._env, "robot"):
                if environment._env.robot.is_connected:
                    environment._env.disconnect()
                    logging.info("✓ 机器人已安全断开连接")
        except Exception as cleanup_error:
            logging.warning(f"清理时出现错误: {cleanup_error}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
