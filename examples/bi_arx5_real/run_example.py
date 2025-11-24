#!/usr/bin/env python3
"""
BiARX5 推理环境运行示例
"""

import logging
import sys

# 添加 lerobot-ARX5 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from examples.bi_arx5_real.main import Args, main

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_bi_arx5_example():
    """运行 BiARX5 推理环境示例"""
    print("=" * 60)
    print("BiARX5 推理环境运行示例")
    print("=" * 60)

    # 配置参数
    args = Args(
        host="0.0.0.0",
        port=8000,
        action_horizon=25,
        num_episodes=1,
        max_episode_steps=1000,
        left_arm_port="can1",
        right_arm_port="can3",
        log_level="INFO",
        use_multithreading=True,
    )

    print("配置参数:")
    print(f"  - 服务器地址: {args.host}:{args.port}")
    print(f"  - 动作序列长度: {args.action_horizon}")
    print(f"  - 最大回合步数: {args.max_episode_steps}")
    print(f"  - 左臂端口: {args.left_arm_port}")
    print(f"  - 右臂端口: {args.right_arm_port}")
    print(f"  - 日志级别: {args.log_level}")

    print("\n⚠️  注意：")
    print("1. 确保 OpenPI 策略服务器已启动")
    print("2. 确保 CAN 总线已配置 (can1, can3)")
    print("3. 确保 BiARX5 硬件已连接")

    user_input = input("\n是否继续运行？(y/N): ").strip().lower()

    if user_input == "y":
        try:
            print("\n启动 BiARX5 推理环境...")
            main(args)
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断 (Ctrl+C)，程序已安全退出")
        except Exception as e:
            print(f"\n❌ 运行失败: {e}")
            logger.exception("详细错误信息:")
    else:
        print("已取消运行")


if __name__ == "__main__":
    run_bi_arx5_example()
