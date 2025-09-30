#!/usr/bin/env python3
"""
测试 BiARX5 推理环境集成
"""

import logging
import sys
import os

import numpy as np

# 添加 openpi 项目根目录到路径
sys.path.insert(0, "/home/ubuntu/openpi")
# 添加 lerobot-ARX5 路径
# sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from examples.bi_arx5_real.env import BiARX5RealEnvironment
from examples.bi_arx5_real.real_env import make_bi_arx5_real_env

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_bi_arx5_real_env():
    """测试 BiARX5 真实环境"""
    print("=" * 60)
    print("测试 BiARX5 真实推理环境")
    print("=" * 60)

    real_env = None
    connected_env = None

    try:
        # 1. 测试底层环境创建 (不连接硬件)
        print("\n1. 创建 BiARX5RealEnv (不连接硬件)...")
        real_env = make_bi_arx5_real_env(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="INFO",
            setup_robot=False,  # 不连接真实硬件
        )
        print("✓ BiARX5RealEnv 创建成功")
        print(f"  - 配置: {real_env.config.id}")
        print(f"  - 左臂端口: {real_env.config.left_arm_port}")
        print(f"  - 右臂端口: {real_env.config.right_arm_port}")

        # 2. 测试 OpenPI Environment 适配器 (会自动连接硬件)
        print("\n2. 创建 BiARX5RealEnvironment 适配器...")
        print("⚠️  注意：这将连接真实的 BiARX5 硬件")

        try:
            user_input = input("是否继续创建适配器？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断 (Ctrl+C)，程序已安全退出")
            return

        if user_input != "y":
            print("跳过硬件连接，仅测试接口兼容性")
            return

        connected_env = BiARX5RealEnvironment(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="INFO",
            render_height=224,
            render_width=224,
        )
        print("✓ BiARX5RealEnvironment 适配器创建成功")

        # 3. 测试动作格式转换
        print("\n3. 测试动作格式转换...")
        test_action = np.random.uniform(-1, 1, 14)  # 14维动作
        print(f"输入动作维度: {test_action.shape}")
        print(f"动作范围: [{test_action.min():.3f}, {test_action.max():.3f}]")

        # 模拟动作转换逻辑
        action_dict = {}
        for i in range(6):
            action_dict[f"left_joint_{i+1}.pos"] = float(test_action[i])
        action_dict["left_gripper.pos"] = float(test_action[6])

        for i in range(6):
            action_dict[f"right_joint_{i+1}.pos"] = float(test_action[7 + i])
        action_dict["right_gripper.pos"] = float(test_action[13])

        print("✓ 动作格式转换成功")
        print(f"  - 转换后动作字典包含 {len(action_dict)} 个键")

        # 3. 测试已连接的环境
        if connected_env:
            print("\n3. 测试已连接的环境...")

            try:
                user_input = input("是否测试机器人观测和动作？(y/N): ").strip().lower()
            except KeyboardInterrupt:
                print("\n⚠️  检测到用户中断 (Ctrl+C)，正在安全断开机器人连接...")
                try:
                    if connected_env._env.robot.is_connected:
                        connected_env._env.disconnect()
                        print("✓ 机器人已安全断开连接")
                except Exception as disconnect_error:
                    print(f"⚠️  断开连接时出现错误: {disconnect_error}")
                print("程序已安全退出")
                return

            if user_input == "y":
                try:
                    # 测试重置
                    print("测试环境重置...")
                    connected_env.reset()
                    print("✓ 环境重置成功")

                    # 获取观测
                    print("获取环境观测...")
                    obs = connected_env.get_observation()
                    print(f"✓ 观测获取成功")
                    print(f"  - state 维度: {obs['state'].shape}")
                    print(f"  - 图像数量: {len(obs['images'])}")
                    for cam_name, img in obs["images"].items():
                        print(f"    - {cam_name}: {img.shape}")

                    # 测试一步动作
                    print("测试执行动作...")
                    test_action_small = obs["state"] + np.random.uniform(
                        -0.01, 0.01, 14
                    )
                    connected_env.apply_action({"actions": test_action_small})
                    print("✓ 动作执行成功")

                except KeyboardInterrupt:
                    print("\n⚠️  检测到用户中断 (Ctrl+C)，正在安全断开机器人连接...")
                    try:
                        if connected_env._env.robot.is_connected:
                            connected_env._env.disconnect()
                            print("✓ 机器人已安全断开连接")
                    except Exception as disconnect_error:
                        print(f"⚠️  断开连接时出现错误: {disconnect_error}")
                    print("程序已安全退出")
                    return
                except Exception as e:
                    print(f"❌ 硬件测试失败: {e}")
                    print("可能的原因:")
                    print("  - 机器人通信异常")
                    print("  - 动作超出安全范围")
            else:
                print("跳过硬件功能测试")
        else:
            print("未连接硬件，跳过硬件测试")

        print("\n" + "=" * 60)
        print("BiARX5 推理环境测试完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，正在清理资源...")
        try:
            if connected_env and connected_env._env.robot.is_connected:
                connected_env._env.disconnect()
                print("✓ 已连接的环境已安全断开")
        except Exception as cleanup_error:
            print(f"⚠️  清理时出现错误: {cleanup_error}")
        print("测试程序已安全退出")
        return
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 最终清理
        try:
            if connected_env and connected_env._env.robot.is_connected:
                print("最终清理：断开已连接的环境...")
                connected_env._env.disconnect()
                print("✓ 环境清理完成")
        except Exception as final_cleanup_error:
            print(f"⚠️  最终清理时出现错误: {final_cleanup_error}")


def test_openpi_integration():
    """测试 OpenPI 集成"""
    print("\n" + "=" * 60)
    print("测试 OpenPI 集成")
    print("=" * 60)

    env = None
    try:
        # 模拟 OpenPI Runtime 的使用方式
        from openpi_client.runtime import environment as _environment

        # 创建环境 (注意：BiARX5RealEnvironment 会自动连接硬件)
        print("⚠️  注意：创建 BiARX5RealEnvironment 会自动连接硬件")
        try:
            user_input = input("是否继续创建环境进行集成测试？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断，跳过 OpenPI 集成测试")
            return

        if user_input != "y":
            print("跳过 OpenPI 集成测试")
            return

        env = BiARX5RealEnvironment(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="INFO",
            use_multithreading=True,
        )

        # 检查接口兼容性
        assert isinstance(env, _environment.Environment)
        print("✓ Environment 接口兼容性检查通过")

        # 检查必要方法
        methods = ["reset", "get_observation", "apply_action", "is_episode_complete"]
        for method in methods:
            assert hasattr(env, method)
            print(f"✓ 方法 {method} 存在")

        print("✓ OpenPI 集成测试通过")

    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，正在清理 OpenPI 集成测试...")
        try:
            if env and hasattr(env, "_env") and hasattr(env._env, "robot"):
                if env._env.robot.is_connected:
                    env._env.disconnect()
                    print("✓ 测试环境已安全断开连接")
        except Exception as cleanup_error:
            print(f"⚠️  清理时出现错误: {cleanup_error}")
        print("OpenPI 集成测试已安全退出")
        return
    except Exception as e:
        print(f"❌ OpenPI 集成测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 确保测试环境被清理
        try:
            if env and hasattr(env, "_env") and hasattr(env._env, "robot"):
                if env._env.robot.is_connected:
                    print("最终清理：断开测试环境连接...")
                    env._env.disconnect()
                    print("✓ 测试环境清理完成")
        except Exception as final_cleanup_error:
            print(f"⚠️  最终清理时出现错误: {final_cleanup_error}")


if __name__ == "__main__":
    try:
        # test_bi_arx5_real_env()
        test_openpi_integration()
    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，测试程序已安全退出")
    except Exception as e:
        print(f"\n❌ 测试程序异常退出: {e}")
        import traceback

        traceback.print_exc()
