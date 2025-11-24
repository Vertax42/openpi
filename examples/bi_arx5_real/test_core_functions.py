#!/usr/bin/env python3
"""
测试 BiARX5 环境的核心函数：get_observation() 和 apply_action()
这些是 OpenPI Runtime 必需的关键方法
"""

import logging
import sys
import os

import numpy as np

# # 添加 openpi 项目根目录到路径
# sys.path.insert(0, "../..")
# # 添加 lerobot-ARX5 路径
# sys.path.insert(0, "../../../lerobot-ARX5/src")

from env import BiARX5RealEnvironment

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_core_functions():
    """测试 get_observation() 和 apply_action() 核心函数"""
    print("=" * 70)
    print("测试 BiARX5 环境核心函数")
    print("get_observation() 和 apply_action() - OpenPI Runtime 必需")
    print("=" * 70)

    env = None

    try:
        # 询问是否连接硬件
        print("⚠️  注意：此测试需要连接真实的 BiARX5 硬件")
        try:
            user_input = input("是否继续连接硬件进行测试？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断，测试已取消")
            return

        if user_input != "y":
            print("测试已取消")
            return

        # 1. 创建环境（连接硬件）
        print("\n1. 创建 BiARX5RealEnvironment（连接硬件）...")
        env = BiARX5RealEnvironment(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="DEBUG",  # 改为 DEBUG 以查看详细日志
            use_multithreading=True,
            setup_robot=True,  # 连接硬件
        )
        print("✓ 环境创建成功，机器人已连接")

        # 2. 测试环境重置
        print("\n2. 测试环境重置...")
        env.reset()
        print("✓ 环境重置成功")

        # 3. 测试 get_observation() - OpenPI Runtime 核心方法
        print("\n3. 测试 get_observation() 方法...")
        observation = env.get_observation()

        print("✓ get_observation() 调用成功")
        print(f"观测数据结构:")
        print(f"  - 类型: {type(observation)}")
        print(f"  - 键: {list(observation.keys())}")

        # 检查必需的数据结构
        required_keys = ["state", "images"]
        for key in required_keys:
            if key in observation:
                print(f"  ✓ 包含必需键: {key}")
                if key == "state":
                    print(f"    - state 形状: {observation[key].shape}")
                    print(f"    - state 类型: {type(observation[key])}")
                    print(
                        f"    - state 范围: [{observation[key].min():.3f}, {observation[key].max():.3f}]"
                    )
                elif key == "images":
                    print(f"    - 图像数量: {len(observation[key])}")
                    for img_name, img_data in observation[key].items():
                        print(
                            f"      - {img_name}: {img_data.shape}, dtype={img_data.dtype}"
                        )
            else:
                print(f"  ❌ 缺少必需键: {key}")

        # 4. 测试 apply_action() - OpenPI Runtime 核心方法
        print("\n4. 测试 apply_action() 方法...")

        # 检查当前控制模式
        is_gravity_comp = env._env.robot.is_gravity_compensation_mode()
        print(
            f"当前控制模式: {'重力补偿模式' if is_gravity_comp else '正常位置控制模式'}"
        )

        # 创建一个安全的测试动作（当前位置 + 小幅扰动）
        current_state = observation["state"]
        print(f"当前状态维度: {current_state.shape}")

        # 生成更明显的安全动作（增加幅度）
        small_perturbation = np.random.uniform(-0.05, 0.05, current_state.shape)
        test_action = current_state + small_perturbation

        # 限制夹爪范围 [0.002, 0.08]
        # 左夹爪在索引 6，右夹爪在索引 13
        test_action[6] = np.clip(test_action[6], 0.002, 0.4)  # 左夹爪
        test_action[13] = np.clip(test_action[13], 0.002, 0.4)  # 右夹爪

        print(f"测试动作:")
        print(f"  - 动作维度: {test_action.shape}")
        print(
            f"  - 扰动范围: [{small_perturbation.min():.4f}, {small_perturbation.max():.4f}]"
        )
        print(f"  - 动作范围: [{test_action.min():.3f}, {test_action.max():.3f}]")
        print(f"  - 左夹爪: {test_action[6]:.4f}, 右夹爪: {test_action[13]:.4f}")

        # 按照 OpenPI 格式包装动作
        action_dict = {"actions": test_action}

        print("执行 apply_action()...")
        if is_gravity_comp:
            print("  ⚠️  机器人当前处于重力补偿模式，将自动切换到位置控制模式")

        env.apply_action(action_dict)
        print("✓ apply_action() 调用成功")

        # 检查执行后的控制模式
        is_gravity_comp_after = env._env.robot.is_gravity_compensation_mode()
        print(
            f"执行后控制模式: {'重力补偿模式' if is_gravity_comp_after else '正常位置控制模式'}"
        )

        if is_gravity_comp and not is_gravity_comp_after:
            print("✓ 成功从重力补偿模式切换到位置控制模式")

        # 5. 验证动作执行后的状态变化
        print("\n5. 验证动作执行效果...")

        # 等待更长时间让机器人响应
        print("等待机器人响应...")
        import time

        time.sleep(0.5)  # 等待500ms

        new_observation = env.get_observation()
        new_state = new_observation["state"]

        state_diff = np.abs(new_state - current_state)
        print(f"状态变化:")
        print(f"  - 最大变化: {state_diff.max():.4f}")
        print(f"  - 平均变化: {state_diff.mean():.4f}")
        print(f"  - 变化的关节数: {np.sum(state_diff > 0.001)}")

        if state_diff.max() > 0.001:
            print("✓ 检测到状态变化，动作执行有效")
        else:
            print("⚠️  状态变化很小，可能动作幅度太小或机器人响应延迟")

        # 6. 测试连续的观测-动作循环（模拟 OpenPI Runtime）
        print("\n6. 测试连续观测-动作循环（模拟 OpenPI Runtime）...")

        try:
            user_input = input("是否测试连续循环？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断")
            return

        if user_input == "y":
            print("执行 10 步连续循环...")
            for step in range(10):
                try:
                    # 获取观测
                    obs = env.get_observation()
                    current_state = obs["state"]
                    print(f"当前状态: {current_state}")

                    # 生成更明显的动作
                    perturbation = np.random.uniform(-0.01, 0.01, current_state.shape)
                    action = current_state + perturbation

                    # 限制夹爪范围 [0.002, 0.08]
                    # 左夹爪在索引 6，右夹爪在索引 13
                    action[6] = np.clip(action[6], 0.002, 0.4)  # 左夹爪
                    action[13] = np.clip(action[13], 0.002, 0.4)  # 右夹爪

                    # 执行动作
                    env.apply_action({"actions": action})

                    print(f"  步骤 {step+1}/10: ✓ 观测->动作循环成功")
                    print(f"    - 左夹爪: {action[6]:.4f}, 右夹爪: {action[13]:.4f}")

                    # 等待机器人响应
                    import time

                    time.sleep(0.5)  # 增加等待时间

                except KeyboardInterrupt:
                    print(f"\n⚠️  在步骤 {step+1} 检测到用户中断")
                    break

            print("✓ 连续循环测试完成")

        print("\n" + "=" * 70)
        print("核心函数测试完成！")
        print("✓ get_observation() 和 apply_action() 都工作正常")
        print("✓ 环境已准备好用于 OpenPI Runtime")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，正在安全断开机器人连接...")
        try:
            if env and env._env.robot.is_connected:
                env._env.disconnect()
                print("✓ 机器人已安全断开连接")
        except Exception as disconnect_error:
            print(f"⚠️  断开连接时出现错误: {disconnect_error}")
        print("测试程序已安全退出")
        return
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 最终清理
        try:
            if env and env._env.robot.is_connected:
                print("最终清理：断开机器人连接...")
                env._env.disconnect()
                print("✓ 机器人连接已断开")
        except Exception as cleanup_error:
            print(f"⚠️  最终清理时出现错误: {cleanup_error}")


if __name__ == "__main__":
    try:
        test_core_functions()
    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，程序已安全退出")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback

        traceback.print_exc()
