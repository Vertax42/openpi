#!/usr/bin/env python3
"""
详细测试观测变化 - 验证动作执行后的状态变化是否正确
"""

import logging
import sys
import traceback
import numpy as np
import time

# 添加路径
sys.path.insert(0, "/home/ubuntu/openpi")
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from examples.bi_arx5_real.env import BiARX5RealEnvironment

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_observation_changes():
    """详细测试观测变化"""
    print("=" * 80)
    print("详细测试观测变化")
    print("=" * 80)

    env = None

    try:
        # 询问是否连接硬件
        print("⚠️  注意：此测试需要连接真实的 BiARX5 硬件")
        try:
            user_input = input("是否继续连接硬件进行测试？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  用户中断，测试取消")
            return

        if user_input != "y":
            print("测试取消")
            return

        # 创建环境
        print("\n1. 创建环境并连接硬件...")
        env = BiARX5RealEnvironment(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="INFO",
            use_multithreading=True,
            setup_robot=True,
        )
        print("✓ 环境创建成功")

        # 重置环境
        print("\n2. 重置环境...")
        env.reset()
        print("✓ 环境重置完成")

        # 获取初始观测
        print("\n3. 获取初始观测...")
        time.sleep(0.5)  # 等待机器人稳定
        obs1 = env.get_observation()
        state1 = obs1["state"].copy()  # 深拷贝避免引用问题

        print(f"初始状态:")
        print(f"  - 左臂关节 (0-5): {state1[0:6]}")
        print(f"  - 左夹爪 (6): {state1[6]:.6f}")
        print(f"  - 右臂关节 (7-12): {state1[7:13]}")
        print(f"  - 右夹爪 (13): {state1[13]:.6f}")

        # 测试1: 发送一个较大的确定性动作
        print("\n4. 测试1: 发送确定性大幅动作...")

        # 创建目标动作：在当前位置基础上每个关节+0.1弧度（约5.7度）
        target_action = state1.copy()

        # 只改变关节1和关节2（较安全的关节）
        target_action[1] += 0.1  # 左臂关节2 +0.1 rad
        target_action[8] += 0.1  # 右臂关节2 +0.1 rad

        # 夹爪保持不变
        target_action[6] = np.clip(target_action[6], 0.002, 0.08)
        target_action[13] = np.clip(target_action[13], 0.002, 0.08)

        print(f"目标动作:")
        print(
            f"  - 左臂关节2 (1): {state1[1]:.6f} -> {target_action[1]:.6f} (变化 +0.1)"
        )
        print(
            f"  - 右臂关节2 (8): {state1[8]:.6f} -> {target_action[8]:.6f} (变化 +0.1)"
        )

        # 执行动作
        print("\n执行动作...")
        env.apply_action({"actions": target_action})
        print("✓ 动作发送成功")

        # 等待机器人响应
        print("\n等待机器人响应...")
        for i in range(5):
            time.sleep(0.2)
            obs2 = env.get_observation()
            state2 = obs2["state"].copy()

            # 计算变化
            state_diff = state2 - state1

            print(f"\n  时刻 {(i+1)*0.2:.1f}s:")
            print(f"    - 左臂关节2 (1): {state2[1]:.6f} (变化: {state_diff[1]:+.6f})")
            print(f"    - 右臂关节2 (8): {state2[8]:.6f} (变化: {state_diff[8]:+.6f})")
            print(f"    - 最大变化: {np.abs(state_diff).max():.6f}")
            print(f"    - 平均变化: {np.abs(state_diff).mean():.6f}")

        # 最终状态对比
        print("\n最终状态对比:")
        final_state = state2
        total_change = final_state - state1

        print(f"  左臂关节2 (1):")
        print(f"    - 初始: {state1[1]:.6f}")
        print(f"    - 目标: {target_action[1]:.6f}")
        print(f"    - 最终: {final_state[1]:.6f}")
        print(f"    - 实际变化: {total_change[1]:+.6f}")
        print(f"    - 期望变化: +0.1")
        print(f"    - 达成率: {(total_change[1]/0.1)*100:.1f}%")

        print(f"\n  右臂关节2 (8):")
        print(f"    - 初始: {state1[8]:.6f}")
        print(f"    - 目标: {target_action[8]:.6f}")
        print(f"    - 最终: {final_state[8]:.6f}")
        print(f"    - 实际变化: {total_change[8]:+.6f}")
        print(f"    - 期望变化: +0.1")
        print(f"    - 达成率: {(total_change[8]/0.1)*100:.1f}%")

        # 分析所有关节的变化
        print("\n所有关节变化分析:")
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

        for i in range(14):
            if abs(total_change[i]) > 0.001:  # 只显示有明显变化的
                print(
                    f"  {joint_names[i]:12s} (索引{i:2d}): {total_change[i]:+.6f} rad"
                )

        # 评估结果
        print("\n" + "=" * 80)
        if abs(total_change[1]) > 0.05 or abs(total_change[8]) > 0.05:
            print("✅ 测试通过：观测变化正常，机器人响应良好")
            print(f"   - 检测到显著的关节运动（>0.05 rad）")
        elif abs(total_change[1]) > 0.01 or abs(total_change[8]) > 0.01:
            print("⚠️  测试部分通过：检测到轻微运动，但未达到目标")
            print(f"   - 可能是 preview_time 或控制增益设置的影响")
        else:
            print("❌ 测试失败：几乎没有检测到运动")
            print(f"   - 可能的原因：")
            print(f"     1. 机器人仍处于重力补偿模式")
            print(f"     2. 动作未正确发送到机器人")
            print(f"     3. 控制器参数设置不当")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            if env and hasattr(env, "_env") and hasattr(env._env, "robot"):
                if env._env.robot.is_connected:
                    print("\n正在断开机器人连接...")
                    env._env.disconnect()
                    print("✓ 机器人已安全断开")
        except Exception as cleanup_error:
            print(f"⚠️  清理时出现错误: {cleanup_error}")


if __name__ == "__main__":
    try:
        test_observation_changes()
    except KeyboardInterrupt:
        print("\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        traceback.print_exc()
