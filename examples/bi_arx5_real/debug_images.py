#!/usr/bin/env python3
"""
调试 BiARX5 相机图像格式
"""

import logging
import sys

import numpy as np

# 添加 openpi 项目根目录到路径
sys.path.insert(0, "/home/ubuntu/openpi")
# 添加 lerobot-ARX5 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

from examples.bi_arx5_real.real_env import make_bi_arx5_real_env

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def debug_camera_images():
    """调试相机图像格式"""
    print("=" * 60)
    print("调试 BiARX5 相机图像格式")
    print("=" * 60)

    env = None

    try:
        # 询问是否连接硬件
        print("⚠️  注意：此调试需要连接真实的 BiARX5 硬件")
        try:
            user_input = input("是否继续连接硬件进行调试？(y/N): ").strip().lower()
        except KeyboardInterrupt:
            print("\n⚠️  检测到用户中断，调试已取消")
            return

        if user_input != "y":
            print("调试已取消")
            return

        # 创建环境（连接硬件）
        print("\n创建 BiARX5RealEnv（连接硬件）...")
        env = make_bi_arx5_real_env(
            left_arm_port="can1",
            right_arm_port="can3",
            log_level="INFO",
            setup_robot=True,
        )
        print("✓ 环境创建成功，机器人已连接")

        # 获取原始观测数据
        print("\n获取原始观测数据...")
        obs = env.get_observation()

        print(f"观测数据结构:")
        print(f"  - 类型: {type(obs)}")
        print(f"  - 键: {list(obs.keys())}")

        # 详细检查图像数据
        if "images" in obs:
            images = obs["images"]
            print(f"\n图像数据详情:")
            print(f"  - 图像数量: {len(images)}")

            for cam_name, img_data in images.items():
                print(f"\n  相机: {cam_name}")
                print(f"    - 类型: {type(img_data)}")

                if img_data is not None:
                    if hasattr(img_data, "shape"):
                        print(f"    - 形状: {img_data.shape}")
                        print(f"    - 数据类型: {img_data.dtype}")
                        print(f"    - 最小值: {img_data.min()}")
                        print(f"    - 最大值: {img_data.max()}")
                        print(f"    - 维度数: {len(img_data.shape)}")

                        # 详细分析形状
                        if len(img_data.shape) == 3:
                            h, w, c = img_data.shape
                            print(f"    - 高度: {h}, 宽度: {w}, 通道: {c}")
                            if c == 3:
                                print(f"    - ✓ 正常的 RGB 图像格式")
                            else:
                                print(f"    - ⚠️  异常的通道数: {c}")
                        elif len(img_data.shape) == 2:
                            h, w = img_data.shape
                            print(f"    - 高度: {h}, 宽度: {w}")
                            print(f"    - ⚠️  灰度图像或缺少通道维度")
                        else:
                            print(f"    - ❌ 异常的图像维度: {img_data.shape}")

                        # 检查数据内容
                        if img_data.size > 0:
                            unique_values = np.unique(img_data)
                            print(f"    - 唯一值数量: {len(unique_values)}")
                            if len(unique_values) <= 10:
                                print(f"    - 唯一值: {unique_values}")

                    else:
                        print(f"    - ❌ 不是 numpy 数组: {img_data}")
                else:
                    print(f"    - ❌ 图像数据为 None")
        else:
            print("❌ 观测数据中没有 'images' 键")

        print("\n" + "=" * 60)
        print("图像格式调试完成！")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)")
    except Exception as e:
        print(f"❌ 调试失败: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理
        try:
            if env and env.robot.is_connected:
                print("断开机器人连接...")
                env.disconnect()
                print("✓ 机器人连接已断开")
        except Exception as cleanup_error:
            print(f"⚠️  清理时出现错误: {cleanup_error}")


if __name__ == "__main__":
    try:
        debug_camera_images()
    except KeyboardInterrupt:
        print("\n⚠️  检测到用户中断 (Ctrl+C)，程序已安全退出")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        import traceback

        traceback.print_exc()
