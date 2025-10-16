import collections
import logging
import sys
import time
from typing import List, Optional

import dm_env
import numpy as np

# 添加 lerobot-ARX5 路径
sys.path.insert(0, "/home/ubuntu/lerobot-ARX5/src")

# 导入 lerobot bi_arx5 (直接基于 ARX5 SDK，无ROS)
from lerobot.robots.bi_arx5.config_bi_arx5 import BiARX5Config
from lerobot.robots.utils import make_robot_from_config

logger = logging.getLogger(__name__)

# 默认重置位置 (关节角度 + 夹爪位置) - 与你的 lerobot 配置保持一致
DEFAULT_RESET_POSITION = [0.0, 0.948, 0.858, -0.573, 0.0, 0.0, 0.002]


class BiARX5RealEnv:
    """
    Environment for real BiARX5 robot bi-manual manipulation
    Based on lerobot BiARX5 implementation (No ROS, direct ARX5 SDK)

    Action space:      [left_joint_1.pos, ..., left_joint_6.pos, left_gripper.pos,
                        right_joint_1.pos, ..., right_joint_6.pos, right_gripper.pos]  # 14 dims

    Observation space: {"qpos": Concat[left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)],
                        "qvel": similar structure (zeros for BiARX5),
                        "images": {"cam_high": (H,W,3), "cam_left_wrist": (H,W,3), "cam_right_wrist": (H,W,3)}}
    """

    def __init__(
        self,
        left_arm_port: str = "can1",
        right_arm_port: str = "can3",
        log_level: str = "INFO",
        use_multithreading: bool = True,
        reset_position: Optional[List[float]] = None,
        setup_robot: bool = True,
    ):
        self._reset_position = (
            reset_position if reset_position else DEFAULT_RESET_POSITION
        )

        # 创建 BiARX5 配置 - 直接使用你现有的 lerobot 配置
        self.config = BiARX5Config(
            id="bi_arx5_real_inference",
            left_arm_model="X5",
            left_arm_port=left_arm_port,
            right_arm_model="X5",
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            inference_mode=True,  # 推理模式，设置合适的 preview_time
            controller_dt=0.002,  # 1ms = 1000Hz 底层控制频率
            preview_time=0.02,  # 20ms 预览时间
        )

        # 创建机器人实例 - 使用你现有的 BiARX5 类
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """连接并初始化机器人 (无ROS，直接ARX5 SDK)"""
        logger.info("Connecting to BiARX5 robot via ARX5 SDK...")
        try:
            self.robot.connect(calibrate=False, go_to_start=True)
            logger.info("BiARX5 robot connected and ready for inference")
        except Exception as e:
            logger.error(f"Failed to connect BiARX5 robot: {e}")
            raise

    def get_qpos(self, obs):
        """获取关节位置 - 直接从你的 BiARX5.get_observation() 获取"""
        # 按照 aloha_real 的格式组织: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
        left_joints = [obs[f"left_joint_{i+1}.pos"] for i in range(6)]
        left_gripper = [obs["left_gripper.pos"]]

        right_joints = [obs[f"right_joint_{i+1}.pos"] for i in range(6)]
        right_gripper = [obs["right_gripper.pos"]]

        return np.concatenate([left_joints, left_gripper, right_joints, right_gripper])

    def get_qvel(self, obs):
        """获取关节速度 - BiARX5 SDK 暂不提供速度反馈，返回零向量"""
        return np.zeros(14)  # 6+1 + 6+1 = 14

    def get_images(self, obs):
        """获取相机图像 - 直接从你的 BiARX5.get_observation() 获取"""
        images = {}

        # 映射你的相机名称到 aloha_real 期望的名称
        camera_mapping = {
            "head": "cam_high",  # 头部相机 -> cam_high
            "left_wrist": "cam_left_wrist",  # 左腕相机 -> cam_left_wrist
            "right_wrist": "cam_right_wrist",  # 右腕相机 -> cam_right_wrist
        }

        for lerobot_name, openpi_name in camera_mapping.items():
            if lerobot_name in obs:
                images[openpi_name] = obs[lerobot_name]
            else:
                logger.warning(f"Camera {lerobot_name} not found in observation")

        return images

    def get_observation(self):
        """获取完整观测 - 兼容 aloha_real 格式"""
        current_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(current_obs)
        obs["qvel"] = self.get_qvel(current_obs)
        obs["images"] = self.get_images(current_obs)
        return obs

    def get_reward(self):
        return 0

    def reset(self, *, fake=False):
        """重置机器人到初始位置 - 使用你的 smooth_go_start()"""
        if not fake:
            logger.info("Resetting BiARX5 to start position...")
            try:
                # 使用你现有的平滑移动方法
                self.robot.smooth_go_start(duration=2.0)
                logger.info("BiARX5 reset completed")
            except Exception as e:
                logger.error(f"Failed to reset BiARX5: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action):
        """执行动作 - 转换格式后调用你的 send_action()"""
        # 确保机器人处于正常位置控制模式（而不是重力补偿模式）
        if self.robot.is_gravity_compensation_mode():
            logger.info(
                "Switching from gravity compensation to normal position control for action execution"
            )
            self.robot.set_to_normal_position_control()
        # else:
        #     logger.info("Robot is already in normal position control mode")

        # 将 OpenPI 的动作数组转换为你的 BiARX5 动作字典格式
        action_dict = {}

        # 左臂动作 (前7个: 6关节 + 夹爪)
        for i in range(6):
            action_dict[f"left_joint_{i+1}.pos"] = float(action[i])
        action_dict["left_gripper.pos"] = float(action[6])

        # 右臂动作 (后7个: 6关节 + 夹爪)
        for i in range(6):
            action_dict[f"right_joint_{i+1}.pos"] = float(action[7 + i])
        action_dict["right_gripper.pos"] = float(action[13])
        # print(
        #     "gripper_action",
        #     action_dict["left_gripper.pos"],
        #     action_dict["right_gripper.pos"],
        # )

        # 发送动作到你的 BiARX5 机器人
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action to BiARX5: {e}")
            raise

        # 等待一个控制周期 (你的配置中 controller_dt = 0.01，即100Hz)
        # time.sleep(self.config.controller_dt)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def disconnect(self):
        """断开机器人连接"""
        if self.robot.is_connected:
            logger.info("Disconnecting BiARX5 robot...")
            try:
                self.robot.disconnect()
                logger.info("BiARX5 robot disconnected")
            except Exception as e:
                logger.warning(f"Error during BiARX5 disconnect: {e}")


def make_bi_arx5_real_env(
    left_arm_port: str = "can1",
    right_arm_port: str = "can3",
    log_level: str = "INFO",
    use_multithreading: bool = True,
    reset_position: Optional[List[float]] = None,
    setup_robot: bool = True,
) -> BiARX5RealEnv:
    """创建 BiARX5 真实环境 (基于 lerobot 实现，无ROS版本)"""
    return BiARX5RealEnv(
        left_arm_port=left_arm_port,
        right_arm_port=right_arm_port,
        log_level=log_level,
        use_multithreading=use_multithreading,
        reset_position=reset_position,
        setup_robot=setup_robot,
    )
