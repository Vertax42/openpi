from typing import List, Optional
import logging

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import numpy as np

import examples.bi_arx5_real.real_env as _real_env

logger = logging.getLogger(__name__)


class BiARX5RealEnvironment(_environment.Environment):
    """An environment for BiARX5 robot on real hardware based on lerobot implementation."""

    def __init__(
        self,
        left_arm_port: str = "can1",
        right_arm_port: str = "can3",
        log_level: str = "INFO",
        use_multithreading: bool = True,
        reset_position: Optional[List[float]] = None,
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,  # 是否立即连接机器人硬件
        controller_dt: float = 0.002,  # 底层控制频率 (秒)
        preview_time: float = 0.02,  # 预览时间 (秒)
    ) -> None:
        self._env = _real_env.make_bi_arx5_real_env(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            reset_position=reset_position,
            setup_robot=setup_robot,  # 传递 setup_robot 参数
            controller_dt=controller_dt,  # 传递控制频率参数
            preview_time=preview_time,  # 传递预览时间参数
        )
        self._render_height = render_height
        self._render_width = render_width
        self._ts = None

    @override
    def reset(self) -> None:
        self._ts = self._env.reset()

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        # 创建新的图像字典，避免修改原始数据
        processed_images = {}

        # 处理图像数据 - 移除深度图像并调整尺寸
        for cam_name in obs["images"]:
            if "_depth" in cam_name:
                continue  # 跳过深度图像

            # 原始图像已经是 uint8 格式 (480, 640, 3)
            single_img = obs["images"][cam_name]

            # 调试：打印原始图像信息
            logger.debug(
                f"Camera {cam_name}: original shape={single_img.shape}, dtype={single_img.dtype}"
            )

            # 将单个图像扩展为批次格式 [1, H, W, C] 以使用 resize_with_pad
            batch_img = np.expand_dims(single_img, axis=0)  # [H, W, C] -> [1, H, W, C]
            logger.debug(f"Camera {cam_name}: batch shape={batch_img.shape}")

            # 调整图像尺寸到指定分辨率
            resized_batch = image_tools.resize_with_pad(
                batch_img, self._render_height, self._render_width
            )
            logger.debug(
                f"Camera {cam_name}: resized batch shape={resized_batch.shape}"
            )

            # 取出批次中的第一个图像 [1, H, W, C] -> [H, W, C]
            resized_img = resized_batch[0]  # 已经是 uint8 格式
            logger.debug(f"Camera {cam_name}: resized shape={resized_img.shape}")

            # 转换为 OpenPI 期望的格式 (H,W,C) -> (C,H,W)
            processed_images[cam_name] = einops.rearrange(resized_img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": processed_images,  # 使用新创建的字典
            # prompt 由 policy server 的 InjectDefaultPrompt 注入，无需在这里添加
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
