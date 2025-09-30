from typing import List, Optional
import logging

import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import numpy as np

from examples.bi_arx5_real import real_env as _real_env

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
    ) -> None:
        self._env = _real_env.make_bi_arx5_real_env(
            left_arm_port=left_arm_port,
            right_arm_port=right_arm_port,
            log_level=log_level,
            use_multithreading=use_multithreading,
            reset_position=reset_position,
            setup_robot=setup_robot,  # 传递 setup_robot 参数
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

        # 处理图像数据 - 移除深度图像
        for cam_name in list(obs["images"].keys()):
            if "_depth" in cam_name:
                del obs["images"][cam_name]

        # 调整图像尺寸并转换格式 (H,W,C) -> (C,H,W)
        for cam_name in obs["images"]:
            # 将单个图像扩展为批次格式 [1, H, W, C]，然后调用 resize_with_pad
            single_img = obs["images"][cam_name]
            batch_img = np.expand_dims(single_img, axis=0)  # [H, W, C] -> [1, H, W, C]

            resized_batch = image_tools.resize_with_pad(
                batch_img, self._render_height, self._render_width
            )

            # 取出批次中的第一个图像 [1, H, W, C] -> [H, W, C]
            resized_img = resized_batch[0]

            img = image_tools.convert_to_uint8(resized_img)
            obs["images"][cam_name] = einops.rearrange(img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": obs["images"],
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])
