"""OpenPI Environment wrapper for Flexiv Rizon4 robot."""

import einops
from lerobot.utils.robot_utils import get_logger
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

import examples.flexiv_rizon4_real.real_env as _real_env

logger = get_logger("FlexivRizon4Env")


class FlexivRizon4RealEnvironment(_environment.Environment):
    """An environment for Flexiv Rizon4 robot on real hardware based on lerobot implementation."""

    def __init__(
        self,
        robot_sn: str = "Rizon4-063423",
        control_mode: str = "cartesian_motion_force",
        use_gripper: bool = True,
        use_force: bool = False,
        go_to_start: bool = True,
        log_level: str = "INFO",
        render_height: int = 224,
        render_width: int = 224,
        setup_robot: bool = True,
        # Flare gripper settings
        flare_gripper_mac_addr: str = "e2b26adbb104",
        flare_gripper_cam_size: tuple[int, int] = (640, 480),
        flare_gripper_rectify_size: tuple[int, int] = (200, 350),
        flare_gripper_max_pos: float = 85.0,
        # External cameras
        cameras: dict | None = None,
    ) -> None:
        self._env = _real_env.make_flexiv_rizon4_real_env(
            robot_sn=robot_sn,
            control_mode=control_mode,
            use_gripper=use_gripper,
            use_force=use_force,
            go_to_start=go_to_start,
            log_level=log_level,
            setup_robot=setup_robot,
            flare_gripper_mac_addr=flare_gripper_mac_addr,
            flare_gripper_cam_size=flare_gripper_cam_size,
            flare_gripper_rectify_size=flare_gripper_rectify_size,
            flare_gripper_max_pos=flare_gripper_max_pos,
            cameras=cameras,
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

    # Camera name mapping from real environment to policy expected names
    # Policy expects: 'observation/wrist_image_left' (required), 'cam_high', 'cam_right_wrist' (optional)
    # Real env produces: 'wrist_cam', 'left_tactile', 'right_tactile'
    CAMERA_NAME_MAP = {
        "wrist_cam": "observation/wrist_image_left",
        # Tactile images not used by current policy, but could be mapped if needed:
        # "left_tactile": "left_tactile",
        # "right_tactile": "right_tactile",
    }

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")

        obs = self._ts.observation

        # Create new image dict to avoid modifying original data
        processed_images = {}

        # Process image data - resize to target resolution
        for cam_name in obs["images"]:
            if "_depth" in cam_name:
                continue  # Skip depth images

            # Skip cameras not in the mapping (e.g., tactile sensors not used by policy)
            if cam_name not in self.CAMERA_NAME_MAP:
                logger.debug(f"Skipping camera {cam_name} (not in CAMERA_NAME_MAP)")
                continue

            # Original image is already uint8 format (H, W, 3)
            single_img = obs["images"][cam_name]

            logger.debug(f"Camera {cam_name}: original shape={single_img.shape}, dtype={single_img.dtype}")

            # Expand single image to batch format [1, H, W, C] for resize_with_pad
            batch_img = np.expand_dims(single_img, axis=0)

            # Resize image to specified resolution
            resized_batch = image_tools.resize_with_pad(batch_img, self._render_height, self._render_width)

            # Extract first image from batch [1, H, W, C] -> [H, W, C]
            resized_img = resized_batch[0]

            # Convert to OpenPI expected format (H,W,C) -> (C,H,W)
            # Use mapped name for policy compatibility
            policy_cam_name = self.CAMERA_NAME_MAP[cam_name]
            processed_images[policy_cam_name] = einops.rearrange(resized_img, "h w c -> c h w")

        return {
            "state": obs["qpos"],
            "images": processed_images,
            # prompt is injected by policy server's InjectDefaultPrompt
        }

    @override
    def apply_action(self, action: dict) -> None:
        self._ts = self._env.step(action["actions"])

    def disconnect(self) -> None:
        """Disconnect from the robot."""
        self._env.disconnect()
