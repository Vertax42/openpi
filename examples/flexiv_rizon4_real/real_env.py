"""Real environment for Flexiv Rizon4 robot.

This module wraps the lerobot FlexivRizon4 implementation for use with OpenPI.
"""

import collections
import time

import dm_env
from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import ControlMode
from lerobot.robots.flexiv_rizon4.config_flexiv_rizon4 import FlexivRizon4Config
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.robot_utils import get_logger
import numpy as np

logger = get_logger("FlexivRizon4RealEnv")

# Constants for Flexiv Rizon4
JOINT_DOF = 7  # 7-DOF robot
CARTESIAN_STATE_DIM = 10  # x, y, z, r6d[6], gripper_pos


class FlexivRizon4RealEnv:
    """Environment for real Flexiv Rizon4 robot manipulation.

    Based on lerobot FlexivRizon4 implementation.

    Action/State space:
        JOINT_IMPEDANCE mode (8 dimensions):
            [joint_1.pos, ..., joint_7.pos, gripper_pos]

        CARTESIAN_MOTION_FORCE mode (10 dimensions with 6D rotation, gripper_first):
            [gripper.pos, tcp.x, tcp.y, tcp.z, tcp.r1, tcp.r2, tcp.r3, tcp.r4, tcp.r5, tcp.r6]

            Where tcp.r1-tcp.r6 is the 6D rotation representation (first two columns of rotation matrix):
            - [tcp.r1, tcp.r2, tcp.r3]: First column of rotation matrix
            - [tcp.r4, tcp.r5, tcp.r6]: Second column of rotation matrix

    Observation space:
        {"qpos": state array (8D or 10D depending on control mode),
         "images": {"wrist_cam": (H,W,3), "left_tactile": (H,W,3), "right_tactile": (H,W,3), ...}}
    """

    def __init__(
        self,
        robot_sn: str = "Rizon4-063423",
        control_mode: str = "joint_impedance_control",
        *,
        use_gripper: bool = True,
        use_force: bool = False,
        go_to_start: bool = True,
        log_level: str = "INFO",
        setup_robot: bool = True,
        # Flare gripper settings
        flare_gripper_mac_addr: str = "e2b26adbb104",
        flare_gripper_cam_size: tuple[int, int] = (640, 480),
        flare_gripper_rectify_size: tuple[int, int] = (400, 700),
        flare_gripper_max_pos: float = 85.0,
        # External cameras (scene cameras)
        cameras: dict | None = None,
    ):
        # Convert control_mode string to enum
        control_mode_enum = ControlMode(control_mode)

        # Create FlexivRizon4 config
        self.config = FlexivRizon4Config(
            robot_sn=robot_sn,
            control_mode=control_mode_enum,
            use_gripper=use_gripper,
            use_force=use_force,
            go_to_start=go_to_start,
            log_level=log_level,
            flare_gripper_mac_addr=flare_gripper_mac_addr,
            flare_gripper_cam_size=flare_gripper_cam_size,
            flare_gripper_rectify_size=flare_gripper_rectify_size,
            flare_gripper_max_pos=flare_gripper_max_pos,
            cameras=cameras or {},
        )

        # Create robot instance
        self.robot = make_robot_from_config(self.config)

        if setup_robot:
            self.setup_robot()

    def setup_robot(self):
        """Connect and initialize robot."""
        logger.info("Connecting to Flexiv Rizon4 robot...")
        try:
            self.robot.connect(calibrate=False, go_to_start=self.config.go_to_start)
            logger.info("Flexiv Rizon4 robot connected and ready for inference")
        except Exception as e:
            logger.error(f"Failed to connect Flexiv Rizon4 robot: {e}")
            raise

    def get_qpos(self, obs: dict) -> np.ndarray:
        """Get state from observation.

        Returns:
            For JOINT_IMPEDANCE mode: [joint_1.pos, ..., joint_7.pos, gripper.pos] (8D)
            For CARTESIAN_MOTION_FORCE mode (gripper_first): [gripper.pos, tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6] (10D)
        """
        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint positions (7D) + gripper (1D)
            joints = [obs[f"joint_{i}.pos"] for i in range(1, JOINT_DOF + 1)]
            gripper = [obs["gripper.pos"]]
            return np.array(joints + gripper, dtype=np.float32)

        if self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # Note: Model expects gripper_first format: [gripper, x, y, z, r1-r6]
            # Gripper (1D)
            gripper = [obs["gripper.pos"]]

            # Position (3D)
            position = [obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]]

            # 6D rotation representation (6D)
            rotation = [obs[f"tcp.r{i}"] for i in range(1, 7)]

            return np.array(gripper + position + rotation, dtype=np.float32)

        raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

    def get_images(self, obs: dict) -> dict:
        """Get camera images from observation.

        Returns dictionary with camera images:
        - wrist_cam: from Flare gripper
        - left_tactile, right_tactile: tactile sensors from Flare gripper
        - Additional external cameras if configured
        """
        images = {}

        # Camera names from Flare gripper and external cameras
        camera_names = ["wrist_cam", "left_tactile", "right_tactile"]
        # Add external camera names from config
        camera_names.extend(self.config.cameras.keys())

        for cam_name in camera_names:
            if cam_name in obs:
                images[cam_name] = obs[cam_name]
            else:
                logger.debug(f"Camera {cam_name} not found in observation")

        return images

    def get_observation(self) -> dict:
        """Get complete observation compatible with OpenPI format."""
        current_obs = self.robot.get_observation()
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(current_obs)
        obs["images"] = self.get_images(current_obs)
        return obs

    def get_reward(self) -> float:
        return 0.0

    def reset(self, *, fake: bool = False) -> dm_env.TimeStep:
        """Reset robot to initial position."""
        if not fake:
            logger.info("Resetting Flexiv Rizon4 to start position...")
            try:
                self.robot.reset_to_initial_position()
                logger.info("Flexiv Rizon4 reset completed")
            except Exception as e:
                logger.error(f"Failed to reset Flexiv Rizon4: {e}")
                raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        """Execute action on the robot.

        Args:
            action: For JOINT_IMPEDANCE mode: [joint_1.pos, ..., joint_7.pos, gripper.pos] (8D)
                    For CARTESIAN_MOTION_FORCE mode (gripper_first): [gripper.pos, tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6] (10D)
        """
        # Convert action array to dictionary format expected by FlexivRizon4
        action_dict = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            # Joint positions (7D) + gripper (1D)
            for i in range(JOINT_DOF):
                action_dict[f"joint_{i + 1}.pos"] = float(action[i])
            action_dict["gripper.pos"] = float(action[JOINT_DOF])

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            # Note: Model outputs gripper_first format: [gripper, x, y, z, r1-r6]
            # Gripper (1D) - at index 0
            action_dict["gripper.pos"] = float(action[0])

            # Position (3D) - at index 1, 2, 3
            action_dict["tcp.x"] = float(action[1])
            action_dict["tcp.y"] = float(action[2])
            action_dict["tcp.z"] = float(action[3])

            # 6D rotation representation (6D) - at index 4-9
            for i in range(1, 7):
                action_dict[f"tcp.r{i}"] = float(action[3 + i])

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Send action to robot
        try:
            self.robot.send_action(action_dict)
        except Exception as e:
            logger.error(f"Failed to send action to Flexiv Rizon4: {e}")
            raise

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def disconnect(self):
        """Disconnect robot connection."""
        if self.robot.is_connected:
            logger.info("Disconnecting Flexiv Rizon4 robot...")
            try:
                self.robot.disconnect()
                time.sleep(1)
                logger.info("Flexiv Rizon4 robot disconnected")
            except Exception as e:
                logger.warn(f"Error during Flexiv Rizon4 disconnect: {e}")


def make_flexiv_rizon4_real_env(
    robot_sn: str = "Rizon4-063423",
    control_mode: str = "joint_impedance_control",
    *,
    use_gripper: bool = True,
    use_force: bool = False,
    go_to_start: bool = True,
    log_level: str = "INFO",
    setup_robot: bool = True,
    flare_gripper_mac_addr: str = "e2b26adbb104",
    flare_gripper_cam_size: tuple[int, int] = (640, 480),
    flare_gripper_rectify_size: tuple[int, int] = (400, 700),
    flare_gripper_max_pos: float = 85.0,
    cameras: dict | None = None,
) -> FlexivRizon4RealEnv:
    """Create Flexiv Rizon4 real environment."""
    return FlexivRizon4RealEnv(
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
