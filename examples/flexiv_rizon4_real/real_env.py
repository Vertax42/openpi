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
JOINT_STATE_DIM = 22  # joint pos (7D) + vel (7D) + effort (7D) + gripper (1D)


class FlexivRizon4RealEnv:
    """Environment for real Flexiv Rizon4 robot manipulation.

    Based on lerobot FlexivRizon4 implementation.

    Action/State space:
        JOINT_IMPEDANCE mode (8 dimensions):
            [joint_1.pos, ..., joint_7.pos, gripper_pos]

        CARTESIAN_MOTION_FORCE mode, use_joint_observation=False (10 dimensions):
            [tcp.x, tcp.y, tcp.z, tcp.r1, tcp.r2, tcp.r3, tcp.r4, tcp.r5, tcp.r6, gripper.pos]

        CARTESIAN_MOTION_FORCE mode, use_joint_observation=True (22 dimensions):
            [joint_1.pos..7, joint_1.vel..7, joint_1.effort..7, gripper.pos]

    Observation space:
        {"qpos": state array, "images": {"wrist_cam": (H,W,3), "top": (H,W,3), ...}}
    """

    def __init__(
        self,
        robot_sn: str = "Rizon4-062855",
        control_mode: str = "cartesian_motion_force_control",
        use_gripper: bool = True,
        gripper_type: str = "flare_gripper",
        use_force: bool = False,
        use_joint_observation: bool = False,
        go_to_start: bool = True,
        log_level: str = "INFO",
        setup_robot: bool = True,
        # Gripper settings
        gripper_mac_addr: str = "bef1504b5391",
        gripper_cam_size: tuple[int, int] = (640, 480),
        gripper_rectify_size: tuple[int, int] = (400, 700),
        gripper_max_pos: float = 85.0,
    ):
        control_mode_enum = ControlMode(control_mode)

        self.config = FlexivRizon4Config(
            robot_sn=robot_sn,
            control_mode=control_mode_enum,
            use_gripper=use_gripper,
            gripper_type=gripper_type,
            use_force=use_force,
            use_joint_observation=use_joint_observation,
            go_to_start=go_to_start,
            log_level=log_level,
            gripper_mac_addr=gripper_mac_addr,
            gripper_cam_size=gripper_cam_size,
            gripper_rectify_size=gripper_rectify_size,
            gripper_max_pos=gripper_max_pos,
        )

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
            JOINT_IMPEDANCE:
                [joint_1.pos, ..., joint_7.pos, gripper.pos] (8D)
            CARTESIAN_MOTION_FORCE + use_joint_observation=False:
                [tcp.x, tcp.y, tcp.z, tcp.r1, ..., tcp.r6, gripper.pos] (10D)
            CARTESIAN_MOTION_FORCE + use_joint_observation=True:
                [joint_1.pos..7, joint_1.vel..7, joint_1.effort..7, gripper.pos] (22D)
        """
        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            joints = [obs[f"joint_{i}.pos"] for i in range(1, JOINT_DOF + 1)]
            gripper = [obs["gripper.pos"]]
            return np.array(joints + gripper, dtype=np.float32)

        if self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            if self.config.use_joint_observation:
                # Joint pos + vel + effort (21D) + gripper (1D)
                pos = [obs[f"joint_{i}.pos"] for i in range(1, JOINT_DOF + 1)]
                vel = [obs[f"joint_{i}.vel"] for i in range(1, JOINT_DOF + 1)]
                effort = [obs[f"joint_{i}.effort"] for i in range(1, JOINT_DOF + 1)]
                gripper = [obs["gripper.pos"]]
                return np.array(pos + vel + effort + gripper, dtype=np.float32)
            else:
                # TCP pose (9D) + gripper (1D)
                position = [obs["tcp.x"], obs["tcp.y"], obs["tcp.z"]]
                rotation = [obs[f"tcp.r{i + 1}"] for i in range(6)]
                gripper = [obs["gripper.pos"]]
                return np.array(position + rotation + gripper, dtype=np.float32)

        raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

    def get_images(self, obs: dict) -> dict:
        """Get camera images from observation.

        Returns dictionary with all available camera images:
        - wrist_cam: from gripper wrist camera
        - left_tactile, right_tactile: tactile sensors
        - top: external RealSense scene camera (if configured)
        """
        images = {}
        # Collect all possible image keys (gripper + external cameras)
        camera_names = ["wrist_cam", "left_tactile", "right_tactile"]
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
            action:
                JOINT_IMPEDANCE: [joint_1.pos, ..., joint_7.pos, gripper.pos] (8D)
                CARTESIAN_MOTION_FORCE: [tcp.x, tcp.y, tcp.z, tcp.r1-r6, gripper.pos] (10D)
        """
        action_dict = {}

        if self.config.control_mode == ControlMode.JOINT_IMPEDANCE:
            for i in range(JOINT_DOF):
                action_dict[f"joint_{i + 1}.pos"] = float(action[i])
            action_dict["gripper.pos"] = float(np.clip(action[JOINT_DOF], 0.0, 1.0))

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION_FORCE:
            action_dict["tcp.x"] = float(action[0])
            action_dict["tcp.y"] = float(action[1])
            action_dict["tcp.z"] = float(action[2])
            for i in range(6):
                action_dict[f"tcp.r{i + 1}"] = float(action[3 + i])
            action_dict["gripper.pos"] = float(np.clip(action[9], 0.0, 1.0))

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

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
                logger.warning(f"Error during Flexiv Rizon4 disconnect: {e}")


def make_flexiv_rizon4_real_env(
    robot_sn: str = "Rizon4-062855",
    control_mode: str = "cartesian_motion_force_control",
    use_gripper: bool = True,
    gripper_type: str = "xense_gripper",
    use_force: bool = False,
    use_joint_observation: bool = False,
    go_to_start: bool = True,
    log_level: str = "INFO",
    setup_robot: bool = True,
    gripper_mac_addr: str = "bef1504b5391",
    gripper_cam_size: tuple[int, int] = (640, 480),
    gripper_rectify_size: tuple[int, int] = (400, 700),
    gripper_max_pos: float = 85.0,
) -> FlexivRizon4RealEnv:
    """Create Flexiv Rizon4 real environment."""
    return FlexivRizon4RealEnv(
        robot_sn=robot_sn,
        control_mode=control_mode,
        use_gripper=use_gripper,
        gripper_type=gripper_type,
        use_force=use_force,
        use_joint_observation=use_joint_observation,
        go_to_start=go_to_start,
        log_level=log_level,
        setup_robot=setup_robot,
        gripper_mac_addr=gripper_mac_addr,
        gripper_cam_size=gripper_cam_size,
        gripper_rectify_size=gripper_rectify_size,
        gripper_max_pos=gripper_max_pos,
    )
