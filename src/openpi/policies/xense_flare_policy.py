import dataclasses
from typing import ClassVar

import einops
import math
import numpy as np

from openpi import transforms


def make_xense_flare_example() -> dict:
    """Creates a random input example for the xense flare policy."""
    return {
        "state": np.ones((8,)),
        "images": {
            "observation/wrist_image_left": np.random.randint(
                256, size=(3, 224, 224), dtype=np.uint8
            ),
        },
        "prompt": "do something",
    }


def _reorder_gripper_first_to_last(arr: np.ndarray) -> np.ndarray:
    """Reorder array from [gripper, ...] to [..., gripper].

    Input format (gripper_first=True):
        [gripper, tcp_x, tcp_y, tcp_z, qw, qx, qy, qz]
    Output format (model internal, gripper_last):
        [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]

    Works for both state (8,) and actions (horizon, 8).
    """
    if arr.ndim == 1:
        # state: (8,) -> move index 0 to last
        return np.concatenate([arr[1:], arr[:1]])
    else:
        # actions: (horizon, 8) -> move column 0 to last
        return np.concatenate([arr[:, 1:], arr[:, :1]], axis=-1)


def _reorder_gripper_last_to_first(arr: np.ndarray) -> np.ndarray:
    """Reorder array from [..., gripper] to [gripper, ...].

    Input format (model output, gripper_last):
        [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]
    Output format (gripper_first=True):
        [gripper, tcp_x, tcp_y, tcp_z, qw, qx, qy, qz]

    Works for both state (8,) and actions (horizon, 8).
    """
    if arr.ndim == 1:
        # state: (8,) -> move last index to first
        return np.concatenate([arr[-1:], arr[:-1]])
    else:
        # actions: (horizon, 8) -> move last column to first
        return np.concatenate([arr[:, -1:], arr[:, :-1]], axis=-1)


@dataclasses.dataclass(frozen=True)
class XenseFlareInputs(transforms.DataTransformFn):
    """Inputs for the xense flare policy.

    Expected inputs (dataset format, 8 dims):
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: array [8] = [gripper, tcp_x, tcp_y, tcp_z, qw, qx, qy, qz] (if gripper_first=True)
    - actions: array [action_horizon, 8]

    Output format (model format, 7 dims, quaternion -> euler):
    - state: [7] = [tcp_x, tcp_y, tcp_z, roll, pitch, yaw, gripper]
    - actions: [action_horizon, 7]
    """

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("observation/wrist_image_left",)

    # If True, input data has gripper at index 0: [gripper, tcp_x, tcp_y, tcp_z, qw, qx, qy, qz]
    # If False, input data has gripper at last index: [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]
    gripper_first: bool = True

    def __call__(self, data: dict) -> dict:
        # reorder state and parse images to model format
        data = _decode_xense_flare(data, gripper_first=self.gripper_first)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
            )

        # Assume that left_wrist_0_rgb image always exists.
        wrist_image_left = in_images["observation/wrist_image_left"]

        images = {
            "left_wrist_0_rgb": wrist_image_left,
        }
        image_masks = {
            "left_wrist_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "base_0_rgb": "cam_high",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(wrist_image_left)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if self.gripper_first:
                actions = _reorder_gripper_first_to_last(actions)
            # Convert quaternion to euler angles: (horizon, 8) -> (horizon, 7)
            actions = _quat_to_rpy_array(actions)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XenseFlareOutputs(transforms.DataTransformFn):
    """Outputs for the Xense Flare policy.

    Model output format (7 dims, euler angles):
        [tcp_x, tcp_y, tcp_z, roll, pitch, yaw, gripper]

    Output format (8 dims, quaternion):
        [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims (in case model outputs padded actions)
        actions = np.asarray(data["actions"][:, :7])
        # Convert euler angles back to quaternion: (horizon, 7) -> (horizon, 8)
        actions = _rpy_to_quat_array(actions)

        return {"actions": actions}


def _quaternion_to_euler_batch(
    qw: np.ndarray, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized quaternion to euler conversion.

    Args:
        qw, qx, qy, qz: Arrays of shape (n,) or scalars

    Returns:
        roll, pitch, yaw: Arrays of same shape as input
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    # Handle gimbal lock
    pitch = np.where(
        np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(np.clip(sinp, -1, 1))
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_to_rpy_array(arr: np.ndarray) -> np.ndarray:
    """Convert quaternion part of state/actions to roll-pitch-yaw.

    Input format (after gripper reorder, 8 dims):
        [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]
    Output format (7 dims):
        [tcp_x, tcp_y, tcp_z, roll, pitch, yaw, gripper]

    Works for both state (8,) and actions (horizon, 8).
    Uses vectorized operations for efficiency.
    """
    if arr.ndim == 1:
        # state: (8,) -> (7,)
        tcp_xyz = arr[:3]
        qw, qx, qy, qz = arr[3], arr[4], arr[5], arr[6]
        gripper = arr[7:8]
        roll, pitch, yaw = _quaternion_to_euler_batch(qw, qx, qy, qz)
        return np.concatenate([tcp_xyz, np.array([roll, pitch, yaw]), gripper])
    else:
        # actions: (horizon, 8) -> (horizon, 7) - vectorized
        tcp_xyz = arr[:, :3]  # (horizon, 3)
        qw, qx, qy, qz = arr[:, 3], arr[:, 4], arr[:, 5], arr[:, 6]  # each (horizon,)
        gripper = arr[:, 7:8]  # (horizon, 1)
        roll, pitch, yaw = _quaternion_to_euler_batch(qw, qx, qy, qz)  # each (horizon,)
        rpy = np.stack([roll, pitch, yaw], axis=-1)  # (horizon, 3)
        return np.concatenate([tcp_xyz, rpy, gripper], axis=-1)  # (horizon, 7)


def _euler_to_quaternion_batch(
    roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized euler to quaternion conversion.

    Uses ZYX intrinsic rotation order (yaw → pitch → roll).

    Args:
        roll, pitch, yaw: Arrays of shape (n,) or scalars

    Returns:
        qw, qx, qy, qz: Arrays of same shape as input
    """
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz


def _rpy_to_quat_array(arr: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw part of actions to quaternion.

    Input format (7 dims):
        [tcp_x, tcp_y, tcp_z, roll, pitch, yaw, gripper]
    Output format (8 dims):
        [tcp_x, tcp_y, tcp_z, qw, qx, qy, qz, gripper]

    Works for both state (7,) and actions (horizon, 7).
    Uses vectorized operations for efficiency.
    """
    if arr.ndim == 1:
        # state: (7,) -> (8,)
        tcp_xyz = arr[:3]
        roll, pitch, yaw = arr[3], arr[4], arr[5]
        gripper = arr[6:7]
        qw, qx, qy, qz = _euler_to_quaternion_batch(roll, pitch, yaw)
        return np.concatenate([tcp_xyz, np.array([qw, qx, qy, qz]), gripper])
    else:
        # actions: (horizon, 7) -> (horizon, 8) - vectorized
        tcp_xyz = arr[:, :3]  # (horizon, 3)
        roll, pitch, yaw = arr[:, 3], arr[:, 4], arr[:, 5]  # each (horizon,)
        gripper = arr[:, 6:7]  # (horizon, 1)
        qw, qx, qy, qz = _euler_to_quaternion_batch(roll, pitch, yaw)  # each (horizon,)
        quat = np.stack([qw, qx, qy, qz], axis=-1)  # (horizon, 4)
        return np.concatenate([tcp_xyz, quat, gripper], axis=-1)  # (horizon, 8)


def _decode_xense_flare(data: dict, *, gripper_first: bool = True) -> dict:
    """Decode xense flare data format.

    Input format (dataset, gripper_first=True, 8 dims):
        [gripper, tcp_x, tcp_y, tcp_z, qw, qx, qy, qz]
    Output format (model, 7 dims):
        [tcp_x, tcp_y, tcp_z, roll, pitch, yaw, gripper]

    Processing steps:
    1. Reorder: move gripper from first to last (if gripper_first=True)
    2. Convert quaternion (qw, qx, qy, qz) to euler angles (roll, pitch, yaw)
    3. Convert images from [C, H, W] to [H, W, C]

    Args:
        data: Input data dict
        gripper_first: If True, input has gripper at index 0, need to move to last
    """
    state = np.asarray(data["state"])
    if gripper_first:
        state = _reorder_gripper_first_to_last(state)
    # Convert quaternion to euler angles: 8 dims -> 7 dims
    state = _quat_to_rpy_array(state)

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data
