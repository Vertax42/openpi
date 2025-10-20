"""Aloha policy with tactile image support.

This extends the base Aloha policy to handle tactile images in addition to
the standard camera images.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import einops

import openpi.transforms as transforms
from openpi.policies import aloha_policy


@dataclasses.dataclass(frozen=True)
class AlohaTactileInputs(aloha_policy.AlohaInputs):
    """Inputs for the Aloha policy with tactile support.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]
      - Visual cameras: cam_high, cam_low, cam_left_wrist, cam_right_wrist
      - Tactile sensors: cam_left_tactile, cam_right_tactile
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The expected camera names (visual + tactile)
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        # Visual cameras (standard Aloha)
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
        # Tactile sensors (new)
        "cam_left_tactile",
        "cam_right_tactile",
    )

    def __call__(self, data: dict) -> dict:
        data = aloha_policy._decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected images to contain subset of {self.EXPECTED_CAMERAS}, got {tuple(in_images)}"
            )

        # Base image must exist
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add visual wrist cameras
        visual_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in visual_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # Add tactile sensors
        tactile_image_names = {
            "left_tactile_0_rgb": "cam_left_tactile",
            "right_tactile_0_rgb": "cam_right_tactile",
        }
        for dest, source in tactile_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                # Use black image as placeholder for missing tactile sensors
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # Actions are only available during training
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = aloha_policy._encode_actions_inv(
                actions, adapt_to_pi=self.adapt_to_pi
            )
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


def make_aloha_tactile_example() -> dict:
    """Create an example input for testing."""
    return {
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_tactile": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_tactile": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "state": np.zeros(14, dtype=np.float32),
        "actions": np.zeros((50, 14), dtype=np.float32),
        "prompt": "pick cube",
    }
