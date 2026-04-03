"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _image_keys_from_repack(repack_inputs: list) -> list[str]:
    """Extract expected image keys from RepackTransform structures."""
    keys: list[str] = []
    for t in repack_inputs:
        if isinstance(t, transforms.RepackTransform):
            structure = t.structure
            if isinstance(structure, dict) and "images" in structure:
                images_struct = structure["images"]
                if isinstance(images_struct, dict):
                    keys.extend(images_struct.keys())
    return keys


class FillDummyImages(transforms.DataTransformFn):
    """Inject zero-filled placeholder images for missing camera keys.

    Used with skip_images=True so that policy Input transforms (which assume
    images are present) can run without crashing.  The dummy image values
    are never used for norm-stat computation.
    Images are (3, 224, 224) uint8 zeros to match the LeRobot [C, H, W] format.
    """

    def __init__(self, expected_keys: list[str]) -> None:
        self._expected_keys = expected_keys

    def __call__(self, data: dict) -> dict:
        if not self._expected_keys:
            return data
        dummy = np.zeros((3, 224, 224), dtype=np.uint8)
        images = data.get("images", {})
        for k in self._expected_keys:
            if k not in images:
                images[k] = dummy
        data["images"] = images
        return data


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    skip_images: bool = True,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config, skip_images=skip_images)

    # When skip_images=True, video keys are absent from each sample.
    # Replace every RepackTransform with a lenient (strict=False) copy so that
    # missing image source-keys are silently dropped instead of raising KeyError.
    def _maybe_lenient(t):
        if skip_images and isinstance(t, transforms.RepackTransform) and t.strict:
            return transforms.RepackTransform(t.structure, strict=False)
        return t

    # When skip_images=True:
    #   1. repack_transforms run with strict=False (missing image source-keys skipped)
    #   2. FillDummyImages injects zero placeholders for the expected camera keys so
    #      that policy Input transforms (e.g. BiFlexivInputs) don't crash on KeyError
    #   3. ALL data_transforms run as normal – this includes DeltaActions which MUST
    #      execute to produce correct (delta) action values for norm stats
    lenient_repacks = [_maybe_lenient(t) for t in data_config.repack_transforms.inputs]

    if skip_images:
        expected_keys = _image_keys_from_repack(data_config.repack_transforms.inputs)
        fill_dummy = [FillDummyImages(expected_keys)]
    else:
        fill_dummy = []

    transform_list = [
        *lenient_repacks,
        *fill_dummy,
        *data_config.data_transforms.inputs,
        RemoveStrings(),
    ]

    dataset = _data_loader.TransformedDataset(dataset, transform_list)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None, skip_images: bool = True):
    """Compute normalization statistics for a config.

    Args:
        config_name: Name of the training config to use.
        max_frames: Optional maximum number of frames to use for computing stats.
        skip_images: Skip video/image decoding (default True). Only state and
            actions are needed for norm stats, so this is much faster. Set to
            False if your repack/data transforms require image fields.
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            skip_images=skip_images,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
