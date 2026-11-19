"""Compute normalization statistics for a configured local LeRobot dataset.

The implementation uses this repository's OpenPI data-loader and normalization APIs.
"""
from __future__ import annotations

import numpy as np
import tqdm
import tyro

import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class RemoveStrings(_transforms.DataTransformFn):
    """Remove text fields that are not needed for state/action statistics."""

    def __call__(self, sample: dict) -> dict:
        return {key: value for key, value in sample.items() if not np.issubdtype(np.asarray(value).dtype, np.str_)}


def compute_for_config(config: _config.TrainConfig, max_frames: int | None = None) -> None:
    """Compute state/action statistics and save them under this config's asset directory."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("The training data config must define repo_id.")
    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("This helper currently supports the project's local LeRobot datasets, not RLDS.")

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )

    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False
    if num_batches < 1:
        raise ValueError("The dataset does not contain a complete batch for normalization.")

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
        seed=config.seed,
    )

    keys = ("state", "actions")
    stats = {key: _normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing normalization statistics"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing normalization statistics to: {output_path}")
    _normalize.save(output_path, norm_stats)


def main(config_name: str, max_frames: int | None = None) -> None:
    compute_for_config(_config.get_config(config_name), max_frames)


if __name__ == "__main__":
    tyro.cli(main)
