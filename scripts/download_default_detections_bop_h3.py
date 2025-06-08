# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

# to use: python -m scripts.download_default_detections_bop_h3
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from utils.logging import get_logger
from huggingface_hub import hf_hub_download, list_repo_files

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_gotrack",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir) / "bop_datasets"
    os.makedirs(root_dir, exist_ok=True)

    # List all files in the repo
    files = list_repo_files("bop-benchmark/bop_extra", repo_type="dataset")

    # Keep only the files that are default detections for model-based tasks on BOP-Classic-Core
    files = [
        f
        for f in files
        if "default_detections/h3_bop24_model_based_unseen/cnos-fastsam-with-mask" in f
    ]

    # Download the default detections.
    for file in files:
        file_path = hf_hub_download(
            repo_id="bop-benchmark/bop_extra",
            filename=f"{file}",
            repo_type="dataset",
            local_dir=f"{root_dir}",
        )
        logger.info(f"Downloaded to: {file_path}")


if __name__ == "__main__":
    download()
