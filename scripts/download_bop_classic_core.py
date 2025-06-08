# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from utils.logging import get_logger
from huggingface_hub import hf_hub_download

logger = get_logger(__name__)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_gotrack",
)
def download(cfg: DictConfig) -> None:
    root_dir = Path(cfg.machine.root_dir) / "bop_datasets"
    os.makedirs(root_dir, exist_ok=True)

    for dataset_name in [
        "lmo",
        "tless",
        "tudl",
        "icbin",
        "itodd",  # gt not available
        "hb",  # gt not available
        "ycbv",
    ]:
        # Select the required files based on the dataset name
        if dataset_name in ["tless", "hb"]:
            required_files = ["base", "models", "test_primesense_bop19"]
        else:
            required_files = ["base", "models", "test_bop19"]
        logger.info(f"Downloading {dataset_name}")
        for file_name in required_files:
            file_path = hf_hub_download(
                repo_id=f"bop-benchmark/{dataset_name}",
                filename=f"{dataset_name}_{file_name}.zip",
                repo_type="dataset",
                local_dir=f"{root_dir}/{dataset_name}",
            )
            logger.info(f"Downloaded to: {file_path}")

            # Unzip the downloaded files
            if file_name == "base":
                os.system(
                    f"unzip -j {root_dir}/{dataset_name}/{dataset_name}_{file_name}.zip -d {root_dir}/{dataset_name}"
                )
            else:
                os.system(
                    f"unzip -o {root_dir}/{dataset_name}/{dataset_name}_{file_name}.zip -d {root_dir}/{dataset_name}"
                )


if __name__ == "__main__":
    download()
