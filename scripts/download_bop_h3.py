# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
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
        "handal",
        "hope",
        "hot3d",
    ]:
        # Select the required files based on the dataset name
        if dataset_name in ["hope", "handal"]:
            required_files = ["base", "models", "test_bop24"]
            required_folders = None
        else:
            required_files = ["base", "models"]
            required_folders = ["test_aria_bop_format", "test_quest3_bop_format"]

        logger.info(f"Downloading {dataset_name}")
        # Download the required files
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
        # Download the required folders
        if required_folders is not None:
            # If the dataset is "hot3d", creating a symbolic link folder "test" that contains all files in the required folders.
            if dataset_name == "hot3d":
                test_folder = f"{root_dir}/{dataset_name}/test"
                os.makedirs(test_folder, exist_ok=True)

            for folder_name in required_folders:
                local_folder_path = f"{root_dir}/{dataset_name}/{folder_name}"
                download_cmd = f"huggingface-cli download bop-benchmark/{dataset_name} --include {folder_name}/*.tar --local-dir {root_dir}/{dataset_name} --repo-type=dataset"
                os.system(download_cmd)
                logger.info(f"Downloaded {folder_name}")

                # Unzip the downloaded files
                files = Path(local_folder_path).glob("*.tar")
                files = sorted(files, key=lambda x: x.name)
                for file in tqdm(files, desc=f"Unzipping {folder_name}"):
                    file_name = file.name.split("-")[-1].split(".")[0]
                    sub_folder = f"{local_folder_path}/{file_name}"
                    os.makedirs(sub_folder, exist_ok=True)

                    unzip_cmd = f"unzip -o {file} -d {local_folder_path}"
                    os.system(unzip_cmd)

                    if dataset_name == "hot3d":
                        sym_link_folder = f"{test_folder}/{file_name}"
                        os.symlink(sub_folder, f"{sym_link_folder}")
                        logger.info(f"Created symlink: {sym_link_folder}")

        # For hope, we rename the entire folder to "hopev2" to use correct dataset parameters from bop_toolkit.
        if dataset_name == "hope":
            os.rename(
                f"{root_dir}/{dataset_name}",
                f"{root_dir}/hopev2",
            )
            logger.info(f"Renamed {dataset_name} to hopev2")


if __name__ == "__main__":
    download()
