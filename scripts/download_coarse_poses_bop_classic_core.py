# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
from utils.logging import get_logger

logger = get_logger(__name__)

foundpose_coarse_poses = {
    "itodd": "foundpose_itodd-test_6bb615c4-51f0-4b99-ac6a-65eabaeaedc9.csv",
    "hb": "foundpose_hb-test_74c59599-186e-46da-8f23-cc0c86a1e9ab.csv",
    "icbin": "foundpose_icbin-test_b3bbb508-19fd-437a-b7b1-1451b6522860.csv",
    "lmo": "foundpose_lmo-test_8788d8c0-2552-4a99-9cff-da3009ffedec.csv",
    "tless": "foundpose_tless-test_498a896a-f39d-4a18-998d-646c62c44c49.csv",
    "ycbv": "foundpose_ycbv-test_d6100df8-7508-4235-a04b-1e41b3df61bd.csv",
    "tudl": "foundpose_tudl-test_037a2901-4de7-4fc8-8dee-6642a0ad827c.csv",
}

gigapose_coarse_poses = {
    "itodd": "gigapose_itodd-test_ba91bea9-8498-43b0-a1ae-9a196fc9217d.csv",
    "hb": "gigapose_hb-test_43281438-85e7-49e8-a660-e7300b5453d3.csv",
    "icbin": "gigapose_icbin-test_a337e2d8-1e71-423b-bdc1-306b6be4d7c5.csv",
    "lmo": "gigapose_lmo-test_5f28db49-675a-43e5-a9a3-9037d571d816.csv",
    "tless": "gigapose_tless-test_ffb21e8c-a5ca-454c-83c1-145e824f0ffb.csv",
    "ycbv": "gigapose_ycbv-test_730adcb8-1bb4-4b07-a926-ebab69644731.csv",
    "tudl": "gigapose_tudl-test_07c7ad01-f0e6-4bf9-8760-845c688617d4.csv",
}


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="inference_gotrack",
)
def download(cfg: DictConfig) -> None:
    source_url = "https://bop.felk.cvut.cz/media/subs"
    root_dir = Path(cfg.machine.root_dir) / "bop_datasets"
    coarse_pose_dir = root_dir / "coarse_poses"
    os.makedirs(coarse_pose_dir, exist_ok=True)

    # Download the coarse poses.
    for method_name in ["foundpose", "gigapose"]:
        method_coarse_pose_dir = coarse_pose_dir / method_name
        os.makedirs(method_coarse_pose_dir, exist_ok=True)
        for dataset_name in [
            "lmo",
            "tless",
            "tudl",
            "icbin",
            "itodd",  # gt not available
            "hb",  # gt not available
            "ycbv",
        ]:
            if method_name == "foundpose":
                file_name = foundpose_coarse_poses[dataset_name]
            elif method_name == "gigapose":
                file_name = gigapose_coarse_poses[dataset_name]
            else:
                raise ValueError(f"Unknown method name: {method_name}")

            download_cmd = (
                f"wget {source_url}/{file_name} -O {method_coarse_pose_dir}/{file_name}"
            )
            os.system(download_cmd)
            logger.info(f"Downloaded to: {method_coarse_pose_dir}/{file_name}")


if __name__ == "__main__":
    download()
