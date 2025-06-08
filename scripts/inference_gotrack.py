# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from utils import data_util, net_util  # noqa: F401
from utils.logging import get_logger
import warnings

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="inference_gotrack"
)
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info("Initializing logger, callbacks and trainer")
    cfg_trainer = cfg.machine.trainer

    if "TensorBoardLogger" in cfg_trainer.logger._target_:
        tensorboard_dir = f"{cfg.save_dir}/{cfg_trainer.logger.name}"
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger.info(f"Tensorboard logger initialized at {tensorboard_dir}")
    else:
        raise NotImplementedError("Only Tensorboard loggers are supported")
    os.makedirs(cfg.save_dir, exist_ok=True)

    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    model = instantiate(cfg.model)
    # Load the model checkpoint
    if cfg.model.checkpoint_path:
        net_util.load_checkpoint(
            model=model,
            checkpoint_path=cfg.model.checkpoint_path,
            checkpoint_key="model_state_dict",
            prefix="models.1.",
        )
    else:
        raise ValueError("Checkpoint path is required for inference")

    if cfg.mode == "pose_refinement":
        # Define the dataloader using dataset_name, coarse_pose_method
        cfg.data.dataloader.dataset_name = cfg.dataset_name
        cfg.data.dataloader.coarse_pose_method = cfg.coarse_pose_method
        test_dataset = instantiate(cfg.data.dataloader)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            collate_fn=data_util.convert_list_scene_observations_to_gotrack_inputs,
        )
        result_dir = (
            Path(cfg.save_dir)
            / f"{cfg.mode}"
            / f"{cfg.dataset_name}_{cfg.coarse_pose_method}"
        )
        result_file_name = test_dataset.coarse_poses_file_name.replace(
            f"{cfg.coarse_pose_method}", f"{cfg.coarse_pose_method}AndGotrack"
        )
    else:
        cfg.data.rbot.dataloader.object_name = cfg.object_name
        cfg.data.rbot.dataloader.sequence_name = cfg.sequence_name
        test_dataset = instantiate(cfg.data.rbot.dataloader)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            collate_fn=data_util.collate_fn,
        )
        result_dir = (
            Path(cfg.save_dir)
            / f"{cfg.mode}"
            / f"{cfg.object_name}_{cfg.sequence_name}"
        )
        result_file_name = f"gotrack-{cfg.object_name}-{cfg.sequence_name}.json"
    logger.info("Dataloaders initialized!")

    model.set_renderer(test_dataset)
    model.result_dir = Path(result_dir)
    model.result_file_name = result_file_name
    logger.info("Model initialized!")
    trainer.test(
        model,
        dataloaders=test_dataloader,
    )

    logger.info("Done!")


if __name__ == "__main__":
    run_inference()
