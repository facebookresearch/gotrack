# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from functools import partial
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from utils import data_util, logging, net_util, template_util, gen_repre_util
import warnings
from torch.utils.data import DataLoader
from model import pipeline

warnings.filterwarnings("ignore")
logger = logging.get_logger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="inference_pose_estimation"
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
    trainer = instantiate(cfg_trainer)
    logger.info("Trainer initialized!")

    # Set up the dataset.
    cfg.data.dataloader.dataset_name = cfg.dataset_name

    # Check whether the task is localization or detection. If localization, the target objects will be loaded.
    is_localization_task = cfg.mode == "localization"
    cfg.data.dataloader.is_localization_task = is_localization_task
    cfg.data.dataloader.use_default_detections = cfg.model.use_default_detections
    test_dataset = instantiate(cfg.data.dataloader)

    collate_fn = partial(
        data_util.convert_list_scene_observations_to_pipeline_inputs,
        is_localization_task=is_localization_task,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=cfg.machine.num_workers,
        collate_fn=collate_fn,
    )

    result_dir = Path(cfg.save_dir) / f"{cfg.mode}" / f"{cfg.dataset_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {result_dir}")

    # Stage 0: Object onboarding (template generation, DINOv2 feature extraction).
    cfg.model.onboarding.gen_templates_opts.dataset_name = cfg.dataset_name
    cfg.model.onboarding.gen_templates_opts.object_lids = [
        obj_id for obj_id in test_dataset.models_info
    ]
    # If templates already exist, this function will skip the generation.
    template_util.batch_synthesize_templates(
        bop_root_dir=test_dataset.root_dir,
        opts=cfg.model.onboarding.gen_templates_opts,
        num_workers=1,
    )
    # Generate object representation.
    cfg.model.onboarding.gen_repre_opts.dataset_name = cfg.dataset_name
    cfg.model.onboarding.gen_repre_opts.object_lids = [
        obj_id for obj_id in test_dataset.models_info
    ]
    # If object representation already exists, this function will skip the generation.
    repre_dir = gen_repre_util.generate_repre_from_list(
        bop_root_dir=test_dataset.root_dir, opts=cfg.model.onboarding.gen_repre_opts
    )
    logger.info("Object onboarding done!")

    # Unifying the crop size and relative padding for all models.
    for model_opts in [
        cfg.model.cnos.opts,
        cfg.model.foundpose.opts,
        cfg.model.gotrack.opts,
    ]:
        model_opts.crop_size = cfg.model.crop_size
        model_opts.crop_rel_pad = cfg.model.crop_rel_pad
        model_opts.debug = cfg.debug

    # Instantiate each stage of the pipeline.
    detector = instantiate(cfg.model.cnos)
    detector.result_dir = Path(result_dir)

    cfg.model.foundpose.opts.run_retrieval_only = cfg.model.fast_pose_estimation
    coarse_pose_model = instantiate(cfg.model.foundpose)
    coarse_pose_model.result_dir = Path(result_dir)
    coarse_pose_model.set_renderer(test_dataset)

    cfg.model.gotrack.opts.num_iterations_test = cfg.model.num_iterations_refinement
    cfg.model.gotrack.opts.process_inputs = not cfg.model.fast_pose_estimation
    gotrack_model = instantiate(cfg.model.gotrack)
    gotrack_model.set_renderer(test_dataset)
    gotrack_model.result_dir = Path(result_dir)
    gotrack_model.result_file_name = "CNOS-FoundPose-GoTrack"

    # Load the model checkpoint
    if cfg.model.gotrack.checkpoint_path:
        net_util.load_checkpoint(
            model=gotrack_model,
            checkpoint_path=cfg.model.gotrack.checkpoint_path,
            checkpoint_key="model_state_dict",
            prefix="models.1.",
        )
    else:
        raise ValueError("Checkpoint path is required for inference")

    # Define the pipeline.
    pose_pipeline = pipeline.PoseEstimationPipeline(
        detection_model=detector,
        coarse_pose_model=coarse_pose_model,
        refiner_model=gotrack_model,
        result_dir=result_dir,
        result_file_name=f"CNOS-FoundPose-GoTrack-{cfg.mode}",
        detection2d_score_threshold=cfg.model.detection2d_score_threshold,
        use_default_detections=cfg.model.use_default_detections,
    )
    pose_pipeline.set_renderer(test_dataset)  # for visualization of whole scene.
    pose_pipeline.onboarding(repre_dir)
    pose_pipeline.post_onboarding_processing()
    logger.info("Pipeline initialized!")

    trainer.test(
        pose_pipeline,
        dataloaders=test_dataloader,
    )
    logger.info("Done!")


if __name__ == "__main__":
    run_inference()
