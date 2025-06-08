# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from pathlib import Path
from typing import Any, Dict, Optional

from model import base, cnos, foundpose, gotrack
from utils import data_util, structs, logging
from bop_toolkit_lib import inout
from PIL import Image

logger = logging.get_logger(__name__)


class PoseEstimationPipeline(base.ModelBase):
    """Three-stage pose estimation pipeline: 2D detection, coarse pose estimation, and refinement."""

    def __init__(
        self,
        detection_model: cnos.CNOS,
        coarse_pose_model: Optional[foundpose.FoundPose],
        refiner_model: Optional[gotrack.GoTrack],
        result_dir: Optional[Path] = None,
        result_file_name: Optional[str] = None,
        detection2d_score_threshold: Optional[float] = 0.0,
        use_default_detections: Optional[bool] = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(base.ModelBase, self).__init__()
        self.detection_model = detection_model
        self.coarse_pose_model = coarse_pose_model
        self.refiner_model = refiner_model
        self.detection2d_score_threshold = detection2d_score_threshold
        self.use_default_detections = use_default_detections
        assert (
            self.detection_model.opts.crop_rel_pad
            == self.coarse_pose_model.opts.crop_rel_pad
            == self.refiner_model.opts.crop_rel_pad
        ), "Crop relative padding should be the same for all models."
        assert (
            self.detection_model.opts.crop_size
            == self.coarse_pose_model.opts.crop_size
            == self.refiner_model.opts.crop_size
        ), "Crop size should be the same for all models."
        self.result_dir = result_dir
        self.result_file_name = result_file_name

    def post_onboarding_processing(self) -> None:
        """Move the model to the specified device."""
        if not self.use_default_detections:
            self.detection_model.objects_repre = self.objects_repre
            self.detection_model.post_onboarding_processing()
        if self.coarse_pose_model is not None:
            self.coarse_pose_model.objects_repre = self.objects_repre
            self.coarse_pose_model.post_onboarding_processing()
        if self.refiner_model is not None:
            self.refiner_model.objects_repre = self.objects_repre

    def forward_pipeline(
        self,
        scene_observation: structs.SceneObservation,
        target_objects: Optional[Dict[str, int]] = None,
        batch_idx: Optional[int] = None,
    ) -> None:
        """Run the full pipeline."""
        # Stage 1: 2D object detection.
        if self.use_default_detections:
            detection_results = (
                data_util.convert_default_detections_to_foundpose_inputs(
                    scene_observation=scene_observation,
                    crop_rel_pad=self.coarse_pose_model.opts.crop_rel_pad,
                    crop_size=self.coarse_pose_model.opts.crop_size,
                )
            )
        else:
            detection_results = self.detection_model.forward_pipeline(
                scene_observation=scene_observation, batch_idx=batch_idx
            )
        # Localization tasks.
        if target_objects is not None:
            detection_results = self.detection_model.filter_detections_by_targets(
                results=detection_results, target_objects=target_objects
            )
        # Detection tasks.
        else:
            detection_results = self.detection_model.filter_detections_by_score(
                results=detection_results,
                score_threshold=self.detection2d_score_threshold,
            )
        # Stage 2. Coarse pose estimation.
        if self.coarse_pose_model is not None:
            coarse_pose_inputs = detection_results
            coarse_pose_results = self.coarse_pose_model.forward_pipeline(
                inputs=coarse_pose_inputs, batch_idx=batch_idx
            )
        # Stage 3. Refiner (tracking) model.
        if self.refiner_model is not None:
            refiner_inputs = data_util.convert_foundpose_outputs_to_gotrack_inputs(
                coarse_pose_results
            )
            self.refiner_model.forward_pipeline(
                inputs=refiner_inputs,
                batch_idx=batch_idx,
            )

    def get_vis_tiles(self, batch_idx: int) -> None:
        # Merge visualization of pose stages if available.
        vis_foundPose_path = self.result_dir / f"vis_{batch_idx:06d}_foundPose.png"
        vis_goTrack_path = self.result_dir / f"vis_{batch_idx:06d}_goTrack.png"
        vis_all_path = self.result_dir / f"vis_{batch_idx:06d}_all.png"
        if vis_foundPose_path.exists() and vis_goTrack_path.exists():
            vis_foundPose = Image.open(str(vis_foundPose_path))
            vis_goTrack = Image.open(str(vis_goTrack_path))

            # Resize the images to have the same width
            width = min(vis_foundPose.width, vis_goTrack.width)
            vis_foundPose = vis_foundPose.resize(
                (width, int(vis_foundPose.height * width / vis_foundPose.width))
            )
            vis_goTrack = vis_goTrack.resize(
                (width, int(vis_goTrack.height * width / vis_goTrack.width))
            )
            # Create a new image with the combined width
            combined_height = vis_foundPose.height + vis_goTrack.height
            combined_image = Image.new("RGB", (width, combined_height))
            # Paste the images into the new image
            combined_image.paste(vis_foundPose, (0, 0))
            combined_image.paste(vis_goTrack, (0, vis_foundPose.height))
            # Save the combined image
            combined_image.save(str(vis_all_path))
            logger.info(f"Merged visualization of pose stages to {vis_all_path}")

    def test_step(self, inputs: Dict[str, Any], batch_idx: int) -> None:
        """Test step for the pipeline."""
        save_path = self.result_dir / f"per_frame_refined_poses_{batch_idx:06d}.json"
        if save_path.exists():
            logger.info(f"Skipping batch {batch_idx}, already processed.")
            return
        scene_observation = inputs["scene_observation"]
        target_objects = inputs["target_objects"]
        self.forward_pipeline(
            scene_observation=scene_observation,
            target_objects=target_objects,
            batch_idx=batch_idx,
        )
        if self.coarse_pose_model.opts.debug:
            self.get_vis_tiles(batch_idx)

    def on_test_epoch_end(self):
        stage_names = ["2d_detections", "coarse_poses", "refined_poses"]
        results_files = {stage_name: [] for stage_name in stage_names}
        model_is_available = {
            "2d_detections": self.detection_model is not None,
            "coarse_poses": self.coarse_pose_model is not None,
            "refined_poses": self.refiner_model is not None,
        }
        if self.global_rank == 0:  # only rank 0 process
            # Collect all per-frame results.
            for stage_name in stage_names:
                if not model_is_available[stage_name]:
                    logger.info(f"{stage_name} not available.")
                    continue
                per_frame_files = self.result_dir.glob(f"per_frame_{stage_name}_*.json")
                per_frame_files = list(per_frame_files)
                per_frame_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
                results_files[stage_name] = per_frame_files
                logger.info(
                    f"{stage_name}: Found {len(per_frame_files)} for {stage_name}."
                )

            # Assemble all per-frame results and save them to a single file.
            all_estimates = {stage_name: [] for stage_name in stage_names}
            for stage_name in stage_names:
                if not model_is_available[stage_name]:
                    logger.info(f"{stage_name} not available.")
                    continue
                for idx in range(len(results_files[stage_name])):
                    result_file = results_files[stage_name][idx]
                    if stage_name == "2d_detections":
                        per_frame_estimates = inout.load_json(
                            str(result_file), keys_to_int=True
                        )
                    else:
                        per_frame_estimates = inout.load_bop_results(result_file)
                    all_estimates[stage_name].extend(per_frame_estimates)
                save_path = self.result_dir / f"{self.result_file_name}-{stage_name}"
                if stage_name == "2d_detections":
                    inout.save_json(
                        str(save_path) + ".json",
                        all_estimates[stage_name],
                    )
                else:
                    inout.save_bop_results(
                        str(save_path) + ".csv",
                        all_estimates[stage_name],
                    )
                logger.info(f"Saved to {save_path}")
