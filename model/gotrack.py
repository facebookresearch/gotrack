# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from einops import rearrange
import numpy as np
import torch
import time
from bop_toolkit_lib import inout
import numpy.typing as npt
from PIL import Image

from model import config, base
from model.blocks import decoder
from model.heads.dpt import model as dpt_head
from utils import (
    corresp_util,
    data_util,
    misc,
    transform3d,
    vis_base_util,
    vis_flow_util,
    structs,
    logging,
    dinov2_util,
)


logger = logging.get_logger(__name__)


class GoTrack(base.ModelBase):
    def __init__(
        self,
        opts: Optional[config.GoTrackOpts] = None,
        result_dir: Optional[Path] = None,
        result_file_name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super(GoTrack, self).__init__()
        if opts is None:
            opts = config.GoTrackOpts()
        self.opts = opts
        self.result_dir = result_dir
        self.result_file_name = result_file_name

        # Initialize the backbone.
        self.backbone: torch.nn.Module = dinov2_util.DinoFeatureExtractor(
            model_name=self.opts.backbone_name
        )

        # Initialize the decoder.
        self.decoder = decoder.Decoder(self.opts.decoder_opts)

        # Initialize the head.
        self.pose_head = dpt_head.DPTHead(self.opts.head_opts)

    def extract_features(
        self,
        inputs: Dict[str, Union[torch.Tensor, structs.Collection]],
    ) -> Dict[str, torch.Tensor]:
        """Extract features from the backbone."""
        # Step 1: Extract features from the backbone.
        images = torch.cat(
            (inputs["rgbs_query"], inputs["rgbs_template"]),  # pyre-ignore
            dim=0,
        )
        start_time = time.time()

        if self.opts.frozen_backbone:
            with torch.no_grad():
                features = self.backbone(images)["feature_maps"]
        else:
            features = self.backbone(images)["feature_maps"]
        feature_time = time.time() - start_time
        features_query, features_reference = features.chunk(2, dim=0)
        return {
            "features_query": features_query,
            "features_reference": features_reference,
            "feature_time": feature_time,
        }

    def compute_flow(
        self,
        inputs: Dict[str, Union[torch.Tensor, structs.Collection]],
    ) -> Dict[str, torch.Tensor]:
        """Computes flow and confidence from inputs.

        There are three main steps in the forward pass:
        1. Extract features from the backbone.
        2. Cross-attention between query and reference features.
        3. Predicting flow and confidences.
        """
        if "features_query" not in inputs:
            # Extract features from the backbone.
            features = self.extract_features(inputs)
            features_query = features["features_query"]
            features_reference = features["features_reference"]
            feature_time = features["feature_time"]
        else:
            features_query = inputs["features_query"]
            features_reference = inputs["features_reference"]
            feature_time = 0.0

        # Step 2: Cross-attention between query and reference features.
        start_time = time.time()
        _, list_features_reference = self.decoder(
            features_query=features_query,
            features_reference=features_reference,
            crop_size=self.opts.crop_size,
        )
        decoder_time = time.time() - start_time
        start_time = time.time()

        # Step 3: Predicting flow and confidences.
        pred_flows, pred_confidences = self.pose_head(
            list_features_reference, self.opts.crop_size
        )

        # Keep only flows and confidences inside template masks.
        pred_flows_masked = pred_flows * inputs["masks_template"].unsqueeze(1)
        pred_confidences_masked = pred_confidences * inputs["masks_template"]

        head_time = time.time() - start_time
        run_time = feature_time + decoder_time + head_time
        return {
            "flows": pred_flows_masked,
            "confidences": pred_confidences_masked,
            "run_time": run_time,
        }

    def compute_poses_from_flows(
        self, forward_inputs: Dict[str, Any], forward_outputs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes poses from flows using PnP.

        Args:
            forward_inputs: Input of the forward pass.
            forward_outputs: Output of the forward pass.
        Returns:
            pred_poses_world_from_model: [batch_size, 4, 4] poses in world coordinate.
            pred_poses_crop_cam_from_model: [batch_size, 4, 4] poses in crop camera coordinate.
            pred_poses_orig_cam_from_model: [batch_size, 4, 4] poses in original camera coordinate.
        """

        batch_size = forward_outputs["flows"].shape[0]
        start_time = time.time()
        (
            pred_poses_crop_cam_from_model,
            pred_scores,
            pred_pnp_err_map,
        ) = corresp_util.poses_from_flows(
            flows=rearrange(forward_outputs["flows"], "b c h w -> b h w c"),
            flow_weights=forward_outputs["confidences"],
            source_depths=forward_inputs["depths_template"],
            source_masks=forward_inputs["masks_template"],
            intrinsics=forward_inputs["crop_cam_intrinsics"],
            Ts_source_cam_from_model=forward_inputs["Ts_crop_cam_from_model_template"],
            pnp_opts=self.opts.pnp_opts,
            weight_threshold=self.opts.visib_threshold,
        )
        pnp_time = time.time() - start_time

        device = forward_outputs["flows"].device
        if len(pred_poses_crop_cam_from_model) == 0:
            # If no poses are predicted, we return dummy values.
            identities = torch.stack(
                [
                    torch.eye(4, dtype=torch.float32, device=device)
                    for _ in range(batch_size)
                ]
            )
            pred_poses_world_from_model = identities.detach().clone()
            pred_poses_crop_cam_from_model = identities.detach().clone()
            pred_poses_orig_cam_from_model = identities.detach().clone()
            pred_scores = torch.zeros(batch_size, dtype=torch.float32)
        else:
            # Recover the poses in original camera coordinates.
            Ts_orig_cam_from_crop_cam = transform3d.inverse_se3(
                forward_inputs["Ts_crop_cam_from_orig_cam"]
            ).to(device)
            pred_poses_orig_cam_from_model = (
                Ts_orig_cam_from_crop_cam @ pred_poses_crop_cam_from_model.to(device)
            )
            Ts_world_from_cam = forward_inputs["Ts_world_from_cam"].to(device)
            pred_poses_world_from_model = (
                Ts_world_from_cam @ pred_poses_orig_cam_from_model
            )

        return (
            pred_poses_world_from_model,
            pred_poses_crop_cam_from_model,
            pred_poses_orig_cam_from_model,
            pred_scores,
            pred_pnp_err_map,
            pnp_time,
        )

    def get_vis_tiles(
        self,
        predictions: Dict[str, Any],
        vis_path: Path,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, npt.NDArray]:
        # Visualize the results only in debug mode.
        if not self.opts.debug or len(predictions["objects"]) == 0:
            return None
        else:
            vis_tiles = vis_flow_util.get_vis_tiles(
                predictions=predictions,
                renderer=self.renderer,
                vis_output_poses=True,
                vis_minimal=self.opts.vis_minimal,
            )
            # Assemble the tiles into a grid and save.
            if len(vis_tiles) > 0:
                # Collect a list of tiles.
                vis_tiles_list = []
                self.vis_tiles_sizes = {}
                if len(self.vis_tiles_sizes) == 0:
                    # Use tiles from the first frame as the reference.
                    vis_tiles_list = list(vis_tiles.values())
                    for tile_name, tile in vis_tiles.items():
                        self.vis_tiles_sizes[tile_name] = (tile.shape[1], tile.shape[0])
                else:
                    for tile_name, tile_size in self.vis_tiles_sizes.items():
                        if tile_name in vis_tiles:
                            tile = vis_tiles[tile_name]
                            assert tile.shape[1] == tile_size[0]
                            assert tile.shape[0] == tile_size[1]
                        else:
                            # Create a black tile if the tile is missing.
                            tile = 225 * np.ones(
                                (tile_size[1], tile_size[0], 3), dtype=np.uint8
                            )
                        vis_tiles_list.append(tile)
                vis_grid = vis_base_util.build_grid(
                    tiles=vis_tiles_list,
                    tile_pad=10,
                )
                vis_grid = Image.fromarray(vis_grid)
                vis_grid.save(vis_path)
                logger.info(f"GoTrack: Saving visualization to: {vis_path}")
            return vis_grid

    def forward_pipeline(
        self,
        inputs: Dict[str, Any],
        batch_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward pass of the refinement model.

        Step 1: Estimate relative pose between the input and reference image in `crop_camera` coordinate frame.
        Step 2: Update the pose in `crop_camera`, then recover the pose in `original_camera`.

        All renderings, predictions, and updates are in the crop camera coordinates.
        """

        # Number of iterations.
        num_iters = self.opts.num_iterations_test

        # Define the output of the forward pass.
        outputs: Dict[str, Any] = {
            "objects": structs.Collection(),
            "num_iters": num_iters,
            "run_time": 0.0,
        }

        # Early termination if there are no object poses to refine.
        if len(inputs["objects"]) == 0:
            return outputs

        # Get the object vertices and ids.
        obj_ids = inputs["objects"].labels.cpu().numpy()
        object_vertices = [self.object_vertices[obj_id] for obj_id in obj_ids]
        # Process inputs (this is done in `collate_postprocessing` when training).
        if self.opts.process_inputs:
            inputs = data_util.process_inputs(
                inputs=inputs,
                crop_size=self.opts.crop_size,
                crop_rel_pad=self.opts.crop_rel_pad,
                cropping_type=self.opts.cropping_type,
                ssaa_factor=self.opts.ssaa_factor,
                object_vertices=object_vertices,
                obj_ids=obj_ids,
                renderer=self.renderer,
                background_type=self.opts.background_type,
            )

        # Intialize the predictions.
        pred_objects = deepcopy(inputs["objects"])

        # Get the initial forward inputs (will be updated after each iteration).
        forward_inputs = data_util.prepare_inputs(inputs)

        total_forward_time = 0.0
        total_pnp_time = 0.0
        for iter_idx in range(num_iters):
            if self.opts.debug:
                logger.info(f"GoTrack: Refinement iteration: {iter_idx + 1}")
            prefix = f"iter={iter_idx}"
            if self.opts.debug:
                outputs[f"{prefix}_inputs"] = forward_inputs

            # Step 1: Running the refiner to get predictions in crop camera coordinates.
            forward_outputs = self.compute_flow(forward_inputs)
            total_forward_time += forward_outputs["run_time"]

            # Step 2: Compute poses from flows.
            (
                pred_poses_world_from_model,
                pred_poses_crop_cam_from_model,
                pred_poses_orig_cam_from_model,
                pred_scores,
                pred_pose_fitting_tiles,
                pnp_time,
            ) = self.compute_poses_from_flows(forward_inputs, forward_outputs)
            total_pnp_time += pnp_time

            # Update the poses in the prediction.
            pred_objects.poses_cam_from_model = pred_poses_orig_cam_from_model.cpu()
            pred_objects.poses_world_from_model = pred_poses_world_from_model.cpu()
            pred_objects.pose_scores = pred_scores

            if self.opts.debug:
                scores_display = list(pred_scores.cpu().numpy())
                scores_display = np.round(scores_display, 3)
                logger.info(f"GoTrack: - Pose scores: {scores_display}")

            # Keep intermediate results for visualization.
            outputs[f"{prefix}_pred_poses_crop_cam_from_model"] = (
                pred_poses_crop_cam_from_model
            )
            outputs[f"{prefix}_pred_poses_orig_cam_from_model"] = (
                pred_poses_orig_cam_from_model
            )
            outputs[f"{prefix}_pose_fitting_tiles"] = pred_pose_fitting_tiles

            if iter_idx < num_iters - 1:
                if self.opts.re_crop_every_iter:
                    # Update init_objects for the next iteration and re-run preprocessing_inputs.
                    inputs["objects"] = pred_objects
                    inputs = data_util.process_inputs(
                        inputs=inputs,
                        crop_size=self.opts.crop_size,
                        crop_rel_pad=self.opts.crop_rel_pad,
                        cropping_type=self.opts.cropping_type,
                        ssaa_factor=self.opts.ssaa_factor,
                        object_vertices=object_vertices,
                        obj_ids=obj_ids,
                        renderer=self.renderer,  # pyre-ignore
                        background_type=self.opts.background_type,
                    )
                    forward_inputs = data_util.prepare_inputs(inputs)
                else:
                    raise NotImplementedError(
                        "GoTrack: Re-cropping is currently required in every iteration."
                    )

            # Keep intermediate results for visualization.
            outputs[f"{prefix}_outputs"] = forward_outputs

        # Add run-time for each instance.
        outputs["run_time"] = total_forward_time + total_pnp_time
        pred_objects.times = torch.as_tensor(
            [outputs["run_time"] for _ in range(len(pred_objects))]
        )
        outputs["objects"] = pred_objects

        # In BOP challenge, run-time = total run-time of all stages (detection, coarse pose, refinement).
        run_times = inputs["images"].times + outputs["run_time"]

        # Save the prediction.
        assert len(outputs["objects"]) == len(inputs["objects"]), (
            "The number of objects in the prediction and input should be the same."
        )
        # In case there are multiple frames in batch, frame_ids is mapping obj_id to (scene_id, im_id)
        assert self.result_dir is not None, "Result directory is not set."
        misc.save_per_frame_prediction(
            scene_ids=inputs["images"].scene_ids.cpu().numpy(),
            im_ids=inputs["images"].im_ids.cpu().numpy(),
            run_times=run_times.cpu().numpy(),
            obj_ids=outputs["objects"].labels.cpu().numpy(),
            obj_frame_ids=inputs["objects"].frame_ids.cpu().numpy(),
            poses_cam_from_model=outputs["objects"].poses_cam_from_model.cpu().numpy(),
            scores=outputs["objects"].pose_scores.cpu().numpy(),
            save_path=self.result_dir / f"per_frame_refined_poses_{batch_idx:06d}.json",
        )
        logger.info(
            f"GoTrack: Run-time: {outputs['run_time']} (Forward:{total_forward_time}, PnP: {total_pnp_time})"
        )
        # Get visualization.
        vis_save_path = self.result_dir / f"vis_{batch_idx:06d}_goTrack.png"
        self.get_vis_tiles(outputs, vis_save_path)

        return outputs

    def test_step(self, inputs: Dict[str, Any], idx: Optional[int] = None):
        # Run the forward pass.
        outputs = self.forward_pipeline(inputs, batch_idx=idx)
        return outputs

    def on_test_epoch_end(self):
        if self.global_rank == 0:  # only rank 0 process
            # List all files in the result directory
            result_files = list(self.result_dir.glob("per_frame_refined_poses*.json"))
            # Sort the files by their names
            result_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
            # Merge the files
            all_predictions = []
            for result_file in result_files:
                per_frame_predictions = inout.load_bop_results(result_file)
                all_predictions.extend(per_frame_predictions)
            # Save the merged predictions
            result_full_path = self.result_dir / f"{self.result_file_name}.csv"
            inout.save_bop_results(result_full_path, all_predictions)
            logger.info(f"GoTrack: Saved all predictions to {result_full_path}")
