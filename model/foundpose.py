# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
from model import config, base
from bop_toolkit_lib import inout
from utils import (
    corresp_util,
    dinov2_util,
    feature_util,
    knn_util,
    logging,
    misc,
    pnp_util,
    renderer_base,
    repre_util,
    structs,
    pca_util,
    torch_helpers,
    transform3d,
    vis_base_util,
    vis_foundpose_util,
)
from PIL import Image

logger = logging.get_logger(__name__)


class FoundPose(base.ModelBase):
    """Adapted from https://github.com/facebookresearch/foundpose"""

    def __init__(
        self,
        opts: config.FoundPoseOpts,
        result_dir: Optional[Path] = None,
        result_file_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.opts = opts
        self.backbone: torch.nn.Module = dinov2_util.DinoFeatureExtractor(
            model_name=self.opts.extractor_name
        ).cuda()
        self.result_dir = result_dir
        self.result_file_name = result_file_name
        self.grid_points = None
        logger.info("FoundPose initialized !")

    def post_onboarding_processing(self):
        """Move the model to the specified device."""

        self.objects_knn_repre = {}
        for obj_id in tqdm(self.objects_repre, "FoundPose: onboarding objects"):
            self.objects_knn_repre[obj_id] = {}
            repre = self.objects_repre[obj_id]
            assert isinstance(repre, repre_util.FeatureBasedObjectRepre)

            # Build a kNN index from object feature vectors.
            if self.opts.match_template_type == "tfidf":
                visual_words_knn_index = knn_util.KNN(
                    k=repre.template_desc_opts.tfidf_knn_k,
                    metric=repre.template_desc_opts.tfidf_knn_metric,
                )
                visual_words_knn_index.fit(repre.feat_cluster_centroids)
                self.objects_knn_repre[obj_id]["visual_words_knn_index"] = (
                    visual_words_knn_index
                )

            # Build per-template KNN index with features from that template.
            template_knn_indices = []
            if self.opts.match_feat_matching_type == "cyclic_buddies":
                for template_id in range(len(repre.template_cameras_cam_from_model)):
                    logger.debug(f"Building KNN index for template {template_id}...")
                    tpl_feat_mask = repre.feat_to_template_ids == template_id
                    tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()
                    template_feats = repre.feat_vectors[tpl_feat_ids]

                    # Build knn index for object features.
                    template_knn_index = knn_util.KNN(k=1, metric="l2")
                    template_knn_index.fit(template_feats.cpu())
                    template_knn_indices.append(template_knn_index)
                logger.debug("Per-template KNN indices built.")
            self.objects_knn_repre[obj_id]["template_knn_indices"] = (
                template_knn_indices
            )
        logger.info("FoundPose: onboarding processed!")

    def get_grid_points(self) -> torch.Tensor:
        if self.grid_points is None:
            self.grid_points = feature_util.generate_grid_points(
                grid_size=self.opts.crop_size,
                cell_size=self.opts.grid_cell_size,
            ).to(self.device)
        return self.grid_points

    def estimate_poses(
        self,
        obj_id: int,
        repre: repre_util.FeatureBasedObjectRepre,
        feature_map_chw: torch.Tensor,
        crop_mask: torch.Tensor,
        crop_camera: structs.CameraModel,
    ) -> Dict[str, torch.Tensor]:
        """This function compute correspondences and retrieve the top k templates for each input crop."""
        times = {}
        timer = misc.Timer(enabled=True)
        timer.start()

        width, height = crop_mask.shape
        grid_points = self.get_grid_points().clone()
        grid_points = grid_points.to(self.device)
        crop_mask = crop_mask.to(self.device)
        # Keep only points inside the object mask.
        query_points = feature_util.filter_points_by_mask(grid_points, crop_mask)
        if query_points.shape[0] == 0:
            query_points = grid_points.clone()

        # Subsample query points if we have too many.
        if query_points.shape[0] > self.opts.max_num_queries:
            perm = torch.randperm(query_points.shape[0])
            query_points = query_points[perm[: self.opts.max_num_queries]]

        query_features = feature_util.sample_feature_map_at_points(
            feature_map_chw=feature_map_chw,
            points=query_points,
            crop_size=(width, height),
        ).contiguous()

        # Get representation for the object.
        knn_repre = self.objects_knn_repre[obj_id]
        template_knn_indices = knn_repre["template_knn_indices"]
        visual_words_knn_index = knn_repre["visual_words_knn_index"]

        # Potentially project features to a PCA space.
        if (
            query_features.shape[1] != repre.feat_vectors.shape[1]
            and len(repre.feat_raw_projectors) != 0
        ):
            query_features_proj = pca_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()
        else:
            query_features_proj = query_features

        # Establish 2D-3D correspondences.
        corresp = []
        if len(query_points) != 0:
            corresp = corresp_util.establish_correspondences(
                query_points=query_points,
                query_features=query_features_proj,
                object_repre=repre,
                template_matching_type=self.opts.match_template_type,
                template_knn_indices=template_knn_indices,
                feat_matching_type=self.opts.match_feat_matching_type,
                top_n_templates=self.opts.match_top_n_templates,
                top_k_buddies=self.opts.match_top_k_buddies,
                visual_words_knn_index=visual_words_knn_index,
                debug=self.opts.debug,
            )
        else:
            logger.info("FoundPose: No correspondences found, skipping.")
        if self.opts.debug:
            logger.info(
                f"FoundPose: Num of corresp: {[len(c['coord_2d']) for c in corresp]}"
            )

        if self.opts.run_retrieval_only:
            assert len(corresp) == 1, "Only one correspondence is expected."
            # If we are only running retrieval, we can skip the pose estimation.
            pose_crop_cam_from_model_template = repre.template_cameras_cam_from_model[
                corresp[0]["template_id"]
            ].T_world_from_eye
            pose_crop_cam_from_model_template = transform3d.inverse_se3_numpy(
                pose_crop_cam_from_model_template
            )
            final_poses = [
                {
                    "R_m2c": pose_crop_cam_from_model_template[:3, :3],
                    "t_m2c": pose_crop_cam_from_model_template[:3, 3:],
                    "corresp_id": 0,
                    "template_id": corresp[0]["template_id"],
                }
            ]
            run_time = timer.elapsed(
                "FoundPose: Time for estimating coarse pose per instance"
            )
        else:
            # Estimate coarse poses from corespondences.
            coarse_poses = []
            for corresp_id, corresp_curr in enumerate(corresp):
                # We need at least 3 correspondences for P3P.
                num_corresp = len(corresp_curr["coord_2d"])
                if num_corresp < 6:
                    logger.info(f"Only {num_corresp} correspondences, skipping.")
                    continue
                (
                    coarse_pose_success,
                    R_m2c_coarse,
                    t_m2c_coarse,
                    inliers_coarse,
                    quality_coarse,
                ) = pnp_util.estimate_pose(
                    corresp=corresp_curr,
                    camera_c2w=crop_camera,
                    pnp_type=self.opts.pnp_type,
                    pnp_ransac_iter=self.opts.pnp_ransac_iter,
                    pnp_inlier_thresh=self.opts.pnp_inlier_thresh,
                    pnp_required_ransac_conf=self.opts.pnp_required_ransac_conf,
                    pnp_refine_lm=self.opts.pnp_refine_lm,
                )
                logger.info(f"Quality of coarse pose {corresp_id}: {quality_coarse}")

                if coarse_pose_success:
                    coarse_poses.append(
                        {
                            "type": "coarse",
                            "R_m2c": R_m2c_coarse,
                            "t_m2c": t_m2c_coarse,
                            "corresp_id": corresp_id,
                            "quality": quality_coarse,
                            "inliers": inliers_coarse,
                            "template_id": corresp_curr["template_id"],
                        }
                    )
                else:
                    logger.info(f"FoundPose: WARNING: failed for corresp {corresp_id}.")
                    coarse_poses.append(
                        {
                            "type": "coarse",
                            "R_m2c": np.eye(3),
                            "t_m2c": np.zeros((3, 1)),
                            "corresp_id": corresp_id,
                            "quality": 0,
                            "inliers": 0,
                            "template_id": corresp_curr["template_id"],
                        }
                    )
            # Find the best coarse pose.
            best_coarse_quality = None
            best_coarse_pose_id = 0
            for coarse_pose_id, pose in enumerate(coarse_poses):
                if best_coarse_quality is None or pose["quality"] > best_coarse_quality:
                    best_coarse_pose_id = coarse_pose_id
                    best_coarse_quality = pose["quality"]

            times["pose_coarse"] = timer.elapsed("Time for coarse pose")
            # Select the final pose estimate.
            final_poses = []
            if self.opts.final_pose_type in [
                "best_coarse",
            ]:
                # If no successful coarse pose, continue.
                if len(coarse_poses) == 0:
                    return None

                # Select the refined pose corresponding to the best coarse pose as the final pose.
                final_pose = None
                if self.opts.final_pose_type in [
                    "best_coarse",
                ]:
                    final_pose = coarse_poses[best_coarse_pose_id]

                if final_pose is not None:
                    final_poses.append(final_pose)

            else:
                raise ValueError(f"Unknown final pose type {self.opts.final_pose_type}")

            run_time = timer.elapsed("Time for estimating coarse pose (per detection)")
        return {
            "corresp": corresp,
            "final_poses": final_poses,
            "run_time": run_time,
        }

    def get_vis_tiles(
        self,
        repre_np: repre_util.FeatureBasedObjectRepre,
        object_lid: int,
        crop_rgb: torch.Tensor,
        feature_map_chw: torch.Tensor,
        renderer: renderer_base.RendererBase,
        pose_m2w: np.ndarray,
        camera_c2w: structs.CameraModel,
        corresp: List[Dict[str, Any]],
        final_pose: Dict[str, Any],
        best_corresp_np: List[Dict[str, Any]],
    ) -> List[np.ndarray]:
        base_image_hwc = crop_rgb.permute(1, 2, 0) * 255.0
        vis_base_image = torch_helpers.tensor_to_array(base_image_hwc).astype(np.uint8)

        # IDs and scores of the matched templates.
        matched_template_ids = [c["template_id"] for c in corresp]
        matched_template_scores = [c["template_score"] for c in corresp]
        vis_tiles = vis_foundpose_util.vis_inference_results(
            base_image=vis_base_image,
            object_repre=repre_np,
            object_lid=object_lid,
            object_pose_m2w=pose_m2w,
            feature_map_chw=feature_map_chw,
            vis_feat_map=self.opts.vis_feat_map,
            camera_c2w=camera_c2w,
            corresp=best_corresp_np,
            matched_template_ids=matched_template_ids,
            matched_template_scores=matched_template_scores,
            best_template_ind=final_pose["corresp_id"],
            renderer=renderer,
            corresp_top_n=self.opts.vis_corresp_top_n,
            extractor=self.backbone,
        )
        return vis_tiles

    def forward_pipeline(
        self,
        inputs: Dict[str, Union[structs.CameraModel, structs.Collection]],
        batch_idx: Optional[int] = None,
    ) -> Dict[str, Union[structs.CameraModel, structs.Collection]]:
        """Estimate the coarse object poses using CNOS outputs."""

        detections = inputs["detections"]
        crop_detections = inputs["crop_detections"]
        scene_observation = inputs["scene_obs"]
        orig_camera = scene_observation.camera
        T_world_from_orig_camera = orig_camera.T_world_from_eye
        T_orig_cam_from_world = transform3d.inverse_se3_numpy(T_world_from_orig_camera)

        timer = misc.Timer(enabled=True)
        if self.opts.debug:
            logger.info(f"FoundPose: Found {len(detections)} detections.")

        # Get the feature map from the backbone.
        features = self.backbone(crop_detections.rgbs.to(self.device))
        feature_maps = features["feature_maps"]

        pred_poses_orig_cam_from_model = []
        pred_poses_world_from_model = []
        matched_templates = {
            "rgbs": [],
            "depths": [],
            "masks": [],
            "cameras": [],
        }
        vis_tiles = []
        run_time = 0.0

        timer.start()
        for idx in range(len(detections)):
            # Establish correspondences and retrievel the top k templates for each input crop.
            # The pose estimates are done in the crop camera.
            obj_id = detections.labels[idx].item()
            repre = self.objects_repre[obj_id]
            assert isinstance(repre, repre_util.FeatureBasedObjectRepre)

            estimate = self.estimate_poses(
                obj_id=obj_id,
                repre=repre,
                feature_map_chw=feature_maps[idx],
                crop_mask=crop_detections.masks[idx],
                crop_camera=crop_detections.cameras[idx],
            )
            if estimate is None:
                logger.info("FoundPose: No estimate found, skipping.")
                pred_pose_world_from_model = np.eye(4)
                pred_pose_orig_cam_from_model = np.eye(4)
                template_id = 0
            else:
                run_time += estimate["run_time"]
                assert len(estimate["final_poses"]) == 1, "One final pose is expected."
                final_pose = estimate["final_poses"][0]
                pose_crop_cam_from_model = transform3d.Rt_to_4x4_numpy(
                    R=final_pose["R_m2c"],
                    t=np.asarray(final_pose["t_m2c"]).reshape(1, 3),
                )
                T_world_from_crop_cam = crop_detections.cameras[idx].T_world_from_eye
                pred_pose_world_from_model = (
                    T_world_from_crop_cam @ pose_crop_cam_from_model
                )
                pred_pose_orig_cam_from_model = (
                    T_orig_cam_from_world @ pred_pose_world_from_model
                )
                template_id = final_pose["template_id"]
                if self.opts.debug:
                    # Visualize the results.
                    pose_m2w = structs.ObjectPose(
                        R=pred_pose_world_from_model[:3, :3],
                        t=pred_pose_world_from_model[:3, 3:],
                    )
                    obj_repre = self.objects_repre[obj_id]
                    vis_tile = self.get_vis_tiles(
                        repre_np=repre_util.convert_object_repre_to_numpy(obj_repre),
                        object_lid=obj_id,
                        crop_rgb=crop_detections.rgbs[idx],
                        feature_map_chw=feature_maps[idx],
                        renderer=self.renderer,
                        pose_m2w=pose_m2w,
                        camera_c2w=crop_detections.cameras[idx],
                        corresp=estimate["corresp"],
                        final_pose=final_pose,
                        best_corresp_np=estimate["corresp"][final_pose["corresp_id"]],
                    )
                    vis_tiles.append(np.vstack(vis_tile))
            pred_poses_world_from_model.append(pred_pose_world_from_model)
            pred_poses_orig_cam_from_model.append(pred_pose_orig_cam_from_model)
            matched_templates["rgbs"].append(repre.templates[template_id])
            matched_templates["depths"].append(repre.template_depths[template_id])
            matched_templates["masks"].append(repre.template_masks[template_id])
            matched_templates["cameras"].append(
                repre.template_cameras_cam_from_model[template_id]
            )
        # Add predicted poses to the detections.
        detections.poses_cam_from_model = torch_helpers.array_to_tensor(
            np.stack(pred_poses_orig_cam_from_model, axis=0)
        ).float()
        detections.poses_world_from_model = torch_helpers.array_to_tensor(
            np.stack(pred_poses_world_from_model, axis=0)
        ).float()

        # Define matched templates as a collection.
        templates = structs.Collection()
        templates.rgbs = torch.stack(matched_templates["rgbs"], dim=0) / 255.0
        templates.rgbs = templates.rgbs.float()
        templates.masks = torch.stack(matched_templates["masks"], dim=0).bool()
        templates.depths = torch.stack(matched_templates["depths"], dim=0)
        templates.cameras = matched_templates["cameras"]
        # Collect the visualization tiles.
        if self.opts.debug:
            vis_path = self.result_dir / f"vis_{batch_idx:06d}_foundPose.png"
            if len(vis_tiles):
                vis_grid = vis_base_util.build_grid(
                    tiles=vis_tiles,
                    tile_pad=10,
                )
                vis_grid = Image.fromarray(vis_grid)
                vis_grid.save(vis_path)
                inout.save_im(vis_path, vis_grid)
                logger.info(f"FoundPose: Saved visualization to {vis_path}")

        # Calculate the run time.
        run_time = timer.elapsed("FoundPose: inference time")
        total_run_time = inputs["run_time"] + run_time

        # TODO: How FoundPose calculate pose confidence score ?
        # For now, we use the score from the detection.
        pred_scores = torch_helpers.tensor_to_array(detections.scores)

        misc.save_per_frame_prediction(
            scene_ids=[scene_observation.scene_id],
            im_ids=[scene_observation.im_id],
            run_times=[total_run_time],
            obj_ids=torch_helpers.tensor_to_array(detections.labels),
            scores=pred_scores,
            poses_cam_from_model=pred_poses_orig_cam_from_model,
            save_path=self.result_dir / f"per_frame_coarse_poses_{batch_idx:06d}.json",
        )
        return {
            "detections": detections,
            "crop_detections": crop_detections,
            "templates": templates,
            "run_time": total_run_time,
            "scene_observation": scene_observation,
        }
