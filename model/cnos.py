# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from utils import (
    crop_generation,
    im_util,
    logging,
    loss_util,
    misc,
    repre_util,
    structs,
    torch_helpers,
    fastsam_util,
)
from model import config, base

logger = logging.get_logger(__name__)


class CNOS(base.ModelBase):
    """Adapted from https://github.com/nv-nguyen/cnos"""

    def __init__(
        self,
        opts: config.CNOSOpts,
        segmentation_model: fastsam_util.FastSAM,
        result_dir: Optional[Path] = None,
        result_file_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.opts = opts
        self.segmentation_model = segmentation_model
        self.descriptor_model = torch.hub.load(
            "facebookresearch/dinov2", self.opts.matching_model_name
        ).cuda()
        self.img_norm = im_util.im_normalize
        self.cosine_similarity = loss_util.PairwiseSimilarity()
        self.result_dir = result_dir
        self.result_file_name = result_file_name
        self.cnos_is_set = False  # use to set the device only once
        logger.info(f"CNOS initialized with opts={self.opts}!")

    def set_device(self):
        """Move the model to the specified device."""
        # Note: segmentation model (FastSAM) is special case and needs to be set separately.
        self.segmentation_model.set_device(self.device)
        self.objects_repre_cnos = self.objects_repre_cnos.to(self.device)
        self.cnos_is_set = True
        logger.info(f"CNOS: model and object representations moved to {self.device}.")

    @torch.no_grad()
    def post_onboarding_processing(self):
        """
        CNOS uses templates with tight crop, black background.
        Descriptors are cls_tokens of the object representations to be stacked together for batching.
        """
        templates_cls_tokens = []
        for obj_id in tqdm(self.objects_repre, desc="CNOS: onboarding objects"):
            repre = self.objects_repre[obj_id]
            assert isinstance(repre, repre_util.FeatureBasedObjectRepre)
            num_templates = repre.templates.shape[0]
            num_inplanes = 14
            if num_templates == 798:
                selected_idx = torch.from_numpy(
                    np.arange(0, num_templates, num_inplanes)
                )
            else:
                raise NotImplementedError("Subsampling is not supported.")
            template_rgbs = repre.templates[selected_idx]
            template_rgbs_bhwc = template_rgbs.permute(0, 2, 3, 1)
            template_masks = repre.template_masks[selected_idx]
            template_bboxes = im_util.masks_to_xyxy_boxes(
                repre.template_masks[selected_idx]
            )
            template_cameras = [
                repre.template_cameras_cam_from_model[idx] for idx in selected_idx
            ]

            crop_templates = crop_generation.batch_cropping_from_bbox(
                source_cameras=template_cameras,
                source_images=torch_helpers.tensor_to_array(template_rgbs_bhwc),
                source_masks=torch_helpers.tensor_to_array(
                    template_masks.unsqueeze(-1)
                ),
                source_xyxy_bboxes=torch_helpers.tensor_to_array(template_bboxes),
                crop_size=self.opts.crop_size,
                crop_rel_pad=self.opts.crop_rel_pad,
            )
            # CNOS uses black background.
            crop_templates.rgbs *= crop_templates.masks.unsqueeze(1)
            if self.opts.debug:
                vis_path = str(self.result_dir / f"template_obj{obj_id:06d}.png")
                save_image(crop_templates.rgbs, vis_path)
                logger.info(f"CNOS: Vis. of object {obj_id} at {vis_path}")

            # Get the cls_tokens of the templates.
            device = template_rgbs.device
            features = self.descriptor_model.forward_features(
                self.img_norm(crop_templates.rgbs.to(device))
            )
            templates_cls_tokens.append(features["x_norm_clstoken"])

        self.objects_repre_cnos = structs.Collection()
        self.objects_repre_cnos.descriptors = torch.stack(templates_cls_tokens, dim=0)
        self.objects_repre_cnos.obj_ids = torch.tensor(
            [obj_id for obj_id in self.objects_repre]
        )
        logger.info(f"CNOS: Loaded object representations to {self.device}.")

    def get_detection_proposals(
        self, scene_observation: structs.SceneObservation
    ) -> structs.Collection:
        """Get detection proposals from the segmentation model."""
        assert isinstance(scene_observation, structs.SceneObservation), (
            f"Expected type SceneObservation, got {type(scene_observation)}"
        )
        proposals = self.segmentation_model.generate_detection_proposals(
            scene_observation.image
        )
        # Filter noisy detections (masks or bboxes are too small).
        keep_idxs = im_util.filter_noisy_detections(
            proposals.boxes,
            proposals.masks,
            min_box_size=self.opts.min_box_size,
            min_mask_size=self.opts.min_mask_size,
            width=scene_observation.image.shape[1],
            height=scene_observation.image.shape[0],
        )
        return proposals[keep_idxs]

    @torch.no_grad()
    def forward_pipeline(
        self,
        scene_observation: structs.SceneObservation,
        batch_idx: Optional[int] = None,
    ) -> Dict[str, Union[structs.CameraModel, structs.Collection]]:
        assert isinstance(scene_observation, structs.SceneObservation), (
            f"Expected type SceneObservation, got {type(scene_observation)}"
        )
        if not self.cnos_is_set:
            self.set_device()

        timer = misc.Timer(enabled=True)
        timer.start()

        # Get 2D detection proposals.
        detections = self.get_detection_proposals(scene_observation)
        proposal_time = timer.elapsed("CNOS: get detection proposals")

        # Process the proposals to get cropp images.
        crop_detections = crop_generation.batch_cropping_from_bbox(
            source_cameras=[scene_observation.camera] * len(detections),
            source_images=[scene_observation.image] * len(detections),
            source_masks=torch_helpers.tensor_to_array(detections.masks.unsqueeze(-1)),
            source_xyxy_bboxes=torch_helpers.tensor_to_array(detections.boxes),
            crop_size=self.opts.crop_size,
            crop_rel_pad=self.opts.crop_rel_pad,
        )

        if self.opts.debug:
            timer.elapsed("CNOS: process proposals")

        # Calculate the descriptors for the processed images.
        timer.start()
        # CNOS uses black background.
        masked_detections = crop_detections.rgbs * crop_detections.masks.unsqueeze(1)
        processed_descriptors = self.descriptor_model.forward_features(
            self.img_norm(masked_detections).to(self.device)
        )["x_norm_clstoken"]

        # Matching descriptors to the descriptors of the onboarded objects.
        scores = self.cosine_similarity(
            query_descriptors=processed_descriptors,
            reference_descriptors=self.objects_repre_cnos.descriptors,
        )  # N_proposals x N_objects x N_templates
        if self.opts.aggregation_function == "avg5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
            # assign each proposal to the object with highest scores
            pred_scores, pred_obj_indexes = torch.max(
                score_per_proposal_and_object, dim=-1
            )  # N_query
            pred_obj_ids = self.objects_repre_cnos.obj_ids[pred_obj_indexes]
        elif self.opts.aggregation_function == "mean":
            score_per_proposal_and_object = torch.mean(scores, dim=-1)
            # assign each proposal to the object with highest scores
            pred_scores, pred_obj_indexes = torch.max(
                score_per_proposal_and_object, dim=-1
            )
            pred_obj_ids = self.objects_repre_cnos.obj_ids[pred_obj_indexes]
        elif self.opts.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
            # assign each proposal to the object with highest scores
            pred_scores, pred_obj_indexes = torch.max(
                score_per_proposal_and_object, dim=-1
            )
            pred_obj_ids = self.objects_repre_cnos.obj_ids[pred_obj_indexes]
        else:
            raise NotImplementedError(
                f"Aggregation function {self.opts.aggregation_function} is not implemented."
            )
        matching_time = timer.elapsed("CNOS: matching time")

        # Adding labels, scores to detections.
        detections.labels = pred_obj_ids
        detections.scores = pred_scores

        # Ranking detections by scores and visualize.
        if self.opts.debug:
            vis_path = str(
                self.result_dir / f"processed_detections_{batch_idx:06d}.png"
            )
            ranking = torch.argsort(detections.scores, descending=True)
            ranking = torch_helpers.tensor_to_array(ranking)
            # Visualize the top 100 detections.
            save_image(masked_detections[ranking[:100]], vis_path)
            logger.info(f"CNOS: Saved processed detections to {vis_path}")

        run_time = proposal_time + matching_time
        logger.info(f"CNOS: run time: {run_time:.2f} seconds")

        assert self.result_dir is not None, "Result directory is not set."
        misc.save_per_frame_prediction(
            scene_ids=[scene_observation.scene_id],
            im_ids=[scene_observation.im_id],
            run_times=[run_time],
            obj_ids=torch_helpers.tensor_to_array(detections.labels),
            scores=torch_helpers.tensor_to_array(detections.scores),
            obj_xywh_boxes=im_util.xyxy_to_xywh(detections.boxes.clone()),
            save_path=self.result_dir / f"per_frame_2d_detections_{batch_idx:06d}.json",
        )
        return {
            "detections": detections,
            "crop_detections": crop_detections,
            "run_time": run_time,
            "scene_obs": scene_observation,
        }

    def filter_detections_by_score(
        self, results: Dict[str, Any], score_threshold: float
    ) -> Dict[str, Any]:
        """Filter the detection results based on the score threshold."""
        detections = results["detections"]
        # Filter the results based on the score threshold.
        keep_masks = detections.scores > score_threshold
        device = detections.scores.device
        num_removed = {detections.scores.shape[0] - keep_masks.sum().item()}
        logger.info(
            f"CNOS: Filtered {num_removed} dets having conf score < {score_threshold}."
        )
        keep_idxs = torch.nonzero(keep_masks, as_tuple=False).squeeze(-1)
        # Handle the case when no detections are kept.
        if keep_idxs.numel() == 0:
            logger.warning(
                "CNOS: No detections kept after filtering by score threshold."
            )
            # Keep all detections to avoid errors in subsequent processing.
            keep_idxs = torch.arange(detections.labels.shape[0], device=device)

        return {
            "detections": detections[keep_idxs],
            "crop_detections": results["crop_detections"].to(device)[keep_idxs],
            "run_time": results["run_time"],
            "scene_obs": results["scene_obs"],
        }

    def filter_detections_by_targets(
        self,
        results: Dict[str, Any],
        target_objects: Dict[int, int],
    ) -> Dict[str, Any]:
        """Filter the detection results based on the target objects.
        For each object_id, keep only exact inst_count instances (ranked by score).
        """
        detections = results["detections"]
        # Filter the results based on the target objects.
        keep_idxs = []
        all_indexes = torch.arange(detections.labels.shape[0]).to(self.device)
        for obj_id in target_objects:
            num_instances = target_objects[obj_id]
            # Get the indices of the objects that match the target object id.
            obj_indexes = all_indexes[detections.labels == obj_id]
            obj_scores = detections.scores[detections.labels == obj_id]
            if obj_scores.shape[0] == 0:
                continue
            # Get the indices of the top num_instances scores.
            _, top_indexes = torch.topk(obj_scores, num_instances)
            keep_idxs.extend(obj_indexes[top_indexes])
        keep_idxs = torch.tensor(keep_idxs)
        return {
            "detections": results["detections"][keep_idxs],
            "crop_detections": results["crop_detections"][keep_idxs],
            "run_time": results["run_time"],
            "scene_obs": results["scene_obs"],
        }

    def test_step(
        self,
        inputs: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Test step for the CNOS model."""
        outputs = self.forward_pipeline(inputs["scene_observation"])
        return outputs
