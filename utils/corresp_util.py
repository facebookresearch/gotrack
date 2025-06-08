# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch
from einops import rearrange
from utils import (
    knn_util,
    misc,
    pnp_util,
    repre_util,
    template_util,
    transform3d,
    config,
    logging,
)


logger = logging.get_logger(__name__)


def cyclic_buddies_matching(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    query_knn_index: knn_util.KNN,
    object_features: torch.Tensor,
    object_knn_index: knn_util.KNN,
    top_k: int,
    debug: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find best buddies via cyclic distance (https://arxiv.org/pdf/2204.03635.pdf)."""

    # Find nearest neighbours in both directions.
    query2obj_nn_ids = object_knn_index.search(query_features)[1].flatten()
    obj2query_nn_ids = query_knn_index.search(object_features)[1].flatten()

    # 2D locations of the query points.
    u1 = query_points

    # 2D locations of the cyclic points.
    cycle_ids = obj2query_nn_ids[query2obj_nn_ids]
    u2 = query_points[cycle_ids]

    # L2 distances between the query and cyclic points.
    cycle_dists = torch.linalg.norm(u1 - u2, axis=1)

    # Keep only top k best buddies.
    top_k = min(top_k, query_points.shape[0])
    _, query_bb_ids = torch.topk(-cycle_dists, k=top_k, sorted=True)

    # Best buddy scores.
    bb_dists = cycle_dists[query_bb_ids]
    bb_scores = torch.as_tensor(1.0 - (bb_dists / bb_dists.max()))

    # Returns IDs of the best buddies.
    object_bb_ids = query2obj_nn_ids[query_bb_ids]

    return query_bb_ids, object_bb_ids, bb_dists, bb_scores


def establish_correspondences(
    query_points: torch.Tensor,
    query_features: torch.Tensor,
    object_repre: repre_util.FeatureBasedObjectRepre,
    template_matching_type: str,
    feat_matching_type: str,
    top_n_templates: int,
    top_k_buddies: int,
    visual_words_knn_index: Optional[knn_util.KNN] = None,
    template_knn_indices: Optional[List[knn_util.KNN]] = None,
    debug: bool = False,
) -> List[Dict]:
    """Establishes 2D-3D correspondences by matching image and object features."""

    timer = misc.Timer(enabled=debug)
    timer.start()
    template_ids, template_scores = template_util.template_matching(
        query_features=query_features,
        object_repre=object_repre,
        top_n_templates=top_n_templates,
        matching_type=template_matching_type,
        visual_words_knn_index=visual_words_knn_index,
        debug=debug,
    )
    if debug:
        timer.elapsed("Time for template matching")
    timer.start()
    # Build knn index for query features.
    query_knn_index = None
    if feat_matching_type == "cyclic_buddies":
        query_knn_index = knn_util.KNN(k=1, metric="l2")
        query_knn_index.fit(query_features)

    # Establish correspondences for each dominant template separately.
    corresps = []
    for template_counter, template_id in enumerate(template_ids):
        # Get IDs of features originating from the current template.
        tpl_feat_mask = torch.as_tensor(
            object_repre.feat_to_template_ids == template_id
        )
        tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

        # Find N best buddies.
        if feat_matching_type == "cyclic_buddies":
            assert object_repre.feat_vectors is not None
            (
                match_query_ids,
                match_obj_ids,
                match_dists,
                match_scores,
            ) = cyclic_buddies_matching(
                query_points=query_points,
                query_features=query_features,
                query_knn_index=query_knn_index,
                object_features=object_repre.feat_vectors[tpl_feat_ids],
                object_knn_index=template_knn_indices[template_id],
                top_k=top_k_buddies,
                debug=debug,
            )
        else:
            raise ValueError(f"Unknown feature matching type ({feat_matching_type}).")

        match_obj_feat_ids = tpl_feat_ids[match_obj_ids]

        # Structures for storing 2D-3D correspondences and related info.
        coord_2d = query_points[match_query_ids]
        coord_2d_ids = match_query_ids
        assert object_repre.vertices is not None
        coord_3d = object_repre.vertices[match_obj_feat_ids]
        coord_conf = match_scores
        full_query_nn_dists = match_dists
        full_query_nn_ids = match_obj_feat_ids
        nn_vertex_ids = match_obj_feat_ids

        template_corresps = {
            "template_id": template_id,
            "template_score": template_scores[template_counter],
            "coord_2d": coord_2d,
            "coord_2d_ids": coord_2d_ids,
            "coord_3d": coord_3d,
            "coord_conf": coord_conf,
            "nn_vertex_ids": nn_vertex_ids,
        }
        # Add items for visualization/debugging.
        if debug:
            template_corresps.update(
                {
                    "nn_dists": full_query_nn_dists,
                    "nn_indices": full_query_nn_ids,
                }
            )

        corresps.append(template_corresps)
    if debug:
        timer.elapsed("Time for establishing corresp")

    return corresps


def flows_from_poses(
    source_masks: torch.Tensor,
    source_depths: torch.Tensor,
    intrinsics: torch.Tensor,
    Ts_target_from_source: torch.Tensor,
) -> torch.Tensor:
    """Calculates 2D flow from source to target images.

    The flow is obtained by projecting 3D points calculated from the source depth map into the target cameras.

    Args:
        source_masks: [batch_size, height, width]: masks of source image (template mask).
        source_depths: [batch_size, height, width]: depth maps of source image (template depth).
        intrinsics: [batch_size, 3, 3] camera intrinsics, assuming both source and target images have the same intrinsics.
        T_target_from_source: [batch_size, 4, 4] transformation from source to target (in camera coordinates).
    Returns:
        Flow source->target by reprojecting the 3d points of shape [batch_size, height, width, 2].
    """
    device = source_masks.device
    assert source_masks.dtype == torch.bool
    height = source_masks.shape[1]

    # Get pixels, 3D points from depth and intrinsics.
    # Get pixels, 3D points from depth and intrinsics.
    source_pixels, source_3d_points = transform3d.get_3d_points_from_depth(
        depths=source_depths,
        intrinsics=intrinsics,
    )
    source_3d_points = source_3d_points.to(device)  # B x H x W x 2
    source_3d_points = source_3d_points.to(device)  # B x H x W x 3

    # Rearrange the 3D points to [batch_size, height * width, 3].
    source_3d_points = rearrange(source_3d_points, "b h w c -> b (h w) c")

    # Transform the source 3D points to the target views.
    source_3d_points_in_target_cam = transform3d.transform_points(
        points=source_3d_points, matrix=Ts_target_from_source[:, None, ...].to(device)
    )

    # Project the source 3D points to the target image.
    proj_source_3d_points_in_target_cam = transform3d.project_3d_points_pinhole(
        source_3d_points_in_target_cam, intrinsics
    )

    # Reshape the reprojection to [batch_size, height, width, 2].
    proj_source_3d_points_in_target_cam = rearrange(
        proj_source_3d_points_in_target_cam, "b (h w) c -> b h w c", h=height
    )

    # Compute the flow = reprojection - original pixel coordinates.
    flows = proj_source_3d_points_in_target_cam - source_pixels

    # Mask out the invalid pixels.
    flows[~source_masks.unsqueeze(-1).repeat(1, 1, 1, 2)] = 0.0
    return flows


def poses_from_flows(
    flows: torch.Tensor,
    source_depths: torch.Tensor,
    source_masks: torch.Tensor,
    intrinsics: torch.Tensor,
    Ts_source_cam_from_model: Optional[torch.Tensor] = None,
    flow_weights: Optional[torch.Tensor] = None,
    weight_threshold: float = 0.3,
    pnp_solver_name: str = "opencv",
    pnp_opts: Optional[config.PnPOpts] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates 6D poses from 3D-to-2D correspondences created from flow(source->target) + depth maps.
    Args:
        flows: [batch_size, height, width, 2] optical flow.
        source_depths: [batch_size, height, width] depth maps of source image (template depth).
        source_masks: [batch_size, height, width] masks of source image (template mask).
        intrinsics: [batch_size, 3, 3] camera intrinsics, assuming both source and target images have the same intrinsics.
        Ts_source_cam_from_model: [batch_size, 4, 4] transformation from model to camera coordinates of source image.
        flow_weights: [batch_size, height, width] weights of the flow vectors.
        weight_threshold: Only flow vectors whose weight is above this threshold are considered.
        pnp_solver_name: Name of the PnP solver to use.
        pnp_opts: Options for the PnP solver.
    Returns:
    - When T_cam_from_model_source is provided (template pose is know), the output is absolute poses of shape [batch_size, 4, 4].
    - When T_cam_from_model_source is not provided (template pose is unknown), the output is the relative transformation from source camera to target camera.
    """
    (batch_size, height, width, _) = flows.shape
    device = flows.device

    # Get pixels, 3D points from depth and intrinsics.
    source_pixels, source_3d_points = transform3d.get_3d_points_from_depth(
        depths=source_depths,
        intrinsics=intrinsics,
    )
    source_3d_points = source_3d_points.to(device)  # B x H x W x 2
    source_3d_points = source_3d_points.to(device)  # B x H x W x 3

    # Get 2D correspondences from flow.
    corresp_2d = source_pixels + flows

    # Get 3D correspondences from depth.
    if Ts_source_cam_from_model is not None:
        # Get 3D points in object coordinates if the transform T_cam_from_model_source is available.
        inv_Ts_cam_from_model_source = transform3d.inverse_se3(Ts_source_cam_from_model)
        corresp_3d = transform3d.transform_points(
            points=source_3d_points,
            matrix=inv_Ts_cam_from_model_source[:, None, None, ...],
        )  # BxHxWx3
    else:
        corresp_3d = source_3d_points  # BxHxWx3

    # Get 6D poses from 3D-2D correspondences.
    assert flow_weights is not None
    output_dict = pnp_util.poses_from_correspondences(
        corresps_2d=corresp_2d,
        corresps_3d=corresp_3d,
        corresps_weight=flow_weights,
        intrinsics=intrinsics,
        masks=source_masks,
        init_poses=Ts_source_cam_from_model,
        pnp_solver_name=pnp_solver_name,
        pnp_opts=pnp_opts,
        weight_threshold=weight_threshold,
    )
    for key in ["estimated_poses", "quality", "proj_err"]:
        assert output_dict[key].shape[0] == batch_size, (
            f"{key} has different size, {output_dict[key].shape[0]} vs batch_size={batch_size}."
        )
    return (
        torch.as_tensor(output_dict["estimated_poses"]),
        torch.as_tensor(output_dict["quality"]),
        torch.as_tensor(output_dict["proj_err"]),
    )
