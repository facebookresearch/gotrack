# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import time
import warnings
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from einops import rearrange
from kornia.geometry.conversions import rotation_matrix_to_axis_angle

from utils import (
    misc,
    structs,
    transform3d,
    vis_base_util,
    logging,
    torch_helpers,
    config,
)


# Ignore warnings about negative z-coordinates in EPnP
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")
logger = logging.get_logger(__name__)


def estimate_pose(
    corresp: Dict[str, Any],
    camera_c2w: structs.PinholePlaneCameraModel,
    pnp_type: str,
    pnp_ransac_iter: int,
    pnp_inlier_thresh: float,
    pnp_required_ransac_conf: float,
    pnp_refine_lm: bool,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, float]:
    """Estimates pose from provided 2D-3D correspondences and camera intrinsics.

    Args:
        corresp: correspondence dictionary as returned by corresp_util. Has the following:
            - coord_2d (num_points, 2): pixel coordinates from query image
            - coord_3d (num_points, 3): point coordinates from the 3d object representation
            - nn_distances (num_points) : cosine distances as returned by KNN
            - nn_indices (num_points).: indices within the object representations
        camera_c2w: camera intrinsics.
    """

    if pnp_type == "opencv":
        object_points = torch_helpers.tensor_to_array(corresp["coord_3d"]).astype(
            np.float32
        )
        image_points = torch_helpers.tensor_to_array(corresp["coord_2d"]).astype(
            np.float32
        )
        K = misc.get_intrinsic_matrix(camera_c2w)
        try:
            pose_est_success, rvec_est_m2c, t_est_m2c, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=K,
                distCoeffs=None,
                iterationsCount=pnp_ransac_iter,
                reprojectionError=pnp_inlier_thresh,
                confidence=pnp_required_ransac_conf,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except Exception:
            # Added to avoid a crash in cv2.solvePnPRansac due to too less correspondences
            # (even though more than 6 are provided, some of them may be colinear...).
            pose_est_success = False
            r_est_m2c = None
            t_est_m2c = None
            inliers = None
            quality = None
        else:
            # Optional LM refinement on inliers.
            if pose_est_success and pnp_refine_lm:
                rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                    objectPoints=object_points[inliers],
                    imagePoints=image_points[inliers],
                    cameraMatrix=K,
                    distCoeffs=None,
                    rvec=rvec_est_m2c,
                    tvec=t_est_m2c,
                )

            r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]
            quality = 0.0
            if pose_est_success:
                quality = float(len(inliers))

    elif pnp_type is None:
        raise ValueError("Unsupported PnP type")

    return pose_est_success, r_est_m2c, t_est_m2c, inliers, quality


def correspondences_2d_from_flows(
    flows: torch.Tensor, flow_masks: torch.Tensor, return_int: bool = True
) -> Tuple[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]:
    """Convert from 2D flows to 2D-2D correspondences.

    Args:
        flows: [batch_size, height, width, 2] optical flow.
        flow_masks: [batch_size, height, width] masks of source image (template mask).
    Returns:
        batch_indexes: [batch_size, num_nonzero_pixels] indices of samples in batches.
        source_u: [batch_size, num_nonzero_pixels] u coordinates of source pixels.
        source_v: [batch_size, num_nonzero_pixels] v coordinates of source pixels.
        target_u: [batch_size, num_nonzero_pixels] u coordinates of target pixels.
        target_v: [batch_size, num_nonzero_pixels] v coordinates of target pixels.
    """
    height, width = flow_masks.shape[1:]

    # Get the indices of the non-zero pixels.
    nonzero_coords_source = flow_masks.nonzero()
    batch_indexes, source_v, source_u = (
        nonzero_coords_source[:, 0],
        nonzero_coords_source[:, 1],
        nonzero_coords_source[:, 2],
    )

    # Applying the flow to the non-zero pixels.
    target_u = source_u + flows[batch_indexes, source_v, source_u, 0]
    target_v = source_v + flows[batch_indexes, source_v, source_u, 1]

    # Clamping the pixel coordinates to the image boundaries.
    target_u = torch.clamp(target_u, 0, width - 1)
    target_v = torch.clamp(target_v, 0, height - 1)
    if return_int:
        target_u = target_u.long()
        target_v = target_v.long()
    return batch_indexes, (source_u, source_v), (target_u, target_v)


def eval_pnp_output(
    width: int,
    height: int,
    corresp_3d: np.ndarray,
    corresp_2d: np.ndarray,
    corresp_weight: np.ndarray,
    intrinsic: np.ndarray,
    t_est_m2c: np.ndarray,
    rvec_est_m2c: np.ndarray,
    pnp_opts: Optional[config.PnPOpts] = None,
) -> Dict[str, Any]:
    """Evaluate the output of the PnP solver to get the projection error and pose quality.
    Args:
        corresp_3d: N x 3 array of 3D points.
        corresp_2d: N x 2 array of 2D points.
        corresp_weight: HxW array of weights.
        intrinsic: 3 x 3 array of camera intrinsics.
        t_est_m2c: 3 x 1 array of translation from model to camera.
        rvec_est_m2c: 3 x 1 array of rotation vector from model to camera.
        pose_m2c: 4 x 4 array of pose from model to camera. Only used when t_est_m2c and rvec_est_m2c are None.
        index_inliers: N x 1 array of inliers.

        visible_masks: HxW array of visible masks.
    Returns:
        outputs: Dict with the following keys:
            is_behind_camera: bool of whether the object pose is behind the camera.
            proj_err: HxW array of reprojection error.
            quality: float of pose quality.
            num_inliers: number of inliers.
    """
    if pnp_opts is None:
        pnp_opts = config.PnPOpts()
    outputs = {}
    pose_m2c = np.eye(4)
    pose_m2c[:3, :3] = cv2.Rodrigues(rvec_est_m2c)[0]
    pose_m2c[:3, 3:] = t_est_m2c

    # Check whether the object pose is behind the camera.
    outputs["is_behind_camera"] = pose_m2c[2, 3] < 0

    # Project the 3D points to the image plane.
    corresp_3d_in_cam = transform3d.transform_points(points=corresp_3d, matrix=pose_m2c)
    assert isinstance(corresp_3d_in_cam, np.ndarray)
    assert isinstance(corresp_2d, np.ndarray)
    proj_corresp_3d_in_cam = transform3d.project_3d_points_pinhole_numpy(
        points=corresp_3d_in_cam,
        intrinsics=intrinsic,
    )

    # Clip the projected points to the image boundaries.
    assert isinstance(proj_corresp_3d_in_cam, np.ndarray)
    proj_corresp_3d_in_cam[:, 0] = np.clip(proj_corresp_3d_in_cam[:, 0], 0, width - 1)
    proj_corresp_3d_in_cam[:, 1] = np.clip(proj_corresp_3d_in_cam[:, 1], 0, height - 1)

    # Calculate the reprojection error and statistics.
    reprojection_error = np.linalg.norm(corresp_2d - proj_corresp_3d_in_cam, axis=1)
    p50 = np.percentile(reprojection_error, 50)
    p95 = np.percentile(reprojection_error, 95)
    p100 = np.max(reprojection_error)

    # Compute inliers if it is None.
    inlier_masks = reprojection_error <= pnp_opts.pnp_inlier_thresh

    # Compute the number of inliers.
    outputs["num_inliers"] = np.sum(inlier_masks)

    # Create an image of the reprojection error.
    proj_err = np.zeros((height, width), dtype=np.float32)
    corresp_2d_u = np.clip(corresp_2d[:, 0], 0, width - 1)
    corresp_2d_v = np.clip(corresp_2d[:, 1], 0, height - 1)
    proj_err[np.int32(corresp_2d_v), np.int32(corresp_2d_u)] = reprojection_error

    # Convert the reprojection error to RGB.
    proj_err = vis_base_util.rgb_from_error_map(proj_err, max_norm=None)

    # Calculate pose quality (MLESAC style).
    # weights = np.exp(-(reprojection_error**2) / (2 * pnp_opts.pnp_inlier_thresh**2))
    # weights *= corresp_weight
    # pose_quality = np.sum(weights) / np.sum(corresp_weight)

    # Calculate pose quality = weight of inliers / weight of all points.
    pose_quality = np.sum(corresp_weight[inlier_masks]) / np.sum(corresp_weight)

    assert pose_quality <= 1.0
    outputs["quality"] = pose_quality

    assert isinstance(proj_err, np.ndarray)
    proj_err_img = vis_base_util.write_text_on_image(
        np.uint8(proj_err),  # pyre-ignore
        [
            {"name": f"Pose quality: {pose_quality:.2f}"},
            {"name": f"Proj err p50|95|100: {p50:.1f}|{p95:.1f}|{p100:.1f}"},
        ],
        size=14,
    )
    outputs["proj_err"] = np.float32(proj_err_img) / 255.0
    return outputs


def poses_from_correspondences(  # noqa: C901
    corresps_2d: torch.Tensor,
    corresps_3d: torch.Tensor,
    corresps_weight: torch.Tensor,
    masks: torch.Tensor,
    intrinsics: torch.Tensor,
    pnp_solver_name: str,
    init_poses: Optional[torch.Tensor] = None,
    pnp_opts: Optional[config.PnPOpts] = None,
    weight_threshold: float = 0.3,
) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
    """Estimates pose from provided 2D-3D correspondences and camera intrinsics.
    Args:
        - corresp_2d (batch_size, height, width, 2): 2D correspondences (pixel coordinates).
        - corresp_3d (batch_size, height, width, 3): 3D correspondences (point coordinates).
        - masks (batch_size, height, width): masks indicating valid correspondences.
        - intrinsic: (batch_size, 3, 3) camera intrinsics.
        - pnp_solver_name: name of the PnP solver to use. Options: "kornia_dlt", "pyt_epnp", "opencv".
        - weights (batch_size, height, width): weights of the correspondences.
        - init_poses (batch_size, 4, 4): initial poses. Only used when pnp_solver_name = "opencv".
        - pnp_opts: options for the PnP solver.
        - pnp_refine_lm: whether to refine the pose using LM. Only used when pnp_solver_name = "opencv".
    Returns:
        - poses (batch_size, 4, 4): estimated poses.
    """
    if pnp_opts is None:
        pnp_opts = config.PnPOpts()
    batch_size = len(corresps_2d)
    width, height = masks.shape[-2], masks.shape[-1]
    device = corresps_2d[0].device

    # Change the input correspondence from BxHxWxC to Bx(HxW)xC.
    corresps_2d = rearrange(corresps_2d, "b h w c -> b (h w) c")
    corresps_3d = rearrange(corresps_3d, "b h w c -> b (h w) c")
    corresps_weight = rearrange(corresps_weight, "b h w -> b (h w)")

    # Computer visible masks.
    visible_masks = corresps_weight > weight_threshold

    # Convert the initial poses to axis-angle and translation use cv2.IterativePnP to refine the poses.
    if init_poses is not None and pnp_solver_name == "opencv":
        init_rvecs = rotation_matrix_to_axis_angle(init_poses[:, :3, :3])
        init_tvecs = init_poses[:, :3, 3]

    outputs = {
        "estimated_poses": [],
        "num_inliers": [],
        "is_behind_camera": [],
        "quality": [],
        "proj_err": [],
        "run_time": [],
        "failed": [],
    }

    # Convert all tensors to numpy arrays.
    corresps_2d = torch_helpers.tensor_to_array(corresps_2d).astype(np.float64)
    corresps_3d = torch_helpers.tensor_to_array(corresps_3d).astype(np.float64)
    weights = torch_helpers.tensor_to_array(corresps_weight).astype(np.float64)
    visible_masks = torch_helpers.tensor_to_array(visible_masks).astype(np.bool_)
    intrinsics = torch_helpers.tensor_to_array(intrinsics).astype(np.float64)

    # Convert the initial poses to array.
    if init_poses is not None:
        init_rvecs = torch_helpers.tensor_to_array(init_rvecs)
        init_tvecs = torch_helpers.tensor_to_array(init_tvecs)

    # Define the PnP solver.
    solvePnPRansac = partial(
        cv2.solvePnPRansac,
        distCoeffs=None,
        iterationsCount=pnp_opts.pnp_ransac_iter,
        reprojectionError=pnp_opts.pnp_inlier_thresh,
        confidence=pnp_opts.pnp_required_ransac_conf,
    )

    # Iterate over the batch and estimate the poses for each sample.
    for i in range(batch_size):
        visible_mask = visible_masks[i]
        corresp_2d_ = corresps_2d[i][visible_mask]
        corresp_3d_ = corresps_3d[i][visible_mask]
        corresp_weight_ = weights[i][visible_mask]
        intrinsic = intrinsics[i]
        if init_poses is not None:
            init_rvecs_ = init_rvecs[i]  # pyre-ignore
            init_tvecs_ = init_tvecs[i]  # pyre-ignore
        else:
            init_rvecs_ = None
            init_tvecs_ = None

        if pnp_solver_name == "opencv":
            num_corresp_avail = np.sum(corresp_weight_ >= weight_threshold)
            pose_est_success = True
            if num_corresp_avail >= 6:
                # If there are too many correspondences, subsample them.
                selected_corresp_2d_ = corresp_2d_
                selected_corresp_3d_ = corresp_3d_
                if num_corresp_avail >= pnp_opts.max_num_corresps:
                    sampled_ids = np.random.choice(
                        len(corresp_2d_),
                        pnp_opts.max_num_corresps,
                        replace=False,
                    )
                    selected_corresp_2d_ = selected_corresp_2d_[sampled_ids]
                    selected_corresp_3d_ = selected_corresp_3d_[sampled_ids]

                # Run PnP solver.
                start_time = time.time()
                pose_est_success, rvec_est_m2c, t_est_m2c, idx_inliers = solvePnPRansac(
                    objectPoints=selected_corresp_3d_,
                    imagePoints=selected_corresp_2d_,
                    cameraMatrix=intrinsic,
                    rvec=init_rvecs_,
                    tvec=init_tvecs_,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                run_time = time.time() - start_time
                if pose_est_success:
                    num_inliers = len(idx_inliers)
                else:
                    num_inliers = 0
                # In some cases, the PnP solver may return a pose behind the camera.
                # In this case, re-run the PnP solver with EPnP.
                if t_est_m2c[2] < 0:
                    start_time = time.time()
                    (
                        pose_est_success,
                        rvec_est_m2c,
                        t_est_m2c,
                        idx_inliers,
                    ) = solvePnPRansac(
                        objectPoints=selected_corresp_3d_,
                        imagePoints=selected_corresp_2d_,
                        cameraMatrix=intrinsic,
                        rvec=init_rvecs_,
                        tvec=init_tvecs_,
                        flags=cv2.SOLVEPNP_EPNP,
                    )
                    run_time = time.time() - start_time
                    num_inliers = len(idx_inliers)

                # Optional LM refinement on inliers.
                if pose_est_success and num_inliers >= 6:
                    start_time = time.time()
                    rvec_est_m2c, t_est_m2c = cv2.solvePnPRefineLM(
                        objectPoints=selected_corresp_3d_[idx_inliers],
                        imagePoints=selected_corresp_2d_[idx_inliers],
                        cameraMatrix=intrinsic,
                        distCoeffs=None,
                        rvec=rvec_est_m2c,
                        tvec=t_est_m2c,
                    )
                    run_time += time.time() - start_time

                # Evaluate the output pose of PnP solver.
                # Attention: we evaluate all the correspondences, not just the subset used for PnP nor the inliers.
                pnp_stats = eval_pnp_output(
                    width=width,
                    height=height,
                    pnp_opts=pnp_opts,
                    corresp_3d=corresp_3d_,  # pyre-ignore
                    corresp_2d=corresp_2d_,  # pyre-ignore
                    corresp_weight=corresp_weight_,
                    intrinsic=intrinsic,  # pyre-ignore
                    t_est_m2c=t_est_m2c,
                    rvec_est_m2c=rvec_est_m2c,
                )
                for k, v in pnp_stats.items():
                    outputs[k].append(v)
                outputs["run_time"].append(run_time)
                outputs["failed"].append(False)

            if num_corresp_avail <= 5 or not pose_est_success:
                rvec_est_m2c, t_est_m2c = init_rvecs_, init_tvecs_  # pyre-ignore
                idx_inliers = []
                outputs["num_inliers"].append(0)
                outputs["is_behind_camera"].append(False)
                outputs["quality"].append(0.0)
                outputs["proj_err"].append(np.zeros((height, width, 3)))
                outputs["run_time"].append(0.0)
                outputs["failed"].append(not pose_est_success)

            # Convert the pose to 4x4 matrix.
            r_est_m2c = cv2.Rodrigues(rvec_est_m2c)[0]  #  pyre-ignore
            pose = transform3d.Rt_to_4x4_numpy(R=r_est_m2c, t=t_est_m2c.reshape(-1, 3))
            pose = torch.from_numpy(pose).float()
            outputs["estimated_poses"].append(pose)

        else:
            raise ValueError(f"Unknown PnP type {pnp_solver_name}")
    assert len(outputs["estimated_poses"]) == batch_size, (
        f"{len(outputs['estimated_poses'])} != {batch_size}"
    )
    for k in [
        "estimated_poses",
        "num_inliers",
        "is_behind_camera",
        "quality",
        "proj_err",
    ]:
        if len(outputs[k]):
            if isinstance(outputs[k][0], torch.Tensor):
                outputs[k] = torch.stack(outputs[k]).to(device)  # pyre-ignore
            else:
                outputs[k] = np.asarray(outputs[k])  # pyre-ignore
    outputs["run_time"] = np.sum(outputs["run_time"])
    outputs["failed"] = np.asarray(outputs["failed"])  # pyre-ignore
    return outputs  # pyre-ignore
