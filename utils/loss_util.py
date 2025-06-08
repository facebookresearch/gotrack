# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-unsafe

"""Losses for GoTrack refiner."""

from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from utils import logging, structs, transform3d, config
import torch.nn.functional as F

logger = logging.get_logger(__name__)
binary_cross_entropy_loss = nn.BCELoss()


def flow_regression_loss(
    pred_flows: torch.Tensor,
    gt_flows: torch.Tensor,
    visib_masks: torch.Tensor,
    loss_opts: config.GoTrackRefinerLossOpts,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Loss function for flow regression.
    # Source: https://github.com/cvlab-epfl/perspective-flow-aggregation/blob/main/flow/train.py#L60
    Args:
        pred_flows: Predicted flows of shape [batch_size, 2, H, W].
        gt_flows: GT flows of shape [batch_size, 2, H, W].
        visib_masks: Valid masks of shape [batch_size, H, W]. A valid mask contains pixels of the source image that are visible in the target image.
        loss_opts: Options for the loss to decide L1 or L2 loss.
        max_flow: Maximum flow value. We ignore pixels having GT flows larger than this value.
    Returns:
        losses: torch.Tensor for backward.
        metrics: Dict[str, torch.Tensor] for visualizing/debugging (not used for backward).
    """
    assert torch.sum(torch.isnan(pred_flows)) == 0
    assert torch.sum(torch.isinf(pred_flows)) == 0

    metrics = {}

    # Get the maximum flow value, then we ignore pixels having GT flows larger than this value.
    height, width = gt_flows.shape[-2:]
    # PFA uses 400px threshold for 256x256 images.
    # https://github.com/cvlab-epfl/perspective-flow-aggregation/blob/main/flow/train.py#L55
    max_flow = max(height, width) * (400 / 256)

    # exlude invalid pixels and extremely large diplacements
    norm_gt_flows = torch.norm(gt_flows, dim=1)
    valid = visib_masks & (norm_gt_flows < max_flow)  # [batch_size, H, W]

    # Compute the loss.
    if loss_opts.distance_type == "l2":
        loss = (pred_flows - gt_flows).pow(2).sum(1).sqrt()
    elif loss_opts.distance_type == "l1":
        loss = (pred_flows - gt_flows).abs().sum(1)
    else:
        raise ValueError(f"Unknown distance type ({loss_opts.distance_type}).")
    # Mask out invalid pixels in the losses.
    loss = loss.view(-1)[valid.view(-1)]
    loss = loss.mean()

    # Calculate end-point-error (EPE).
    epe = torch.sum((pred_flows - gt_flows).pow(2), dim=1).sqrt()

    # Mask out invalid pixels in the metrics.
    epe = epe.view(-1)[valid.view(-1)]

    # Calculate the accuracies with 1px, 3px, 5px thresholds.
    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }
    return loss, metrics  # pyre-ignore


def compute_flow_losses_and_metrics(
    num_iters: int,
    name_prefix: str,
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
    loss_opts: config.GoTrackRefinerLossOpts,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Computes losses for flow regression.
    Args:
        num_iters: Number of iterations.
        name_prefix: Prefix for the loss names.
        predictions: Predictions from the model.
            - "flows": Predicted flows of shape [batch_size, 2, H, W].
        gt_objects: Ground truth objects.
            - "flows": Correspondences of shape [batch_size, 2, H, W].
            - "visib_masks": Valid masks of shape [batch_size, H, W]. Valid masks contains pixels of source masks that visible in the target view.
    Return:
        Losses for backward.
        Metrics for debugging (not used for backward).
    """
    losses = {}
    metrics = {}
    loss_name_tpl = name_prefix + "{name}_loss_iter={iter_idx:01d}"
    metric_name_tpl = name_prefix + "{name}_metric_iter={iter_idx:01d}"

    # Get GT flows and GT valid masks.
    gt_flows = gt_objects.flows.cuda()
    visib_masks = gt_objects.visib_masks.cuda().bool()
    assert gt_flows.shape[1] == 2

    # Get predicted flows.
    pred_flows = predictions["flows"]
    assert pred_flows.shape[1] == 2
    assert gt_flows.shape == pred_flows.shape

    # Calculate the losses and metrics.
    loss, flow_metrics = flow_regression_loss(
        pred_flows=pred_flows,
        gt_flows=gt_flows,
        visib_masks=visib_masks,
        loss_opts=loss_opts,
    )

    # Update the losses and metrics.
    loss_name = loss_name_tpl.format(iter_idx=0, name="sequence_flow")
    losses[loss_name] = loss

    for metric_name, metric_value in flow_metrics.items():
        metric_name = metric_name_tpl.format(iter_idx=0, name=metric_name)
        metrics[metric_name] = metric_value
    return losses, metrics


def compute_mask_losses_and_metrics(
    num_iters: int,
    name_prefix: str,
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
    loss_opts: config.GoTrackRefinerLossOpts,
    binary_threshold: float = 0.5,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Computes losses for mask/confidence score regression.
    Args:
        num_iters: Number of iterations.
        name_prefix: Prefix for the loss names.
        predictions: Predictions from the model.
            - "confidences": Predicted confidences of shape [batch_size, H, W]. 0 mean not visible, 1 mean visible.
        gt_objects: Ground truth objects.
            - "visib_masks": Valid masks of shape [batch_size, H, W]. Valid masks contain pixels of source images that are visible in target images.
            - "masks": Masks of shape [batch_size, H, W]. Masks contains pixels of source images.
    Return:
        Losses for backward.
        Metrics for debugging (not used for backward).
    """
    losses = {}
    metrics = {}
    loss_name_tpl = name_prefix + "{name}_loss_iter={iter_idx:01d}"
    metric_name_tpl = name_prefix + "{name}_metric_iter={iter_idx:01d}"

    # Get GT valid masks.
    gt_visib_masks = gt_objects.visib_masks.cuda().float()

    # Get GT masks (!= valid masks).
    gt_masks = gt_objects.masks.cuda()

    # Get predicted confidences.
    pred_conf = predictions["confidences"]

    # Binary cross entropy loss has log(prediction) iterm which can lead to NaNs when prediction is 0.
    # Discussion: https://stackoverflow.com/questions/37044600/sudden-drop-in-accuracy-while-training-a-deep-neural-net
    # So we clamp the prediction to be at least 1e-6 to avoid NaNs.
    pred_conf = torch.clamp(pred_conf, min=1e-6)

    # Compute binary cross entropy loss (for only pixels inside template masks).
    loss = binary_cross_entropy_loss(
        pred_conf[gt_masks],
        gt_visib_masks[gt_masks],
    )

    # Compute accuracy.
    predicted_mask = (pred_conf > binary_threshold).float()
    accuracy = (predicted_mask == gt_visib_masks)[gt_masks]
    accuracy = accuracy.float().mean()

    loss_name = loss_name_tpl.format(iter_idx=0, name="binary_mask")
    losses[loss_name] = loss
    metric_name = metric_name_tpl.format(iter_idx=0, name="accuracy")
    metrics[metric_name] = accuracy
    return losses, metrics


def symmetric_pose_loss(
    poses_possible_gt: torch.Tensor, poses_pred: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """Symmetry-aware mean L1 distance between vertices in two poses.

    Args:
        poses_possible_gt: Possible GT poses of shape [num_insts, max_gt_poses,
            4, 4]. There may be more possible GT poses due to symmetries. For
            objects with less than max_gt_poses possible GT poses, the tensor is
            assumed to be filled up with one of the possible GT poses.
        poses_pred: Predicted poses of shape [num_insts, 4, 4].
        vertices: 3D model vertices of shape [num_insts, num_vertices, 3].
    Returns:
        The calculated loss value.
    """

    # Vertices in all possible GT poses.
    vertices_possible_gt = transform3d.transform_points(poses_possible_gt, vertices)

    # Vertices in the predicted pose.
    vertices_pred = transform3d.transform_points(poses_pred, vertices)

    # Mean L1 distance between the vertices in the predicted pose and vertices
    # in each of the possible GT poses.
    losses_possible = (
        (vertices_pred.unsqueeze(1) - vertices_possible_gt)
        .flatten(-2, -1)
        .abs()
        .mean(-1)
    )

    # Take the minimum mean distance as the loss.
    loss, _ = losses_possible.min(dim=1)

    # loss *= 0.001  # From mm to m (as in the original CosyPose implementation).

    return loss


def symmetric_disentangled_pose_loss(
    poses_possible_gt: torch.Tensor,
    poses_input: torch.Tensor,
    poses_pred_delta: torch.Tensor,
    delta_pose_repre: config.PoseRepre,
    cameras: List[structs.CameraModel],
    vertices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Disentangled symmetry-aware mean L1 distance between vertices in two poses.

    For details, see page 21 in: https://arxiv.org/pdf/2008.08465

    Args:
        poses_possible_gt: Possible GT poses of shape [num_insts, max_gt_poses,
            4, 4]. See the doc of `loss_symmetric` for more details.
        poses_input: Input poses of shape [num_insts, 4, 4].
        poses_pred_delta: Predicted delta poses of shape [num_poses, pose_repre_size].
        delta_pose_repre: The used representation of the delta poses.
        cameras: The cameras used for input rendered images.
        vertices: 3D model vertices of shape [num_insts, num_vertices, 3].
    Returns:
        A dictionary with disentangled loss terms w.r.t. the rotation ("rot"),
        XY translation ("xy"), and Z translation ("z").
    """

    bsz = poses_possible_gt.shape[0]
    assert poses_possible_gt.device == poses_input.device == poses_pred_delta.device
    assert poses_possible_gt.shape[0] == bsz
    assert poses_input.shape[0] == bsz
    assert vertices.dim() == 3 and vertices.shape[0] == bsz and vertices.shape[-1] == 3
    assert poses_possible_gt.dim() == 4 and poses_possible_gt.shape[-2:] == (4, 4)

    if delta_pose_repre.value == config.PoseRepre.XYZ_CONT6D.value:
        # The first possible GT pose is considered the "true" GT.
        poses_gt = poses_possible_gt[:, 0]
        z_gt = poses_gt[:, 2, [3]]

        # GT translation, predicted rotation.
        R_delta = transform3d.rotation_matrix_from_cont6d(poses_pred_delta[:, :6])
        poses_pred_rot = poses_gt.clone()
        poses_pred_rot[:, :3, :3] = R_delta @ poses_input[:, :3, :3]
        loss_rot = symmetric_pose_loss(poses_possible_gt, poses_pred_rot, vertices)

        # GT rotation and Z translation, predicted XY translation.
        xy_delta = poses_pred_delta[:, 6:8]
        xy_in = poses_input[:, :2, 3]
        z_in = poses_input[:, 2, [3]]
        fxfy = torch.as_tensor([camera.f for camera in cameras])
        fxfy = fxfy.to(poses_gt.device)
        poses_pred_xy = poses_gt.clone()
        poses_pred_xy[:, :2, 3] = (
            (xy_delta / fxfy) + (xy_in / z_in.repeat(1, 2))
        ) * z_gt.repeat(1, 2)
        loss_xy = symmetric_pose_loss(poses_possible_gt, poses_pred_xy, vertices)

        # GT rotation and XY translation, predicted Z translation.
        z_delta = poses_pred_delta[:, [8]]
        poses_pred_z = poses_gt.clone()
        poses_pred_z[:, [2], [3]] = z_delta * z_in
        loss_z = symmetric_pose_loss(poses_possible_gt, poses_pred_z, vertices)

    else:
        raise ValueError("Unknown pose representation.")

    return {"rot": loss_rot, "xy": loss_xy, "z": loss_z}


def get_symmetric_poses_tensor(
    poses_cam_from_obj: torch.Tensor,
    symmetries_obj_from_obj: List[torch.Tensor],
) -> torch.Tensor:
    """Returns all symmetric versions of the provided poses.

    Args:
        poses_cam_from_obj: Poses of shape [batch_size, 4, 4].
        symmetries_obj_from_obj: List of batch_size tensors of shape [num_sym, 4, 4] that stores
            symmetry transformations. symmetries_obj_from_obj[i] are symmetry transformations of
            an object instance whose pose is in poses_cam_from_obj[i]. Note that each object
            may have a different number of symmetry transformations (i.e. num_sym may be
            different for each object).
    Returns:
        Symmetric poses of shape [batch_size, max_sym_count, 4, 4], where max_sym_count is
        the maximum number of symmetry transformations that appears in the batch. For
        objects that have fewer symmetry transformations, the remaining slots are filled
        with the original poses.
    """

    # Get symmetry transformations in a tensor of shape (batch_size, max_sym_count, 4, 4)
    # padded with identities for objects that have less than max_sym_count symmetries.
    num_objects = len(symmetries_obj_from_obj)
    max_sym_count = max([len(sym) for sym in symmetries_obj_from_obj])
    symmetries_obj_from_obj_tensor = torch.tile(
        torch.eye((4)), [num_objects, max_sym_count, 1, 1]
    )
    for inst_id, syms in enumerate(symmetries_obj_from_obj):
        symmetries_obj_from_obj_tensor[inst_id, : len(syms)] = syms

    # Collect all possible GT object poses by applying the symmetry
    # transformations to the actual GT object poses.
    device = poses_cam_from_obj.unsqueeze(1).device
    symmetric_poses_cam_from_obj = poses_cam_from_obj.unsqueeze(
        1
    ) @ symmetries_obj_from_obj_tensor.to(device)

    return symmetric_poses_cam_from_obj


def compute_losses(
    pose_loss_opts: config.PoseLossOpts,
    pose_repre: config.PoseRepre,
    num_iters: int,
    name_prefix: str,
    object_symmetries: Dict[int, Any],
    predictions: Dict[str, Any],
    gt_objects: structs.Collection,
) -> Dict[str, torch.Tensor]:
    """Computes training losses (see refiner_base.py)."""

    # Collect all possible GT object poses by applying the symmetry
    # transformations to the actual GT object poses.
    losses = {}
    loss_name_tpl = name_prefix + "_{name}_loss_iter={iter_idx:03d}"

    for iter_idx in range(num_iters):
        prefix = f"iter={iter_idx}"

        # Each iteration uses different cropping, so have different GT poses.
        Ts_crop_cam_from_orig_cam = predictions[f"{prefix}_Ts_crop_cam_from_orig_cam"]
        gt_poses_crop_cam_from_model = (
            Ts_crop_cam_from_orig_cam @ gt_objects.poses_cam_from_model
        )

        poses_possible_gt_cam_from_obj = get_symmetric_poses_tensor(
            poses_cam_from_obj=gt_poses_crop_cam_from_model,
            symmetries_obj_from_obj=[
                object_symmetries[label.item()] for label in gt_objects.labels
            ],
        )
        poses_possible_gt_cam_from_obj = poses_possible_gt_cam_from_obj.to(
            gt_objects.target_vertices.device
        )

        # Calculate the disentangled loss terms from CosyPose.
        losses_curr = symmetric_disentangled_pose_loss(
            poses_possible_gt=poses_possible_gt_cam_from_obj,
            poses_input=predictions[f"{prefix}_init_Ts_crop_cam_from_model"],
            poses_pred_delta=predictions[f"{prefix}_delta_poses_in_crop_cam"],
            delta_pose_repre=pose_repre,
            cameras=predictions[f"{prefix}_crop_cameras"],
            vertices=gt_objects.target_vertices,
        )

        # Calculate the final loss as a linear combination of the loss terms.
        for loss_name, loss_values in losses_curr.items():
            if loss_name == "rot":
                weight = pose_loss_opts.loss_disentangled_rot_weight
            elif loss_name == "xy":
                weight = pose_loss_opts.loss_disentangled_xy_weight
            elif loss_name == "z":
                weight = pose_loss_opts.loss_disentangled_z_weight
            else:
                raise ValueError(f"Unknown loss ({loss_name}).")
            loss_name = loss_name_tpl.format(iter_idx=iter_idx, name=loss_name)
            losses[loss_name] = loss_values.mean() * weight
    return losses


def calculate_delta_pose_in_cam(
    pose_repre: config.PoseRepre,
    source_Ts_cam_from_model: torch.Tensor,
    target_Ts_cam_from_model: torch.Tensor,
    cameras: List[structs.CameraModel],
) -> torch.Tensor:
    """Calculates the delta pose from the source poses to target poses under camera coordinate frame.

    Args:
        pose_repre: Pose representation.
        Ts_cam_from_model: Original poses of shape [batch_size, 4, 4].
        Ts_delta: Delta poses of shape [batch_size, 4, 4].
    Returns:
        Delta poses of shape [batch_size, pose_repre_size].
    """
    batch_size = source_Ts_cam_from_model.shape[0]
    device = source_Ts_cam_from_model.device

    if pose_repre == config.PoseRepre.XYZ_CONT6D:
        delta_poses_in_cam = torch.zeros(
            (batch_size, 9), dtype=torch.float32, device=device
        )

        # Calculate the delta pose in the camera space.
        R_delta = target_Ts_cam_from_model[:, :3, :3] @ source_Ts_cam_from_model[
            :, :3, :3
        ].permute(0, 2, 1)

        # Convert the delta pose to the continuous 6D representation.
        delta_poses_in_cam[:, :6] = transform3d.cont6d_from_rotation_matrix(R_delta)
        R_delta_recovered = transform3d.rotation_matrix_from_cont6d(
            delta_poses_in_cam[:, :6]
        )
        assert torch.allclose(R_delta, R_delta_recovered, atol=1e-4)

        # Delta Z is relative scale change.
        assert torch.sum(source_Ts_cam_from_model[:, 2, 3] == 0) == 0
        delta_poses_in_cam[:, 8] = (
            target_Ts_cam_from_model[:, 2, 3] / source_Ts_cam_from_model[:, 2, 3]
        )

        # Delta XY is relative displacement in pixels.
        fxfy = torch.as_tensor([camera.f for camera in cameras])
        fxfy = fxfy.to(device)

        # Normalize to get homogenuous coordinates.
        source_reproj = (
            source_Ts_cam_from_model[:, :2, 3] / source_Ts_cam_from_model[:, 2, 3:]
        )
        target_reproj = (
            target_Ts_cam_from_model[:, :2, 3] / target_Ts_cam_from_model[:, 2, 3:]
        )
        delta_poses_in_cam[:, 6:8] = (target_reproj - source_reproj) * fxfy

    else:
        raise ValueError(f"Not supported pose representation: {pose_repre}")
    return delta_poses_in_cam


class PairwiseSimilarity(nn.Module):
    """Pairwise similarity module.
    Source: https://github.com/nv-nguyen/cnos/blob/main/src/model/loss.py#L21"""

    def __init__(self, metric="cosine", chunk_size=64):
        super(PairwiseSimilarity, self).__init__()
        self.metric = metric
        self.chunk_size = chunk_size

    def forward(self, query_descriptors, reference_descriptors):
        N_query = query_descriptors.shape[0]
        N_objects, N_templates = (
            reference_descriptors.shape[0],
            reference_descriptors.shape[1],
        )
        reference_descriptors_ = (
            reference_descriptors.clone().unsqueeze(0).repeat(N_query, 1, 1, 1)
        )
        query_descriptors_ = (
            query_descriptors.clone().unsqueeze(1).repeat(1, N_templates, 1)
        )
        query_descriptors_ = F.normalize(query_descriptors_, dim=-1)
        reference_descriptors_ = F.normalize(reference_descriptors_, dim=-1)

        similarity = structs.BatchedData(batch_size=None)
        for idx_obj in range(N_objects):
            sim = F.cosine_similarity(
                query_descriptors_, reference_descriptors_[:, idx_obj], dim=-1
            )  # N_query x N_templates
            similarity.append(sim)
        similarity.stack()
        similarity = similarity.data
        similarity = similarity.permute(1, 0, 2)  # N_query x N_objects x N_templates
        return similarity.clamp(min=0.0, max=1.0)
