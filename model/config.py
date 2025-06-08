# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import NamedTuple, Optional, Tuple
from utils.config import (
    GoTrackRefinerLossOpts,
    InitPoseOpts,
    PnPOpts,
)
from model.blocks.config import (
    DecoderOpts,
)
from model.heads.dpt.config import (
    DPTHeadOpts,
)


class FastSAMOpts(NamedTuple):
    """Options for the FastSAM model.

    model_path: Path to the model.
    iou_threshold: IoU threshold.
    conf_threshold: Confidence threshold.
    max_det: Maximum number of detections.
    im_width_size: Image width size.
    verbose: Whether to print verbose output.
    """

    model_path: str
    iou_threshold: float = 0.9
    conf_threshold: float = 0.05
    max_det: int = 200
    im_width_size: int = 640
    verbose: bool = True


class CNOSOpts(NamedTuple):
    """Options for the CNOSFastSAM model.

    aggregation_function: Function to aggregate the matching scores.
    min_box_size: (Post processing for detection stage) Minimum box size.
    min_mask_size: (Post processing for detection stage) Minimum mask size.
    """

    aggregation_function: str = "avg5"
    matching_model_name: str = "dinov2_vitl14"
    crop_rel_pad: float = 0.0
    crop_size: Tuple[int, int] = (280, 280)
    min_box_size: float = 0.05  # relative to image size
    min_mask_size: float = 3e-4  # relative to image size
    debug: bool = True


class FoundPoseOpts(NamedTuple):
    """Options for the FoundPose model.

    aggregation_function: Function to aggregate the matching scores.
    """

    # Whether running only retrieval and use template pose as predicted pose
    run_retrieval_only: bool = False

    crop_size: Tuple[int, int] = (280, 280)
    crop_rel_pad: float = 0.2

    # Feature extraction options.
    extractor_name: str = "dinov2_vits14-reg"
    grid_cell_size: float = 1.0
    max_num_queries: int = 1000000

    # Feature matching options.
    match_template_type: str = "tfidf"
    match_top_n_templates: int = 5
    match_feat_matching_type: str = "cyclic_buddies"
    match_top_k_buddies: int = 300

    # PnP options.
    pnp_type: str = "opencv"
    pnp_ransac_iter: int = 1000
    pnp_required_ransac_conf: float = 0.99
    pnp_inlier_thresh: float = 10.0
    pnp_refine_lm: bool = True
    final_pose_type: str = "best_coarse"
    vis_feat_map: bool = True
    vis_corresp_top_n: int = 100
    debug: bool = True


class GoTrackOpts(NamedTuple):
    """Options for the Flow-based refiner.
    These options are generic and can be used for any refiner.
    For each refiner, there are additional specific options, such as backbone, head, etc.

    head_opts: Options for the head.
    decoder_opts: Options for the decoder.
    loss_opts: Options for the losses.
    backbone_name: Name of the backbone network.
    frozen_backbone: Whether to freeze the backbone.

    num_iterations_train: Number of refinement iterations during training.
    num_iterations_test: Number of refinement iterations during testing.

    data_prepocessing_opts: Options for data processing.

    crop_size: Size of the input image.
    background_type: Type of the background (e.g., "black", etc).
    re_crop_every_iter: Whether to re-crop the image every iteration.

    pnp_solver_name: Name of the PnP solver to be used.
    name_prefix: for naming the refiner loss.

    vis_minimal: Whether to generate only a minimal visualization.
    vis_only_last_iter: Whether to visualize only the last iteration.
    """

    head_opts: DPTHeadOpts = DPTHeadOpts()
    decoder_opts: DecoderOpts = DecoderOpts()
    loss_opts: GoTrackRefinerLossOpts = GoTrackRefinerLossOpts()
    backbone_name: str = "dinov2_vits14-reg"
    frozen_backbone: bool = True

    num_iterations_train: int = 1
    num_iterations_test: int = 5

    # whether process inputs when running the model
    process_inputs: bool = True

    # Parameters for data processing.
    init_pose_opts: InitPoseOpts = InitPoseOpts()
    crop_size: Tuple[int, int] = (280, 280)
    crop_rel_pad: float = 0.1
    cropping_type: str = "perspective_2d_box"
    background_type: str = "gray"
    ssaa_factor: float = 1.0

    re_crop_every_iter: bool = True

    visib_threshold: float = 0.3
    pnp_solver_name: str = "opencv"
    pnp_opts: Optional[PnPOpts] = None
    name_prefix: str = ""

    vis_minimal: bool = False
    vis_only_last_iter: bool = True

    debug: bool = True
