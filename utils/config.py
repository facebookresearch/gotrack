# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from enum import Enum
from typing import List, NamedTuple, Optional, Tuple


class FeatureOpts(NamedTuple):
    extractor_name: str


class TemplateDescOpts(NamedTuple):
    desc_type: str = "tfidf"

    # Options for tfidf template descriptor.
    tfidf_knn_metric: str = "l2"
    tfidf_knn_k: int = 3
    tfidf_soft_assign: bool = False
    tfidf_soft_sigma_squared: float = 10.0


class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    dataset_name: str
    object_lids: Optional[List[int]] = None

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    # Mesh pre-processing options.
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    # Rendering options.
    ssaa_factor: float = 4.0
    background_type: str = "gray"
    light_type: str = "multi_directional"

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True


class GenRepreOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    templates_version: str
    dataset_name: str
    object_lids: Optional[List[int]] = None

    # Feature extraction options.
    extractor_name: str = "dinov2_vits14_reg"
    grid_cell_size: float = 14.0

    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # Feature clustering options.
    cluster_features: bool = True
    cluster_num: int = 2048

    # Template descriptor options.
    template_desc_opts: Optional[TemplateDescOpts] = None

    # Other options.
    overwrite: bool = True
    debug: bool = True


class PnPOpts(NamedTuple):
    """PnP Options.
    Args:
        pnp_ransac_iter: number of iterations for RANSAC.
        pnp_inlier_thresh: inlier threshold for RANSAC.
        pnp_required_ransac_conf: required RANSAC confidence.
        sub_sample_correspondences: whether to sub-sample correspondences.
        max_num_corresps: maximum number of (sampled) correspondences to use.
    """

    pnp_ransac_iter: int = 3000
    pnp_inlier_thresh: float = 2.0
    pnp_required_ransac_conf: float = 0.999
    sub_sample_correspondences: bool = True
    max_num_corresps: int = 10000


class GoTrackRefinerLossOpts(NamedTuple):
    """Options for the losses."""

    distance_type: str = "l1"


class InitPoseType(Enum):
    """Type of initial poses used in the pose estimator.

    GT_POSES: Use GT poses.
    FROM_GT_BOXES: Calculate the initial poses from GT 2D bounding boxes.
    FROM_DETECTIONS: Calculate the initial poses from detected 2D bounding boxes.
    FROM_COARSE_POSES: Use poses predicted by a coarse pose estimator.
    """

    GT_POSES = "gt_poses"
    FROM_GT_BOXES = "from_gt_boxes"
    FROM_DETECTIONS = "from_detections"
    FROM_COARSE_POSES = "from_coarse_poses"


class InitPoseOpts(NamedTuple):
    """Options for initial poses during training.

    pose_type: The type of initial poses.
    noise_name: The name of the reference method for generate noises (CosyPose / FoundationPose).
    noise_rotation_std: The standard deviation of noise to add to the rotation (exponential map) in degrees.
    noise_translation_std: The standard deviation of noise to add to the XYZ translation in mm.
    References of std in papers:
    - CosyPose/MegaPose: https://arxiv.org/pdf/2212.06870 Page 14
    - FoundationPose: https://arxiv.org/pdf/2312.08344 Page 14
    """

    pose_type: InitPoseType = InitPoseType.GT_POSES
    noise_name: str = "cosypose"
    noise_rotation_std: List[float] = [15.0, 15.0, 15.0]
    noise_translation_std: List[float] = [15.0, 15.0, 15.0]


class PoseLossOpts(NamedTuple):
    """Disentangled pose loss options for training, defined in same way as CosyPose/MegaPose."""

    loss_disentangled: bool = True
    loss_disentangled_rot_weight: float = 0.1
    loss_disentangled_xy_weight: float = 0.1
    loss_disentangled_z_weight: float = 0.1
    loss_add_weight: float = 1.0
    num_model_vertices: int = 500


class PoseRepre(Enum):
    """Rigid pose representations.

    XYZ_CONT6D: 3D translation and continuous 6D representation for rotation
        from "On the continuity of rotation representations in neural networks"
        by Zhou et al.
    XYZ_QUAT: 3D translation and 4D quaternion.
    """

    XYZ_CONT6D = "xyz_cont6d"
    XYZ_QUAT = "xyz_quat"
