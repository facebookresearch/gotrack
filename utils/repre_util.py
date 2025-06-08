# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import os

from dataclasses import dataclass, field
from bop_toolkit_lib import inout
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from utils import (
    json_util,
    misc,
    pca_util,
    logging,
    structs,
    torch_helpers,
    config,
    feature_util,
)

logger = logging.get_logger(__name__)


@dataclass
class FeatureBasedObjectRepre:
    """Stores visual object features registered in 3D."""

    # 3D vertices of the object model of shape (num_vertices, 3).
    vertices: Optional[torch.Tensor] = None

    # 3D vertex normals of shape (num_vertices, 3).
    vertex_normals: Optional[torch.Tensor] = None

    # Class feature vectors of shape (feat_dims).
    cls_feat_vectors: Optional[torch.Tensor] = None

    # Feature vectors of shape (num_features, feat_dims).
    feat_vectors: Optional[torch.Tensor] = None

    # Feature options.
    feat_opts: Optional[config.FeatureOpts] = None

    # Mapping from feature to associated vertex ID of shape (num_features).
    feat_to_vertex_ids: Optional[torch.Tensor] = None

    # Mapping from feature to source template ID of shape (num_features).
    feat_to_template_ids: Optional[torch.Tensor] = None

    # Mapping from feature to assigned feature ID of shape (num_features).
    feat_to_cluster_ids: Optional[torch.Tensor] = None

    # Centroids of feature clusters of shape (num_clusters, feat_dims).
    feat_cluster_centroids: Optional[torch.Tensor] = None

    # Inverse document frequency (for tfidf template descriptors) of shape (num_clusters).
    # Ref: https://www.di.ens.fr/~josef/publications/torii13.pdf
    feat_cluster_idfs: Optional[torch.Tensor] = None

    # Projectors of raw extracted features to features saved in `self.feat_vectors`.
    feat_raw_projectors: List[pca_util.Projector] = field(default_factory=list)

    # Projectors for visualizing features from `self.feat_vectors` (typically a PCA projector).
    feat_vis_projectors: List[pca_util.Projector] = field(default_factory=list)

    # Templates of shape (num_templates, channels, height, width).
    templates: Optional[torch.Tensor] = None

    # Template masks of shape (num_templates, height, width).
    template_masks: Optional[torch.Tensor] = None

    # Template depth of shape (num_templates, height, width).
    template_depths: Optional[torch.Tensor] = None

    # Per-template camera with extrinsics expressed as a transformation from the model space
    # to the camera space.
    template_cameras_cam_from_model: List[structs.PinholePlaneCameraModel] = field(
        default_factory=list
    )

    # Template descriptors of shape (num_templates, desc_dims).
    template_descs: Optional[torch.Tensor] = None

    # Configuration of template descriptors.
    template_desc_opts: Optional[config.TemplateDescOpts] = None


def get_object_repre_dir_path(
    base_dir: str, repre_type: str, dataset: str, lid: int
) -> str:
    """Get a path to a directory where a representation of the specified object is stored."""

    return os.path.join(
        base_dir,
        repre_type,
        dataset,
        str(lid),
    )


def save_object_repre(
    repre: FeatureBasedObjectRepre,
    repre_dir: str,
) -> None:
    # Save the object into torch data.
    if not isinstance(repre.template_desc_opts, config.TemplateDescOpts):
        repre.template_desc_opts = config.TemplateDescOpts(repre.template_desc_opts)

    object_dict = {}
    for key, value in repre.__dict__.items():
        if value is not None and torch.is_tensor(value):
            object_dict[key] = value

    # Save camera metadata.
    object_dict["template_cameras_cam_from_model"] = []
    for camera in repre.template_cameras_cam_from_model:
        cam_data = {
            "f": torch.tensor(camera.f),
            "c": torch.tensor(camera.c),
            "width": camera.width,
            "height": camera.height,
            "T_world_from_eye": torch.tensor(camera.T_world_from_eye),
        }
        object_dict["template_cameras_cam_from_model"].append(cam_data)

    object_dict["feat_opts"] = repre.feat_opts._asdict()
    object_dict["template_desc_opts"] = repre.template_desc_opts._asdict()

    object_dict["feat_raw_projectors"] = []
    for projector in repre.feat_raw_projectors:
        object_dict["feat_raw_projectors"].append(
            pca_util.projector_to_tensordict(projector)
        )

    object_dict["feat_vis_projectors"] = []
    for projector in repre.feat_vis_projectors:
        object_dict["feat_vis_projectors"].append(
            pca_util.projector_to_tensordict(projector)
        )

    # Save the dictionary of tensors to the file
    repre_path = os.path.join(repre_dir, "repre.pth")
    logger.info(f"Saving repre to: {repre_path}")

    torch.save(object_dict, repre_path)


def load_object_repre(
    repre_dir: str,
    tensor_device: str = "cuda",
    load_fields: Optional[List[str]] = None,
) -> FeatureBasedObjectRepre:
    """Loads a representation of the specified object."""

    repre_path = os.path.join(repre_dir, "repre.pth")
    logger.info(f"Loading repre from: {repre_path}")
    object_dict = torch.load(repre_path)
    logger.info("Repre loaded.")

    repre_dict: Dict[str, Any] = {}

    for key, value in object_dict.items():
        if value is not None and (isinstance(value, torch.Tensor)):
            repre_dict[key] = value

    if object_dict["feat_opts"] is not None and (
        load_fields is None or "feat_opts" in load_fields
    ):
        repre_dict["feat_opts"] = config.FeatureOpts(**dict(object_dict["feat_opts"]))

    repre_dict["feat_raw_projectors"] = []
    if load_fields is None or "feat_raw_projectors" in load_fields:
        for projector in object_dict["feat_raw_projectors"]:
            repre_dict["feat_raw_projectors"].append(
                pca_util.projector_from_tensordict(projector)
            )

    repre_dict["feat_vis_projectors"] = []
    if load_fields is None or "feat_vis_projectors" in load_fields:
        for projector in object_dict["feat_vis_projectors"]:
            repre_dict["feat_vis_projectors"].append(
                pca_util.projector_from_tensordict(projector)
            )

    repre_dict["template_cameras_cam_from_model"] = []
    if load_fields is None or "template_cameras_cam_from_model" in load_fields:
        for camera in object_dict["template_cameras_cam_from_model"]:
            repre_dict["template_cameras_cam_from_model"].append(
                structs.PinholePlaneCameraModel(
                    f=camera["f"],  ## needs conversion
                    c=camera["c"],  ## needs conversion
                    width=camera["width"],
                    height=camera["height"],
                    T_world_from_eye=camera["T_world_from_eye"],  ## needs conversion
                )
            )

    if load_fields is None or "template_desc_opts" in load_fields:
        if object_dict["template_desc_opts"] is not None:
            repre_dict["template_desc_opts"] = config.TemplateDescOpts(
                **dict(object_dict["template_desc_opts"])
            )

    # Convert to the corresponding Python structure.
    repre = FeatureBasedObjectRepre(**repre_dict)
    return repre


def convert_object_repre_to_numpy(
    repre: FeatureBasedObjectRepre,
) -> FeatureBasedObjectRepre:
    repre_out = FeatureBasedObjectRepre()
    for name, value in repre.__dict__.items():
        if value is not None and isinstance(value, torch.Tensor):
            value = torch_helpers.tensor_to_array(value)
        setattr(repre_out, name, value)

    return repre_out


def generate_raw_repre(
    bop_root_dir: Path,
    opts: config.GenRepreOpts,
    dataset_name: str,
    object_lid: int,
    extractor: torch.nn.Module,
    output_dir: str,
    device: str = "cuda",
    debug: bool = False,
) -> FeatureBasedObjectRepre:
    # Prepare a timer.
    timer = misc.Timer(enabled=debug)

    # Load the template metadata.
    metadata_path = os.path.join(
        bop_root_dir,
        "templates",
        opts.templates_version,
        opts.dataset_name,
        str(object_lid),
        "metadata.json",
    )
    metadata = json_util.load_json(metadata_path)

    # Prepare structures for storing data.
    cls_feat_vector_list = []
    feat_vectors_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_masks_list = []
    template_depths_list = []
    template_cameras_cam_from_model_list = []

    # Use template images specified in the metadata.
    template_id = 0
    num_templates = len(metadata)
    for data_id, data_sample in enumerate(metadata):
        logger.info(f"Processing dataset {data_id}/{num_templates}, ")

        timer.start()

        camera_sample = data_sample["cameras"]
        camera_world_from_cam = structs.PinholePlaneCameraModel(
            width=camera_sample["ImageSizeX"],
            height=camera_sample["ImageSizeY"],
            f=(camera_sample["fx"], camera_sample["fy"]),
            c=(camera_sample["cx"], camera_sample["cy"]),
            T_world_from_eye=np.array(camera_sample["T_WorldFromCamera"]),
        )
        # RGB/monochrome and depth images (in mm).
        image_path = data_sample["rgb_image_path"]
        depth_path = data_sample["depth_map_path"]
        mask_path = data_sample["binary_mask_path"]

        image_arr = inout.load_im(image_path)  # H,W,C
        depth_image_arr = inout.load_depth(depth_path)
        mask_image_arr = inout.load_im(mask_path)

        image_chw = (
            torch_helpers.array_to_tensor(image_arr)
            .to(torch.float32)
            .permute(2, 0, 1)
            .to(device)
            / 255.0
        )
        depth_image_hw = (
            torch_helpers.array_to_tensor(depth_image_arr).to(torch.float32).to(device)
        )
        object_mask_modal = (
            torch_helpers.array_to_tensor(mask_image_arr).to(torch.float32).to(device)
        )

        # Get the object annotation.
        assert data_sample["dataset"] == dataset_name
        assert data_sample["lid"] == object_lid
        assert data_sample["template_id"] == data_id

        object_pose = data_sample["pose"]

        # Transformations.
        object_pose_rigid_matrix = np.eye(4)
        object_pose_rigid_matrix[:3, :3] = object_pose["R"]
        object_pose_rigid_matrix[:3, 3:] = object_pose["t"]
        T_world_from_model = (
            torch_helpers.array_to_tensor(object_pose_rigid_matrix)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_world = torch.linalg.inv(T_world_from_model)
        T_world_from_camera = (
            torch_helpers.array_to_tensor(camera_world_from_cam.T_world_from_eye)
            .to(torch.float32)
            .to(device)
        )
        T_model_from_camera = torch.matmul(T_model_from_world, T_world_from_camera)

        timer.elapsed("Time for getting template data")
        timer.start()

        # Extract features from the current template.
        (
            cls_feat_vector,
            feat_vectors,
            feat_to_vertex_ids,
            vertices_in_model,
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera_world_from_cam,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=False,
        )
        timer.elapsed("Time for feature extraction")
        timer.start()

        # Store data.
        cls_feat_vector_list.append(cls_feat_vector)
        feat_vectors_list.append(feat_vectors)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids = template_id * torch.ones(
            feat_vectors.shape[0], dtype=torch.int32, device=device
        )
        feat_to_template_ids_list.append(feat_to_template_ids)

        # Save the template as uint8 to save space.
        image_chw_uint8 = (image_chw * 255).to(torch.uint8)
        templates_list.append(image_chw_uint8)
        template_masks_list.append(object_mask_modal)
        template_depths_list.append(depth_image_hw)
        # Store camera model of the current template.
        camera_model = camera_world_from_cam.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

        # Increment the template ID.
        template_id += 1

        timer.elapsed("Time for storing data")

    logger.info("Processing done.")
    # Build the object representation from the collected data.
    return FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        cls_feat_vectors=torch.stack(cls_feat_vector_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_opts=config.FeatureOpts(extractor_name=opts.extractor_name),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_masks=torch.stack(template_masks_list),
        template_depths=torch.stack(template_depths_list),
        template_cameras_cam_from_model=template_cameras_cam_from_model_list,
    )
