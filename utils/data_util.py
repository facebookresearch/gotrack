# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Any, Dict, Tuple, Optional, List

import numpy as np

import torch
from utils import (
    crop_generation,
    im_util,
    json_util,
    poser_util,
    renderer_base,
    structs,
    misc,
    torch_helpers,
    transform3d,
    logging,
)

logger = logging.get_logger(__name__)


def parse_chunk_cameras(
    cameras_json: Any,
    im_size: Optional[Tuple[int, int]] = None,
) -> Dict[int, structs.PinholePlaneCameraModel]:
    """Parses per-image camera parameters of a dataset chunk from JSON format.

    Args:
        cameras_json: Cameras in JSON format.
        im_size: The image size (needs to be either provided via this argument
            or must be present in the JSON file).
    Returns:
        A dictionary mapping the image ID to camera parameters.
        For each image, there is a dictionary with the following items:
    """

    cameras = {}
    for im_id, camera_json in cameras_json.items():
        width = None
        height = None
        fx = None
        fy = None
        cx = None
        cy = None
        depth_scale = None
        is_fisheye = False  # Whether the camera model is fisheye.
        camera_model_name = None  # In case of a fisheye camera, the model type.
        projection_params = None  # In case of a fisheye camera.
        # World to camera transformation.
        extrinsics_w2c = np.eye(4, dtype=np.float32)

        if im_size is not None:
            width = im_size[0]
            height = im_size[1]

        for k, v in camera_json.items():
            if k == "im_size":
                width = int(v[0])
                height = int(v[1])
            elif k == "cam_K":
                K = np.array(v, np.float32).reshape((3, 3))
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            elif k == "cam_R_w2c":
                extrinsics_w2c[:3, :3] = np.array(v, np.float32).reshape((3, 3))
            elif k == "cam_t_w2c":
                extrinsics_w2c[:3, 3:] = np.array(v, np.float32).reshape((3, 1))
            elif k == "depth_scale":
                depth_scale = float(v)  # noqa: F841
            elif k == "cam_model":
                is_fisheye = True
                camera_model_name = v["projection_model_type"]
                projection_params = v["projection_params"]

        # Camera to world transformation.
        extrinsics_c2w = np.linalg.inv(extrinsics_w2c)

        if is_fisheye:
            if (
                camera_model_name == "CameraModelType.FISHEYE624"
                and len(projection_params) == 15
            ):
                # TODO: Aria data hack
                f, cx, cy = projection_params[:3]
                fx = fy = f
                coeffs = projection_params[3:]
            else:
                fx, fy, cx, cy = projection_params[:4]
                coeffs = projection_params[4:]
            try:
                from hand_tracking_toolkit.camera import model_by_name
            except ImportError:
                logger.error(
                    "hand_tracking_toolkit is not installed. "
                    "Please run pip install git+https://github.com/facebookresearch/hand_tracking_toolkit."
                )
                raise
            cls = model_by_name[camera_model_name]
            camera_model = cls(width, height, (fx, fy), (cx, cy), coeffs)
        else:
            camera_model = structs.PinholePlaneCameraModel(
                width=width,
                height=height,
                f=(fx, fy),
                c=(cx, cy),
                T_world_from_eye=extrinsics_c2w,
            )
        cameras[im_id] = camera_model
    return cameras


def load_chunk_cameras(
    path: str, im_size: Optional[Tuple[int, int]] = None
) -> Dict[int, structs.PinholePlaneCameraModel]:
    """Loads per-image camera parameters of a dataset chunk from a JSON file.

    Args:
        path: The path to the input JSON file.
        im_size: The image size (needs to be either provided via this argument
            or must be present in the JSON file).
    Returns:
        A dictionary mapping the image ID to camera parameters.
        For each image, there is a dictionary with the following items:
    """

    cameras_json = json_util.load_json(path, keys_to_int=True)

    return parse_chunk_cameras(cameras_json, im_size)


def parse_chunk_gts(
    gts_json: Any,
    dataset: str,
) -> Dict[int, List[structs.ObjectAnnotation]]:
    """Parses GT annotations of a dataset chunk from JSON format.

    Args:
        gts_json: Chunk GTs in JSON format.
        dataset: A dataset which the JSON file belongs to.
    Returns:
        A dictionary with the loaded GT annotations.
    """

    gts = {}
    for im_id, im_gts_json in gts_json.items():
        gts[im_id] = []
        for gt_raw in im_gts_json:
            dataset_curr, lid, R, t = None, None, None, None
            for k, v in gt_raw.items():
                if k == "dataset":
                    dataset_curr = str(v)
                if k == "obj_id":
                    lid = int(v)
                elif k == "cam_R_m2c":
                    R = np.array(v, np.float32).reshape((3, 3))
                elif k == "cam_t_m2c":
                    t = np.array(v, np.float32).reshape((3, 1))
            if dataset_curr is None:
                dataset_curr = dataset
            if lid is None:
                raise ValueError("Local ID must be specified.")
            gts[im_id].append(
                structs.ObjectAnnotation(
                    dataset=dataset_curr, lid=lid, pose=structs.ObjectPose(R=R, t=t)
                )
            )
    return gts


def compute_gotrack_inputs_from_init_poses(
    input_rgbs: torch.Tensor,
    input_cameras: List[structs.CameraModel],
    init_poses_cam_from_model: torch.Tensor,
    renderer: renderer_base.RendererBase,
    obj_ids: List[int],
    object_vertices: np.ndarray,
    crop_size: Tuple[int, int],
    crop_rel_pad: float,
    cropping_type: str,
    ssaa_factor: float,
    background_type: str,
    input_masks: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, Any], List[structs.CameraModel], torch.Tensor]:
    """
    This function prepares crop data from initial poses used in both training and testing (one or multiple iterations):
    - First, use the initial pose + CAD model to crop the input image, and get `crop_cameras` & T_orig_cam_to_crop_cam
    - Second, generate the reference image with `crop_cameras` and `init_poses_cam_from_model`
    Args:
        input_rgbs: [B, 3, H, W] input images.
        input_cameras: List[CameraModel], input (original) cameras.
        init_poses_cam_from_model: [B, 4, 4] initial pose in camera coordinate frame.
        renderer: renderer used to render the template data.
        asset_keys: asset keys for current batch.
        asset_vertices: [B, N, 3] vertices of the object.
        data_preprocessing_opts: options for cropping.
        input_masks: optional, [B, H, W] input masks, for training it is modal masks, used for generating visible masks.
    Returns:
        A dictionary with the following keys:
            crop_inputs_rgbs: [B, 3, H, W] cropping of the input images.
            template_data: rendered template data having:
                - rgbs: [B, 3, H, W] template images.
                - masks: [B, H, W] template masks.
                - depths: [B, H, W] template depths.
            crop_masks: if used [B, H, W] cropping of the input masks.
        crop_cameras: Cropped cameras.
        Ts_crop_cam_from_orig_cam: [B, 4, 4] transformation from original camera crop cameras.
    """
    device = init_poses_cam_from_model.device

    # Define outputs as a dictionary.
    outputs = {}

    input_images = {"rgbs": input_rgbs}
    if input_masks is not None:
        input_images["masks"] = input_masks

    # Transform object poses from camera coordinate frame to world coordinate frame.
    Ts_world_from_cam = poser_util.get_Ts_world_from_cam(input_cameras)
    init_poses_world_from_model = (
        Ts_world_from_cam.to(device) @ init_poses_cam_from_model
    )

    # Crop images using initial poses.
    (
        Ts_crop_cam_from_orig_cam,
        crop_cameras,
        cropped_inputs,
    ) = crop_generation.cropping_inputs(
        input_images=input_images,
        Ts_world_from_model=torch_helpers.tensor_to_array(init_poses_world_from_model),
        cameras=input_cameras,
        target_size=crop_size,
        object_vertices=object_vertices,  # pyre-ignore
        pad_ratio=crop_rel_pad,
        cropping_type=cropping_type,
    )
    device = init_poses_cam_from_model.device
    Ts_crop_cam_from_orig_cam = Ts_crop_cam_from_orig_cam.to(device)

    outputs["crop_rgbs"] = cropped_inputs["rgbs"]
    if input_masks is not None:
        outputs["crop_masks"] = cropped_inputs["masks"]

    # Transform object poses from original cameras to crop cameras.
    init_poses_crop_cam_from_model = (
        Ts_crop_cam_from_orig_cam @ init_poses_cam_from_model
    )

    # Rendering the template data.
    templates = poser_util.batch_object_render_pinhole(
        obj_ids=obj_ids,
        Ts_cam_from_model=torch_helpers.tensor_to_array(init_poses_crop_cam_from_model),
        cameras=crop_cameras,
        renderer=renderer,
        ssaa_factor=ssaa_factor,
        background=[0.5, 0.5, 0.5] if background_type == "gray" else None,
    )
    outputs["templates"] = templates
    return outputs, crop_cameras, Ts_crop_cam_from_orig_cam


def process_inputs(
    inputs: Dict[str, Any],
    crop_size: Tuple[int, int],
    crop_rel_pad: float,
    cropping_type: str,
    ssaa_factor: float,
    object_vertices: Dict[int, torch.Tensor],
    obj_ids: List[int],
    renderer: renderer_base.RendererBase,
    background_type: str,
) -> Dict[str, Any]:
    """
    Preparing inputs and gts for GoTrack refiner: flows, masks, valid_masks, etc.

    Args:
        inputs: output of convert_batched_frame_sequences_to_gotrack_inputs
    Returns: same inputs with additional fields:
        - crop_cameras: Cropped cameras.
        - gotrack_inputs: inputs used in generic_refiner, i.e initial poses before refinement, and in crop cameras.:
            - input images in crop cameras: [B, 3, H_crop, W_crop] cropping of the input images.
            - init_poses_cam_from_model: [B, 4, 4] initial pose in original camera coordinate frame.
            - init_poses_world_from_model: [B, 4, 4] initial pose in world coordinate frame.
            - init_poses_crop_cam_from_model: [B, 4, 4] initial pose in crop camera coordinate frame.
            - poses_crop_cam_from_model: [B, 4, 4] GT poses in crop camera coordinate frame.
            - Ts_crop_cam_from_orig_cam: [B, 4, 4] transformation from original camera crop cameras.
            - template_data: rendered template data having:
                - rgbs: [B, 3, H_crop, W_crop] template images.
                - masks: [B, H_crop, W_crop] template masks.
                - depths: [B, H_crop, W_crop] template depths.
        - generic_refiner_gts: gt used in refiner, i.e flows, masks, valid_masks.
            - flows: [B, H_crop, W_crop, 2] flows from template to query.
            - masks: [B, H_crop, W_crop] flow masks, i.e dubplicate of template masks.
            - valid_masks: [B, H_crop, W_crop] valid masks, i.e template pixels that are visible in query images.
    """
    objects = inputs["objects"]
    images = inputs["images"]
    # Create a collection for the refiner inputs.
    gotrack_inputs = structs.Collection()

    # Set initial poses.
    gotrack_inputs.init_poses_cam_from_model = objects.poses_cam_from_model
    gotrack_inputs.init_poses_world_from_model = objects.poses_world_from_model

    # Get the original images, cameras for each object.
    frame_ids = objects.frame_ids
    orig_rgbs = images.bitmaps[frame_ids]
    orig_cameras = [images.cameras[frame_id] for frame_id in frame_ids.cpu().numpy()]
    Ts_world_from_cam = poser_util.get_Ts_world_from_cam(orig_cameras)

    # Crop the input images and get the crop cameras.
    (
        crop_inputs,
        crop_cameras,
        Ts_crop_cam_from_orig_cam,
    ) = compute_gotrack_inputs_from_init_poses(
        input_rgbs=orig_rgbs,
        input_cameras=orig_cameras,
        init_poses_cam_from_model=gotrack_inputs.init_poses_cam_from_model,
        object_vertices=object_vertices,  # pyre-ignore
        obj_ids=obj_ids,
        crop_size=crop_size,
        crop_rel_pad=crop_rel_pad,
        cropping_type=cropping_type,
        ssaa_factor=ssaa_factor,
        renderer=renderer,
        background_type=background_type,
    )

    # Get the intrinsics from the crop cameras.
    crop_cam_intrinsics = [
        misc.get_intrinsic_matrix(crop_cameras[i]) for i in range(len(crop_cameras))
    ]
    gotrack_inputs.crop_cam_intrinsics = torch.from_numpy(
        np.stack(crop_cam_intrinsics)
    ).float()

    for key in crop_inputs:
        setattr(gotrack_inputs, key, crop_inputs[key])

    # Transform object poses from original cameras to crop cameras.
    gotrack_inputs.init_poses_crop_cam_from_model = (
        Ts_crop_cam_from_orig_cam @ gotrack_inputs.init_poses_cam_from_model
    )
    gotrack_inputs.poses_crop_cam_from_model = (
        Ts_crop_cam_from_orig_cam @ objects.poses_cam_from_model
    )

    # Update inputs with crop data.
    inputs["gotrack_inputs"] = gotrack_inputs
    inputs["crop_cameras"] = crop_cameras
    inputs["Ts_crop_cam_from_orig_cam"] = Ts_crop_cam_from_orig_cam
    inputs["Ts_world_from_cam"] = Ts_world_from_cam
    inputs["obj_ids"] = obj_ids

    # Keep the original images and cameras for visualizations.
    inputs["orig_cameras"] = orig_cameras
    inputs["orig_rgbs"] = orig_rgbs

    return inputs


def prepare_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare the inputs for the refiner and move the tensors to the correct device."""

    device = torch.cuda.current_device()

    # Get the refiner inputs.
    gotrack_inputs = inputs["gotrack_inputs"]

    # Get the current device.
    rgbs_query = gotrack_inputs.crop_rgbs.to(device)
    device = rgbs_query.device

    # Move the refiner inputs to the current device.
    gotrack_inputs = gotrack_inputs.to(device)

    # Get the template data.
    templates = gotrack_inputs.templates

    # Define the input forward pass.
    forward_inputs = {
        "rgbs_query": rgbs_query,
        "rgbs_template": templates.rgbs,
        "masks_template": templates.masks,
        "depths_template": templates.depths,
        "crop_cam_intrinsics": gotrack_inputs.crop_cam_intrinsics,
        "Ts_crop_cam_from_model_template": gotrack_inputs.init_poses_crop_cam_from_model,
        "Ts_crop_cam_from_orig_cam": inputs["Ts_crop_cam_from_orig_cam"],
        "Ts_world_from_cam": inputs["Ts_world_from_cam"],
        "crop_cameras": inputs["crop_cameras"],
        "orig_cameras": inputs["orig_cameras"],
        "orig_rgbs": inputs["orig_rgbs"],
    }
    if "features_query" in inputs:
        forward_inputs["features_query"] = inputs["features_query"].to(device)
        forward_inputs["features_reference"] = inputs["features_reference"].to(device)
    return forward_inputs


def convert_list_scene_observations_to_pipeline_inputs(
    inputs: Dict[str, Any],
    is_localization_task: bool,
) -> Dict[str, Any]:
    """Collate function to convert a list of SceneObservation objects to a format that supported by PytorchLightning."""
    assert len(inputs) == 1, "Batch size must be 1 for BOP datasets."
    scene_observation = inputs[0]["scene_observation"]
    target_objects = inputs[0]["target_objects"]
    if not is_localization_task:
        target_objects = None
    return {"scene_observation": scene_observation, "target_objects": target_objects}


def convert_list_scene_observations_to_gotrack_inputs(  # noqa: C901
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """A collate function used by GoTrack-based models.

    Args:
        input_data: Batched frame sequences.
    Returns:
        "objects": Collection with object data.
    """
    list_scene_observations = [data["scene_observation"] for data in inputs]

    # Count number of objects in the batch.
    num_images = len(list_scene_observations)
    num_objects = 0
    object_fields = set()
    for frame_id, frame_anno in enumerate(list_scene_observations):
        assert frame_anno.objects_anno is not None
        num_objects += len(frame_anno.objects_anno)
        if frame_anno.objects_anno is None:
            for anno in frame_anno.objects_anno:
                for field, value in anno._asdict().items():
                    if value is not None:
                        object_fields.add(field)

    # Get image shape.
    first_image = list_scene_observations[0].image
    assert first_image is not None
    im_hwc_shape = first_image.shape[-3:]
    height, width = im_hwc_shape[:2]

    # Prepare object collection.
    # Initialize with -1 to indicate invalid values.
    images = structs.Collection()
    images.bitmaps = -torch.ones((num_images, 3, height, width), dtype=torch.float32)
    images.scene_ids = -torch.ones((num_images), dtype=torch.int32)
    images.im_ids = -torch.ones((num_images), dtype=torch.int32)
    images.cameras = [None for _ in range(num_images)]
    images.times = -torch.ones((num_images), dtype=torch.float32)

    objects = structs.Collection()
    objects.labels = -torch.ones((num_objects), dtype=torch.int32)
    objects.frame_ids = -torch.ones((num_objects), dtype=torch.int32)
    objects.inst_ids = -torch.ones((num_objects), dtype=torch.int32)
    objects.poses_cam_from_model = -torch.ones((num_objects, 4, 4), dtype=torch.float32)
    objects.poses_world_from_model = -torch.ones(
        (num_objects, 4, 4), dtype=torch.float32
    )
    if "masks_modal" in object_fields:
        objects.masks_modal = torch.zeros(
            (num_objects, height, width), dtype=torch.bool
        )
    if "masks_amodal" in object_fields:
        objects.masks_amodal = torch.zeros(
            (num_objects, height, width), dtype=torch.bool
        )
    if "boxes_modal" in object_fields:
        objects.boxes_modal = -torch.ones((num_objects, 4), dtype=torch.float32)
    if "boxes_amodal" in object_fields:
        objects.boxes_amodal = -torch.ones((num_objects, 4), dtype=torch.float32)

    anno_id = 0
    for frame_id, frame_anno in enumerate(list_scene_observations):
        # Get camera.
        camera = frame_anno.camera
        images.bitmaps[frame_id] = (
            torch_helpers.array_to_tensor(im_util.hwc_to_chw(frame_anno.image)) / 255.0
        )
        images.cameras[frame_id] = camera
        images.scene_ids[frame_id] = frame_anno.scene_id
        images.im_ids[frame_id] = frame_anno.im_id
        images.times[frame_id] = frame_anno.time

        # Transformation from world to camera.
        T_cam_from_world = torch.as_tensor(
            transform3d.inverse_se3(torch.as_tensor(camera.T_world_from_eye)),
            dtype=torch.float32,
        )
        assert frame_anno.objects_anno is not None
        for anno_id, anno in enumerate(frame_anno.objects_anno):
            objects.frame_ids[anno_id] = frame_id
            # Object label.
            objects.labels[anno_id] = anno.lid
            objects.inst_ids[anno_id] = anno_id
            assert frame_anno.image is not None

            # Object pose.
            T_world_from_model = transform3d.Rt_to_4x4(
                R=torch.as_tensor(anno.pose.R),
                t=torch.as_tensor(anno.pose.t.T),
            ).to(torch.float32)
            objects.poses_cam_from_model[anno_id] = torch.matmul(
                T_cam_from_world, T_world_from_model
            )
            objects.poses_world_from_model[anno_id] = T_world_from_model

            # Modal and amodal masks and boxes.
            if objects.has("masks_modal"):
                assert anno.masks_modal is not None
                objects.masks_modal[anno_id] = anno.masks_modal
            if objects.has("masks_amodal"):
                assert anno.masks_amodal is not None
                objects.masks_amodal[anno_id] = anno.masks_amodal
            if objects.has("boxes_modal"):
                assert anno.boxes_modal is not None
                objects.boxes_modal[anno_id] = anno.boxes_modal
            if objects.has("boxes_amodal"):
                assert anno.boxes_amodal is not None
                objects.boxes_amodal[anno_id] = anno.boxes_amodal
            anno_id += 1
    return {"objects": objects, "images": images}


def convert_default_detections_to_foundpose_inputs(
    scene_observation: structs.SceneObservation,
    crop_size: Tuple[int, int],
    crop_rel_pad: float,
) -> Dict[str, Any]:
    """ "Convert default detections to FoundPose inputs."""
    device = torch.cuda.current_device()
    detections = structs.Collection()
    masks = []
    boxes = []
    labels = []
    scores = []
    for obj_anno in scene_observation.objects_anno:
        boxes.append(obj_anno.boxes_modal)
        masks.append(obj_anno.masks_modal)
        labels.append(obj_anno.lid)
        scores.append(obj_anno.score)
    detections.labels = torch.tensor(labels, dtype=torch.int32)
    detections.boxes = torch.tensor(boxes, dtype=torch.float32)
    detections.masks = torch.tensor(masks, dtype=torch.bool).to(device)
    detections.scores = torch.tensor(scores, dtype=torch.float32)
    detections = detections.to(device)

    crop_detections = crop_generation.batch_cropping_from_bbox(
        source_cameras=[scene_observation.camera] * len(detections),
        source_images=[scene_observation.image] * len(detections),
        source_masks=torch_helpers.tensor_to_array(detections.masks.unsqueeze(-1)),
        source_xyxy_bboxes=torch_helpers.tensor_to_array(detections.boxes),
        crop_size=crop_size,
        crop_rel_pad=crop_rel_pad,
    )
    return {
        "detections": detections,
        "crop_detections": crop_detections,
        "run_time": scene_observation.time,
        "scene_obs": scene_observation._replace(objects_anno=None),
    }


def convert_foundpose_outputs_to_gotrack_inputs(  # noqa: C901
    inputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Function to convert FoundPose's outputs to GoTrack's inputs.
    This function is a combination of convert_list_scene_observations_to_gotrack_inputs and process_inputs.
    """

    scene_observation = inputs["scene_observation"]
    crop_detections = inputs["crop_detections"]
    detections = inputs["detections"]
    assert len(detections) == len(crop_detections)

    # Crop of query images are used to get the transform from original camera to crop camera.
    crop_cameras_query = crop_detections.cameras
    # Crop cameras of the template images are used to recover the object poses in the crop cameras.
    crop_cameras_template = inputs["templates"].cameras

    # Similar to convert_list_scene_observations_to_gotrack_inputs
    detections.frame_ids = torch.zeros(len(detections), dtype=torch.int32)
    outputs = {"objects": detections}
    images = structs.Collection()
    bitmaps = (
        torch_helpers.array_to_tensor(im_util.hwc_to_chw(scene_observation.image))
        / 255.0
    )
    images.bitmaps = bitmaps.unsqueeze(0).float()
    images.cameras = [scene_observation.camera]
    images.scene_ids = torch.tensor([scene_observation.scene_id], dtype=torch.int32)
    images.im_ids = torch.tensor([scene_observation.im_id], dtype=torch.int32)
    images.times = torch.tensor([inputs["run_time"]], dtype=torch.float32)
    outputs["images"] = images

    # Similar to process_inputs
    Ts_crop_cam_from_orig_cam = poser_util.get_T_crop_cam_from_orig_cam(
        crop_cameras=crop_cameras_query,
        orig_cameras=[scene_observation.camera for _ in range(len(detections))],
    )
    Ts_crop_cam_from_orig_cam = torch.from_numpy(Ts_crop_cam_from_orig_cam)
    Ts_world_from_cam = poser_util.get_Ts_world_from_cam(
        [scene_observation.camera for _ in range(len(detections))]
    )
    outputs["crop_cameras"] = crop_cameras_template
    outputs["Ts_crop_cam_from_orig_cam"] = Ts_crop_cam_from_orig_cam
    outputs["Ts_world_from_cam"] = Ts_world_from_cam
    outputs["obj_ids"] = torch_helpers.tensor_to_array(detections.labels)

    # Keep the original images and cameras for visualizations.
    outputs["orig_cameras"] = [scene_observation.camera]
    outputs["orig_rgbs"] = images.bitmaps

    gotrack_inputs = structs.Collection()
    gotrack_inputs.init_poses_cam_from_model = detections.poses_cam_from_model
    gotrack_inputs.init_poses_world_from_model = detections.poses_world_from_model
    gotrack_inputs.templates = inputs["templates"]
    gotrack_inputs.crop_rgbs = crop_detections.rgbs
    gotrack_inputs.crop_masks = crop_detections.masks
    # Transform object poses from original cameras to crop cameras.
    gotrack_inputs.init_poses_crop_cam_from_model = (
        Ts_crop_cam_from_orig_cam @ gotrack_inputs.init_poses_cam_from_model
    )
    crop_cam_intrinsics = [
        misc.get_intrinsic_matrix(camera) for camera in crop_cameras_template
    ]
    gotrack_inputs.crop_cam_intrinsics = torch.from_numpy(
        np.stack(crop_cam_intrinsics)
    ).float()
    outputs["gotrack_inputs"] = gotrack_inputs
    return outputs
