# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-unsafe

"""Utilities for posers."""

from typing import Dict, List, Optional
import numpy.typing as npt
import numpy as np
import torch
from torch.nn.functional import interpolate

from utils import structs, transform3d, config, renderer_base, logging

logger = logging.get_logger(__name__)


def get_Ts_world_from_cam(
    cameras: List[structs.CameraModel], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Returns T_world_from_cam matrices of shape [num_cameras, 3, 3]."""
    Ts_world_from_cam = np.asarray([cam.T_world_from_eye for cam in cameras])
    Ts_world_from_cam = torch.as_tensor(Ts_world_from_cam, dtype=torch.float32)
    if device is not None:
        Ts_world_from_cam = Ts_world_from_cam.to(device)
    return Ts_world_from_cam


def get_T_crop_cam_from_orig_cam(
    crop_cameras: List[structs.CameraModel],
    orig_cameras: List[structs.CameraModel],
) -> npt.NDArray:
    """Returns the transformation that transforms original camera to the crop camera frame.
    Args:
        crop_camera: Crop camera model with extrinsics set to the world->camera
        orig_camera: Original camera model with extrinsics set to the camera->world
    Returns:
        Ts_crop_cam_from_orig_cam: [batch x 4 x 4]
    """
    # Crop camera and original camera have the same translation
    Ts_crop_cam_from_orig_cam = np.zeros((len(crop_cameras), 4, 4), dtype=np.float32)
    for idx in range(len(crop_cameras)):
        # Assumption is that crop and original camera are at the same position.
        assert np.allclose(
            crop_cameras[idx].T_world_from_eye[0:3, 3],
            orig_cameras[idx].T_world_from_eye[0:3, 3],
            rtol=1e-03,
            atol=1e-03,
        )
        # Only compute relative rotation because above assert.
        Ts_crop_cam_from_orig_cam[idx] = (
            transform3d.inverse_se3_numpy(crop_cameras[idx].T_world_from_eye)
            @ orig_cameras[idx].T_world_from_eye
        )
    return Ts_crop_cam_from_orig_cam


def single_object_render_pinhole(
    camera_obj_from_cam: structs.PinholePlaneCameraModel,
    obj_id: int,
    renderer: renderer_base.RendererBase,
    render_types: List[renderer_base.RenderType],
    ssaa_factor: float,
    background: Optional[np.ndarray],
) -> Dict[renderer_base.RenderType, structs.ArrayData]:
    """Renders a single object with a single camera."""

    # Prepare a camera for rendering, upsampled for SSAA (supersampling anti-aliasing).
    rendering_camera = structs.PinholePlaneCameraModel(
        width=int(camera_obj_from_cam.width * ssaa_factor),
        height=int(camera_obj_from_cam.height * ssaa_factor),
        f=(
            camera_obj_from_cam.f[0] * ssaa_factor,
            camera_obj_from_cam.f[1] * ssaa_factor,
        ),
        c=(
            camera_obj_from_cam.c[0] * ssaa_factor,
            camera_obj_from_cam.c[1] * ssaa_factor,
        ),
        # distort_coeffs=[],
        # undistort_coeffs=[],
        T_world_from_eye=camera_obj_from_cam.T_world_from_eye,
    )
    rendering = renderer.render_object_model(
        obj_id=obj_id,
        camera_model_c2w=rendering_camera,
        render_types=render_types,
        return_tensors=True,
        background=background,
    )

    # Downsample the rendered images in case of SSAA.
    if ssaa_factor != 1.0:
        scale_factor = 1.0 / ssaa_factor

        # Downsample the rendered RGB image.
        color_hwc = torch.as_tensor(rendering[renderer_base.RenderType.COLOR])
        color_bchw = color_hwc.permute(2, 0, 1).unsqueeze(0)
        color_bchw = interpolate(
            input=color_bchw,
            scale_factor=scale_factor,
            mode="area",
        ).squeeze(0)
        rendering[renderer_base.RenderType.COLOR] = color_bchw.squeeze(0).permute(
            1, 2, 0
        )

        # Downsample the rendered mask.
        mask_hw = torch.as_tensor(rendering[renderer_base.RenderType.MASK])
        mask_bchw = mask_hw.unsqueeze(0).unsqueeze(0).to(torch.uint8)
        mask_bchw = interpolate(
            input=mask_bchw,
            scale_factor=scale_factor,
            mode="nearest",
        )
        rendering[renderer_base.RenderType.MASK] = (
            mask_bchw.squeeze(0).squeeze(0).to(torch.bool)
        )

        # Downsample the rendered depth image.
        mask_hw = torch.as_tensor(rendering[renderer_base.RenderType.DEPTH])
        mask_bchw = mask_hw.unsqueeze(0).unsqueeze(0)
        mask_bchw = interpolate(
            input=mask_bchw,
            scale_factor=scale_factor,
            mode="nearest",
        )
        rendering[renderer_base.RenderType.DEPTH] = mask_bchw.squeeze(0).squeeze(0)

    return rendering


def batch_object_render_pinhole(  # noqa: C901
    obj_ids: List[int],
    Ts_cam_from_model: np.ndarray,
    cameras: List[structs.CameraModel],
    renderer: renderer_base.RendererBase,
    ssaa_factor: float = 1.0,
    background: Optional[structs.Color] = None,
) -> structs.Collection:
    render_types = [
        renderer_base.RenderType.COLOR,
        renderer_base.RenderType.MASK,
        renderer_base.RenderType.DEPTH,
    ]
    B = len(Ts_cam_from_model)
    H, W = cameras[0].height, cameras[0].width

    batch_rendering = structs.Collection()
    batch_rendering.rgbs = torch.zeros((B, 3, H, W), dtype=torch.float32)
    batch_rendering.masks = torch.zeros((B, H, W), dtype=torch.bool)
    batch_rendering.depths = torch.zeros((B, H, W), dtype=torch.float32)

    assert len(obj_ids) == len(cameras), "Number of assets and cameras must match."

    for sample_id, camera in enumerate(cameras):
        # Camera model for rendering.
        camera_obj_from_cam = camera.copy()

        # Assuming the world coordinate = model coordinate.
        T_world_from_eye = transform3d.inverse_se3_numpy(Ts_cam_from_model[sample_id])
        camera_obj_from_cam.T_world_from_eye = T_world_from_eye

        # Render the object.
        rendering = single_object_render_pinhole(
            camera_obj_from_cam=camera_obj_from_cam,
            obj_id=obj_ids[sample_id],
            renderer=renderer,
            render_types=render_types,
            ssaa_factor=ssaa_factor,
            background=background,
        )

        rgb_rendering = rendering[renderer_base.RenderType.COLOR]
        rgb_rendering = rgb_rendering.permute(2, 0, 1)  # pyre-ignore
        batch_rendering.rgbs[sample_id] = rgb_rendering
        batch_rendering.masks[sample_id] = rendering[renderer_base.RenderType.MASK]
        batch_rendering.depths[sample_id] = rendering[renderer_base.RenderType.DEPTH]

        # Report NaNs warning when it is significant.
        num_nans = torch.sum(torch.isnan(batch_rendering.rgbs[sample_id]))
        if num_nans > (H * W * 0.01):
            logger.warning(
                f"NaNs ({num_nans}) in rendered image with {obj_ids[sample_id]}"
            )
        # Fix NaNs to 0.
        batch_rendering.rgbs[sample_id][
            torch.isnan(batch_rendering.rgbs[sample_id])
        ] = 0
        batch_rendering.masks[sample_id][
            torch.isnan(batch_rendering.masks[sample_id])
        ] = 0
        batch_rendering.depths[sample_id][
            torch.isnan(batch_rendering.depths[sample_id])
        ] = 0

    return batch_rendering


def update_poses(
    Ts_cam_from_model: torch.Tensor,
    deltas: torch.Tensor,
    cameras: List[structs.CameraModel],
    pose_repre: config.PoseRepre,
) -> Dict[str, torch.Tensor]:
    """Applies predicted delta poses to the original poses.

    Args:
        poses: Rigid transformations of shape [num_poses, 4, 4].
        deltas: Predicted delta poses of shape [num_poses, pose_repre_size].
        Ks: Intrinsics matrices of shape [num_poses, 3, 3].
    Returns:
        Updated poses of shape [num_poses, 4, 4].
    """
    Ts_delta = torch.zeros_like(Ts_cam_from_model, dtype=torch.float32)
    Ts_delta[:, 3, 3] = 1.0

    if pose_repre.value == config.PoseRepre.XYZ_CONT6D.value:
        # Apply the rotational update.
        R_delta = transform3d.rotation_matrix_from_cont6d(deltas[:, :6])
        Ts_delta[:, :3, :3] = R_delta.clone()

        R_out = R_delta @ Ts_cam_from_model[:, :3, :3]

        # Apply the translational update.
        # z_delta is a relative multiplicative displacement, z_in and z_out
        # are in millimeters.
        z_delta = deltas[:, [8]]
        z_in = Ts_cam_from_model[:, 2, [3]]
        z_out = z_delta * z_in
        Ts_delta[:, 2, [3]] = z_out - z_in

        # xy_delta and fxfy are in pixels, xy_in and xy_out in millimeters.
        xy_delta = deltas[:, 6:8]
        xy_in = Ts_cam_from_model[:, :2, 3]
        fxfy = torch.as_tensor([camera.f for camera in cameras])
        fxfy = fxfy.to(deltas.device)
        xy_out = ((xy_delta / fxfy) + (xy_in / z_in.repeat(1, 2))) * z_out.repeat(1, 2)
        Ts_delta[:, :2, 3] = xy_out - xy_in

        # Assemble the output poses.
        poses_out = Ts_cam_from_model.clone()
        poses_out[:, :3, :3] = R_out
        poses_out[:, :2, 3] = xy_out
        poses_out[:, 2, 3] = z_out.flatten()

    else:
        raise ValueError(f"Not supported pose representation: {pose_repre}")
    return {"T_crop_cam_from_model": poses_out, "Ts_delta": Ts_delta}


def generate_pose_perturbation(
    init_pose_opts: config.InitPoseOpts,
    gt_Ts_cam_from_model: Optional[torch.Tensor] = None,
    poses_coarse_cam_from_obj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prepares initial poses for pose refinement.

    Args:
        ...
        init_pose_opts: Options for generating the initial poses.
    Returns:
        Initial poses of shape [batch_size, 4, 4].
    """

    poses_init: Optional[torch.Tensor] = None

    # Use GT poses as the initial poses.
    if init_pose_opts.pose_type.value == config.InitPoseType.GT_POSES.value:
        if gt_Ts_cam_from_model is None:
            raise ValueError("GT poses are required for GT pose initialization.")
        poses_init = torch.clone(gt_Ts_cam_from_model)

    # Use coarse poses as the initial poses.
    elif init_pose_opts.pose_type.value == config.InitPoseType.FROM_COARSE_POSES.value:
        poses_init = poses_coarse_cam_from_obj

    # Calculate initial poses from GT 2D bounding boxes.
    elif init_pose_opts.pose_type.value == config.InitPoseType.FROM_GT_BOXES.value:
        raise NotImplementedError

    # Calculate initial poses from detected 2D bounding boxes.
    elif init_pose_opts.pose_type.value == config.InitPoseType.FROM_DETECTIONS.value:
        raise NotImplementedError

    if poses_init is None:
        raise ValueError("Initial poses could not be defined.")

    # Potentially add a noise to the initial poses.
    poses_perturb = transform3d.perturb_transforms_numpy(
        poses_init,
        # pyre-fixme[6]: For 2nd argument expected `Union[Sequence[float], float,
        #  Tensor]` but got `ndarray[typing.Any, dtype[typing.Any]]`.
        angle_std_in_rad=np.asarray(init_pose_opts.noise_rotation_std) / 180 * np.pi,
        # pyre-fixme[6]: For 3rd argument expected `Union[Sequence[float], float,
        #  Tensor]` but got `ndarray[typing.Any, dtype[typing.Any]]`.
        trans_std=np.asarray(init_pose_opts.noise_translation_std),
    )
    return poses_perturb
