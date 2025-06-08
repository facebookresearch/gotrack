# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional, Tuple
from math import atan2
import numpy as np
import numpy.typing as npt
import torch
from utils import im_util, torch_helpers, transform3d, logging, structs, poser_util

logger = logging.get_logger(__name__)


def approximate_bounding_sphere(pts: npt.NDArray) -> Tuple[npt.NDArray, float]:
    """
    Quick and approximate bounding sphere from a collection of points
    pts: N k-dimensional points stored as an N x k array
    """
    bbox_max = pts.max(axis=0)
    bbox_min = pts.min(axis=0)
    centroid = (bbox_max + bbox_min) / 2.0
    radius_sqr = np.max((((pts - centroid) ** 2).sum(axis=1)))
    radius = np.sqrt(radius_sqr)

    return centroid, radius


def align_crop_camera_right_angle(R_crop_cam_device: npt.NDArray) -> npt.NDArray:
    """
    Update the camera rotation via a rotation around the principal axis (roll) such that the final
    crop left direction is aligned with the HMD down direction. The camera is rotated by either 0,
    90, -90 or 180. The aignement is performed on the left direction to match the behaviour of the
    Hollywood cameras
    Args:
      R_crop_cam_device the camera rotation w.r.t. the headset
    Returns:
      r_alignedcropcam_cropcam: a roll rotation to align the camera
    """

    device_direction_in_crop_cam = R_crop_cam_device @ np.array([0, -1, 0])
    roll_angle = np.pi - atan2(
        device_direction_in_crop_cam[1], device_direction_in_crop_cam[0]
    )
    if np.isnan(roll_angle):
        logger.warning(
            f"Roll angle is NaN, device_direction_in_crop_cam={device_direction_in_crop_cam}"
        )
        roll_angle = np.pi - atan2(
            device_direction_in_crop_cam[1], device_direction_in_crop_cam[0] + 1e-6
        )

    # snap to pi/2 multiples
    roll_angle = 0.5 * np.pi * (int(round(roll_angle / (0.5 * np.pi))) % 4)

    R_alignedcropcam_cropcam = transform3d.axis_angle(
        axis=np.array([0, 0, 1], dtype=np.float32), theta=roll_angle
    )

    return R_alignedcropcam_cropcam


def construct_crop_camera(
    box: structs.AlignedBox2f,
    camera_model_c2w: structs.CameraModel,
    viewport_size: Tuple[int, int],
    viewport_rel_pad: float,
) -> structs.CameraModel:
    """Constructs a virtual pinhole camera from the specified 2D bounding box.

    Args:
        camera_model_c2w: Original camera model with extrinsics set to the
            camera->world transformation.

        viewport_crop_size: Viewport size of the new camera.
        viewport_scaling_factor: Requested scaling of the viewport.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center
        of the specified 2D bounding box and whose focal length is set such as
        the sphere representing the bounding box (+ requested padding) is visible
        in the camera viewport.
    """

    # Get centroid and radius of the reference sphere (the virtual camera will
    # be constructed such as the projection of the sphere fits the viewport.
    f = 0.5 * (camera_model_c2w.f[0] + camera_model_c2w.f[1])
    cx, cy = camera_model_c2w.c
    box_corners_in_c = np.array(
        [
            [box.left - cx, box.top - cy, f],
            [box.right - cx, box.top - cy, f],
            [box.left - cx, box.bottom - cy, f],
            [box.right - cx, box.bottom - cy, f],
        ]
    )
    box_corners_in_c /= np.linalg.norm(box_corners_in_c, axis=1, keepdims=True)
    centroid_in_c = np.mean(box_corners_in_c, axis=0)
    centroid_in_c_h = np.hstack([centroid_in_c, 1]).reshape((4, 1))
    centroid_in_w = camera_model_c2w.T_world_from_eye.dot(centroid_in_c_h)[:3, 0]

    radius = np.linalg.norm(box_corners_in_c - centroid_in_c, axis=1).max()

    # Transformations from world to the original and virtual cameras.
    trans_w2c = np.linalg.inv(camera_model_c2w.T_world_from_eye)
    trans_w2vc = transform3d.gen_look_at_matrix(trans_w2c, centroid_in_w)

    # Transform the centroid from world to the virtual camera.
    centroid_in_vc = transform3d.transform_points(
        trans_w2vc, np.expand_dims(centroid_in_w, axis=0)
    ).squeeze()

    # Project the sphere radius to the image plane of the virtual camera and
    # enlarge it by the specified padding. This defines the 2D extent that
    # should be visible in the virtual camera.
    fx_fy_orig = np.array(camera_model_c2w.f, dtype=np.float32)
    radius_2d = fx_fy_orig * radius / centroid_in_vc[2]
    extent_2d = (1.0 + viewport_rel_pad) * radius_2d

    cx_cy = np.array(viewport_size, dtype=np.float32) / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = fx_fy_orig * cx_cy / extent_2d

    # Parameters of the virtual camera.
    return structs.PinholePlaneCameraModel(
        width=viewport_size[0],
        height=viewport_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        T_world_from_eye=np.linalg.inv(trans_w2vc),
    )


def construct_center_crop_camera(
    camera_model_orig: structs.PinholePlaneCameraModel,
    center_crop_size: Tuple[int, int],
) -> Tuple[structs.PinholePlaneCameraModel, structs.AlignedBox2f]:
    """Constructs a virtual camera focused on the viewport center.

    Args:
        camera_model_orig: Original camera model.
        center_crop_size: Size of the region to crop from the center of the
            camera viewport.
    Returns:
        camera_model: A virtual camera whose viewport has the specified size and
        is centered on the original camera viewport. The camera distortion is the
        same as for the original camera.
        center_crop_box: A box corresponding to the new camera viewport.
    """

    if (
        center_crop_size[0] > camera_model_orig.width
        or center_crop_size[1] > camera_model_orig.height
    ):
        raise ValueError(
            "The center crop cannot be larger than the original camera viewport."
        )

    camera_model = camera_model_orig.copy()

    camera_model.width = center_crop_size[0]
    camera_model.height = center_crop_size[1]

    left = int(0.5 * (camera_model_orig.width - camera_model.width))
    top = int(0.5 * (camera_model_orig.height - camera_model.height))
    camera_model.c = (camera_model_orig.c[0] - left, camera_model_orig.c[1] - top)

    center_crop_box = structs.AlignedBox2f(
        left, top, left + center_crop_size[0], top + center_crop_size[1]
    )

    return camera_model, center_crop_box


def construct_perspective_crop_camera(
    crop_size: Tuple[int, int],
    crop_pad_ratio: float,
    look_at_point_in_world: npt.NDArray,
    canonical_content_radius: float,
    T_world_from_camera: npt.NDArray,
    T_world_from_device: Optional[npt.NDArray] = None,
) -> structs.PinholePlaneCameraModel:
    """Constructs a virtual pinhole camera for perspective cropping.

    Similarly to `ml_libs.lib.hand.crop.create_crop_perspective_camera`,
    the virtual pinhole camera is supposed to be used for cropping by
    mapping the original image to the virtual camera.

    Args:
        crop_size: Viewport size of the crop camera.
        crop_pad_ratio: Requested padding around the cropped content.
        look_at_point_in_world: 3D point in the world through which
            the optical axis of the crop camera should pass.
        canonical_content_radius: Radius of the content to crop expressed
            in the crop camera when its focal length is set to 1.
        object_vertices_in_w: 3D vertices of the object in world coordinates.
        T_world_from_camera: Transformation from the original camera to world.
        T_world_from_device: Transformation from a device (on which the original
            camera is attached) to world.
    Returns:
        A virtual pinhole camera whose optical axis passes through the specified
        3D point (`look_at_point_in_world`) and whose focal length is set such
        as the content with the requested padding fits the camera viewport. The
        crop is potentially rotated around the optical axis such that the HMD
        bottom direction points to the left (similar to Hollywood images).
        The applied rotation has 0, 90, 180, or 270 degrees.
    """

    # Transformations from world to the original and virtual cameras.
    T_camera_from_world = transform3d.inverse_se3_numpy(T_world_from_camera)
    T_crop_from_world = transform3d.gen_look_at_matrix(
        T_camera_from_world, look_at_point_in_world
    )

    if T_world_from_device is None:
        # World to device transformation simulating Hollywood cameras (raw images
        # from these cameras are "lying" on the left side).
        R_rot90 = np.eye(4)
        R_rot90[:3, :3] = transform3d.axis_angle(
            axis=np.array([0, 0, 1], dtype=np.float32), theta=-0.5 * np.pi
        )
        T_world_from_device = T_world_from_camera @ R_rot90

    # Align crop camera with right angle.
    R_world_from_device = T_world_from_device[0:3, 0:3]
    R_crop_from_world = T_crop_from_world[0:3, 0:3]
    T_aligned_crop_from_crop = np.eye(4)
    T_aligned_crop_from_crop[0:3, 0:3] = align_crop_camera_right_angle(
        R_crop_from_world @ R_world_from_device
    )
    T_crop_from_world = T_aligned_crop_from_crop @ T_crop_from_world

    # Project the sphere radius to the image plane of the virtual camera (with
    # unit focal length) and enlarge it by the specified padding. This defines
    # the 2D extent that should be visible in the virtual camera.
    canonical_extent = (1.0 + 2.0 * crop_pad_ratio) * canonical_content_radius

    # Calculate the principal point of the virtual camera, following Nimble's
    # pixel coordinate convention, i.e., coordinate (0.0, 0.0) maps to the center
    # of the pixel at `image[0, 0]` (see fbcode/nimble/pylib/camera/camera.py).
    im_size = np.array(crop_size, dtype=np.float32)
    cx_cy = im_size / 2.0 - 0.5

    # Set the focal length such as all projected points fit the viewport of the
    # virtual camera.
    fx_fy = cx_cy / max(canonical_extent, 1e-06)

    # Parameters of the virtual camera.
    return structs.PinholePlaneCameraModel(
        width=crop_size[0],
        height=crop_size[1],
        f=tuple(fx_fy),
        c=tuple(cx_cy),
        # distort_coeffs=(),
        T_world_from_eye=transform3d.inverse_se3_numpy(T_crop_from_world),
    )


def construct_perspective_crop_camera_from_3d_points(
    crop_size: Tuple[int, int],
    crop_pad_ratio: float,
    points_in_world: npt.NDArray,
    T_world_from_camera: npt.NDArray,
    T_world_from_device: Optional[npt.NDArray] = None,
) -> structs.PinholePlaneCameraModel:
    """Constructs a perspective crop camera from specified 3D points.

    Args:
        crop_size: Viewport size of the crop camera.
        crop_pad_ratio: Requested padding around the cropped content.
        points_in_world: 3D points representing the content to crop.
        T_world_from_camera: Transformation from the original camera to world.
        T_world_from_device: Transformation from a device (on which the original
            camera is attached) to world.
    Returns:
        A virtual pinhole camera whose optical axis passes through the centroid of
        the provided 3D points and whose focal length is set such as the 2D projection
        of the bounding sphere of the 3D points (+ requested padding) fits the camera
        viewport. The crop is potentially rotated around the optical axis based on
        `T_world_from_device` (see `construct_perspective_crop_camera`).
    """

    # 3D centroid and radius of the approximate bounding sphere.
    ref_point_in_world, radius = approximate_bounding_sphere(points_in_world)

    # Project the radius to the canonical version of the crop camera.
    ref_point_in_camera = transform3d.transform_points(
        transform3d.inverse_se3_numpy(T_world_from_camera),
        ref_point_in_world.reshape((1, 3)),
    )
    canonical_radius = radius / np.linalg.norm(ref_point_in_camera)

    return construct_perspective_crop_camera(
        crop_size=crop_size,
        crop_pad_ratio=crop_pad_ratio,
        look_at_point_in_world=ref_point_in_world,
        canonical_content_radius=canonical_radius,
        T_world_from_camera=T_world_from_camera,
        T_world_from_device=T_world_from_device,
    )


def construct_perspective_crop_camera_from_2d_box(
    crop_size: Tuple[int, int],
    crop_pad_ratio: float,
    box: structs.AlignedBox2f,
    camera: structs.CameraModel,
    T_world_from_device: Optional[npt.NDArray] = None,
) -> structs.PinholePlaneCameraModel:
    """Constructs a perspective crop camera from a specified 2D box.

    Args:
        crop_size: Viewport size of the crop camera.
        crop_pad_ratio: Requested padding around the cropped content.
        box: 2D bounding box of the content to crop from the original camera.
        camera: The original camera.
        T_world_from_device: Transformation from a device (on which the original
            camera is attached) to world.
    Returns:
        A virtual pinhole camera whose optical axis passes through the center of
        the warped 2D bounding box and whose focal length is set such as the warped
        box (+ requested padding) fits the camera viewport. The crop is potentially
        rotated around the optical axis based on `T_world_from_device`
        (see `construct_perspective_crop_camera`).
    """

    # Centers of 2D box sides expressed as 3D points in the camera coordinates.
    f = 0.5 * (camera.f[0] + camera.f[1])
    middle_hor = 0.5 * (box.left + box.right)
    middle_ver = 0.5 * (box.top + box.bottom)
    points_in_window = np.array(
        [
            [middle_hor, box.top, f],
            [box.right, middle_ver, f],
            [middle_hor, box.bottom, f],
            [box.left, middle_ver, f],
        ]
    )
    points_in_camera = camera.window_to_eye3(points_in_window)

    # Unit rays pointing to the reference points.
    rays_in_camera = points_in_camera / np.linalg.norm(
        points_in_camera, axis=1, keepdims=True
    )

    # Mean ray representing the center of the 2D box warped to the crop camera.
    center_ray_in_camera = np.mean(rays_in_camera, axis=0)

    # Reference point via which the optical axis of the crop camera will pass
    # is defined by the mean ray.
    ref_point_in_world = transform3d.transform_points(
        camera.T_world_from_eye, center_ray_in_camera
    )

    # Radius covering the 2D box warped to the crop camera, expressed in the canonical
    # version of the crop camera (i.e. with unit focal length).
    canonical_radius = np.linalg.norm(
        rays_in_camera - center_ray_in_camera, axis=1
    ).max()
    canonical_radius /= np.linalg.norm(center_ray_in_camera)

    return construct_perspective_crop_camera(
        crop_size=crop_size,
        crop_pad_ratio=crop_pad_ratio,
        look_at_point_in_world=ref_point_in_world,
        canonical_content_radius=canonical_radius,
        T_world_from_camera=camera.T_world_from_eye,
        T_world_from_device=T_world_from_device,
    )


def cropping_inputs(
    input_images: Dict[str, torch.Tensor],
    cameras: List[structs.CameraModel],
    Ts_world_from_model: np.ndarray,
    target_size: Tuple[int, int],
    object_vertices: List[np.ndarray],
    pad_ratio: float,
    cropping_type: str = "perspective_2d_box",
    input_crop_cameras: Optional[List[structs.CameraModel]] = None,
) -> Tuple[torch.Tensor, List[structs.CameraModel], Dict[str, torch.Tensor]]:
    """Cropping inputs for the forward pass in refiner. The cropping is define in same way as tdg_from_bop_gar.py"""
    image_keys = list(input_images.keys())
    cropped_inputs = {k: [] for k in image_keys}
    crop_cameras = []

    # Convert input images to numpy arrays.
    input_images_np = {}
    for image_key in image_keys:
        input_images_np[image_key] = input_images[image_key].cpu().numpy()

    batch_size = len(input_images[image_keys[0]])
    assert len(cameras) == batch_size
    for sample_id in range(batch_size):
        sample_image = {}
        for image_key in image_keys:
            if image_key.startswith("rgb") or image_key.startswith("monochrome"):
                rgb = im_util.chw_to_hwc(
                    input_images_np[image_key][sample_id],
                )
                sample_image[image_key] = np.uint8(rgb * 255)
            elif image_key.startswith("mask"):
                sample_image[image_key] = np.int16(
                    input_images_np[image_key][sample_id]
                )
            elif image_key.startswith("depth"):
                sample_image[image_key] = input_images_np[image_key][sample_id]
            else:
                raise ValueError(f"Unknown image key {image_key}")

        camera = cameras[sample_id]

        if input_crop_cameras is not None:
            crop_camera = input_crop_cameras[sample_id]
        else:
            # Get object mesh vertices in the model space.
            vertices_in_m = object_vertices[sample_id]

            # Transform the vertices to the world space.
            vertices_in_w = transform3d.transform_points(
                Ts_world_from_model[sample_id], vertices_in_m
            )

            # A world to device transformation simulating Hollywood cameras.
            R_rot90 = np.eye(4)
            R_rot90[:3, :3] = transform3d.axis_angle(
                axis=np.array([0, 0, 1], dtype=np.float32), theta=-0.5 * np.pi
            )
            T_world_from_device = camera.T_world_from_eye @ R_rot90

            # Perspectively crop the object in the given images.
            crop_camera = construct_perspective_crop_camera_from_3d_points(
                crop_size=target_size,
                crop_pad_ratio=pad_ratio,
                points_in_world=vertices_in_w,
                T_world_from_camera=camera.T_world_from_eye,
                T_world_from_device=T_world_from_device,
            )
            # Calculate 2D bounding box of the object in the current crop camera.
            if cropping_type == "perspective_2d_box":
                projs = crop_camera.world_to_window(vertices_in_w)
                box = np.array(im_util.calc_2d_box(projs[:, 0], projs[:, 1]))

                # Recompute the crop camera based on the 2D box.
                crop_camera = construct_perspective_crop_camera_from_2d_box(
                    crop_size=target_size,
                    crop_pad_ratio=pad_ratio,
                    box=structs.AlignedBox2f(*box),
                    camera=crop_camera,
                )
        crops = {}
        for image_key in sample_image.keys():
            crops[image_key] = im_util.warp_image(
                src_image=sample_image[image_key],
                src_camera=camera,
                dst_camera=crop_camera,
            )

        assert isinstance(crop_camera, structs.CameraModel)
        crop_cameras.append(crop_camera)
        for image_key in image_keys:
            if image_key.startswith("rgb") or image_key.startswith("monochrome"):
                cropped_inputs[image_key].append(
                    crops[image_key].transpose(2, 0, 1) / 255.0
                )
            else:
                cropped_inputs[image_key].append(crops[image_key])

    for image_key in image_keys:
        np_type = np.int16 if image_key.startswith("mask") else np.float32
        image_key_data = np.asarray(cropped_inputs[image_key], dtype=np_type)
        cropped_inputs[image_key] = torch.from_numpy(image_key_data)  # pyre-ignore

    # Get transformation from input cameras to crop cameras.
    Ts_crop_cam_from_orig_cam = poser_util.get_T_crop_cam_from_orig_cam(
        crop_cameras=crop_cameras, orig_cameras=cameras
    )
    Ts_crop_cam_from_orig_cam = torch.from_numpy(Ts_crop_cam_from_orig_cam)
    return Ts_crop_cam_from_orig_cam, crop_cameras, cropped_inputs  # pyre-ignore


def batch_cropping_from_bbox(
    source_images: npt.NDArray,
    source_cameras: List[structs.CameraModel],
    source_xyxy_bboxes: npt.NDArray,
    crop_size: Tuple[int, int],
    crop_rel_pad: float,
    source_masks: Optional[npt.NDArray] = None,
) -> structs.Collection:
    """Cropping the input images in batch to get crop images and crop cameras."""
    num_proposals = len(source_xyxy_bboxes)
    crop_camera_models = []
    crop_rgbs = []
    crop_masks = []
    for idx in range(num_proposals):
        xyxy_bbox = source_xyxy_bboxes[idx]
        xyxy_bbox = structs.AlignedBox2f(
            left=xyxy_bbox[0],
            top=xyxy_bbox[1],
            right=xyxy_bbox[2],
            bottom=xyxy_bbox[3],
        )
        # Get squared box for cropping.
        crop_box = im_util.calc_crop_box(
            box=xyxy_bbox,
            make_square=True,
        )
        # Construct a virtual camera focused on the crop.
        crop_camera_model = construct_crop_camera(
            box=crop_box,
            camera_model_c2w=source_cameras[idx],
            viewport_size=crop_size,
            viewport_rel_pad=crop_rel_pad,
        )
        crop_camera_models.append(crop_camera_model)

        image_np_hwc = im_util.warp_image(
            src_camera=source_cameras[idx],
            dst_camera=crop_camera_model,
            src_image=source_images[idx] / 255.0,
        )
        crop_rgbs.append(image_np_hwc)

        if source_masks is not None:
            mask_modal = im_util.warp_image(
                src_camera=source_cameras[idx],
                dst_camera=crop_camera_model,
                src_image=np.uint8(source_masks[idx] * 255.0),
            )
            crop_masks.append(mask_modal)

    crop_rgbs = np.stack(crop_rgbs)
    crop_rgbs = torch_helpers.array_to_tensor(crop_rgbs)
    crop_rgbs = crop_rgbs.to(torch.float32).permute(0, 3, 1, 2)

    if source_masks is not None:
        crop_masks = np.stack(crop_masks)
        crop_masks = torch_helpers.array_to_tensor(crop_masks)
        crop_masks = (crop_masks / 255.0).squeeze(1).bool()

    crop_inputs = structs.Collection()
    crop_inputs.rgbs = crop_rgbs
    crop_inputs.cameras = crop_camera_models
    if source_masks is not None:
        crop_inputs.masks = crop_masks
    return crop_inputs
