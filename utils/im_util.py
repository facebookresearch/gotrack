# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import numpy.typing as npt
import torch
from utils import logging, structs, torch_helpers
from PIL import Image
import torchvision.transforms as T
from torchvision.ops.boxes import box_area

logger = logging.get_logger(__name__)

im_normalize = T.Compose(
    [
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def mask_rgb(im: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    mask_hwc = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    masked_im = im.copy() * mask_hwc
    return masked_im


def xyxy_to_xywh(
    bbox: Union[npt.NDArray, torch.Tensor],
) -> Union[npt.NDArray, torch.Tensor]:
    if len(bbox.shape) == 1:
        """Convert [x1, y1, x2, y2] box format to [x, y, w, h] format."""
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2 - x1, y2 - y1]
    elif len(bbox.shape) == 2:
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return torch.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")


def masks_to_xyxy_boxes(masks: torch.Tensor) -> torch.Tensor:
    xyxy_bboxes = []
    for mask in masks:
        mask_arr = torch_helpers.tensor_to_array(mask)
        mask_arr = mask_arr.astype(np.uint8)
        mask_pil = Image.fromarray(mask_arr)
        xyxy_bboxes.append(mask_pil.getbbox())
    xyxy_bboxes = torch.tensor(xyxy_bboxes).to(masks.device)
    return xyxy_bboxes


def calc_crop_box(
    box: structs.AlignedBox2f,
    box_scaling_factor: float = 1.0,
    make_square: bool = False,
) -> structs.AlignedBox2f:
    """Adjusts a bounding box to the specified aspect and scale.

    Args:
        box: Bounding box.
        box_aspect: The aspect ratio of the target box.
        box_scaling_factor: The scaling factor to apply to the box.
    Returns:
        Adjusted box.
    """

    # Potentially inflate the box and adjust aspect ratio.
    crop_box_width = box.width * box_scaling_factor
    crop_box_height = box.height * box_scaling_factor

    # Optionally make the box square.
    if make_square:
        crop_box_side = max(crop_box_width, crop_box_height)
        crop_box_width = crop_box_side
        crop_box_height = crop_box_side

    # Calculate padding.
    x_pad = 0.5 * (crop_box_width - box.width)
    y_pad = 0.5 * (crop_box_height - box.height)

    return structs.AlignedBox2f(
        left=box.left - x_pad,
        top=box.top - y_pad,
        right=box.right + x_pad,
        bottom=box.bottom + y_pad,
    )


def calc_2d_box(
    xs: torch.Tensor,
    ys: torch.Tensor,
    im_size: Optional[torch.Tensor] = None,
    clip: bool = False,
) -> torch.Tensor:
    """Calculates the 2D bounding box of a set of 2D points.

    Args:
        xs: A 1D tensor with x-coordinates of 2D points.
        ys: A 1D tensor with y-coordinates of 2D points.
        im_size: The image size (width, height), used for optional clipping.
        clip: Whether to clip the bounding box (default == False).
    Returns:
        The 2D bounding box (x1, y1, x2, y2), where (x1, y1) and (x2, y2) is the
        minimum and the maximum corner respectively.
    """
    if len(xs) == 0 or len(ys) == 0:
        return torch.Tensor([0.0, 0.0, 0.0, 0.0])

    box_min = torch.as_tensor([xs.min(), ys.min()])
    box_max = torch.as_tensor([xs.max(), ys.max()])
    if clip:
        if im_size is None:
            raise ValueError("Image size needs to be provided for clipping.")
        box_min = clip_2d_point(box_min, im_size)  # type: ignore # noqa: F821
        box_max = clip_2d_point(box_max, im_size)  # type: ignore # noqa: F821
    return torch.hstack([box_min, box_max])


def clip_2d_point(point: torch.Tensor, im_size: torch.Tensor) -> torch.Tensor:
    """Clips a 2D point to the image.

    A point outside the image is replaced with the closest point inside the image.

    Args:
        point: An 2D point to clip.
        im_size: (width, height) of the image.
    Returns:
        The clipped point.
    """

    point_min = torch.Tensor([0, 0]).type(point.dtype)
    point_max = torch.Tensor([im_size[0] - 1, im_size[1] - 1]).type(point.dtype)
    return torch.minimum(torch.maximum(point, point_min), point_max)


def hwc_to_chw(im: npt.NDArray) -> npt.NDArray:
    """Converts a Numpy array from HWC to CHW (C = channels, H = height, W = width).

    Args:
        data: A Numpy array width dimensions in the HWC order.
    Returns:
        A Numpy array width dimensions in the CHW order.
    """

    return np.transpose(im, (2, 0, 1))


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: Optional[Any] = None,
) -> np.ndarray:
    """Resizes an image.

    Args:
      image: An input image.
      size: The size of the output image (width, height).
      interpolation: An interpolation method (a suitable one is picked if undefined).
    Returns:
      The resized image.
    """

    if interpolation is None:
        interpolation = (
            cv2.INTER_AREA if image.shape[0] >= size[1] else cv2.INTER_LINEAR
        )
    return cv2.resize(image, size, interpolation=interpolation)


def chw_to_hwc(data: np.ndarray) -> np.ndarray:
    """Converts a Numpy array from CHW to HWC (C = channels, H = height, W = width).

    Args:
        data: A Numpy array width dimensions in the CHW order.
    Returns:
        A Numpy array width dimensions in the HWC order.
    """

    return np.transpose(data, (1, 2, 0))


def crop_image(image: np.ndarray, crop_box: structs.AlignedBox2f) -> np.ndarray:
    """Crops an image.

    Args:
        image: The input HWC image.
        crop_box: The bounding box for cropping given by (x1, y1, x2, y2).
    Returns:
        Cropped image.
    """

    return image[crop_box.top : crop_box.bottom, crop_box.left : crop_box.right]


def float_to_uint8_image(im: npt.NDArray) -> npt.NDArray:
    """Scales image values from [0, 1] to [0, 255] and converts the type to uint8.

    Args:
        im: The input image with values of a floating-point type in [0, 1].
    Returns:
        The image with values of uint8 type in [0, 255].
    """

    if im.max() > 1.0:
        raise ValueError("The input image values should be in range [0, 1].")
    if im.dtype not in [float, np.float32, np.float64]:
        raise ValueError("The input image should be of a floating-point type.")
    return np.round(255.0 * im).astype(np.uint8)


def uint8_to_float_image(im: npt.NDArray) -> npt.NDArray:
    """Scales image values from [0, 255] to [0, 1] and converts the type to float.

    Args:
        im: The input image with values of uint8 type in [0, 255].
    Returns:
        The image with values of float32 type in [0, 1].
    """
    if im.dtype != np.uint8:
        raise ValueError(f"The input image should be of uint8 type. It is {im.dtype}")
    return im.astype(np.float32) / 255.0


def rgb_to_mono_image(im: npt.NDArray) -> npt.NDArray:
    """Converts an RGB image to a monochrome image.

    Ref: nimble/common/Image/Conversions/ColorR8G8B8Conversions.cpp?lines=26-29

    Args:
        im: The input RGB image of shape (h x w x 3).
    Returns:
        A monochrome image.
    """
    mono = im[..., 0] * 0.2125 + im[..., 1] * 0.7154 + im[..., 2] * 0.0721

    if im.dtype == np.uint8:
        out = np.empty(mono.shape, dtype=np.uint8)
        np.rint(mono, casting="unsafe", out=out)
        return out


def warp_depth_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_depth_image: np.ndarray,
    depth_check: bool = True,
) -> np.ndarray:
    # Copy the source depth image.
    depth_image = np.array(src_depth_image)

    # If the camera extrinsics changed, update the depth values.
    if not np.allclose(src_camera.T_world_from_eye, dst_camera.T_world_from_eye):
        # Image coordinates with valid depth values.
        valid_mask = depth_image > 0
        ys, xs = np.nonzero(valid_mask)

        # Transform the source depth image to a point cloud.
        pts_in_src = src_camera.window_to_eye(np.vstack([xs, ys]).T)
        pts_in_src *= np.expand_dims(depth_image[valid_mask] / pts_in_src[:, 2], axis=1)

        # Transform the point cloud from the source to the target camera.
        pts_in_w = src_camera.eye_to_world(pts_in_src)
        pts_in_trg = dst_camera.world_to_eye(pts_in_w)

        depth_image[valid_mask] = pts_in_trg[:, 2]

    # Warp the depth image to the target camera.
    return warp_image(
        src_camera=src_camera,
        dst_camera=dst_camera,
        src_image=depth_image,
        interpolation=cv2.INTER_NEAREST,
        depth_check=depth_check,
    )


def warp_images_with_adaptive_interpolation(
    images: Dict[str, npt.NDArray],
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
) -> Dict[str, npt.NDArray]:
    """
    Warp images with adaptive interpolation.

    The interpolation is set based on the image type (RGB, monochrome, depth, mask) and,
    in case of some image types, on whether the image is being up- or down-sampled.

    Parameters
    ----------
    images: Dictionary of images to crop. Keys must start with one of:
        "rgb", "monochrome", "depth", "mask". All images must have the same
        resolution and the same camera model (`src_camera`).
    src_camera: Source camera model.
    dst_camera: Destination camera model.

    Returns
    -------
    Cropped images.
    """

    # Crop the images.
    crops: Dict[str, npt.NDArray] = {}
    for image_key, image in images.items():
        # All images are assumed to be of the same size defined by the camera model.
        assert (
            image.shape[0] == src_camera.height and image.shape[1] == src_camera.width
        )

        # Crop RGB/monochrome image.
        if image_key.startswith("rgb") or image_key.startswith("monochrome"):
            # Compute diagonals of the corresponding regions in the source and
            # destination cameras.
            f = 0.5 * (dst_camera.f[0] + dst_camera.f[1])
            corners_in_dst = np.array(
                [[0, 0, f], [dst_camera.width, dst_camera.height, f]], dtype=np.float32
            )
            corners_in_world = dst_camera.window_to_world3(corners_in_dst)
            corners_in_src = src_camera.world_to_window(corners_in_world)
            src_diagonal = np.linalg.norm(corners_in_src[1] - corners_in_src[0])
            dst_diagonal = np.linalg.norm(corners_in_dst[1] - corners_in_dst[0])

            # Use area interpolation when downsampling and bilinear when upsampling
            # (this is the default in OpenCV).
            if src_diagonal >= dst_diagonal:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR

            crop = warp_image(
                src_camera=src_camera,
                dst_camera=dst_camera,
                src_image=image,
                interpolation=interpolation,
            )

            # Make sure the crop has the last dimension preserved (if it was present
            # in the input image; warp_image removes last singleton dimension).
            if image.ndim == 3 and crop.ndim == 2:
                crop = np.expand_dims(crop, -1)

            crops[image_key] = crop

            # Ensure the last dimension for channels exists for single-channel
            # images (it is dropped in `warp_image`).
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)

        # Crop depth image.
        elif image_key.startswith("depth"):
            crops[image_key] = warp_depth_image(
                src_camera=src_camera,
                dst_camera=dst_camera,
                src_depth_image=image,
            )

        # Crop mask image.
        elif image_key.startswith("mask"):
            crops[image_key] = warp_image(
                src_camera=src_camera,
                dst_camera=dst_camera,
                src_image=image,
                interpolation=cv2.INTER_NEAREST,
            )

        else:
            raise ValueError(f"Unknown image key: {image_key}")

        assert crops[image_key].shape[0] == dst_camera.height
        assert crops[image_key].shape[1] == dst_camera.width

    return crops


def warp_image(
    src_camera: structs.CameraModel,
    dst_camera: structs.CameraModel,
    src_image: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    depth_check: bool = True,
    factor_to_downsample: int = 1,
) -> np.ndarray:
    """
    Warp an image from the source camera to the destination camera.

    Parameters
    ----------
    src_camera :
        Source camera model
    dst_camera :
        Destination camera model
    src_image :
        Source image
    interpolation :
        Interpolation method
    depth_check :
        If True, mask out points with negative z coordinates
    factor_to_downsample :
        If this value is greater than 1, it will downsample the input image prior to warping.
        This improves downsampling performance, in an attempt to replicate
        area interpolation for crop+undistortion warps.
    """

    if factor_to_downsample > 1:
        src_image = cv2.resize(
            src_image,
            (
                int(src_image.shape[1] / factor_to_downsample),
                int(src_image.shape[0] / factor_to_downsample),
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Rescale source camera
        src_camera = adjust_camera_model(src_camera, factor_to_downsample)  # type: ignore # noqa: F821

    W, H = dst_camera.width, dst_camera.height
    px, py = np.meshgrid(np.arange(W), np.arange(H))
    dst_win_pts = np.column_stack((px.flatten(), py.flatten()))

    dst_eye_pts = dst_camera.window_to_eye(dst_win_pts)
    world_pts = dst_camera.eye_to_world(dst_eye_pts)
    src_eye_pts = src_camera.world_to_eye(world_pts)
    src_win_pts = src_camera.eye_to_window(src_eye_pts)

    # Mask out points with negative z coordinates
    if depth_check:
        mask = src_eye_pts[:, 2] < 0
        src_win_pts[mask] = -1

    src_win_pts = src_win_pts.astype(np.float32)

    map_x = src_win_pts[:, 0].reshape((H, W))
    map_y = src_win_pts[:, 1].reshape((H, W))

    return cv2.remap(src_image, map_x, map_y, interpolation)


def filter_noisy_detections(
    boxes: torch.Tensor,
    masks: torch.Tensor,
    width: int,
    height: int,
    min_box_size: float,
    min_mask_size: float,
    debug: bool = False,
) -> torch.Tensor:
    """Filters out noisy detections based on box and mask sizes."""
    img_area = width * height
    box_areas = box_area(boxes) / img_area
    mask_areas = masks.sum(dim=(1, 2)) / img_area
    keep_idxs = torch.logical_and(
        box_areas > min_box_size**2,
        mask_areas > min_mask_size,
    )
    if debug:
        num_noisy_detections = len(keep_idxs) - keep_idxs.sum()
        logger.info(f"CNOS: Filtered {num_noisy_detections} noisy detections.")
    return keep_idxs
