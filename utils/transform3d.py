# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict

import math
import numpy as np
import torch
from typing import Sequence, Tuple, TypeVar, Union, Optional
from scipy.spatial.transform import Rotation

from einops import rearrange


AnyTensor = TypeVar("AnyTensor", np.ndarray, "torch.Tensor")
_EPS = 1e-12


def transform_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Transform an array of 3D points with an SE3 transform (rotation and translation).

    *WARNING* this function does not support arbitrary affine transforms that also scale
    the coordinates (i.e., if a 4x4 matrix is provided as input, the last row of the
    matrix must be `[0, 0, 0, 1]`).

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points [..., 3]

    Returns:
        Transformed points [..., 3]
    """
    return rotate_points(matrix, points) + matrix[..., :3, 3]


def rotate_points(matrix: AnyTensor, points: AnyTensor) -> AnyTensor:
    """
    Rotates an array of 3D points with an affine transform,
    which is equivalent to transforming an array of 3D rays.

    *WARNING* This ignores the translation in `m`; to transform 3D *points*, use
    `transform_points()` instead.

    Note that we specifically optimize for ndim=2, which is a frequent
    use case, for better performance. See n388920 for the comparison.

    Matrix or points can be batched as long as the batch shapes are broadcastable.

    Args:
        matrix: SE3 transform(s)  [..., 3, 4] or [..., 4, 4]
        points: Array of 3d points or 3d direction vectors [..., 3]

    Returns:
        Rotated points / direction vectors [..., 3]
    """
    if matrix.ndim == 2:
        return (points.reshape(-1, 3) @ matrix[:3, :3].T).reshape(points.shape)
    else:
        return (matrix[..., :3, :3] @ points[..., None]).squeeze(-1)


def gen_look_at_matrix(
    orig_camera_from_world: np.ndarray,
    center: np.ndarray,
    camera_angle: float = 0,
    return_camera_from_world: bool = True,
) -> np.ndarray:
    """
    Rotates the input camera such that the new transformation align the z-direction to the provided point in world.
    Args:
      camera_angle is used to apply a roll rotation around the new z
      return_camera_from_world is used to return the inverse

    Returns:
        world_from_aligned_camera or aligned_camera_from_world
    """

    center_local = transform_points(orig_camera_from_world, center)
    z_dir_local = center_local / np.linalg.norm(center_local)
    delta_r_local = from_two_vectors(
        np.array([0, 0, 1], dtype=center.dtype), z_dir_local
    )
    orig_world_from_camera = np.linalg.inv(orig_camera_from_world)

    world_from_aligned_camera = orig_world_from_camera.copy()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ delta_r_local
    )

    # Locally rotate the z axis to align with the camera angle
    z_local_rot = Rotation.from_euler("z", camera_angle, degrees=True).as_matrix()
    world_from_aligned_camera[0:3, 0:3] = (
        world_from_aligned_camera[0:3, 0:3] @ z_local_rot
    )

    if return_camera_from_world:
        return np.linalg.inv(world_from_aligned_camera)
    return world_from_aligned_camera


def from_two_vectors(a_orig: np.ndarray, b_orig: np.ndarray) -> np.ndarray:
    # Convert the vectors to unit vectors.
    a = normalized(a_orig)
    b = normalized(b_orig)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    v_mat = skew_matrix(v)

    rot = (
        np.eye(3, 3, dtype=a_orig.dtype)
        + v_mat
        + np.matmul(v_mat, v_mat) * (1 - c) / (max(s * s, 1e-15))
    )

    return rot


def skew_matrix(v: np.ndarray) -> np.ndarray:
    res = np.array(
        [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=v.dtype
    )
    return res


def rotation_matrix_numpy(angle: float, direction: np.ndarray) -> np.ndarray:
    """Return a homogeneous transformation matrix [4x4] to rotate a point around the
    provided direction by a mangnitude set by angle.

    Args:
        angle: Angle to rotate around axis [rad].
        direction: Direction vector (3-vector, does not need to be normalized)

    Returns:
        M: A 4x4 matrix with the rotation component set and translation to zero.

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = normalized(direction[:3])
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float64
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float64,
    )
    M = np.identity(4)
    M[:3, :3] = R
    return M


def as_4x4(a: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """
    Append [0,0,0,1] to convert 3x4 matrices to a 4x4 homogeneous matrices

    If the matrices are already 4x4 they will be returned unchanged.
    """
    if a.shape[-2:] == (4, 4):
        if copy:
            a = np.array(a)
        return a
    if a.shape[-2:] == (3, 4):
        return np.concatenate(
            (
                a,
                np.broadcast_to(
                    np.array([0, 0, 0, 1], dtype=a.dtype), a.shape[:-2] + (1, 4)
                ),
            ),
            axis=-2,
        )
    raise ValueError("expected 3x4 or 4x4 affine transform")


def normalized(v: AnyTensor, axis: int = -1, eps: float = 5.43e-20) -> AnyTensor:
    """
    Return a unit-length copy of vector(s) v

    Parameters
    ----------
    axis : int = -1
        Which axis to normalize on

    eps
        Epsilon to avoid division by zero. Vectors with length below
        eps will not be normalized. The default is 2^-64, which is
        where squared single-precision floats will start to lose
        precision.
    """
    d = np.maximum(eps, (v * v).sum(axis=axis, keepdims=True) ** 0.5)
    return v / d


def inverse_Rt(
    R_A_B: torch.Tensor, t_A_B: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper for replacement of torch.inverse() function which cannot run on device
    R_A_B: [*, 3, 3]
    t_A_B: [*, 3]
    return: (R_A_B, [*, 3])
    """
    torch._assert(R_A_B.shape[-2:] == (3, 3), "Invalid rotation matrix")
    torch._assert(t_A_B.shape[-1] == 3, "Invalid translation vector")
    R_B_A = R_A_B.transpose(-2, -1)
    t_B_A = -R_B_A.matmul(t_A_B.unsqueeze(-1)).squeeze(-1)
    return R_B_A.clone(), t_B_A.clone()


def inverse_se3(T_A_B: torch.Tensor) -> torch.Tensor:
    """Replacement for torch.inverse() function which cannot run on device
    Works on a tensor of 4x4 transformation matrices and returns the respective inverse matrices.
    Assumes input dimension is [*, 4, 4] wheras the 4th row is always [0, 0, 0, 1]
    and the first three rows represent a valid rotation and translation pair.
    """
    torch._assert(T_A_B.shape[-2:] == (4, 4), "Invalid SE3 matrix")

    R_A_B = T_A_B[..., :3, :3]
    t_A_B = T_A_B[..., :3, 3]
    R_B_A, t_B_A = inverse_Rt(R_A_B, t_A_B)
    T_B_A_3x4 = torch.cat((R_B_A, t_B_A.unsqueeze(-1)), dim=-1)
    T_B_A_4x4 = torch.cat((T_B_A_3x4, T_A_B[..., -1:, :]), dim=-2)
    return T_B_A_4x4


def inverse_se3_numpy(
    matrix: np.ndarray,
) -> np.ndarray:
    """Return inverse of square 4x4 transformation matrix."""

    inverse = np.empty(matrix.shape, dtype=matrix.dtype)
    # R = R.T
    inverse[..., :3, :3] = np.swapaxes(matrix[..., :3, :3], axis1=-1, axis2=-2)
    # t = -R.T * t
    np.matmul(-inverse[..., :3, :3], matrix[..., :3, 3:4], out=inverse[..., :3, 3:4])
    # Homogenous dimension
    inverse[..., 3, 0:3] = 0
    inverse[..., 3, 3] = 1
    return inverse


def Rt_to_4x4(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Helper to convert a batch of rotation matrices and translation vectors to
    4x4 transformation matrices.

    Args:
        R: Rotation matrices of shape (..., 3, 3)
        t: Translation vectors of shape (..., 3)

    Returns:
        T: Transformation matrices of shape (..., 4, 4)
    """
    torch._assert(R.shape[-2:] == (3, 3), "Invalid rotation matrices")
    torch._assert(t.shape[-1] == 3, "Invalid translation vectors")
    *dim, _, _ = R.shape
    T = torch.zeros((*dim, 4, 4), dtype=R.dtype, device=R.device)
    T[..., 3, 3] = 1
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def Rt_to_4x4_numpy(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Helper to convert a batch of rotation matrices and translation vectors to
    4x4 transformation matrices.

    Args:
        R: Rotation matrices of shape (..., 3, 3)
        t: Translation vectors of shape (..., 3)

    Returns:
        T: Transformation matrices of shape (..., 4, 4)
    """
    assert R.shape[-2:] == (3, 3), "Invalid rotation matrices"
    assert t.shape[-1] == 3, "Invalid translation vectors"
    *dim, _, _ = R.shape
    T = np.zeros((*dim, 4, 4), dtype=R.dtype)
    T[..., 3, 3] = 1
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def perturb_transforms_numpy(
    poses: np.ndarray,
    angle_std_in_rad: Union[float, np.ndarray, Sequence[float]],
    trans_std: Union[float, np.ndarray, Sequence[float]],
    perturb_on_rhs: bool = True,
) -> np.ndarray:
    """Perturb a 4x4 transformation matrices by adding normally distributed
    noise to the rotation and translation components.

    Args:
        poses: The 4x4 transformation matrices to be perturbed.
        angle_std_in_rad: The standard deviation of the angular perturbations.
        trans_std: The standard deviation of the translational perturbations.
        perturb_on_rhs: Whether the perturbation should be applied on the right hand side.

    Returns:
        The perturbed transformations.
    """
    h, w = poses.shape
    assert h == 4 and w == 4
    if isinstance(angle_std_in_rad, (float)):
        angle_std_in_rad = np.array([1, 1, 1]) * angle_std_in_rad
    if isinstance(trans_std, (float)):
        trans_std = np.array([1, 1, 1]) * trans_std
    assert len(angle_std_in_rad) == 3
    assert len(trans_std) == 3

    zero_mean = np.zeros(3)
    angle_std_in_rad = np.array(angle_std_in_rad)
    trans_std = np.array(trans_std)
    rotvec_offset = np.stack(np.random.normal(loc=zero_mean, scale=angle_std_in_rad))
    pos_offset = np.stack(np.random.normal(loc=zero_mean, scale=trans_std))
    T = Rt_to_4x4_numpy(
        so3_exp_map_numpy(rotvec_offset),  # type: ignore  # noqa: F821
        pos_offset,
    )
    if perturb_on_rhs:
        return poses @ T
    else:
        return T @ poses


def cont6d_from_rotation_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """Creates a 6D representation of rotations from a 3x3 rotation matrix.

    Ref: Zhou et al., On the Continuity of Rotation Representations in Neural
    Networks, CVPR19, https://zhouyisjtu.github.io/project_rotation/rotation.html.

    Args:
        rotations: 3x3 rotation matrices of
            shape [batch_size, 3, 3].
    Returns:
        6D continuous representation of rotations of shape [batch_size, 6].
    """
    assert rotations.shape[-2:] == (3, 3)
    x = rotations[:, :, 0]
    z = rotations[:, :, 1]

    # x, z should be already normalized
    batch_size = rotations.shape[0]
    device = rotations.device
    assert torch.allclose(torch.norm(x, dim=1), torch.ones(batch_size, device=device))
    assert torch.allclose(torch.norm(z, dim=1), torch.ones(batch_size, device=device))

    return torch.stack((x, z), dim=1).reshape(batch_size, 6)


def rotation_matrix_from_cont6d(rotations: torch.Tensor) -> torch.Tensor:
    """Creates a 3x3 rotation matrix from the 6D representation of Zhou et al.

    Ref: Zhou et al., On the Continuity of Rotation Representations in Neural
    Networks, CVPR19, https://zhouyisjtu.github.io/project_rotation/rotation.html.

    Code from: https://github.com/papagina/RotationContinuity

    Args:
        rotations: 6D continuous representation of rotations of shape [batch_size, 6].
    Returns:
        3x3 rotation matrices of shape [batch_size, 3, 3].
    """

    x_raw = rotations[..., :3]
    y_raw = rotations[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)

    return torch.stack((x, y, z), -1)


def axis_angle(axis, theta):
    """Construct from axis and angle of rotation"""
    axis /= np.linalg.norm(axis)

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    ex = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )

    rotation = (
        costheta * np.eye(3)
        + (1.0 - costheta) * np.dot(axis.reshape(3, 1), axis.reshape(1, 3))
        + sintheta * ex
    )
    return rotation


def axis_angle2(axis, theta):
    """Compute a rotation matrix from an axis and an angle.
    Returns 3x3 Matrix.
    Is the same as transformations.rotation_matrix(theta, axis).
    cfo, 2015/08/13

    """
    if theta * theta > _EPS:
        wx = axis[0]
        wy = axis[1]
        wz = axis[2]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        c_1 = 1.0 - costheta
        wx_sintheta = wx * sintheta
        wy_sintheta = wy * sintheta
        wz_sintheta = wz * sintheta
        C00 = c_1 * wx * wx
        C01 = c_1 * wx * wy
        C02 = c_1 * wx * wz
        C11 = c_1 * wy * wy
        C12 = c_1 * wy * wz
        C22 = c_1 * wz * wz
        R = np.zeros((3, 3), dtype=np.float64)
        R[0, 0] = costheta + C00
        R[1, 0] = wz_sintheta + C01
        R[2, 0] = -wy_sintheta + C02
        R[0, 1] = -wz_sintheta + C01
        R[1, 1] = costheta + C11
        R[2, 1] = wx_sintheta + C12
        R[0, 2] = wy_sintheta + C02
        R[1, 2] = -wx_sintheta + C12
        R[2, 2] = costheta + C22
        return R
    else:
        raise ValueError("Axis angle is too small.")


def project_3d_points_pinhole_numpy(
    points: np.ndarray,
    intrinsics: np.ndarray,
    check_if_point_behind_camera: Optional[bool] = False,
    invalid_pixel_value: Optional[float] = -1,
    eps: Optional[float] = 1e-6,
) -> np.ndarray:
    """Projects 3D points to the image plane using pinhole intrinsics.

    Args:
        points (torch.Tensor): Points to project (batch_size x num_points x 3) or
            (num_points x 3)
        intrinsics: Pinhole camera intrinsics (batch_size x 3 x 3) or (3x3). May be the
            intrinsic matrix or the intrinsic matrix composed with an additional transform.
        check_if_points_behind_camera: If true, will check whether the z component of a point is
            negative and replace it with invalid_pixel_value
        invalid_value: Value to assign to any points behind the camera.
        eps: A small value used to avoid division by zero.

    Returns:
        pixels: Pixel coordinates of the projected points (batch_size x num_points x 2)
    """

    assert points.shape[-1] == 3
    assert intrinsics.shape[-2:] == (3, 3)

    # Project the points.
    uvs = (intrinsics @ points.swapaxes(-1, -2)).swapaxes(-1, -2)

    # Make sure we do not divide by zero.
    # pyre-fixme[6]: For 2nd argument expected `Union[_SupportsArray[dtype[typing.Any...
    z = np.maximum(uvs[..., -1], eps)
    pxs = uvs[..., :2] / np.expand_dims(z, -1)

    # Replace any points behind the camera with an invalid pixel value.
    if check_if_point_behind_camera:
        pxs = np.where(
            np.tile(
                np.expand_dims(uvs[..., -1] < 0, -1),
                [1] * (len(points.shape) - 1) + [2],  # Repeat twice in last dim.
            ),
            np.ones(pxs.shape) * invalid_pixel_value,
            pxs,
        )
    return pxs


def project_3d_points_pinhole(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    check_if_point_behind_camera: Optional[bool] = False,
    invalid_pixel_value: Optional[float] = -1,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    """Projects 3D points to the image plane using pinhole intrinsics.

    Args:
        points (torch.Tensor): Points to project (batch_size x num_points x 3) or
            (num_points x 3)
        intrinsics: Pinhole camera intrinsics (batch_size x 3 x 3) or (3x3). May be the
            intrinsic matrix or the intrinsic matrix composed with an additional transform.
        check_if_points_behind_camera: If true, will check whether the z component of a point is
            negative and replace it with invalid_pixel_value
        invalid_value: Value to assign to any points behind the camera.
        eps: A small value used to avoid division by zero.

    Returns:
        pixels: Pixel coordinates of the projected points (batch_size x num_points x 2)
    """
    assert points.shape[-1] == 3, "Input points must be of dimension BxNx3"
    assert intrinsics.shape[-2:] == (3, 3), f"{intrinsics.shape} is not Bx3x3"

    # Project the points.
    uvs = (intrinsics @ points.transpose(-1, -2)).transpose(-1, -2)

    # Make sure we do not divide by zero.
    z = torch.clamp(uvs[..., -1], min=eps)
    pxs = uvs[..., :2] / z.unsqueeze(-1)

    # Replace any points behind the camera with an invalid pixel value.
    if check_if_point_behind_camera:
        pxs = torch.where(
            (uvs[..., -1] < 0)
            .unsqueeze(-1)
            .expand([-1] * (len(points.shape) - 1) + [2]),  # Repeat twice in last dim.
            # pyre-fixme[58]: `*` is not supported for operand types `Tensor` and
            #  `Optional[float]`.
            torch.ones_like(pxs) * invalid_pixel_value,
            pxs,
        )
    return pxs


def invert_intrinsics(
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """
    Inverts a batch of camera intrinsics matrices.

    Args:
      intrinsics: A torch tensor of shape (num_frames, 3, 3) representing the camera intrinsics matrices to be inverted.

    Returns:
      A torch tensor of the same shape as the input containing the inverted intrinsics matrices.

    """

    # Calculate inverses of a and b
    ones_tensor = torch.ones(intrinsics.shape[0], device=intrinsics.device)
    inv_a = ones_tensor / intrinsics[..., 0, 0]
    inv_b = ones_tensor / intrinsics[..., 1, 1]

    # Invert the matrix element-wise
    intrinsics_inverse = intrinsics.clone()
    intrinsics_inverse[..., 0, 0] = inv_a
    intrinsics_inverse[..., 1, 1] = inv_b
    intrinsics_inverse[..., 0, 2] = -inv_a * intrinsics[..., 0, 2]
    intrinsics_inverse[..., 1, 2] = -inv_b * intrinsics[..., 1, 2]

    return intrinsics_inverse


def homogeneous_pixel_grid(
    width: int, height: int, offset_center_pixel: float = 0.5
) -> torch.Tensor:
    """
    Create a grid of homogeneous points for every pixel in an image of size (height, width).

    Returns:
        An tensor of shape `(height, width, 3)` containing the homogeneous points for every pixel in an image of size (height, width).
    """
    # Create a grid of coordinates (x, y) for each pixel in the image
    x = torch.linspace(0, width - 1, width) + offset_center_pixel
    y = torch.linspace(0, height - 1, height) + offset_center_pixel
    x, y = torch.meshgrid(x, y)
    one = torch.ones_like(x)

    # Tranpose to get the correct shape.
    homogeneous_pixel = torch.stack([x, y, one], dim=-1).transpose(1, 0)
    assert homogeneous_pixel.shape == (height, width, 3)
    return homogeneous_pixel


def get_3d_points_from_depth(
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets 3D points from depth.
    Args:
        depth: [batch_size, height, width] depth image.
        intrinsics: [batch_size, 3, 3] camera intrinsics.
    Returns:
        pixels: 3D points [batch_size, height, width, 3] in camera coordinates.
        3d_points: 3D points [batch_size, height, width, 3] in camera coordinates.
    """
    (batch_size, height, width) = depths.shape
    assert depths.device == intrinsics.device, (
        "Depth and intrinsics must be on the same device."
    )

    # Get the homogeneous points if not provided.
    homogeneous_pixels = homogeneous_pixel_grid(width, height)
    homogeneous_pixels = homogeneous_pixels.to(intrinsics.device)

    # Repeat homogeneous pixels for each batch element.
    homogeneous_pixels = homogeneous_pixels.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Get 3D points of template in camera coordinates.
    inv_intrinsics = invert_intrinsics(intrinsics)  # B x 3 x 3

    # Get uncalibrated pixels.
    pixels_uncalib = inv_intrinsics @ rearrange(
        homogeneous_pixels, "b h w c -> b c (h w)"
    )

    # Convert back to B x H x W x C.
    pixels_uncalib = rearrange(pixels_uncalib, "b c (h w) -> b h w c", h=height)

    # Multiply with the depth to get the 3D points.
    points3d = pixels_uncalib * depths.unsqueeze(-1)

    # Return both pixels and points3d as we need both to define 3D-to-2D correspondences.
    pixels = homogeneous_pixels[:, :, :, :2]
    return pixels, points3d
