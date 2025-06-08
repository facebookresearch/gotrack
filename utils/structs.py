# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


from __future__ import annotations
import itertools
from typing import (
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    Any,
)

import numpy as np
import torch

import abc
import math

from utils import torch_helpers, transform3d

ArrayData = Union[np.ndarray, torch.Tensor]
Color = Tuple[float, float, float]


class RigidTransform(NamedTuple):
    """An N-D rigid transformation.

    R: An NxN rotation matrix.
    t: An Nx1 translation vector
    """

    R: ArrayData
    t: ArrayData


ObjectPose = RigidTransform


class ObjectAnnotation(NamedTuple):
    """An annotation of a single object instance.

    dataset: The name of the dataset to which the object belogs (e.g. "tless").
    lid: A local ID (e.g. BOP IDs: https://bop.felk.cvut.cz/datasets/).
    uid: A globally unique ID (e.g. an FBID of an object model saved in Gaia).
    pose: The object pose (a transformation from the model to the world).
    masks_modal: Per-view 2D modal masks (of the same size as the image).
    masks_amodal: Per-view 2D modal masks (of the same size as the image).
    boxes_modal: Per-view 2D modal bounding boxes (of the same size as the image).
    boxes_amodal: Per-view 2D amodal bounding boxes (of the same size as the image).
    active: Binary signal whether the object is active (used for action
        recognition, e.g., for the Toy Assembly demo).
    visibilities: Per-view fractions of the object silhouette that is visible.
    """

    dataset: str
    lid: int
    uid: Optional[str] = None
    pose: Optional[ObjectPose] = None
    masks_modal: Optional[ArrayData] = None
    masks_amodal: Optional[ArrayData] = None
    boxes_modal: Optional[ArrayData] = None
    boxes_amodal: Optional[ArrayData] = None
    active: Optional[bool] = None
    visibilities: Optional[ArrayData] = None
    score: Optional[float] = None


class ImageObjectAnnotations(NamedTuple):
    """Annotations of objects in an image.

    Having annotations of all objects from an image in one structure is convenient
    for single-image methods such as Mask R-CNN.

    labels: Class labels of shape (num_objs).
    poses: 3D transformation matrices of shape (num_objs, 4, 4). The
        transformations are from the model to the camera space.
    masks: Object masks of shape (num_objs, im_height, im_width).
    boxes: 2D bounding boxes of shape (num_objs, 4).
    active_labels: Active labels of shape (num_objs).
    visibilities: Visible fractions of shape (num_objs).
    """

    labels: Optional[ArrayData] = None
    poses: Optional[ArrayData] = None
    masks: Optional[ArrayData] = None
    boxes: Optional[ArrayData] = None
    active_labels: Optional[ArrayData] = None
    visibilities: Optional[ArrayData] = None


class SceneObservation(NamedTuple):
    """A data sample/unit of a Torch dataset.

    image: Color/monochrome images of one of the following shapes:
        (1) (num_frames, num_views, num_channels, im_height, im_width)
        (2) (num_frames, num_views, im_height, im_width, num_channels)
    depth_image: Depth images of shape (num_frames, num_views, im_height, im_width).
    camera: Per-view camera properties (cameras[i][j] are camera properties for
        view j of frame i).
    objects_anno: Per-frame object annotations (objects_anno[i][o] are annotations
        for object o in frame i).
    time: Any time that related to the observation: timestamp or run-time.
    """

    scene_id: int
    im_id: int
    image: Optional[ArrayData] = None
    depth_image: Optional[ArrayData] = None
    camera: Optional[List[List[CameraModel]]] = None
    objects_anno: Optional[List[List[ObjectAnnotation]]] = None
    time: Optional[float] = None


class AlignedBox2f:
    """
    An 2D axis aligned box in floating point.

    Assumptions:
    * The origin is at top-left corner
    * `right` and `bottom` are not inclusive in the region, i.e. the width and
        height can be simply calculated by `right - left` and `bottom - top`.
    * This is the implementation of definition 2 in this doc:
        https://docs.google.com/document/d/14YNR6ebjMgnPaeP0-RsNTEjyCHznwRLDqWmHKBNXMLo/edit
    """

    def __init__(self, left: float, top: float, right: float, bottom: float):
        """Initializes the bounding box given (left, top, right, bottom)"""
        self._left: float = left
        self._top: float = top
        self._right: float = right
        self._bottom: float = bottom

    def __repr__(self):
        return f"AlignedBox2f(left: {self._left}, top: {self._top}, right: {self._right}, bottom: {self._bottom})"

    @property
    def left(self) -> float:
        """Left of the aligned box on x-axis."""
        return self._left

    @property
    def top(self) -> float:
        """Top of the aligned box on y-axis."""
        return self._top

    @property
    def right(self) -> float:
        """Right of the aligned box on x-axis."""
        return self._right

    @property
    def bottom(self) -> float:
        """Bottom of the aligned box on y-axis."""
        return self._bottom

    @property
    def width(self) -> float:
        """Width of the aligned box.

        Returns:
            Width computed by right - left
        """
        return self.right - self.left

    @property
    def height(self) -> float:
        """Height of the aligned box.

        Returns:
            Height computed by bottom - top
        """
        return self.bottom - self.top

    def pad(self, width: float, height: float) -> AlignedBox2f:
        """Pads the region by extending `width` and `height` on four sides.

        Args:
            width (float): length to pad on left and right sides
            height (float): length to pad on top and bottom sides
        Returns:
            a new AlignedBox2f object with padded region
        """
        return AlignedBox2f(
            self.left - width,
            self.top - height,
            self.right + width,
            self.bottom + height,
        )

    def array_ltrb(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,):  (left, top, right, bottom).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return np.array([self.left, self.top, self.right, self.bottom])

    def array_ltwh(self) -> np.ndarray:
        """Converts the box into a float np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            a float np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return np.array([self.left, self.top, self.width, self.height])

    def int_array_ltrb(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, right, bottom)
        """
        return self.array_ltrb().astype(int)

    def int_array_ltwh(self) -> np.ndarray:
        """Converts the box into an int np.ndarray of shape (4,): (left, top, width, height).

        Returns:
            an int np.ndarray of shape (4,) representing (left, top, width, height)
        """
        return self.array_ltwh().astype(int)

    def round(self) -> AlignedBox2f:
        """Rounds the float values to int.

        Returns:
            a new AlignedBox2f object with rounded values (still float)
        """
        return AlignedBox2f(
            np.round(self.left),
            np.round(self.top),
            np.round(self.right),
            np.round(self.bottom),
        )

    def clip(self, boundary: AlignedBox2f) -> AlignedBox2f:
        """Clips the region by the boundary

        Args:
            boundary (AlignedBox2f): boundary of box to be clipped
                (boundary.left: minimum left / right value,
                 boundary.top: minimum top / bottom value,
                 boundary.right: maximum left / right value,
                 boundary.bottom: maximum top / bottom value)
        Returns:
            a new clipped AlignedBox2f object
        """
        return AlignedBox2f(
            min(max(self.left, boundary.left), boundary.right),
            min(max(self.top, boundary.top), boundary.bottom),
            min(max(self.right, boundary.left), boundary.right),
            min(max(self.bottom, boundary.top), boundary.bottom),
        )


class CameraModel(abc.ABC):
    """
    Parameters
    ----------
    width, height : int
        Size of the sensor window

    f : float or tuple(float, float)
        Focal length

    c : tuple(float, float)
        Optical center in window coordinates

    T_world_from_eye : np.ndarray
        Camera's position and orientation in world space, represented as
        a 3x4 or 4x4 matrix.

        The matrix be a rigid transform (only rotation and translation).

        You can change a camera's extrinsics after construction by
        assigning to or modifying this matrix.

    serial : string
        Arbitrary string identifying the specific camera.

    Attributes
    ----------
    Most attributes are the same as constructor parameters.

    zmin
        Smallest z coordinate of a visible unit-length eye vector.
        (Unit-length) eye rays with z < zmin are known not to be visible
        without doing any extra work.

        This check is needed because for points far outside the window,
        as the distortion polynomial explodes and incorrectly maps some
        distant points back to coordinates inside the window.

        `zmin = cos(max_angle)`

    max_angle
        Maximum angle from +Z axis of a visible eye vector.
    """

    width: int
    height: int

    f: Tuple[float, float]
    c: Tuple[float, float]

    T_world_from_eye: np.ndarray

    _zmin: Optional[float]
    _max_angle: Optional[float]

    def __init__(
        self,
        width,
        height,
        f,
        c,
        T_world_from_eye=None,
        serial="",
    ):  # pylint: disable=super-init-not-called (see issue 4790 on pylint github)
        self.width = width
        self.height = height
        self.serial = serial

        # f can be either a scalar or (fx,fy) pair. We only fit scalars,
        # but may load (fx, fy) from a stored file.
        self.f = tuple(np.broadcast_to(f, 2))
        self.c = tuple(c)

        if T_world_from_eye is None:
            self.T_world_from_eye = np.eye(4)
        else:
            self.T_world_from_eye = transform3d.as_4x4(T_world_from_eye, copy=True)
            if (
                np.abs(
                    (self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3]
                    - np.eye(3)
                ).max()
                >= 1.0e-5
            ):
                info_str = "camera T_world_from_eye must be a rigid transform\n"
                info_str = info_str + "T\n{}\n".format(self.T_world_from_eye.T)
                info_str = info_str + "(T*T_t - I).max()\n{}\n".format(
                    np.abs(
                        (self.T_world_from_eye.T @ self.T_world_from_eye)[:3, :3]
                        - np.eye(3)
                    ).max()
                )
                raise ValueError(info_str)

        # These are computed only when needed, use the getters zmin() and max_angle()
        self._zmin = None
        self._max_angle = None

    def __repr__(self):
        return (
            f"{type(self).__name__}({self.width}x{self.height}, f={self.f} c={self.c}"
        )

    def to_json(self):
        js = {}
        js["ImageSizeX"] = self.width
        js["ImageSizeY"] = self.height
        js["T_WorldFromCamera"] = self.T_world_from_eye.tolist()

        js["ModelViewMatrix"] = np.linalg.inv(self.T_world_from_eye).tolist()

        js["fx"], js["fy"] = np.asarray(self.f).tolist()
        js["cx"], js["cy"] = np.asarray(self.c).tolist()

        return js

    def copy(
        self,
        T_world_from_eye=None,
    ):
        """Return a copy of this camera

        Arguments
        ---------
        T_world_from_eye : 4x4 np.ndarray
            Optional new extrinsics for the new camera model.
            Default is to copy this camera's extrinsics.

        serial : str
            Optional replacement serial number.
            Default is to copy this camera's serial number.
        """
        return self.crop(
            0,
            0,
            self.width,
            self.height,
            T_world_from_eye=T_world_from_eye,
        )

    def compute_zmin(self):
        corners = (
            np.array(
                [[0, 0], [self.width, 0], [0, self.height], [self.width, self.height]]
            )
            - 0.5
        )
        self._zmin = self.window_to_eye(corners)[:, 2].min()
        self._max_angle = np.arccos(self._zmin)

    def zmin(self):
        if self._zmin is None:
            self.compute_zmin()
        return self._zmin

    def max_angle(self):
        if self._max_angle is None:
            self.compute_zmin()
        return self._max_angle

    def world_to_window(self, v):
        """Project world space points to 2D window coordinates"""
        return self.eye_to_window(self.world_to_eye(v))

    def world_to_window3(self, v):
        """Project world space points to 3D window coordinates (uv + depth)"""
        return self.eye_to_window3(self.world_to_eye(v))

    @staticmethod
    def project(v):
        # map to [x/z, y/z]
        assert v.shape[-1] == 3
        return v[..., :2] / v[..., 2, None]

    @staticmethod
    def unproject(p):
        # map to [u,v,1] and renormalize
        assert p.shape[-1] == 2
        x, y = np.moveaxis(p, -1, 0)
        v = np.stack((x, y, np.ones(shape=x.shape, dtype=x.dtype)), axis=-1)
        v = transform3d.normalized(v, axis=-1)
        return v

    @staticmethod
    def project3(v):
        # map to [x/z, y/z, z]
        x, y, z = np.moveaxis(v, -1, 0)
        return np.stack([x / z, y / z, z], axis=-1)

    @staticmethod
    def unproject3(p):
        # map to [p*z, v*z, z]
        x, y, z = np.moveaxis(p, -1, 0)
        return np.stack((x * z, y * z, z), axis=-1)

    def pos(self):
        """Return world position of camera"""
        return self.T_world_from_eye[:3, 3]

    def orient(self):
        """Return world orientation of camera as 3x3 matrix"""
        return self.T_world_from_eye[:3, :3]

    def window_to_world_ray(self, w):
        """
        Unproject 2D window coordinates to world rays.

        Returns a tuple of (origin, direction)
        """
        v = transform3d.rotate_points(self.T_world_from_eye, self.window_to_eye(w))
        o = np.broadcast_to(self.pos(), v.shape)
        return (o, v)

    def window_to_world3(self, w):
        """Unproject 3D window coordinates (uv + depth) to world points"""
        return self.eye_to_world(self.window_to_eye3(w))

    def world_visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def world_to_eye(self, v):
        """
        Apply camera inverse extrinsics to points `v` to get eye coords
        """
        return transform3d.rotate_points(
            self.T_world_from_eye.T, v - self.T_world_from_eye[:3, 3]
        )

    def eye_to_world(self, v):
        """
        Apply camera extrinsics to eye points `v` to get world coords
        """
        return transform3d.transform_points(self.T_world_from_eye, v)

    def eye_to_window(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        return p * self.f + self.c

    def window_to_eye(self, w):
        """Unproject 2d window coordinates to unit-length 3D eye coordinates"""

        q = (np.asarray(w) - self.c) / self.f
        return self.unproject(q)

    def eye_to_window3(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)
        q = self.distort.evaluate(p[..., :2])
        p[..., :2] = q * self.f + self.c
        return p

    def window_to_eye3(self, w):
        """Unproject 3d window coordinates (uv + depth) to eye coordinates"""
        # assert self.undistort is not None
        temp = np.array(w, dtype=np.float64)
        temp[..., :2] -= self.c
        temp[..., :2] /= self.f
        # temp[..., :2] = self.undistort.evaluate(temp[..., :2])
        return self.unproject3(temp)

    def visible(self, v):
        """
        Returns true if the given world-space points are visible in this camera
        """
        return self.eye_visible(self.world_to_eye(v))

    def eye_visible(self, v):
        """
        Returns true if the given eye points are visible in this camera
        """
        v = transform3d.normalized(v)
        w = self.eye_to_window(v)
        return (v[..., 2] >= self.zmin()) & self.w_visible(w)

    def w_visible(self, w, *, margin=0):
        """
        Return True if the 2d window coordinate `w` is inside the window

        Can be called with an array, returning a bool array.
        """
        x, y = np.moveaxis(w, -1, 0)
        x0 = -margin - 0.5
        y0 = -margin - 0.5
        x1 = self.width + margin - 0.5
        y1 = self.height + margin - 0.5
        return (x > x0) & (x < x1) & (y >= y0) & (y < y1)

    def crop(
        self,
        src_x,
        src_y,
        target_width,
        target_height,
        scale=1,
        T_world_from_eye=None,
        serial=None,
    ):
        """
        Return intrinsics for a crop of the sensor image.

        No scaling is applied; this just returns the model for a sub-
        array of image data. (Or for a larger array, if (x,y)<=0 and
        (width, height) > (self.width, self.height).

        To do both cropping and scaling, use :meth:`subrect`

        Parameters
        ----------
        x, y, width, height
            Location and size in this camera's window coordinates
        """
        return type(self)(
            target_width,
            target_height,
            np.asarray(self.f) * scale,
            (np.array(self.c) - (src_x, src_y) + 0.5) * scale - 0.5,
            self.T_world_from_eye if T_world_from_eye is None else T_world_from_eye,
            self.serial if serial is None else serial,
        )

    def subrect(self, transform, width, height, bypass_fit_undistort_coeffs=False):
        """
        Return intrinsics for a scaled crop of the sensor image.

        Parameters
        ----------
        Transform
            a 2x3 affine transform matrix that takes coordinates in the
            old image rect to coordinates in the new image rect, as for
            `cv.WarpAffine`.

            The transform is given in continuous coords, so it must
            follow the "pixel center on integer grid coordinates"
            convention. E.g. resizing an image by 1/N is not just
            scaling by 1/N, but scaling by `1/N` and translating by
            `(1-N)/(2N)`

            Yes, this is confusing. Blame the CV community for failing
            to learn anything from the graphics community.

        width, height : int
            size of target image
        """
        # Currently only support scale and translation.
        #
        # We could support 90 degree rotation by careful manipulation of polynomial
        # coefficients, or arbitrary rotation by applying an 2D affine transform
        # instead of just (f, c) to convert distorted coords to window coords.
        f = np.diag(transform[:-2, :-2])
        c = transform[:-2, 2]
        offdiag = np.diag(np.flip(transform[..., :2, :2], -1))
        if not np.all(offdiag == 0.0):
            raise NotImplementedError("transforms with rotation not yet supported")
        cam = type(self)(
            width,
            height,
            self.f * f,
            self.c * f + c,
            self.distort_coeffs,
            self.undistort_coeffs,
            self.T_world_from_eye,
            self.serial,
            bypass_fit_undistort_coeffs,
        )
        return cam


class PinholePlaneCameraModel(CameraModel):
    model_fov_limit = 50 * (math.pi / 180)

    def uv_to_window_matrix(self):
        """Return the 3x3 intrinsics matrix"""
        return np.array(
            [[self.f[0], 0, self.c[0]], [0, self.f[1], self.c[1]], [0, 0, 1]]
        )


class Collection:
    """Represents a set of samples (e.g. a set of images or objects).

    It stores attributes of samples as "fields". For example, in the case
    of objects, it can store boxes, masks, labels, scores, etc.). All fields
    must have the same ``__len__`` which is the number of samples and can be
    tensors, numpy arrays, lists, etc.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          collection.boxes = tensor.rand(num_objs, 4)
          collection.scores = tensor.rand(num_objs)
          collection.names = ["mouse", "keyboard", ...]  # num_objs names
          print(collection.scores)
          print('boxes' in collection)

    2. ``len(collection)`` returns the number of samples
    3. Indexing: ``collection[indices]`` will apply the indexing on all the
       fields and returns a new :class:`Collection`.
       Typically, ``indices`` is an integer vector of indices,
       or a binary mask of length ``num_samples``

       .. code-block:: python

          category_3_detections = collection[collection.pred_classes == 3]
          confident_detections = collection[collection.scores > 0.9]

    Bootstrapped from: detectron2/structures/instances.py
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            kwargs: fields to add to this `Collection`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:  # pyre-ignore
        if name == "_fields" or name not in self._fields:
            raise AttributeError(
                "Cannot find field '{}' in the given Collection!".format(name)
            )
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:  # pyre-ignore
        """Sets the field named `name` to `value`.

        The length of `value` must be the number of samples,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, (
                "Adding a field {} of length {} to a Collection of length {}".format(
                    name, data_len, len(self)
                )
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """Checks whether the specified field exists."""
        return name in self._fields

    def remove(self, name: str) -> None:
        """Removes the field called `name`."""
        del self._fields[name]

    def get(self, name: str) -> Any:  # pyre-ignore
        """Returns the field called `name`."""
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """Returns a dict mapping names to data of the fields."""
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Collection":
        """Calls `to(device)` on all fields that support this method."""
        ret = Collection()
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(
        self,
        # pyre-fixme[24]: Generic type `slice` expects 3 type parameters.
        item: Union[int, slice, torch.Tensor, torch.BoolTensor],
    ) -> "Collection":
        """
        Args:
            item: An index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Collection` where all fields are indexed by `item`.
        """
        if type(item) is int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Collection index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Collection()
        for k, v in self._fields.items():
            if isinstance(v, List):
                ret.set(k, [v[i] for i in item])
            else:
                ret.set(k, v[item])

        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        return 0

    def __iter__(self) -> None:
        raise NotImplementedError("`Collection` object is not iterable!")

    def tensors_to_arrays(self) -> None:
        for field in self._fields.keys():
            if torch.is_tensor(self._fields[field]):
                self._fields[field] = torch_helpers.tensor_to_array(self._fields[field])

    def arrays_to_tensors(self) -> None:
        for field in self._fields.keys():
            if isinstance(self._fields[field], np.ndarray):
                self._fields[field] = torch_helpers.array_to_tensor(self._fields[field])

    @staticmethod
    def cat(collection_list: List["Collection"]) -> "Collection":
        """
        Args:
            collection_list (list[Collection])

        Returns:
            Collection
        """
        assert all(isinstance(i, Collection) for i in collection_list)
        assert len(collection_list) > 0
        if len(collection_list) == 1:
            return collection_list[0]

        ret = Collection()
        for k in collection_list[0]._fields.keys():
            values = [i.get(k) for i in collection_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError(
                    "Unsupported type {} for concatenation".format(type(v0))
                )
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_samples={}, ".format(len(self))
        s += "fields=[{}])".format(
            ", ".join((f"{k}: {v}" for k, v in self._fields.items()))
        )
        return s

    __repr__ = __str__  # pyre-ignore


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)
