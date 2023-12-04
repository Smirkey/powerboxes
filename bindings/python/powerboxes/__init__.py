import numpy as np
import numpy.typing as npt
from ._boxes import (
    _dtype_to_func_box_areas,
    _dtype_to_func_box_convert,
    _dtype_to_func_remove_small_boxes,
)
from ._giou import (
    _dtype_to_func_giou_distance,
    _dtype_to_func_parallel_giou_distance,
)
from ._iou import (
    _dtype_to_func_iou_distance,
    _dtype_to_func_parallel_iou_distance,
)
from ._powerboxes import masks_to_boxes as _masks_to_boxes

from typing import TypeVar, Union
BOXES_NOT_SAME_TYPE = "boxes1 and boxes2 must have the same dtype"
BOXES_NOT_NP_ARRAY = "boxes must be numpy array"
supported_dtypes = [
    "float64",
    "float32",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]
__version__ = "0.1.3"

T = TypeVar("T", bound=Union[np.float64, np.float32, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])

def iou_distance(boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes pairwise box iou distances.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_iou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(BOXES_NOT_SAME_TYPE)


def parallel_iou_distance(boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes pairwise box iou distances, in parallel.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_parallel_iou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(BOXES_NOT_SAME_TYPE)


def parallel_giou_distance(boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes pairwise box giou distances, in parallel.

    see: https://giou.stanford.edu/

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_parallel_giou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(BOXES_NOT_SAME_TYPE)


def giou_distance(boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes pairwise box giou distances.

    see: https://giou.stanford.edu/

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_giou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(BOXES_NOT_SAME_TYPE)


def remove_small_boxes(boxes: npt.NDArray[T], min_size) -> npt.NDArray[T]:
    """Removes boxes with area less than min_area.

    Args:
        boxes: 2d array of boxes in xyxy format
        min_size: minimum area of boxes to keep

    Raises:
        TypeError: if boxes is not numpy array

    Returns:
        np.ndarray: 2d array of boxes in xyxy format
    """
    if not isinstance(boxes, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    return _dtype_to_func_remove_small_boxes[boxes.dtype](boxes, min_size)


def boxes_areas(boxes: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes areas of boxes.

    Args:
        boxes: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 1d array of areas
    """
    if not isinstance(boxes, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    return _dtype_to_func_box_areas[boxes.dtype](boxes)


def box_convert(boxes: npt.NDArray[T], in_fmt: str, out_fmt: str) -> npt.NDArray[T]:
    """Converts boxes from one format to another.

    Available formats are:
        - 'xyxy': (xmin, ymin, xmax, ymax)
        - 'xywh': (xmin, ymin, width, height)
        - 'cxcywh': (center_x, center_y, width, height)

    Args:
        boxes: 2d array of boxes in in_fmt
        in_fmt: format of input boxes
        out_fmt: format of output boxes

    Returns:
        np.ndarray: boxes in out_fmt
    """
    if not isinstance(boxes, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    return _dtype_to_func_box_convert[boxes.dtype](boxes, in_fmt, out_fmt)

def masks_to_boxes(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if not isinstance(masks, np.ndarray):
        raise TypeError(BOXES_NOT_NP_ARRAY)
    return _masks_to_boxes(masks)

__all__ = [
    "iou_distance",
    "parallel_iou_distance",
    "remove_small_boxes",
    "boxes_areas",
    "box_convert",
    "giou_distance",
    "parallel_giou_distance",
    "masks_to_boxes",
    "supported_dtypes"
]
