from typing import TypeVar, Union

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
from ._nms import _dtype_to_func_nms, _dtype_to_func_rtree_nms
from ._powerboxes import masks_to_boxes as _masks_to_boxes

_BOXES_NOT_SAME_TYPE = "boxes1 and boxes2 must have the same dtype"
_BOXES_NOT_NP_ARRAY = "boxes must be numpy array"
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

T = TypeVar(
    "T",
    bound=Union[
        np.float64,
        np.float32,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ],
)


def iou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_iou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(_BOXES_NOT_SAME_TYPE)


def parallel_iou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_parallel_iou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(_BOXES_NOT_SAME_TYPE)


def parallel_giou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_parallel_giou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(_BOXES_NOT_SAME_TYPE)


def giou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype:
        return _dtype_to_func_giou_distance[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError(_BOXES_NOT_SAME_TYPE)


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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    return _dtype_to_func_remove_small_boxes[boxes.dtype](boxes, min_size)


def boxes_areas(boxes: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Computes areas of boxes.

    Args:
        boxes: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 1d array of areas
    """
    if not isinstance(boxes, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
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
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    return _dtype_to_func_box_convert[boxes.dtype](boxes, in_fmt, out_fmt)


def masks_to_boxes(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint64]:
    """Converts masks to boxes in xyxy format.

    Args:
        masks: 3d array of masks in (N, H, W) format

    Raises:
        TypeError: if masks is not numpy array

    Returns:
        npt.NDArray[np.uint64]: 2d array of boxes in xyxy format
    """
    if not isinstance(masks, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    return _masks_to_boxes(masks)


def nms(
    boxes: npt.NDArray[T],
    scores: npt.NDArray[np.float64],
    iou_threshold: float,
    score_threshold: float,
) -> npt.NDArray[np.uint64]:
    """Applies non-maximum suppression to boxes.

    Args:
        boxes: 2d array of boxes in xyxy format
        scores: 1d array of scores
        iou_threshold: threshold for iou
        score_threshold: threshold for scores

    Raises:
        TypeError: if boxes or scores are not numpy arrays

    Returns:
        npt.NDArray[np.uint64]: 1d array of indices to keep
    """
    if not isinstance(boxes, np.ndarray) or not isinstance(scores, np.ndarray):
        raise TypeError("Boxes and scores must be numpy arrays")
    return _dtype_to_func_nms[boxes.dtype](
        boxes, scores, iou_threshold, score_threshold
    )


def rtree_nms(
    boxes: npt.NDArray[T],
    scores: npt.NDArray[np.float64],
    iou_threshold: float,
    score_threshold: float,
) -> npt.NDArray[np.uint64]:
    """Applies non-maximum suppression to boxes.

    Uses an rtree to speed up computation. This is only available for
    signed integer dtypes and float32 and float64.

    Args:
        boxes: 2d array of boxes in xyxy format
        scores: 1d array of scores
        iou_threshold: threshold for iou
        score_threshold: threshold for scores

    Raises:
        TypeError: if boxes or scores are not numpy arrays

    Returns:
        npt.NDArray[np.uint64]: 1d array of indices to keep
    """
    if not isinstance(boxes, np.ndarray) or not isinstance(scores, np.ndarray):
        raise TypeError("Boxes and scores must be numpy arrays")
    return _dtype_to_func_rtree_nms[boxes.dtype](
        boxes, scores, iou_threshold, score_threshold
    )


__all__ = [
    "iou_distance",
    "parallel_iou_distance",
    "remove_small_boxes",
    "boxes_areas",
    "box_convert",
    "giou_distance",
    "parallel_giou_distance",
    "masks_to_boxes",
    "supported_dtypes",
    "nms",
    "rtree_nms",
    "__version__",
]
