from typing import TypeVar, Union

import numpy as np
import numpy.typing as npt

from ._dispatch import (
    _dtype_to_func_box_areas,
    _dtype_to_func_box_convert,
    _dtype_to_func_diou_distance,
    _dtype_to_func_giou_distance,
    _dtype_to_func_iou_distance,
    _dtype_to_func_nms,
    _dtype_to_func_parallel_giou_distance,
    _dtype_to_func_parallel_iou_distance,
    _dtype_to_func_remove_small_boxes,
    _dtype_to_func_rotated_nms,
    _dtype_to_func_rtree_nms,
    _dtype_to_func_rtree_rotated_nms,
    _dtype_to_func_tiou_distance,
)
from ._powerboxes import draw_boxes as _draw_boxes
from ._powerboxes import masks_to_boxes as _masks_to_boxes
from ._powerboxes import (
    parallel_rotated_giou_distance as _parallel_rotated_giou_distance,
)
from ._powerboxes import parallel_rotated_iou_distance as _parallel_rotated_iou_distance
from ._powerboxes import (
    parallel_rotated_tiou_distance as _parallel_rotated_tiou_distance,
)
from ._powerboxes import rotated_giou_distance as _rotated_giou_distance
from ._powerboxes import rotated_iou_distance as _rotated_iou_distance
from ._powerboxes import rotated_tiou_distance as _rotated_tiou_distance

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
__version__ = "0.3.0"

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


# ---------------------------------------------------------------------------
# Internal dispatch helpers
# ---------------------------------------------------------------------------


def _dispatch2(dispatch_map, boxes1, boxes2, *args):
    """Dispatch a (boxes1, boxes2[, *args]) call via dtype."""
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype != boxes2.dtype:
        raise ValueError(_BOXES_NOT_SAME_TYPE)
    try:
        return dispatch_map[boxes1.dtype](boxes1, boxes2, *args)
    except KeyError:
        raise TypeError(
            f"Box dtype: {boxes1.dtype} not in supported dtypes {supported_dtypes}"
        )


def _dispatch1(dispatch_map, boxes, *args):
    """Dispatch a (boxes[, *args]) call via dtype."""
    if not isinstance(boxes, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    try:
        return dispatch_map[boxes.dtype](boxes, *args)
    except KeyError:
        raise TypeError(
            f"Box dtype: {boxes.dtype} not in supported dtypes {supported_dtypes}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diou_distance(
    boxes1: npt.NDArray[Union[np.float32, np.float64]],
    boxes2: npt.NDArray[Union[np.float32, np.float64]],
) -> npt.NDArray[np.float64]:
    """Compute pairwise box diou distances.

    DIoU distance is defined in https://arxiv.org/pdf/1911.08287.pdf

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_diou_distance, boxes1, boxes2)


def iou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box iou distances.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_iou_distance, boxes1, boxes2)


def parallel_iou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box iou distances, in parallel.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_parallel_iou_distance, boxes1, boxes2)


def giou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box giou distances.

    see [here](https://giou.stanford.edu/) for giou distance definition

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_giou_distance, boxes1, boxes2)


def parallel_giou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box giou distances, in parallel.

    see [here](https://giou.stanford.edu/) for giou distance definition

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_parallel_giou_distance, boxes1, boxes2)


def tiou_distance(
    boxes1: npt.NDArray[T], boxes2: npt.NDArray[T]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box tiou (tracking iou) distances.

    see [here](https://arxiv.org/pdf/2310.05171.pdf) for tiou definition

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    return _dispatch2(_dtype_to_func_tiou_distance, boxes1, boxes2)


def rotated_iou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the pairwise iou distance between rotated boxes.

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _rotated_iou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def parallel_rotated_iou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the pairwise iou distance between rotated boxes, in parallel.

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _parallel_rotated_iou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def rotated_giou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the pairwise giou distance between rotated boxes.

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _rotated_giou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def parallel_rotated_giou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the pairwise giou distance between rotated boxes, in parallel.

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _parallel_rotated_giou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def rotated_tiou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box tiou (tracking iou) distances between rotated boxes.

    see [here](https://arxiv.org/pdf/2310.05171.pdf) for tiou definition

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _rotated_tiou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def parallel_rotated_tiou_distance(
    boxes1: npt.NDArray[np.float64], boxes2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute pairwise box tiou (tracking iou) distances between rotated boxes, in parallel.

    see [here](https://arxiv.org/pdf/2310.05171.pdf) for tiou definition

    Boxes should be in (cx, cy, w, h, a) format
    where cx and cy are center coordinates, w and h
    width and height and a, the angle in degrees

    Args:
        boxes1: 2d array of boxes in cxywha format
        boxes2: 2d array of boxes in cxywha format

    Raises:
        TypeError: if boxes1 or boxes2 are not numpy arrays
        ValueError: if boxes1 and boxes2 have different dtypes

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    if not isinstance(boxes1, np.ndarray) or not isinstance(boxes2, np.ndarray):
        raise TypeError(_BOXES_NOT_NP_ARRAY)
    if boxes1.dtype == boxes2.dtype == np.dtype("float64"):
        return _parallel_rotated_tiou_distance(boxes1, boxes2)
    raise TypeError(f"Boxes dtype: {boxes1.dtype}, {boxes2.dtype} not in float64 dtype")


def remove_small_boxes(boxes: npt.NDArray[T], min_size: float) -> npt.NDArray[T]:
    """Remove boxes with area less than min_area.

    Args:
        boxes: 2d array of boxes in xyxy format
        min_size: minimum area of boxes to keep

    Raises:
        TypeError: if boxes is not numpy array

    Returns:
        np.ndarray: 2d array of boxes in xyxy format
    """
    return _dispatch1(_dtype_to_func_remove_small_boxes, boxes, min_size)


def boxes_areas(boxes: npt.NDArray[T]) -> npt.NDArray[np.float64]:
    """Compute areas of boxes.

    Args:
        boxes: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 1d array of areas
    """
    return _dispatch1(_dtype_to_func_box_areas, boxes)


def box_convert(boxes: npt.NDArray[T], in_fmt: str, out_fmt: str) -> npt.NDArray[T]:
    """Convert boxes from one format to another.

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
    return _dispatch1(_dtype_to_func_box_convert, boxes, in_fmt, out_fmt)


def masks_to_boxes(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint64]:
    """Convert masks to boxes in xyxy format.

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
    """Apply non-maximum suppression to boxes.

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
    return _dispatch1(_dtype_to_func_nms, boxes, scores, iou_threshold, score_threshold)


def rotated_nms(
    boxes: npt.NDArray[T],
    scores: npt.NDArray[np.float64],
    iou_threshold: float,
    score_threshold: float,
) -> npt.NDArray[np.uint64]:
    """Apply non-maximum suppression to oriented bounding boxes.

    Args:
        boxes: 2d array of boxes in cxcywha format
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
    return _dispatch1(
        _dtype_to_func_rotated_nms, boxes, scores, iou_threshold, score_threshold
    )


def rtree_nms(
    boxes: npt.NDArray[Union[np.float64, np.float32, np.int64, np.int32, np.int16]],
    scores: npt.NDArray[np.float64],
    iou_threshold: float,
    score_threshold: float,
) -> npt.NDArray[np.uint64]:
    """Apply non-maximum suppression to boxes using an R-tree index.

    Uses an rtree to speed up computation. Only available for
    signed integer dtypes and float32/float64.

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
    return _dispatch1(
        _dtype_to_func_rtree_nms, boxes, scores, iou_threshold, score_threshold
    )


def rtree_rotated_nms(
    boxes: npt.NDArray[Union[np.float64, np.float32, np.int64, np.int32, np.int16]],
    scores: npt.NDArray[np.float64],
    iou_threshold: float,
    score_threshold: float,
) -> npt.NDArray[np.uint64]:
    """Apply non-maximum suppression to oriented boxes using an R-tree index.

    Uses an rtree to speed up computation. Only available for
    signed integer dtypes and float32/float64.

    Args:
        boxes: 2d array of boxes in cxcywha format
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
    return _dispatch1(
        _dtype_to_func_rtree_rotated_nms, boxes, scores, iou_threshold, score_threshold
    )


def draw_boxes(
    image: npt.NDArray[np.uint8],
    boxes: npt.NDArray[np.float64],
    colors: npt.NDArray[np.uint8] = None,
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw bounding boxes on an image.

    Args:
        image: 3d array of shape (3, H, W) in CHW format, uint8
        boxes: 2d array of boxes in xyxy format, float64
        colors: optional 2d array of shape (N, 3) with RGB colors per box, uint8
        thickness: line thickness in pixels (default 2)

    Raises:
        TypeError: if image or boxes are not numpy arrays

    Returns:
        np.ndarray: 3d array of shape (3, H, W) with boxes drawn
    """
    if not isinstance(image, np.ndarray) or not isinstance(boxes, np.ndarray):
        raise TypeError("image and boxes must be numpy arrays")
    return _draw_boxes(image, boxes, colors, thickness)


__all__ = [
    "diou_distance",
    "iou_distance",
    "parallel_iou_distance",
    "giou_distance",
    "parallel_giou_distance",
    "tiou_distance",
    "rotated_iou_distance",
    "parallel_rotated_iou_distance",
    "rotated_giou_distance",
    "parallel_rotated_giou_distance",
    "rotated_tiou_distance",
    "parallel_rotated_tiou_distance",
    "remove_small_boxes",
    "boxes_areas",
    "box_convert",
    "masks_to_boxes",
    "nms",
    "rotated_nms",
    "rtree_nms",
    "draw_boxes",
    "supported_dtypes",
    "__version__",
]
