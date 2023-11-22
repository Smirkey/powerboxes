import numpy as np

from .powerboxesrs import (
    box_areas_f32,
    box_areas_f64,
    box_areas_i16,
    box_areas_i32,
    box_areas_i64,
    box_areas_u16,
    box_areas_u32,
    box_areas_u64,
    box_convert_f32,
    box_convert_f64,
    box_convert_i16,
    box_convert_i32,
    box_convert_i64,
    box_convert_u8,
    box_convert_u16,
    box_convert_u32,
    box_convert_u64,
    giou_distance_f32,
    giou_distance_f64,
    giou_distance_i16,
    giou_distance_i32,
    giou_distance_i64,
    giou_distance_u8,
    giou_distance_u16,
    giou_distance_u32,
    giou_distance_u64,
    iou_distance_f32,
    iou_distance_f64,
    iou_distance_i16,
    iou_distance_i32,
    iou_distance_i64,
    iou_distance_u8,
    iou_distance_u16,
    iou_distance_u32,
    iou_distance_u64,
    parallel_giou_distance_f32,
    parallel_giou_distance_f64,
    parallel_giou_distance_i16,
    parallel_giou_distance_i32,
    parallel_giou_distance_i64,
    parallel_giou_distance_u8,
    parallel_giou_distance_u16,
    parallel_giou_distance_u32,
    parallel_giou_distance_u64,
    parallel_iou_distance_f32,
    parallel_iou_distance_f64,
    parallel_iou_distance_i16,
    parallel_iou_distance_i32,
    parallel_iou_distance_i64,
    parallel_iou_distance_u8,
    parallel_iou_distance_u16,
    parallel_iou_distance_u32,
    parallel_iou_distance_u64,
    remove_small_boxes_f32,
    remove_small_boxes_f64,
    remove_small_boxes_i16,
    remove_small_boxes_i32,
    remove_small_boxes_i64,
    remove_small_boxes_u8,
    remove_small_boxes_u16,
    remove_small_boxes_u32,
    remove_small_boxes_u64,
)

_dtype_to_func_box_convert = {
    np.dtype("float64"): box_convert_f64,
    np.dtype("float32"): box_convert_f32,
    np.dtype("int64"): box_convert_i64,
    np.dtype("int32"): box_convert_i32,
    np.dtype("int16"): box_convert_i16,
    np.dtype("uint64"): box_convert_u64,
    np.dtype("uint32"): box_convert_u32,
    np.dtype("uint16"): box_convert_u16,
    np.dtype("uint8"): box_convert_u8,
}
_dtype_to_func_box_areas = {
    np.dtype("float64"): box_areas_f64,
    np.dtype("float32"): box_areas_f32,
    np.dtype("int64"): box_areas_i64,
    np.dtype("int32"): box_areas_i32,
    np.dtype("int16"): box_areas_i16,
    np.dtype("uint64"): box_areas_u64,
    np.dtype("uint32"): box_areas_u32,
    np.dtype("uint16"): box_areas_u16,
}
_dtype_to_func_remove_small_boxes = {
    np.dtype("float64"): remove_small_boxes_f64,
    np.dtype("float32"): remove_small_boxes_f32,
    np.dtype("int64"): remove_small_boxes_i64,
    np.dtype("int32"): remove_small_boxes_i32,
    np.dtype("int16"): remove_small_boxes_i16,
    np.dtype("uint64"): remove_small_boxes_u64,
    np.dtype("uint32"): remove_small_boxes_u32,
    np.dtype("uint16"): remove_small_boxes_u16,
    np.dtype("uint8"): remove_small_boxes_u8,
}
_dtype_to_func_giou_distance = {
    np.dtype("float64"): giou_distance_f64,
    np.dtype("float32"): giou_distance_f32,
    np.dtype("int64"): giou_distance_i64,
    np.dtype("int32"): giou_distance_i32,
    np.dtype("int16"): giou_distance_i16,
    np.dtype("uint64"): giou_distance_u64,
    np.dtype("uint32"): giou_distance_u32,
    np.dtype("uint16"): giou_distance_u16,
    np.dtype("uint8"): giou_distance_u8,
}
_dtype_to_func_parallel_giou_distance = {
    np.dtype("float64"): parallel_giou_distance_f64,
    np.dtype("float32"): parallel_giou_distance_f32,
    np.dtype("int64"): parallel_giou_distance_i64,
    np.dtype("int32"): parallel_giou_distance_i32,
    np.dtype("int16"): parallel_giou_distance_i16,
    np.dtype("uint64"): parallel_giou_distance_u64,
    np.dtype("uint32"): parallel_giou_distance_u32,
    np.dtype("uint16"): parallel_giou_distance_u16,
    np.dtype("uint8"): parallel_giou_distance_u8,
}
_dtype_to_func_parallel_iou_distance = {
    np.dtype("float64"): parallel_iou_distance_f64,
    np.dtype("float32"): parallel_iou_distance_f32,
    np.dtype("int64"): parallel_iou_distance_i64,
    np.dtype("int32"): parallel_iou_distance_i32,
    np.dtype("int16"): parallel_iou_distance_i16,
    np.dtype("uint64"): parallel_iou_distance_u64,
    np.dtype("uint32"): parallel_iou_distance_u32,
    np.dtype("uint16"): parallel_iou_distance_u16,
    np.dtype("uint8"): parallel_iou_distance_u8,
}
_dtype_to_func_iou_distance = {
    np.dtype("float64"): iou_distance_f64,
    np.dtype("float32"): iou_distance_f32,
    np.dtype("int64"): iou_distance_i64,
    np.dtype("int32"): iou_distance_i32,
    np.dtype("int16"): iou_distance_i16,
    np.dtype("uint64"): iou_distance_u64,
    np.dtype("uint32"): iou_distance_u32,
    np.dtype("uint16"): iou_distance_u16,
    np.dtype("uint8"): iou_distance_u8,
}
BOXES_NOT_SAME_TYPE = "boxes1 and boxes2 must have the same dtype"
BOXES_NOT_NP_ARRAY = "boxes1 and boxes2 must be numpy arrays"
__version__ = "0.1.2"


def iou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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


def parallel_iou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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


def parallel_giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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


def giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
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


def remove_small_boxes(boxes: np.ndarray, min_size) -> np.ndarray:
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
        raise TypeError("boxes must be numpy array")
    return _dtype_to_func_remove_small_boxes[boxes.dtype](boxes, min_size)


def boxes_areas(boxes: np.ndarray) -> np.ndarray:
    """Computes areas of boxes.

    Args:
        boxes: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 1d array of areas
    """
    if not isinstance(boxes, np.ndarray):
        raise TypeError("boxes must be numpy array")
    return _dtype_to_func_box_areas[boxes.dtype](boxes)


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
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
        raise TypeError("boxes must be numpy array")
    return _dtype_to_func_box_convert[boxes.dtype](boxes, in_fmt, out_fmt)


__all__ = [
    "iou_distance",
    "parallel_iou_distance",
    "remove_small_boxes",
    "boxes_areas",
    "box_convert",
    "giou_distance",
    "parallel_giou_distance",
    "iou_distance_f64",
]
