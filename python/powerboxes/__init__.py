import numpy as np

from .powerboxesrs import (
    box_convert,
    boxes_areas,
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
        raise TypeError("boxes1 and boxes2 must be numpy arrays")
    dtype_to_func = {
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
    if boxes1.dtype == boxes2.dtype:
        return dtype_to_func[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError("boxes1 and boxes2 must have the same dtype")


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
        raise TypeError("boxes1 and boxes2 must be numpy arrays")
    dtype_to_func = {
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
    if boxes1.dtype == boxes2.dtype:
        return dtype_to_func[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError("boxes1 and boxes2 must have the same dtype")


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
        raise TypeError("boxes1 and boxes2 must be numpy arrays")
    dtype_to_func = {
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
    if boxes1.dtype == boxes2.dtype:
        return dtype_to_func[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError("boxes1 and boxes2 must have the same dtype")


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
        raise TypeError("boxes1 and boxes2 must be numpy arrays")
    dtype_to_func = {
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
    if boxes1.dtype == boxes2.dtype:
        return dtype_to_func[boxes1.dtype](boxes1, boxes2)
    else:
        raise ValueError("boxes1 and boxes2 must have the same dtype")


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
    dtype_to_func = {
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
    return dtype_to_func[boxes.dtype](boxes, min_size)


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
