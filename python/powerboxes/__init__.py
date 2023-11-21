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
    parallel_giou_distance,
    parallel_iou_distance,
    remove_small_boxes,
)


def iou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise box iou distances.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    match (boxes1.dtype, boxes2.dtype):
        case (np.float64, np.float64):
            return iou_distance_f64(boxes1, boxes2)
        case (np.float32, np.float32):
            return iou_distance_f32(boxes1, boxes2)
        case (np.int64, np.int64):
            return iou_distance_i64(boxes1, boxes2)
        case (np.int32, np.int32):
            return iou_distance_i32(boxes1, boxes2)
        case (np.int16, np.int16):
            return iou_distance_i16(boxes1, boxes2)
        case (np.uint64, np.uint64):
            return iou_distance_u64(boxes1, boxes2)
        case (np.uint32, np.uint32):
            return iou_distance_u32(boxes1, boxes2)
        case (np.uint16, np.uint16):
            return iou_distance_u16(boxes1, boxes2)
        case (np.uint8, np.uint8):
            return iou_distance_u8(boxes1, boxes2)
        case _:
            raise ValueError("Unsupported dtype")


def giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise box giou distances.

    see: https://giou.stanford.edu/

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """
    match (boxes1.dtype, boxes2.dtype):
        case (np.float64, np.float64):
            return giou_distance_f64(boxes1, boxes2)
        case (np.float32, np.float32):
            return giou_distance_f32(boxes1, boxes2)
        case (np.int64, np.int64):
            return giou_distance_i64(boxes1, boxes2)
        case (np.int32, np.int32):
            return giou_distance_i32(boxes1, boxes2)
        case (np.int16, np.int16):
            return giou_distance_i16(boxes1, boxes2)
        case (np.uint64, np.uint64):
            return giou_distance_u64(boxes1, boxes2)
        case (np.uint32, np.uint32):
            return giou_distance_u32(boxes1, boxes2)
        case (np.uint16, np.uint16):
            return giou_distance_u16(boxes1, boxes2)
        case (np.uint8, np.uint8):
            return giou_distance_u8(boxes1, boxes2)
        case _:
            raise ValueError("Unsupported dtype")


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
