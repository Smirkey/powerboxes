import numpy as np
from ._powerboxes import box_convert as _box_convert
from ._powerboxes import boxes_areas as _boxes_areas
from ._powerboxes import distance_box_iou as _distance_box_iou
from ._powerboxes import parallel_distance_box_iou as _parallel_distance_box_iou
from ._powerboxes import remove_small_boxes as _remove_small_boxes


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str):
    return _box_convert(boxes, in_fmt, out_fmt)


def boxes_areas(boxes: np.ndarray):
    return _boxes_areas(boxes)


def distance_box_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    return _distance_box_iou(boxes1, boxes2)


def parallel_distance_box_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    return _parallel_distance_box_iou(boxes1, boxes2)


def remove_small_boxes(boxes: np.ndarray, min_area: float):
    return _remove_small_boxes(boxes, min_area)


__all__ = [
    "distance_box_iou",
    "parallel_distance_box_iou",
    "remove_small_boxes",
    "boxes_areas",
    "box_convert",
]
