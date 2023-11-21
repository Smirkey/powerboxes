import numpy as np

def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
    """Converts boxes from one format to another.

    Available formats are:
        - xyxy: (xmin, ymin, xmax, ymax)
        - xywh: (xmin, ymin, width, height)
        - cxcywh: (center_x, center_y, width, height)

    Args:
        boxes: 2d array of boxes in in_fmt
        in_fmt: format of input boxes
        out_fmt: format of output boxes

    Returns:
        np.ndarray: boxes in out_fmt
    """

def boxes_areas(boxes: np.ndarray) -> np.ndarray:
    """Computes areas of boxes.

    Args:
        boxes: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 1d array of areas
    """

def parallel_iou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise box iou distances, in parallel.

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """

def giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise box giou distances.

    see: https://giou.stanford.edu/

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """

def parallel_giou_distance(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise box giou distances, in parallel.

    see: https://giou.stanford.edu/

    Args:
        boxes1: 2d array of boxes in xyxy format
        boxes2: 2d array of boxes in xyxy format

    Returns:
        np.ndarray: 2d matrix of pairwise distances
    """

def remove_small_boxes(boxes: np.ndarray, min_size) -> np.ndarray:
    """Removes boxes with area less than min_area.

    Args:
        boxes: 2d array of boxes in xyxy format
        min_size: minimum area of boxes to keep

    Returns:
        np.ndarray: 2d array of boxes in xyxy format
    """
