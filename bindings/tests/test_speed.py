import numpy as np
import pytest
from powerboxes import (
    box_convert,
    boxes_areas,
    giou_distance,
    iou_distance,
    parallel_giou_distance,
    parallel_iou_distance,
    remove_small_boxes,
)


@pytest.mark.benchmark(group="giou_distance")
def test_giou_distance(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(giou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="parallel_giou_distance")
def test_parallel_giou_distance(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(parallel_giou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="iou_distance")
def test_iou_distance(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(iou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="parallel_iou_distance")
def test_parallel_iou_distance(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(parallel_iou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="remove_small_boxes")
def test_remove_small_boxes(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(remove_small_boxes, boxes, 0.4)


@pytest.mark.benchmark(group="remove_small_boxes")
def test_boxes_areas(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(boxes_areas, boxes)


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_xyxy_xywh(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "xyxy", "xywh")


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_xyxy_cxcywh(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "xyxy", "cxcywh")


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_cxcywh_xywh(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "cxcywh", "xywh")


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_cxcywh_xyxy(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "cxcywh", "xywh")


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_xywh_cxcywh(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "xywh", "cxcywh")


@pytest.mark.benchmark(group="box_convert")
def test_box_convert_xywh_xyxy(benchmark):
    boxes = np.random.random((100, 4))
    benchmark(box_convert, boxes, "xywh", "xyxy")
