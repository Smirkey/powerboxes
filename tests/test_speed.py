import numpy as np
import pytest
from powerboxes import (
    box_convert,
    boxes_areas,
    distance_box_iou,
    parallel_distance_box_iou,
    remove_small_boxes,
)


@pytest.mark.benchmark(group="distance_box_iou")
def test_distance_box_iou(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(distance_box_iou, boxes1, boxes2)


@pytest.mark.benchmark(group="parallel_distance_box_iou")
def test_parallel_distance_box_iou(benchmark):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    benchmark(parallel_distance_box_iou, boxes1, boxes2)


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
