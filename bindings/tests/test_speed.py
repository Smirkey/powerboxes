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
    masks_to_boxes,
    supported_dtypes
)

np.random.seed(42)

@pytest.mark.benchmark(group="giou_distance")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_giou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(giou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="parallel_giou_distance")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_parallel_giou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(parallel_giou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="iou_distance")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_iou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(iou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="parallel_iou_distance")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_parallel_iou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(parallel_iou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="remove_small_boxes")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_remove_small_boxes(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(remove_small_boxes, boxes, 0.4)


@pytest.mark.benchmark(group="remove_small_boxes")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_boxes_areas(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(boxes_areas, boxes)


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_xyxy_xywh(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "xyxy", "xywh")


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_xyxy_cxcywh(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "xyxy", "cxcywh")


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_cxcywh_xywh(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "cxcywh", "xywh")


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_cxcywh_xyxy(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "cxcywh", "xywh")


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_xywh_cxcywh(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "xywh", "cxcywh")


@pytest.mark.benchmark(group="box_convert")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert_xywh_xyxy(benchmark, dtype):
    boxes = np.random.random((100, 4)).astype(dtype)
    benchmark(box_convert, boxes, "xywh", "xyxy")

@pytest.mark.benchmark(group="masks_to_boxes")
def test_masks_to_boxes(benchmark):
    masks = np.array([True]*(100*100*100)).reshape((100, 100, 100))
    benchmark(masks_to_boxes, masks)
