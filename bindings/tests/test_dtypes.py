import numpy as np
import pytest
from powerboxes import (
    BOXES_NOT_NP_ARRAY,
    BOXES_NOT_SAME_TYPE,
    box_convert,
    boxes_areas,
    giou_distance,
    iou_distance,
    parallel_giou_distance,
    parallel_iou_distance,
    remove_small_boxes,
    supported_dtypes
)

np.random.seed(42)


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_giou_distance(dtype):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    giou_distance(boxes1.astype(dtype), boxes2.astype(dtype))


def test_giou_distance_wrong_dtype():
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    with pytest.raises(ValueError, match=BOXES_NOT_SAME_TYPE):
        giou_distance(boxes1.astype(np.float64), boxes2.astype(np.uint8))


def test_giou_distance_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        giou_distance("bonjour", "how are you?")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_parallel_giou_distance(dtype):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    parallel_giou_distance(boxes1.astype(dtype), boxes2.astype(dtype))


def test_parallel_giou_distance_wrong_dtype():
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    with pytest.raises(ValueError, match=BOXES_NOT_SAME_TYPE):
        parallel_giou_distance(boxes1.astype(np.float64), boxes2.astype(np.uint8))


def test_parallel_giou_distance_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        parallel_giou_distance("bonjour", "how are you?")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_parallel_iou_distance(dtype):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    parallel_iou_distance(boxes1.astype(dtype), boxes2.astype(dtype))


def test_parallel_iou_distance_wrong_dtype():
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    with pytest.raises(ValueError, match=BOXES_NOT_SAME_TYPE):
        parallel_iou_distance(boxes1.astype(np.float64), boxes2.astype(np.uint8))


def test_parallel_iou_distance_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        parallel_iou_distance("bonjour", "how are you?")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_iou_distance(dtype):
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    iou_distance(boxes1.astype(dtype), boxes2.astype(dtype))


def test_iou_distance_wrong_dtype():
    boxes1 = np.random.random((100, 4))
    boxes2 = np.random.random((100, 4))
    with pytest.raises(ValueError, match=BOXES_NOT_SAME_TYPE):
        iou_distance(boxes1.astype(np.float64), boxes2.astype(np.uint8))


def test_iou_distance_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        iou_distance("bonjour", "how are you?")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_remove_small_boxes(dtype):
    boxes = np.random.random((100, 4))
    remove_small_boxes(boxes.astype(dtype), 0.4)


def test_remove_small_boxes_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        remove_small_boxes("bonjour", "how are you?")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_boxes_areas(dtype):
    boxes = np.random.random((100, 4))
    boxes_areas(boxes.astype(dtype))


def test_boxes_areas_bad_inpus():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        boxes_areas("hey")


@pytest.mark.parametrize("dtype", supported_dtypes)
def test_box_convert(dtype):
    boxes = np.random.random((100, 4))
    box_convert(boxes.astype(dtype), "xyxy", "xywh")


def test_box_convert_bad_inputs():
    with pytest.raises(TypeError, match=BOXES_NOT_NP_ARRAY):
        box_convert("foo", "xyxy", "xywh")
