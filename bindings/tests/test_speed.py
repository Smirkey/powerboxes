import numpy as np
import pytest
from powerboxes import (
    box_convert,
    boxes_areas,
    diou_distance,
    giou_distance,
    iou_distance,
    masks_to_boxes,
    nms,
    parallel_giou_distance,
    parallel_iou_distance,
    remove_small_boxes,
    rotated_giou_distance,
    rotated_iou_distance,
    rotated_tiou_distance,
    rtree_nms,
    supported_dtypes,
    tiou_distance,
)

np.random.seed(42)

SCORES = np.random.random((100,))


@pytest.fixture(scope="function")
def generate_boxes(request):
    n_boxes = request.param if hasattr(request, "param") else 100
    im_size = 10_000
    topleft = np.random.uniform(0.0, high=im_size, size=(n_boxes, 2))
    wh = np.random.uniform(15, 45, size=topleft.shape)
    return np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)


@pytest.mark.benchmark(group="rotated_tiou_distance")
@pytest.mark.parametrize("dtype", ["float64"])
def test_rotated_tiou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 5)).astype(dtype)
    boxes2 = np.random.random((100, 5)).astype(dtype)
    benchmark(rotated_tiou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="rotated_iou_distance")
@pytest.mark.parametrize("dtype", ["float64"])
def test_rotated_iou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 5)).astype(dtype)
    boxes2 = np.random.random((100, 5)).astype(dtype)
    benchmark(rotated_iou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="rotated_giou_distance")
@pytest.mark.parametrize("dtype", ["float64"])
def test_rotated_giou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 5)).astype(dtype)
    boxes2 = np.random.random((100, 5)).astype(dtype)
    benchmark(rotated_giou_distance, boxes1, boxes2)


@pytest.mark.benchmark(group="tiou_distance")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_tiou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(tiou_distance, boxes1, boxes2)


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


@pytest.mark.benchmark(group="diou_distance")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_diou_distance(benchmark, dtype):
    boxes1 = np.random.random((100, 4)).astype(dtype)
    boxes2 = np.random.random((100, 4)).astype(dtype)
    benchmark(diou_distance, boxes1, boxes2)


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
    masks = np.array([True] * (100 * 100 * 100)).reshape((100, 100, 100))
    benchmark(masks_to_boxes, masks)


@pytest.mark.benchmark(group="nms")
@pytest.mark.parametrize("dtype", supported_dtypes)
def test_nms(benchmark, dtype, generate_boxes):
    boxes = generate_boxes
    boxes = boxes.astype(dtype)
    benchmark(nms, boxes, SCORES, 0.5, 0.5)


@pytest.mark.benchmark(group="nms")
@pytest.mark.parametrize("dtype", ["float64", "float32", "int64", "int32", "int16"])
def test_rtree_nms(benchmark, dtype, generate_boxes):
    boxes = generate_boxes
    boxes = boxes.astype(dtype)
    print(dtype)
    benchmark(rtree_nms, boxes, SCORES, 0.5, 0.5)


@pytest.mark.benchmark(group="nms_many_boxes")
@pytest.mark.parametrize("generate_boxes", [1000, 5000, 10000, 20000], indirect=True)
def test_nms_many_boxes(benchmark, generate_boxes):
    boxes = generate_boxes
    scores = np.random.random(len(boxes))
    benchmark(nms, boxes, scores, 0.5, 0.5)


@pytest.mark.benchmark(group="nms_many_boxes")
@pytest.mark.parametrize("generate_boxes", [1000, 5000, 10000, 20000], indirect=True)
def test_rtree_nms_many_boxes(benchmark, generate_boxes):
    boxes = generate_boxes
    scores = np.random.random(len(boxes))
    benchmark(rtree_nms, boxes, scores, 0.5, 0.5)
