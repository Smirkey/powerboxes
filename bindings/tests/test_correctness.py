"""Correctness tests against shapely reference implementations.

Every distance / NMS function is tested against a pure-Python + shapely
reference so we can be confident the Rust implementations are correct.
"""

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box as shapely_box

from powerboxes import (
    boxes_areas,
    giou_distance,
    iou_distance,
    nms,
    parallel_giou_distance,
    parallel_iou_distance,
    rotated_giou_distance,
    rotated_iou_distance,
    rotated_nms,
    rotated_tiou_distance,
    rtree_nms,
    rtree_rotated_nms,
    tiou_distance,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _xyxy_to_polygon(x1, y1, x2, y2):
    return shapely_box(x1, y1, x2, y2)


def _cxcywha_to_polygon(cx, cy, w, h, a):
    dx, dy = w / 2, h / 2
    poly = Polygon([(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)])
    poly = rotate(poly, a, origin=(0, 0), use_radians=False)
    poly = translate(poly, xoff=cx, yoff=cy)
    return poly


# ---------------------------------------------------------------------------
# axis-aligned reference helpers
# ---------------------------------------------------------------------------


def _ref_iou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.zeros((n1, n2))
    for i in range(n1):
        p1 = _xyxy_to_polygon(*boxes1[i])
        a1 = p1.area
        for j in range(n2):
            p2 = _xyxy_to_polygon(*boxes2[j])
            a2 = p2.area
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union == 0:
                result[i, j] = 1.0
            else:
                result[i, j] = 1.0 - inter / union
    return result


def _ref_giou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.zeros((n1, n2))
    for i in range(n1):
        p1 = _xyxy_to_polygon(*boxes1[i])
        a1 = p1.area
        for j in range(n2):
            p2 = _xyxy_to_polygon(*boxes2[j])
            a2 = p2.area
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union == 0:
                iou = 0.0
            else:
                iou = inter / union
            # enclosing box
            cx1 = min(boxes1[i][0], boxes2[j][0])
            cy1 = min(boxes1[i][1], boxes2[j][1])
            cx2 = max(boxes1[i][2], boxes2[j][2])
            cy2 = max(boxes1[i][3], boxes2[j][3])
            c_area = (cx2 - cx1) * (cy2 - cy1)
            giou = iou - (c_area - union) / c_area
            result[i, j] = 1.0 - giou
    return result


def _ref_tiou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.zeros((n1, n2))
    for i in range(n1):
        a1 = _xyxy_to_polygon(*boxes1[i]).area
        for j in range(n2):
            a2 = _xyxy_to_polygon(*boxes2[j]).area
            cx1 = min(boxes1[i][0], boxes2[j][0])
            cy1 = min(boxes1[i][1], boxes2[j][1])
            cx2 = max(boxes1[i][2], boxes2[j][2])
            cy2 = max(boxes1[i][3], boxes2[j][3])
            c_area = (cx2 - cx1) * (cy2 - cy1)
            result[i, j] = 1.0 - min(a1 / c_area, a2 / c_area)
    return result


def _ref_nms(boxes, scores, iou_threshold, score_threshold):
    order = np.argsort(-scores)
    if score_threshold > 0:
        order = order[scores[order] >= score_threshold]
    keep = []
    suppressed = set()
    for i, idx in enumerate(order):
        if idx in suppressed:
            continue
        keep.append(idx)
        p1 = _xyxy_to_polygon(*boxes[idx])
        a1 = p1.area
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if jdx in suppressed:
                continue
            p2 = _xyxy_to_polygon(*boxes[jdx])
            a2 = p2.area
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(jdx)
    return np.array(keep)


# ---------------------------------------------------------------------------
# rotated reference helpers
# ---------------------------------------------------------------------------


def _ref_rotated_iou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.ones((n1, n2))
    for i in range(n1):
        p1 = _cxcywha_to_polygon(*boxes1[i])
        a1 = p1.area
        for j in range(n2):
            p2 = _cxcywha_to_polygon(*boxes2[j])
            a2 = p2.area
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union > 0:
                result[i, j] = 1.0 - inter / union
    return result


def _ref_rotated_giou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.ones((n1, n2))
    for i in range(n1):
        p1 = _cxcywha_to_polygon(*boxes1[i])
        a1 = p1.area
        b1 = p1.bounds  # (minx, miny, maxx, maxy)
        for j in range(n2):
            p2 = _cxcywha_to_polygon(*boxes2[j])
            a2 = p2.area
            b2 = p2.bounds
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union == 0:
                continue
            iou = inter / union
            # enclosing AABB of both polygons' AABBs
            cx1 = min(b1[0], b2[0])
            cy1 = min(b1[1], b2[1])
            cx2 = max(b1[2], b2[2])
            cy2 = max(b1[3], b2[3])
            c_area = (cx2 - cx1) * (cy2 - cy1)
            giou = iou - (c_area - union) / c_area
            result[i, j] = 1.0 - giou
    return result


def _ref_rotated_tiou_distance(boxes1, boxes2):
    n1, n2 = len(boxes1), len(boxes2)
    result = np.ones((n1, n2))
    for i in range(n1):
        p1 = _cxcywha_to_polygon(*boxes1[i])
        a1 = p1.area
        b1 = p1.bounds
        for j in range(n2):
            p2 = _cxcywha_to_polygon(*boxes2[j])
            a2 = p2.area
            b2 = p2.bounds
            cx1 = min(b1[0], b2[0])
            cy1 = min(b1[1], b2[1])
            cx2 = max(b1[2], b2[2])
            cy2 = max(b1[3], b2[3])
            c_area = (cx2 - cx1) * (cy2 - cy1)
            result[i, j] = 1.0 - min(a1 / c_area, a2 / c_area)
    return result


def _ref_rotated_nms(boxes, scores, iou_threshold, score_threshold):
    order = np.argsort(-scores)
    if score_threshold > 0:
        order = order[scores[order] >= score_threshold]
    keep = []
    suppressed = set()
    for i, idx in enumerate(order):
        if idx in suppressed:
            continue
        keep.append(idx)
        p1 = _cxcywha_to_polygon(*boxes[idx])
        a1 = p1.area
        if a1 == 0:
            continue
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if jdx in suppressed:
                continue
            p2 = _cxcywha_to_polygon(*boxes[jdx])
            a2 = p2.area
            if a2 == 0:
                continue
            inter = p1.intersection(p2).area
            union = a1 + a2 - inter
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(jdx)
    return np.array(keep)


# ---------------------------------------------------------------------------
# random box generators
# ---------------------------------------------------------------------------


def _random_xyxy_boxes(rng, n, im_size=1000):
    topleft = rng.uniform(0, im_size, size=(n, 2))
    wh = rng.uniform(10, 100, size=(n, 2))
    return np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)


def _random_rotated_boxes(rng, n, im_size=1000):
    center = rng.uniform(10, im_size, size=(n, 2))
    wh = rng.uniform(10, 100, size=(n, 2))
    angle = rng.uniform(-90, 90, size=(n, 1))
    return np.concatenate([center, wh, angle], axis=1).astype(np.float64)


# ===================================================================
# Axis-aligned distance tests
# ===================================================================


@pytest.mark.parametrize("seed", range(5))
def test_iou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_xyxy_boxes(rng, 20)
    boxes2 = _random_xyxy_boxes(rng, 15)
    result = iou_distance(boxes1, boxes2)
    ref = _ref_iou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_parallel_iou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_xyxy_boxes(rng, 20)
    boxes2 = _random_xyxy_boxes(rng, 15)
    result = parallel_iou_distance(boxes1, boxes2)
    ref = _ref_iou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_giou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_xyxy_boxes(rng, 20)
    boxes2 = _random_xyxy_boxes(rng, 15)
    result = giou_distance(boxes1, boxes2)
    ref = _ref_giou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_parallel_giou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_xyxy_boxes(rng, 20)
    boxes2 = _random_xyxy_boxes(rng, 15)
    result = parallel_giou_distance(boxes1, boxes2)
    ref = _ref_giou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_tiou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_xyxy_boxes(rng, 20)
    boxes2 = _random_xyxy_boxes(rng, 15)
    result = tiou_distance(boxes1, boxes2)
    ref = _ref_tiou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_boxes_areas_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes = _random_xyxy_boxes(rng, 30)
    result = boxes_areas(boxes)
    ref = np.array([_xyxy_to_polygon(*b).area for b in boxes])
    np.testing.assert_allclose(result, ref, atol=1e-10)


# ===================================================================
# Axis-aligned NMS tests
# ===================================================================


@pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("score_threshold", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("seed", range(5))
def test_nms_against_shapely(seed, iou_threshold, score_threshold):
    rng = np.random.default_rng(seed)
    boxes = _random_xyxy_boxes(rng, 50)
    scores = rng.random(len(boxes))
    result = nms(boxes, scores, iou_threshold, score_threshold)
    ref = _ref_nms(boxes, scores, iou_threshold, score_threshold)
    np.testing.assert_array_equal(result, ref)


@pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("score_threshold", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("seed", range(5))
def test_rtree_nms_against_shapely(seed, iou_threshold, score_threshold):
    rng = np.random.default_rng(seed)
    boxes = _random_xyxy_boxes(rng, 50)
    scores = rng.random(len(boxes))
    result = rtree_nms(boxes, scores, iou_threshold, score_threshold)
    ref = _ref_nms(boxes, scores, iou_threshold, score_threshold)
    np.testing.assert_array_equal(result, ref)


# ===================================================================
# Rotated distance tests
# ===================================================================


@pytest.mark.parametrize("seed", range(5))
def test_rotated_iou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_rotated_boxes(rng, 20)
    boxes2 = _random_rotated_boxes(rng, 15)
    result = rotated_iou_distance(boxes1, boxes2)
    ref = _ref_rotated_iou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_rotated_giou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_rotated_boxes(rng, 20)
    boxes2 = _random_rotated_boxes(rng, 15)
    result = rotated_giou_distance(boxes1, boxes2)
    ref = _ref_rotated_giou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-8)


@pytest.mark.parametrize("seed", range(5))
def test_rotated_tiou_distance_against_shapely(seed):
    rng = np.random.default_rng(seed)
    boxes1 = _random_rotated_boxes(rng, 20)
    boxes2 = _random_rotated_boxes(rng, 15)
    result = rotated_tiou_distance(boxes1, boxes2)
    ref = _ref_rotated_tiou_distance(boxes1, boxes2)
    np.testing.assert_allclose(result, ref, atol=1e-8)


# ===================================================================
# Rotated NMS tests
# ===================================================================


@pytest.mark.parametrize("iou_threshold", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("score_threshold", [0.0, 0.3, 0.5])
@pytest.mark.parametrize("seed", range(5))
def test_rotated_nms_against_shapely(seed, iou_threshold, score_threshold):
    rng = np.random.default_rng(seed)
    n = rng.integers(10, 100)
    boxes = _random_rotated_boxes(rng, n)
    scores = rng.random(n)
    result = rotated_nms(boxes, scores, iou_threshold, score_threshold)
    ref = _ref_rotated_nms(boxes, scores, iou_threshold, score_threshold)
    np.testing.assert_array_equal(result, ref)
    result_rtree = rtree_rotated_nms(boxes, scores, iou_threshold, score_threshold)
    np.testing.assert_array_equal(result_rtree, ref)
