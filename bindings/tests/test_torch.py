"""Tests comparing powerboxes NMS against torchvision.ops.nms.

Inspired by torchvision's own NMS test suite (test/test_ops.py::TestNMS).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from torchvision.ops import nms as torch_nms

from powerboxes import nms, rtree_nms

NMS_FUNCS = pytest.mark.parametrize(
    "nms_func", [nms, rtree_nms], ids=["nms", "rtree_nms"]
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _random_xyxy_boxes(rng, n, im_size=1000):
    topleft = rng.uniform(0, im_size, size=(n, 2))
    wh = rng.uniform(10, 100, size=(n, 2))
    return np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)


def _create_boxes_with_iou(rng, n, iou_thresh):
    """Create N random boxes where the last box has a controlled IoU with the first.

    Mirrors torchvision's _create_tensors_with_iou: the last box is a copy of
    the first, widened so that IoU(box[0], box[-1]) just barely exceeds
    iou_thresh.  This guarantees at least one suppression at the boundary.
    """
    boxes = _random_xyxy_boxes(rng, n)
    boxes[-1, :] = boxes[0, :]
    x0, y0, x1, y1 = boxes[-1]
    adjusted_thresh = iou_thresh + 1e-5
    boxes[-1, 2] += (x1 - x0) * (1 - adjusted_thresh) / adjusted_thresh
    scores = rng.random(n)
    return boxes, scores


def _pb_nms(nms_func, boxes, scores, iou_threshold):
    return nms_func(boxes, scores, iou_threshold, score_threshold=0.0)


def _tv_nms(boxes, scores, iou_threshold):
    torch_boxes = torch.from_numpy(boxes.astype(np.float32))
    torch_scores = torch.from_numpy(scores.astype(np.float32))
    return torch_nms(torch_boxes, torch_scores, iou_threshold).numpy()


# ---------------------------------------------------------------------------
# random box tests (torchvision-style _create_tensors_with_iou)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iou_threshold", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", range(10))
@NMS_FUNCS
def test_nms_against_torchvision(seed, iou_threshold, nms_func):
    rng = np.random.default_rng(seed)
    boxes, scores = _create_boxes_with_iou(rng, 1000, iou_threshold)
    pb_keep = _pb_nms(nms_func, boxes, scores, iou_threshold)
    tv_keep = _tv_nms(boxes, scores, iou_threshold)
    np.testing.assert_array_equal(pb_keep, tv_keep)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


@NMS_FUNCS
def test_nms_single_box(nms_func):
    boxes = np.array([[10.0, 20.0, 50.0, 60.0]])
    scores = np.array([0.9])
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.5)
    tv_keep = _tv_nms(boxes, scores, 0.5)
    np.testing.assert_array_equal(pb_keep, tv_keep)
    assert len(pb_keep) == 1


@NMS_FUNCS
def test_nms_identical_boxes(nms_func):
    """All boxes identical — only the highest-scoring one should survive."""
    n = 10
    boxes = np.tile([100.0, 100.0, 200.0, 200.0], (n, 1))
    scores = np.arange(n, dtype=np.float64)
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.5)
    tv_keep = _tv_nms(boxes, scores, 0.5)
    np.testing.assert_array_equal(pb_keep, tv_keep)
    assert len(pb_keep) == 1


@NMS_FUNCS
def test_nms_no_overlap(nms_func):
    """Non-overlapping boxes — all should be kept."""
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [50.0, 50.0, 60.0, 60.0],
            [100.0, 100.0, 110.0, 110.0],
            [200.0, 200.0, 210.0, 210.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.5)
    tv_keep = _tv_nms(boxes, scores, 0.5)
    np.testing.assert_array_equal(pb_keep, tv_keep)
    assert len(pb_keep) == 4


@NMS_FUNCS
def test_nms_descending_score_order(nms_func):
    """Kept indices should follow descending-score order."""
    rng = np.random.default_rng(42)
    boxes = _random_xyxy_boxes(rng, 200)
    scores = rng.random(200)
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.5)
    tv_keep = _tv_nms(boxes, scores, 0.5)
    np.testing.assert_array_equal(pb_keep, tv_keep)
    # scores of kept boxes should be in non-increasing order
    kept_scores = scores[pb_keep]
    assert np.all(kept_scores[:-1] >= kept_scores[1:])


@NMS_FUNCS
def test_nms_high_overlap_low_threshold(nms_func):
    """Highly overlapping boxes with a strict threshold — heavy suppression."""
    base = np.array([100.0, 100.0, 200.0, 200.0])
    rng = np.random.default_rng(7)
    # Small random jitter so boxes overlap heavily
    boxes = base + rng.uniform(-5, 5, size=(50, 4))
    boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)  # ensure valid
    scores = rng.random(50)
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.2)
    tv_keep = _tv_nms(boxes, scores, 0.2)
    np.testing.assert_array_equal(pb_keep, tv_keep)


@NMS_FUNCS
def test_nms_concrete_float16_case(nms_func):
    """Concrete example from torchvision's test_nms_float16."""
    boxes = np.array(
        [
            [285.3538, 185.5758, 1193.5110, 851.4551],
            [285.1472, 188.7374, 1192.4984, 851.0669],
            [279.2440, 197.9812, 1189.4746, 849.2019],
        ]
    )
    scores = np.array([0.6370, 0.7569, 0.3966])
    pb_keep = _pb_nms(nms_func, boxes, scores, 0.2)
    tv_keep = _tv_nms(boxes, scores, 0.2)
    np.testing.assert_array_equal(pb_keep, tv_keep)
