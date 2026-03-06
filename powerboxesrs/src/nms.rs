// Largely inspired by lsnms: https://github.com/remydubois/lsnms
use std::cmp::Ordering;

use crate::utils;
use crate::rotation;
#[cfg(feature = "ndarray")]
use ndarray::{ArrayView1, ArrayView2, Axis};
use num_traits::{Num, ToPrimitive};
use rstar::{RTree, RTreeNum, AABB};

#[inline(always)]
fn area_f64<N>(bx: N, by: N, bxx: N, byy: N) -> f64
where
    N: ToPrimitive,
{
    (bxx.to_f64().unwrap() - bx.to_f64().unwrap()) * (byy.to_f64().unwrap() - by.to_f64().unwrap())
}

// ─── Slice-based core ───

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
///
/// # Arguments
///
/// * `boxes` - A flat slice of length `n * 4` representing the coordinates in xyxy format
///   of the bounding boxes (row-major).
/// * `scores` - A slice of length `n` representing the scores of the bounding boxes.
/// * `n` - The number of bounding boxes.
/// * `iou_threshold` - The IoU threshold to use for filtering.
/// * `score_threshold` - The score threshold to use for filtering.
///
/// # Returns
///
/// A `Vec<usize>` representing the indices of the bounding boxes to keep.
pub fn nms_slice<N>(
    boxes: &[N],
    scores: &[f64],
    n: usize,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy,
{
    let order: Vec<usize> = {
        let mut indices: Vec<_> = if score_threshold > utils::ZERO {
            scores
                .iter()
                .enumerate()
                .take(n)
                .filter(|(_, &score)| score >= score_threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..n).collect()
        };
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
        });
        indices
    };

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; order.len()];

    for (i, &idx) in order.iter().enumerate() {
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let (b1x, b1y, b1xx, b1yy) = utils::row4(boxes, idx);
        let area1 = area_f64(b1x, b1y, b1xx, b1yy);
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let (b2x, b2y, b2xx, b2yy) = utils::row4(boxes, order[j]);
            let x = utils::max(b1x, b2x);
            let y = utils::max(b1y, b2y);
            let xx = utils::min(b1xx, b2xx);
            let yy = utils::min(b1yy, b2yy);
            if x > xx || y > yy {
                continue;
            }
            let intersection = area_f64(x, y, xx, yy);
            let area2 = area_f64(b2x, b2y, b2xx, b2yy);
            let union = area1 + area2 - intersection;
            let iou = intersection / union;
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }
    keep
}

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
/// This function internally uses an R-tree to speed up the computation. It is recommended to use
/// this function when the number of boxes is large.
/// The R-tree implementation is based on the rstar crate. It allows queries in O(log n) time.
///
/// # Arguments
///
/// * `boxes` - A flat slice of length `n * 4` representing the coordinates in xyxy format
///   of the bounding boxes (row-major).
/// * `scores` - A slice of length `n` representing the scores of the bounding boxes.
/// * `n` - The number of bounding boxes.
/// * `iou_threshold` - The IoU threshold to use for filtering.
/// * `score_threshold` - The score threshold to use for filtering.
///
/// # Returns
///
/// A `Vec<usize>` representing the indices of the bounding boxes to keep.
pub fn rtree_nms_slice<N>(
    boxes: &[N],
    scores: &[f64],
    n: usize,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: RTreeNum + PartialEq + PartialOrd + ToPrimitive + Copy + Send + Sync,
{
    let order: Vec<usize> = {
        let mut indices: Vec<_> = if score_threshold > utils::ZERO {
            scores
                .iter()
                .enumerate()
                .take(n)
                .filter(|(_, &score)| score >= score_threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..n).collect()
        };
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
        });
        indices
    };

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; n];

    let rtree: RTree<utils::Bbox<N>> = RTree::bulk_load(
        order
            .iter()
            .map(|&idx| {
                let (x1, y1, x2, y2) = utils::row4(boxes, idx);
                utils::Bbox {
                    x1,
                    y1,
                    x2,
                    y2,
                    index: idx,
                }
            })
            .collect(),
    );

    for &idx in &order {
        if suppress[idx] {
            continue;
        }
        keep.push(idx);
        let (b1x, b1y, b1xx, b1yy) = utils::row4(boxes, idx);
        let area1 = area_f64(b1x, b1y, b1xx, b1yy);
        for bbox in
            rtree.locate_in_envelope_intersecting(&AABB::from_corners([b1x, b1y], [b1xx, b1yy]))
        {
            let idx_j = bbox.index;
            if suppress[idx_j] {
                continue;
            }
            let (b2x, b2y, b2xx, b2yy) = utils::row4(boxes, idx_j);
            let x = utils::max(b1x, b2x);
            let y = utils::max(b1y, b2y);
            let xx = utils::min(b1xx, b2xx);
            let yy = utils::min(b1yy, b2yy);
            if x > xx || y > yy {
                continue;
            }
            let intersection = area_f64(x, y, xx, yy);
            let area2 = area_f64(b2x, b2y, b2xx, b2yy);
            let union = area1 + area2 - intersection;
            let iou = intersection / union;
            if iou > iou_threshold {
                suppress[idx_j] = true;
            }
        }
    }
    keep
}

/// Performs non-maximum suppression (NMS) on a set of oriented bounding boxes using their scores and IoU.
///
/// # Arguments
///
/// * `boxes` - A flat slice of length `n * 5` representing the coordinates in cxcywha format
///   of the oriented bounding boxes (row-major).
/// * `scores` - A slice of length `n` representing the scores of the bounding boxes.
/// * `n` - The number of bounding boxes.
/// * `iou_threshold` - The IoU threshold to use for filtering.
/// * `score_threshold` - The score threshold to use for filtering.
///
/// # Returns
///
/// A `Vec<usize>` representing the indices of the oriented bounding boxes to keep.
pub fn rotated_nms_slice<N>(
    boxes: &[N],
    scores: &[f64],
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy,
{
    let order: Vec<usize> = {
        let mut indices: Vec<_> = if score_threshold > utils::ZERO {
            // filter out boxes lower than score threshold
            scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score >= score_threshold)
                .map(|(idx, _)| idx)
                .collect()
        } else {
            (0..scores.len()).collect()
        };
        // sort box indices by scores
        indices.sort_unstable_by(|&a, &b| {
            scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal)
        });
        indices
    };

    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = vec![false; order.len()];

    for (i, &idx) in order.iter().enumerate() {
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let box1 = utils::row5(boxes, idx);
        let w1 = box1.2.to_f64().unwrap();
        let h1 = box1.3.to_f64().unwrap();
        let area1 = h1 * w1;
        if area1 == 0.0 {
            continue;
        }
        let rect1 = rotation::Rect::new(
            box1.0.to_f64().unwrap(),
            box1.1.to_f64().unwrap(),
            w1,
            h1,
            box1.4.to_f64().unwrap(),
        );
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let box2 = utils::row5(boxes, order[j]);
            let w2 = box2.2.to_f64().unwrap();
            let h2 = box2.3.to_f64().unwrap();
            let area2 = w2 * h2;
            if area2 == 0.0 {
                continue;
            }
            let rect2 = rotation::Rect::new(
                box2.0.to_f64().unwrap(),
                box2.1.to_f64().unwrap(),
                w2,
                h2,
                box2.4.to_f64().unwrap(),
            );
            let intersection = rotation::intersection_area(&rect1, &rect2);

            if intersection == 0.0 {
                continue;
            }
            let union = area1 + area2 - intersection;
            let iou: f64 = intersection / union;
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }
    keep
}

// ─── ndarray wrappers ───

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
///
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 4)` representing the coordinates in xyxy format
///   of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A `Vec<usize>` representing the indices of the bounding boxes to keep.
///
/// # Examples
///
/// ```
/// use ndarray::{arr2, Array1};
/// use powerboxesrs::nms::nms;
///
/// let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
/// let scores = Array1::from(vec![1.0, 1.0]);
/// let keep = nms(&boxes, &scores, 0.8, 0.0);
/// assert_eq!(keep, vec![0, 1]);
/// ```
#[cfg(feature = "ndarray")]
pub fn nms<'a, N, BA, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy + PartialEq + 'a,
    BA: Into<ArrayView2<'a, N>>,
    SA: Into<ArrayView1<'a, f64>>,
{
    let boxes = boxes.into();
    let scores = scores.into();
    assert_eq!(boxes.nrows(), scores.len_of(Axis(0)));
    let n = boxes.nrows();
    let boxes_slice = boxes.as_slice().expect("boxes must be contiguous");
    let scores_slice = scores.as_slice().expect("scores must be contiguous");
    nms_slice(boxes_slice, scores_slice, n, iou_threshold, score_threshold)
}

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
/// This function internally uses an R-tree to speed up the computation. It is recommended to use
/// this function when the number of boxes is large.
/// The R-tree implementation is based on the rstar crate. It allows queries in O(log n) time.
///
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 4)` representing the coordinates in xyxy format
///   of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A `Vec<usize>` representing the indices of the bounding boxes to keep.
///
/// # Examples
///
/// ```
/// use ndarray::{arr2, Array1};
/// use powerboxesrs::nms::rtree_nms;
///
/// let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
/// let scores = Array1::from(vec![1.0, 1.0]);
/// let keep = rtree_nms(&boxes, &scores, 0.8, 0.0);
/// assert_eq!(keep, vec![0, 1]);
/// ```
#[cfg(feature = "ndarray")]
pub fn rtree_nms<'a, N, BA, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: RTreeNum + PartialEq + PartialOrd + ToPrimitive + Copy + PartialEq + Send + Sync + 'a,
    BA: Into<ArrayView2<'a, N>>,
    SA: Into<ArrayView1<'a, f64>>,
{
    let scores = scores.into();
    let boxes = boxes.into();
    let n = boxes.nrows();
    let boxes_slice = boxes.as_slice().expect("boxes must be contiguous");
    let scores_slice = scores.as_slice().expect("scores must be contiguous");
    rtree_nms_slice(boxes_slice, scores_slice, n, iou_threshold, score_threshold)
}

/// Performs non-maximum suppression (NMS) on a set of oriented bounding boxes using their scores and IoU.
///
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 5)` representing the coordinates in cxcywha format of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A 1D array of shape `(num_boxes,)` representing the indices of the oriented bounding boxes to keep.
///
/// # Examples
///
/// ```
/// use ndarray::{arr2, Array1};
/// use powerboxesrs::nms::rotated_nms;
///
/// let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0, 45.0], [1.0, 1.0, 3.0, 3.0, -45.0]]);
/// let scores = Array1::from(vec![1.0, 0.8]);
/// let keep = rotated_nms(&boxes, &scores, 0.2, 0.8);
/// assert_eq!(keep, vec![0]);
/// ```
#[cfg(feature = "ndarray")]
pub fn rotated_nms<'a, N, BA, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy + PartialEq + 'a,
    BA: Into<ArrayView2<'a, N>>,
    SA: Into<ArrayView1<'a, f64>>,
{
    let boxes = boxes.into();
    let scores = scores.into();
    assert_eq!(boxes.nrows(), scores.len_of(Axis(0)));
    let boxes_slice = boxes.as_slice().expect("boxes must be contiguous");
    let scores_slice = scores.as_slice().expect("scores must be contiguous");
    rotated_nms_slice(boxes_slice, scores_slice, iou_threshold, score_threshold)

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_slice_normal() {
        let boxes = vec![
            184.68927598, 850.65932762, 201.47437531, 866.02327337,
            185.68927598, 851.65932762, 200.47437531, 865.02327337,
            875.33814954, 706.46958933, 902.14487263, 737.14697788,
            874.33814954, 703.46958933, 901.14487263, 732.14697788,
            277.71729109, 744.81869575, 308.13768447, 777.11413807,
            275.71729109, 740.81869575, 310.13768447, 765.11413807,
        ];
        let scores = vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let keep = nms_slice(&boxes, &scores, 6, 0.5, 0.0);
        let keep_rtree = rtree_nms_slice(&boxes, &scores, 6, 0.5, 0.0);
        assert_eq!(keep, vec![0, 2, 4]);
        assert_eq!(keep_rtree, keep);
    }

    #[test]
    fn test_rotated_nms_slice_normal() {
        let boxes = vec![
            1.0, 2.0, 10.0, 5.0, 45.0,
            0.0, 1.0, 9.0, 4.0, 30.0,
            10.0, 20.0, 5.0, 8.0, -45.0,
        ];
        let scores = vec![0.9, 0.8, 0.7];
        let keep = rotated_nms_slice(&boxes, &scores, 0.5, 0.0);
        assert_eq!(keep, vec![0, 2]);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use ndarray::{arr2, Array1};
        use super::*;

        #[test]
        fn test_nms_normal_case() {
            let boxes = arr2(&[
                [184.68927598, 850.65932762, 201.47437531, 866.02327337],
                [185.68927598, 851.65932762, 200.47437531, 865.02327337],
                [875.33814954, 706.46958933, 902.14487263, 737.14697788],
                [874.33814954, 703.46958933, 901.14487263, 732.14697788],
                [277.71729109, 744.81869575, 308.13768447, 777.11413807],
                [275.71729109, 740.81869575, 310.13768447, 765.11413807],
            ]);
            let scores = Array1::from(vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4]);
            let keep = nms(&boxes, &scores, 0.5, 0.0);
            let keep_rtree = rtree_nms(&boxes, &scores, 0.5, 0.0);
            assert_eq!(keep, vec![0, 2, 4]);
            assert_eq!(keep_rtree, keep);
        }

        #[test]
        fn test_nms_empty_case() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
            let scores = Array1::from(vec![0.0, 0.0]);
            let keep = nms(&boxes, &scores, 0.5, 1.0);
            let keep_rtree = rtree_nms(&boxes, &scores, 0.5, 1.0);
            assert_eq!(keep, vec![]);
            assert_eq!(keep, keep_rtree)
        }

        #[test]
        fn test_nms_score_threshold() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
            let scores = Array1::from(vec![0.0, 1.0]);
            let keep = nms(&boxes, &scores, 0.5, 0.5);
            let keep_rtree = rtree_nms(&boxes, &scores, 0.5, 0.5);
            assert_eq!(keep, vec![1]);
            assert_eq!(keep, keep_rtree)
        }

        #[test]
        fn test_nms_iou_threshold() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
            let scores = Array1::from(vec![1.0, 1.0]);
            let keep = nms(&boxes, &scores, 0.8, 0.0);
            let keep_rtree = rtree_nms(&boxes, &scores, 0.8, 0.0);
            assert_eq!(keep, vec![0, 1]);
            assert_eq!(keep, keep_rtree)
        }

        #[test]
        fn test_rotated_nms_normal_case() {
            let boxes = arr2(&[
                [1.0, 2.0, 10.0, 5.0, 45.0],
                [0.0, 1.0, 9.0, 4.0, 30.0],
                [10.0, 20.0, 5.0, 8.0, -45.0],
            ]);
            let scores = Array1::from(vec![0.9, 0.8, 0.7]);
            let keep = rotated_nms(&boxes, &scores, 0.5, 0.0);
            assert_eq!(keep, vec![0, 2]);
        }

        #[test]
        fn test_rotated_nms_empty_case() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0, 10.0], [1.0, 1.0, 3.0, 3.0, 10.0]]);
            let scores = Array1::from(vec![0.0, 0.0]);
            let keep = rotated_nms(&boxes, &scores, 0.5, 1.0);
            assert_eq!(keep, vec![]);
        }

        #[test]
        fn test_rotated_nms_score_threshold() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0, 10.0], [1.0, 1.0, 3.0, 3.0, 12.0]]);
            let scores = Array1::from(vec![0.0, 1.0]);
            let keep = rotated_nms(&boxes, &scores, 0.5, 0.5);
            assert_eq!(keep, vec![1]);
        }

        #[test]
        fn test_rotated_nms_iou_threshold() {
            let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0, 10.0], [1.0, 1.0, 3.0, 3.0, 45.0]]);
            let scores = Array1::from(vec![1.0, 1.0]);
            let keep = rotated_nms(&boxes, &scores, 0.8, 0.0);
            assert_eq!(keep, vec![0, 1]);
        }

    }
}
