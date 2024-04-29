// Largely inspired by lsnms: https://github.com/remydubois/lsnms
use std::cmp::Ordering;

use crate::utils;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use num_traits::{Num, ToPrimitive};
use rstar::{RTree, RTreeNum, AABB};

#[inline(always)]
pub fn area<N>(bx: N, by: N, bxx: N, byy: N) -> N
where
    N: Num + PartialEq + PartialOrd + ToPrimitive,
{
    (bxx - bx) * (byy - by)
}

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores and IoU.
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 4)` representing the coordinates in xyxy format of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A 1D array of shape `(num_boxes,)` representing the indices of the bounding boxes to keep.
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
        let box1 = boxes.row(idx);
        let b1x = box1[0];
        let b1y = box1[1];
        let b1xx = box1[2];
        let b1yy = box1[3];
        let area1 = area(b1x, b1y, b1xx, b1yy);
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let box2 = boxes.row(order[j]);
            let b2x = box2[0];
            let b2y = box2[1];
            let b2xx = box2[2];
            let b2yy = box2[3];

            // Intersection-over-union
            let x = utils::max(b1x, b2x);
            let y = utils::max(b1y, b2y);
            let xx = utils::min(b1xx, b2xx);
            let yy = utils::min(b1yy, b2yy);
            if x > xx || y > yy {
                // Boxes are not intersecting at all
                continue;
            };
            // Boxes are intersecting
            let intersection: N = area(x, y, xx, yy);
            let area2: N = area(b2x, b2y, b2xx, b2yy);
            let union: N = area1 + area2 - intersection;
            let iou: f64 = intersection.to_f64().unwrap() / union.to_f64().unwrap();
            if iou > iou_threshold {
                suppress[j] = true;
            }
        }
    }
    keep
}

/// Performs non-maximum suppression (NMS) on a set of bounding using their score and IoU.
/// This function internally uses an RTree to speed up the computation. It is recommended to use this function
/// when the number of boxes is large.
/// The RTree implementation is based on the rstar crate. It allows to perform queries in O(log n) time.
///
/// # Arguments
///
/// * `boxes` - A 2D array of shape `(num_boxes, 4)` representing the coordinates in xyxy format of the bounding boxes.
/// * `scores` - A 1D array of shape `(num_boxes,)` representing the scores of the bounding boxes.
/// * `iou_threshold` - A float representing the IoU threshold to use for filtering.
/// * `score_threshold` - A float representing the score threshold to use for filtering.
///
/// # Returns
///
/// A 1D array of shape `(num_boxes,)` representing the indices of the bounding boxes to keep.
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
    let mut suppress = Array1::from_elem(scores.len(), false);

    // build rtree
    let rtree: RTree<utils::Bbox<N>> = RTree::bulk_load(
        order
            .iter()
            .map(|&idx| {
                let box_ = boxes.row(idx);
                utils::Bbox {
                    x1: box_[0],
                    y1: box_[1],
                    x2: box_[2],
                    y2: box_[3],
                    index: idx,
                }
            })
            .collect(),
    );
    for i in 0..order.len() {
        let idx = order[i];
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let box1 = boxes.row(idx);
        let b1x = box1[0];
        let b1y = box1[1];
        let b1xx = box1[2];
        let b1yy = box1[3];
        let area1 = area(b1x, b1y, b1xx, b1yy);
        for bbox in
            rtree.locate_in_envelope_intersecting(&AABB::from_corners([b1x, b1y], [b1xx, b1yy]))
        {
            let idx_j = bbox.index;
            if suppress[idx_j] {
                continue;
            }
            let box2 = boxes.row(idx_j);
            let b2x = box2[0];
            let b2y = box2[1];
            let b2xx = box2[2];
            let b2yy = box2[3];

            // Intersection-over-union
            let x = utils::max(b1x, b2x);
            let y = utils::max(b1y, b2y);
            let xx = utils::min(b1xx, b2xx);
            let yy = utils::min(b1yy, b2yy);
            if x > xx || y > yy {
                // Boxes are not intersecting at all
                continue;
            };
            // Boxes are intersecting
            let intersection: N = area(x, y, xx, yy);
            let area2: N = area(b2x, b2y, b2xx, b2yy);
            let union: N = area1 + area2 - intersection;
            let iou: f64 = intersection.to_f64().unwrap() / union.to_f64().unwrap();
            if iou > iou_threshold {
                suppress[idx_j] = true;
            }
        }
    }
    keep
}

#[cfg(test)]
mod tests {
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
        // empty case
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![0.0, 0.0]);
        let keep = nms(&boxes, &scores, 0.5, 1.0);
        let keep_rtree = rtree_nms(&boxes, &scores, 0.5, 1.0);

        assert_eq!(keep, vec![]);
        assert_eq!(keep, keep_rtree)
    }

    #[test]
    fn test_nms_score_threshold() {
        // score threshold
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![0.0, 1.0]);
        let keep = nms(&boxes, &scores, 0.5, 0.5);
        let keep_rtree = rtree_nms(&boxes, &scores, 0.5, 0.5);
        assert_eq!(keep, vec![1]);
        assert_eq!(keep, keep_rtree)
    }

    #[test]
    fn test_nms_iou_threshold() {
        // iou threshold
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![1.0, 1.0]);
        let keep = nms(&boxes, &scores, 0.8, 0.0);
        let keep_rtree = rtree_nms(&boxes, &scores, 0.8, 0.0);
        assert_eq!(keep, vec![0, 1]);
        assert_eq!(keep, keep_rtree)
    }
}
