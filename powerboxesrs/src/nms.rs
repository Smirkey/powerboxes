// Largely inspired by lsnms: https://github.com/remydubois/lsnms
use std::cmp::Ordering;

use crate::{boxes, utils};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Num, ToPrimitive};
use rstar::{RTree, RTreeNum, AABB};

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
pub fn nms<'a, N, BA, S, SA>(
    boxes: BA,
    scores: SA,
    iou_threshold: f64,
    score_threshold: S,
) -> Vec<usize>
where
    N: Num + PartialOrd + ToPrimitive + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
    S: Num + PartialOrd + ToPrimitive + Copy + 'a,
    SA: Into<ArrayView1<'a, S>>,
{
    let boxes = boxes.into();
    let scores = scores.into();
    let mut above_score_threshold: Vec<usize> = (0..scores.len()).collect();
    if score_threshold > S::zero() {
        // filter out boxes lower than score threshold
        above_score_threshold = scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= score_threshold)
            .map(|(idx, _)| idx)
            .collect();
    }

    let filtered_boxes = boxes.select(Axis(0), &above_score_threshold);
    // Compute areas once
    let areas = boxes::box_areas(&filtered_boxes);
    // sort box indices by scores
    above_score_threshold
        .sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let order = Array1::from(above_score_threshold);
    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = Array1::from_elem(order.len(), false);

    for i in 0..order.len() {
        let idx = order[i];
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let area1 = areas[i];
        let box1 = boxes.row(idx);
        for j in (i + 1)..order.len() {
            let idx_j = order[j];
            if suppress[j] {
                continue;
            }
            let area2 = areas[j];
            let box2 = boxes.row(idx_j);

            let mut iou = 0.0;
            let x1 = utils::max(box1[0], box2[0]);
            let x2 = utils::min(box1[2], box2[2]);
            let y1 = utils::max(box1[1], box2[1]);
            let y2 = utils::min(box1[3], box2[3]);
            if y2 > y1 && x2 > x1 {
                let intersection = (x2 - x1) * (y2 - y1);
                let intersection = intersection.to_f64().unwrap();
                let intersection = f64::min(intersection, f64::min(area1, area2));
                iou = intersection / (area1 + area2 - intersection + utils::EPS);
            }
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
pub fn rtree_nms<N>(
    boxes: &Array2<N>,
    scores: &Array1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> Vec<usize>
where
    N: RTreeNum + ToPrimitive + Send + Sync,
{
    let mut above_score_threshold: Vec<usize> = (0..scores.len()).collect();
    if score_threshold > utils::EPS {
        // filter out boxes lower than score threshold
        above_score_threshold = scores
            .iter()
            .enumerate()
            .filter(|(_, &score)| score >= score_threshold)
            .map(|(idx, _)| idx)
            .collect();
    }
    // Compute areas once
    let areas = boxes::box_areas(&boxes);
    // sort box indices by scores
    above_score_threshold
        .sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let order = Array1::from(above_score_threshold);
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
        let area1 = areas[i];
        let box1 = boxes.row(idx);

        for bbox in rtree.locate_in_envelope_intersecting(&AABB::from_corners(
            [box1[0], box1[1]],
            [box1[2], box1[3]],
        )) {
            let idx_j = bbox.index;
            if suppress[idx_j] {
                continue;
            }
            let area2 = areas[idx_j];
            let box2 = boxes.row(idx_j);

            let mut iou = 0.0;
            let x1 = utils::max(box1[0], box2[0]);
            let x2 = utils::min(box1[2], box2[2]);
            let y1 = utils::max(box1[1], box2[1]);
            let y2 = utils::min(box1[3], box2[3]);
            if y2 > y1 && x2 > x1 {
                let intersection = (x2 - x1) * (y2 - y1);
                let intersection = intersection.to_f64().unwrap();
                let intersection = f64::min(intersection, f64::min(area1, area2));
                iou = intersection / (area1 + area2 - intersection + utils::EPS);
            }
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
