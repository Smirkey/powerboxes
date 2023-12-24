use std::cmp::Ordering;

use ndarray::{Array1, Array2};
use num_traits::{Num, ToPrimitive};

use crate::{boxes, utils};

/// Performs non-maximum suppression (NMS) on a set of bounding boxes using their scores.
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
/// assert_eq!(keep, Array1::from(vec![0, 1]));
/// ```
pub fn nms<N>(
    boxes: &Array2<N>,
    scores: &Array1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> Array1<usize>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    // Compute areas once
    let areas = boxes::box_areas(&boxes);
    // sort boxes by scores
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let order = Array1::from(indices);
    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = Array1::from_elem(scores.len(), false);

    for i in 0..scores.len() {
        let idx = order[i];
        if scores[idx] < score_threshold {
            break;
        }
        if suppress[i] {
            continue;
        }
        keep.push(idx);
        let area1 = areas[idx];
        let box1 = boxes.row(idx);
        for j in (i + 1)..scores.len() {
            if suppress[j] {
                continue;
            }
            let idx_j = order[j];
            let area2 = areas[idx_j];
            let box2 = boxes.row(idx_j);

            let mut iou = 0.0;
            let a1_x1 = box1[0];
            let a1_y1 = box1[1];
            let a1_x2 = box1[2];
            let a1_y2 = box1[3];
            let a2_x1 = box2[0];
            let a2_y1 = box2[1];
            let a2_x2 = box2[2];
            let a2_y2 = box2[3];
            let x1 = utils::max(a1_x1, a2_x1);
            let x2 = utils::min(a1_x2, a2_x2);
            let y1 = utils::max(a1_y1, a2_y1);
            let y2 = utils::min(a1_y2, a2_y2);
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
    return Array1::from(keep);
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
        assert_eq!(keep, Array1::from(vec![0, 2, 4]));
    }

    #[test]
    fn test_nms_empty_case() {
        // empty case
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![0.0, 0.0]);
        let keep = nms(&boxes, &scores, 0.5, 1.0);
        assert_eq!(keep, Array1::from(vec![]));
    }

    #[test]
    fn test_nms_score_threshold() {
        // score threshold
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![0.0, 1.0]);
        let keep = nms(&boxes, &scores, 0.5, 0.5);
        assert_eq!(keep, Array1::from(vec![1]));
    }

    #[test]
    fn test_nms_iou_threshold() {
        // iou threshold
        let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
        let scores = Array1::from(vec![1.0, 1.0]);
        let keep = nms(&boxes, &scores, 0.8, 0.0);
        assert_eq!(keep, Array1::from(vec![0, 1]));
    }
}
