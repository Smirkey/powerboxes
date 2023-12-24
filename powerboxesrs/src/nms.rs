use ndarray::{Array1, Array2};
use num_traits::{Num, ToPrimitive};

use crate::{boxes, iou};

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
    let areas = boxes::box_areas(boxes);
    // sort boxes by scores
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let order = Array1::from(indices);
    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = Array1::from_elem(scores.len(), false);
    // initialize cache for intersection
    for i in 0..scores.len() {
        if suppress[i] {
            continue;
        }
        let idx = order[i];
        if scores[idx] < score_threshold {
            break;
        }
        keep.push(idx);
        for j in (i + 1)..order.len() {
            if suppress[j] {
                continue;
            }
            let idx_j = order[j];
            let box1 = boxes.row(idx).to_owned();
            let box2 = boxes.row(idx_j).to_owned();

            let iou = iou::box_iou(&box1, &box2, areas[idx], areas[idx_j]);
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
        assert_eq!(keep, Array1::from(vec![1, 0]));
    }
}
