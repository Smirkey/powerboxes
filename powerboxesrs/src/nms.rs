use ndarray::{Array, Array1, Array2};
use num_traits::{Num, ToPrimitive};

use crate::iou;

fn nms<N>(
    boxes: &Array2<N>,
    scores: &Array1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> Array1<usize>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    indices = indices.into_iter().rev().collect::<Vec<usize>>();
    let order = Array1::from(indices);
    let mut keep: Vec<usize> = Vec::new();
    let mut suppress = Array1::from_elem(scores.len(), false);
    for i in 0..scores.len() {
        if suppress[i] {
            continue;
        }
        let idx = order[i];
        if scores[idx] < score_threshold {
            break;
        }
        keep.push(idx);
        for j in (i + 1)..scores.len() {
            if suppress[j] {
                continue;
            }
            let idx_j = order[j];
            let box1_as_array2 = Array::from_shape_vec((1, 4), boxes.row(idx).to_vec()).unwrap();
            let box2_as_array2 = Array::from_shape_vec((1, 4), boxes.row(idx_j).to_vec()).unwrap();
            let iou = iou::iou(&box1_as_array2, &box2_as_array2).row(0)[0];
            if iou > iou_threshold {
                suppress[idx_j] = true;
            }
        }
    }
    return Array1::from(keep);
}

#[test]
fn test_nms() {
    use ndarray::arr2;

    // empty case
    let boxes = arr2(&[[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]]);
    let scores = Array1::from(vec![0.0, 0.0]);
    let keep = nms(&boxes, &scores, 0.5, 1.0);
    assert_eq!(keep, Array1::from(vec![]));

    // normal case
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
