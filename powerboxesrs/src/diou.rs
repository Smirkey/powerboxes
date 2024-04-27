use crate::{boxes, utils};
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, Num};

/// Calculates the intersection over union (DIoU) distance between two sets of bounding boxes.
/// https://arxiv.org/pdf/1911.08287.pdf
///
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape (N, 4) representing N bounding boxes in xyxy format.
/// * `boxes2` - A 2D array of shape (M, 4) representing M bounding boxes in xyxy format.
///
/// # Returns
///
/// A 2D array of shape (N, M) representing the DIoU distance between each pair of bounding boxes
/// ```
pub fn diou_distance<'a, BA, N>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    N: Num + Float + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes1 = boxes1.into();
    let boxes2 = boxes2.into();
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();
    let two = N::from(2).unwrap();
    let mut diou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::box_areas(&boxes1);
    let areas_boxes2 = boxes::box_areas(&boxes2);
    for (i, a1) in boxes1.outer_iter().enumerate() {
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = areas_boxes1[i];

        for (j, a2) in boxes2.outer_iter().enumerate() {
            let a2_x1 = a2[0];
            let a2_y1 = a2[1];
            let a2_x2 = a2[2];
            let a2_y2 = a2[3];
            let area2 = areas_boxes2[j];
            let x1 = utils::max(a1_x1, a2_x1);
            let y1 = utils::max(a1_y1, a2_y1);
            let x2 = utils::min(a1_x2, a2_x2);
            let y2 = utils::min(a1_y2, a2_y2);
            if x2 < x1 || y2 < y1 {
                diou_matrix[[i, j]] = utils::ONE;
                continue;
            }
            let intersection = (x2 - x1) * (y2 - y1);
            let intersection = intersection.to_f64().unwrap();
            let intersection = utils::min(intersection, utils::min(area1, area2));
            let iou = intersection / (area1 + area2 - intersection + utils::EPS);

            let center_box1 = [(a1_x1 + a1_x2) / two, (a1_y1 + a1_y2) / two];
            let center_box2 = [(a2_x1 + a2_x2) / two, (a2_y1 + a2_y2) / two];

            let d = Float::sqrt(
                Float::powf(center_box1[0] - center_box2[0], two)
                    + Float::powf(center_box1[1] - center_box2[1], two),
            );
            let c = Float::sqrt(Float::powf(x2 - x1, two) + Float::powf(y2 - y1, two));
            let diou_penalty = Float::powf(d, two) / Float::powf(c, two);
            diou_matrix[[i, j]] = utils::ONE - (iou - diou_penalty.to_f64().unwrap());
        }
    }

    diou_matrix
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_diou_distance() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
        let diou_distance_result = diou_distance(&boxes1, &boxes2);

        assert_eq!(diou_distance_result, arr2(&[[1.8571428571428572]]));
    }

    #[test]
    fn test_diou_distance_distance2() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
        let diou_distance_result = diou_distance(&boxes1, &boxes2);
        assert_eq!(diou_distance_result, arr2(&[[1.0]]));
    }

    #[test]
    fn test_diou_distance_distance3() {
        let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
        let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
        let diou_distance_result = diou_distance(&boxes1, &boxes2);
        assert_eq!(diou_distance_result, arr2(&[[3.187499999999999]]));
    }

    #[test]
    fn test_diou_distance_distance4() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let diou_distance_result = diou_distance(&boxes1, &boxes2);
        assert_eq!(diou_distance_result, arr2(&[[0.0]]));
    }

    #[test]
    fn test_diou_distance_distance5() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
        let diou_distance_result = diou_distance(&boxes1, &boxes2);
        assert_eq!(diou_distance_result, arr2(&[[1.0]]));
    }
}
