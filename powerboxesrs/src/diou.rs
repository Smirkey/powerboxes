use crate::{
    utils, boxes,
};
use ndarray::{Array2};
use num_traits::{Num, ToPrimitive, Float, real::Real};

/// Calculates the intersection over union (IoU) distance between two sets of bounding boxes.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape (N, 4) representing N bounding boxes in xyxy format.
/// * `boxes2` - A 2D array of shape (M, 4) representing M bounding boxes in xyxy format.
///
/// # Returns
///
/// A 2D array of shape (N, M) representing the IoU distance between each pair of bounding boxes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxesrs::iou::iou_distance;
///
/// let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
/// let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
/// let iou = iou_distance(&boxes1, &boxes2);
/// assert_eq!(iou, array![[0.8571428571428572, 1.],[1., 0.8571428571428572]]);
/// ```
pub fn iou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Float + Real,
{
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

            let d = Float::sqrt(Float::powf(center_box1[0] - center_box2[0], two) + Float::powf(center_box1[1] - center_box2[1], two));
            let c = Float::sqrt(Float::powf(x2 - x1, N::from(2).unwrap()) + Float::powf(y2 - y1, N::from(2).unwrap()));
            let diou_penalty = Float::powf(d, two) / Float::powf(c, two);
            diou_matrix[[i, j]] = iou - diou_penalty.to_f64().unwrap();
            
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
    }

}
