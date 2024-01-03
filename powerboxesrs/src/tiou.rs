use ndarray::Array2;
use num_traits::{Num, ToPrimitive};

use crate::{boxes, utils};
/// Computes the Tracking Intersection over Union (TIOU) distance between two sets of bounding boxes.
/// see https://arxiv.org/pdf/2310.05171.pdf
/// # Arguments
///
/// * `boxes1` - A 2D array of shape `(num_boxes1, 4)` representing the coordinates in xyxy format of the first set of bounding boxes.
/// * `boxes2` - A 2D array of shape `(num_boxes2, 4)` representing the coordinates in xyxy format of the second set of bounding boxes.
///
/// # Returns
///
/// A 2D array of shape `(num_boxes1, num_boxes2)` representing the GIOU distance between each pair of bounding boxes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxesrs::tiou::tiou_distance;
///
/// let boxes1 = array![[0., 0., 10., 10.], [20., 20., 30., 30.]];
/// let boxes2 = array![[0., 0., 10., 10.], [15., 15., 25., 25.], [20., 20., 30., 30.]];
///
/// let tiou = tiou_distance(&boxes1, &boxes2);
///
/// assert_eq!(tiou.shape(), &[2, 3]);
/// assert_eq!(tiou, array![[0., 0.84, 0.8888888888888888], [0.8888888888888888, 0.5555555555555556, 0.]]);
/// ```
pub fn tiou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut tiou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
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

            // Calculate the enclosing box (C) coordinates
            let c_x1 = utils::min(a1_x1, a2_x1);
            let c_y1 = utils::min(a1_y1, a2_y1);
            let c_x2 = utils::max(a1_x2, a2_x2);
            let c_y2 = utils::max(a1_y2, a2_y2);
            // Calculate the area of the enclosing box (C)
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            let c_area = c_area.to_f64().unwrap();
            tiou_matrix[[i, j]] =
                utils::max(utils::ONE - utils::min(area1 / c_area, area2 / c_area), 0.);
        }
    }

    tiou_matrix
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_giou() {
        let boxes1 = arr2(&[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]]);
        let boxes2 = arr2(&[[2.0, 2.0, 5.0, 5.0], [3.0, 3.0, 6.0, 6.0]]);

        let tiou_matrix = tiou_distance(&boxes1, &boxes2);
        assert_eq!(tiou_matrix, arr2(&[[0.64, 0.75], [0.4375, 0.64]]));
    }
}
