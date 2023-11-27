use ndarray::{Array2, Zip};
use num_traits::{Num, ToPrimitive};

use crate::{boxes, utils};
/// Computes the Generalized Intersection over Union (GIOU) distance between two sets of bounding boxes.
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
/// use powerboxesrs::giou::giou_distance;
///
/// let boxes1 = array![[0., 0., 10., 10.], [20., 20., 30., 30.]];
/// let boxes2 = array![[0., 0., 10., 10.], [15., 15., 25., 25.], [20., 20., 30., 30.]];
///
/// let giou = giou_distance(&boxes1, &boxes2);
///
/// assert_eq!(giou.shape(), &[2, 3]);
/// assert_eq!(giou, array![[0., 1.6800000000000002, 1.7777777777777777], [1.7777777777777777, 1.0793650793650793, 0.]]);
/// ```
pub fn giou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut giou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::box_areas(&boxes1);
    let areas_boxes2 = boxes::box_areas(&boxes2);

    for i in 0..num_boxes1 {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];

        let area1 = areas_boxes1[i];

        for j in 0..num_boxes2 {
            let a2 = boxes2.row(j);
            let a2_x1 = a2[0];
            let a2_y1 = a2[1];
            let a2_x2 = a2[2];
            let a2_y2 = a2[3];
            let area2 = areas_boxes2[j];

            let x1 = utils::max(a1_x1, a2_x1);
            let y1 = utils::max(a1_y1, a2_y1);
            let x2 = utils::min(a1_x2, a2_x2);
            let y2 = utils::min(a1_y2, a2_y2);
            let (iou, union) = if x2 < x1 || y2 < y1 {
                (0.0, area1 + area2)
            } else {
                let intersection = (x2 - x1) * (y2 - y1);
                let intersection = intersection.to_f64().unwrap();
                let intersection = utils::min(intersection, utils::min(area1, area2));
                let union = area1 + area2 - intersection + 1e-16;
                (intersection / union, union)
            };
            // Calculate the enclosing box (C) coordinates
            let c_x1 = utils::min(a1_x1, a2_x1);
            let c_y1 = utils::min(a1_y1, a2_y1);
            let c_x2 = utils::max(a1_x2, a2_x2);
            let c_y2 = utils::max(a1_y2, a2_y2);
            // Calculate the area of the enclosing box (C)
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            let c_area = c_area.to_f64().unwrap();
            let giou = iou - ((c_area - union) / c_area);
            giou_matrix[[i, j]] = 1.0 - giou;
        }
    }

    giou_matrix
}
/// Computes the parallelized version of the Generalized Intersection over Union (GIOU) distance between two sets of bounding boxes.
/// Usually better when a high number of bounding boxes is used.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape `(num_boxes1, 4)` representing the coordinates of the first set of bounding boxes.
/// * `boxes2` - A 2D array of shape `(num_boxes2, 4)` representing the coordinates of the second set of bounding boxes.
///
/// # Returns
///
/// A 2D array of shape `(num_boxes1, num_boxes2)` representing the GIOU distance between each pair of bounding boxes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxesrs::giou::parallel_giou_distance;
///
/// let boxes1 = array![[0., 0., 10., 10.], [20., 20., 30., 30.]];
/// let boxes2 = array![[0., 0., 10., 10.], [15., 15., 25., 25.], [20., 20., 30., 30.]];
///
/// let giou = parallel_giou_distance(&boxes1, &boxes2);
///
/// assert_eq!(giou.shape(), &[2, 3]);
/// assert_eq!(giou, array![[0., 1.6800000000000002, 1.7777777777777777], [1.7777777777777777, 1.0793650793650793, 0.]]);
/// ```
pub fn parallel_giou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Sync + Send,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut giou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::parallel_box_areas(&boxes1);
    let areas_boxes2 = boxes::parallel_box_areas(&boxes2);
    Zip::indexed(giou_matrix.rows_mut()).par_for_each(|i, mut row| {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = areas_boxes1[i];
        row.indexed_iter_mut()
            .zip(boxes2.rows())
            .for_each(|((j, d), box2)| {
                let a2_x1 = box2[0];
                let a2_y1 = box2[1];
                let a2_x2 = box2[2];
                let a2_y2 = box2[3];
                let area2 = areas_boxes2[j];

                let x1 = utils::max(a1_x1, a2_x1);
                let y1 = utils::max(a1_y1, a2_y1);
                let x2 = utils::min(a1_x2, a2_x2);
                let y2 = utils::min(a1_y2, a2_y2);
                let (iou, union) = if x2 < x1 || y2 < y1 {
                    (0.0, area1 + area2)
                } else {
                    let intersection = (x2 - x1) * (y2 - y1);
                    let intersection = intersection.to_f64().unwrap();
                    let intersection = utils::min(intersection, utils::min(area1, area2));
                    let union = area1 + area2 - intersection + 1e-16;
                    (intersection / union, union)
                };
                // Calculate the enclosing box (C) coordinates
                let c_x1 = utils::min(a1_x1, a2_x1);
                let c_y1 = utils::min(a1_y1, a2_y1);
                let c_x2 = utils::max(a1_x2, a2_x2);
                let c_y2 = utils::max(a1_y2, a2_y2);
                // Calculate the area of the enclosing box (C)
                let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
                let c_area = c_area.to_f64().unwrap();
                let giou = iou - ((c_area - union) / c_area);

                *d = 1.0 - giou;
            });
    });

    giou_matrix
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;

    #[test]
    fn test_giou() {
        let boxes1 = arr2(&[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]]);
        let boxes2 = arr2(&[[2.0, 2.0, 5.0, 5.0], [3.0, 3.0, 6.0, 6.0]]);

        let giou_matrix = giou_distance(&boxes1, &boxes2);
        let parallel_giou_matrix = parallel_giou_distance(&boxes1, &boxes2);
        assert_eq!(giou_matrix[[0, 0]], 1.2611764705882353);
        assert_eq!(giou_matrix[[0, 1]], 1.5);
        assert_eq!(giou_matrix[[1, 0]], 0.8392857142857143);
        assert_eq!(giou_matrix[[1, 1]], 1.2611764705882353);
        assert_eq!(giou_matrix, parallel_giou_matrix);
    }
}
