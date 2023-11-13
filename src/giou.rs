use ndarray::{Array2, Zip};

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
/// assert_eq!(giou[[0, 0]], 0.);
/// assert_eq!(giou[[0, 1]], 1.5948840131957898);
/// assert_eq!(giou[[0, 2]], 1.3293605909992827);
/// assert_eq!(giou[[1, 0]], 1.3293605909992827);
/// assert_eq!(giou[[1, 1]], 1.020555218446602);
/// assert_eq!(giou[[1, 2]], 0.);
/// ```
pub fn giou_distance(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut giou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));

    for i in 0..num_boxes1 {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];

        let area1 = (a1_x2 - a1_x1 + 1.) * (a1_y2 - a1_y1 + 1.);

        for j in 0..num_boxes2 {
            let a2 = boxes2.row(j);
            let a2_x1 = a2[0];
            let a2_y1 = a2[1];
            let a2_x2 = a2[2];
            let a2_y2 = a2[3];
            let area2 = (a2_x2 - a2_x1 + 1.) * (a2_y2 - a2_y1 + 1.);

            let x1 = f64::max(a1_x1, a2_x1);
            let y1 = f64::max(a1_y1, a2_y1);
            let x2 = f64::min(a1_x2, a2_x2);
            let y2 = f64::min(a1_y2, a2_y2);

            let intersection = (x2 - x1 + 1.) * (y2 - y1 + 1.);
            let union = area1 + area2 - intersection;
            let iou = intersection / union;

            // Calculate the enclosing box (C) coordinates
            let c_x1 = f64::min(a1_x1, a2_x1);
            let c_y1 = f64::min(a1_y1, a2_y1);
            let c_x2 = f64::max(a1_x2, a2_x2);
            let c_y2 = f64::max(a1_y2, a2_y2);

            // Calculate the area of the enclosing box (C)
            let c_area = (c_x2 - c_x1 + 1.) * (c_y2 - c_y1 + 1.);

            let giou = iou - ((c_area - union) / c_area);

            giou_matrix[[i, j]] = 1. - giou;
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
/// assert_eq!(giou[[0, 0]], 0.);
/// assert_eq!(giou[[0, 1]], 1.5948840131957898);
/// assert_eq!(giou[[0, 2]], 1.3293605909992827);
/// assert_eq!(giou[[1, 0]], 1.3293605909992827);
/// assert_eq!(giou[[1, 1]], 1.020555218446602);
/// assert_eq!(giou[[1, 2]], 0.);
/// ```
pub fn parallel_giou_distance(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut giou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));

    Zip::indexed(giou_matrix.rows_mut()).par_for_each(|i, mut row| {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = (a1_x2 - a1_x1 + 1.) * (a1_y2 - a1_y1 + 1.);
        row.iter_mut().zip(boxes2.rows()).for_each(|(d, box2)| {
            let a2_x1 = box2[0];
            let a2_y1 = box2[1];
            let a2_x2 = box2[2];
            let a2_y2 = box2[3];
            let area2 = (a2_x2 - a2_x1 + 1.) * (a2_y2 - a2_y1 + 1.);

            let x1 = f64::max(a1_x1, a2_x1);
            let y1 = f64::max(a1_y1, a2_y1);
            let x2 = f64::min(a1_x2, a2_x2);
            let y2 = f64::min(a1_y2, a2_y2);

            let intersection = (x2 - x1 + 1.) * (y2 - y1 + 1.);
            let union = area1 + area2 - intersection;
            let iou = intersection / union;

            // Calculate the enclosing box (C) coordinates
            let c_x1 = f64::min(a1_x1, a2_x1);
            let c_y1 = f64::min(a1_y1, a2_y1);
            let c_x2 = f64::max(a1_x2, a2_x2);
            let c_y2 = f64::max(a1_y2, a2_y2);

            // Calculate the area of the enclosing box (C)
            let c_area = (c_x2 - c_x1 + 1.) * (c_y2 - c_y1 + 1.);

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
        assert_eq!(giou_matrix[[0, 0]], 1.0793650793650793);
        assert_eq!(giou_matrix[[0, 1]], 1.3350888742593812);
        assert_eq!(giou_matrix[[1, 0]], 0.688695652173913);
        assert_eq!(giou_matrix[[1, 1]], 1.0793650793650793);
        assert_eq!(giou_matrix, parallel_giou_matrix);
    }
}
