use crate::{boxes, utils};
use ndarray::{Array2, Zip};
use num_traits::{Num, ToPrimitive};

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
/// assert_eq!(iou, array![[0.6086956521739131, 0.967741935483871],[0.967741935483871, 0.6086956521739131]]);
/// ```
pub fn iou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<N>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<N>::zeros((num_boxes1, num_boxes2));
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
            let intersection = (x2 - x1 + N::one()) * (y2 - y1 + N::one());
            let iou = intersection / (area1 + area2 - intersection);

            iou_matrix[[i, j]] = N::one() - iou;
        }
    }

    iou_matrix
}
/// Calculates the intersection over union (IoU) distance between two sets of bounding boxes.
/// This function uses rayon to parallelize the computation, which can be faster than the
/// non-parallelized version for large numbers of boxes.
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
/// assert_eq!(iou, array![[0.6086956521739131, 0.967741935483871],[0.967741935483871, 0.6086956521739131]]);
/// ```
pub fn parallel_iou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<N>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Clone + Sync + Send,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<N>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::box_areas(&boxes1);
    let areas_boxes2 = boxes::box_areas(&boxes2);
    Zip::indexed(iou_matrix.rows_mut()).par_for_each(|i, mut row| {
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
                if x2 < x1 || y2 < y1 {
                    *d = N::zero();
                }
                let intersection = (x2 - x1 + N::one()) * (y2 - y1 + N::one());

                let iou = intersection / (area1 + area2 - intersection);

                *d = N::one() - iou;
            });
    });

    return iou_matrix;
}

#[test]
fn test_iou_distance() {
    use ndarray::arr2;
    // Test case 1
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
    let iou_result = iou_distance(&boxes1, &boxes2);
    let parallel_iou_result = parallel_iou_distance(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.7142857142857143]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.7142857142857143]]));

    // Test case 2
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
    let iou_result = iou_distance(&boxes1, &boxes2);
    let parallel_iou_result = parallel_iou_distance(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[1.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[1.0]]));

    // Test case 3
    let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
    let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
    let iou_result = iou_distance(&boxes1, &boxes2);
    let parallel_iou_result = parallel_iou_distance(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.75]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.75]]));

    // Test case 4
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let iou_result = iou_distance(&boxes1, &boxes2);
    let parallel_iou_result = parallel_iou_distance(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.0]]));

    // Test case 5
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
    let iou_result = iou_distance(&boxes1, &boxes2);
    let parallel_iou_result = parallel_iou_distance(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[1.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[1.0]]));
}
