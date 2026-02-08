#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2, Zip};
#[cfg(feature = "ndarray")]
use num_traits::real::Real;
use num_traits::{Num, ToPrimitive};
use rstar::RTree;

use crate::{
    boxes,
    rotation::{intersection_area, minimal_bounding_rect, Rect},
    utils,
};

// ─── Slice-based core ───

/// Computes the Generalized Intersection over Union (GIoU) distance between two sets of bounding boxes.
///
/// # Arguments
///
/// * `boxes1` - A flat slice of length `n1 * 4` representing the coordinates in xyxy format
///   of the first set of bounding boxes (row-major).
/// * `boxes2` - A flat slice of length `n2 * 4` representing the coordinates in xyxy format
///   of the second set of bounding boxes (row-major).
/// * `n1` - The number of boxes in the first set.
/// * `n2` - The number of boxes in the second set.
///
/// # Returns
///
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the GIoU distance
/// between each pair of bounding boxes.
pub fn giou_distance_slice<N>(boxes1: &[N], boxes2: &[N], n1: usize, n2: usize) -> Vec<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let mut result = vec![0.0f64; n1 * n2];
    let areas1 = boxes::box_areas_slice(boxes1, n1);
    let areas2 = boxes::box_areas_slice(boxes2, n2);

    for i in 0..n1 {
        let (a1_x1, a1_y1, a1_x2, a1_y2) = utils::row4(boxes1, i);
        let area1 = areas1[i];

        for j in 0..n2 {
            let (a2_x1, a2_y1, a2_x2, a2_y2) = utils::row4(boxes2, j);
            let area2 = areas2[j];

            let x1 = utils::max(a1_x1, a2_x1);
            let y1 = utils::max(a1_y1, a2_y1);
            let x2 = utils::min(a1_x2, a2_x2);
            let y2 = utils::min(a1_y2, a2_y2);
            let (iou, union) = if x2 < x1 || y2 < y1 {
                (utils::ZERO, area1 + area2)
            } else {
                let intersection = (x2 - x1) * (y2 - y1);
                let intersection = intersection.to_f64().unwrap();
                let intersection = utils::min(intersection, utils::min(area1, area2));
                let union = area1 + area2 - intersection;
                (intersection / union, union)
            };
            let c_x1 = utils::min(a1_x1, a2_x1);
            let c_y1 = utils::min(a1_y1, a2_y1);
            let c_x2 = utils::max(a1_x2, a2_x2);
            let c_y2 = utils::max(a1_y2, a2_y2);
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            let c_area = c_area.to_f64().unwrap();
            let giou = iou - ((c_area - union) / c_area);
            result[i * n2 + j] = utils::ONE - giou;
        }
    }

    result
}

/// Calculates the rotated Generalized IoU (GIoU) distance between two sets of rotated bounding boxes.
///
/// Given two sets of rotated bounding boxes, this function computes the rotated GIoU distance
/// matrix between them. The rotated GIoU distance is a measure of dissimilarity between two
/// rotated bounding boxes, taking into account both their overlap and the encompassing area.
///
/// # Arguments
///
/// * `boxes1` - A flat slice of length `n1 * 5`. Each box contains
///   parameters [center_x, center_y, width, height, angle].
/// * `boxes2` - A flat slice of length `n2 * 5`. Each box contains
///   parameters [center_x, center_y, width, height, angle].
/// * `n1` - The number of boxes in the first set.
/// * `n2` - The number of boxes in the second set.
///
/// # Returns
///
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the rotated GIoU distance matrix.
pub fn rotated_giou_distance_slice(boxes1: &[f64], boxes2: &[f64], n1: usize, n2: usize) -> Vec<f64> {
    let mut result = vec![utils::ONE; n1 * n2];
    let areas1 = boxes::rotated_box_areas_slice(boxes1, n1);
    let areas2 = boxes::rotated_box_areas_slice(boxes2, n2);

    let boxes1_rects: Vec<Rect> = (0..n1)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes1, i);
            Rect::new(cx, cy, w, h, a)
        })
        .collect();
    let boxes2_rects: Vec<Rect> = (0..n2)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes2, i);
            Rect::new(cx, cy, w, h, a)
        })
        .collect();

    let boxes1_bounding_rects: Vec<utils::Bbox<f64>> = boxes1_rects
        .iter()
        .enumerate()
        .map(|(idx, rect)| {
            let (min_x, min_y, max_x, max_y) = minimal_bounding_rect(&rect.points());
            utils::Bbox { index: idx, x1: min_x, y1: min_y, x2: max_x, y2: max_y }
        })
        .collect();
    let boxes2_bounding_rects: Vec<utils::Bbox<f64>> = boxes2_rects
        .iter()
        .enumerate()
        .map(|(idx, rect)| {
            let (min_x, min_y, max_x, max_y) = minimal_bounding_rect(&rect.points());
            utils::Bbox { index: idx, x1: min_x, y1: min_y, x2: max_x, y2: max_y }
        })
        .collect();

    let box1_rtree: RTree<utils::Bbox<f64>> = RTree::bulk_load(boxes1_bounding_rects);
    let box2_rtree: RTree<utils::Bbox<f64>> = RTree::bulk_load(boxes2_bounding_rects);

    for (box1, box2) in box1_rtree.intersection_candidates_with_other_tree(&box2_rtree) {
        let area1 = areas1[box1.index];
        let area2 = areas2[box2.index];
        let rect1 = boxes1_rects[box1.index];
        let rect2 = boxes2_rects[box2.index];
        let intersection = intersection_area(&rect1, &rect2);
        let union = area1 + area2 - intersection;
        let c_x1 = utils::min(box1.x1, box2.x1);
        let c_y1 = utils::min(box1.y1, box2.y1);
        let c_x2 = utils::max(box1.x2, box2.x2);
        let c_y2 = utils::max(box1.y2, box2.y2);
        let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
        let c_area = c_area.to_f64().unwrap();
        result[box1.index * n2 + box2.index] = intersection / union - ((c_area - union) / c_area);
    }

    result
}

// ─── ndarray wrappers ───

/// Computes the Generalized Intersection over Union (GIoU) distance between two sets of bounding boxes.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape `(num_boxes1, 4)` representing the coordinates in xyxy format
///   of the first set of bounding boxes.
/// * `boxes2` - A 2D array of shape `(num_boxes2, 4)` representing the coordinates in xyxy format
///   of the second set of bounding boxes.
///
/// # Returns
///
/// A 2D array of shape `(num_boxes1, num_boxes2)` representing the GIoU distance between each pair
/// of bounding boxes.
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
#[cfg(feature = "ndarray")]
pub fn giou_distance<'a, N, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = giou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

/// Computes the parallelized version of the Generalized Intersection over Union (GIoU) distance
/// between two sets of bounding boxes. Usually better when a high number of bounding boxes is used.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape `(num_boxes1, 4)` representing the coordinates in xyxy format
///   of the first set of bounding boxes.
/// * `boxes2` - A 2D array of shape `(num_boxes2, 4)` representing the coordinates in xyxy format
///   of the second set of bounding boxes.
///
/// # Returns
///
/// A 2D array of shape `(num_boxes1, num_boxes2)` representing the GIoU distance between each pair
/// of bounding boxes.
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
#[cfg(feature = "ndarray")]
pub fn parallel_giou_distance<'a, N, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    N: Real + Sync + Send + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes1 = boxes1.into();
    let boxes2 = boxes2.into();
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut giou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::parallel_box_areas(boxes1);
    let areas_boxes2 = boxes::parallel_box_areas(boxes2);
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
                    (utils::ZERO, area1 + area2)
                } else {
                    let intersection = (x2 - x1) * (y2 - y1);
                    let intersection = intersection.to_f64().unwrap();
                    let intersection = utils::min(intersection, utils::min(area1, area2));
                    let union = area1 + area2 - intersection;
                    (intersection / union, union)
                };
                let c_x1 = utils::min(a1_x1, a2_x1);
                let c_y1 = utils::min(a1_y1, a2_y1);
                let c_x2 = utils::max(a1_x2, a2_x2);
                let c_y2 = utils::max(a1_y2, a2_y2);
                let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
                let c_area = c_area.to_f64().unwrap();
                let giou = iou - ((c_area - union) / c_area);

                *d = utils::ONE - giou;
            });
    });

    giou_matrix
}

/// Calculates the rotated Generalized IoU (GIoU) distance between two sets of rotated bounding boxes.
///
/// Given two sets of rotated bounding boxes represented by `boxes1` and `boxes2`, this function
/// computes the rotated GIoU distance matrix between them. The rotated GIoU distance is a measure
/// of dissimilarity between two rotated bounding boxes, taking into account both their overlap
/// and the encompassing area.
///
/// # Arguments
///
/// * `boxes1` - A 2D array containing the parameters of the first set of rotated bounding boxes.
///   Each row represents a rotated bounding box with parameters [center_x, center_y, width, height, angle].
/// * `boxes2` - A 2D array containing the parameters of the second set of rotated bounding boxes.
///   Each row represents a rotated bounding box with parameters [center_x, center_y, width, height, angle].
///
/// # Returns
///
/// A 2D array representing the rotated GIoU distance matrix. The element at position (i, j)
/// represents the rotated GIoU distance between the i-th box in `boxes1` and the j-th box in `boxes2`.
#[cfg(feature = "ndarray")]
pub fn rotated_giou_distance<'a, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = rotated_giou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_giou_slice() {
        let boxes1 = vec![0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 4.0, 4.0];
        let boxes2 = vec![2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0];
        let result = giou_distance_slice(&boxes1, &boxes2, 2, 2);
        assert_eq!(result[0], 1.2611764705882353);
        assert_eq!(result[1], 1.5);
        assert_eq!(result[2], 0.8392857142857143);
        assert_eq!(result[3], 1.2611764705882353);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
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

        #[test]
        fn test_rotated_giou() {
            let boxes1 = arr2(&[[5.0, 5.0, 2.0, 2.0, 0.0]]);
            let boxes2 = arr2(&[[4.0, 4.0, 2.0, 2.0, 0.0]]);
            let rotated_iou_distance_result = rotated_giou_distance(&boxes1, &boxes2);
            assert_eq!(rotated_iou_distance_result, arr2(&[[-0.07936507936507936]]));
        }
    }
}
