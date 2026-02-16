use rayon::prelude::*;

#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};
use num_traits::{Num, ToPrimitive};

use crate::{
    boxes,
    rotation::{minimal_bounding_rect, Rect},
    utils,
};

// ─── Slice-based core ───

/// Computes the Tracking Intersection over Union (TIoU) distance between two sets of bounding boxes.
/// See <https://arxiv.org/pdf/2310.05171.pdf>.
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
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the TIoU distance
/// between each pair of bounding boxes.
pub fn tiou_distance_slice<N>(boxes1: &[N], boxes2: &[N], n1: usize, n2: usize) -> Vec<f64>
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
            let c_x1 = utils::min(a1_x1, a2_x1);
            let c_y1 = utils::min(a1_y1, a2_y1);
            let c_x2 = utils::max(a1_x2, a2_x2);
            let c_y2 = utils::max(a1_y2, a2_y2);
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            let c_area = c_area.to_f64().unwrap();
            result[i * n2 + j] = utils::ONE - utils::min(area1 / c_area, area2 / c_area);
        }
    }

    result
}

/// Calculates the rotated Tracking IoU (TIoU) distance between two sets of rotated bounding boxes.
///
/// Given two sets of rotated bounding boxes, this function computes the rotated TIoU distance
/// matrix between them. The rotated TIoU distance is a measure of dissimilarity between two
/// rotated bounding boxes, taking into account both their overlap and the encompassing area.
///
/// # Arguments
///
/// * `boxes1` - A flat slice of length `n1 * 5`. Each box contains
///   parameters [center_x, center_y, width, height, angle in degrees].
/// * `boxes2` - A flat slice of length `n2 * 5`. Each box contains
///   parameters [center_x, center_y, width, height, angle in degrees].
/// * `n1` - The number of boxes in the first set.
/// * `n2` - The number of boxes in the second set.
///
/// # Returns
///
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the rotated TIoU distance matrix.
pub fn rotated_tiou_distance_slice(boxes1: &[f64], boxes2: &[f64], n1: usize, n2: usize) -> Vec<f64> {
    let mut result = vec![utils::ONE; n1 * n2];
    let areas1 = boxes::rotated_box_areas_slice(boxes1, n1);
    let areas2 = boxes::rotated_box_areas_slice(boxes2, n2);

    let boxes1_rects: Vec<(f64, f64, f64, f64)> = (0..n1)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes1, i);
            minimal_bounding_rect(&Rect::new(cx, cy, w, h, a).points())
        })
        .collect();
    let boxes2_rects: Vec<(f64, f64, f64, f64)> = (0..n2)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes2, i);
            minimal_bounding_rect(&Rect::new(cx, cy, w, h, a).points())
        })
        .collect();

    for i in 0..n1 {
        let area1 = areas1[i];
        let (x1_r1, y1_r1, x2_r1, y2_r1) = boxes1_rects[i];

        for j in 0..n2 {
            let area2 = areas2[j];
            let (x1_r2, y1_r2, x2_r2, y2_r2) = boxes2_rects[j];

            let c_x1 = utils::min(x1_r1, x1_r2);
            let c_y1 = utils::min(y1_r1, y1_r2);
            let c_x2 = utils::max(x2_r1, x2_r2);
            let c_y2 = utils::max(y2_r1, y2_r2);
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            let c_area = c_area.to_f64().unwrap();
            result[i * n2 + j] = utils::ONE - utils::min(area1 / c_area, area2 / c_area);
        }
    }

    result
}

/// Calculates the rotated TIoU distance in parallel using Rayon.
///
/// Each row of the result matrix is computed in a separate thread.
///
/// # Arguments
///
/// * `boxes1` - A flat slice of length `n1 * 5`. Each box: (cx, cy, w, h, angle).
/// * `boxes2` - A flat slice of length `n2 * 5`. Each box: (cx, cy, w, h, angle).
/// * `n1` - The number of boxes in the first set.
/// * `n2` - The number of boxes in the second set.
///
/// # Returns
///
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the rotated TIoU distance matrix.
pub fn parallel_rotated_tiou_distance_slice(
    boxes1: &[f64],
    boxes2: &[f64],
    n1: usize,
    n2: usize,
) -> Vec<f64> {
    let mut result = vec![utils::ONE; n1 * n2];
    let areas1 = boxes::rotated_box_areas_slice(boxes1, n1);
    let areas2 = boxes::rotated_box_areas_slice(boxes2, n2);

    let boxes1_rects: Vec<(f64, f64, f64, f64)> = (0..n1)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes1, i);
            minimal_bounding_rect(&Rect::new(cx, cy, w, h, a).points())
        })
        .collect();
    let boxes2_rects: Vec<(f64, f64, f64, f64)> = (0..n2)
        .map(|i| {
            let (cx, cy, w, h, a) = utils::row5(boxes2, i);
            minimal_bounding_rect(&Rect::new(cx, cy, w, h, a).points())
        })
        .collect();

    result.par_chunks_mut(n2).enumerate().for_each(|(i, row)| {
        let area1 = areas1[i];
        let (x1_r1, y1_r1, x2_r1, y2_r1) = boxes1_rects[i];

        for j in 0..n2 {
            let area2 = areas2[j];
            let (x1_r2, y1_r2, x2_r2, y2_r2) = boxes2_rects[j];

            let c_x1 = utils::min(x1_r1, x1_r2);
            let c_y1 = utils::min(y1_r1, y1_r2);
            let c_x2 = utils::max(x2_r1, x2_r2);
            let c_y2 = utils::max(y2_r1, y2_r2);
            let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);
            row[j] = utils::ONE - utils::min(area1 / c_area, area2 / c_area);
        }
    });

    result
}

// ─── ndarray wrappers ───

/// Computes the Tracking Intersection over Union (TIoU) distance between two sets of bounding boxes.
/// See <https://arxiv.org/pdf/2310.05171.pdf>.
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
/// A 2D array of shape `(num_boxes1, num_boxes2)` representing the TIoU distance between each pair
/// of bounding boxes.
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
#[cfg(feature = "ndarray")]
pub fn tiou_distance<'a, N, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
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
    let result = tiou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

/// Calculates the rotated Tracking IoU (TIoU) distance between two sets of rotated bounding boxes.
///
/// Given two sets of rotated bounding boxes, this function computes the rotated TIoU distance
/// matrix between them. The rotated TIoU distance is a measure of dissimilarity between two
/// rotated bounding boxes, taking into account both their overlap and the encompassing area.
///
/// # Arguments
///
/// * `boxes1` - A 2D array containing the parameters of the first set of rotated bounding boxes.
///   Each row represents a rotated bounding box with parameters [center_x, center_y, width, height, angle in degrees].
/// * `boxes2` - A 2D array containing the parameters of the second set of rotated bounding boxes.
///   Each row represents a rotated bounding box with parameters [center_x, center_y, width, height, angle in degrees].
///
/// # Returns
///
/// A 2D array representing the rotated TIoU distance matrix. The element at position (i, j)
/// represents the rotated TIoU distance between the i-th box in `boxes1` and the j-th box in `boxes2`.
#[cfg(feature = "ndarray")]
pub fn rotated_tiou_distance<'a, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = rotated_tiou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

/// Calculates the rotated TIoU distance matrix in parallel using Rayon.
///
/// # Arguments
///
/// * `boxes1` - A 2D array with rows (center_x, center_y, width, height, angle).
/// * `boxes2` - A 2D array with rows (center_x, center_y, width, height, angle).
///
/// # Returns
///
/// A 2D array representing the rotated TIoU distance matrix.
#[cfg(feature = "ndarray")]
pub fn parallel_rotated_tiou_distance<'a, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = parallel_rotated_tiou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiou_slice() {
        let boxes1 = vec![0.0, 0.0, 3.0, 3.0, 1.0, 1.0, 4.0, 4.0];
        let boxes2 = vec![2.0, 2.0, 5.0, 5.0, 3.0, 3.0, 6.0, 6.0];
        let result = tiou_distance_slice(&boxes1, &boxes2, 2, 2);
        assert_eq!(result, vec![0.64, 0.75, 0.4375, 0.64]);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use ndarray::arr2;
        use super::*;

        #[test]
        fn test_tiou() {
            let boxes1 = arr2(&[[0.0, 0.0, 3.0, 3.0], [1.0, 1.0, 4.0, 4.0]]);
            let boxes2 = arr2(&[[2.0, 2.0, 5.0, 5.0], [3.0, 3.0, 6.0, 6.0]]);

            let tiou_matrix = tiou_distance(&boxes1, &boxes2);
            assert_eq!(tiou_matrix, arr2(&[[0.64, 0.75], [0.4375, 0.64]]));
        }

        #[test]
        fn test_rotated_tiou() {
            let boxes1 = arr2(&[[0.0, 0.0, 3.0, 3.0, 20.0], [1.0, 1.0, 4.0, 4.0, 19.0]]);
            let boxes2 = arr2(&[[2.0, 2.0, 5.0, 5.0, 0.0], [3.0, 3.0, 6.0, 6.0, 20.0]]);

            let tiou_matrix = rotated_tiou_distance(&boxes1, &boxes2);
            let parallel_result = parallel_rotated_tiou_distance(&boxes1, &boxes2);
            assert_eq!(
                tiou_matrix,
                arr2(&[
                    [0.7818149787949012, 0.8829233169330242],
                    [0.561738213456193, 0.7725560385451797]
                ])
            );
            assert_eq!(tiou_matrix, parallel_result);
        }
    }
}
