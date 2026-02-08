#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};
use num_traits::{Num, ToPrimitive};

use crate::{
    boxes,
    rotation::{minimal_bounding_rect, Rect},
    utils,
};

// ─── Slice-based core ───

/// Computes the TIoU distance between two sets of bounding boxes (slice-based).
///
/// `boxes1` is a flat slice of length `n1 * 4` (row-major xyxy format).
/// `boxes2` is a flat slice of length `n2 * 4`.
/// Returns a flat `Vec<f64>` of length `n1 * n2` (row-major).
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

/// Calculates rotated TIoU distance (slice-based).
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

// ─── ndarray wrappers ───

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
            assert_eq!(
                tiou_matrix,
                arr2(&[
                    [0.7818149787949012, 0.8829233169330242],
                    [0.561738213456193, 0.7725560385451797]
                ])
            );
        }
    }
}
