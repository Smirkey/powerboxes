use crate::{
    boxes,
    rotation::{intersection_area, minimal_bounding_rect, Rect},
    utils,
};
#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2, Zip};
use num_traits::{Num, ToPrimitive};
use rstar::RTree;

// ─── Slice-based core ───

/// Calculates the IoU distance between two sets of bounding boxes (slice-based).
///
/// `boxes1` is a flat slice of length `n1 * 4` (row-major xyxy format).
/// `boxes2` is a flat slice of length `n2 * 4`.
/// Returns a flat `Vec<f64>` of length `n1 * n2` (row-major).
pub fn iou_distance_slice<N>(boxes1: &[N], boxes2: &[N], n1: usize, n2: usize) -> Vec<f64>
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
            if x2 < x1 || y2 < y1 {
                result[i * n2 + j] = utils::ONE;
                continue;
            }
            let intersection = (x2 - x1) * (y2 - y1);
            let intersection = intersection.to_f64().unwrap();
            let intersection = utils::min(intersection, utils::min(area1, area2));
            result[i * n2 + j] = utils::ONE - (intersection / (area1 + area2 - intersection));
        }
    }

    result
}

/// Calculates rotated IoU distance (slice-based).
/// `boxes1` is a flat slice of length `n1 * 5` (cxcywha format).
/// `boxes2` is a flat slice of length `n2 * 5`.
/// Returns a flat `Vec<f64>` of length `n1 * n2` (row-major).
pub fn rotated_iou_distance_slice(boxes1: &[f64], boxes2: &[f64], n1: usize, n2: usize) -> Vec<f64> {
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
            utils::Bbox {
                index: idx,
                x1: min_x,
                y1: min_y,
                x2: max_x,
                y2: max_y,
            }
        })
        .collect();
    let boxes2_bounding_rects: Vec<utils::Bbox<f64>> = boxes2_rects
        .iter()
        .enumerate()
        .map(|(idx, rect)| {
            let (min_x, min_y, max_x, max_y) = minimal_bounding_rect(&rect.points());
            utils::Bbox {
                index: idx,
                x1: min_x,
                y1: min_y,
                x2: max_x,
                y2: max_y,
            }
        })
        .collect();

    let box1_rtree: RTree<utils::Bbox<f64>> = RTree::bulk_load(boxes1_bounding_rects);
    let box2_rtree: RTree<utils::Bbox<f64>> = RTree::bulk_load(boxes2_bounding_rects);

    for (box1, box2) in box1_rtree.intersection_candidates_with_other_tree(&box2_rtree) {
        let area1 = areas1[box1.index];
        let area2 = areas2[box2.index];
        let intersection = intersection_area(&boxes1_rects[box1.index], &boxes2_rects[box2.index]);
        let union = area1 + area2 - intersection;
        result[box1.index * n2 + box2.index] = utils::ONE - intersection / union;
    }

    result
}

// ─── ndarray wrappers ───

#[cfg(feature = "ndarray")]
pub fn iou_distance<'a, N, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
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
    let result = iou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(feature = "ndarray")]
pub fn parallel_iou_distance<'a, N, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Send + Sync + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes1 = boxes1.into();
    let boxes2 = boxes2.into();
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
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
                    *d = utils::ONE;
                } else {
                    let intersection = (x2 - x1) * (y2 - y1);
                    let intersection = intersection.to_f64().unwrap();
                    let intersection = utils::min(intersection, utils::min(area1, area2));
                    *d = 1. - (intersection / (area1 + area2 - intersection));
                }
            });
    });

    iou_matrix
}

#[cfg(feature = "ndarray")]
pub fn rotated_iou_distance<'a, BA>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = rotated_iou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_distance_slice() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![1.0, 1.0, 3.0, 3.0];
        let result = iou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert_eq!(result, vec![0.8571428571428572]);
    }

    #[test]
    fn test_iou_distance_slice_no_overlap() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![3.0, 3.0, 4.0, 4.0];
        let result = iou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn test_iou_distance_slice_perfect() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![0.0, 0.0, 2.0, 2.0];
        let result = iou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert_eq!(result, vec![0.0]);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use ndarray::arr2;
        use super::*;

        #[test]
        fn test_iou_distance() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
            let iou_distance_result = iou_distance(&boxes1, &boxes2);
            let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
            assert_eq!(iou_distance_result, arr2(&[[0.8571428571428572]]));
            assert_eq!(parallel_iou_distance_result, arr2(&[[0.8571428571428572]]));
        }

        #[test]
        fn test_iou_distance2() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
            let iou_distance_result = iou_distance(&boxes1, &boxes2);
            let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
            assert_eq!(iou_distance_result, arr2(&[[1.0]]));
            assert_eq!(parallel_iou_distance_result, arr2(&[[1.0]]));
        }

        #[test]
        fn test_iou_distance3() {
            let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
            let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
            let iou_distance_result = iou_distance(&boxes1, &boxes2);
            let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
            assert_eq!(iou_distance_result, arr2(&[[0.9375]]));
            assert_eq!(parallel_iou_distance_result, arr2(&[[0.9375]]));
        }

        #[test]
        fn test_iou_distance4() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let iou_distance_result = iou_distance(&boxes1, &boxes2);
            let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
            assert_eq!(iou_distance_result, arr2(&[[0.0]]));
            assert_eq!(parallel_iou_distance_result, arr2(&[[0.0]]));
        }

        #[test]
        fn test_rotated_iou_distance() {
            let boxes1 = arr2(&[[5.0, 5.0, 2.0, 2.0, 0.0]]);
            let boxes2 = arr2(&[[4.0, 4.0, 2.0, 2.0, 0.0]]);
            let rotated_iou_distance_result = rotated_iou_distance(&boxes1, &boxes2);
            assert_eq!(rotated_iou_distance_result, arr2(&[[0.8571428571428572]]));
        }
    }
}
