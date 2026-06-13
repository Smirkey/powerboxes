use crate::iou::{iou_distance_slice, parallel_iou_distance_slice};
#[cfg(feature = "ndarray")]
use ndarray::ArrayView2;
use num_traits::{Num, ToPrimitive};
use perfect_matching::sapjv::{lsap_scalar, lsap_simd};

// Cost matrix size above which we use parallel_iou_distance_slice
const PARALLEL_IOU_MIN_BOXES: usize = 90_000;
// If more than 98% boxes dont overlap, lsap_simd is slower than lsap
const SIMD_MAX_SPARSITY: f64 = 0.98;

/// Compute the optimal assignment between two sets of axis-aligned bounding boxes
/// using the LSAP algorithm, minimising the total IoU distance.
///
/// Builds the `min(n1,n2) × max(n1,n2)` cost matrix from `iou_distance_slice`,
/// casts the `f64` costs to `f32`, then dispatches to `lsap_simd` or `lsap_scalar`
/// depending on cost-matrix sparsity. After matching, pairs whose IoU is below
/// `iou_threshold` are discarded.
///
/// # Arguments
///
/// * `boxes1`        - Flat slice of length `n1 * 4` (xyxy, row-major).
/// * `boxes2`        - Flat slice of length `n2 * 4` (xyxy, row-major).
/// * `n1`            - Number of boxes in the first set.
/// * `n2`            - Number of boxes in the second set.
/// * `iou_threshold` - Minimum IoU required to keep a match. Use `0.0` to keep all.
///
/// # Returns
///
/// A `Vec<(usize, usize)>` of matched index pairs `(i, j)` such that
/// `boxes1[i]` is matched to `boxes2[j]`. Length is at most `min(n1, n2)`.
pub fn lsap_iou_slice<N>(
    boxes1: &[N],
    boxes2: &[N],
    n1: usize,
    n2: usize,
    iou_threshold: f64,
) -> Vec<(usize, usize)>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Sync,
{
    if n1 == 0 || n2 == 0 {
        return vec![];
    }

    // benchmark showed that parallel iou distance can be faster above 300 x 300 boxes
    let iou_func = if n1 * n2 > PARALLEL_IOU_MIN_BOXES {
        parallel_iou_distance_slice
    } else {
        iou_distance_slice
    };

    // lsap requires rows <= columns; transpose when n1 > n2.
    let transposed = n1 > n2;
    let (nrows, ncols) = if transposed { (n2, n1) } else { (n1, n2) };

    // Build the cost matrix
    let iou_dist = if !transposed {
        iou_func(boxes1, boxes2, nrows, ncols)
    } else {
        iou_func(boxes2, boxes1, nrows, ncols)
    };

    let costs_flat: Vec<f32> = iou_dist.iter().map(|&d| d as f32).collect();

    // check matrix "sparsity": ie many boxes dont overlap (iou is 0, distance is 1)
    let non_overlapping_ratio = costs_flat
        .iter()
        .filter_map(|x| if *x < 1_f32 { None } else { Some(1_f64) })
        .sum::<f64>()
        / (costs_flat.len() as f64);

    // if cost matrix is too sparse, the overhead of calling InnerScan in the SIMD
    // implementation dominates so it's much faster to use the scalar implementation.
    let lsap_func = if non_overlapping_ratio > SIMD_MAX_SPARSITY {
        lsap_scalar
    } else {
        lsap_simd
    };

    let assignments = lsap_func(&costs_flat, nrows, ncols);

    let (raw_idx1, raw_idx2) = if transposed {
        (assignments, (0..nrows).collect())
    } else {
        ((0..nrows).collect(), assignments)
    };

    // Discard pairs whose IoU falls below the threshold.
    let max_dist = 1.0 - iou_threshold;
    raw_idx1
        .into_iter()
        .zip(raw_idx2)
        .filter(|&(i, j)| {
            iou_dist[if transposed {
                j * ncols + i
            } else {
                i * ncols + j
            }] <= max_dist
        })
        .collect()
}

/// Compute the optimal assignment between two sets of axis-aligned bounding boxes
/// using the LSAP algorithm, minimising the total IoU distance.
///
/// Wraps [`lsap_iou_slice`] for `ndarray` inputs.
///
/// # Arguments
///
/// * `boxes1`        - Array of shape `(N, 4)` in xyxy format.
/// * `boxes2`        - Array of shape `(M, 4)` in xyxy format.
/// * `iou_threshold` - Minimum IoU required to keep a match. Use `0.0` to keep all.
///
/// # Returns
///
/// A `Vec<(usize, usize)>` of matched index pairs of length at most `min(N, M)`.
#[cfg(feature = "ndarray")]
pub fn lsap_iou<'a, N, BA>(boxes1: BA, boxes2: BA, iou_threshold: f64) -> Vec<(usize, usize)>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Sync + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    lsap_iou_slice(s1, s2, n1, n2, iou_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    // tests for lsap_iou_slice
    #[test]
    fn test_identical_boxes() {
        let boxes = vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let pairs = lsap_iou_slice(&boxes, &boxes, 2, 2, 0.0);
        assert_eq!(pairs.len(), 2);
        for pair in pairs.iter() {
            assert_eq!(pair.0, pair.1);
        }
    }

    #[test]
    fn test_more_gt_than_pred() {
        let gt = vec![
            0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        ];
        let pred = vec![
            2.0, 2.0, 3.0, 3.0, // match with gt 1
            4.0_f64, 4.0, 5.0, 5.0, // match with gt 2
        ];
        let pairs = lsap_iou_slice(&gt, &pred, 3, 2, 0.0);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs, vec![(1, 0), (2, 1)]);
    }

    #[test]
    fn test_more_pred_than_gt() {
        let gt = vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let pred = vec![
            0.0_f64, 0.0, 1.0, 1.0, // match with gt 0
            2.0, 2.0, 3.0, 3.0, // match with gt 1
            4.0, 4.0, 5.0, 5.0, // no match
        ];
        let pairs = lsap_iou_slice(&gt, &pred, 2, 3, 0.0);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn test_multiple_overlap() {
        let gt = vec![
            0.0_f64, 0.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        let pred = vec![
            1.0_f64, 1.0, 3.0, 3.0, // match with gt 1
            1.0, 1.0, 4.0, 4.0, // match with gt 0
            2.5, 2.5, 4.0, 4.0, // match with gt 2
        ];
        let pairs = lsap_iou_slice(&gt, &pred, 3, 3, 0.0);
        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs, vec![(0, 1), (1, 0), (2, 2)]);
    }

    #[test]
    fn test_empty_inputs() {
        let boxes: Vec<f64> = vec![];
        let pairs = lsap_iou_slice::<f64>(&boxes, &boxes, 0, 0, 0.0);
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_single_pair() {
        let b1 = vec![0.0_f64, 0.0, 2.0, 2.0];
        let b2 = vec![1.0_f64, 1.0, 3.0, 3.0];
        let pairs = lsap_iou_slice(&b1, &b2, 1, 1, 0.0);
        assert_eq!(pairs, vec![(0, 0)]);
    }

    #[test]
    fn test_optimal_over_greedy() {
        // Arrange boxes so that the greedy (nearest-first) choice is suboptimal.
        //
        //  gt0 = [0,0,2,2],  gt1 = [3,3,5,5]
        //  p0  = [3,3,5,5],  p1  = [0,0,2,2]
        //
        // Optimal: gt0→p1, gt1→p0 (total cost 0). Greedy starting at gt0 might pick p0.
        let gt = vec![0.0_f64, 0.0, 2.0, 2.0, 3.0, 3.0, 5.0, 5.0];
        let pred = vec![3.0_f64, 3.0, 5.0, 5.0, 0.0, 0.0, 2.0, 2.0];
        let pairs = lsap_iou_slice(&gt, &pred, 2, 2, 0.0);
        // gt0 should match pred1, gt1 should match pred0.
        assert_eq!(pairs, vec![(0, 1), (1, 0)]);
    }

    #[test]
    fn test_parallel_iou_branch() {
        // n1 * n2 = 90_601 > PARALLEL_IOU_MIN_BOXES (90_000), so this exercises
        // the parallel_iou_distance_slice arm. Identity boxes keep the assignment
        // trivial: pair i with i for all i.
        let n = 301;
        let mut boxes = Vec::with_capacity(n * 4);
        for i in 0..n {
            let f = i as f64;
            boxes.extend_from_slice(&[f, f, f + 1.0, f + 1.0]);
        }
        let pairs = lsap_iou_slice(&boxes, &boxes, n, n, 0.0);
        assert_eq!(pairs.len(), n);
        for pair in pairs.iter() {
            assert_eq!(pair.0, pair.1);
        }
    }

    #[test]
    fn test_no_overlap_no_match() {
        let gt = vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let pred = vec![10.0_f64, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0];
        // With a non-zero IoU threshold, every candidate match should be discarded.
        let pairs = lsap_iou_slice(&gt, &pred, 2, 2, 0.1);
        assert!(
            pairs.is_empty(),
            "expected no matches when IoU is 0 for all pairs, got {:?}",
            pairs
        );
    }

    // test for ndarray lsap_iou
    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use super::*;
        use ndarray::arr2;

        #[test]
        fn test_ndarray_wrapper() {
            let boxes1 = arr2(&[[0.0_f64, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]);
            let boxes2 = arr2(&[[0.0_f64, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]);
            let pairs = lsap_iou(&boxes1, &boxes2, 0.0);
            assert_eq!(pairs, vec![(0, 0), (1, 1)]);
        }
    }
}
