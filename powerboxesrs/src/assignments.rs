use crate::iou::{iou_distance_slice, parallel_iou_distance_slice};
#[cfg(feature = "ndarray")]
use ndarray::ArrayView2;
use num_traits::{Num, ToPrimitive};

const SCALE: f64 = 1_000_000_000_000_000.0;

/// Shortest Augmenting Path algorithm for the Linear Sum Assignment Problem.
///
/// Based on: Crouse, "On implementing 2D rectangular assignment algorithms",
/// https://ui.adsabs.harvard.edu/abs/2016ITAES..52.1679C/abstract
/// Given `J` jobs and `W` workers (`J <= W`), computes the minimum cost to assign each jobs
/// to distinct workers.
///
/// # Arguments
/// * `c` - A slice representing a `J x W` cost matrix where `c[j][w]` is the cost to
///   assign job `j` to worker `w`. The slice is a row-major representation of the cost matrix.
///
/// # Returns
/// A `Vec<T>` of length `J`, where entry `j` is the worker's index assigned to this job.
///
/// # Panics
/// Panics if `weights` is empty, rows have inconsistent lengths, or `J > W`.
///
/// # Examples
///
/// ```
/// use powerboxesrs::assignments::lsap;
/// let costs = vec![8_i64, 5, 9, 4, 2, 4, 7, 3, 8];
/// let assignments = lsap(&costs, 3, 3);
/// assert_eq!(assignments, vec![0, 2, 1]);
/// ```
pub fn lsap<T>(c: &[T], nrow: usize, ncol: usize) -> Vec<usize>
where
    T: Copy + PartialOrd + std::ops::Sub<Output = T> + std::ops::Add<Output = T>,
    T: num_traits::Bounded + num_traits::Zero,
{
    assert!(nrow <= ncol);

    let inf = T::max_value();

    let mut u = vec![T::zero(); nrow]; // row potentials
    let mut v = vec![T::zero(); ncol]; // col potentials
    let mut col4row = vec![usize::MAX; nrow]; // col assigned to each row
    let mut row4col = vec![usize::MAX; ncol]; // row assigned to each col

    for cur_row in 0..nrow {
        // Dijkstra-like shortest path from cur_row to any unassigned col
        let mut shortest_path_costs = vec![inf; ncol];
        let mut path = vec![usize::MAX; ncol];
        let mut visited = vec![false; ncol];

        let mut i = cur_row;
        let mut sink = usize::MAX;
        let mut min_val = T::zero();

        while sink == usize::MAX {
            let mut idx = usize::MAX;
            let mut lowest = inf;

            for j in 0..ncol {
                if !visited[j] {
                    let r = c[i * ncol + j] - u[i] - v[j] + min_val;
                    if r < shortest_path_costs[j] {
                        shortest_path_costs[j] = r;
                        path[j] = i;
                    }
                    if shortest_path_costs[j] < lowest
                        || (shortest_path_costs[j] == lowest && row4col[j] == usize::MAX)
                    {
                        lowest = shortest_path_costs[j];
                        idx = j;
                    }
                }
            }

            min_val = lowest;
            let j = idx;
            visited[j] = true;

            if row4col[j] == usize::MAX {
                sink = j;
            } else {
                i = row4col[j];
            }
        }

        // Update potentials along the path
        u[cur_row] = u[cur_row] + min_val;
        for j in 0..ncol {
            if visited[j] {
                let r = row4col[j];
                if r != usize::MAX {
                    u[r] = u[r] - shortest_path_costs[j] + min_val;
                }
                v[j] = v[j] + shortest_path_costs[j] - min_val;
            }
        }

        // Augment along the path back to cur_row
        let mut j = sink;
        loop {
            let i = path[j];
            row4col[j] = i;
            let prev_j = col4row[i];
            col4row[i] = j;
            if i == cur_row {
                break;
            }
            j = prev_j;
        }
    }

    col4row
}

/// Compute the optimal assignment between two sets of axis-aligned bounding boxes
/// using the LSAP algorithm, minimising the total IoU distance.
///
/// Given `n1` ground-truth boxes and `n2` predicted boxes the function builds the
/// `min(n1,n2) × max(n1,n2)` cost matrix from `iou_distance_slice`, scales the
/// `f64` costs to `i64` (multiplied by `1e15`), and calls `lsap`.
/// After matching, pairs whose IoU is strictly below `iou_threshold` are discarded.
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
/// A pair `(indices1, indices2)` of equal-length `Vec<usize>` such that
/// `boxes1[indices1[k]]` is matched to `boxes2[indices2[k]]`.
/// The length of both vectors is at most `min(n1, n2)`.
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
    let iou_func = if n1 * n2 > 90_000 {
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
    // lsap function needs a numeric type that implements the `Ord` trait
    let costs_flat: Vec<i64> = iou_dist.iter().map(|&d| (d * SCALE) as i64).collect();

    let assignments = lsap(&costs_flat, nrows, ncols);

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
        .filter(|&(i, j)| iou_dist[i * ncols + j] <= max_dist)
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
/// A pair `(indices1, indices2)` of equal-length `Vec<usize>` of length at most `min(N, M)`.
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
            0.0_f64, 0.0, 1.0, 1.0, // match with gt 0
        ];
        let pairs = lsap_iou_slice(&gt, &pred, 3, 2, 0.0);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs, vec![(1, 0), (0, 1)]);
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
