use crate::iou::{iou_distance_slice, parallel_iou_distance_slice};
use bytemuck::cast_slice;
#[cfg(feature = "ndarray")]
use ndarray::ArrayView2;
use num_traits::{Num, ToPrimitive};
use pulp::{Arch, Simd, WithSimd};

// Cost matrix size above which we use parallel_iou_distance_slice
const PARALLEL_IOU_MIN_BOXES: usize = 90_000;
// If more than 75% boxes dont overlap, lsap_simd is slower than lsap
const SIMD_MAX_SPARSITY: f32 = 0.75;


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

/// SIMD implementation for LSAP
struct InnerScan<'a> {
    c_row: &'a [f32],
    v: &'a [f32],
    visited: &'a [u8],
    spc: &'a mut [f32],
    path: &'a mut [u32],
    row4col: &'a [u32],
    u_i: f32,
    min_val: f32,
    row_i: u32,
}

struct ScanResult {
    pub best_cost: f32,
    pub best_col: usize,
}

impl WithSimd for InnerScan<'_> {
    type Output = ScanResult;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> ScanResult {
        let Self {
            c_row,
            v,
            visited,
            spc,
            path,
            row4col,
            u_i,
            min_val,
            row_i,
        } = self;
        let ncol = c_row.len();

        let offset = simd.splat_f32s(min_val - u_i);
        let inf_v = simd.splat_f32s(f32::INFINITY);
        let row_v = simd.splat_u32s(row_i);
        let zero_u = simd.splat_u32s(0u32);

        let mut best_cost_v = simd.splat_f32s(f32::INFINITY);
        let mut best_col_v = simd.splat_u32s(u32::MAX);

        let (c_chunks, c_tail) = S::as_simd_f32s(c_row);
        let (v_chunks, _) = S::as_simd_f32s(v);
        let (spc_chunks, _) = S::as_mut_simd_f32s(spc);
        let (path_chunks, _) = S::as_mut_simd_u32s(path);

        let lanes = std::mem::size_of::<S::f32s>() / 4;

        for (chunk_idx, (((c_v, v_v), spc_v), path_v)) in c_chunks
            .iter()
            .zip(v_chunks.iter())
            .zip(spc_chunks.iter_mut())
            .zip(path_chunks.iter_mut())
            .enumerate()
        {
            let base = chunk_idx * lanes;

            // Widen u8 visited flags → u32 lanes, then build mask where != 0
            let vis_u32: S::u32s = {
                let mut buf = zero_u;
                let buf_slice: &mut [u32] =
                    bytemuck::cast_slice_mut(std::slice::from_mut(&mut buf));
                for (dst, &src) in buf_slice.iter_mut().zip(visited[base..].iter()) {
                    *dst = src as u32;
                }
                buf
            };
            let is_visited = simd.greater_than_u32s(vis_u32, zero_u);

            // r = c[j] - v[j] + (min_val - u_i)
            let r = simd.add_f32s(simd.sub_f32s(*c_v, *v_v), offset);

            // spc[j] = min(spc[j], r)  for unvisited lanes only
            let old_spc_v = *spc_v;
            let new_spc = simd.min_f32s(*spc_v, r);
            *spc_v = simd.select_f32s_m32s(is_visited, *spc_v, new_spc);

            // path[j] = row_i  where r improved spc AND lane is unvisited
            let improved = simd.less_than_f32s(r, old_spc_v);
            let update_path = simd.and_m32s(simd.not_m32s(is_visited), improved);
            *path_v = simd.select_u32s_m32s(update_path, row_v, *path_v);

            // For argmin: mask visited lanes out with infinity
            let cost_for_min = simd.select_f32s_m32s(is_visited, inf_v, *spc_v);

            // Column indices for this chunk
            let col_indices: S::u32s = {
                let mut buf = zero_u;
                let buf_slice: &mut [u32] =
                    bytemuck::cast_slice_mut(std::slice::from_mut(&mut buf));
                for (k, dst) in buf_slice.iter_mut().enumerate() {
                    *dst = (base + k) as u32;
                }
                buf
            };

            // Update running per-lane argmin
            let new_is_better = simd.less_than_f32s(cost_for_min, best_cost_v);
            best_cost_v = simd.select_f32s_m32s(new_is_better, cost_for_min, best_cost_v);
            best_col_v = simd.select_u32s_m32s(new_is_better, col_indices, best_col_v);
        }

        // Horizontal reduction: fold SIMD lanes down to a scalar argmin
        let costs: &[f32] = cast_slice(std::slice::from_ref(&best_cost_v));
        let cols: &[u32] = cast_slice(std::slice::from_ref(&best_col_v));
        let mut best_cost = f32::INFINITY;
        let mut best_col = usize::MAX;
        for (&cost, &col) in costs.iter().zip(cols.iter()) {
            if cost < best_cost
                || (cost == best_cost && col != u32::MAX && row4col[col as usize] == u32::MAX)
            {
                best_cost = cost;
                best_col = col as usize;
            }
        }

        // Scalar tail: remainder columns that don't fill a full SIMD vector
        let tail_start = ncol - c_tail.len();
        for j in tail_start..ncol {
            if visited[j] == 0 {
                let r = c_row[j] - u_i - v[j] + min_val;
                if r < spc[j] {
                    spc[j] = r;
                    path[j] = row_i;
                }
                if spc[j] < best_cost || (spc[j] == best_cost && row4col[j] == u32::MAX) {
                    best_cost = spc[j];
                    best_col = j;
                }
            }
        }

        ScanResult {
            best_cost,
            best_col,
        }
    }
}

/// Shortest Augmenting Path algorithm with SIMD instructions for the Linear Sum Assignment Problem.
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
/// use powerboxesrs::assignments::lsap_simd;
/// let costs = vec![8_f32, 5., 9., 4., 2., 4., 7., 3., 8.];
/// let assignments = lsap_simd(&costs, 3, 3);
/// assert_eq!(assignments, vec![0, 2, 1]);
/// ```
pub fn lsap_simd(c: &[f32], nrow: usize, ncol: usize) -> Vec<usize> {
    assert!(nrow <= ncol);

    let arch = Arch::new();

    let mut u = vec![0f32; nrow];
    let mut v = vec![0f32; ncol];
    let mut col4row = vec![usize::MAX; nrow];
    let mut row4col = vec![u32::MAX; ncol];
    let mut visited = vec![0u8; ncol];

    for cur_row in 0..nrow {
        let mut spc = vec![f32::INFINITY; ncol];
        let mut path = vec![u32::MAX; ncol];
        visited.fill(0);

        let mut i = cur_row;
        let mut sink = usize::MAX;
        let mut min_val = 0f32;

        while sink == usize::MAX {
            let res = arch.dispatch(InnerScan {
                c_row: &c[i * ncol..(i + 1) * ncol],
                v: &v,
                visited: &visited,
                spc: &mut spc,
                path: &mut path,
                row4col: &row4col,
                u_i: u[i],
                min_val,
                row_i: i as u32,
            });

            min_val = res.best_cost;
            let j = res.best_col;
            visited[j] = 1;

            if row4col[j] == u32::MAX {
                sink = j;
            } else {
                i = row4col[j] as usize;
            }
        }

        u[cur_row] += min_val;
        for j in 0..ncol {
            if visited[j] != 0 {
                let r = row4col[j];
                if r != u32::MAX {
                    u[r as usize] += min_val - spc[j];
                }
                v[j] += spc[j] - min_val;
            }
        }

        let mut j = sink;
        loop {
            let pi = path[j] as usize;
            row4col[j] = pi as u32;
            let prev_j = col4row[pi];
            col4row[pi] = j;
            if pi == cur_row {
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
/// After matching, pairs whose IoU is below `iou_threshold` are discarded.
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
    let non_overlapping_ratio =
        costs_flat
            .iter()
            .fold(0_f32, |acc, x| if *x == 1_f32 { &acc + 1_f32 } else { acc })
            / costs_flat.len() as f32;

    // if cost matrix is too sparse, the overhead of calling InnerScan in the SIMD
    // implementation dominates so it's much faster to use the scalar implementation.
    let lsap_func = if non_overlapping_ratio > SIMD_MAX_SPARSITY {
        lsap
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
