use crate::{boxes, utils};
#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, Num, ToPrimitive};

// ─── Slice-based core ───

/// Calculates the Complete Intersection over Union (CIoU) distance between two sets of bounding boxes.
/// See <https://arxiv.org/pdf/1911.08287.pdf>.
///
/// CIoU = IoU - (ρ²(b, b_gt) / c²) - αv
/// where v = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))² and α = v / ((1 - IoU) + v).
///
/// # Arguments
///
/// * `boxes1` - A flat slice of length `n1 * 4` representing N bounding boxes in xyxy format (row-major).
/// * `boxes2` - A flat slice of length `n2 * 4` representing M bounding boxes in xyxy format (row-major).
/// * `n1` - The number of boxes in the first set.
/// * `n2` - The number of boxes in the second set.
///
/// # Returns
///
/// A flat `Vec<f64>` of length `n1 * n2` (row-major) representing the CIoU distance
/// between each pair of bounding boxes.
pub fn ciou_distance_slice<N>(boxes1: &[N], boxes2: &[N], n1: usize, n2: usize) -> Vec<f64>
where
    N: Num + PartialOrd + ToPrimitive + Float,
{
    let two = N::one() + N::one();
    let four_over_pi_sq: f64 = 4.0 / (std::f64::consts::PI * std::f64::consts::PI);
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
            let iou = intersection / (area1 + area2 - intersection);

            // Center distance squared
            let center_box1 = [(a1_x1 + a1_x2) / two, (a1_y1 + a1_y2) / two];
            let center_box2 = [(a2_x1 + a2_x2) / two, (a2_y1 + a2_y2) / two];
            let rho_sq = Float::powf(center_box1[0] - center_box2[0], two)
                + Float::powf(center_box1[1] - center_box2[1], two);

            // Enclosing box diagonal squared
            let enc_x1 = utils::min(a1_x1, a2_x1);
            let enc_y1 = utils::min(a1_y1, a2_y1);
            let enc_x2 = utils::max(a1_x2, a2_x2);
            let enc_y2 = utils::max(a1_y2, a2_y2);
            let c_sq = Float::powf(enc_x2 - enc_x1, two) + Float::powf(enc_y2 - enc_y1, two);

            let diou_penalty = rho_sq / c_sq;

            // Aspect ratio consistency term
            let w1 = (a1_x2 - a1_x1).to_f64().unwrap();
            let h1 = (a1_y2 - a1_y1).to_f64().unwrap();
            let w2 = (a2_x2 - a2_x1).to_f64().unwrap();
            let h2 = (a2_y2 - a2_y1).to_f64().unwrap();
            let atan_diff = h2.atan2(w2) - h1.atan2(w1);
            let v = four_over_pi_sq * atan_diff * atan_diff;
            let alpha = if v > 0.0 {
                v / ((utils::ONE - iou) + v)
            } else {
                0.0
            };

            result[i * n2 + j] = utils::ONE - (iou - diou_penalty.to_f64().unwrap() - alpha * v);
        }
    }

    result
}

// ─── ndarray wrapper ───

/// Calculates the Complete Intersection over Union (CIoU) distance between two sets of bounding boxes.
/// See <https://arxiv.org/pdf/1911.08287.pdf>.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape (N, 4) representing N bounding boxes in xyxy format.
/// * `boxes2` - A 2D array of shape (M, 4) representing M bounding boxes in xyxy format.
///
/// # Returns
///
/// A 2D array of shape (N, M) representing the CIoU distance between each pair of bounding boxes.
#[cfg(feature = "ndarray")]
pub fn ciou_distance<'a, BA, N>(boxes1: BA, boxes2: BA) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Float + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let b1 = boxes1.into();
    let b2 = boxes2.into();
    let n1 = b1.nrows();
    let n2 = b2.nrows();
    let s1 = b1.as_slice().expect("boxes1 must be contiguous");
    let s2 = b2.as_slice().expect("boxes2 must be contiguous");
    let result = ciou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ciou_identical_boxes() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![0.0, 0.0, 2.0, 2.0];
        let result = ciou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert!((result[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ciou_non_overlapping_boxes() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![3.0, 3.0, 4.0, 4.0];
        let result = ciou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_ciou_aspect_ratio_penalty() {
        // Wide box vs tall box: aspect ratio penalty should increase CIoU distance
        let boxes1 = vec![0.0, 0.0, 4.0, 1.0]; // wide box (4:1)
        let boxes2 = vec![0.0, 0.0, 1.0, 4.0]; // tall box (1:4)
        let ciou = ciou_distance_slice(&boxes1, &boxes2, 1, 1);
        // Same aspect ratio: no extra penalty
        let boxes3 = vec![0.0, 0.0, 4.0, 1.0]; // same wide box
        let ciou_same = ciou_distance_slice(&boxes1, &boxes3, 1, 1);
        assert!(
            ciou[0] > ciou_same[0],
            "Different aspect ratios should increase CIoU distance"
        );
    }

    #[test]
    fn test_ciou_same_aspect_ratio() {
        // When aspect ratios match, v=0 so CIoU = DIoU
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![1.0, 1.0, 3.0, 3.0];
        let ciou = ciou_distance_slice(&boxes1, &boxes2, 1, 1);
        // Both boxes are square, so aspect ratio penalty v=0, CIoU differs from DIoU
        // only by using the correct enclosing box diagonal
        assert!(ciou[0] > 0.0);
        assert!(ciou[0] < 2.0);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use super::*;
        use ndarray::arr2;

        #[test]
        fn test_ciou_distance_identical() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let result = ciou_distance(&boxes1, &boxes2);
            assert!((result[[0, 0]] - 0.0).abs() < 1e-10);
        }

        #[test]
        fn test_ciou_distance_non_overlapping() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
            let result = ciou_distance(&boxes1, &boxes2);
            assert_eq!(result[[0, 0]], 1.0);
        }
    }
}
