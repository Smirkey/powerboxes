use crate::{boxes, utils};
#[cfg(feature = "ndarray")]
use ndarray::{Array2, ArrayView2};
use num_traits::{Float, Num, ToPrimitive};

// ─── Slice-based core ───

/// Calculates the DIoU distance between two sets of bounding boxes (slice-based).
///
/// `boxes1` is a flat slice of length `n1 * 4` (row-major xyxy format).
/// `boxes2` is a flat slice of length `n2 * 4`.
/// Returns a flat `Vec<f64>` of length `n1 * n2` (row-major).
pub fn diou_distance_slice<N>(boxes1: &[N], boxes2: &[N], n1: usize, n2: usize) -> Vec<f64>
where
    N: Num + PartialOrd + ToPrimitive + Float,
{
    let two = N::one() + N::one();
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

            let center_box1 = [(a1_x1 + a1_x2) / two, (a1_y1 + a1_y2) / two];
            let center_box2 = [(a2_x1 + a2_x2) / two, (a2_y1 + a2_y2) / two];

            let d = Float::sqrt(
                Float::powf(center_box1[0] - center_box2[0], two)
                    + Float::powf(center_box1[1] - center_box2[1], two),
            );
            let c = Float::sqrt(Float::powf(x2 - x1, two) + Float::powf(y2 - y1, two));
            let diou_penalty = Float::powf(d, two) / Float::powf(c, two);
            result[i * n2 + j] = utils::ONE - (iou - diou_penalty.to_f64().unwrap());
        }
    }

    result
}

// ─── ndarray wrapper ───

#[cfg(feature = "ndarray")]
pub fn diou_distance<'a, BA, N>(boxes1: BA, boxes2: BA) -> Array2<f64>
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
    let result = diou_distance_slice(s1, s2, n1, n2);
    Array2::from_shape_vec((n1, n2), result).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diou_distance_slice() {
        let boxes1 = vec![0.0, 0.0, 2.0, 2.0];
        let boxes2 = vec![1.0, 1.0, 3.0, 3.0];
        let result = diou_distance_slice(&boxes1, &boxes2, 1, 1);
        assert_eq!(result, vec![1.8571428571428572]);
    }

    #[cfg(feature = "ndarray")]
    mod ndarray_tests {
        use ndarray::arr2;
        use super::*;

        #[test]
        fn test_diou_distance() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
            let diou_distance_result = diou_distance(&boxes1, &boxes2);
            assert_eq!(diou_distance_result, arr2(&[[1.8571428571428572]]));
        }

        #[test]
        fn test_diou_distance_distance2() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
            let diou_distance_result = diou_distance(&boxes1, &boxes2);
            assert_eq!(diou_distance_result, arr2(&[[1.0]]));
        }

        #[test]
        fn test_diou_distance_distance3() {
            let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
            let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
            let diou_distance_result = diou_distance(&boxes1, &boxes2);
            assert_eq!(diou_distance_result, arr2(&[[3.187499999999999]]));
        }

        #[test]
        fn test_diou_distance_distance4() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let diou_distance_result = diou_distance(&boxes1, &boxes2);
            assert_eq!(diou_distance_result, arr2(&[[0.0]]));
        }

        #[test]
        fn test_diou_distance_distance5() {
            let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
            let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
            let diou_distance_result = diou_distance(&boxes1, &boxes2);
            assert_eq!(diou_distance_result, arr2(&[[1.0]]));
        }
    }
}
