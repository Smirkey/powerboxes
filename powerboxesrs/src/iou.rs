use crate::{
    boxes::{self, rotated_box_areas},
    rotation::{intersection_area, Rect},
    utils,
};
use ndarray::{Array2, Zip};
use num_traits::{Num, ToPrimitive};
use rstar::RTree;

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
/// assert_eq!(iou, array![[0.8571428571428572, 1.],[1., 0.8571428571428572]]);
/// ```
pub fn iou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));
    let areas_boxes1 = boxes::box_areas(&boxes1);
    let areas_boxes2 = boxes::box_areas(&boxes2);
    for (i, a1) in boxes1.outer_iter().enumerate() {
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = areas_boxes1[i];

        for (j, a2) in boxes2.outer_iter().enumerate() {
            let a2_x1 = a2[0];
            let a2_y1 = a2[1];
            let a2_x2 = a2[2];
            let a2_y2 = a2[3];
            let area2 = areas_boxes2[j];
            let x1 = utils::max(a1_x1, a2_x1);
            let y1 = utils::max(a1_y1, a2_y1);
            let x2 = utils::min(a1_x2, a2_x2);
            let y2 = utils::min(a1_y2, a2_y2);
            if x2 < x1 || y2 < y1 {
                iou_matrix[[i, j]] = utils::ONE;
                continue;
            }
            let intersection = (x2 - x1) * (y2 - y1);
            let intersection = intersection.to_f64().unwrap();
            let intersection = utils::min(intersection, utils::min(area1, area2));
            iou_matrix[[i, j]] =
                utils::ONE - (intersection / (area1 + area2 - intersection + utils::EPS));
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
/// use powerboxesrs::iou::parallel_iou_distance;
///
/// let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
/// let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
/// let iou = parallel_iou_distance(&boxes1, &boxes2);
/// assert_eq!(iou, array![[0.8571428571428572, 1.],[1., 0.8571428571428572]]);
/// ```
pub fn parallel_iou_distance<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy + Clone + Sync + Send,
{
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
                    *d = 1. - (intersection / (area1 + area2 - intersection + utils::EPS));
                }
            });
    });

    return iou_matrix;
}

pub fn rotated_iou_distance(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::ones((num_boxes1, num_boxes2));
    let areas1 = rotated_box_areas(boxes1);
    let areas2 = rotated_box_areas(boxes2);

    let boxes1_rects: Vec<Rect> = boxes1
        .rows()
        .into_iter()
        .map(|row| Rect::new(row[0], row[1], row[2], row[3], row[4]))
        .collect();
    let boxes2_rects: Vec<Rect> = boxes2
        .rows()
        .into_iter()
        .map(|row| Rect::new(row[0], row[1], row[2], row[3], row[4]))
        .collect();
    let boxes1_bounding_rects: Vec<utils::Bbox<f64>> = boxes1_rects
        .iter()
        .enumerate()
        .map(|(idx, rect)| {
            let points = rect.points();
            let (min_x, max_x) = points
                .iter()
                .map(|p| p.x)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, x| {
                    (acc.0.min(x), acc.1.max(x))
                });
            let (min_y, max_y) = points
                .iter()
                .map(|p| p.y)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, y| {
                    (acc.0.min(y), acc.1.max(y))
                });
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
            let points = rect.points();
            let (min_x, max_x) = points
                .iter()
                .map(|p| p.x)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, x| {
                    (acc.0.min(x), acc.1.max(x))
                });
            let (min_y, max_y) = points
                .iter()
                .map(|p| p.y)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |acc, y| {
                    (acc.0.min(y), acc.1.max(y))
                });
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
        let intersection = intersection_area(boxes1_rects[box1.index], boxes2_rects[box2.index]);
        let union = area1 + area2 - intersection + utils::EPS;
        iou_matrix[[box1.index, box2.index]] = utils::ONE - intersection / union;
    }

    return iou_matrix;
}

#[cfg(test)]
mod tests {
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
    fn test_iou_disstance5() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
        let iou_distance_result = iou_distance(&boxes1, &boxes2);
        let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
        assert_eq!(iou_distance_result, arr2(&[[1.0]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[1.0]]));
    }

    #[test]
    fn test_rotated_iou_disstance() {
        let boxes1 = arr2(&[[5.0, 5.0, 2.0, 2.0, 0.0]]);
        let boxes2 = arr2(&[[4.0, 4.0, 2.0, 2.0, 0.0]]);
        let rotated_iou_distance_result = rotated_iou_distance(&boxes1, &boxes2);
        assert_eq!(rotated_iou_distance_result, arr2(&[[0.8571428571428572]]));
    }
}
