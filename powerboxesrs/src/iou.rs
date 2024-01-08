use crate::{boxes, rotation::cxcywha_to_points, utils};
use ndarray::{Array2, Axis, Zip};
use num_traits::{Num, ToPrimitive};
use rstar::{Envelope, RStarInsertionStrategy, RTree, RTreeNum, RTreeObject, RTreeParams, AABB};

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

/// Calculates the intersection over union (IoU) between two sets of bounding boxes.
///
/// # Arguments
///
/// * `boxes1` - A 2D array of shape (N, 4) representing N bounding boxes in xyxy format.
/// * `boxes2` - A 2D array of shape (M, 4) representing M bounding boxes in xyxy format.
///
/// # Returns
///
/// A 2D array of shape (N, M) representing the IoU between each pair of bounding boxes.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxesrs::iou::iou;
///
/// let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
/// let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
/// let iou = iou(&boxes1, &boxes2);
/// assert_eq!(iou, array![[0.1428571428571428, 0.],[0., 0.1428571428571428]]);
/// ```
pub fn iou<N>(boxes1: &Array2<N>, boxes2: &Array2<N>) -> Array2<f64>
where
    N: Num + PartialOrd + ToPrimitive + Copy,
{
    let iou_distance = iou_distance(boxes1, boxes2);
    let iou = utils::ONE - iou_distance;
    return iou;
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

// Struct we use to represent a bbox object in rstar R-tree
struct OrientedBbox<T> {
    index: usize,
    x1: T,
    y1: T,
    x2: T,
    y2: T,
    x3: T,
    y3: T,
    x4: T,
    y4: T,
}

// Implement RTreeObject for Bbox
impl<T> RTreeObject for OrientedBbox<T>
where
    T: RTreeNum + ToPrimitive + Sync + Send,
{
    type Envelope = AABB<[T; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_points([
            &[self.x1, self.y1],
            &[self.x2, self.y2],
            &[self.x3, self.y3],
            &[self.x4, self.y4],
        ])
    }
}

impl<T> RTreeParams for OrientedBbox<T>
where
    T: RTreeNum + ToPrimitive + Sync + Send,
{
    const MIN_SIZE: usize = 16;
    const MAX_SIZE: usize = 256;
    const REINSERTION_COUNT: usize = 5;
    type DefaultInsertionStrategy = RStarInsertionStrategy;
}

pub fn rotated_iou_distance(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::ones((num_boxes1, num_boxes2));
    let points_boxes_1: Vec<OrientedBbox<f64>> = boxes1
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(i, row)| {
            let (p1, p2, p3, p4) = cxcywha_to_points(row[0], row[1], row[2], row[3], row[4]);
            OrientedBbox {
                index: i,
                x1: p1.x,
                y1: p1.y,
                x2: p2.x,
                y2: p2.y,
                x3: p3.x,
                y3: p3.y,
                x4: p4.x,
                y4: p4.y,
            }
        })
        .collect();
    let points_boxes_2: Vec<OrientedBbox<f64>> = boxes2
        .axis_iter(Axis(0))
        .enumerate()
        .map(|(i, row)| {
            let (p1, p2, p3, p4) = cxcywha_to_points(row[0], row[1], row[2], row[3], row[4]);
            OrientedBbox {
                index: i,
                x1: p1.x,
                y1: p1.y,
                x2: p2.x,
                y2: p2.y,
                x3: p3.x,
                y3: p3.y,
                x4: p4.x,
                y4: p4.y,
            }
        })
        .collect();
    let rtree_boxes_1: RTree<OrientedBbox<f64>> = RTree::bulk_load(points_boxes_1);
    let rtree_boxes_2: RTree<OrientedBbox<f64>> = RTree::bulk_load(points_boxes_2);

    for (box1, box2) in rtree_boxes_1.intersection_candidates_with_other_tree(&rtree_boxes_2) {
        let box1_envelope = box1.envelope();
        let box2_envelope = box2.envelope();
        let intersection = box1_envelope.intersection_area(&box2_envelope);
        let iou = intersection
            / (box1_envelope.area() + box2_envelope.area() - intersection + utils::EPS);
        iou_matrix[[box1.index, box2.index]] = utils::ONE - iou;
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
        let iou_result = iou(&boxes1, &boxes2);

        assert_eq!(iou_distance_result, arr2(&[[0.8571428571428572]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[0.8571428571428572]]));
        assert_eq!(1. - iou_distance_result, iou_result);
    }

    #[test]
    fn test_iou_distance2() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
        let iou_distance_result = iou_distance(&boxes1, &boxes2);
        let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
        let iou_result = iou(&boxes1, &boxes2);
        assert_eq!(iou_distance_result, arr2(&[[1.0]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[1.0]]));
        assert_eq!(1. - iou_distance_result, iou_result);
    }

    #[test]
    fn test_iou_distance3() {
        let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
        let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
        let iou_distance_result = iou_distance(&boxes1, &boxes2);
        let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
        let iou_result = iou(&boxes1, &boxes2);
        assert_eq!(iou_distance_result, arr2(&[[0.9375]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[0.9375]]));
        assert_eq!(1. - iou_distance_result, iou_result);
    }

    #[test]
    fn test_iou_distance4() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let iou_distance_result = iou_distance(&boxes1, &boxes2);
        let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
        let iou_result = iou(&boxes1, &boxes2);
        assert_eq!(iou_distance_result, arr2(&[[0.0]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[0.0]]));
        assert_eq!(1. - iou_distance_result, iou_result);
    }

    #[test]
    fn test_iou_disstance5() {
        let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
        let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
        let iou_distance_result = iou_distance(&boxes1, &boxes2);
        let parallel_iou_distance_result = parallel_iou_distance(&boxes1, &boxes2);
        let iou_result = iou(&boxes1, &boxes2);
        assert_eq!(iou_distance_result, arr2(&[[1.0]]));
        assert_eq!(parallel_iou_distance_result, arr2(&[[1.0]]));
        assert_eq!(1. - iou_distance_result, iou_result);
    }

    #[test]
    fn test_rotated_iou_disstance() {
        let boxes1 = arr2(&[[5.0, 5.0, 2.0, 2.0, 0.0]]);
        let boxes2 = arr2(&[[4.0, 4.0, 2.0, 2.0, 0.0]]);
        let rotated_iou_distance_result = rotated_iou_distance(&boxes1, &boxes2);
        assert_eq!(rotated_iou_distance_result, arr2(&[[0.8571428571428572]]));
    }
}
