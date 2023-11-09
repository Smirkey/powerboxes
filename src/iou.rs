use ndarray::{Array2, Zip};

pub fn distance_box_iou(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));

    for i in 0..num_boxes1 {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = (a1_x2 - a1_x1 + 1.) * (a1_y2 - a1_y1 + 1.);

        for j in 0..num_boxes2 {
            let a2 = boxes2.row(j);
            let a2_x1 = a2[0];
            let a2_y1 = a2[1];
            let a2_x2 = a2[2];
            let a2_y2 = a2[3];
            let area2 = (a2_x2 - a2_x1 + 1.) * (a2_y2 - a2_y1 + 1.);

            let x1 = f64::max(a1_x1, a2_x1);
            let y1 = f64::max(a1_y1, a2_y1);
            let x2 = f64::min(a1_x2, a2_x2);
            let y2 = f64::min(a1_y2, a2_y2);

            let intersection = (x2 - x1 + 1.) * (y2 - y1 + 1.);
            let iou = intersection / (area1 + area2 - intersection);

            iou_matrix[[i, j]] = 1. - iou;
        }
    }

    return iou_matrix;
}

pub fn parallel_distance_box_iou(boxes1: &Array2<f64>, boxes2: &Array2<f64>) -> Array2<f64> {
    let num_boxes1 = boxes1.nrows();
    let num_boxes2 = boxes2.nrows();

    let mut iou_matrix = Array2::<f64>::zeros((num_boxes1, num_boxes2));

    Zip::indexed(iou_matrix.rows_mut()).par_for_each(|i, mut row| {
        let a1 = boxes1.row(i);
        let a1_x1 = a1[0];
        let a1_y1 = a1[1];
        let a1_x2 = a1[2];
        let a1_y2 = a1[3];
        let area1 = (a1_x2 - a1_x1 + 1.) * (a1_y2 - a1_y1 + 1.);
        row.iter_mut().zip(boxes2.rows()).for_each(|(d, box2)| {
            let a2_x1 = box2[0];
            let a2_y1 = box2[1];
            let a2_x2 = box2[2];
            let a2_y2 = box2[3];
            let area2 = (a2_x2 - a2_x1 + 1.) * (a2_y2 - a2_y1 + 1.);

            let x1 = f64::max(a1_x1, a2_x1);
            let y1 = f64::max(a1_y1, a2_y1);
            let x2 = f64::min(a1_x2, a2_x2);
            let y2 = f64::min(a1_y2, a2_y2);
            if x2 < x1 || y2 < y1 {
                *d = 0.0;
            }

            let intersection = (x2 - x1 + 1.0) * (y2 - y1 + 1.0);

            let iou = intersection / (area1 + area2 - intersection);

            *d = 1.0 - iou;
        });
    });

    return iou_matrix;
}

#[test]
fn test_distance_box_iou() {
    use ndarray::arr2;
    // Test case 1
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
    let iou_result = distance_box_iou(&boxes1, &boxes2);
    let parallel_iou_result = parallel_distance_box_iou(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.7142857142857143]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.7142857142857143]]));

    // Test case 2
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
    let iou_result = distance_box_iou(&boxes1, &boxes2);
    let parallel_iou_result = parallel_distance_box_iou(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[1.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[1.0]]));

    // Test case 3
    let boxes1 = arr2(&[[2.5, 2.5, 3.0, 3.0]]);
    let boxes2 = arr2(&[[1.0, 1.0, 3.0, 3.0]]);
    let iou_result = distance_box_iou(&boxes1, &boxes2);
    let parallel_iou_result = parallel_distance_box_iou(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.75]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.75]]));

    // Test case 4
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let iou_result = distance_box_iou(&boxes1, &boxes2);
    let parallel_iou_result = parallel_distance_box_iou(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[0.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[0.0]]));

    // Test case 5
    let boxes1 = arr2(&[[0.0, 0.0, 2.0, 2.0]]);
    let boxes2 = arr2(&[[3.0, 3.0, 4.0, 4.0]]);
    let iou_result = distance_box_iou(&boxes1, &boxes2);
    let parallel_iou_result = parallel_distance_box_iou(&boxes1, &boxes2);
    assert_eq!(iou_result, arr2(&[[1.0]]));
    assert_eq!(parallel_iou_result, arr2(&[[1.0]]));
}
