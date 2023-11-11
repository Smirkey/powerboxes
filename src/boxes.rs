use ndarray::Array2;
use ndarray::{Array1, Axis, Zip};

pub enum BoxFormat {
    XYXY,
    XYWH,
    CXCYWH,
}

/// Calculates the areas of a 2D array of boxes.
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes represented as an `Array2<f64>` in xyxy format.
///
/// # Returns
///
/// An `Array1<f64>` containing the areas of each box in the same order as the input array.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxes::boxes::box_areas;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
///
/// let areas = box_areas(&boxes);
///
/// assert_eq!(areas, array![9., 121.]);
/// ```
pub fn box_areas(boxes: &Array2<f64>) -> Array1<f64> {
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);

    Zip::indexed(&mut areas).for_each(|i, area| {
        let box1 = boxes.row(i);
        let x1 = box1[0];
        let y1 = box1[1];
        let x2 = box1[2];
        let y2 = box1[3];
        *area = (x2 - x1 + 1.) * (y2 - y1 + 1.);
    });

    return areas;
}

/// Calculates the areas of a 2D array of boxes in parallel.
/// This function is only faster than `box_areas` for large arrays
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes represented as an `Array2<f64>` in xyxy format.
///
/// # Returns
///
/// An `Array1<f64>` containing the areas of each box in the same order as the input array.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxes::boxes::parallel_box_areas;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
///
/// let areas = parallel_box_areas(&boxes);
///
/// assert_eq!(areas, array![9., 121.]);
/// ```
pub fn parallel_box_areas(boxes: &Array2<f64>) -> Array1<f64> {
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);

    Zip::indexed(&mut areas).par_for_each(|i, area| {
        let box1 = boxes.row(i);
        let x1 = box1[0];
        let y1 = box1[1];
        let x2 = box1[2];
        let y2 = box1[3];
        *area = (x2 - x1 + 1.) * (y2 - y1 + 1.);
    });

    return areas;
}

/// Removes all boxes from the input array that have a size smaller than `min_size`.
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes represented as an `Array2<f64>` in xyxy format.
/// * `min_size` - The minimum size of boxes to keep.
///
/// # Returns
///
/// A new 2D array with all boxes smaller than `min_size` removed.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxes::boxes::remove_small_boxes;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
/// let min_size = 10;
/// let result = remove_small_boxes(&boxes, min_size);
///
/// assert_eq!(result, array![[0., 0., 10., 10.]]));
/// ```
pub fn remove_small_boxes(boxes: &Array2<f64>, min_size: f64) -> Array2<f64> {
    let areas = box_areas(boxes);
    let keep: Vec<usize> = areas
        .indexed_iter()
        .filter(|(_, &area)| area >= min_size)
        .map(|(index, _)| index)
        .collect();
    return boxes.select(Axis(0), &keep);
}

/// Converts a 2D array of boxes from one format to another.
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes in the input format.
/// * `in_fmt` - The input format of the boxes.
/// * `out_fmt` - The desired output format of the boxes.
///
/// # Returns
///
/// A 2D array of boxes in the output format.
///
/// # Example
///
/// ```
/// use ndarray::arr2;
/// use powerboxes::{BoxFormat, box_convert};
///
/// let boxes = arr2(&[
///     [10.0, 20.0, 30.0, 40.0],
///     [75.0, 25.0, 100.0, 200.0],
///     [100.0, 100.0, 101.0, 101.0],
/// ]);
/// let in_fmt = BoxFormat::XYXY;
/// let out_fmt = BoxFormat::CXCYWH;
/// let expected_output = arr2(&[
///     [20.0, 30.0, 20.0, 20.0],
///     [87.5, 112.5, 25.0, 175.0],
///     [100.5, 100.5, 1.0, 1.0],
/// ]);
/// let output = box_convert(&boxes, &in_fmt, &out_fmt);
/// assert_eq!(output, expected_output);
/// ```
pub fn box_convert(boxes: &Array2<f64>, in_fmt: &BoxFormat, out_fmt: &BoxFormat) -> Array2<f64> {
    let num_boxes: usize = boxes.nrows();
    let mut converted_boxes = Array2::<f64>::zeros((num_boxes, 4));

    Zip::indexed(converted_boxes.rows_mut()).for_each(|i, mut box1| {
        let box2 = boxes.row(i);
        match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x2 - x1;
                box1[3] = y2 - y1;
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = (x1 + x2) / 2.;
                box1[1] = (y1 + y2) / 2.;
                box1[2] = x2 - x1;
                box1[3] = y2 - y1;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1;
                box1[1] = y1;
                box1[2] = x1 + w;
                box1[3] = y1 + h;
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1 + w / 2.;
                box1[1] = y1 + h / 2.;
                box1[2] = w;
                box1[3] = h;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / 2.;
                box1[1] = cy - h / 2.;
                box1[2] = cx + w / 2.;
                box1[3] = cy + h / 2.;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / 2.;
                box1[1] = cy - h / 2.;
                box1[2] = w;
                box1[3] = h;
            }
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    });
    return converted_boxes;
}
/// Converts a 2D array of boxes from one format to another, in parallel.
/// This function is only faster than `box_convert` for large arrays
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes in the input format.
/// * `in_fmt` - The input format of the boxes.
/// * `out_fmt` - The desired output format of the boxes.
///
/// # Returns
///
/// A 2D array of boxes in the output format.
///
/// # Example
///
/// ```
/// use ndarray::arr2;
/// use powerboxes::{BoxFormat, parallel_box_convert};
///
/// let boxes = arr2(&[
///     [10.0, 20.0, 30.0, 40.0],
///     [75.0, 25.0, 100.0, 200.0],
///     [100.0, 100.0, 101.0, 101.0],
/// ]);
/// let in_fmt = BoxFormat::XYXY;
/// let out_fmt = BoxFormat::CXCYWH;
/// let expected_output = arr2(&[
///     [20.0, 30.0, 20.0, 20.0],
///     [87.5, 112.5, 25.0, 175.0],
///     [100.5, 100.5, 1.0, 1.0],
/// ]);
/// let output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
/// assert_eq!(expected_output, output);
/// ```
pub fn parallel_box_convert(
    boxes: &Array2<f64>,
    in_fmt: &BoxFormat,
    out_fmt: &BoxFormat,
) -> Array2<f64> {
    let mut converted_boxes = boxes.clone();

    Zip::indexed(converted_boxes.rows_mut()).par_for_each(|i, mut box1| {
        let box2 = boxes.row(i);
        match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[2] = x2 - x1;
                box1[3] = y2 - y1;
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let x2 = box2[2];
                let y2 = box2[3];
                box1[0] = (x1 + x2) / 2.;
                box1[1] = (y1 + y2) / 2.;
                box1[2] = x2 - x1;
                box1[3] = y2 - y1;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[2] = x1 + w;
                box1[3] = y1 + h;
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let x1 = box2[0];
                let y1 = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = x1 + w / 2.;
                box1[1] = y1 + h / 2.;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / 2.;
                box1[1] = cy - h / 2.;
                box1[2] = cx + w / 2.;
                box1[3] = cy + h / 2.;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / 2.;
                box1[1] = cy - h / 2.;
            }
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    });
    return converted_boxes;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, array};
    #[test]
    fn test_box_convert_xyxy_to_xywh() {
        let boxes = arr2(&[
            [10., 20., 30., 40.],
            [75., 25., 100., 200.],
            [100., 100., 101., 101.],
        ]);
        let in_fmt = BoxFormat::XYXY;
        let out_fmt = BoxFormat::XYWH;
        let expected_output = arr2(&[
            [10.0, 20.0, 20.0, 20.0],
            [75.0, 25.0, 25.0, 175.0],
            [100.0, 100.0, 1.0, 1.0],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_box_convert_xyxy_to_cxcywh() {
        let boxes = arr2(&[
            [10.0, 20.0, 30.0, 40.0],
            [75.0, 25.0, 100.0, 200.0],
            [100.0, 100.0, 101.0, 101.0],
        ]);
        let in_fmt = BoxFormat::XYXY;
        let out_fmt = BoxFormat::CXCYWH;
        let expected_output = arr2(&[
            [20.0, 30.0, 20.0, 20.0],
            [87.5, 112.5, 25.0, 175.0],
            [100.5, 100.5, 1.0, 1.0],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_box_convert_xywh_to_xyxy() {
        let boxes = arr2(&[
            [10.0, 20.0, 20.0, 20.0],
            [75.0, 25.0, 25.0, 175.0],
            [100.0, 100.0, 1.0, 1.0],
        ]);
        let in_fmt = BoxFormat::XYWH;
        let out_fmt = BoxFormat::XYXY;
        let expected_output = arr2(&[
            [10.0, 20.0, 30.0, 40.0],
            [75.0, 25.0, 100.0, 200.0],
            [100.0, 100.0, 101.0, 101.0],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_box_convert_xywh_to_cxcywh() {
        let boxes = arr2(&[
            [10.0, 20.0, 20.0, 20.0],
            [75.0, 25.0, 25.0, 175.0],
            [100.0, 100.0, 1.0, 1.0],
        ]);
        let in_fmt = BoxFormat::XYWH;
        let out_fmt = BoxFormat::CXCYWH;
        let expected_output = arr2(&[
            [20.0, 30.0, 20.0, 20.0],
            [87.5, 112.5, 25.0, 175.0],
            [100.5, 100.5, 1.0, 1.0],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_box_convert_cxcywh_to_xyxy() {
        let boxes = arr2(&[
            [20.0, 30.0, 20.0, 20.0],
            [87.5, 112.5, 25.0, 175.0],
            [100.5, 100.5, 1.0, 1.0],
        ]);
        let in_fmt = BoxFormat::CXCYWH;
        let out_fmt = BoxFormat::XYXY;
        let expected_output = arr2(&[
            [10., 20., 30., 40.],
            [75., 25., 100., 200.],
            [100., 100., 101., 101.],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_box_convert_cxcywh_to_xywh() {
        let boxes = arr2(&[
            [20.0, 30.0, 20.0, 20.0],
            [87.5, 112.5, 25.0, 175.0],
            [100.5, 100.5, 1.0, 1.0],
        ]);
        let in_fmt = BoxFormat::CXCYWH;
        let out_fmt = BoxFormat::XYWH;
        let expected_output = arr2(&[
            [10.0, 20.0, 20.0, 20.0],
            [75.0, 25.0, 25.0, 175.0],
            [100.0, 100.0, 1.0, 1.0],
        ]);
        let output = box_convert(&boxes, &in_fmt, &out_fmt);
        let parallel_output = parallel_box_convert(&boxes, &in_fmt, &out_fmt);
        assert_eq!(output, expected_output);
        assert_eq!(output, parallel_output);
    }

    #[test]
    fn test_coherence() {
        let boxes = arr2(&[
            [10., 20., 30., 40.],
            [75., 25., 100., 200.],
            [100., 100., 101., 101.],
        ]);
        let xywh = parallel_box_convert(&boxes, &BoxFormat::XYXY, &BoxFormat::XYWH);
        let cxcywh = parallel_box_convert(&xywh, &BoxFormat::XYWH, &BoxFormat::CXCYWH);
        assert_eq!(
            parallel_box_convert(&cxcywh, &BoxFormat::CXCYWH, &BoxFormat::XYXY),
            boxes
        );
        assert_eq!(
            parallel_box_convert(&xywh, &BoxFormat::XYWH, &BoxFormat::XYXY),
            boxes
        );
    }
    #[test]
    fn test_box_areas_single_box() {
        let boxes = array![[1., 2., 3., 4.]];
        let areas = box_areas(&boxes);
        let parallel_areas = box_areas(&boxes);
        assert_eq!(areas, array![9.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_multiple_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let areas = box_areas(&boxes);
        let parallel_areas = box_areas(&boxes);
        assert_eq!(areas, array![9., 121.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_zero_area() {
        let boxes = array![[1., 2., 1., 2.]];
        let areas = box_areas(&boxes);
        let parallel_areas = box_areas(&boxes);
        assert_eq!(areas, array![1.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_negative_coordinates() {
        let boxes = array![[-1., -1., 1., 1.]];
        let areas = box_areas(&boxes);
        let parallel_areas = box_areas(&boxes);
        assert_eq!(areas, array![9.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_remove_small_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let min_size = 10.;
        let filtered_boxes = remove_small_boxes(&boxes, min_size);
        assert_eq!(filtered_boxes, array![[0., 0., 10., 10.]]);
    }
}
