use ndarray::{Array1, Array2, ArrayView2, ArrayView3, ArrayViewMut2, Axis, Zip};
use num_traits::{real::Real, Num, ToPrimitive};

#[derive(Copy, Clone)]
pub enum BoxFormat {
    XYXY,
    XYWH,
    CXCYWH,
}

/// Calculates the areas of a 2D array of boxes.
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes represented as an `ArrayView2<N>` in xyxy format.
///
/// # Returns
///
/// An `Array1<f64>` containing the areas of each box in the same order as the input array.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use powerboxesrs::boxes::box_areas;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
///
/// let areas = box_areas(&boxes);
///
/// assert_eq!(areas, array![4., 100.]);
/// ```
pub fn box_areas<'a, N, BA>(boxes: BA) -> Array1<f64>
where
    N: Num + PartialEq + ToPrimitive + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes = boxes.into();
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);
    Zip::indexed(&mut areas).for_each(|i, area| {
        let box1 = boxes.row(i);
        let area_ = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        *area = area_.to_f64().unwrap();
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
/// use powerboxesrs::boxes::parallel_box_areas;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
///
/// let areas = parallel_box_areas(&boxes);
///
/// assert_eq!(areas, array![4., 100.]);
/// ```
pub fn parallel_box_areas<'a, N, BA>(boxes: BA) -> Array1<f64>
where
    N: Real + Send + Sync + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes = boxes.into();
    let num_boxes = boxes.nrows();
    let mut areas = Array1::<f64>::zeros(num_boxes);

    Zip::indexed(&mut areas).par_for_each(|i, area| {
        let box1 = boxes.row(i);
        let x1: N = box1[0];
        let y1: N = box1[1];
        let x2: N = box1[2];
        let y2: N = box1[3];
        let _area = (x2 - x1) * (y2 - y1);
        *area = _area.to_f64().unwrap();
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
/// use powerboxesrs::boxes::remove_small_boxes;
///
/// let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
/// let min_size = 10.0;
/// let result = remove_small_boxes(&boxes, min_size);
///
/// assert_eq!(result, array![[0., 0., 10., 10.]]);
/// ```
pub fn remove_small_boxes<'a, N, BA>(boxes: BA, min_size: f64) -> Array2<N>
where
    N: Num + PartialEq + Clone + PartialOrd + ToPrimitive + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes = boxes.into();
    let areas = box_areas(boxes);
    let keep: Vec<usize> = areas
        .indexed_iter()
        .filter(|(_, &area)| area >= min_size)
        .map(|(index, _)| index)
        .collect();
    return boxes.select(Axis(0), &keep);
}

/// Converts a 2D array of boxes from one format to another, in-place.
/// This works because all box formats use 4 values in their representations.
///
/// # Arguments
///
/// * `boxes` - A 2D array of boxes in the input format.
/// * `in_fmt` - The input format of the boxes.
/// * `out_fmt` - The desired output format of the boxes.
///
/// # Example
///
/// ```
/// use ndarray::arr2;
/// use powerboxesrs::boxes::{BoxFormat, box_convert_inplace};
///
/// let mut boxes = arr2(&[
///     [10.0, 20.0, 30.0, 40.0],
///     [75.0, 25.0, 100.0, 200.0],
///     [100.0, 100.0, 101.0, 101.0],
/// ]);
/// let expected_output = arr2(&[
///     [20.0, 30.0, 20.0, 20.0],
///     [87.5, 112.5, 25.0, 175.0],
///     [100.5, 100.5, 1.0, 1.0],
/// ]);
/// box_convert_inplace(&mut boxes, BoxFormat::XYXY, BoxFormat::CXCYWH);
/// assert_eq!(boxes, expected_output);
/// ```
pub fn box_convert_inplace<'a, N, BA>(boxes: BA, in_fmt: BoxFormat, out_fmt: BoxFormat)
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Copy + 'a,
    BA: Into<ArrayViewMut2<'a, N>>,
{
    boxes
        .into()
        .rows_mut()
        .into_iter()
        .for_each(|mut bx| match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                bx[2] = bx[2] - bx[0];
                bx[3] = bx[3] - bx[1];
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = bx[0];
                let y1 = bx[1];
                let x2 = bx[2];
                let y2 = bx[3];
                bx[0] = (x1 + x2) / (N::one() + N::one());
                bx[1] = (y1 + y2) / (N::one() + N::one());
                bx[2] = x2 - x1;
                bx[3] = y2 - y1;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                bx[2] = bx[0] + bx[2];
                bx[3] = bx[1] + bx[3];
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let w = bx[2];
                let h = bx[3];
                bx[0] = bx[0] + w / (N::one() + N::one());
                bx[1] = bx[1] + h / (N::one() + N::one());
                bx[2] = w;
                bx[3] = h;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => {
                let cx = bx[0];
                let cy = bx[1];
                let wd2 = bx[2] / (N::one() + N::one());
                let hd2 = bx[3] / (N::one() + N::one());
                bx[0] = cx - wd2;
                bx[1] = cy - hd2;
                bx[2] = cx + wd2;
                bx[3] = cy + hd2;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => {
                let w = bx[2];
                let h = bx[3];
                bx[0] = bx[0] - w / (N::one() + N::one());
                bx[1] = bx[1] - h / (N::one() + N::one());
                bx[2] = w;
                bx[3] = h;
            }
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        });
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
/// use powerboxesrs::boxes::{BoxFormat, box_convert};
///
/// let boxes = arr2(&[
///     [10.0, 20.0, 30.0, 40.0],
///     [75.0, 25.0, 100.0, 200.0],
///     [100.0, 100.0, 101.0, 101.0],
/// ]);
/// let expected_output = arr2(&[
///     [20.0, 30.0, 20.0, 20.0],
///     [87.5, 112.5, 25.0, 175.0],
///     [100.5, 100.5, 1.0, 1.0],
/// ]);
/// let output = box_convert(&boxes, BoxFormat::XYXY, BoxFormat::CXCYWH);
/// assert_eq!(output, expected_output);
/// ```
pub fn box_convert<'a, N, BA>(boxes: BA, in_fmt: BoxFormat, out_fmt: BoxFormat) -> Array2<N>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let mut converted_boxes = boxes.into().to_owned();
    box_convert_inplace(&mut converted_boxes, in_fmt, out_fmt);
    converted_boxes
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
/// use powerboxesrs::boxes::{BoxFormat, parallel_box_convert};
///
/// let boxes = arr2(&[
///     [10.0, 20.0, 30.0, 40.0],
///     [75.0, 25.0, 100.0, 200.0],
///     [100.0, 100.0, 101.0, 101.0],
/// ]);
/// let expected_output = arr2(&[
///     [20.0, 30.0, 20.0, 20.0],
///     [87.5, 112.5, 25.0, 175.0],
///     [100.5, 100.5, 1.0, 1.0],
/// ]);
/// let output = parallel_box_convert(&boxes, BoxFormat::XYXY, BoxFormat::CXCYWH);
/// assert_eq!(expected_output, output);
/// ```
pub fn parallel_box_convert<N>(
    boxes: &Array2<N>,
    in_fmt: BoxFormat,
    out_fmt: BoxFormat,
) -> Array2<N>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Sync + Send + Copy,
{
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
                box1[0] = (x1 + x2) / (N::one() + N::one());
                box1[1] = (y1 + y2) / (N::one() + N::one());
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
                box1[0] = x1 + w / (N::one() + N::one());
                box1[1] = y1 + h / (N::one() + N::one());
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / (N::one() + N::one());
                box1[1] = cy - h / (N::one() + N::one());
                box1[2] = cx + w / (N::one() + N::one());
                box1[3] = cy + h / (N::one() + N::one());
            }
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => {
                let cx = box2[0];
                let cy = box2[1];
                let w = box2[2];
                let h = box2[3];
                box1[0] = cx - w / (N::one() + N::one());
                box1[1] = cy - h / (N::one() + N::one());
            }
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    });
    return converted_boxes;
}

/// Compute the bounding boxes around the provided masks.
/// Returns a [N, 4] array containing bounding boxes. The boxes are in xyxy format
///
/// # Arguments
///
/// * `masks` - A [N, H, W] array of masks to transform where N is the number of masks and (H, W) are the spatial dimensions of the image.
///
/// # Returns
///
/// A [N, 4] array of boxes in xyxy format.
/// # Example
/// ```
/// use ndarray::{arr3, array};
/// use powerboxesrs::boxes::masks_to_boxes;
/// let masks = arr3(&[
///   [[true, true, true], [false, false, false]],
///   [[false, false, false], [true, true, true]],
///   [[false, false, false], [false, false, true]],
/// ]);
/// let boxes = masks_to_boxes(&masks);
/// assert_eq!(boxes, array![[0, 0, 2, 0], [0, 1, 2, 1], [2, 1, 2, 1]]);
pub fn masks_to_boxes<'a, MA>(masks: MA) -> Array2<usize>
where
    MA: Into<ArrayView3<'a, bool>>,
{
    let masks = masks.into();
    let num_masks = masks.shape()[0];
    let height = masks.shape()[1];
    let width = masks.shape()[2];
    let mut boxes = Array2::<usize>::zeros((num_masks, 4));

    for (i, mask) in masks.outer_iter().enumerate() {
        let mut x1 = width;
        let mut y1 = height;
        let mut x2 = 0;
        let mut y2 = 0;

        // get the indices where the mask is true
        mask.indexed_iter().for_each(|(index, &value)| {
            if value {
                let (y, x) = index;
                if x < x1 {
                    x1 = x;
                }
                if x > x2 {
                    x2 = x;
                }
                if y < y1 {
                    y1 = y;
                }
                if y > y2 {
                    y2 = y;
                }
            }
        });
        boxes[[i, 0]] = x1;
        boxes[[i, 1]] = y1;
        boxes[[i, 2]] = x2;
        boxes[[i, 3]] = y2;
    }

    return boxes;
}

/// Calculates the areas of rotated boxes in the cxcywha format.
///
/// Given an array of rotated boxes represented in the cxcywha format, where each row
/// corresponds to a box and the columns represent center-x, center-y, width, height,
/// and orientation angle, this function computes the area of each box and returns
/// an array containing the computed areas.
///
/// # Arguments
///
/// * `boxes` - A 2D array representing rotated boxes in the cxcywha format.
///
/// # Returns
///
/// A 1D array containing the computed areas of each rotated box.
///
pub fn rotated_box_areas<'a, BA>(boxes: BA) -> Array1<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let boxes = boxes.into();
    let n_boxes = boxes.nrows();

    let mut areas = Array1::zeros(n_boxes);

    for i in 0..n_boxes {
        areas[i] = boxes[[i, 2]] * boxes[[i, 3]]
    }

    areas
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3, array, Array3};
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let output = box_convert(&boxes, in_fmt, out_fmt);
        let parallel_output = parallel_box_convert(&boxes, in_fmt, out_fmt);
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
        let xywh = parallel_box_convert(&boxes, BoxFormat::XYXY, BoxFormat::XYWH);
        let cxcywh = parallel_box_convert(&xywh, BoxFormat::XYWH, BoxFormat::CXCYWH);
        assert_eq!(
            parallel_box_convert(&cxcywh, BoxFormat::CXCYWH, BoxFormat::XYXY),
            boxes
        );
        assert_eq!(
            parallel_box_convert(&xywh, BoxFormat::XYWH, BoxFormat::XYXY),
            boxes
        );
    }
    #[test]
    fn test_box_areas_single_box() {
        let boxes = array![[1., 2., 3., 4.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_multiple_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4., 100.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_zero_area() {
        let boxes = array![[1., 2., 1., 2.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![0.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_box_areas_negative_coordinates() {
        let boxes = array![[-1., -1., 1., 1.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4.]);
        assert_eq!(parallel_areas, areas);
    }

    #[test]
    fn test_remove_small_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let min_size = 10.;
        let filtered_boxes = remove_small_boxes(&boxes, min_size);
        assert_eq!(filtered_boxes, array![[0., 0., 10., 10.]]);
    }

    #[test]
    fn test_masks_to_boxes() {
        let masks: Array3<bool> = arr3(&[
            [[true, true, true], [false, false, false]],
            [[false, false, false], [true, true, true]],
            [[false, false, false], [false, false, true]],
        ]);
        let boxes = masks_to_boxes(&masks);
        assert_eq!(boxes, array![[0, 0, 2, 0], [0, 1, 2, 1], [2, 1, 2, 1]]);
    }
    #[test]
    fn test_rotated_box_areas_single_box() {
        let boxes = array![[1., 2., 3., 4., 100.]];
        let areas = rotated_box_areas(&boxes);
        assert_eq!(areas, array![12.]);
    }
}
