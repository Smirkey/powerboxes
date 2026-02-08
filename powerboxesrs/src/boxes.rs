#[cfg(feature = "ndarray")]
use ndarray::{Array1, Array2, ArrayView2, ArrayView3, ArrayViewMut2, Axis, Zip};
#[cfg(feature = "ndarray")]
use num_traits::real::Real;
use num_traits::{Num, ToPrimitive};

use crate::utils;

#[derive(Copy, Clone)]
pub enum BoxFormat {
    XYXY,
    XYWH,
    CXCYWH,
}

// ─── Slice-based core functions ───

/// Calculates areas for N boxes stored in a flat xyxy slice of length N*4.
/// Returns a Vec<f64> of length N.
#[inline]
pub fn box_areas_slice<N>(boxes: &[N], n: usize) -> Vec<f64>
where
    N: Num + PartialEq + ToPrimitive + Copy,
{
    let mut areas = vec![0.0f64; n];
    for i in 0..n {
        let (x1, y1, x2, y2) = utils::row4(boxes, i);
        let area = (x2 - x1) * (y2 - y1);
        areas[i] = area.to_f64().unwrap();
    }
    areas
}

/// Removes boxes with area < min_size.
/// Returns a flat Vec<N> containing the remaining boxes (row-major, 4 cols).
pub fn remove_small_boxes_slice<N>(boxes: &[N], n: usize, min_size: f64) -> Vec<N>
where
    N: Num + PartialEq + Clone + PartialOrd + ToPrimitive + Copy,
{
    let areas = box_areas_slice(boxes, n);
    let mut result = Vec::new();
    for i in 0..n {
        if areas[i] >= min_size {
            let base = i * 4;
            result.extend_from_slice(&boxes[base..base + 4]);
        }
    }
    result
}

/// Converts boxes in-place from one format to another.
/// `boxes` is a mutable flat slice of length N*4.
pub fn box_convert_inplace_slice<N>(boxes: &mut [N], n: usize, in_fmt: BoxFormat, out_fmt: BoxFormat)
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Copy,
{
    let two = N::one() + N::one();
    for i in 0..n {
        let base = i * 4;
        match (in_fmt, out_fmt) {
            (BoxFormat::XYXY, BoxFormat::XYWH) => {
                boxes[base + 2] = boxes[base + 2] - boxes[base];
                boxes[base + 3] = boxes[base + 3] - boxes[base + 1];
            }
            (BoxFormat::XYXY, BoxFormat::CXCYWH) => {
                let x1 = boxes[base];
                let y1 = boxes[base + 1];
                let x2 = boxes[base + 2];
                let y2 = boxes[base + 3];
                boxes[base] = (x1 + x2) / two;
                boxes[base + 1] = (y1 + y2) / two;
                boxes[base + 2] = x2 - x1;
                boxes[base + 3] = y2 - y1;
            }
            (BoxFormat::XYWH, BoxFormat::XYXY) => {
                boxes[base + 2] = boxes[base] + boxes[base + 2];
                boxes[base + 3] = boxes[base + 1] + boxes[base + 3];
            }
            (BoxFormat::XYWH, BoxFormat::CXCYWH) => {
                let w = boxes[base + 2];
                let h = boxes[base + 3];
                boxes[base] = boxes[base] + w / two;
                boxes[base + 1] = boxes[base + 1] + h / two;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYXY) => {
                let cx = boxes[base];
                let cy = boxes[base + 1];
                let wd2 = boxes[base + 2] / two;
                let hd2 = boxes[base + 3] / two;
                boxes[base] = cx - wd2;
                boxes[base + 1] = cy - hd2;
                boxes[base + 2] = cx + wd2;
                boxes[base + 3] = cy + hd2;
            }
            (BoxFormat::CXCYWH, BoxFormat::XYWH) => {
                let w = boxes[base + 2];
                let h = boxes[base + 3];
                boxes[base] = boxes[base] - w / two;
                boxes[base + 1] = boxes[base + 1] - h / two;
            }
            (BoxFormat::XYXY, BoxFormat::XYXY) => (),
            (BoxFormat::XYWH, BoxFormat::XYWH) => (),
            (BoxFormat::CXCYWH, BoxFormat::CXCYWH) => (),
        }
    }
}

/// Converts boxes from one format to another, returning a new Vec.
pub fn box_convert_slice<N>(boxes: &[N], n: usize, in_fmt: BoxFormat, out_fmt: BoxFormat) -> Vec<N>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Copy,
{
    let mut result = boxes.to_vec();
    box_convert_inplace_slice(&mut result, n, in_fmt, out_fmt);
    result
}

/// Compute bounding boxes from masks.
/// `masks` is a flat slice of length N * H * W (row-major, N masks of H rows and W cols).
pub fn masks_to_boxes_slice(masks: &[bool], num_masks: usize, height: usize, width: usize) -> Vec<usize> {
    let mut result = vec![0usize; num_masks * 4];
    for i in 0..num_masks {
        let mask_offset = i * height * width;
        let mut x1 = width;
        let mut y1 = height;
        let mut x2 = 0usize;
        let mut y2 = 0usize;
        for y in 0..height {
            for x in 0..width {
                if masks[mask_offset + y * width + x] {
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
            }
        }
        let base = i * 4;
        result[base] = x1;
        result[base + 1] = y1;
        result[base + 2] = x2;
        result[base + 3] = y2;
    }
    result
}

/// Calculates areas of rotated boxes in cxcywha format.
/// `boxes` is a flat slice of length N*5.
pub fn rotated_box_areas_slice(boxes: &[f64], n: usize) -> Vec<f64> {
    let mut areas = vec![0.0f64; n];
    for i in 0..n {
        let base = i * 5;
        areas[i] = boxes[base + 2] * boxes[base + 3];
    }
    areas
}

// ─── ndarray wrappers ───

#[cfg(feature = "ndarray")]
pub fn box_areas<'a, N, BA>(boxes: BA) -> Array1<f64>
where
    N: Num + PartialEq + ToPrimitive + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let boxes = boxes.into();
    let n = boxes.nrows();
    let slice = boxes.as_slice().expect("boxes must be contiguous");
    Array1::from(box_areas_slice(slice, n))
}

#[cfg(feature = "ndarray")]
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

    areas
}

#[cfg(feature = "ndarray")]
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
    boxes.select(Axis(0), &keep)
}

#[cfg(feature = "ndarray")]
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

#[cfg(feature = "ndarray")]
pub fn box_convert<'a, N, BA>(boxes: BA, in_fmt: BoxFormat, out_fmt: BoxFormat) -> Array2<N>
where
    N: Num + PartialEq + PartialOrd + ToPrimitive + Clone + Copy + 'a,
    BA: Into<ArrayView2<'a, N>>,
{
    let mut converted_boxes = boxes.into().to_owned();
    box_convert_inplace(&mut converted_boxes, in_fmt, out_fmt);
    converted_boxes
}

#[cfg(feature = "ndarray")]
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
    converted_boxes
}

#[cfg(feature = "ndarray")]
pub fn masks_to_boxes<'a, MA>(masks: MA) -> Array2<usize>
where
    MA: Into<ArrayView3<'a, bool>>,
{
    let masks = masks.into();
    let num_masks = masks.shape()[0];
    let height = masks.shape()[1];
    let width = masks.shape()[2];
    let flat = if let Some(s) = masks.as_slice() {
        masks_to_boxes_slice(s, num_masks, height, width)
    } else {
        // fallback for non-contiguous
        let owned: Vec<bool> = masks.iter().copied().collect();
        masks_to_boxes_slice(&owned, num_masks, height, width)
    };
    Array2::from_shape_vec((num_masks, 4), flat).unwrap()
}

#[cfg(feature = "ndarray")]
pub fn rotated_box_areas<'a, BA>(boxes: BA) -> Array1<f64>
where
    BA: Into<ArrayView2<'a, f64>>,
{
    let boxes = boxes.into();
    let n = boxes.nrows();
    let slice = boxes.as_slice().expect("boxes must be contiguous");
    Array1::from(rotated_box_areas_slice(slice, n))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "ndarray")]
    use ndarray::{arr2, arr3, array, Array3};

    #[test]
    fn test_box_areas_slice() {
        let boxes = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 10.0, 10.0];
        let areas = box_areas_slice(&boxes, 2);
        assert_eq!(areas, vec![4.0, 100.0]);
    }

    #[test]
    fn test_box_convert_slice_xyxy_to_cxcywh() {
        let boxes = vec![10.0, 20.0, 30.0, 40.0];
        let result = box_convert_slice(&boxes, 1, BoxFormat::XYXY, BoxFormat::CXCYWH);
        assert_eq!(result, vec![20.0, 30.0, 20.0, 20.0]);
    }

    #[test]
    fn test_remove_small_boxes_slice() {
        let boxes = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 10.0, 10.0];
        let result = remove_small_boxes_slice(&boxes, 2, 10.0);
        assert_eq!(result, vec![0.0, 0.0, 10.0, 10.0]);
    }

    #[test]
    fn test_masks_to_boxes_slice() {
        let masks = vec![
            // mask 0: top row all true
            true, true, true, false, false, false,
            // mask 1: bottom row all true
            false, false, false, true, true, true,
        ];
        let boxes = masks_to_boxes_slice(&masks, 2, 2, 3);
        assert_eq!(boxes, vec![0, 0, 2, 0, 0, 1, 2, 1]);
    }

    #[test]
    fn test_rotated_box_areas_slice() {
        let boxes = vec![1.0, 2.0, 3.0, 4.0, 100.0];
        let areas = rotated_box_areas_slice(&boxes, 1);
        assert_eq!(areas, vec![12.0]);
    }

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_box_areas_single_box() {
        let boxes = array![[1., 2., 3., 4.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4.]);
        assert_eq!(parallel_areas, areas);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_box_areas_multiple_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4., 100.]);
        assert_eq!(parallel_areas, areas);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_box_areas_zero_area() {
        let boxes = array![[1., 2., 1., 2.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![0.]);
        assert_eq!(parallel_areas, areas);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_box_areas_negative_coordinates() {
        let boxes = array![[-1., -1., 1., 1.]];
        let areas = box_areas(&boxes);
        let parallel_areas = parallel_box_areas(&boxes);
        assert_eq!(areas, array![4.]);
        assert_eq!(parallel_areas, areas);
    }

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_remove_small_boxes() {
        let boxes = array![[1., 2., 3., 4.], [0., 0., 10., 10.]];
        let min_size = 10.;
        let filtered_boxes = remove_small_boxes(&boxes, min_size);
        assert_eq!(filtered_boxes, array![[0., 0., 10., 10.]]);
    }

    #[cfg(feature = "ndarray")]
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

    #[cfg(feature = "ndarray")]
    #[test]
    fn test_rotated_box_areas_single_box() {
        let boxes = array![[1., 2., 3., 4., 100.]];
        let areas = rotated_box_areas(&boxes);
        assert_eq!(areas, array![12.]);
    }
}
