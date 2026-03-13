mod utils;

use std::fmt::Debug;

use ndarray::Array1;
use num_traits::{Bounded, Float, Num, Signed, ToPrimitive};
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods};
use powerboxesrs::{boxes, ciou, diou, draw, giou, iou, nms, tiou};
use pyo3::prelude::*;
use utils::{preprocess_array1, preprocess_array3, preprocess_boxes, preprocess_rotated_boxes};

// ---------------------------------------------------------------------------
// Type-dispatch helpers
// ---------------------------------------------------------------------------

/// Call `$mac!(prefix, generic_fn, $T, $suffix)` for all 9 numeric types.
macro_rules! for_each_numeric_type {
    ($mac:ident, $prefix:ident, $generic:ident) => {
        $mac!($prefix, $generic, f64, f64);
        $mac!($prefix, $generic, f32, f32);
        $mac!($prefix, $generic, i64, i64);
        $mac!($prefix, $generic, i32, i32);
        $mac!($prefix, $generic, i16, i16);
        $mac!($prefix, $generic, u64, u64);
        $mac!($prefix, $generic, u32, u32);
        $mac!($prefix, $generic, u16, u16);
        $mac!($prefix, $generic, u8, u8);
    };
    ($mac:ident, $prefix:ident, $generic:ident, floats) => {
        $mac!($prefix, $generic, f64, f64);
        $mac!($prefix, $generic, f32, f32);
    };
    ($mac:ident, $prefix:ident, $generic:ident, signed) => {
        $mac!($prefix, $generic, f64, f64);
        $mac!($prefix, $generic, f32, f32);
        $mac!($prefix, $generic, i64, i64);
        $mac!($prefix, $generic, i32, i32);
        $mac!($prefix, $generic, i16, i16);
    };
    ($mac:ident, $prefix:ident, $generic:ident, integers) => {
        $mac!($prefix, $generic, i64, i64);
        $mac!($prefix, $generic, i32, i32);
        $mac!($prefix, $generic, i16, i16);
        $mac!($prefix, $generic, u64, u64);
        $mac!($prefix, $generic, u32, u32);
        $mac!($prefix, $generic, u16, u16);
        $mac!($prefix, $generic, u8, u8);
    };
}

// ---------------------------------------------------------------------------
// Function-shape macros
// ---------------------------------------------------------------------------

/// Generate a typed `#[pyfunction]` for `(py, boxes1, boxes2) -> Array2<f64>`.
macro_rules! impl_distance2_fn {
    ($prefix:ident, $generic:ident, $T:ty, $suffix:ident) => {
        ::paste::paste! {
            #[pyfunction]
            fn [<$prefix _ $suffix>](
                py: Python,
                boxes1: &Bound<'_, PyArray2<$T>>,
                boxes2: &Bound<'_, PyArray2<$T>>,
            ) -> PyResult<Py<PyArray2<f64>>> {
                $generic(py, boxes1, boxes2)
            }
        }
    };
}

/// Generate a typed `#[pyfunction]` for `(py, boxes) -> Array1<f64>`.
macro_rules! impl_unary_f64_fn {
    ($prefix:ident, $generic:ident, $T:ty, $suffix:ident) => {
        ::paste::paste! {
            #[pyfunction]
            fn [<$prefix _ $suffix>](
                py: Python,
                boxes: &Bound<'_, PyArray2<$T>>,
            ) -> PyResult<Py<PyArray1<f64>>> {
                $generic(py, boxes)
            }
        }
    };
}

/// Generate a typed `#[pyfunction]` for `(py, boxes, min_size: f64) -> Array2<T>`.
macro_rules! impl_filter_fn {
    ($prefix:ident, $generic:ident, $T:ty, $suffix:ident) => {
        ::paste::paste! {
            #[pyfunction]
            fn [<$prefix _ $suffix>](
                py: Python,
                boxes: &Bound<'_, PyArray2<$T>>,
                min_size: f64,
            ) -> PyResult<Py<PyArray2<$T>>> {
                $generic(py, boxes, min_size)
            }
        }
    };
}

/// Generate a typed `#[pyfunction]` for `(py, boxes, in_fmt, out_fmt) -> Array2<T>`.
macro_rules! impl_convert_fn {
    ($prefix:ident, $generic:ident, $T:ty, $suffix:ident) => {
        ::paste::paste! {
            #[pyfunction]
            fn [<$prefix _ $suffix>](
                py: Python,
                boxes: &Bound<'_, PyArray2<$T>>,
                in_fmt: &str,
                out_fmt: &str,
            ) -> PyResult<Py<PyArray2<$T>>> {
                $generic(py, boxes, in_fmt, out_fmt)
            }
        }
    };
}

/// Generate a typed `#[pyfunction]` for `(py, boxes, scores, iou_threshold, score_threshold) -> Array1<usize>`.
macro_rules! impl_nms_fn {
    ($prefix:ident, $generic:ident, $T:ty, $suffix:ident) => {
        ::paste::paste! {
            #[pyfunction]
            fn [<$prefix _ $suffix>](
                py: Python,
                boxes: &Bound<'_, PyArray2<$T>>,
                scores: &Bound<'_, PyArray1<f64>>,
                iou_threshold: f64,
                score_threshold: f64,
            ) -> PyResult<Py<PyArray1<usize>>> {
                $generic(py, boxes, scores, iou_threshold, score_threshold)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register all 9 typed variants of a function with the Python module.
/// Usage: `register_typed!(m, prefix, [f64, f32, ...])`
macro_rules! register_typed {
    ($m:ident, $prefix:ident, [$( $suffix:ident ),+]) => {
        $(
            ::paste::paste! {
                $m.add_function(wrap_pyfunction!([<$prefix _ $suffix>], $m)?)?;
            }
        )+
    };
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _powerboxes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TIoU
    register_typed!(
        m,
        tiou_distance,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // IoU
    register_typed!(
        m,
        iou_distance,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // Parallel IoU
    register_typed!(
        m,
        parallel_iou_distance,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // CIoU (float only)
    register_typed!(m, ciou_distance, [f64, f32]);
    // DIoU (float only)
    register_typed!(m, diou_distance, [f64, f32]);
    // GIoU
    register_typed!(
        m,
        giou_distance,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // Parallel GIoU
    register_typed!(
        m,
        parallel_giou_distance,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // Remove small boxes
    register_typed!(
        m,
        remove_small_boxes,
        [f64, f32, i64, i32, i16, u64, u32, u16, u8]
    );
    // Box areas
    register_typed!(m, box_areas, [f64, f32, i64, i32, i16, u64, u32, u16, u8]);
    // Box convert
    register_typed!(m, box_convert, [f64, f32, i64, i32, i16, u64, u32, u16, u8]);
    // NMS
    register_typed!(m, nms, [f64, f32, i64, i32, i16, u64, u32, u16, u8]);
    // Rotated NMS
    register_typed!(m, rotated_nms, [f64, f32, i64, i32, i16, u64, u32, u16, u8]);
    // Rtree NMS (signed + float only)
    register_typed!(m, rtree_nms, [f64, f32, i64, i32, i16]);
    // Rtree Rotated NMS (signed + float only)
    register_typed!(m, rtree_rotated_nms, [f64, f32, i64, i32, i16]);
    // Masks to boxes
    m.add_function(wrap_pyfunction!(masks_to_boxes, m)?)?;
    // Rotated IoU
    m.add_function(wrap_pyfunction!(rotated_iou_distance, m)?)?;
    // Parallel Rotated IoU
    m.add_function(wrap_pyfunction!(parallel_rotated_iou_distance, m)?)?;
    // Rotated GIoU
    m.add_function(wrap_pyfunction!(rotated_giou_distance, m)?)?;
    // Parallel Rotated GIoU
    m.add_function(wrap_pyfunction!(parallel_rotated_giou_distance, m)?)?;
    // Rotated TIoU
    m.add_function(wrap_pyfunction!(rotated_tiou_distance, m)?)?;
    // Parallel Rotated TIoU
    m.add_function(wrap_pyfunction!(parallel_rotated_tiou_distance, m)?)?;
    // Draw
    m.add_function(wrap_pyfunction!(draw_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(draw_rotated_boxes, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// One-off functions (rotated/f64-only, masks, draw)
// ---------------------------------------------------------------------------

#[pyfunction]
fn masks_to_boxes(py: Python, masks: &Bound<'_, PyArray3<bool>>) -> PyResult<Py<PyArray2<usize>>> {
    let masks = preprocess_array3(masks);
    let boxes = boxes::masks_to_boxes(masks);
    let boxes_as_numpy = utils::array_to_numpy(py, boxes).unwrap();
    Ok(boxes_as_numpy.unbind())
}

#[pyfunction]
fn rotated_iou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = iou::rotated_iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
fn parallel_rotated_iou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = iou::parallel_rotated_iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
fn rotated_giou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = giou::rotated_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
fn parallel_rotated_giou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = giou::parallel_rotated_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
fn rotated_tiou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = tiou::rotated_tiou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
fn parallel_rotated_tiou_distance(
    py: Python,
    boxes1: &Bound<'_, PyArray2<f64>>,
    boxes2: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = tiou::parallel_rotated_tiou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}

#[pyfunction]
#[pyo3(signature = (image, boxes, colors=None, thickness=2, filled=false, opacity=1.0))]
fn draw_boxes(
    py: Python,
    image: &Bound<'_, PyArray3<u8>>,
    boxes: &Bound<'_, PyArray2<f64>>,
    colors: Option<&Bound<'_, PyArray2<u8>>>,
    thickness: usize,
    filled: bool,
    opacity: f64,
) -> PyResult<Py<PyArray3<u8>>> {
    let image_array = preprocess_array3(image);
    let image_shape = image_array.shape();
    if image_shape[0] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must have shape (3, H, W)",
        ));
    }
    let height = image_shape[1];
    let width = image_shape[2];

    let boxes_array = unsafe { boxes.as_array() };
    let boxes_shape = boxes_array.shape();
    if boxes_shape.len() != 2 || (boxes_shape[0] > 0 && boxes_shape[1] != 4) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boxes must have shape (N, 4)",
        ));
    }
    let num_boxes = boxes_shape[0];

    let image_slice: Vec<u8> = image_array.iter().copied().collect();
    let boxes_slice: Vec<f64> = boxes_array.iter().copied().collect();

    let colors_vec: Option<Vec<u8>> = colors
        .map(|c| {
            let c_array = unsafe { c.as_array() };
            let c_shape = c_array.shape();
            if c_shape.len() != 2 || c_shape[0] != num_boxes || c_shape[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "colors must have shape (N, 3)",
                ));
            }
            Ok(c_array.iter().copied().collect())
        })
        .transpose()?;

    if !(0.0..=1.0).contains(&opacity) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opacity must be between 0.0 and 1.0",
        ));
    }

    let result = draw::draw_boxes_slice(
        &image_slice,
        height,
        width,
        &boxes_slice,
        num_boxes,
        colors_vec.as_deref(),
        draw::DrawOptions {
            thickness,
            filled,
            opacity,
        },
    );

    let result_array = ndarray::Array3::from_shape_vec((3, height, width), result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let numpy_array = utils::array_to_numpy(py, result_array)?;
    Ok(numpy_array.unbind())
}

#[pyfunction]
#[pyo3(signature = (image, boxes, colors=None, thickness=2, filled=false, opacity=1.0))]
fn draw_rotated_boxes(
    py: Python,
    image: &Bound<'_, PyArray3<u8>>,
    boxes: &Bound<'_, PyArray2<f64>>,
    colors: Option<&Bound<'_, PyArray2<u8>>>,
    thickness: usize,
    filled: bool,
    opacity: f64,
) -> PyResult<Py<PyArray3<u8>>> {
    let image_array = preprocess_array3(image);
    let image_shape = image_array.shape();
    if image_shape[0] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "image must have shape (3, H, W)",
        ));
    }
    let height = image_shape[1];
    let width = image_shape[2];

    let boxes_array = unsafe { boxes.as_array() };
    let boxes_shape = boxes_array.shape();
    if boxes_shape.len() != 2 || (boxes_shape[0] > 0 && boxes_shape[1] != 5) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "boxes must have shape (N, 5)",
        ));
    }
    let num_boxes = boxes_shape[0];

    let colors_vec: Option<Vec<u8>> = colors
        .map(|c| {
            let c_array = unsafe { c.as_array() };
            let c_shape = c_array.shape();
            if c_shape.len() != 2 || c_shape[0] != num_boxes || c_shape[1] != 3 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "colors must have shape (N, 3)",
                ));
            }
            Ok(c_array.iter().copied().collect())
        })
        .transpose()?;

    if !(0.0..=1.0).contains(&opacity) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "opacity must be between 0.0 and 1.0",
        ));
    }

    let image_slice: Vec<u8> = image_array.iter().copied().collect();
    let boxes_slice: Vec<f64> = boxes_array.iter().copied().collect();
    let result = draw::draw_rotated_boxes_slice(
        &image_slice,
        height,
        width,
        &boxes_slice,
        num_boxes,
        colors_vec.as_deref(),
        draw::DrawOptions {
            thickness,
            filled,
            opacity,
        },
    );

    let result_array = ndarray::Array3::from_shape_vec((3, height, width), result)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let numpy_array = utils::array_to_numpy(py, result_array)?;
    Ok(numpy_array.unbind())
}

// ---------------------------------------------------------------------------
// Generic implementations + typed wrappers via macros
// ---------------------------------------------------------------------------

// CIoU
fn ciou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + Float + numpy::Element,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = ciou::ciou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_distance2_fn,
    ciou_distance,
    ciou_distance_generic,
    floats
);

// DIoU
fn diou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + Float + numpy::Element,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = diou::diou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_distance2_fn,
    diou_distance,
    diou_distance_generic,
    floats
);

// IoU
fn iou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = iou::iou_distance(boxes1, boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(impl_distance2_fn, iou_distance, iou_distance_generic);

// Parallel IoU
fn parallel_iou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy + Sync + Send,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = iou::parallel_iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_distance2_fn,
    parallel_iou_distance,
    parallel_iou_distance_generic
);

// TIoU
fn tiou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let tiou = tiou::tiou_distance(&boxes1, &boxes2);
    let tiou_as_numpy = utils::array_to_numpy(py, tiou).unwrap();
    Ok(tiou_as_numpy.unbind())
}
for_each_numeric_type!(impl_distance2_fn, tiou_distance, tiou_distance_generic);

// GIoU
fn giou_distance_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(impl_distance2_fn, giou_distance, giou_distance_generic);

// Parallel GIoU — float types use the parallel implementation
fn parallel_giou_distance_float_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy + Sync + Send + Float,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_distance2_fn,
    parallel_giou_distance,
    parallel_giou_distance_float_generic,
    floats
);

// Parallel GIoU — integer types fall back to sequential (parallel requires Float/Real)
fn parallel_giou_distance_int_generic<T>(
    py: Python,
    boxes1: &Bound<'_, PyArray2<T>>,
    boxes2: &Bound<'_, PyArray2<T>>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(py, iou).unwrap();
    Ok(iou_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_distance2_fn,
    parallel_giou_distance,
    parallel_giou_distance_int_generic,
    integers
);

// Remove small boxes
fn remove_small_boxes_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    min_size: f64,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(py, filtered_boxes).unwrap();
    Ok(filtered_boxes_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_filter_fn,
    remove_small_boxes,
    remove_small_boxes_generic
);

// Box areas
fn generic_box_areas<T>(py: Python, boxes: &Bound<'_, PyArray2<T>>) -> PyResult<Py<PyArray1<f64>>>
where
    T: Num + numpy::Element + PartialOrd + ToPrimitive + Sync + Send + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let areas = boxes::box_areas(boxes);
    let areas_as_numpy = utils::array_to_numpy(py, areas).unwrap();
    Ok(areas_as_numpy.unbind())
}
for_each_numeric_type!(impl_unary_f64_fn, box_areas, generic_box_areas);

// Box convert
fn box_convert_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Num + numpy::Element + PartialOrd + ToPrimitive + Sync + Send + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let in_fmt = match in_fmt {
        "xyxy" => boxes::BoxFormat::XYXY,
        "xywh" => boxes::BoxFormat::XYWH,
        "cxcywh" => boxes::BoxFormat::CXCYWH,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid input format",
            ))
        }
    };
    let out_fmt = match out_fmt {
        "xyxy" => boxes::BoxFormat::XYXY,
        "xywh" => boxes::BoxFormat::XYWH,
        "cxcywh" => boxes::BoxFormat::CXCYWH,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid output format",
            ))
        }
    };
    let converted_boxes = boxes::box_convert(boxes, in_fmt, out_fmt);
    let converted_boxes_as_numpy = utils::array_to_numpy(py, converted_boxes).unwrap();
    Ok(converted_boxes_as_numpy.unbind())
}
for_each_numeric_type!(impl_convert_fn, box_convert, box_convert_generic);

// NMS
fn nms_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    scores: &Bound<'_, PyArray1<f64>>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: numpy::Element + Num + PartialEq + PartialOrd + ToPrimitive + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let scores = preprocess_array1(scores);
    let keep = nms::nms(boxes, scores, iou_threshold, score_threshold);
    let keep_as_ndarray = Array1::from(keep);
    let keep_as_numpy = utils::array_to_numpy(py, keep_as_ndarray).unwrap();
    Ok(keep_as_numpy.unbind())
}
for_each_numeric_type!(impl_nms_fn, nms, nms_generic);

// Rotated NMS
fn rotated_nms_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    scores: &Bound<'_, PyArray1<f64>>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: numpy::Element + Num + PartialEq + PartialOrd + ToPrimitive + Copy,
{
    let boxes = preprocess_rotated_boxes(boxes)?;
    let scores = preprocess_array1(scores);
    let keep = nms::rotated_nms(boxes, scores, iou_threshold, score_threshold);
    let keep_as_ndarray = Array1::from(keep);
    let keep_as_numpy = utils::array_to_numpy(py, keep_as_ndarray)?;
    Ok(keep_as_numpy.unbind())
}
for_each_numeric_type!(impl_nms_fn, rotated_nms, rotated_nms_generic);

// Rtree NMS (signed + float types only)
fn rtree_nms_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    scores: &Bound<'_, PyArray1<f64>>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: numpy::Element
        + Num
        + Signed
        + Bounded
        + Debug
        + PartialEq
        + PartialOrd
        + ToPrimitive
        + Copy
        + Sync
        + Send,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let scores = preprocess_array1(scores);
    let keep = nms::rtree_nms(boxes, scores, iou_threshold, score_threshold);
    let keep_as_ndarray = Array1::from(keep);
    let keep_as_numpy = utils::array_to_numpy(py, keep_as_ndarray).unwrap();
    Ok(keep_as_numpy.unbind())
}
for_each_numeric_type!(impl_nms_fn, rtree_nms, rtree_nms_generic, signed);

// Rtree Rotated NMS (signed + float types only)
fn rtree_rotated_nms_generic<T>(
    py: Python,
    boxes: &Bound<'_, PyArray2<T>>,
    scores: &Bound<'_, PyArray1<f64>>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: numpy::Element
        + Num
        + Signed
        + Bounded
        + Debug
        + PartialEq
        + PartialOrd
        + ToPrimitive
        + Copy
        + Sync
        + Send,
{
    let boxes = preprocess_rotated_boxes(boxes).unwrap();
    let scores = preprocess_array1(scores);
    let keep = nms::rtree_rotated_nms(boxes, scores, iou_threshold, score_threshold);
    let keep_as_ndarray = Array1::from(keep);
    let keep_as_numpy = utils::array_to_numpy(py, keep_as_ndarray).unwrap();
    Ok(keep_as_numpy.unbind())
}
for_each_numeric_type!(
    impl_nms_fn,
    rtree_rotated_nms,
    rtree_rotated_nms_generic,
    signed
);
