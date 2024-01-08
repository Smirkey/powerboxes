mod utils;

use std::fmt::Debug;

use num_traits::{Bounded, Num, Signed, ToPrimitive};
use numpy::{PyArray1, PyArray2, PyArray3};
use powerboxesrs::{boxes, giou, iou, nms, tiou};
use pyo3::prelude::*;
use utils::{preprocess_array1, preprocess_array3, preprocess_boxes, preprocess_rotated_boxes};

#[pymodule]
fn _powerboxes(_py: Python, m: &PyModule) -> PyResult<()> {
    // TIoU
    m.add_function(wrap_pyfunction!(tiou_distance_f64, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_f32, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_i64, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_i32, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_i16, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_u64, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_u32, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_u16, m)?)?;
    m.add_function(wrap_pyfunction!(tiou_distance_u8, m)?)?;
    // IoU
    m.add_function(wrap_pyfunction!(iou_distance_f64, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_f32, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_i64, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_i32, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_i16, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_u64, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_u32, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_u16, m)?)?;
    m.add_function(wrap_pyfunction!(iou_distance_u8, m)?)?;
    // Parallel IoU
    m.add_function(wrap_pyfunction!(parallel_iou_distance_f64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_f32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_i64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_i32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_i16, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_u64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_u32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_u16, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance_u8, m)?)?;
    // GIoU
    m.add_function(wrap_pyfunction!(giou_distance_f64, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_f32, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_i64, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_i32, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_i16, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_u64, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_u32, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_u16, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance_u8, m)?)?;
    // Parallel GIoU
    m.add_function(wrap_pyfunction!(parallel_giou_distance_f64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_f32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_i64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_i32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_i16, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_u64, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_u32, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_u16, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance_u8, m)?)?;
    // Remove small boxes
    m.add_function(wrap_pyfunction!(remove_small_boxes_f64, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_f32, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_i64, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_i32, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_i16, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_u64, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_u32, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_u16, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes_u8, m)?)?;
    // Boxes areas
    m.add_function(wrap_pyfunction!(box_areas_f64, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_f32, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_i64, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_i32, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_i16, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_u64, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_u32, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_u16, m)?)?;
    m.add_function(wrap_pyfunction!(box_areas_u8, m)?)?;
    // Box convert
    m.add_function(wrap_pyfunction!(box_convert_f64, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_f32, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_i64, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_i32, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_i16, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_u64, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_u32, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_u16, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert_u8, m)?)?;
    // nms
    m.add_function(wrap_pyfunction!(nms_f64, m)?)?;
    m.add_function(wrap_pyfunction!(nms_f32, m)?)?;
    m.add_function(wrap_pyfunction!(nms_i64, m)?)?;
    m.add_function(wrap_pyfunction!(nms_i32, m)?)?;
    m.add_function(wrap_pyfunction!(nms_i16, m)?)?;
    m.add_function(wrap_pyfunction!(nms_u64, m)?)?;
    m.add_function(wrap_pyfunction!(nms_u32, m)?)?;
    m.add_function(wrap_pyfunction!(nms_u16, m)?)?;
    m.add_function(wrap_pyfunction!(nms_u8, m)?)?;
    // rtree nms
    m.add_function(wrap_pyfunction!(rtree_nms_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rtree_nms_f32, m)?)?;
    m.add_function(wrap_pyfunction!(rtree_nms_i64, m)?)?;
    m.add_function(wrap_pyfunction!(rtree_nms_i32, m)?)?;
    m.add_function(wrap_pyfunction!(rtree_nms_i16, m)?)?;
    // Masks to boxes
    m.add_function(wrap_pyfunction!(masks_to_boxes, m)?)?;
    // Rotated IoU
    m.add_function(wrap_pyfunction!(rotated_iou_distance, m)?)?;
    Ok(())
}
// Masks to boxes
#[pyfunction]
fn masks_to_boxes(_py: Python, masks: &PyArray3<bool>) -> PyResult<Py<PyArray2<usize>>> {
    let masks = preprocess_array3(masks);
    let boxes = boxes::masks_to_boxes(&masks);
    let boxes_as_numpy = utils::array_to_numpy(_py, boxes).unwrap();
    return Ok(boxes_as_numpy.to_owned());
}

// Rotated box IoU

#[pyfunction]
fn rotated_iou_distance(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_rotated_boxes(boxes1).unwrap();
    let boxes2 = preprocess_rotated_boxes(boxes2).unwrap();
    let iou = iou::rotated_iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}

// IoU
fn iou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = iou::iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn iou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
// Parallel IoU
fn parallel_iou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy + Sync + Send,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = iou::parallel_iou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_iou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
// TIoU
fn tiou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let tiou = tiou::tiou_distance(&boxes1, &boxes2);
    let tiou_as_numpy = utils::array_to_numpy(_py, tiou).unwrap();
    return Ok(tiou_as_numpy.to_owned());
}

#[pyfunction]
fn tiou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn tiou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(tiou_distance_generic(_py, boxes1, boxes2)?);
}
// GIoU
fn giou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn giou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(giou_distance_generic(_py, boxes1, boxes2)?);
}
// Parallel GIoU
fn parallel_giou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<f64>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_boxes(boxes1).unwrap();
    let boxes2 = preprocess_boxes(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_giou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(parallel_giou_distance_generic(_py, boxes1, boxes2)?);
}
// Remove small boxes
fn remove_small_boxes_generic<T>(
    _py: Python,
    boxes: &PyArray2<T>,
    min_size: f64,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_f64(
    _py: Python,
    boxes: &PyArray2<f64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_f32(
    _py: Python,
    boxes: &PyArray2<f32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<f32>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_i64(
    _py: Python,
    boxes: &PyArray2<i64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i64>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_i32(
    _py: Python,
    boxes: &PyArray2<i32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i32>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_i16(
    _py: Python,
    boxes: &PyArray2<i16>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i16>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_u64(
    _py: Python,
    boxes: &PyArray2<u64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u64>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_u32(
    _py: Python,
    boxes: &PyArray2<u32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u32>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_u16(
    _py: Python,
    boxes: &PyArray2<u16>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u16>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
#[pyfunction]
fn remove_small_boxes_u8(
    _py: Python,
    boxes: &PyArray2<u8>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u8>>> {
    return Ok(remove_small_boxes_generic(_py, boxes, min_size)?);
}
// Boxes areas
fn generic_box_areas<T>(_py: Python, boxes: &PyArray2<T>) -> PyResult<Py<PyArray1<f64>>>
where
    T: Num + numpy::Element + PartialOrd + ToPrimitive + Sync + Send + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let areas = boxes::box_areas(&boxes);
    let areas_as_numpy = utils::array_to_numpy(_py, areas).unwrap();
    return Ok(areas_as_numpy.to_owned());
}

#[pyfunction]
fn box_areas_f64(_py: Python, boxes: &PyArray2<f64>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_f32(_py: Python, boxes: &PyArray2<f32>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_i64(_py: Python, boxes: &PyArray2<i64>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_i32(_py: Python, boxes: &PyArray2<i32>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_i16(_py: Python, boxes: &PyArray2<i16>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_u64(_py: Python, boxes: &PyArray2<u64>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_u32(_py: Python, boxes: &PyArray2<u32>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_u16(_py: Python, boxes: &PyArray2<u16>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}
#[pyfunction]
fn box_areas_u8(_py: Python, boxes: &PyArray2<u8>) -> PyResult<Py<PyArray1<f64>>> {
    return Ok(generic_box_areas(_py, boxes)?);
}

fn box_convert_generic<T>(
    _py: Python,
    boxes: &PyArray2<T>,
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
    let converted_boxes = boxes::box_convert(&boxes, &in_fmt, &out_fmt);
    let converted_boxes_as_numpy = utils::array_to_numpy(_py, converted_boxes).unwrap();
    return Ok(converted_boxes_as_numpy.to_owned());
}

#[pyfunction]
fn box_convert_f64(
    _py: Python,
    boxes: &PyArray2<f64>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_f32(
    _py: Python,
    boxes: &PyArray2<f32>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<f32>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_i64(
    _py: Python,
    boxes: &PyArray2<i64>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<i64>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_i32(
    _py: Python,
    boxes: &PyArray2<i32>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<i32>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_i16(
    _py: Python,
    boxes: &PyArray2<i16>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<i16>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_u64(
    _py: Python,
    boxes: &PyArray2<u64>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<u64>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_u32(
    _py: Python,
    boxes: &PyArray2<u32>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<u32>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_u16(
    _py: Python,
    boxes: &PyArray2<u16>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<u16>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}
#[pyfunction]
fn box_convert_u8(
    _py: Python,
    boxes: &PyArray2<u8>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray2<u8>>> {
    return Ok(box_convert_generic(_py, boxes, in_fmt, out_fmt).unwrap());
}

// nms
fn nms_generic<T>(
    _py: Python,
    boxes: &PyArray2<T>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: Num + numpy::Element + PartialOrd + ToPrimitive + Copy,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let scores = preprocess_array1(scores);
    let keep = nms::nms(&boxes, &scores, iou_threshold, score_threshold);
    let keep_as_numpy = utils::array_to_numpy(_py, keep).unwrap();
    return Ok(keep_as_numpy.to_owned());
}
#[pyfunction]
fn nms_f64(
    _py: Python,
    boxes: &PyArray2<f64>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_f32(
    _py: Python,
    boxes: &PyArray2<f32>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_i64(
    _py: Python,
    boxes: &PyArray2<i64>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_i32(
    _py: Python,
    boxes: &PyArray2<i32>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_i16(
    _py: Python,
    boxes: &PyArray2<i16>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_u64(
    _py: Python,
    boxes: &PyArray2<u64>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_u32(
    _py: Python,
    boxes: &PyArray2<u32>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_u16(
    _py: Python,
    boxes: &PyArray2<u16>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn nms_u8(
    _py: Python,
    boxes: &PyArray2<u8>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}

// rtree nms
fn rtree_nms_generic<T>(
    _py: Python,
    boxes: &PyArray2<T>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>>
where
    T: Num
        + numpy::Element
        + PartialOrd
        + ToPrimitive
        + Copy
        + Signed
        + Bounded
        + Debug
        + Sync
        + Send,
{
    let boxes = preprocess_boxes(boxes).unwrap();
    let scores = preprocess_array1(scores);
    let keep = nms::rtree_nms(&boxes, &scores, iou_threshold, score_threshold);
    let keep_as_numpy = utils::array_to_numpy(_py, keep).unwrap();
    return Ok(keep_as_numpy.to_owned());
}
#[pyfunction]
fn rtree_nms_f64(
    _py: Python,
    boxes: &PyArray2<f64>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(rtree_nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn rtree_nms_f32(
    _py: Python,
    boxes: &PyArray2<f32>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(rtree_nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn rtree_nms_i64(
    _py: Python,
    boxes: &PyArray2<i64>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(rtree_nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn rtree_nms_i32(
    _py: Python,
    boxes: &PyArray2<i32>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(rtree_nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
#[pyfunction]
fn rtree_nms_i16(
    _py: Python,
    boxes: &PyArray2<i16>,
    scores: &PyArray1<f64>,
    iou_threshold: f64,
    score_threshold: f64,
) -> PyResult<Py<PyArray1<usize>>> {
    return Ok(rtree_nms_generic(
        _py,
        boxes,
        scores,
        iou_threshold,
        score_threshold,
    )?);
}
