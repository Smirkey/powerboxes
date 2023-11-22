#![crate_name = "powerboxesrs"]

//! Powerboxes is a package containing utility functions for transforming bounding boxes and computing metrics from them.
//! # Powerboxesrs
//!
//! `powerboxesrs` is a Rust package containing utility functions for transforming bounding boxes and computing metrics from them.
//!
//! ## Installation
//!
//! Add the following to your `Cargo.toml` file:
//!
//! ```toml
//! [dependencies]
//! powerboxesrs = "0.1.2"
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use ndarray::array;
//! use powerboxesrs::iou::iou_distance;
//!
//! let boxes1 = array![[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]];
//! let boxes2 = array![[0.5, 0.5, 1.5, 1.5], [2.5, 2.5, 3.5, 3.5]];
//! let iou = iou_distance(&boxes1, &boxes2);
//! assert_eq!(iou, array![[0.6086956521739131, 0.967741935483871],[0.967741935483871, 0.6086956521739131]]);
//! ```
//!
//! ## Features
//!
//! - Bounding box transformations
//! - Intersection over union (IoU) and generalized IoU (GIoU) metrics
//!

pub mod boxes;
pub mod giou;
pub mod iou;
mod utils;

use num_traits::{Num, ToPrimitive};
use numpy::PyArray2;
use pyo3::prelude::*;
use utils::{preprocess_array, GenericArray2};

#[pymodule]
fn powerboxesrs(_py: Python, m: &PyModule) -> PyResult<()> {
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
    m.add_function(wrap_pyfunction!(boxes_areas, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert, m)?)?;
    Ok(())
}

// IoU
fn iou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy,
{
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
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
) -> PyResult<Py<PyArray2<f32>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<i64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<i32>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<i16>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<u64>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<u32>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<u16>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn iou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<u8>>> {
    return Ok(iou_distance_generic(_py, boxes1, boxes2)?);
}
// Parallel IoU
fn parallel_iou_distance_generic<T>(
    _py: Python,
    boxes1: &PyArray2<T>,
    boxes2: &PyArray2<T>,
) -> PyResult<Py<PyArray2<T>>>
where
    T: Num + ToPrimitive + PartialOrd + numpy::Element + Copy + Sync + Send,
{
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
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
) -> PyResult<Py<PyArray2<f32>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<i64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<i32>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<i16>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<u64>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<u32>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<u16>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
#[pyfunction]
fn parallel_iou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<u8>>> {
    return Ok(parallel_iou_distance_generic(_py, boxes1, boxes2)?);
}
// GIoU
#[pyfunction]
fn giou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<i64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<i32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<i16>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<u64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<u32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<u16>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn giou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<u8>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
// Parallel GIOU
#[pyfunction]
fn parallel_giou_distance_f64(
    _py: Python,
    boxes1: &PyArray2<f64>,
    boxes2: &PyArray2<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_f32(
    _py: Python,
    boxes1: &PyArray2<f32>,
    boxes2: &PyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_i64(
    _py: Python,
    boxes1: &PyArray2<i64>,
    boxes2: &PyArray2<i64>,
) -> PyResult<Py<PyArray2<i64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_i32(
    _py: Python,
    boxes1: &PyArray2<i32>,
    boxes2: &PyArray2<i32>,
) -> PyResult<Py<PyArray2<i32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_i16(
    _py: Python,
    boxes1: &PyArray2<i16>,
    boxes2: &PyArray2<i16>,
) -> PyResult<Py<PyArray2<i16>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_u64(
    _py: Python,
    boxes1: &PyArray2<u64>,
    boxes2: &PyArray2<u64>,
) -> PyResult<Py<PyArray2<u64>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_u32(
    _py: Python,
    boxes1: &PyArray2<u32>,
    boxes2: &PyArray2<u32>,
) -> PyResult<Py<PyArray2<u32>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_u16(
    _py: Python,
    boxes1: &PyArray2<u16>,
    boxes2: &PyArray2<u16>,
) -> PyResult<Py<PyArray2<u16>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
#[pyfunction]
fn parallel_giou_distance_u8(
    _py: Python,
    boxes1: &PyArray2<u8>,
    boxes2: &PyArray2<u8>,
) -> PyResult<Py<PyArray2<u8>>> {
    let boxes1 = preprocess_array(boxes1).unwrap();
    let boxes2 = preprocess_array(boxes2).unwrap();
    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);
    let iou_as_numpy = utils::array_to_numpy(_py, iou).unwrap();
    return Ok(iou_as_numpy.to_owned());
}
// Remove small boxes
#[pyfunction]
fn remove_small_boxes_f64(
    _py: Python,
    boxes: &PyArray2<f64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_f32(
    _py: Python,
    boxes: &PyArray2<f32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<f32>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_i64(
    _py: Python,
    boxes: &PyArray2<i64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i64>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_i32(
    _py: Python,
    boxes: &PyArray2<i32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i32>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_i16(
    _py: Python,
    boxes: &PyArray2<i16>,
    min_size: f64,
) -> PyResult<Py<PyArray2<i16>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_u64(
    _py: Python,
    boxes: &PyArray2<u64>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u64>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_u32(
    _py: Python,
    boxes: &PyArray2<u32>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u32>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_u16(
    _py: Python,
    boxes: &PyArray2<u16>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u16>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}
#[pyfunction]
fn remove_small_boxes_u8(
    _py: Python,
    boxes: &PyArray2<u8>,
    min_size: f64,
) -> PyResult<Py<PyArray2<u8>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);
    let filtered_boxes_as_numpy = utils::array_to_numpy(_py, filtered_boxes).unwrap();
    return Ok(filtered_boxes_as_numpy.to_owned());
}

#[pyfunction]
fn boxes_areas(_py: Python, boxes: GenericArray2) -> PyResult<Py<pyo3::PyAny>> {
    fn compute_boxes_areas<T>(_py: Python, boxes: &PyArray2<T>) -> PyResult<Py<pyo3::PyAny>>
    where
        T: Num + numpy::Element + PartialOrd + ToPrimitive + Sync + Send + Copy,
    {
        let boxes = preprocess_array(boxes).unwrap();
        let areas = boxes::box_areas(&boxes);

        let areas_as_numpy = utils::array_to_numpy(_py, areas)
            .unwrap()
            .to_owned()
            .to_object(_py);

        return Ok(areas_as_numpy.to_owned());
    }
    match boxes {
        GenericArray2::U8(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::U16(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::U32(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::U64(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::I8(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::I16(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::I32(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::I64(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::F32(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
        GenericArray2::F64(boxes) => return Ok(compute_boxes_areas(_py, &boxes).unwrap()),
    }
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
    let boxes = preprocess_array(boxes).unwrap();
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
fn box_convert(
    _py: Python,
    boxes: GenericArray2,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<pyo3::PyAny>> {
    fn compute_convert_boxes<T>(
        _py: Python,
        boxes: &PyArray2<T>,
        in_fmt: &str,
        out_fmt: &str,
    ) -> PyResult<Py<pyo3::PyAny>>
    where
        T: Num + numpy::Element + PartialOrd + ToPrimitive + Sync + Send + Copy,
    {
        let boxes = preprocess_array(boxes).unwrap();

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

        let converted_boxes_as_numpy = utils::array_to_numpy(_py, converted_boxes)
            .unwrap()
            .to_object(_py);

        return Ok(converted_boxes_as_numpy);
    }
    match boxes {
        GenericArray2::U8(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::U16(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::U32(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::U64(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::I8(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::I16(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::I32(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::I64(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::F32(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
        GenericArray2::F64(boxes) => {
            return Ok(compute_convert_boxes(_py, &boxes, in_fmt, out_fmt).unwrap())
        }
    }
}
