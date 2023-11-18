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
//! powerboxesrs = "0.1.1"
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

use ndarray::Dim;
use numpy::{PyArray, PyArrayDyn};
use pyo3::prelude::*;
use utils::preprocess_array;

#[pymodule]
fn powerboxesrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(iou_distance, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_iou_distance, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(boxes_areas, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert, m)?)?;
    m.add_function(wrap_pyfunction!(giou_distance, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_giou_distance, m)?)?;
    Ok(())
}

#[pyfunction]
fn giou_distance(
    _py: Python,
    boxes1: &PyArrayDyn<f64>,
    boxes2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let boxes1 = utils::preprocess_array(boxes1).unwrap();
    let boxes2 = utils::preprocess_array(boxes2).unwrap();

    let iou = giou::giou_distance(&boxes1, &boxes2);

    let iou_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, iou).unwrap().to_owned();

    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn parallel_giou_distance(
    _py: Python,
    boxes1: &PyArrayDyn<f64>,
    boxes2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let boxes1 = utils::preprocess_array(boxes1).unwrap();
    let boxes2 = utils::preprocess_array(boxes2).unwrap();

    let iou = giou::parallel_giou_distance(&boxes1, &boxes2);

    let iou_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, iou).unwrap().to_owned();

    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn iou_distance(
    _py: Python,
    boxes1: &PyArrayDyn<f64>,
    boxes2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let boxes1 = utils::preprocess_array(boxes1).unwrap();
    let boxes2 = utils::preprocess_array(boxes2).unwrap();

    let iou = iou::iou_distance(&boxes1, &boxes2);

    let iou_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, iou).unwrap().to_owned();

    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn parallel_iou_distance(
    _py: Python,
    boxes1: &PyArrayDyn<f64>,
    boxes2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let boxes1 = utils::preprocess_array(boxes1).unwrap();
    let boxes2 = utils::preprocess_array(boxes2).unwrap();

    let iou = iou::parallel_iou_distance(&boxes1, &boxes2);

    let iou_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, iou).unwrap().to_owned();

    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn remove_small_boxes(
    _py: Python,
    boxes: &PyArrayDyn<f64>,
    min_size: f64,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let filtered_boxes = boxes::remove_small_boxes(&boxes, min_size);

    let filtered_boxes_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, filtered_boxes)
            .unwrap()
            .to_owned();

    return Ok(filtered_boxes_as_numpy.to_owned());
}

#[pyfunction]
fn boxes_areas(
    _py: Python,
    boxes: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 1]>>>> {
    let boxes = preprocess_array(boxes).unwrap();
    let areas = boxes::box_areas(&boxes);

    let areas_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 1]>>> =
        utils::array_to_numpy(_py, areas).unwrap().to_owned();

    return Ok(areas_as_numpy.to_owned());
}

#[pyfunction]
fn box_convert(
    _py: Python,
    boxes: &PyArrayDyn<f64>,
    in_fmt: &str,
    out_fmt: &str,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
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

    let converted_boxes_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, converted_boxes)
            .unwrap()
            .to_owned();

    return Ok(converted_boxes_as_numpy.to_owned());
}
