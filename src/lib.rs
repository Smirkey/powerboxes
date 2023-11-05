pub mod boxes;
pub mod iou;
pub mod utils;

use ndarray::Dim;
use numpy::{PyArray, PyArrayDyn};
use pyo3::prelude::*;
use utils::preprocess_array;

/// A Python module implemented in Rust.
#[pymodule]
fn powerboxes(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(distance_box_iou, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_distance_box_iou, m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_boxes, m)?)?;
    m.add_function(wrap_pyfunction!(boxes_areas, m)?)?;
    m.add_function(wrap_pyfunction!(box_convert, m)?)?;
    Ok(())
}

#[pyfunction]
fn distance_box_iou(
    _py: Python,
    array1: &PyArrayDyn<f64>,
    array2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let array1 = utils::preprocess_array(array1).unwrap();
    let array2 = utils::preprocess_array(array2).unwrap();

    let iou = iou::distance_box_iou(&array1, &array2);

    let iou_as_numpy: Py<numpy::PyArray<f64, Dim<[usize; 2]>>> =
        utils::array_to_numpy(_py, iou).unwrap().to_owned();

    return Ok(iou_as_numpy.to_owned());
}

#[pyfunction]
fn parallel_distance_box_iou(
    _py: Python,
    array1: &PyArrayDyn<f64>,
    array2: &PyArrayDyn<f64>,
) -> PyResult<Py<PyArray<f64, Dim<[usize; 2]>>>> {
    let array1 = utils::preprocess_array(array1).unwrap();
    let array2 = utils::preprocess_array(array2).unwrap();

    let iou = iou::parallel_distance_box_iou(&array1, &array2);

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
