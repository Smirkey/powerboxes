use ndarray::{ArrayBase, Dim, OwnedRepr};
use numpy::{IntoPyArray, PyArray, PyArrayDyn};
use pyo3::prelude::*;

pub fn array_to_numpy<T: numpy::Element, D: ndarray::Dimension>(
    py: Python,
    array: ArrayBase<OwnedRepr<T>, D>,
) -> PyResult<&PyArray<T, D>> {
    let numpy_array = array.into_pyarray(py);

    return Ok(numpy_array);
}

pub fn numpy_to_array<T: numpy::Element, D: ndarray::Dimension>(
    numpy_array: &PyArray<T, D>,
) -> ArrayBase<OwnedRepr<T>, D> {
    let array = unsafe { numpy_array.as_array().to_owned() };

    return array;
}

pub fn preprocess_array(
    array: &PyArrayDyn<f64>,
) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, PyErr> {
    let array = numpy_to_array(array);

    let array_shape = array.shape();

    if array_shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have shape (N, 4)",
        ));
    } else {
        let num_boxes = array_shape[0];

        if num_boxes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Arrays must have shape (N, 4) with N > 0",
            ));
        }
    }

    let array = array
        .to_owned()
        .into_shape((array_shape[0], array_shape[1]))
        .unwrap();
    return Ok(array);
}
