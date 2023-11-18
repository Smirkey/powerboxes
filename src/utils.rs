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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayBase, IxDyn};
    use numpy::{PyArray, PyArrayDyn};

    #[test]
    fn test_array_to_numpy() {
        let data = vec![1., 2., 3., 4.];
        let array = ArrayBase::from_shape_vec((1, 4), data).unwrap();
        Python::with_gil(|py| {
            let result = array_to_numpy(py, array).unwrap();
            assert_eq!(result.readonly().shape(), &[1, 4]);
            assert_eq!(result.readonly().shape(), &[1, 4]);
        });
    }

    #[test]
    fn test_numpy_to_array() {
        Python::with_gil(|python| {
            let array = PyArray::<f64, _>::zeros(python, [2, 3], false);
            let result = numpy_to_array(&array);
            assert_eq!(result.shape(), &[2, 3]);
        });
    }

    #[test]
    fn test_preprocess_array() {
        Python::with_gil(|python| {
            let array = PyArrayDyn::<f64>::zeros(python, IxDyn(&[2, 4]), false);
            let result = preprocess_array(&array);
            assert!(result.is_ok());
            let unwrapped_result = result.unwrap();
            assert_eq!(unwrapped_result.shape(), &[2, 4]);
        });
    }
    #[test]
    fn test_preprocess_array_bad_shape() {
        Python::with_gil(|python| {
            let array = PyArrayDyn::<f64>::zeros(python, IxDyn(&[2, 3]), false);
            let result = preprocess_array(&array);
            assert!(result.is_err());
        });
    }
}
