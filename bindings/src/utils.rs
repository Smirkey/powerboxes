use ndarray::{ArrayBase, Dim, ViewRepr};
use num_traits::Num;
use numpy::{PyArray, PyArray1, PyArray2, PyArray3, ToPyArray};
use pyo3::prelude::*;

/// Converts a 2-dimensional Rust ndarray to a NumPy array.
///
/// # Arguments
///
/// * `py` - The Python interpreter context.
/// * `array` - The 2-dimensional Rust ndarray to convert.
///
/// # Returns
///
/// A reference to the converted NumPy array.
///
/// # Example
///
/// ```rust
/// let py = Python::acquire_gil().python();
/// let array_2d: Array2<f64> = Array2::ones((3, 3));
/// let numpy_array_2d = array2_to_numpy(py, array_2d).unwrap();
/// ```
pub fn array_to_numpy<'a, T, D>(
    py: Python<'a>,
    array: ArrayBase<ViewRepr<&T>, D>,
) -> PyResult<&'a PyArray<T, D>>
where
    T: numpy::Element + 'a,
    D: ndarray::Dimension,
{
    let numpy_array = array.to_owned().to_pyarray(py);

    return Ok(numpy_array);
}

pub fn preprocess_boxes<N>(
    array: &PyArray2<N>,
) -> Result<ArrayBase<ViewRepr<&N>, Dim<[usize; 2]>>, PyErr>
where
    N: numpy::Element,
{
    let array = unsafe { array.as_array() };
    let array_shape = array.shape();

    if array_shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have at least shape (N, 4)",
        ));
    } else {
        let num_boxes = array_shape[0];

        if num_boxes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Arrays must have shape (N, 4) with N > 0",
            ));
        }
    }

    return Ok(array);
}

pub fn preprocess_rotated_boxes<'a, N>(
    array: &PyArray2<N>,
) -> Result<ArrayBase<ViewRepr<&N>, Dim<[usize; 2]>>, PyErr>
where
    N: Num + numpy::Element + Send + 'a,
{
    let array = unsafe { array.as_array() };
    let array_shape = array.shape();

    if array_shape[1] != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have at least shape (N, 5)",
        ));
    } else {
        let num_boxes = array_shape[0];

        if num_boxes == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Arrays must have shape (N, 5) with N > 0",
            ));
        }
    }

    return Ok(array);
}

pub fn preprocess_array3<'a, N>(array: &PyArray3<N>) -> ArrayBase<ViewRepr<&N>, Dim<[usize; 3]>>
where
    N: numpy::Element + 'a,
{
    let array = unsafe { array.as_array() };
    return array;
}

pub fn preprocess_array1<'a, N>(array: &PyArray1<N>) -> ArrayBase<ViewRepr<&N>, Dim<[usize; 1]>>
where
    N: numpy::Element + 'a,
{
    let array: ArrayBase<ViewRepr<&N>, ndarray::prelude::Dim<[usize; 1]>> =
        unsafe { array.as_array() };
    return array;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_array_to_numpy() {
        let array = Array1::from(vec![1., 2., 3., 4.]);
        Python::with_gil(|py| {
            let result = array_to_numpy(py, array.view()).unwrap();
            assert_eq!(result.readonly().shape(), &[4]);
        });
    }

    #[test]
    fn test_preprocess_boxes() {
        Python::with_gil(|python| {
            let array = PyArray2::<f32>::zeros(python, [2, 4], false);
            let result = preprocess_boxes::<f32>(array);
            assert!(result.is_ok());
            let unwrapped_result = result.unwrap();
            assert_eq!(unwrapped_result.shape(), &[2, 4]);
        });
    }
    #[test]
    fn test_preprocess_boxes_bad_shape() {
        Python::with_gil(|python| {
            let array = PyArray2::<f32>::zeros(python, [2, 16], false);
            let result = preprocess_boxes::<f32>(array);
            assert!(result.is_err());
        });
    }

    #[test]
    fn test_preprocess_array1() {
        Python::with_gil(|python| {
            let array = PyArray1::<f32>::zeros(python, [2], false);
            let result = preprocess_array1::<f32>(array);
            assert_eq!(result.shape(), &[2]);
        });
    }
}
