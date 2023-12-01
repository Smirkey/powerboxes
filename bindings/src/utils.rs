use ndarray::{ArrayBase, Dim, OwnedRepr};
use num_traits::Num;
use numpy::{IntoPyArray, PyArray, PyArray2};
use pyo3::prelude::*;

pub fn array_to_numpy<T: numpy::Element, D: ndarray::Dimension>(
    py: Python,
    array: ArrayBase<OwnedRepr<T>, D>,
) -> PyResult<&PyArray<T, D>> {
    let numpy_array: &PyArray<T, D> = array.into_pyarray(py);
    return Ok(numpy_array);
}

pub fn preprocess_array<N>(
    array: &PyArray2<N>,
) -> Result<ArrayBase<OwnedRepr<N>, Dim<[usize; 2]>>, PyErr>
where
    N: Num + numpy::Element + Send,
{
    // Usage:
    let array: ArrayBase<OwnedRepr<N>, Dim<[usize; 2]>> = unsafe { array.as_array().to_owned() };
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
    use ndarray::ArrayBase;

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
    fn test_preprocess_array() {
        Python::with_gil(|python| {
            let array = PyArray2::<f32>::zeros(python, [2, 4], false);
            let result = preprocess_array::<f32>(array);
            assert!(result.is_ok());
            let unwrapped_result = result.unwrap();
            assert_eq!(unwrapped_result.shape(), &[2, 4]);
        });
    }
    #[test]
    fn test_preprocess_array_bad_shape() {
        Python::with_gil(|python| {
            let array = PyArray2::<f32>::zeros(python, [2, 16], false);
            let result = preprocess_array::<f32>(array);
            assert!(result.is_err());
        });
    }
}
