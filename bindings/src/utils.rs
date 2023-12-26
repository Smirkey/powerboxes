use ndarray::{Array1, Array2, Array3, ArrayBase, OwnedRepr};
use num_traits::Num;
use numpy::{IntoPyArray, PyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;

pub fn array_to_numpy<T: numpy::Element, D: ndarray::Dimension>(
    py: Python,
    array: ArrayBase<OwnedRepr<T>, D>,
) -> PyResult<&PyArray<T, D>> {
    let numpy_array: &PyArray<T, D> = array.into_pyarray(py);
    return Ok(numpy_array);
}

pub fn preprocess_boxes<N>(array: &PyArray2<N>) -> Result<Array2<N>, PyErr>
where
    N: Num + numpy::Element + Send,
{
    let array = unsafe { array.as_array() };
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

pub fn preprocess_array3<N>(array: &PyArray3<N>) -> Array3<N>
where
    N: numpy::Element,
{
    let array = unsafe { array.as_array().to_owned() };
    return array;
}

pub fn preprocess_array1<N>(array: &PyArray1<N>) -> Array1<N>
where
    N: numpy::Element,
{
    let array = unsafe { array.as_array().to_owned() };
    return array;
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
