use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

/// Compute cosine-similarity top-k over a matrix of embeddings.
///
/// `query`: 1-D array of length D.
/// `matrix`: 2-D array of shape (N, D).
/// `k`: number of results to return (default: all rows).
#[pyfunction]
#[pyo3(signature = (query, matrix, k=None))]
fn top_k(
    py: Python,
    query: PyReadonlyArray1<f32>,
    matrix: PyReadonlyArray2<f32>,
    k: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let q = query.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("query is not contiguous: {}", e))
    })?;
    let m = matrix.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("matrix is not contiguous: {}", e))
    })?;

    let shape = matrix.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must be 2-dimensional",
        ));
    }
    let n_rows = shape[0];
    let dim = shape[1];

    if q.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "query dimension {} does not match matrix dimension {}",
            q.len(),
            dim
        )));
    }

    let norm_q: f32 = q.iter().map(|v| v * v).sum::<f32>().sqrt();

    let mut scores: Vec<(usize, f32)> = (0..n_rows)
        .into_par_iter()
        .map(|i| {
            let row = &m[i * dim..(i + 1) * dim];
            let dot: f32 = q.iter().zip(row).map(|(a, b)| a * b).sum();
            let norm_row: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            let score = if norm_q > 0.0 && norm_row > 0.0 {
                dot / (norm_q * norm_row)
            } else {
                0.0
            };
            (i, score)
        })
        .collect();

    scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = k.unwrap_or(n_rows).min(n_rows);
    let top = &scores[..k];

    let result = PyDict::new(py);
    result.set_item(
        "indices",
        top.iter().map(|(i, _)| *i).collect::<Vec<usize>>(),
    )?;
    result.set_item("scores", top.iter().map(|(_, s)| *s).collect::<Vec<f32>>())?;
    Ok(result.into())
}

/// Register vector functions in the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(top_k, m)?)?;
    Ok(())
}
