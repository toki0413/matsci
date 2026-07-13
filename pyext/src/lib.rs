mod analysis;
mod files;
mod lammps;
mod sandbox;
mod vasp;
mod vectors;

use numpy::{PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Parse a VASP OUTCAR file.
#[pyfunction]
#[pyo3(signature = (path))]
fn parse_outcar(py: Python, path: &str) -> PyResult<Py<PyDict>> {
    use std::path::Path;

    let outcar_path = Path::new(path);
    if !outcar_path.exists() {
        let result = PyDict::new(py);
        result.set_item("error", "OUTCAR file not found")?;
        return Ok(result.into());
    }

    let state = py
        .allow_threads(|| vasp::parse_outcar_file(outcar_path))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let result = vasp::build_outcar_dict(py, state)?;
    Ok(result.into())
}

/// Compute mean squared displacement from a NumPy position array.
///
/// `positions` shape: `(n_frames, n_atoms, 3)`.
/// `timesteps`: optional 1-D array of frame timesteps; if None, uses frame indices.
/// `box_dims`: optional `[lx, ly, lz]`. When supplied, the minimum image
/// convention is applied so wrapped coordinates (LAMMPS dump default) work.
#[pyfunction]
#[pyo3(signature = (positions, timesteps=None, box_dims=None))]
fn compute_msd(
    py: Python,
    positions: PyReadonlyArray3<f64>,
    timesteps: Option<Vec<i64>>,
    box_dims: Option<[f64; 3]>,
) -> PyResult<Py<PyDict>> {
    let arr = positions.as_array();
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions must have shape (n_frames, n_atoms, 3)",
        ));
    }
    let n_frames = shape[0];
    let n_atoms = shape[1];

    // 零拷贝: as_slice 直接借 numpy buffer, 之前 iter().copied().collect()
    // 会把整个 positions 复制成 Vec<f64> (MD 轨迹动辄几百万 f64, 浪费内存).
    // 要求 C-contiguous, 不连续直接报错 (caller reshape / ascontiguousarray 即可).
    let positions_slice: &[f64] = positions.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "positions array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    // 释放 GIL: MSD 纯数值计算, 不碰 Python 对象, 持 GIL 阻塞其他 Python 线程.
    let msd = py.allow_threads(|| {
        analysis::msd_from_slice(positions_slice, n_frames, n_atoms, box_dims)
    });

    let ts_slice = timesteps.as_deref();
    let result = analysis::build_msd_dict(py, &msd, ts_slice)?;
    Ok(result.into())
}

/// Compute radial distribution function from a NumPy position array.
///
/// `positions` shape: `(n_atoms, 3)` for a single frame.
/// `box` is a length-3 sequence `[lx, ly, lz]`.
#[pyfunction]
#[pyo3(signature = (positions, box_dims, bins=100, r_max=None))]
fn compute_rdf(
    py: Python,
    positions: PyReadonlyArray2<f64>,
    box_dims: [f64; 3],
    bins: usize,
    r_max: Option<f64>,
) -> PyResult<Py<PyDict>> {
    let arr = positions.as_array();
    let shape = arr.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "positions must have shape (n_atoms, 3)",
        ));
    }
    let n_atoms = shape[0];

    // 零拷贝 + 释放 GIL, 同 compute_msd.
    let positions_slice: &[f64] = positions.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "positions array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    let (r_values, g, r_max) = py
        .allow_threads(|| {
            analysis::rdf_from_slice(positions_slice, n_atoms, box_dims, bins, r_max)
        })
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Failed to compute RDF"))?;

    let result = analysis::build_rdf_dict(py, &r_values, &g, bins, r_max)?;
    Ok(result.into())
}

/// Rust performance extensions for Huginn.
#[pymodule]
fn huginn_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    files::register_module(m)?;
    lammps::register_module(m)?;
    sandbox::register_module(m)?;
    vectors::register_module(m)?;

    m.add_function(wrap_pyfunction!(parse_outcar, m)?)?;
    m.add_function(wrap_pyfunction!(compute_msd, m)?)?;
    m.add_function(wrap_pyfunction!(compute_rdf, m)?)?;
    Ok(())
}
