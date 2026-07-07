use pyo3::prelude::*;
use pyo3::types::PyList;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Return the last `n` lines of a text file without loading the whole file.
#[pyfunction]
#[pyo3(signature = (path, n=100))]
fn tail_lines(py: Python, path: &str, n: usize) -> PyResult<Py<PyList>> {
    let file_path = Path::new(path);
    if !file_path.exists() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "file not found: {}",
            path
        )));
    }

    let lines = py
        .allow_threads(|| read_last_lines(file_path, n))
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to read file: {}", e))
        })?;

    let py_list = PyList::new(py, lines)?;
    Ok(py_list.into())
}

fn read_last_lines(path: &Path, n: usize) -> Result<Vec<String>, std::io::Error> {
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut file = File::open(path)?;
    let len = file.seek(SeekFrom::End(0))?;

    let mut buf = Vec::new();
    let mut pos = len as i64;
    let mut newline_count = 0;

    // Read backwards in chunks until we have enough newlines or reach the start.
    const CHUNK_SIZE: usize = 8192;
    while pos > 0 && newline_count <= n {
        let chunk_start = (pos - CHUNK_SIZE as i64).max(0) as u64;
        let chunk_len = (pos - chunk_start as i64) as usize;
        file.seek(SeekFrom::Start(chunk_start))?;
        buf.resize(chunk_len, 0);
        file.read_exact(&mut buf)?;
        newline_count += buf.iter().filter(|&&b| b == b'\n').count();
        pos = chunk_start as i64;
    }

    // Read the remaining prefix if we stopped before the start.
    if pos > 0 {
        file.seek(SeekFrom::Start(0))?;
        let prefix_len = pos as usize;
        let mut prefix = vec![0u8; prefix_len];
        file.read_exact(&mut prefix)?;
        buf.splice(0..0, prefix.iter().copied());
    }

    let text = String::from_utf8_lossy(&buf);
    let lines: Vec<&str> = text.lines().collect();

    let start = lines.len().saturating_sub(n);
    Ok(lines[start..].iter().map(|s| s.to_string()).collect())
}

/// Register file utility functions in the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tail_lines, m)?)?;
    Ok(())
}
