use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// Compute mean squared displacement for a set of positions.
///
/// `positions` is a flat slice of length `n_frames * n_atoms * 3`,
/// ordered as [frame0_atom0_x, frame0_atom0_y, frame0_atom0_z, ...].
pub fn msd_from_slice(positions: &[f64], n_frames: usize, n_atoms: usize) -> Vec<(i64, f64)> {
    if n_frames < 2 || n_atoms == 0 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(n_frames - 1);

    for frame_idx in 1..n_frames {
        let mut sum = 0.0;
        for atom_idx in 0..n_atoms {
            let ref_base = atom_idx * 3;
            let cur_base = (frame_idx * n_atoms + atom_idx) * 3;
            let dx = positions[cur_base] - positions[ref_base];
            let dy = positions[cur_base + 1] - positions[ref_base + 1];
            let dz = positions[cur_base + 2] - positions[ref_base + 2];
            sum += dx * dx + dy * dy + dz * dz;
        }
        result.push((frame_idx as i64, sum / n_atoms as f64));
    }

    result
}

/// Compute radial distribution function for a single frame.
///
/// `positions` is a flat slice of length `n_atoms * 3`.
/// `box_dims` gives the orthorhombic box lengths `[lx, ly, lz]`.
pub fn rdf_from_slice(
    positions: &[f64],
    n_atoms: usize,
    box_dims: [f64; 3],
    bins: usize,
    r_max: Option<f64>,
) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    if n_atoms < 2 || bins == 0 {
        return None;
    }

    let lx = box_dims[0];
    let ly = box_dims[1];
    let lz = box_dims[2];

    let r_max = r_max.unwrap_or_else(|| lx.min(ly).min(lz) / 2.0);
    if r_max <= 0.0 {
        return None;
    }

    let dr = r_max / bins as f64;
    let mut g = vec![0.0; bins];

    for i in 0..n_atoms {
        let ibase = i * 3;
        for j in (i + 1)..n_atoms {
            let jbase = j * 3;
            let mut dx = positions[jbase] - positions[ibase];
            let mut dy = positions[jbase + 1] - positions[ibase + 1];
            let mut dz = positions[jbase + 2] - positions[ibase + 2];

            // Minimum image convention
            dx -= lx * (dx / lx).round();
            dy -= ly * (dy / ly).round();
            dz -= lz * (dz / lz).round();

            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < r_max {
                let idx = (r / dr) as usize;
                if idx < bins {
                    g[idx] += 2.0;
                }
            }
        }
    }

    // Normalize
    let volume = lx * ly * lz;
    let rho = n_atoms as f64 / volume;
    let pi = std::f64::consts::PI;

    let mut r_values = Vec::with_capacity(bins);
    for i in 0..bins {
        let r_inner = i as f64 * dr;
        let r_outer = (i + 1) as f64 * dr;
        let shell_vol = 4.0 / 3.0 * pi * (r_outer.powi(3) - r_inner.powi(3));
        if shell_vol > 0.0 {
            g[i] /= n_atoms as f64 * rho * shell_vol;
        }
        r_values.push((i as f64 + 0.5) * dr);
    }

    Some((r_values, g, r_max))
}

/// Build the Python MSD result dict from raw (timestep, msd) pairs.
pub fn build_msd_dict<'py>(
    py: Python<'py>,
    msd: &[(i64, f64)],
    timesteps: Option<&[i64]>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let py_msd = PyList::empty(py);

    for (frame_idx, value) in msd.iter() {
        let ts = timesteps
            .map(|t| t[*frame_idx as usize])
            .unwrap_or(*frame_idx);
        let item = PyDict::new(py);
        item.set_item("timestep", ts)?;
        item.set_item("msd", *value)?;
        py_msd.append(item)?;
    }

    result.set_item("msd", py_msd)?;
    Ok(result)
}

/// Build the Python RDF result dict.
pub fn build_rdf_dict<'py>(
    py: Python<'py>,
    r_values: &[f64],
    g: &[f64],
    bins: usize,
    r_max: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("r", r_values.to_vec())?;
    result.set_item("g", g.to_vec())?;
    result.set_item("bins", bins)?;
    result.set_item("r_max", r_max)?;
    Ok(result)
}
