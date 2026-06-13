use crate::analysis::{build_msd_dict, build_rdf_dict, msd_from_slice, rdf_from_slice};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone)]
struct Atom {
    fields: HashMap<String, AtomValue>,
}

#[derive(Debug, Clone)]
enum AtomValue {
    Number(f64),
    Text(String),
}

impl AtomValue {
    fn to_py_object<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            AtomValue::Number(n) => Ok(n.into_pyobject(py)?.into_any()),
            AtomValue::Text(s) => Ok(s.into_pyobject(py)?.into_any()),
        }
    }
}

#[derive(Debug, Clone)]
struct Frame {
    timestep: i64,
    n_atoms: usize,
    box_bounds: [[f64; 2]; 3],
    atoms: Vec<Atom>,
}

fn parse_dump_file(path: &Path) -> Result<Vec<Frame>, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open trajectory file: {e}"))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut frames = Vec::new();
    let mut current_frame: Option<Frame> = None;
    let mut atom_headers: Vec<String>;

    while let Some(Ok(line)) = lines.next() {
        let trimmed = line.trim();

        if trimmed == "ITEM: TIMESTEP" {
            if let Some(frame) = current_frame.take() {
                frames.push(frame);
            }

            let ts_line = lines
                .next()
                .ok_or("Unexpected end of file after ITEM: TIMESTEP")?
                .map_err(|e| format!("Failed to read timestep: {e}"))?;
            let timestep = ts_line
                .trim()
                .parse::<i64>()
                .map_err(|e| format!("Failed to parse timestep: {e}"))?;

            current_frame = Some(Frame {
                timestep,
                n_atoms: 0,
                box_bounds: [[0.0, 0.0]; 3],
                atoms: Vec::new(),
            });
        } else if trimmed.starts_with("ITEM: NUMBER OF ATOMS") {
            let n_line = lines
                .next()
                .ok_or("Unexpected end of file after ITEM: NUMBER OF ATOMS")?
                .map_err(|e| format!("Failed to read atom count: {e}"))?;
            let n_atoms = n_line
                .trim()
                .parse::<usize>()
                .map_err(|e| format!("Failed to parse atom count: {e}"))?;

            if let Some(ref mut frame) = current_frame {
                frame.n_atoms = n_atoms;
            }
        } else if trimmed.starts_with("ITEM: BOX BOUNDS") {
            let mut bounds = [[0.0; 2]; 3];
            for i in 0..3 {
                let b_line = lines
                    .next()
                    .ok_or("Unexpected end of file in ITEM: BOX BOUNDS")?
                    .map_err(|e| format!("Failed to read box bound: {e}"))?;
                let parts: Vec<&str> = b_line.trim().split_whitespace().collect();
                if parts.len() < 2 {
                    return Err("Invalid box bound line".to_string());
                }
                bounds[i][0] = parts[0]
                    .parse::<f64>()
                    .map_err(|e| format!("Failed to parse box bound: {e}"))?;
                bounds[i][1] = parts[1]
                    .parse::<f64>()
                    .map_err(|e| format!("Failed to parse box bound: {e}"))?;
            }
            if let Some(ref mut frame) = current_frame {
                frame.box_bounds = bounds;
            }
        } else if trimmed.starts_with("ITEM: ATOMS") {
            atom_headers = trimmed
                .strip_prefix("ITEM: ATOMS")
                .unwrap_or("")
                .trim()
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();

            let n_atoms = current_frame.as_ref().map(|f| f.n_atoms).unwrap_or(0);
            let mut atoms = Vec::with_capacity(n_atoms);

            for _ in 0..n_atoms {
                let a_line = lines
                    .next()
                    .ok_or("Unexpected end of file in ITEM: ATOMS")?
                    .map_err(|e| format!("Failed to read atom line: {e}"))?;
                let parts: Vec<&str> = a_line.trim().split_whitespace().collect();
                let mut fields = HashMap::with_capacity(atom_headers.len());

                for (header, value) in atom_headers.iter().zip(parts.iter()) {
                    let parsed = match value.parse::<f64>() {
                        Ok(n) => AtomValue::Number(n),
                        Err(_) => AtomValue::Text(value.to_string()),
                    };
                    fields.insert(header.clone(), parsed);
                }

                atoms.push(Atom { fields });
            }

            if let Some(ref mut frame) = current_frame {
                frame.atoms = atoms;
            }
        }
    }

    if let Some(frame) = current_frame.take() {
        frames.push(frame);
    }

    Ok(frames)
}

fn frame_positions_to_vec(frames: &[Frame]) -> (Vec<f64>, Vec<i64>, Vec<[f64; 3]>) {
    let n_frames = frames.len();
    let n_atoms = frames.first().map(|f| f.atoms.len()).unwrap_or(0);
    let mut positions = Vec::with_capacity(n_frames * n_atoms * 3);
    let mut timesteps = Vec::with_capacity(n_frames);
    let mut box_dims = Vec::with_capacity(n_frames);

    for frame in frames {
        timesteps.push(frame.timestep);
        let lx = frame.box_bounds[0][1] - frame.box_bounds[0][0];
        let ly = frame.box_bounds[1][1] - frame.box_bounds[1][0];
        let lz = frame.box_bounds[2][1] - frame.box_bounds[2][0];
        box_dims.push([lx, ly, lz]);

        for atom in &frame.atoms {
            let x = atom.fields.get("x").and_then(|v| v.as_number()).unwrap_or(0.0);
            let y = atom.fields.get("y").and_then(|v| v.as_number()).unwrap_or(0.0);
            let z = atom.fields.get("z").and_then(|v| v.as_number()).unwrap_or(0.0);
            positions.push(x);
            positions.push(y);
            positions.push(z);
        }
    }

    (positions, timesteps, box_dims)
}

impl AtomValue {
    fn as_number(&self) -> Option<f64> {
        match self {
            AtomValue::Number(n) => Some(*n),
            _ => None,
        }
    }
}

fn build_result_dict<'py>(
    py: Python<'py>,
    frames: Vec<Frame>,
    compute_msd_flag: bool,
    compute_rdf_flag: bool,
    rdf_bins: usize,
    rdf_r_max: Option<f64>,
    include_frames: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);

    let n_frames = frames.len();
    result.set_item("n_frames", n_frames)?;

    let n_atoms = frames.first().map(|f| f.n_atoms).unwrap_or(0);
    result.set_item("n_atoms", n_atoms)?;

    let mut atom_types = HashSet::new();
    let mut timesteps = Vec::with_capacity(n_frames);
    let mut box_bounds: Option<[[f64; 2]; 3]> = None;

    for frame in &frames {
        timesteps.push(frame.timestep);
        if box_bounds.is_none() {
            box_bounds = Some(frame.box_bounds);
        }
        for atom in &frame.atoms {
            if let Some(AtomValue::Number(t)) = atom.fields.get("type") {
                atom_types.insert(*t as i64);
            }
        }
    }

    let mut atom_types: Vec<i64> = atom_types.into_iter().collect();
    atom_types.sort();
    result.set_item("atom_types", atom_types)?;
    result.set_item("timesteps", timesteps.clone())?;

    if let Some(bounds) = box_bounds {
        let py_bounds = PyList::new(py, Vec::<Vec<f64>>::new())?;
        for b in &bounds {
            py_bounds.append(vec![b[0], b[1]])?;
        }
        result.set_item("box_bounds", py_bounds)?;
    } else {
        result.set_item("box_bounds", PyList::empty(py))?;
    }

    if include_frames {
        let py_frames = PyList::empty(py);
        for frame in &frames {
            let py_frame = PyDict::new(py);
            py_frame.set_item("timestep", frame.timestep)?;
            py_frame.set_item("n_atoms", frame.n_atoms)?;

            let py_box = PyList::new(py, Vec::<Vec<f64>>::new())?;
            for b in &frame.box_bounds {
                py_box.append(vec![b[0], b[1]])?;
            }
            py_frame.set_item("box", py_box)?;

            let py_atoms = PyList::empty(py);
            for atom in &frame.atoms {
                let py_atom = PyDict::new(py);
                for (key, value) in &atom.fields {
                    py_atom.set_item(key.as_str(), value.to_py_object(py)?)?;
                }
                py_atoms.append(py_atom)?;
            }
            py_frame.set_item("atoms", py_atoms)?;

            py_frames.append(py_frame)?;
        }
        result.set_item("frames", py_frames)?;
    }

    if compute_msd_flag && !frames.is_empty() {
        let (positions, ts, _) = frame_positions_to_vec(&frames);
        let msd = msd_from_slice(&positions, n_frames, n_atoms);
        let msd_dict = build_msd_dict(py, &msd, Some(&ts))?;
        if let Ok(py_msd) = msd_dict.get_item("msd") {
            result.set_item("msd", py_msd)?;
        }
    }

    if compute_rdf_flag && !frames.is_empty() {
        if let Some(frame) = frames.last() {
            let n_atoms = frame.atoms.len();
            let mut positions = Vec::with_capacity(n_atoms * 3);
            for atom in &frame.atoms {
                let x = atom.fields.get("x").and_then(|v| v.as_number()).unwrap_or(0.0);
                let y = atom.fields.get("y").and_then(|v| v.as_number()).unwrap_or(0.0);
                let z = atom.fields.get("z").and_then(|v| v.as_number()).unwrap_or(0.0);
                positions.push(x);
                positions.push(y);
                positions.push(z);
            }
            let lx = frame.box_bounds[0][1] - frame.box_bounds[0][0];
            let ly = frame.box_bounds[1][1] - frame.box_bounds[1][0];
            let lz = frame.box_bounds[2][1] - frame.box_bounds[2][0];
            if let Some((r_values, g, r_max)) = rdf_from_slice(&positions, n_atoms, [lx, ly, lz], rdf_bins, rdf_r_max) {
                let rdf_dict = build_rdf_dict(py, &r_values, &g, rdf_bins, r_max)?;
                result.set_item("rdf", rdf_dict)?;
            }
        }
    }

    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (path, compute_msd=false, compute_rdf=false, rdf_bins=100, rdf_r_max=None, include_frames=false))]
fn parse_lammps_dump(
    py: Python,
    path: &str,
    compute_msd: bool,
    compute_rdf: bool,
    rdf_bins: usize,
    rdf_r_max: Option<f64>,
    include_frames: bool,
) -> PyResult<Py<PyDict>> {
    let traj_path = Path::new(path);
    if !traj_path.exists() {
        let result = PyDict::new(py);
        result.set_item("error", "Trajectory file not found")?;
        return Ok(result.into());
    }

    let frames = py
        .allow_threads(|| parse_dump_file(traj_path))
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let result = build_result_dict(
        py,
        frames,
        compute_msd,
        compute_rdf,
        rdf_bins,
        rdf_r_max,
        include_frames,
    )?;

    Ok(result.into())
}

/// Register LAMMPS functions in the Python module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_lammps_dump, m)?)?;
    Ok(())
}
