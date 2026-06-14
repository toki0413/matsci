use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parsed state accumulated while streaming through OUTCAR.
#[derive(Debug, Default)]
pub struct OutcarState {
    energy: Option<f64>,
    converged: bool,
    encut: Option<f64>,
    ispin: Option<i64>,
    nelm: Option<i64>,
    nelmin: Option<i64>,
    kpoints_found: bool,
    volume: Option<f64>,
    efermi: Option<f64>,
    lattice_vectors: [[f64; 3]; 3],
    has_lattice: bool,
    forces: Vec<ForceEntry>,
    magnetic_moments: Vec<f64>,
    band_gap_note: Option<String>,
}

#[derive(Debug)]
struct ForceEntry {
    position: [f64; 3],
    force: [f64; 3],
}

/// Parse a VASP OUTCAR file in a single streaming pass.
pub fn parse_outcar_file(path: &Path) -> Result<OutcarState, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open OUTCAR: {e}"))?;
    let reader = BufReader::new(file);

    let mut state = OutcarState::default();
    let mut lines = reader.lines().peekable();

    while let Some(Ok(line)) = lines.next() {
        let trimmed = line.trim();

        if trimmed.starts_with("free  energy   TOTEN") {
            if let Some(val) = parse_trailing_float(&line, "=") {
                state.energy = Some(val);
            }
        } else if trimmed == "reached required accuracy - stopping structural energy minimisation" {
            state.converged = true;
        } else if trimmed.starts_with("ENCUT") {
            if let Some(val) = parse_value_after_keyword(&line, "ENCUT") {
                state.encut = Some(val);
            }
        } else if trimmed.starts_with("ISPIN") {
            if let Some(val) = parse_int_after_keyword(&line, "ISPIN") {
                state.ispin = Some(val);
            }
        } else if trimmed.starts_with("NELMIN") {
            if let Some(val) = parse_int_after_keyword(&line, "NELMIN") {
                state.nelmin = Some(val);
            }
        } else if trimmed.starts_with("NELM") {
            if let Some(val) = parse_int_after_keyword(&line, "NELM") {
                state.nelm = Some(val);
            }
        } else if trimmed.starts_with("k-points in units of 2pi/SCALE and weight:") {
            state.kpoints_found = true;
        } else if trimmed.starts_with("direct lattice vectors") {
            // Read next 3 lines: direct vectors, ignore reciprocal part.
            let mut lattice = [[0.0; 3]; 3];
            for i in 0..3 {
                if let Some(Ok(l)) = lines.next() {
                    let parts: Vec<&str> = l.trim().split_whitespace().collect();
                    if parts.len() >= 3 {
                        if let (Ok(a), Ok(b), Ok(c)) = (
                            parts[0].parse::<f64>(),
                            parts[1].parse::<f64>(),
                            parts[2].parse::<f64>(),
                        ) {
                            lattice[i] = [a, b, c];
                        }
                    }
                }
            }
            state.lattice_vectors = lattice;
            state.has_lattice = true;
        } else if trimmed.starts_with("volume of cell :") {
            if let Some(val) = parse_trailing_float(&line, ":") {
                state.volume = Some(val);
            }
        } else if trimmed.starts_with("E-fermi") {
            if let Some(val) = parse_value_after_keyword(&line, "E-fermi") {
                state.efermi = Some(val);
            }
        } else if trimmed.starts_with("TOTAL-FORCE") {
            // Read forces until a blank line or a new ITEM.
            let mut forces = Vec::new();
            while let Some(Ok(next)) = lines.peek() {
                let next_trimmed = next.trim();
                if next_trimmed.is_empty() || next_trimmed.starts_with("ITEM:") {
                    break;
                }
                let parts: Vec<&str> = next_trimmed.split_whitespace().collect();
                if parts.len() >= 6 {
                    if let (Ok(px), Ok(py), Ok(pz), Ok(fx), Ok(fy), Ok(fz)) = (
                        parts[0].parse::<f64>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                        parts[5].parse::<f64>(),
                    ) {
                        forces.push(ForceEntry {
                            position: [px, py, pz],
                            force: [fx, fy, fz],
                        });
                    }
                }
                lines.next();
            }
            if !forces.is_empty() {
                state.forces = forces;
            }
        } else if trimmed.starts_with("magnetization (x)") {
            // Read magnetic moments until a blank line or a new ITEM.
            let mut moments = Vec::new();
            // Skip header line(s) until we hit data.
            while let Some(Ok(next)) = lines.peek() {
                let next_trimmed = next.trim();
                if next_trimmed.is_empty() || next_trimmed.starts_with("ITEM:") {
                    break;
                }
                let parts: Vec<&str> = next_trimmed.split_whitespace().collect();
                if parts.len() >= 5 {
                    if let Ok(m) = parts.last().unwrap().parse::<f64>() {
                        moments.push(m);
                    }
                }
                lines.next();
            }
            if !moments.is_empty() {
                state.magnetic_moments = moments;
            }
        } else if trimmed.starts_with("band No.") {
            state.band_gap_note = Some("see vasprun.xml or use py4vasp".to_string());
        }
    }

    Ok(state)
}

/// Convert parsed OUTCAR state into a Python dict matching VaspTool._parse_outcar output.
pub fn build_outcar_dict<'py>(py: Python<'py>, state: OutcarState) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);

    result.set_item("energy", state.energy)?;
    result.set_item("converged", state.converged)?;
    result.set_item("encut", state.encut)?;
    result.set_item("ispin", state.ispin)?;
    result.set_item("nelm", state.nelm)?;
    result.set_item("nelmin", state.nelmin)?;
    let kpoints: Option<&str> = if state.kpoints_found {
        Some("found")
    } else {
        None
    };
    result.set_item("kpoints", kpoints)?;
    result.set_item("volume", state.volume)?;
    result.set_item("band_gap", state.band_gap_note)?;
    result.set_item("efermi", state.efermi)?;

    if state.has_lattice {
        let lat = PyList::new(py, Vec::<Vec<f64>>::new())?;
        for row in &state.lattice_vectors {
            lat.append(vec![row[0], row[1], row[2]])?;
        }
        result.set_item("lattice_vectors", lat)?;
    } else {
        result.set_item("lattice_vectors", PyList::empty(py))?;
    }

    let forces = PyList::empty(py);
    for f in &state.forces {
        let entry = PyDict::new(py);
        entry.set_item(
            "position",
            vec![f.position[0], f.position[1], f.position[2]],
        )?;
        entry.set_item("force", vec![f.force[0], f.force[1], f.force[2]])?;
        forces.append(entry)?;
    }
    result.set_item("forces", forces)?;

    result.set_item("magnetic_moments", state.magnetic_moments.clone())?;

    Ok(result)
}

fn parse_trailing_float(line: &str, delimiter: &str) -> Option<f64> {
    line.rsplit(delimiter)
        .next()?
        .trim()
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn parse_value_after_keyword(line: &str, keyword: &str) -> Option<f64> {
    let start = line.find(keyword)? + keyword.len();
    let rest = &line[start..];
    // Skip non-numeric chars until we hit a number
    for token in rest.split_whitespace() {
        if let Ok(n) = token.parse::<f64>() {
            return Some(n);
        }
    }
    None
}

fn parse_int_after_keyword(line: &str, keyword: &str) -> Option<i64> {
    let start = line.find(keyword)? + keyword.len();
    let rest = &line[start..];
    for token in rest.split_whitespace() {
        if let Ok(n) = token.parse::<i64>() {
            return Some(n);
        }
    }
    None
}
