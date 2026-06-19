import { useState, useMemo, useRef, useCallback } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import {
  Upload, Eye, EyeOff, Crosshair, Info, Box,
} from "lucide-react";

/* ── Element color/size data (CPK) ── */
const ELEMENT_COLORS: Record<string, string> = {
  H: "#FFFFFF", He: "#D9FFFF", Li: "#CC80FF", Be: "#C2FF00", B: "#FFB5B5",
  C: "#909090", N: "#3050F8", O: "#FF0D0D", F: "#90E050", Ne: "#B3E3F5",
  Na: "#AB5CF2", Mg: "#8AFF00", Al: "#BFA6A6", Si: "#F0C8A0", P: "#FF8000",
  S: "#FFFF30", Cl: "#1FF01F", Ar: "#80D1E3", K: "#8F40D4", Ca: "#3DFF00",
  Sc: "#E6E6E6", Ti: "#BFC2C7", V: "#A6A6AB", Cr: "#8A99C7", Mn: "#9C7AC7",
  Fe: "#E06633", Co: "#F090A0", Ni: "#50D050", Cu: "#C88033", Zn: "#7D80B0",
  Ga: "#C28F8F", Ge: "#668F8F", As: "#BD80E3", Se: "#FFA100", Br: "#A62929",
  Kr: "#5CB8D1", Rb: "#702EB0", Sr: "#00FF00", Y: "#94FFFF", Zr: "#94E0E0",
  Nb: "#73C2C9", Mo: "#54B5B5", Tc: "#3B9E9E", Ru: "#248F8F", Rh: "#0A7D8C",
  Pd: "#006985", Ag: "#C0C0C0", Cd: "#FFD98F", In: "#A67573", Sn: "#668080",
  Sb: "#9E63B5", Te: "#D47A00", I: "#940094", Xe: "#429EB0", Cs: "#57178F",
  Ba: "#00C900", La: "#70D4FF", Ce: "#FFFFC7", Pr: "#D9FFC7", Nd: "#C7FFC7",
  Pm: "#A3FFC7", Sm: "#8FFFC7", Eu: "#61FFC7", Gd: "#45FFC7", Tb: "#30FFC7",
  Dy: "#1FFFC7", Ho: "#00FF9C", Er: "#00E675", Tm: "#00D452", Yb: "#00BF38",
  Lu: "#00AB24", Hf: "#4DC2FF", Ta: "#4DA6FF", W: "#2194D6", Re: "#267DAB",
  Os: "#266696", Ir: "#175487", Pt: "#D0D0E0", Au: "#FFD123", Hg: "#B8B8D0",
  Tl: "#A6544D", Pb: "#575961", Bi: "#9E4FB5", Po: "#AB5C00", At: "#754F45",
  Rn: "#428296", Fr: "#420066", Ra: "#007D00", Ac: "#70ABFA", Th: "#00BAFF",
  Pa: "#00A1FF", U: "#008FFF", Np: "#0080FF", Pu: "#006BFF",
};
const DEFAULT_COLOR = "#FF69B4";

const ELEMENT_RADII: Record<string, number> = {
  H: 0.25, He: 0.31, C: 0.40, N: 0.38, O: 0.36, F: 0.32, S: 0.50, P: 0.44,
  Si: 0.46, Fe: 0.48, Ti: 0.46, Cu: 0.42, Zn: 0.42, Al: 0.50, Na: 0.68,
  K: 0.75, Ca: 0.66, Mg: 0.58, Cl: 0.48, Br: 0.56, I: 0.65,
};
const DEFAULT_RADIUS = 0.45;

/* ── Structure parsing ── */
interface Atom { symbol: string; x: number; y: number; z: number; }
interface Lattice { a: THREE.Vector3; b: THREE.Vector3; c: THREE.Vector3; }
interface StructureData { atoms: Atom[]; lattice?: Lattice; title?: string; }

function parseXYZ(text: string): StructureData {
  const lines = text.trim().split("\n").map((l) => l.trim());
  const count = parseInt(lines[0], 10);
  const title = lines[1] || "";
  const atoms: Atom[] = [];
  for (let i = 2; i < 2 + count && i < lines.length; i++) {
    const parts = lines[i].split(/\s+/);
    if (parts.length >= 4) {
      atoms.push({ symbol: parts[0], x: +parts[1], y: +parts[2], z: +parts[3] });
    }
  }
  return { atoms, title };
}

function parsePOSCAR(text: string): StructureData {
  const lines = text.trim().split("\n");
  const title = lines[0]?.trim() || "";
  const scale = parseFloat(lines[1]?.trim() || "1");
  const lattice: Lattice = {
    a: new THREE.Vector3(...lines[2].trim().split(/\s+/).map(Number)).multiplyScalar(scale),
    b: new THREE.Vector3(...lines[3].trim().split(/\s+/).map(Number)).multiplyScalar(scale),
    c: new THREE.Vector3(...lines[4].trim().split(/\s+/).map(Number)).multiplyScalar(scale),
  };
  const species = lines[5].trim().split(/\s+/);
  const counts = lines[6].trim().split(/\s+/).map(Number);
  const direct = lines[7]?.trim().toLowerCase().startsWith("d");
  const atoms: Atom[] = [];
  let lineIdx = 8;
  for (let s = 0; s < species.length; s++) {
    for (let j = 0; j < counts[s] && lineIdx < lines.length; j++, lineIdx++) {
      const coords = lines[lineIdx].trim().split(/\s+/).map(Number);
      let pos: THREE.Vector3;
      if (direct) {
        pos = lattice.a.clone().multiplyScalar(coords[0])
          .add(lattice.b.clone().multiplyScalar(coords[1]))
          .add(lattice.c.clone().multiplyScalar(coords[2]));
      } else {
        pos = new THREE.Vector3(coords[0] * scale, coords[1] * scale, coords[2] * scale);
      }
      atoms.push({ symbol: species[s], x: pos.x, y: pos.y, z: pos.z });
    }
  }
  return { atoms, lattice, title };
}

function detectBonds(atoms: Atom[]): [number, number][] {
  const bonds: [number, number][] = [];
  for (let i = 0; i < atoms.length; i++) {
    const ri = (ELEMENT_RADII[atoms[i].symbol] ?? DEFAULT_RADIUS) * 1.8;
    for (let j = i + 1; j < atoms.length; j++) {
      const rj = (ELEMENT_RADII[atoms[j].symbol] ?? DEFAULT_RADIUS) * 1.8;
      const dx = atoms[i].x - atoms[j].x;
      const dy = atoms[i].y - atoms[j].y;
      const dz = atoms[i].z - atoms[j].z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < ri + rj && dist > 0.4) bonds.push([i, j]);
    }
  }
  return bonds;
}

/* ── 3D sub-components ── */
function AtomSphere({ position, color, radius, label, onClick }: {
  position: [number, number, number]; color: string; radius: number; label: string; onClick?: () => void;
}) {
  return (
    <mesh position={position} onClick={onClick}>
      <sphereGeometry args={[radius, 24, 24]} />
      <meshStandardMaterial color={color} roughness={0.35} metalness={0.15} />
      {onClick && (
        <Text position={[0, radius + 0.2, 0]} fontSize={0.22} color="#2a2520" anchorX="center" anchorY="bottom">
          {label}
        </Text>
      )}
    </mesh>
  );
}

function BondCylinder({ start, end }: { start: [number, number, number]; end: [number, number, number] }) {
  const s = new THREE.Vector3(...start);
  const e = new THREE.Vector3(...end);
  const mid = s.clone().add(e).multiplyScalar(0.5);
  const dir = e.clone().sub(s);
  const len = dir.length();
  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());

  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <cylinderGeometry args={[0.06, 0.06, len, 8]} />
      <meshStandardMaterial color="#888075" roughness={0.5} metalness={0.1} />
    </mesh>
  );
}

function UnitCell({ lattice }: { lattice: Lattice }) {
  const edges = useMemo(() => {
    const o = new THREE.Vector3(0, 0, 0);
    const pts: [THREE.Vector3, THREE.Vector3][] = [];
    const { a, b, c } = lattice;
    // 12 edges of parallelepiped
    const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
    const edgeIdx: [number, number][] = [
      [0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7],
    ];
    edgeIdx.forEach(([i, j]) => pts.push([corners[i], corners[j]]));
    return pts;
  }, [lattice]);

  return (
    <group>
      {edges.map(([s, e], i) => (
        <BondCylinder key={i} start={[s.x, s.y, s.z]} end={[e.x, e.y, e.z]} />
      ))}
    </group>
  );
}

function CameraFitter({ atoms }: { atoms: Atom[] }) {
  const { camera } = useThree();
  useMemo(() => {
    if (atoms.length === 0) return;
    const center = new THREE.Vector3();
    atoms.forEach((a) => center.add(new THREE.Vector3(a.x, a.y, a.z)));
    center.divideScalar(atoms.length);
    let maxDist = 0;
    atoms.forEach((a) => {
      const d = center.distanceTo(new THREE.Vector3(a.x, a.y, a.z));
      if (d > maxDist) maxDist = d;
    });
    const dist = Math.max(maxDist * 2.5, 5);
    (camera as THREE.PerspectiveCamera).position.set(center.x + dist * 0.5, center.y + dist * 0.3, center.z + dist);
    (camera as THREE.PerspectiveCamera).lookAt(center);
  }, [atoms, camera]);
  return null;
}

/* ── Main component ── */
export default function StructureViewer({ API_BASE }: { API_BASE: string }) {
  const [structure, setStructure] = useState<StructureData | null>(null);
  const [rawInput, setRawInput] = useState("");
  const [showUnitCell, setShowUnitCell] = useState(true);
  const [showBonds, setShowBonds] = useState(true);
  const [selectedAtom, setSelectedAtom] = useState<number | null>(null);
  const [inputFormat, setInputFormat] = useState<"xyz" | "poscar">("xyz");
  const [info, setInfo] = useState<string>("");
  const fileRef = useRef<HTMLInputElement>(null);

  const parseInput = useCallback((text: string, fmt: "xyz" | "poscar") => {
    try {
      const data = fmt === "xyz" ? parseXYZ(text) : parsePOSCAR(text);
      if (data.atoms.length === 0) { setInfo("No atoms parsed."); return; }
      setStructure(data);
      setSelectedAtom(null);
      // Compute info
      const elements = new Set(data.atoms.map((a) => a.symbol));
      let infoStr = `${data.atoms.length} atoms, ${elements.size} elements (${[...elements].join(", ")})`;
      if (data.lattice) {
        const { a, b, c } = data.lattice;
        infoStr += `\nLattice: |a|=${a.length().toFixed(2)} |b|=${b.length().toFixed(2)} |c|=${c.length().toFixed(2)} Å`;
      }
      setInfo(infoStr);
    } catch (e: any) {
      setInfo(`Parse error: ${e.message}`);
    }
  }, []);

  const handleLoad = useCallback(() => {
    if (rawInput.trim()) parseInput(rawInput, inputFormat);
  }, [rawInput, inputFormat, parseInput]);

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      setRawInput(text);
      const fmt = file.name.toLowerCase().endsWith(".xyz") ? "xyz" : "poscar";
      setInputFormat(fmt);
      parseInput(text, fmt);
    };
    reader.readAsText(file);
  }, [parseInput]);

  const handleBackendAnalyze = useCallback(async () => {
    if (!rawInput.trim()) return;
    setInfo("Analyzing via backend…");
    try {
      const res = await fetch(`${API_BASE}/tools/structure_tool`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "analyze", content: rawInput, format: inputFormat }),
      });
      const data = await res.json();
      if (data.result) setInfo(typeof data.result === "string" ? data.result : JSON.stringify(data.result, null, 2));
    } catch (e: any) {
      setInfo(`Backend error: ${e.message}`);
    }
  }, [API_BASE, rawInput, inputFormat]);

  const bonds = useMemo(() => structure && showBonds ? detectBonds(structure.atoms) : [], [structure, showBonds]);

  const center = useMemo(() => {
    if (!structure || structure.atoms.length === 0) return [0, 0, 0] as [number, number, number];
    const c = new THREE.Vector3();
    structure.atoms.forEach((a) => c.add(new THREE.Vector3(a.x, a.y, a.z)));
    c.divideScalar(structure.atoms.length);
    return [c.x, c.y, c.z] as [number, number, number];
  }, [structure]);

  return (
    <div className="flex h-full gap-4">
      {/* Left: 3D viewport */}
      <div className="flex flex-1 flex-col">
        {/* Toolbar */}
        <div className="mb-2 flex items-center gap-2">
          <select
            value={inputFormat}
            onChange={(e) => setInputFormat(e.target.value as any)}
            className="input-field text-xs w-24"
          >
            <option value="xyz">XYZ</option>
            <option value="poscar">POSCAR/CIF</option>
          </select>
          <button onClick={() => fileRef.current?.click()} className="btn-secondary text-xs gap-1.5">
            <Upload size={14} /> File
          </button>
          <input ref={fileRef} type="file" accept=".xyz,.poscar,.cif,.vasp" className="hidden" onChange={handleFileUpload} />
          <button onClick={handleLoad} className="btn-secondary text-xs">Load</button>
          <button onClick={handleBackendAnalyze} className="btn-secondary text-xs">Analyze</button>
          <div className="mx-2 h-4 w-px bg-border" />
          <button
            onClick={() => setShowBonds((p) => !p)}
            className={`rounded-lg p-1.5 text-xs transition-colors ${showBonds ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
            title="Toggle bonds"
          >
            {showBonds ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
          {structure?.lattice && (
            <button
              onClick={() => setShowUnitCell((p) => !p)}
              className={`rounded-lg p-1.5 text-xs transition-colors ${showUnitCell ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
              title="Toggle unit cell"
            >
              <Box size={14} />
            </button>
          )}
          <button
            onClick={() => setSelectedAtom(null)}
            className="rounded-lg p-1.5 text-text-muted hover:text-text-secondary"
            title="Deselect atom"
          >
            <Crosshair size={14} />
          </button>
        </div>

        {/* 3D Canvas */}
        <div className="flex-1 overflow-hidden rounded-xl border border-border bg-bg-secondary" style={{ minHeight: 300 }}>
          {structure ? (
            <Canvas camera={{ fov: 50, near: 0.1, far: 1000 }}>
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={0.8} />
              <pointLight position={[-10, -5, -10]} intensity={0.3} />
              <CameraFitter atoms={structure.atoms} />
              <OrbitControls target={center} enableDamping dampingFactor={0.1} />

              {structure.atoms.map((atom, i) => (
                <AtomSphere
                  key={i}
                  position={[atom.x, atom.y, atom.z]}
                  color={ELEMENT_COLORS[atom.symbol] ?? DEFAULT_COLOR}
                  radius={ELEMENT_RADII[atom.symbol] ?? DEFAULT_RADIUS}
                  label={atom.symbol}
                  onClick={() => setSelectedAtom(i)}
                />
              ))}

              {bonds.map(([i, j], k) => (
                <BondCylinder
                  key={`b${k}`}
                  start={[structure.atoms[i].x, structure.atoms[i].y, structure.atoms[i].z]}
                  end={[structure.atoms[j].x, structure.atoms[j].y, structure.atoms[j].z]}
                />
              ))}

              {showUnitCell && structure.lattice && <UnitCell lattice={structure.lattice} />}
            </Canvas>
          ) : (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Box size={48} className="text-text-muted opacity-30" />
              <p className="mt-4 text-sm font-medium text-text-secondary">No structure loaded</p>
              <p className="mt-1 max-w-xs text-xs text-text-muted">
                Paste XYZ or POSCAR data in the panel on the right, or upload a file to visualize the crystal structure.
              </p>
            </div>
          )}
        </div>

        {/* Info bar */}
        {info && (
          <div className="mt-2 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-xs text-text-secondary whitespace-pre-wrap">
            {info}
          </div>
        )}
      </div>

      {/* Right: Input + atom info */}
      <div className="flex w-72 flex-col gap-3">
        <div className="flex flex-1 flex-col">
          <label className="mb-1 text-xs font-medium text-text-secondary">Structure input</label>
          <textarea
            value={rawInput}
            onChange={(e) => setRawInput(e.target.value)}
            placeholder={inputFormat === "xyz" ? "3\nWater molecule\nO  0.000  0.000  0.117\nH  0.000  0.756 -0.469\nH  0.000 -0.756 -0.469" : "Si bulk\n1.0\n0.0 2.715 2.715\n2.715 0.0 2.715\n2.715 2.715 0.0\nSi\n2\nDirect\n0.0 0.0 0.0\n0.25 0.25 0.25"}
            className="flex-1 resize-none rounded-xl border border-border bg-bg-secondary p-3 font-mono text-xs text-text-primary outline-none focus:border-accent"
            style={{ minHeight: 160 }}
          />
        </div>

        {/* Selected atom */}
        {selectedAtom !== null && structure && (
          <div className="rounded-xl border border-border bg-bg-secondary p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-text-secondary">
              <Info size={14} />
              <span>Selected atom #{selectedAtom + 1}</span>
            </div>
            <div className="mt-2 space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-text-muted">Element</span>
                <span className="font-semibold" style={{ color: ELEMENT_COLORS[structure.atoms[selectedAtom].symbol] ?? DEFAULT_COLOR }}>
                  {structure.atoms[selectedAtom].symbol}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Position</span>
                <span className="font-mono text-text-secondary">
                  {structure.atoms[selectedAtom].x.toFixed(3)}, {structure.atoms[selectedAtom].y.toFixed(3)}, {structure.atoms[selectedAtom].z.toFixed(3)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Sample data button */}
        <div>
          <label className="mb-1 block text-xs font-medium text-text-secondary">Quick load</label>
          <div className="flex flex-col gap-1.5">
            <button
              onClick={() => {
                const xyz = "3\nWater molecule (H2O)\nO  0.000  0.000  0.117\nH  0.000  0.756 -0.469\nH  0.000 -0.756 -0.469";
                setRawInput(xyz); setInputFormat("xyz"); parseInput(xyz, "xyz");
              }}
              className="btn-secondary text-xs justify-start"
            >
              Water (H₂O)
            </button>
            <button
              onClick={() => {
                const poscar = `Si diamond
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25`;
                setRawInput(poscar); setInputFormat("poscar"); parseInput(poscar, "poscar");
              }}
              className="btn-secondary text-xs justify-start"
            >
              Si diamond
            </button>
            <button
              onClick={() => {
                const xyz = `5
Methane (CH4)
C  0.000  0.000  0.000
H  0.629  0.629  0.629
H -0.629 -0.629  0.629
H -0.629  0.629 -0.629
H  0.629 -0.629 -0.629`;
                setRawInput(xyz); setInputFormat("xyz"); parseInput(xyz, "xyz");
              }}
              className="btn-secondary text-xs justify-start"
            >
              Methane (CH₄)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
