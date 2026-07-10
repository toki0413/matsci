/**
 * Compact inline 3D structure viewer for chat messages.
 * Reuses parsing logic from StructureViewer but in a minimal canvas.
 */
import { useMemo } from "react";
import * as THREE from "three";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

// Ponytail: duplicate the minimum parsing logic rather than extracting a shared
// module — StructureViewer already works, don't touch it. ~40 lines of overlap
// is cheaper than the refactor risk.

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
const ELEMENT_RADII: Record<string, number> = {
  H: 0.25, He: 0.31, C: 0.40, N: 0.38, O: 0.36, F: 0.32, S: 0.50, P: 0.44,
  Si: 0.46, Fe: 0.48, Ti: 0.46, Cu: 0.42, Zn: 0.42, Al: 0.50, Na: 0.68,
  K: 0.75, Ca: 0.66, Mg: 0.58, Cl: 0.48, Br: 0.56, I: 0.65,
};
const DEFAULT_COLOR = "#FF69B4";
const DEFAULT_RADIUS = 0.45;

interface Atom { symbol: string; x: number; y: number; z: number; }
interface Lattice { a: THREE.Vector3; b: THREE.Vector3; c: THREE.Vector3; }
interface StructureData { atoms: Atom[]; lattice?: Lattice; title?: string; }

function parseXYZ(text: string): StructureData {
  const lines = text.trim().split("\n").map(l => l.trim());
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
    a: new THREE.Vector3(...(lines[2]?.trim().split(/\s+/).map(Number) || [0,0,0])).multiplyScalar(scale),
    b: new THREE.Vector3(...(lines[3]?.trim().split(/\s+/).map(Number) || [0,0,0])).multiplyScalar(scale),
    c: new THREE.Vector3(...(lines[4]?.trim().split(/\s+/).map(Number) || [0,0,0])).multiplyScalar(scale),
  };
  const species = lines[5]?.trim().split(/\s+/) || [];
  const counts = lines[6]?.trim().split(/\s+/).map(Number) || [];
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

/** Check whether a string is XYZ or POSCAR format. Exported for ToolResultRenderer. */
export function detectStructure(content: string): "xyz" | "poscar" | null {
  const text = content.trim();
  if (!text) return null;
  const lines = text.split("\n");

  // XYZ: first line is a small integer, third line starts with element symbol
  const count = parseInt(lines[0]?.trim(), 10);
  if (!isNaN(count) && count > 0 && count < 10000 && lines.length >= count + 2) {
    const atomLine = lines[2]?.trim().split(/\s+/);
    if (atomLine && atomLine.length >= 4 && /^[A-Z][a-z]?$/.test(atomLine[0])) {
      return "xyz";
    }
  }

  // POSCAR: line 2 is a float (scale), lines 3-5 are 3 numbers each
  const scale = parseFloat(lines[1]?.trim() || "");
  if (!isNaN(scale) && scale > 0) {
    const v2 = lines[2]?.trim().split(/\s+/).map(Number);
    const v3 = lines[3]?.trim().split(/\s+/).map(Number);
    if (v2?.length === 3 && v3?.length === 3 && v2.every(n => !isNaN(n))) {
      return "poscar";
    }
  }

  return null;
}

function AtomMesh({ atom }: { atom: Atom }) {
  return (
    <mesh position={[atom.x, atom.y, atom.z]}>
      <sphereGeometry args={[ELEMENT_RADII[atom.symbol] ?? DEFAULT_RADIUS, 20, 20]} />
      <meshStandardMaterial color={ELEMENT_COLORS[atom.symbol] ?? DEFAULT_COLOR} roughness={0.35} metalness={0.15} />
    </mesh>
  );
}

function BondMesh({ a, b }: { a: Atom; b: Atom }) {
  const s = new THREE.Vector3(a.x, a.y, a.z);
  const e = new THREE.Vector3(b.x, b.y, b.z);
  const mid = s.clone().add(e).multiplyScalar(0.5);
  const dir = e.clone().sub(s);
  const len = dir.length();
  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <cylinderGeometry args={[0.05, 0.05, len, 6]} />
      <meshStandardMaterial color="#888075" roughness={0.5} />
    </mesh>
  );
}

function CellEdge({ s, e }: { s: THREE.Vector3; e: THREE.Vector3 }) {
  const mid = s.clone().add(e).multiplyScalar(0.5);
  const dir = e.clone().sub(s);
  const len = dir.length();
  const quat = new THREE.Quaternion();
  quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return (
    <mesh position={[mid.x, mid.y, mid.z]} quaternion={quat}>
      <cylinderGeometry args={[0.04, 0.04, len, 6]} />
      <meshStandardMaterial color="#aaa" transparent opacity={0.4} />
    </mesh>
  );
}

function CameraFitter({ atoms }: { atoms: Atom[] }) {
  const { camera } = useThree();
  useMemo(() => {
    if (atoms.length === 0) return;
    const center = new THREE.Vector3();
    atoms.forEach(a => center.add(new THREE.Vector3(a.x, a.y, a.z)));
    center.divideScalar(atoms.length);
    let maxDist = 0;
    atoms.forEach(a => {
      const d = center.distanceTo(new THREE.Vector3(a.x, a.y, a.z));
      if (d > maxDist) maxDist = d;
    });
    const dist = Math.max(maxDist * 2.5, 5);
    (camera as THREE.PerspectiveCamera).position.set(
      center.x + dist * 0.5, center.y + dist * 0.3, center.z + dist
    );
    (camera as THREE.PerspectiveCamera).lookAt(center);
  }, [atoms, camera]);
  return null;
}

export default function InlineStructure3D({ content }: { content: string }) {
  const structure = useMemo(() => {
    const fmt = detectStructure(content);
    if (!fmt) return null;
    try {
      return fmt === "xyz" ? parseXYZ(content) : parsePOSCAR(content);
    } catch {
      return null;
    }
  }, [content]);

  const bonds = useMemo(
    () => structure ? detectBonds(structure.atoms) : [],
    [structure]
  );

  const center = useMemo(() => {
    if (!structure || structure.atoms.length === 0) return [0, 0, 0] as [number, number, number];
    const c = new THREE.Vector3();
    structure.atoms.forEach(a => c.add(new THREE.Vector3(a.x, a.y, a.z)));
    c.divideScalar(structure.atoms.length);
    return [c.x, c.y, c.z] as [number, number, number];
  }, [structure]);

  if (!structure || structure.atoms.length === 0) return null;

  const elements = new Set(structure.atoms.map(a => a.symbol));
  const formula = [...elements].join(", ");

  return (
    <div className="rounded-lg border border-border overflow-hidden bg-bg-secondary">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border bg-bg-tertiary">
        <span className="text-xs font-medium text-text-secondary">
          {structure.atoms.length} atoms · {formula}
        </span>
        <span className="text-[10px] text-text-muted">
          {structure.lattice ? "crystal" : "molecule"}
        </span>
      </div>
      <div style={{ height: 280 }}>
        <Canvas camera={{ fov: 50, near: 0.1, far: 1000 }}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[10, 10, 5]} intensity={0.8} />
          <pointLight position={[-10, -5, -10]} intensity={0.3} />
          <CameraFitter atoms={structure.atoms} />
          <OrbitControls target={center} enableDamping dampingFactor={0.1} />
          {structure.atoms.map((atom, i) => <AtomMesh key={i} atom={atom} />)}
          {bonds.map(([i, j], k) => (
            <BondMesh key={`b${k}`} a={structure.atoms[i]} b={structure.atoms[j]} />
          ))}
          {structure.lattice && (
            <group>
              {(() => {
                const o = new THREE.Vector3(0,0,0);
                const { a, b, c } = structure.lattice;
                const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
                const edges: [number, number][] = [
                  [0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7],
                ];
                return edges.map(([i, j], k) => (
                  <CellEdge key={`e${k}`} s={corners[i]} e={corners[j]} />
                ));
              })()}
            </group>
          )}
        </Canvas>
      </div>
    </div>
  );
}
