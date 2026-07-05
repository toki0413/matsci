import { useState, useMemo, useRef, useCallback, useEffect } from "react";
import { Canvas, useThree, ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Text, Line, Html } from "@react-three/drei";
import * as THREE from "three";
import {
  Upload, Eye, EyeOff, Crosshair, Info, Box, Camera,
  Play, Pause, SkipBack, SkipForward, Zap, Activity, Radio,
} from "lucide-react";

/* ─────────────────────────────────────────────────────────────
 * 元素数据 (CPK 颜色 + 共价半径)
 * 与后端 viewer3d.py 保持一致, 但前端也内置一份用于离线渲染
 * ───────────────────────────────────────────────────────────── */
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
const BOND_TOLERANCE = 1.3; // 共价半径之和的放大系数

/* ─────────────────────────────────────────────────────────────
 * 类型定义
 * ───────────────────────────────────────────────────────────── */
interface Atom {
  element: string;
  position: [number, number, number];
  radius: number;
  color: string;
}
interface Cell {
  a: [number, number, number];
  b: [number, number, number];
  c: [number, number, number];
  scale?: number;
}
interface TrajectoryFrame {
  positions: [number, number, number][];
  energy: number | null;
  step: number;
}
type RepresentationMode = "ball-stick" | "space-fill" | "wireframe" | "surface";

/* ─────────────────────────────────────────────────────────────
 * 结构解析 (POSCAR / XYZ)
 * ───────────────────────────────────────────────────────────── */
function parseXYZ(text: string): { atoms: Atom[]; title: string } {
  const lines = text.trim().split("\n").map((l) => l.trim());
  const count = parseInt(lines[0], 10) || lines.length - 2;
  const title = lines[1] || "";
  const atoms: Atom[] = [];
  for (let i = 2; i < 2 + count && i < lines.length; i++) {
    const parts = lines[i].split(/\s+/);
    if (parts.length >= 4) {
      const sym = parts[0];
      atoms.push({
        element: sym,
        position: [+parts[1], +parts[2], +parts[3]],
        radius: ELEMENT_RADII[sym] ?? DEFAULT_RADIUS,
        color: ELEMENT_COLORS[sym] ?? DEFAULT_COLOR,
      });
    }
  }
  return { atoms, title };
}

function parsePOSCAR(text: string): { atoms: Atom[]; cell: Cell; title: string } {
  const lines = text.trim().split("\n");
  const title = lines[0]?.trim() || "";
  const scale = parseFloat(lines[1]?.trim() || "1");
  const a = lines[2].trim().split(/\s+/).map(Number) as [number, number, number];
  const b = lines[3].trim().split(/\s+/).map(Number) as [number, number, number];
  const c = lines[4].trim().split(/\s+/).map(Number) as [number, number, number];
  // 缩放
  a.forEach((_, i) => (a[i] *= scale));
  b.forEach((_, i) => (b[i] *= scale));
  c.forEach((_, i) => (c[i] *= scale));

  const species = lines[5].trim().split(/\s+/);
  const countsLine = lines[6].trim().split(/\s+/);
  const counts = countsLine.map(Number);

  let isDirect = true;
  let firstCoord = 8;
  // 处理没有元素行的退化情况
  if (species.every((s) => /^\d+$/.test(s))) {
    isDirect = lines[6]?.trim().toLowerCase().startsWith("d");
    firstCoord = 7;
  } else {
    isDirect = lines[7]?.trim().toLowerCase().startsWith("d") ||
               lines[7]?.trim().toLowerCase().startsWith("f");
  }

  const atoms: Atom[] = [];
  let idx = firstCoord;
  for (let s = 0; s < species.length; s++) {
    for (let j = 0; j < counts[s] && idx < lines.length; j++, idx++) {
      const coords = lines[idx].trim().split(/\s+/).map(Number);
      let pos: [number, number, number];
      if (isDirect) {
        pos = [
          coords[0] * a[0] + coords[1] * b[0] + coords[2] * c[0],
          coords[0] * a[1] + coords[1] * b[1] + coords[2] * c[1],
          coords[0] * a[2] + coords[1] * b[2] + coords[2] * c[2],
        ];
      } else {
        pos = [coords[0], coords[1], coords[2]];
      }
      atoms.push({
        element: species[s],
        position: pos,
        radius: ELEMENT_RADII[species[s]] ?? DEFAULT_RADIUS,
        color: ELEMENT_COLORS[species[s]] ?? DEFAULT_COLOR,
      });
    }
  }
  return { atoms, cell: { a, b, c, scale }, title };
}

function detectBonds(atoms: Atom[]): [number, number][] {
  const bonds: [number, number][] = [];
  const n = atoms.length;
  // 大体系跳过 O(n^2) 成键判定, 交给后端
  if (n > 3000) return bonds;
  for (let i = 0; i < n; i++) {
    const ri = (atoms[i].radius) * 1.8; // 显示半径放大
    for (let j = i + 1; j < n; j++) {
      const rj = (atoms[j].radius) * 1.8;
      const dx = atoms[i].position[0] - atoms[j].position[0];
      const dy = atoms[i].position[1] - atoms[j].position[1];
      const dz = atoms[i].position[2] - atoms[j].position[2];
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < (ri + rj) * BOND_TOLERANCE && dist > 0.4) {
        bonds.push([i, j]);
      }
    }
  }
  return bonds;
}

/* ─────────────────────────────────────────────────────────────
 * 3D 子组件
 * ───────────────────────────────────────────────────────────── */

// 单个原子球: 小体系用独立 mesh, 支持点击和力拖拽
function AtomMesh({
  atom, index, mode, selected, onSelect, onDragForce,
}: {
  atom: Atom; index: number; mode: RepresentationMode;
  selected: boolean; onSelect: (i: number) => void;
  onDragForce: (i: number, force: [number, number, number]) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const dragging = useRef(false);
  const dragStart = useRef<THREE.Vector3>(new THREE.Vector3());

  const radius = mode === "space-fill" ? atom.radius * 2.2 : atom.radius;
  const segments = mode === "wireframe" ? 12 : 24;

  const handlePointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    onSelect(index);
    if (e.shiftKey) {
      // Shift + 拖拽 = 施加力
      dragging.current = true;
      dragStart.current.copy(e.point);
      (e.target as Element).setPointerCapture?.(e.pointerId);
    }
  };

  const handlePointerMove = (e: ThreeEvent<PointerEvent>) => {
    if (!dragging.current) return;
    // 力 = 当前指针位置 - 起点, 转换到世界坐标
    const force = e.point.clone().sub(dragStart.current);
    onDragForce(index, [force.x, force.y, force.z]);
  };

  const handlePointerUp = (e: ThreeEvent<PointerEvent>) => {
    if (dragging.current) {
      dragging.current = false;
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    }
  };

  if (mode === "wireframe") {
    return (
      <mesh ref={meshRef} position={atom.position}>
        <sphereGeometry args={[radius, segments, segments]} />
        <meshBasicMaterial color={atom.color} wireframe />
      </mesh>
    );
  }

  return (
    <mesh
      ref={meshRef}
      position={atom.position}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      <sphereGeometry args={[radius, segments, segments]} />
      <meshStandardMaterial
        color={atom.color}
        roughness={0.35}
        metalness={0.15}
        emissive={selected ? "#ffaa00" : "#000000"}
        emissiveIntensity={selected ? 0.5 : 0}
      />
      {selected && (
        <Text
          position={[0, radius + 0.3, 0]}
          fontSize={0.28}
          color="#ffffff"
          outlineColor="#000000"
          outlineWidth={0.02}
          anchorX="center"
          anchorY="bottom"
        >
          {`${atom.element} #${index + 1}`}
        </Text>
      )}
    </mesh>
  );
}

// 大体系用 InstancedMesh, 牺牲交互换性能
function InstancedAtoms({ atoms, mode }: { atoms: Atom[]; mode: RepresentationMode }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current) return;
    atoms.forEach((atom, i) => {
      dummy.position.set(...atom.position);
      const r = mode === "space-fill" ? atom.radius * 2.2 : atom.radius;
      dummy.scale.setScalar(r / DEFAULT_RADIUS);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(i, dummy.matrix);
      const c = new THREE.Color(atom.color);
      meshRef.current!.setColorAt(i, c);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) meshRef.current.instanceColor.needsUpdate = true;
  }, [atoms, mode, dummy]);

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, atoms.length]}>
      <sphereGeometry args={[DEFAULT_RADIUS, 16, 16]} />
      <meshStandardMaterial roughness={0.35} metalness={0.15} />
    </instancedMesh>
  );
}

// 键: 圆柱体
function BondMesh({ start, end, color = "#888075" }: {
  start: [number, number, number]; end: [number, number, number]; color?: string;
}) {
  const { position, quaternion, length } = useMemo(() => {
    const s = new THREE.Vector3(...start);
    const e = new THREE.Vector3(...end);
    const mid = s.clone().add(e).multiplyScalar(0.5);
    const dir = e.clone().sub(s);
    const len = dir.length();
    const quat = new THREE.Quaternion();
    quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    return { position: mid, quaternion: quat, length: len };
  }, [start, end]);

  return (
    <mesh position={position} quaternion={quaternion}>
      <cylinderGeometry args={[0.05, 0.05, length, 8]} />
      <meshStandardMaterial color={color} roughness={0.5} metalness={0.1} />
    </mesh>
  );
}

// 键的 InstancedMesh 版本, 用于大体系
function InstancedBonds({ atoms, bonds }: { atoms: Atom[]; bonds: [number, number][] }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  useEffect(() => {
    if (!meshRef.current || bonds.length === 0) return;
    const up = new THREE.Vector3(0, 1, 0);
    bonds.forEach(([i, j], k) => {
      const s = new THREE.Vector3(...atoms[i].position);
      const e = new THREE.Vector3(...atoms[j].position);
      const mid = s.clone().add(e).multiplyScalar(0.5);
      const dir = e.clone().sub(s);
      const len = dir.length();
      dummy.position.copy(mid);
      dummy.quaternion.setFromUnitVectors(up, dir.normalize());
      dummy.scale.set(1, len, 1);
      dummy.updateMatrix();
      meshRef.current!.setMatrixAt(k, dummy.matrix);
    });
    meshRef.current.instanceMatrix.needsUpdate = true;
  }, [atoms, bonds, dummy]);

  if (bonds.length === 0) return null;
  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, bonds.length]}>
      <cylinderGeometry args={[0.05, 0.05, 1, 6]} />
      <meshStandardMaterial color="#888075" roughness={0.5} metalness={0.1} />
    </instancedMesh>
  );
}

// 晶胞线框
function UnitCell({ cell }: { cell: Cell }) {
  const edges = useMemo(() => {
    const o = new THREE.Vector3(0, 0, 0);
    const a = new THREE.Vector3(...cell.a);
    const b = new THREE.Vector3(...cell.b);
    const c = new THREE.Vector3(...cell.c);
    const corners = [o, a, b, c, a.clone().add(b), a.clone().add(c), b.clone().add(c), a.clone().add(b).add(c)];
    const edgeIdx: [number, number][] = [
      [0,1],[0,2],[0,3],[1,4],[1,5],[2,4],[2,6],[3,5],[3,6],[4,7],[5,7],[6,7],
    ];
    return edgeIdx.map(([i, j]) => [corners[i], corners[j]] as [THREE.Vector3, THREE.Vector3]);
  }, [cell]);

  return (
    <group>
      {edges.map(([s, e], i) => (
        <Line key={i} points={[s, e]} color="#5fa8ff" lineWidth={1.5} dashed dashSize={0.2} gapSize={0.1} />
      ))}
    </group>
  );
}

// 力箭头: 用户拖拽原子时显示
function ForceArrow({ origin, force }: { origin: [number, number, number]; force: [number, number, number] }) {
  const end = useMemo(() => {
    const o = new THREE.Vector3(...origin);
    const f = new THREE.Vector3(...force);
    return o.clone().add(f.multiplyScalar(3));
  }, [origin, force]);

  return (
    <group>
      <Line points={[new THREE.Vector3(...origin), end]} color="#ff4444" lineWidth={3} />
      {/* 箭头头部 */}
      <mesh position={end.toArray()}>
        <coneGeometry args={[0.15, 0.4, 8]} />
        <meshBasicMaterial color="#ff4444" />
      </mesh>
    </group>
  );
}

// 相机自动适配
function CameraFitter({ atoms }: { atoms: Atom[] }) {
  const { camera } = useThree();
  useMemo(() => {
    if (atoms.length === 0) return;
    const center = new THREE.Vector3();
    atoms.forEach((a) => center.add(new THREE.Vector3(...a.position)));
    center.divideScalar(atoms.length);
    let maxDist = 0;
    atoms.forEach((a) => {
      const d = center.distanceTo(new THREE.Vector3(...a.position));
      if (d > maxDist) maxDist = d;
    });
    const dist = Math.max(maxDist * 2.8, 5);
    (camera as THREE.PerspectiveCamera).position.set(
      center.x + dist * 0.5, center.y + dist * 0.3, center.z + dist
    );
    (camera as THREE.PerspectiveCamera).lookAt(center);
  }, [atoms, camera]);
  return null;
}

// 表面表示: 用半透明大球近似范德华表面
function SurfaceMesh({ atoms }: { atoms: Atom[] }) {
  return (
    <group>
      {atoms.map((atom, i) => (
        <mesh key={i} position={atom.position}>
          <sphereGeometry args={[atom.radius * 1.8, 16, 16]} />
          <meshStandardMaterial
            color={atom.color}
            transparent
            opacity={0.45}
            roughness={0.6}
            metalness={0.0}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

// HUD: 实时能量 / 温度 / 帧信息
function HUD({ energy, temperature, step, nAtoms, mode }: {
  energy: number; temperature: number; step: number; nAtoms: number; mode: RepresentationMode;
}) {
  return (
    <Html position={[-10, 8, 0]} style={{ pointerEvents: "none" }} zIndexRange={[10, 0]}>
      <div style={{
        background: "rgba(15, 20, 30, 0.85)",
        border: "1px solid #2a3a5a",
        borderRadius: "8px",
        padding: "10px 14px",
        color: "#c8d4e8",
        fontFamily: "Arial, sans-serif",
        fontSize: "12px",
        minWidth: "180px",
        backdropFilter: "blur(4px)",
      }}>
        <div style={{ fontWeight: "bold", color: "#5fa8ff", marginBottom: "6px", fontSize: "13px" }}>
          TELEMETRY
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "2px 8px" }}>
          <span>Energy</span>
          <span style={{ color: "#ffaa44", fontFamily: "monospace" }}>{energy.toFixed(4)} eV</span>
          <span>Temperature</span>
          <span style={{ color: "#ff6b6b", fontFamily: "monospace" }}>{temperature.toFixed(2)} K</span>
          <span>Step</span>
          <span style={{ color: "#8aff8a", fontFamily: "monospace" }}>{step}</span>
          <span>Atoms</span>
          <span style={{ color: "#c8d4e8", fontFamily: "monospace" }}>{nAtoms}</span>
          <span>Mode</span>
          <span style={{ color: "#c8d4e8" }}>{mode}</span>
        </div>
      </div>
    </Html>
  );
}

/* ─────────────────────────────────────────────────────────────
 * 主组件
 * ───────────────────────────────────────────────────────────── */
export default function MolecularViewer({ API_BASE }: { API_BASE: string }) {
  const [atoms, setAtoms] = useState<Atom[]>([]);
  const [cell, setCell] = useState<Cell | null>(null);
  const [title, setTitle] = useState<string>("");
  const [bonds, setBonds] = useState<[number, number][]>([]);
  const [rawInput, setRawInput] = useState("");
  const [inputFormat, setInputFormat] = useState<"xyz" | "poscar" | "cif">("xyz");
  const [mode, setMode] = useState<RepresentationMode>("ball-stick");
  const [showUnitCell, setShowUnitCell] = useState(true);
  const [showBonds, setShowBonds] = useState(true);
  const [showHUD, setShowHUD] = useState(true);
  const [selectedAtom, setSelectedAtom] = useState<number | null>(null);
  const [info, setInfo] = useState<string>("");
  const [forceVec, setForceVec] = useState<{ atom: number; force: [number, number, number] } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const canvasContainerRef = useRef<HTMLDivElement>(null);

  // 轨迹播放
  const [frames, setFrames] = useState<TrajectoryFrame[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);

  // 实时模拟 (WebSocket)
  const [liveMode, setLiveMode] = useState(false);
  const [liveEnergy, setLiveEnergy] = useState(0);
  const [liveTemp, setLiveTemp] = useState(0);
  const [liveStep, setLiveStep] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const livePositionsRef = useRef<[number, number, number][]>([]);

  const useInstancing = atoms.length > 1000;

  // 解析输入
  const parseInput = useCallback((text: string, fmt: "xyz" | "poscar" | "cif") => {
    try {
      let parsedAtoms: Atom[] = [];
      let parsedCell: Cell | null = null;
      let parsedTitle = "";
      if (fmt === "xyz") {
        const r = parseXYZ(text);
        parsedAtoms = r.atoms;
        parsedTitle = r.title;
      } else if (fmt === "poscar") {
        const r = parsePOSCAR(text);
        parsedAtoms = r.atoms;
        parsedCell = r.cell;
        parsedTitle = r.title;
      } else if (fmt === "cif") {
        // CIF 走后端解析更准确
        loadFromBackend(text, "cif");
        return;
      }
      if (parsedAtoms.length === 0) {
        setInfo("No atoms parsed.");
        return;
      }
      setAtoms(parsedAtoms);
      setCell(parsedCell);
      setTitle(parsedTitle);
      setBonds(detectBonds(parsedAtoms));
      setSelectedAtom(null);
      setFrames([]);
      const elements = new Set(parsedAtoms.map((a) => a.element));
      let infoStr = `${parsedAtoms.length} atoms, ${elements.size} elements (${[...elements].join(", ")})`;
      if (parsedCell) {
        const norm = (v: [number, number, number]) => Math.sqrt(v[0]**2 + v[1]**2 + v[2]**2);
        infoStr += `\nCell: |a|=${norm(parsedCell.a).toFixed(2)} |b|=${norm(parsedCell.b).toFixed(2)} |c|=${norm(parsedCell.c).toFixed(2)} Å`;
      }
      setInfo(infoStr);
    } catch (e: any) {
      setInfo(`Parse error: ${e.message}`);
    }
  }, []);

  // 后端加载 (支持 CIF)
  const loadFromBackend = useCallback(async (content: string, fmt: string) => {
    setInfo("Loading via backend…");
    try {
      const res = await fetch(`${API_BASE}/v1/viewer3d/load`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content, format: fmt }),
      });
      const data = await res.json();
      if (data.error) { setInfo(`Backend error: ${data.error}`); return; }
      const parsedAtoms: Atom[] = (data.atoms || []).map((a: any) => ({
        element: a.element,
        position: a.position,
        radius: ELEMENT_RADII[a.element] ?? DEFAULT_RADIUS,
        color: ELEMENT_COLORS[a.element] ?? DEFAULT_COLOR,
      }));
      setAtoms(parsedAtoms);
      setBonds(data.bonds || []);
      setCell(data.cell || null);
      setTitle(data.title || "");
      setSelectedAtom(null);
      setFrames([]);
      const elements = new Set(parsedAtoms.map((a) => a.element));
      setInfo(`${parsedAtoms.length} atoms, ${elements.size} elements (${[...elements].join(", ")})`);
    } catch (e: any) {
      setInfo(`Backend error: ${e.message}`);
    }
  }, [API_BASE]);

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
      const name = file.name.toLowerCase();
      const fmt = name.endsWith(".xyz") ? "xyz" : name.endsWith(".cif") ? "cif" : "poscar";
      setInputFormat(fmt);
      parseInput(text, fmt);
    };
    reader.readAsText(file);
  }, [parseInput]);

  // 截图导出 PNG
  const handleScreenshot = useCallback(() => {
    const canvas = canvasContainerRef.current?.querySelector("canvas");
    if (!canvas) return;
    const url = (canvas as HTMLCanvasElement).toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = `molecular-viewer-${Date.now()}.png`;
    a.click();
  }, []);

  // 加载轨迹
  const handleLoadTrajectory = useCallback(async () => {
    if (!rawInput.trim()) return;
    setInfo("Loading trajectory…");
    try {
      const res = await fetch(`${API_BASE}/v1/viewer3d/trajectory`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: rawInput, format: "xyz" }),
      });
      const data = await res.json();
      if (data.error) { setInfo(`Trajectory error: ${data.error}`); return; }
      const parsedFrames: TrajectoryFrame[] = data.frames || [];
      setFrames(parsedFrames);
      setCurrentFrame(0);
      setPlaying(false);
      setInfo(`Loaded ${parsedFrames.length} frames`);
    } catch (e: any) {
      setInfo(`Trajectory error: ${e.message}`);
    }
  }, [API_BASE, rawInput]);

  // 轨迹播放: 每 100ms 推进一帧
  useEffect(() => {
    if (!playing || frames.length === 0) return;
    const timer = setInterval(() => {
      setCurrentFrame((f) => (f + 1) % frames.length);
    }, 100);
    return () => clearInterval(timer);
  }, [playing, frames.length]);

  // 当播放轨迹时, 用当前帧的坐标覆盖原子位置
  const displayAtoms = useMemo(() => {
    if (frames.length === 0) return atoms;
    const frame = frames[currentFrame];
    if (!frame) return atoms;
    return atoms.map((a, i) => ({
      ...a,
      position: frame.positions[i] ?? a.position,
    }));
  }, [atoms, frames, currentFrame]);

  // WebSocket 实时模式
  const startLiveStream = useCallback(() => {
    if (!rawInput.trim()) {
      setInfo("Load a structure first to start live simulation.");
      return;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    // WebSocket URL: http -> ws, 加 /ws/viewer3d
    const wsBase = API_BASE.replace(/^http/, "ws");
    const ws = new WebSocket(`${wsBase}/ws/viewer3d`);
    wsRef.current = ws;
    setLiveMode(true);
    setInfo("Connecting to live simulation…");

    ws.onopen = () => {
      setInfo("Live simulation connected.");
      ws.send(JSON.stringify({
        type: "hello",
        content: rawInput,
        format: inputFormat,
        stream: true,
      }));
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "structure") {
          const parsedAtoms: Atom[] = (msg.atoms || []).map((a: any) => ({
            element: a.element,
            position: a.position,
            radius: ELEMENT_RADII[a.element] ?? DEFAULT_RADIUS,
            color: ELEMENT_COLORS[a.element] ?? DEFAULT_COLOR,
          }));
          setAtoms(parsedAtoms);
          setBonds(msg.bonds || []);
          setCell(msg.cell || null);
          setTitle(msg.title || "");
          livePositionsRef.current = parsedAtoms.map((a) => a.position);
        } else if (msg.type === "frame") {
          livePositionsRef.current = msg.positions;
          setLiveEnergy(msg.energy ?? 0);
          setLiveTemp(msg.temperature ?? 0);
          setLiveStep(msg.step ?? 0);
          // 触发重新渲染: 用一个 state 推一下
          setAtoms((prev) => prev.map((a, i) => ({
            ...a,
            position: msg.positions[i] ?? a.position,
          })));
        } else if (msg.type === "force_ack") {
          // 力已确认, 可以清掉箭头
        }
      } catch (e) {
        // 忽略解析错误
      }
    };
    ws.onerror = () => setInfo("WebSocket error.");
    ws.onclose = () => {
      setLiveMode(false);
      setInfo("Live simulation disconnected.");
    };
  }, [API_BASE, rawInput, inputFormat]);

  const stopLiveStream = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setLiveMode(false);
  }, []);

  // 发送力到后端
  const handleDragForce = useCallback((atomIdx: number, force: [number, number, number]) => {
    setForceVec({ atom: atomIdx, force });
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: "force",
        atom_idx: atomIdx,
        force,
      }));
    }
    // 2 秒后清除箭头
    setTimeout(() => setForceVec(null), 2000);
  }, []);

  // 卸载时关闭 WS
  useEffect(() => () => {
    if (wsRef.current) wsRef.current.close();
  }, []);

  const center = useMemo(() => {
    if (atoms.length === 0) return [0, 0, 0] as [number, number, number];
    const c = new THREE.Vector3();
    atoms.forEach((a) => c.add(new THREE.Vector3(...a.position)));
    c.divideScalar(atoms.length);
    return [c.x, c.y, c.z] as [number, number, number];
  }, [atoms]);

  // 示例结构
  const sampleStructures: Record<string, { content: string; format: "xyz" | "poscar" }> = {
    water: {
      content: "3\nWater molecule (H2O)\nO  0.000  0.000  0.117\nH  0.000  0.756 -0.469\nH  0.000 -0.756 -0.469",
      format: "xyz",
    },
    methane: {
      content: "5\nMethane (CH4)\nC  0.000  0.000  0.000\nH  0.629  0.629  0.629\nH -0.629 -0.629  0.629\nH -0.629  0.629 -0.629\nH  0.629 -0.629 -0.629",
      format: "xyz",
    },
    silicon: {
      content: `Si diamond
1.0
0.0 2.715 2.715
2.715 0.0 2.715
2.715 2.715 0.0
Si
2
Direct
0.0 0.0 0.0
0.25 0.25 0.25`,
      format: "poscar",
    },
  };

  return (
    <div className="flex h-full gap-4">
      {/* 左侧: 3D 视口 */}
      <div className="flex flex-1 flex-col">
        {/* 工具栏 */}
        <div className="mb-2 flex flex-wrap items-center gap-2">
          <select
            value={inputFormat}
            onChange={(e) => setInputFormat(e.target.value as any)}
            className="input-field text-xs w-24"
          >
            <option value="xyz">XYZ</option>
            <option value="poscar">POSCAR</option>
            <option value="cif">CIF</option>
          </select>
          <button onClick={() => fileRef.current?.click()} className="btn-secondary text-xs gap-1.5">
            <Upload size={14} /> File
          </button>
          <input ref={fileRef} type="file" accept=".xyz,.poscar,.cif,.vasp" className="hidden" onChange={handleFileUpload} />
          <button onClick={handleLoad} className="btn-secondary text-xs">Load</button>
          <button onClick={handleLoadTrajectory} className="btn-secondary text-xs gap-1.5">
            <Activity size={14} /> Trajectory
          </button>

          <div className="mx-1 h-4 w-px bg-border" />

          {/* 表示模式 */}
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as RepresentationMode)}
            className="input-field text-xs w-36"
            title="Representation mode"
          >
            <option value="ball-stick">Ball & Stick</option>
            <option value="space-fill">Space Filling</option>
            <option value="wireframe">Wireframe</option>
            <option value="surface">Surface</option>
          </select>

          <div className="mx-1 h-4 w-px bg-border" />

          <button
            onClick={() => setShowBonds((p) => !p)}
            className={`rounded-lg p-1.5 text-xs transition-colors ${showBonds ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
            title="Toggle bonds"
          >
            {showBonds ? <Eye size={14} /> : <EyeOff size={14} />}
          </button>
          {cell && (
            <button
              onClick={() => setShowUnitCell((p) => !p)}
              className={`rounded-lg p-1.5 text-xs transition-colors ${showUnitCell ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
              title="Toggle unit cell"
            >
              <Box size={14} />
            </button>
          )}
          <button
            onClick={() => setShowHUD((p) => !p)}
            className={`rounded-lg p-1.5 text-xs transition-colors ${showHUD ? "bg-bg-tertiary text-accent" : "text-text-muted hover:text-text-secondary"}`}
            title="Toggle HUD"
          >
            <Info size={14} />
          </button>
          <button
            onClick={handleScreenshot}
            className="rounded-lg p-1.5 text-text-muted hover:text-text-secondary"
            title="Export PNG"
          >
            <Camera size={14} />
          </button>

          <div className="mx-1 h-4 w-px bg-border" />

          {/* 实时模拟控制 */}
          {!liveMode ? (
            <button
              onClick={startLiveStream}
              className="btn-secondary text-xs gap-1.5"
              title="Start live simulation (WebSocket)"
            >
              <Radio size={14} /> Live
            </button>
          ) : (
            <button
              onClick={stopLiveStream}
              className="rounded-lg bg-red-500/20 px-2 py-1 text-xs text-red-400"
            >
              <Radio size={14} /> Stop
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

        {/* 轨迹播放栏 */}
        {frames.length > 0 && (
          <div className="mb-2 flex items-center gap-2 rounded-lg border border-border bg-bg-secondary px-3 py-1.5">
            <button onClick={() => setCurrentFrame((f) => Math.max(0, f - 1))} className="text-text-muted hover:text-text-secondary">
              <SkipBack size={14} />
            </button>
            <button onClick={() => setPlaying((p) => !p)} className="text-accent">
              {playing ? <Pause size={16} /> : <Play size={16} />}
            </button>
            <button onClick={() => setCurrentFrame((f) => Math.min(frames.length - 1, f + 1))} className="text-text-muted hover:text-text-secondary">
              <SkipForward size={14} />
            </button>
            <input
              type="range"
              min={0}
              max={frames.length - 1}
              value={currentFrame}
              onChange={(e) => setCurrentFrame(+e.target.value)}
              className="flex-1"
            />
            <span className="font-mono text-xs text-text-secondary">
              {currentFrame + 1}/{frames.length}
              {frames[currentFrame]?.energy != null && ` | E=${frames[currentFrame].energy.toFixed(3)}`}
            </span>
          </div>
        )}

        {/* 3D 画布 */}
        <div ref={canvasContainerRef} className="flex-1 overflow-hidden rounded-xl border border-border bg-bg-secondary" style={{ minHeight: 400 }}>
          {atoms.length > 0 ? (
            <Canvas camera={{ fov: 50, near: 0.1, far: 1000 }} gl={{ preserveDrawingBuffer: true }}>
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={0.8} />
              <pointLight position={[-10, -5, -10]} intensity={0.3} />
              <CameraFitter atoms={displayAtoms} />
              <OrbitControls target={center} enableDamping dampingFactor={0.1} />

              {/* 原子渲染: 大体系用 instanced mesh */}
              {useInstancing ? (
                <InstancedAtoms atoms={displayAtoms} mode={mode} />
              ) : (
                displayAtoms.map((atom, i) => (
                  <AtomMesh
                    key={i}
                    atom={atom}
                    index={i}
                    mode={mode}
                    selected={selectedAtom === i}
                    onSelect={setSelectedAtom}
                    onDragForce={handleDragForce}
                  />
                ))
              )}

              {/* 表面模式叠加半透明球 */}
              {mode === "surface" && <SurfaceMesh atoms={displayAtoms} />}

              {/* 键 */}
              {showBonds && mode !== "space-fill" && mode !== "surface" && bonds.length > 0 && (
                useInstancing ? (
                  <InstancedBonds atoms={displayAtoms} bonds={bonds} />
                ) : (
                  bonds.map(([i, j], k) => (
                    <BondMesh
                      key={`b${k}`}
                      start={displayAtoms[i].position}
                      end={displayAtoms[j].position}
                    />
                  ))
                )
              )}

              {/* 晶胞 */}
              {showUnitCell && cell && <UnitCell cell={cell} />}

              {/* 力箭头 */}
              {forceVec && forceVec.force.some((v) => Math.abs(v) > 0.01) && (
                <ForceArrow
                  origin={displayAtoms[forceVec.atom].position}
                  force={forceVec.force}
                />
              )}

              {/* HUD */}
              {showHUD && (
                <HUD
                  energy={liveMode ? liveEnergy : (frames[currentFrame]?.energy ?? 0)}
                  temperature={liveMode ? liveTemp : 0}
                  step={liveMode ? liveStep : currentFrame}
                  nAtoms={atoms.length}
                  mode={mode}
                />
              )}
            </Canvas>
          ) : (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <Box size={48} className="text-text-muted opacity-30" />
              <p className="mt-4 text-sm font-medium text-text-secondary">No structure loaded</p>
              <p className="mt-1 max-w-xs text-xs text-text-muted">
                Paste XYZ / POSCAR / CIF data on the right, or use a sample below. Shift+drag an atom to apply a force.
              </p>
            </div>
          )}
        </div>

        {/* 信息栏 */}
        {info && (
          <div className="mt-2 rounded-lg border border-border bg-bg-secondary px-3 py-2 text-xs text-text-secondary whitespace-pre-wrap">
            {title && <div className="mb-1 text-xs font-semibold text-accent">{title}</div>}
            {info}
          </div>
        )}
      </div>

      {/* 右侧: 输入 + 原子信息 */}
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

        {/* 选中的原子 */}
        {selectedAtom !== null && atoms[selectedAtom] && (
          <div className="rounded-xl border border-border bg-bg-secondary p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-text-secondary">
              <Info size={14} />
              <span>Atom #{selectedAtom + 1}</span>
            </div>
            <div className="mt-2 space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-text-muted">Element</span>
                <span className="font-semibold" style={{ color: atoms[selectedAtom].color }}>
                  {atoms[selectedAtom].element}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Position</span>
                <span className="font-mono text-text-secondary">
                  {atoms[selectedAtom].position.map((v) => v.toFixed(3)).join(", ")}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Radius</span>
                <span className="font-mono text-text-secondary">{atoms[selectedAtom].radius.toFixed(2)} Å</span>
              </div>
              <div className="mt-2 rounded bg-bg-tertiary p-2 text-[10px] text-text-muted">
                <Zap size={10} className="inline" /> Shift + drag this atom to apply a force
              </div>
            </div>
          </div>
        )}

        {/* 示例结构 */}
        <div>
          <label className="mb-1 block text-xs font-medium text-text-secondary">Quick load</label>
          <div className="flex flex-col gap-1.5">
            <button
              onClick={() => {
                const s = sampleStructures.water;
                setRawInput(s.content); setInputFormat(s.format); parseInput(s.content, s.format);
              }}
              className="btn-secondary text-xs justify-start"
            >
              Water (H₂O)
            </button>
            <button
              onClick={() => {
                const s = sampleStructures.methane;
                setRawInput(s.content); setInputFormat(s.format); parseInput(s.content, s.format);
              }}
              className="btn-secondary text-xs justify-start"
            >
              Methane (CH₄)
            </button>
            <button
              onClick={() => {
                const s = sampleStructures.silicon;
                setRawInput(s.content); setInputFormat(s.format); parseInput(s.content, s.format);
              }}
              className="btn-secondary text-xs justify-start"
            >
              Si diamond
            </button>
          </div>
        </div>

        {/* 实时遥测 */}
        {liveMode && (
          <div className="rounded-xl border border-accent/30 bg-accent/5 p-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-accent">
              <Radio size={14} className="animate-pulse" />
              <span>LIVE TELEMETRY</span>
            </div>
            <div className="mt-2 space-y-1 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-text-muted">Energy</span>
                <span className="text-accent">{liveEnergy.toFixed(4)} eV</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Temperature</span>
                <span className="text-red-400">{liveTemp.toFixed(2)} K</span>
              </div>
              <div className="flex justify-between">
                <span className="text-text-muted">Step</span>
                <span className="text-green-400">{liveStep}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
