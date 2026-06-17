import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const API_BASE = "http://localhost:8000";

type PetMood = "idle" | "thinking" | "working" | "success" | "error" | "sleeping" | "happy" | "eating" | "hungry" | "levelup";
type PetPersonality = "cheerful" | "nerdy" | "calm" | "sassy";
type AccessoryId = "none" | "crown" | "glasses" | "scarf";

interface PetState {
  mood: PetMood;
  message: string;
  idle_seconds: number;
  active_tasks: number;
  recent_events: Array<{
    timestamp: number;
    mood: PetMood;
    message: string;
    details?: Record<string, any>;
  }>;
  name: string;
  personality: PetPersonality;
  experience?: number;
  level?: number;
  hunger?: number;
  happiness?: number;
  accessories?: string[];
}

const MOOD_EMOJI: Record<PetMood, string> = {
  idle: "\u2728", thinking: "\uD83D\uDCAD", working: "\u2699\uFE0F",
  success: "\uD83C\uDF89", error: "\uD83D\uDCA5", sleeping: "\uD83D\uDCA4",
  happy: "\uD83D\uDC26", eating: "\uD83C\uDF5E", hungry: "\uD83D\uDE22",
  levelup: "\u2B50",
};

const ACCESSORIES: Record<AccessoryId, { label: string; minLevel: number }> = {
  none: { label: "None", minLevel: 0 },
  crown: { label: "Crown", minLevel: 5 },
  glasses: { label: "Glasses", minLevel: 3 },
  scarf: { label: "Scarf", minLevel: 7 },
};

interface PersonalityPack {
  greeting: string;
  click: string[];
  sleep: string;
  wake: string;
  feed: string;
  pet: string;
  tips: string[];
  offline: string;
  reconnect: string;
  errorSoothing: string;
  hungry: string;
}

const PERSONALITIES: Record<PetPersonality, PersonalityPack> = {
  cheerful: {
    greeting: "Hi! I'm {name}. Ready to help!",
    click: ["Chirp!", "Yay, you clicked me!", "Let's do some science!", "I'm so happy you're here!"],
    sleep: "Zzz... dreaming of crystals...",
    wake: "Good morning! Let's go!",
    feed: "Yum! That was delicious!",
    pet: "Aww, that feels nice!",
    tips: [
      "Tip: Team mode lets multiple agents work together!",
      "Tip: You can drag me anywhere on the screen.",
      "Tip: Set a context budget to keep prompts small.",
      "Tip: Local-only mode keeps your data private.",
      "Tip: Try asking me about DFT or molecular dynamics!",
    ],
    offline: "Oh no, backend offline! Click me to check.",
    reconnect: "Let me try waking the backend up...",
    errorSoothing: "Don't worry, we can fix this together!",
    hungry: "I'm getting hungry... feed me?",
  },
  nerdy: {
    greeting: "Greetings. I am {name}, your research assistant.",
    click: ["Beep boop.", "Computing... click acknowledged.", "Did you know silicon has a diamond cubic structure?", "Optimal click detected."],
    sleep: "Entering low-power standby mode...",
    wake: "System resumed. Awaiting input.",
    feed: "Energy intake sufficient. Thank you.",
    pet: "Tactile input registered. Pleasant.",
    tips: [
      "Tip: Use code_tool for reproducible Python snippets.",
      "Tip: Context budget = system prompt + tools + history tokens.",
      "Tip: Encrypted configs use Fernet + PBKDF2.",
      "Tip: vLLM and Ollama are valid local-only providers.",
      "Tip: Lean proofs can be checked with the lean_tool.",
    ],
    offline: "Backend connection lost. Diagnostics recommended.",
    reconnect: "Attempting to re-establish backend socket...",
    errorSoothing: "Error detected. Review logs for traceback.",
    hungry: "Energy reserves at {hunger}%. Recommend refueling.",
  },
  calm: {
    greeting: "Hello, I'm {name}. Take a deep breath.",
    click: ["Chirp.", "Peaceful click.", "All is well.", "Breathe in, breathe out."],
    sleep: "Resting quietly...",
    wake: "Welcome back.",
    feed: "Thank you. Nourished.",
    pet: "Gentle. Peaceful.",
    tips: [
      "Tip: Save your settings when switching models.",
      "Tip: A smaller context budget keeps responses focused.",
      "Tip: You can mute these tips in the menu.",
      "Tip: Right-click me for quick actions.",
      "Tip: Team mode is great for complex comparisons.",
    ],
    offline: "The backend seems quiet. Click to reconnect.",
    reconnect: "Gently checking the backend...",
    errorSoothing: "It's okay. Errors are just information.",
    hungry: "A little nourishment would be welcome.",
  },
  sassy: {
    greeting: "Hey, I'm {name}. Try not to break anything.",
    click: ["What?", "I'm busy being cute.", "Yes, human?", "You again? Fine."],
    sleep: "Do not disturb. Seriously.",
    wake: "Ugh, morning already?",
    feed: "About time you fed me.",
    pet: "Hey, watch the feathers!",
    tips: [
      "Tip: Stop clicking me and do some work.",
      "Tip: If you keep asking silly things, I'll mute myself.",
      "Tip: Team mode = make the AI do the teamwork.",
      "Tip: Local-only mode means no cloud snooping.",
      "Tip: Context budget stops you from crashing the LLM.",
    ],
    offline: "Backend's napping. Wake it up, will ya?",
    reconnect: "Poking the backend...",
    errorSoothing: "Well, that went wrong. Classic.",
    hungry: "Feed me or I'm going on strike.",
  },
};

const XP_PER_LEVEL_BASE = 100;
const XP_PER_SUCCESS = 15;
const HUNGER_DECAY_INTERVAL = 30000; // 30s
const MOOD_DECAY_INTERVAL = 45000;   // 45s
const HUNGER_DECAY_AMOUNT = 2;
const MOOD_DECAY_AMOUNT = 1;
const HUNGER_LOW_THRESHOLD = 25;
const MOOD_LOW_THRESHOLD = 25;

function xpForLevel(level: number): number {
  return Math.floor(XP_PER_LEVEL_BASE * Math.pow(1.15, level - 1));
}

function formatMsg(template: string, name: string, vars?: Record<string, string | number>) {
  let msg = template.replace("{name}", name);
  if (vars) {
    for (const [k, v] of Object.entries(vars)) {
      msg = msg.replace(`{${k}}`, String(v));
    }
  }
  return msg;
}

function moodClass(mood: PetMood, hopping: boolean): string {
  if (hopping) return "raven-hop";
  switch (mood) {
    case "thinking": return "raven-thinking";
    case "working": return "raven-working";
    case "success":
    case "happy": return "raven-happy";
    case "error": return "raven-error";
    case "sleeping":
    case "hungry": return "raven-sleeping";
    case "eating": return "raven-eating";
    case "levelup": return "raven-levelup";
    default: return "raven-idle";
  }
}

/* ── Accessory SVG fragments ── */
function AccessorySvg({ id }: { id: AccessoryId }) {
  if (id === "none") return null;
  if (id === "crown") {
    return (
      <g className="pet-accessory">
        <path d="M40 14 L42 8 L45 12 L48 6 L51 12 L54 8 L56 14" stroke="#fbbf24" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        <rect x="40" y="14" width="16" height="3" rx="1" fill="#fbbf24" opacity="0.8" />
      </g>
    );
  }
  if (id === "glasses") {
    return (
      <g className="pet-accessory">
        <circle cx="38" cy="32" r="6" stroke="#94a3b8" strokeWidth="1.2" fill="none" />
        <circle cx="52" cy="32" r="6" stroke="#94a3b8" strokeWidth="1.2" fill="none" />
        <line x1="44" y1="32" x2="46" y2="32" stroke="#94a3b8" strokeWidth="1.2" />
        <line x1="32" y1="32" x2="28" y2="30" stroke="#94a3b8" strokeWidth="1" />
        <rect x="35" y="29" width="6" height="6" rx="3" fill="rgba(148,163,184,0.15)" />
        <rect x="49" y="29" width="6" height="6" rx="3" fill="rgba(148,163,184,0.15)" />
      </g>
    );
  }
  if (id === "scarf") {
    return (
      <g className="pet-accessory">
        <path d="M32 44 Q40 50 48 48 Q56 46 64 50" stroke="#ef4444" strokeWidth="3" fill="none" strokeLinecap="round" />
        <path d="M34 46 Q42 52 48 50 Q54 48 62 52" stroke="#dc2626" strokeWidth="2" fill="none" strokeLinecap="round" opacity="0.6" />
        <path d="M58 48 L62 56 L58 54 L56 58" stroke="#ef4444" strokeWidth="2" fill="none" strokeLinecap="round" />
      </g>
    );
  }
  return null;
}

/* ── Raven SVG ── */
function RavenAvatar({ mood: _mood, accessory, svgRef }: { mood: PetMood; accessory: AccessoryId; svgRef?: React.Ref<SVGSVGElement> }) {
  return (
    <svg ref={svgRef} viewBox="0 0 100 105" className="w-full h-full raven-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="bodyGrad" cx="0.45" cy="0.3" r="0.65">
          <stop offset="0%" stopColor="#4a4a52" />
          <stop offset="50%" stopColor="#2d2d35" />
          <stop offset="100%" stopColor="#1a1a22" />
        </radialGradient>
        <radialGradient id="bellyGrad" cx="0.5" cy="0.35" r="0.55">
          <stop offset="0%" stopColor="#3d3d48" />
          <stop offset="100%" stopColor="#2a2a34" />
        </radialGradient>
        <radialGradient id="eyeGrad" cx="0.4" cy="0.35" r="0.55">
          <stop offset="0%" stopColor="#3a2a1a" />
          <stop offset="100%" stopColor="#0a0604" />
        </radialGradient>
        <radialGradient id="wingGrad" cx="0.35" cy="0.25" r="0.7">
          <stop offset="0%" stopColor="#3a3a4a" />
          <stop offset="40%" stopColor="#252530" />
          <stop offset="100%" stopColor="#15151e" />
        </radialGradient>
        <linearGradient id="beakGrad" x1="0" y1="0" x2="0.3" y2="1">
          <stop offset="0%" stopColor="#4a4a4a" />
          <stop offset="100%" stopColor="#1a1a1a" />
        </linearGradient>
        <linearGradient id="sheenGrad" x1="0.2" y1="0" x2="0.8" y2="1">
          <stop offset="0%" stopColor="#6a4a9a" stopOpacity={0.22} />
          <stop offset="40%" stopColor="#3a6a7a" stopOpacity={0.16} />
          <stop offset="100%" stopColor="#1a1a22" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="rimLight" x1="0" y1="0" x2="1" y2="0.5">
          <stop offset="0%" stopColor="#6a6a7a" stopOpacity={0.2} />
          <stop offset="50%" stopColor="#4a4a5a" stopOpacity={0.08} />
          <stop offset="100%" stopColor="#1a1a22" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="crestGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3a3a45" />
          <stop offset="100%" stopColor="#1a1a22" />
        </linearGradient>
        <filter id="ravenShadow" x="-10%" y="-5%" width="120%" height="130%">
          <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#000" floodOpacity="0.3" />
        </filter>
        <filter id="eyeGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur in="SourceGraphic" stdDeviation="1.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      {/* Ground shadow */}
      <ellipse cx="50" cy="100" rx="22" ry="3" fill="rgba(0,0,0,0.22)" />

      {/* Tail feathers (5 curved, raven-like) */}
      <path d="M 36 82 Q 30 90 28 96 Q 32 93 36 88 Z" fill="#1a1a22" />
      <path d="M 44 85 Q 40 94 38 99 Q 43 95 46 89 Z" fill="#1e1e28" />
      <path d="M 50 86 Q 50 96 50 100 Q 54 95 53 89 Z" fill="#1a1a22" />
      <path d="M 56 85 Q 60 94 62 99 Q 57 95 54 89 Z" fill="#1e1e28" />
      <path d="M 64 82 Q 70 90 72 96 Q 68 93 64 88 Z" fill="#1a1a22" />

      {/* Feet (dark raven claws) */}
      <g fill="#2a2a2a" stroke="#1a1a1a" strokeWidth="0.3">
        <path d="M 37 88 Q 34 91 30 95 L 33 93 Q 31 96 32 98 L 35 94 Q 37 96 38 98 L 37 93 Z" />
        <path d="M 63 88 Q 66 91 70 95 L 67 93 Q 69 96 68 98 L 65 94 Q 63 96 62 98 L 63 93 Z" />
      </g>

      {/* Body */}
      <ellipse cx="50" cy="52" rx="33" ry="38" fill="url(#bodyGrad)" filter="url(#ravenShadow)" />
      {/* Rim light on body edge */}
      <ellipse cx="38" cy="42" rx="30" ry="34" fill="url(#rimLight)" />
      {/* Iridescent sheen overlay (blue-purple raven shimmer) */}
      <ellipse cx="40" cy="38" rx="26" ry="30" fill="url(#sheenGrad)" />
      {/* Belly (subtle lighter area) */}
      <ellipse cx="50" cy="62" rx="18" ry="18" fill="url(#bellyGrad)" opacity={0.6} />

      {/* Throat hackles (distinctive raven feature) */}
      <g stroke="#1a1a22" fill="none" strokeWidth="1.4" strokeLinecap="round" opacity={0.45}>
        <path d="M 37 52 Q 34 58 37 64" />
        <path d="M 43 54 Q 40 60 42 66" />
        <path d="M 50 55 Q 50 62 50 68" />
        <path d="M 57 54 Q 60 60 58 66" />
        <path d="M 63 52 Q 66 58 63 64" />
      </g>

      {/* Feather texture on body (6 lines, 3 per side) */}
      <g stroke="#4a4a5a" fill="none" strokeWidth="0.5" opacity={0.18}>
        <path d="M 28 42 Q 34 40 40 42" />
        <path d="M 60 42 Q 66 40 72 42" />
        <path d="M 30 52 Q 36 50 42 52" />
        <path d="M 58 52 Q 64 50 70 52" />
        <path d="M 33 62 Q 38 60 43 62" />
        <path d="M 57 62 Q 62 60 67 62" />
      </g>

      {/* Left wing */}
      <g id="ravenWingL" className="raven-wing-left">
        <path d="M 18 44 Q 5 38 7 56 Q 9 72 18 80 Q 22 68 20 55 Q 19 48 18 44 Z" fill="url(#wingGrad)" />
        <path d="M 11 48 Q 9 56 11 64" fill="none" stroke="#4a4a5a" strokeWidth="0.5" opacity={0.2} />
        <path d="M 14 50 Q 12 58 14 66" fill="none" stroke="#4a4a5a" strokeWidth="0.4" opacity={0.18} />
        <path d="M 17 52 Q 16 60 17 68" fill="none" stroke="#4a4a5a" strokeWidth="0.4" opacity={0.15} />
        {/* Wing edge highlight (iridescent) */}
        <path d="M 18 44 Q 5 38 7 56" fill="none" stroke="#5a4a7a" strokeWidth="0.6" opacity={0.15} />
      </g>

      {/* Right wing */}
      <g id="ravenWingR" className="raven-wing-right">
        <path d="M 82 44 Q 95 38 93 56 Q 91 72 82 80 Q 78 68 80 55 Q 81 48 82 44 Z" fill="url(#wingGrad)" />
        <path d="M 89 48 Q 91 56 89 64" fill="none" stroke="#4a4a5a" strokeWidth="0.5" opacity={0.2} />
        <path d="M 86 50 Q 88 58 86 66" fill="none" stroke="#4a4a5a" strokeWidth="0.4" opacity={0.18} />
        <path d="M 83 52 Q 84 60 83 68" fill="none" stroke="#4a4a5a" strokeWidth="0.4" opacity={0.15} />
        <path d="M 82 44 Q 95 38 93 56" fill="none" stroke="#5a4a7a" strokeWidth="0.6" opacity={0.15} />
      </g>

      {/* Eyes group */}
      <g id="ravenEyes" className="raven-head">
        {/* Left eye */}
        <ellipse cx="36" cy="36" rx="10.5" ry="11" fill="#0a0604" opacity={0.35} />
        <circle cx="36" cy="36" r="10" fill="url(#eyeGrad)" />
        <circle cx="36" cy="37" r="7.5" fill="#12080a" />
        <circle cx="36" cy="36" r="4" fill="#8b5e14" opacity={0.2} />
        <circle cx="32" cy="32.5" r="3.5" fill="rgba(255,255,255,0.92)" />
        <circle cx="33.5" cy="34.5" r="1.8" fill="rgba(255,255,255,0.45)" />
        <circle cx="39.5" cy="39.5" r="1.3" fill="rgba(255,255,255,0.5)" />
        <circle cx="36" cy="36" r="1.5" fill="#fbbf24" opacity={0.25} />
        {/* Right eye */}
        <ellipse cx="64" cy="36" rx="10.5" ry="11" fill="#0a0604" opacity={0.35} />
        <circle cx="64" cy="36" r="10" fill="url(#eyeGrad)" />
        <circle cx="64" cy="37" r="7.5" fill="#12080a" />
        <circle cx="64" cy="36" r="4" fill="#8b5e14" opacity={0.2} />
        <circle cx="60" cy="32.5" r="3.5" fill="rgba(255,255,255,0.92)" />
        <circle cx="61.5" cy="34.5" r="1.8" fill="rgba(255,255,255,0.45)" />
        <circle cx="67.5" cy="39.5" r="1.3" fill="rgba(255,255,255,0.5)" />
        <circle cx="64" cy="36" r="1.5" fill="#fbbf24" opacity={0.25} />
      </g>

      {/* Beak group — raven-style: clean pointed */}
      <g id="ravenBeak">
        <path d="M 43 44 Q 50 42 57 44 Q 54 52 50 56 Q 46 52 43 44 Z" fill="url(#beakGrad)" />
        <path d="M 44 46.5 Q 50 48.5 56 46.5" fill="none" stroke="#555" strokeWidth="0.5" opacity={0.35} />
      </g>

      {/* Nostril dots */}
      <circle cx="48" cy="44" r="0.7" fill="#111" opacity={0.25} />
      <circle cx="52" cy="44" r="0.7" fill="#111" opacity={0.25} />

      {/* Subtle cheek glow */}
      <circle cx="24" cy="44" r="5" fill="#6a4a8a" opacity={0.12} />
      <circle cx="76" cy="44" r="5" fill="#6a4a8a" opacity={0.12} />

      {/* Head crest feathers (shaggy raven crown — 5 feathers) */}
      <path d="M 40 16 Q 36 2 42 9" fill="none" stroke="url(#crestGrad)" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M 46 14 Q 44 0 48 7" fill="none" stroke="#2a2a35" strokeWidth="3" strokeLinecap="round" />
      <path d="M 50 13 Q 50 -1 54 7" fill="none" stroke="url(#crestGrad)" strokeWidth="4" strokeLinecap="round" />
      <path d="M 54 14 Q 56 0 52 7" fill="none" stroke="#2a2a35" strokeWidth="3" strokeLinecap="round" />
      <path d="M 60 16 Q 64 2 58 9" fill="none" stroke="url(#crestGrad)" strokeWidth="3.5" strokeLinecap="round" />

      {/* Accessory slot */}
      <AccessorySvg id={accessory} />
    </svg>
  );
}

/* ── Particle types ── */
type ParticleKind = "star" | "xp" | "firework" | "zzz";

interface Particle {
  id: number;
  kind: ParticleKind;
  x: number;
  y: number;
  delay: number;
  color?: string;
  variant: number;
  size?: number;
  glowRadius?: number;
  fontSize?: number;
}

let particleIdCounter = 0;

function makeParticles(kind: ParticleKind, count: number): Particle[] {
  const arr: Particle[] = [];
  const colors = ["#fbbf24", "#0d9488", "#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#ec4899", "#06b6d4"];
  for (let i = 0; i < count; i++) {
    const size = (kind === "star" || kind === "xp") ? (3 + Math.random() * 3) : undefined;
    const glowRadius = kind === "firework" ? (4 + Math.random() * 8) : undefined;
    const fontSize = kind === "zzz" ? (10 + i * 3) : undefined;
    arr.push({
      id: ++particleIdCounter,
      kind,
      x: 35 + Math.random() * 30,
      y: 40 + Math.random() * 20,
      delay: Math.random() * 0.3,
      color: kind === "firework" ? colors[i % colors.length] : undefined,
      variant: (i % 5) + 1,
      size,
      glowRadius,
      fontSize,
    });
  }
  return arr;
}

/* ── Progress ring ── */
function ProgressRing({ progress, visible }: { progress: number; visible: boolean }) {
  const circumference = 2 * Math.PI * 60;
  const offset = circumference * (1 - progress);
  return (
    <div className={`pet-progress-ring ${visible ? "pet-progress-ring-visible" : ""}`}>
      <svg viewBox="0 0 130 130" className="w-full h-full" style={{ transform: "rotate(-90deg)" }}>
        <circle cx="65" cy="65" r="60" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="2" />
        <circle cx="65" cy="65" r="60" fill="none" stroke="var(--seed-primary, #3b82f6)" strokeWidth="2.5" strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 0.8s ease" }} />
      </svg>
    </div>
  );
}

/* ── Main Pet Component ── */
export default function Pet() {
  const [mood, setMood] = useState<PetMood>("idle");
  const [message, setMessage] = useState("");
  const [showBubble, setShowBubble] = useState(true);
  const [hopping, setHopping] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);
  const [activeTasks, setActiveTasks] = useState(0);
  const [, setRecentEvents] = useState<PetState["recent_events"]>([]);
  const [tipIndex, setTipIndex] = useState(0);
  const [persistent, setPersistent] = useState(false);
  const [muted, setMuted] = useState(false);
  const [petName, setPetName] = useState("Muninn");
  const [personality, setPersonality] = useState<PetPersonality>("cheerful");
  const [clickIndex, setClickIndex] = useState(0);
  const [experience, setExperience] = useState(0);
  const [level, setLevel] = useState(1);
  const [hunger, setHunger] = useState(80);
  const [happiness, setHappiness] = useState(80);
  const [accessory, setAccessory] = useState<AccessoryId>("none");
  const [particles, setParticles] = useState<Particle[]>([]);
  const [xpGainPopups, setXpGainPopups] = useState<Array<{ id: number; amount: number }>>([]);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const decayTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const appWindow = useRef<ReturnType<typeof getCurrentWindow> | null>(null);
  const lastActivityRef = useRef<number>(Date.now());
  const ravenSvgRef = useRef<SVGSVGElement>(null);
  const particleContainerRef = useRef<HTMLDivElement>(null);
  const pack = PERSONALITIES[personality] || PERSONALITIES.cheerful;
  const xpMax = useMemo(() => xpForLevel(level), [level]);

  useEffect(() => {
    appWindow.current = getCurrentWindow();
  }, []);

  useEffect(() => {
    if (!message) {
      setMessage(formatMsg(pack.greeting, petName));
    }
  }, [pack, petName]); // eslint-disable-line react-hooks/exhaustive-deps

  // Blink cycle
  useEffect(() => {
    if (mood === 'sleeping') return;
    let timeout: ReturnType<typeof setTimeout>;
    function scheduleNext() {
      const delay = 2500 + Math.random() * 3500;
      timeout = setTimeout(() => {
        if (ravenSvgRef.current) {
          ravenSvgRef.current.classList.add('raven-blink');
          setTimeout(() => ravenSvgRef.current?.classList.remove('raven-blink'), 200);
        }
        // 25% chance double blinking
        if (Math.random() < 0.25) {
          setTimeout(() => {
            if (ravenSvgRef.current) {
              ravenSvgRef.current.classList.add('raven-blink');
              setTimeout(() => ravenSvgRef.current?.classList.remove('raven-blink'), 200);
            }
          }, 400);
        }
        scheduleNext();
      }, delay);
    }
    scheduleNext();
    return () => clearTimeout(timeout);
  }, [mood]);

  const clearAutoHide = useCallback(() => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = null;
  }, []);

  const autoHide = useCallback((ms: number) => {
    clearAutoHide();
    hideTimer.current = setTimeout(() => setShowBubble(false), ms);
  }, [clearAutoHide]);

  const spawnParticles = useCallback((kind: ParticleKind, count: number, durationMs = 2000) => {
    const newP = makeParticles(kind, count);
    setParticles(prev => [...prev, ...newP]);
    setTimeout(() => {
      setParticles(prev => prev.filter(p => !newP.find(np => np.id === p.id)));
    }, durationMs);
  }, []);

  const showXpGain = useCallback((amount: number) => {
    const id = ++particleIdCounter;
    setXpGainPopups(prev => [...prev, { id, amount }]);
    setTimeout(() => setXpGainPopups(prev => prev.filter(p => p.id !== id)), 1100);
  }, []);

  function spawnHearts() {
    if (!particleContainerRef.current) return;
    const hearts = ['\u2665', '\u2661', '\u2764'];
    for (let i = 0; i < 4; i++) {
      const p = document.createElement('div');
      p.className = 'particle particle-heart particle-heart-fly-' + ((i % 3) + 1);
      p.textContent = hearts[Math.floor(Math.random() * hearts.length)];
      p.style.left = (40 + Math.random() * 20) + '%';
      p.style.top = (38 + Math.random() * 12) + '%';
      p.style.fontSize = (9 + Math.random() * 5) + 'px';
      p.style.animationDelay = (Math.random() * 0.3) + 's';
      particleContainerRef.current.appendChild(p);
    }
    setTimeout(() => {
      particleContainerRef.current?.querySelectorAll('.particle-heart').forEach(el => el.remove());
    }, 1500);
  }

  const gainXp = useCallback((amount: number) => {
    setExperience(prev => {
      let newXp = prev + amount;
      let lvl = level;
      let max = xpMax;
      while (newXp >= max) {
        newXp -= max;
        lvl++;
        max = xpForLevel(lvl);
      }
      if (lvl > level) {
        setLevel(lvl);
        spawnParticles("firework", 8, 2000);
        setMood("levelup");
        setMessage(`Level ${lvl}! New powers unlocked.`);
        setShowBubble(true);
        setPersistent(false);
        autoHide(4000);
      }
      return newXp;
    });
    showXpGain(amount);
    spawnParticles("xp", 4, 1800);
  }, [level, xpMax, spawnParticles, showXpGain, autoHide]);

  const updateFromState = useCallback((state: PetState) => {
    setMood(state.mood);
    setMessage(state.message);
    setActiveTasks(state.active_tasks || 0);
    setRecentEvents(state.recent_events || []);
    if (state.name) setPetName(state.name);
    if (state.personality && PERSONALITIES[state.personality]) {
      setPersonality(state.personality as PetPersonality);
    }
    if (state.experience !== undefined) setExperience(state.experience);
    if (state.level !== undefined) setLevel(state.level);
    if (state.hunger !== undefined) setHunger(state.hunger);
    if (state.happiness !== undefined) setHappiness(state.happiness);
    if (state.accessories?.length) setAccessory(state.accessories[0] as AccessoryId);
    lastActivityRef.current = Date.now() - (state.idle_seconds || 0) * 1000;
  }, []);

  const speak = useCallback((text: string, nextMood: PetMood = "happy", persist = false) => {
    setMood(nextMood);
    setMessage(text);
    setShowBubble(true);
    setPersistent(persist);
    lastActivityRef.current = Date.now();
    if (persist) {
      clearAutoHide();
    } else {
      autoHide(nextMood === "error" ? 8000 : 4000);
    }
  }, [autoHide, clearAutoHide]);

  const hop = useCallback(() => {
    setHopping(true);
    if (hopTimer.current) clearTimeout(hopTimer.current);
    hopTimer.current = setTimeout(() => setHopping(false), 500);
  }, []);

  // SSE + idle + decay timers
  useEffect(() => {
    document.body.style.background = "transparent";
    document.documentElement.style.background = "transparent";

    const es = new EventSource(`${API_BASE}/events`);
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        setBackendOnline(true);
        if (data.type === "heartbeat" && data.state) {
          updateFromState(data.state as PetState);
          return;
        }
        if (data.type === "state") {
          updateFromState(data.state as PetState);
        } else if (data.type === "event") {
          const moodEvt = data.mood as PetMood;
          const msg = data.message as string;
          const persist = moodEvt === "error";
          speak(msg, moodEvt, persist);
          hop();
          // Auto XP gain on success
          if (moodEvt === "success") {
            gainXp(XP_PER_SUCCESS);
            spawnParticles("star", 6, 2000);
          }
        }
      } catch {
        // ignore malformed events
      }
    };
    es.onerror = () => {
      setBackendOnline(false);
      speak(formatMsg(pack.offline, petName), "sleeping", true);
    };

    idleTimer.current = setInterval(() => {
      const idleMs = Date.now() - lastActivityRef.current;
      if (backendOnline && !persistent && !muted) {
        if (idleMs > 45000) {
          // Show tip
          const tip = pack.tips[tipIndex % pack.tips.length];
          setMood("idle");
          setMessage(tip);
          setShowBubble(true);
          setTipIndex((i) => (i + 1) % pack.tips.length);
          lastActivityRef.current = Date.now();
        } else if (hunger < HUNGER_LOW_THRESHOLD) {
          // Hungry warning
          setMood("hungry");
          setMessage(formatMsg(pack.hungry, petName, { hunger }));
          setShowBubble(true);
        } else if (idleMs > 20000 && mood !== "sleeping") {
          setMood("sleeping");
          setMessage(formatMsg(pack.sleep, petName));
          setShowBubble(true);
          spawnParticles("zzz", 3, 3500);
        }
      }
    }, 1000);

    // Hunger and mood decay
    decayTimer.current = setInterval(() => {
      setHunger(prev => Math.max(0, prev - HUNGER_DECAY_AMOUNT));
      setHappiness(prev => Math.max(0, prev - MOOD_DECAY_AMOUNT));
    }, Math.min(HUNGER_DECAY_INTERVAL, MOOD_DECAY_INTERVAL));

    return () => {
      es.close();
      if (hideTimer.current) clearTimeout(hideTimer.current);
      if (hopTimer.current) clearTimeout(hopTimer.current);
      if (idleTimer.current) clearInterval(idleTimer.current);
      if (decayTimer.current) clearInterval(decayTimer.current);
      document.body.style.background = "";
      document.documentElement.style.background = "";
    };
  }, [backendOnline, hop, mood, muted, persistent, pack, petName, speak, tipIndex, updateFromState, gainXp, spawnParticles, hunger]);

  const handlePointerDown = () => {
    appWindow.current?.startDragging();
  };

  const handleClick = () => {
    if (menuOpen) {
      setMenuOpen(false);
      return;
    }
    hop();
    spawnHearts();
    if (!backendOnline) {
      speak(formatMsg(pack.reconnect, petName), "thinking");
      return;
    }
    if (mood === "sleeping") {
      speak(formatMsg(pack.wake, petName), "idle");
    } else if (mood === "error") {
      speak(formatMsg(pack.errorSoothing, petName), "happy");
    } else {
      const replies = pack.click;
      const reply = replies[clickIndex % replies.length];
      setClickIndex((i) => (i + 1) % replies.length);
      speak(reply, "happy");
      setHappiness(prev => Math.min(100, prev + 5));
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen((open) => !open);
  };

  const cycleAccessory = () => {
    const ids: AccessoryId[] = ["none", "crown", "glasses", "scarf"];
    const idx = ids.indexOf(accessory);
    const next = ids[(idx + 1) % ids.length];
    setAccessory(next);
    speak(next === "none" ? "No accessories." : `Wearing ${ACCESSORIES[next].label}!`, "happy");
  };

  const doFeed = () => {
    hop();
    setMood("eating");
    setMessage(formatMsg(pack.feed, petName));
    setShowBubble(true);
    setHunger(prev => Math.min(100, prev + 25));
    spawnParticles("star", 4, 1500);
    setMenuOpen(false);
    setTimeout(() => { setMood("happy"); }, 1200);
  };

  const doPet = () => {
    hop();
    spawnHearts();
    speak(formatMsg(pack.pet, petName), "happy");
    setHappiness(prev => Math.min(100, prev + 15));
    setMenuOpen(false);
  };

  const doSleepToggle = () => {
    if (mood === "sleeping") {
      speak(formatMsg(pack.wake, petName), "idle");
    } else {
      speak(formatMsg(pack.sleep, petName), "sleeping");
      spawnParticles("zzz", 3, 3500);
    }
    setMenuOpen(false);
  };

  const showStatus = () => {
    speak(`Lv.${level} | XP: ${experience}/${xpMax} | Hunger: ${hunger}% | Mood: ${happiness}%`, "idle", true);
    setMenuOpen(false);
  };

  const progressVisible = activeTasks > 0 || mood === "working";
  const progressValue = activeTasks > 0 ? Math.min(activeTasks * 0.3, 1) : 0.6;

  return (
    <div
      className="pet-container"
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      {/* Speech Bubble */}
      {showBubble && (
        <div className={`pet-bubble ${persistent ? "pet-bubble-persistent" : ""}`}>
          <span className="pet-bubble-emoji">{MOOD_EMOJI[mood]}</span>
          <span className="pet-bubble-text">{message}</span>
        </div>
      )}

      {/* Task indicator */}
      {activeTasks > 0 && (
        <div className="pet-task-indicator">
          <span className="pet-task-dot" />
          <span>{activeTasks} {activeTasks === 1 ? "task" : "tasks"}</span>
        </div>
      )}

      {/* XP gain popups */}
      {xpGainPopups.map(p => (
        <div key={p.id} className="pet-xp-gain">+{p.amount} XP</div>
      ))}

      {/* Raven avatar with particles & progress ring */}
      <div className="pet-avatar-wrapper">
        <ProgressRing progress={progressValue} visible={progressVisible} />
        <div className="pet-particle-layer" ref={particleContainerRef}>
          {particles.map(p => (
            <div key={p.id}
              className={`pet-particle pet-particle-${p.kind} pet-particle-fly-${p.variant}`}
              style={{
                left: `${p.x}%`,
                top: `${p.y}%`,
                animationDelay: `${p.delay}s`,
                ...(p.size ? { width: `${p.size}px`, height: `${p.size}px` } : {}),
                ...(p.color ? { background: p.color, boxShadow: `0 0 ${p.glowRadius || 8}px ${p.color}` } : {}),
                ...(p.fontSize ? { fontSize: `${p.fontSize}px` } : {}),
              }}
            >
              {p.kind === "zzz" ? "z" : undefined}
            </div>
          ))}
        </div>
        <div className={`pet-avatar ${moodClass(mood, hopping)}`}>
          <RavenAvatar mood={mood} accessory={accessory} svgRef={ravenSvgRef} />
        </div>
      </div>

      {/* Status bars */}
      <div className="pet-status-panel">
        {/* XP bar */}
        <div className="pet-xp-bar-wrapper">
          <div className="pet-level-badge">{level}</div>
          <div className="pet-xp-bar">
            <div className="pet-xp-bar-fill" style={{ width: `${(experience / xpMax) * 100}%` }} />
          </div>
          <span className="pet-xp-label">{experience}/{xpMax}</span>
        </div>
        {/* Stat bars */}
        <div className="pet-stat-bars">
          <div className="pet-stat-bar">
            <span className="pet-stat-icon">{"\uD83C\uDF56"}</span>
            <div className="pet-stat-track">
              <div className={`pet-stat-fill pet-stat-fill-hunger ${hunger < HUNGER_LOW_THRESHOLD ? "pet-stat-fill-low" : ""}`} style={{ width: `${hunger}%` }} />
            </div>
          </div>
          <div className="pet-stat-bar">
            <span className="pet-stat-icon">{"\uD83D\uDC99"}</span>
            <div className="pet-stat-track">
              <div className={`pet-stat-fill pet-stat-fill-mood ${happiness < MOOD_LOW_THRESHOLD ? "pet-stat-fill-low" : ""}`} style={{ width: `${happiness}%` }} />
            </div>
          </div>
        </div>
      </div>

      {/* Context menu */}
      {menuOpen && (
        <div className="pet-menu">
          <div className="pet-menu-header">
            <span className={`pet-menu-dot ${backendOnline ? "pet-menu-dot-online" : "pet-menu-dot-offline"}`} />
            <span className="pet-menu-name">{petName}</span>
            <span className="pet-menu-level">Lv.{level}</span>
          </div>
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); doFeed(); }}>
            <span className="pet-menu-item-icon">{"\uD83C\uDF3E"}</span> Feed
          </button>
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); doPet(); }}>
            <span className="pet-menu-item-icon">{"\uD83E\uDD1A"}</span> Pet
          </button>
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); doSleepToggle(); }}>
            <span className="pet-menu-item-icon">{mood === "sleeping" ? "\u2600\uFE0F" : "\uD83C\uDF19"}</span>
            {mood === "sleeping" ? "Wake" : "Sleep"}
          </button>
          <div className="pet-menu-divider" />
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); cycleAccessory(); }}>
            <span className="pet-menu-item-icon">{"\uD83C\uDFA9"}</span> Accessories
          </button>
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); showStatus(); }}>
            <span className="pet-menu-item-icon">{"\uD83D\uDCCA"}</span> Status
          </button>
          <div className="pet-menu-divider" />
          <button className="pet-menu-item" onClick={(e) => { e.stopPropagation(); setShowBubble(s => !s); setMenuOpen(false); }}>
            <span className="pet-menu-item-icon">{showBubble ? "\uD83E\uDD10" : "\uD83D\uDCAC"}</span>
            {showBubble ? "Hide bubble" : "Show bubble"}
          </button>
          <button className="pet-menu-item" onClick={(e) => {
            e.stopPropagation();
            setMuted(m => !m);
            speak(muted ? "Tips are back!" : "Tips muted.", "happy");
            setMenuOpen(false);
          }}>
            <span className="pet-menu-item-icon">{muted ? "\uD83D\uDD0A" : "\uD83D\uDD07"}</span>
            {muted ? "Unmute tips" : "Mute tips"}
          </button>
        </div>
      )}
    </div>
  );
}
