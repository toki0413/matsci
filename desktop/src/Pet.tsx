import { useEffect, useState, useRef, useCallback, useMemo } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { invoke } from "@tauri-apps/api/core";
import {
  playPetChirp, playHappyCoo, playFeedTick, playLevelUp,
  startAmbient, stopAmbient, setMuted as setSoundMuted,
} from "./sounds";
import { getApiBase, setApiBase, getAuthToken } from "./lib/api-client";
import { ReconnectingWebSocket } from "./lib/ws-client";

// API_BASE is now managed by the shared api-client module, but we
// keep a local reference for SSE URL construction. The syncBackendUrl()
// call below keeps them in sync.

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
/* ── Raven Image Avatar (watercolor logo) ── */
function RavenAvatar({ mood: _mood, accessory: _accessory, imgRef }: { mood: PetMood; accessory: AccessoryId; imgRef?: React.Ref<HTMLDivElement> }) {
  return (
    <div ref={imgRef} className="raven-img-wrapper">
      <img src="/raven-logo.png" alt="Huginn" className="raven-img" draggable={false} />
    </div>
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
        <circle cx="65" cy="65" r="60" fill="none" stroke="rgba(42,37,32,0.08)" strokeWidth="2" />
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
  const [petName, setPetName] = useState(() => {
    try {
      const raw = localStorage.getItem("huginn:config");
      if (raw) {
        const cfg = JSON.parse(raw);
        if (cfg.pet_name) return cfg.pet_name;
      }
    } catch {
      // localStorage might not be available
    }
    return "Muninn";
  });
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
  const petWsRef = useRef<ReconnectingWebSocket | null>(null);
  const lastActivityRef = useRef<number>(Date.now());
  const ravenSvgRef = useRef<HTMLDivElement>(null);
  const particleContainerRef = useRef<HTMLDivElement>(null);
  const pack = PERSONALITIES[personality] || PERSONALITIES.cheerful;
  const xpMax = useMemo(() => xpForLevel(level), [level]);

  // Day/night cycle based on local time
  const [timeOfDay, setTimeOfDay] = useState<'dawn' | 'day' | 'dusk' | 'night'>(() => {
    const h = new Date().getHours();
    if (h >= 5 && h < 7) return 'dawn';
    if (h >= 7 && h < 17) return 'day';
    if (h >= 17 && h < 19) return 'dusk';
    return 'night';
  });
  useEffect(() => {
    const update = () => {
      const h = new Date().getHours();
      if (h >= 5 && h < 7) setTimeOfDay('dawn');
      else if (h >= 7 && h < 17) setTimeOfDay('day');
      else if (h >= 17 && h < 19) setTimeOfDay('dusk');
      else setTimeOfDay('night');
    };
    const id = setInterval(update, 60_000);
    return () => clearInterval(id);
  }, []);

  // Sync ambient sound with time-of-day
  useEffect(() => {
    if (!muted) {
      startAmbient(timeOfDay);
    }
    return () => stopAmbient();
  }, [timeOfDay, muted]);

  // Sync mute state with sound engine
  useEffect(() => {
    setSoundMuted(muted);
  }, [muted]);

  // Play level-up sound when level increases
  const prevLevelRef = useRef(level);
  useEffect(() => {
    if (level > prevLevelRef.current) {
      playLevelUp();
    }
    prevLevelRef.current = level;
  }, [level]);

  useEffect(() => {
    appWindow.current = getCurrentWindow();
  }, []);

  useEffect(() => {
    if (!message) {
      setMessage(formatMsg(pack.greeting, petName));
    }
  }, [pack, petName]); // eslint-disable-line react-hooks/exhaustive-deps

  // Blink cycle (subtle opacity dip for image-based avatar)
  useEffect(() => {
    if (mood === 'sleeping') return;
    let timeout: ReturnType<typeof setTimeout>;
    function scheduleNext() {
      const delay = 3000 + Math.random() * 4000;
      timeout = setTimeout(() => {
        if (ravenSvgRef.current) {
          ravenSvgRef.current.classList.add('raven-blink');
          setTimeout(() => ravenSvgRef.current?.classList.remove('raven-blink'), 250);
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

    // Sync backend port before opening SSE so we connect to the right URL.
    let es: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connectSSE = () => {
      const baseUrl = getApiBase();
      es = new EventSource(`${baseUrl}/events`);
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
        // Auto-reconnect with delay (previously no reconnect at all)
        es?.close();
        if (reconnectTimer) clearTimeout(reconnectTimer);
        reconnectTimer = setTimeout(connectSSE, 5_000);
      };
    };

    const init = async () => {
      try {
        const port: number = await invoke("get_backend_port");
        if (port && port > 0) {
          setApiBase(`http://localhost:${port}`);
        }
      } catch {
        // Tauri IPC not available — keep default
      }
      connectSSE();

      // WS for real-time pet_update pushes (mood, xp, level, hunger, happiness)
      const wsUrl = getApiBase().replace("http", "ws") + "/ws/agent";
      petWsRef.current?.close();
      petWsRef.current = new ReconnectingWebSocket({
        url: wsUrl,
        authToken: () => getAuthToken(),
        pingInterval: 30_000,
        onMessage: (data) => {
          if (typeof data !== "object" || data === null) return;
          const msg = data as { type?: string; mood?: string; xp?: number; level?: number; hunger?: number; happiness?: number };
          if (msg.type !== "pet_update") return;
          if (msg.mood) setMood(msg.mood as PetMood);
          if (msg.xp !== undefined) setExperience(msg.xp);
          if (msg.level !== undefined) setLevel(msg.level);
          if (msg.hunger !== undefined) setHunger(msg.hunger);
          if (msg.happiness !== undefined) setHappiness(msg.happiness);
        },
      });
      petWsRef.current.connect();
    };
    init();

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
      es?.close();
      petWsRef.current?.close();
      petWsRef.current = null;
      if (reconnectTimer) clearTimeout(reconnectTimer);
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
    playPetChirp();
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
    playFeedTick();
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
    playPetChirp();
    playHappyCoo();
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
        {/* Day/night scene background */}
        <div className={`pet-scene pet-scene-${timeOfDay}`} />
        {/* Ambient particles based on time of day */}
        {(timeOfDay === 'day' || timeOfDay === 'dawn') && Array.from({ length: 3 }, (_, i) => (
          <div key={`dust-${i}`} className="pet-ambient-particle ambient-dust"
            style={{ left: `${20 + i * 25}%`, top: `${15 + i * 10}%`, animationDelay: `${i * 2}s` }} />
        ))}
        {timeOfDay === 'night' && Array.from({ length: 4 }, (_, i) => (
          <div key={`fly-${i}`} className="pet-ambient-particle ambient-firefly"
            style={{ left: `${10 + i * 22}%`, top: `${10 + i * 15}%`, animationDelay: `${i * 1.3}s` }} />
        ))}
        {timeOfDay === 'dusk' && Array.from({ length: 2 }, (_, i) => (
          <div key={`speck-${i}`} className="pet-ambient-particle ambient-speck"
            style={{ left: `${30 + i * 30}%`, top: `${20 + i * 12}%`, animationDelay: `${i * 3}s` }} />
        ))}
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
          <RavenAvatar mood={mood} accessory={accessory} imgRef={ravenSvgRef} />
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
