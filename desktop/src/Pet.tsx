import { useEffect, useState, useRef, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const API_BASE = "http://localhost:8000";

type PetMood = "idle" | "thinking" | "working" | "success" | "error" | "sleeping" | "happy";

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
}

const MOOD_EMOJI: Record<PetMood, string> = {
  idle: "✨",
  thinking: "💭",
  working: "⚙️",
  success: "🎉",
  error: "💥",
  sleeping: "💤",
  happy: "🐦",
};

const TIPS = [
  "Tip: Use @coder to delegate coding tasks.",
  "Tip: Team mode splits objectives across agents.",
  "Tip: Set context budget to avoid huge prompts.",
  "Tip: Local-only mode keeps data on your machine.",
  "Tip: Drag me anywhere you like!",
  "Tip: Right-click for quick actions.",
];

function moodClass(mood: PetMood, hopping: boolean): string {
  if (hopping) return "animate-hop";
  switch (mood) {
    case "thinking":
      return "animate-bounce";
    case "working":
      return "animate-pulse";
    case "success":
    case "happy":
      return "animate-bounce";
    case "error":
      return "animate-shake";
    case "sleeping":
      return "animate-breathe";
    default:
      return "animate-float";
  }
}

function BirdAvatar({ mood }: { mood: PetMood }) {
  const eye = mood === "sleeping" ? (
    <path d="M38 38 Q42 42 46 38" stroke="#1f2937" strokeWidth="2" fill="none" />
  ) : (
    <circle cx="42" cy="40" r="3" fill="#1f2937" />
  );

  return (
    <svg viewBox="0 0 100 100" className="w-full h-full drop-shadow-xl">
      {/* Body */}
      <ellipse cx="50" cy="60" rx="28" ry="24" fill="#60a5fa" />
      {/* Belly */}
      <ellipse cx="50" cy="68" rx="18" ry="12" fill="#dbeafe" />
      {/* Head */}
      <circle cx="50" cy="38" r="20" fill="#60a5fa" />
      {/* Beak */}
      <path d="M66 36 L80 42 L66 48 Z" fill="#f59e0b" />
      {/* Eye */}
      {eye}
      {/* Wing */}
      <path
        d="M34 58 Q48 70 62 58 Q58 74 34 58"
        fill="#3b82f6"
      />
      {/* Feet */}
      <path d="M42 82 L42 92 M58 82 L58 92" stroke="#f59e0b" strokeWidth="3" strokeLinecap="round" />
      {/* Hair tuft */}
      <path d="M46 20 Q50 10 54 20" stroke="#2563eb" strokeWidth="3" fill="none" strokeLinecap="round" />
    </svg>
  );
}

export default function Pet() {
  const [mood, setMood] = useState<PetMood>("idle");
  const [message, setMessage] = useState<string>("Hi!");
  const [showBubble, setShowBubble] = useState(true);
  const [hopping, setHopping] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);
  const [activeTasks, setActiveTasks] = useState(0);
  const [recentEvents, setRecentEvents] = useState<PetState["recent_events"]>([]);
  const [tipIndex, setTipIndex] = useState(0);
  const [persistent, setPersistent] = useState(false);
  const [muted, setMuted] = useState(false);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const appWindow = useRef<ReturnType<typeof getCurrentWindow> | null>(null);
  const lastActivityRef = useRef<number>(Date.now());

  useEffect(() => {
    appWindow.current = getCurrentWindow();
  }, []);

  const clearAutoHide = useCallback(() => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = null;
  }, []);

  const autoHide = useCallback((ms: number) => {
    clearAutoHide();
    hideTimer.current = setTimeout(() => setShowBubble(false), ms);
  }, [clearAutoHide]);

  const updateFromState = useCallback((state: PetState) => {
    setMood(state.mood);
    setMessage(state.message);
    setActiveTasks(state.active_tasks || 0);
    setRecentEvents(state.recent_events || []);
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
        }
      } catch {
        // ignore malformed events
      }
    };
    es.onerror = () => {
      setBackendOnline(false);
      speak("Backend offline. Click to check.", "sleeping", true);
    };

    idleTimer.current = setInterval(() => {
      const idleMs = Date.now() - lastActivityRef.current;
      if (backendOnline && !persistent && !muted) {
        if (idleMs > 45000) {
          setMood("idle");
          setMessage(TIPS[tipIndex % TIPS.length]);
          setShowBubble(true);
          setTipIndex((i) => (i + 1) % TIPS.length);
          lastActivityRef.current = Date.now();
        } else if (idleMs > 20000 && mood !== "sleeping") {
          setMood("sleeping");
          setMessage("Zzz…");
          setShowBubble(true);
        }
      }
    }, 1000);

    return () => {
      es.close();
      if (hideTimer.current) clearTimeout(hideTimer.current);
      if (hopTimer.current) clearTimeout(hopTimer.current);
      if (idleTimer.current) clearInterval(idleTimer.current);
      document.body.style.background = "";
      document.documentElement.style.background = "";
    };
  }, [backendOnline, hop, mood, muted, persistent, speak, tipIndex, updateFromState]);

  const handlePointerDown = () => {
    appWindow.current?.startDragging();
  };

  const handleClick = () => {
    if (menuOpen) {
      setMenuOpen(false);
      return;
    }
    hop();
    if (!backendOnline) {
      speak("Trying to reconnect…", "thinking");
      return;
    }
    if (mood === "sleeping") {
      speak("Good morning!", "idle");
    } else if (mood === "error") {
      speak("I hope that helps!", "happy");
    } else {
      speak("Chirp!", "happy");
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen((open) => !open);
  };

  const openExternal = (path: string) => {
    // The main window is responsible for navigation; we just emit a hint via URL
    window.open(`${API_BASE}${path}`, "_blank");
  };

  const actions = [
    { label: "Feed 🌾", onClick: () => { hop(); speak("Yum!", "happy"); setMenuOpen(false); } },
    { label: mood === "sleeping" ? "Wake ☀️" : "Sleep 🌙", onClick: () => {
      if (mood === "sleeping") {
        speak("I'm awake!", "idle");
      } else {
        speak("Good night…", "sleeping");
      }
      setMenuOpen(false);
    }},
    { label: showBubble ? "Hide bubble 🤐" : "Show bubble 💬", onClick: () => {
      setShowBubble((s) => !s);
      setMenuOpen(false);
    }},
    { label: muted ? "Unmute tips 🔊" : "Mute tips 🔇", onClick: () => {
      setMuted((m) => !m);
      speak(muted ? "Tips are back!" : "Tips muted.", "happy");
      setMenuOpen(false);
    }},
    { label: "Open team 👥", onClick: () => { openExternal("/"); setMenuOpen(false); } },
    { label: "Open settings ⚙️", onClick: () => { openExternal("/"); setMenuOpen(false); } },
  ];

  return (
    <div
      className="pet-container"
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      {showBubble && (
        <div className={`pet-bubble ${persistent ? "pet-bubble-persistent" : ""}`}>
          <span className="pet-bubble-emoji">{MOOD_EMOJI[mood]}</span>
          <span className="pet-bubble-text">{message}</span>
        </div>
      )}
      <div className={`pet-avatar ${moodClass(mood, hopping)}`}>
        <BirdAvatar mood={mood} />
      </div>
      {menuOpen && (
        <div className="pet-menu">
          <div className="pet-status">
            <span className={`h-2 w-2 rounded-full ${backendOnline ? "bg-success" : "bg-error"}`} />
            <span>{backendOnline ? "Backend online" : "Backend offline"}</span>
            {activeTasks > 0 && <span className="text-text-muted">• {activeTasks} active</span>}
          </div>
          {recentEvents.slice(-3).map((ev, i) => (
            <div key={i} className="pet-status text-text-secondary">
              <span>{MOOD_EMOJI[ev.mood]}</span>
              <span className="truncate">{ev.message}</span>
            </div>
          ))}
          {actions.map((a) => (
            <button key={a.label} className="pet-menu-item" onClick={(e) => { e.stopPropagation(); a.onClick(); }}>
              {a.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
