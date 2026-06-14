import { useEffect, useState, useRef, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const API_BASE = "http://localhost:8000";

type PetMood = "idle" | "thinking" | "working" | "success" | "error" | "sleeping" | "happy";
type PetPersonality = "cheerful" | "nerdy" | "calm" | "sassy";

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

interface PersonalityPack {
  greeting: string;
  click: string[];
  sleep: string;
  wake: string;
  feed: string;
  tips: string[];
  offline: string;
  reconnect: string;
  errorSoothing: string;
}

const PERSONALITIES: Record<PetPersonality, PersonalityPack> = {
  cheerful: {
    greeting: "Hi! I'm {name}. Ready to help! 🌟",
    click: ["Chirp!", "Yay, you clicked me!", "Let's do some science!", "I'm so happy you're here!"],
    sleep: "Zzz… dreaming of crystals…",
    wake: "Good morning! Let's go!",
    feed: "Yum! That was delicious!",
    tips: [
      "Tip: Team mode lets multiple agents work together!",
      "Tip: You can drag me anywhere on the screen.",
      "Tip: Set a context budget to keep prompts small.",
      "Tip: Local-only mode keeps your data private.",
      "Tip: Try asking me about DFT or molecular dynamics!",
    ],
    offline: "Oh no, backend offline! Click me to check.",
    reconnect: "Let me try waking the backend up…",
    errorSoothing: "Don't worry, we can fix this together!",
  },
  nerdy: {
    greeting: "Greetings. I am {name}, your research assistant.",
    click: ["Beep boop.", "Computing… click acknowledged.", "Did you know silicon has a diamond cubic structure?", "Optimal click detected."],
    sleep: "Entering low-power standby mode…",
    wake: "System resumed. Awaiting input.",
    feed: "Energy intake sufficient. Thank you.",
    tips: [
      "Tip: Use `code_tool` for reproducible Python snippets.",
      "Tip: Context budget ≈ system prompt + tools + history tokens.",
      "Tip: Encrypted configs use Fernet + PBKDF2.",
      "Tip: vLLM and Ollama are valid local-only providers.",
      "Tip: Lean proofs can be checked with the `lean_tool`.",
    ],
    offline: "Backend connection lost. Diagnostics recommended.",
    reconnect: "Attempting to re-establish backend socket…",
    errorSoothing: "Error detected. Review logs for traceback.",
  },
  calm: {
    greeting: "Hello, I'm {name}. Take a deep breath. 🍃",
    click: ["Chirp.", "Peaceful click.", "All is well.", "Breathe in, breathe out."],
    sleep: "Resting quietly…",
    wake: "Welcome back.",
    feed: "Thank you. Nourished.",
    tips: [
      "Tip: Save your settings when switching models.",
      "Tip: A smaller context budget keeps responses focused.",
      "Tip: You can mute these tips in the menu.",
      "Tip: Right-click me for quick actions.",
      "Tip: Team mode is great for complex comparisons.",
    ],
    offline: "The backend seems quiet. Click to reconnect.",
    reconnect: "Gently checking the backend…",
    errorSoothing: "It's okay. Errors are just information.",
  },
  sassy: {
    greeting: "Hey, I'm {name}. Try not to break anything.",
    click: ["What?", "I'm busy being cute.", "Yes, human?", "You again? Fine."],
    sleep: "Do not disturb. Seriously.",
    wake: "Ugh, morning already?",
    feed: "About time you fed me.",
    tips: [
      "Tip: Stop clicking me and do some work.",
      "Tip: If you keep asking silly things, I'll mute myself.",
      "Tip: Team mode = make the AI do the teamwork.",
      "Tip: Local-only mode means no cloud snooping.",
      "Tip: Context budget stops you from crashing the LLM.",
    ],
    offline: "Backend's napping. Wake it up, will ya?",
    reconnect: "Poking the backend…",
    errorSoothing: "Well, that went wrong. Classic.",
  },
};

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
      <ellipse cx="50" cy="60" rx="28" ry="24" fill="#60a5fa" />
      <ellipse cx="50" cy="68" rx="18" ry="12" fill="#dbeafe" />
      <circle cx="50" cy="38" r="20" fill="#60a5fa" />
      <path d="M66 36 L80 42 L66 48 Z" fill="#f59e0b" />
      {eye}
      <path d="M34 58 Q48 70 62 58 Q58 74 34 58" fill="#3b82f6" />
      <path d="M42 82 L42 92 M58 82 L58 92" stroke="#f59e0b" strokeWidth="3" strokeLinecap="round" />
      <path d="M46 20 Q50 10 54 20" stroke="#2563eb" strokeWidth="3" fill="none" strokeLinecap="round" />
    </svg>
  );
}

function formatMsg(template: string, name: string) {
  return template.replace("{name}", name);
}

export default function Pet() {
  const [mood, setMood] = useState<PetMood>("idle");
  const [message, setMessage] = useState<string>("");
  const [showBubble, setShowBubble] = useState(true);
  const [hopping, setHopping] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);
  const [activeTasks, setActiveTasks] = useState(0);
  const [recentEvents, setRecentEvents] = useState<PetState["recent_events"]>([]);
  const [tipIndex, setTipIndex] = useState(0);
  const [persistent, setPersistent] = useState(false);
  const [muted, setMuted] = useState(false);
  const [petName, setPetName] = useState("Muninn");
  const [personality, setPersonality] = useState<PetPersonality>("cheerful");
  const [clickIndex, setClickIndex] = useState(0);
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const appWindow = useRef<ReturnType<typeof getCurrentWindow> | null>(null);
  const lastActivityRef = useRef<number>(Date.now());
  const pack = PERSONALITIES[personality] || PERSONALITIES.cheerful;

  useEffect(() => {
    appWindow.current = getCurrentWindow();
  }, []);

  useEffect(() => {
    if (!message) {
      setMessage(formatMsg(pack.greeting, petName));
    }
  }, [pack, petName]); // eslint-disable-line react-hooks/exhaustive-deps

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
    if (state.name) setPetName(state.name);
    if (state.personality && PERSONALITIES[state.personality]) {
      setPersonality(state.personality as PetPersonality);
    }
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
      speak(formatMsg(pack.offline, petName), "sleeping", true);
    };

    idleTimer.current = setInterval(() => {
      const idleMs = Date.now() - lastActivityRef.current;
      if (backendOnline && !persistent && !muted) {
        if (idleMs > 45000) {
          const tip = pack.tips[tipIndex % pack.tips.length];
          setMood("idle");
          setMessage(tip);
          setShowBubble(true);
          setTipIndex((i) => (i + 1) % pack.tips.length);
          lastActivityRef.current = Date.now();
        } else if (idleMs > 20000 && mood !== "sleeping") {
          setMood("sleeping");
          setMessage(formatMsg(pack.sleep, petName));
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
  }, [backendOnline, hop, mood, muted, persistent, pack, petName, speak, tipIndex, updateFromState]);

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
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen((open) => !open);
  };

  const actions = [
    { label: `Feed 🌾`, onClick: () => { hop(); speak(formatMsg(pack.feed, petName), "happy"); setMenuOpen(false); } },
    { label: mood === "sleeping" ? `Wake ☀️` : `Sleep 🌙`, onClick: () => {
      if (mood === "sleeping") {
        speak(formatMsg(pack.wake, petName), "idle");
      } else {
        speak(formatMsg(pack.sleep, petName), "sleeping");
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
            <span className="font-semibold">{petName}</span>
            <span className="text-text-muted">({personality})</span>
          </div>
          <div className="pet-status">
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
