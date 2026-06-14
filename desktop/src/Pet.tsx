import { useEffect, useState, useRef, useCallback } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

const API_BASE = "http://localhost:8000";

type PetMood = "idle" | "thinking" | "working" | "success" | "error" | "sleeping" | "happy";

interface PetState {
  mood: PetMood;
  message: string;
  idle_seconds: number;
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
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const appWindow = useRef<ReturnType<typeof getCurrentWindow> | null>(null);

  useEffect(() => {
    appWindow.current = getCurrentWindow();
  }, []);

  const clearMessage = useCallback(() => {
    if (hideTimer.current) clearTimeout(hideTimer.current);
    hideTimer.current = setTimeout(() => setShowBubble(false), 4000);
  }, []);

  const speak = useCallback((text: string, nextMood: PetMood = "happy") => {
    setMood(nextMood);
    setMessage(text);
    setShowBubble(true);
    clearMessage();
  }, [clearMessage]);

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
        if (data.type === "heartbeat") return;
        if (data.type === "state") {
          const s = data.state as PetState;
          setMood(s.mood);
          setMessage(s.message);
          setShowBubble(true);
          clearMessage();
        } else if (data.type === "event") {
          setMood(data.mood);
          setMessage(data.message);
          setShowBubble(true);
          if (hideTimer.current) clearTimeout(hideTimer.current);
          hideTimer.current = setTimeout(() => setShowBubble(false), 4000);
        }
      } catch {
        // ignore malformed events
      }
    };
    es.onerror = () => {
      setMood("sleeping");
      setMessage("Zzz… backend offline");
    };
    return () => {
      es.close();
      if (hideTimer.current) clearTimeout(hideTimer.current);
      if (hopTimer.current) clearTimeout(hopTimer.current);
      document.body.style.background = "";
      document.documentElement.style.background = "";
    };
  }, [clearMessage]);

  const handlePointerDown = () => {
    appWindow.current?.startDragging();
  };

  const handleClick = () => {
    if (menuOpen) {
      setMenuOpen(false);
      return;
    }
    hop();
    if (mood === "sleeping") {
      speak("Good morning!", "idle");
    } else {
      speak("Chirp!", "happy");
    }
  };

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    setMenuOpen((open) => !open);
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
  ];

  return (
    <div
      className="pet-container"
      onPointerDown={handlePointerDown}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
    >
      {showBubble && (
        <div className="pet-bubble">
          <span className="pet-bubble-emoji">{MOOD_EMOJI[mood]}</span>
          <span className="pet-bubble-text">{message}</span>
        </div>
      )}
      <div className={`pet-avatar ${moodClass(mood, hopping)}`}>
        <BirdAvatar mood={mood} />
      </div>
      {menuOpen && (
        <div className="pet-menu">
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
