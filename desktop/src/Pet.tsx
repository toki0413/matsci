import { useEffect, useState, useRef } from "react";

const API_BASE = "http://localhost:8000";

type PetMood = "idle" | "thinking" | "working" | "success" | "error" | "sleeping";

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
};

function moodClass(mood: PetMood): string {
  switch (mood) {
    case "thinking":
      return "animate-bounce";
    case "working":
      return "animate-pulse";
    case "success":
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
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

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
      document.body.style.background = "";
      document.documentElement.style.background = "";
    };
  }, []);

  return (
    <div className="pet-container">
      {showBubble && (
        <div className="pet-bubble">
          <span className="pet-bubble-emoji">{MOOD_EMOJI[mood]}</span>
          <span className="pet-bubble-text">{message}</span>
        </div>
      )}
      <div className={`pet-avatar ${moodClass(mood)}`}>
        <BirdAvatar mood={mood} />
      </div>
    </div>
  );
}
