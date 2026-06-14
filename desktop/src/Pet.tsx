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
      <img
        src="/pet.png"
        alt="MatSci pet"
        className={`pet-avatar ${moodClass(mood)}`}
        draggable={false}
      />
    </div>
  );
}
