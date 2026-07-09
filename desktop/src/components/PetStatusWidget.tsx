import { useTranslation } from "react-i18next";

export interface PetStatusState {
  mood: string;
  xp: number;
  level: number;
  hunger: number;
  happiness: number;
}

const MOOD_EMOJI: Record<string, string> = {
  idle: "\u2728",
  thinking: "\uD83D\uDCAD",
  working: "\u2699\uFE0F",
  success: "\uD83C\uDF89",
  error: "\uD83D\uDCA5",
  sleeping: "\uD83D\uDCA4",
  happy: "\uD83D\uDC26",
  eating: "\uD83C\uDF5E",
  hungry: "\uD83D\uDE22",
  levelup: "\u2B50",
};

const XP_PER_LEVEL_BASE = 100;

function xpForLevel(level: number): number {
  return Math.floor(XP_PER_LEVEL_BASE * Math.pow(1.15, level - 1));
}

export function PetStatusWidget({ petState }: { petState: PetStatusState | null }) {
  const { t } = useTranslation();

  if (!petState) return null;

  const mood = petState.mood || "idle";
  const isBusy = mood === "thinking" || mood === "working";
  const level = petState.level || 1;
  const xpMax = xpForLevel(level);
  const xpPct = Math.min(100, ((petState.xp || 0) / xpMax) * 100);
  const hungerPct = Math.min(100, petState.hunger || 0);
  const happyPct = Math.min(100, petState.happiness || 0);

  return (
    <div className="border-t border-border px-3 py-2">
      {/* mood + level + busy indicator */}
      <div className="flex items-center gap-2">
        <span className="text-base leading-none">{MOOD_EMOJI[mood] || "\u2728"}</span>
        <span className="flex h-5 min-w-[1.25rem] items-center justify-center rounded bg-accent/20 px-1 text-[11px] font-bold text-accent">
          Lv.{level}
        </span>
        {isBusy && (
          <span className="ml-auto flex items-center gap-1 text-[11px] text-text-muted">
            <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            {mood === "thinking" ? t("pet.thinking") : t("pet.working")}
          </span>
        )}
      </div>

      {/* XP bar */}
      <div className="mt-1.5 flex items-center gap-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-bg-tertiary">
          <div
            className="h-full rounded-full bg-accent transition-all duration-500"
            style={{ width: `${xpPct}%` }}
          />
        </div>
        <span className="text-[10px] tabular-nums text-text-muted">
          {petState.xp || 0}/{xpMax}
        </span>
      </div>

      {/* hunger / happiness mini-bars */}
      <div className="mt-1.5 flex items-center gap-3">
        <div className="flex items-center gap-1" title={t("pet.hunger")}>
          <span className="text-[10px]">{"\uD83C\uDF56"}</span>
          <div className="h-1 w-10 overflow-hidden rounded-full bg-bg-tertiary">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                hungerPct < 25 ? "bg-error" : "bg-warning"
              }`}
              style={{ width: `${hungerPct}%` }}
            />
          </div>
        </div>
        <div className="flex items-center gap-1" title={t("pet.mood")}>
          <span className="text-[10px]">{"\uD83D\uDC99"}</span>
          <div className="h-1 w-10 overflow-hidden rounded-full bg-bg-tertiary">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                happyPct < 25 ? "bg-error" : "bg-success"
              }`}
              style={{ width: `${happyPct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
