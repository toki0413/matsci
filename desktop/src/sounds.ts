/**
 * Pet Sound Engine — Web Audio API synthesized sounds
 * Zero external audio file dependencies.
 */

let ctx: AudioContext | null = null;
let ambientNodes: { osc: OscillatorNode; gain: GainNode }[] = [];
let ambientInterval: ReturnType<typeof setInterval> | null = null;
let muted = false;

function getCtx(): AudioContext {
  if (!ctx) {
    ctx = new AudioContext();
  }
  if (ctx.state === 'suspended') {
    ctx.resume();
  }
  return ctx;
}

export function setMuted(m: boolean) {
  muted = m;
  if (m) stopAmbient();
}

// ── Helpers ──

function playTone(
  freq: number,
  type: OscillatorType,
  duration: number,
  volume: number,
  delay = 0,
  freqEnd?: number,
) {
  if (muted) return;
  const c = getCtx();
  const now = c.currentTime + delay;

  const osc = c.createOscillator();
  const gain = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, now);
  if (freqEnd !== undefined) {
    osc.frequency.exponentialRampToValueAtTime(freqEnd, now + duration);
  }

  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(volume, now + 0.01);
  gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

  osc.connect(gain);
  gain.connect(c.destination);
  osc.start(now);
  osc.stop(now + duration + 0.05);
}

// ── Interaction Sounds ──

/** Gentle chirp when petting/clicking the raven */
export function playPetChirp() {
  // Two quick ascending notes (bird-like)
  playTone(1200, 'sine', 0.08, 0.12, 0, 1800);
  playTone(1500, 'sine', 0.1, 0.1, 0.07, 2000);
}

/** Soft coo for happy mood */
export function playHappyCoo() {
  playTone(600, 'sine', 0.15, 0.08, 0, 800);
  playTone(700, 'sine', 0.12, 0.06, 0.12, 900);
}

// ── Event Sounds ──

/** Ascending arpeggio for level up (3 notes) */
export function playLevelUp() {
  playTone(523, 'sine', 0.2, 0.12, 0);       // C5
  playTone(659, 'sine', 0.2, 0.12, 0.15);     // E5
  playTone(784, 'sine', 0.35, 0.14, 0.3);     // G5
  playTone(1047, 'triangle', 0.5, 0.08, 0.4); // C6 shimmer
}

/** Bright chime for task completion */
export function playTaskComplete() {
  playTone(880, 'sine', 0.15, 0.1, 0);
  playTone(1175, 'sine', 0.25, 0.08, 0.1);
}

/** Low descending tone for error */
export function playError() {
  playTone(300, 'triangle', 0.3, 0.1, 0, 180);
}

/** Soft tick for feeding */
export function playFeedTick() {
  playTone(1000, 'sine', 0.05, 0.08);
  playTone(1200, 'sine', 0.05, 0.06, 0.06);
}

// ── Ambient Sounds (Day/Night) ──

type AmbientType = 'dawn' | 'day' | 'dusk' | 'night' | 'off';

function createAmbientOsc(
  freq: number,
  type: OscillatorType,
  volume: number,
  wobbleAmt: number,
  wobbleSpeed: number,
): { osc: OscillatorNode; gain: GainNode } {
  const c = getCtx();

  const osc = c.createOscillator();
  const gain = c.createGain();
  const lfo = c.createOscillator();
  const lfoGain = c.createGain();

  osc.type = type;
  osc.frequency.setValueAtTime(freq, c.currentTime);

  // LFO for organic wobble
  lfo.type = 'sine';
  lfo.frequency.setValueAtTime(wobbleSpeed, c.currentTime);
  lfoGain.gain.setValueAtTime(wobbleAmt, c.currentTime);
  lfo.connect(lfoGain);
  lfoGain.connect(osc.frequency);

  gain.gain.setValueAtTime(0, c.currentTime);
  gain.gain.linearRampToValueAtTime(volume, c.currentTime + 2); // 2s fade-in

  osc.connect(gain);
  gain.connect(c.destination);
  osc.start();
  lfo.start();

  return { osc, gain };
}

function stopAmbientNodes() {
  const c = ctx;
  if (!c) return;
  ambientNodes.forEach(({ osc, gain }) => {
    try {
      gain.gain.linearRampToValueAtTime(0, c.currentTime + 1.5);
      osc.stop(c.currentTime + 2);
    } catch {
      // already stopped
    }
  });
  ambientNodes = [];
}

/** Start or switch ambient sound based on time of day */
export function startAmbient(period: AmbientType) {
  if (muted && period !== 'off') return;
  stopAmbientNodes();
  if (ambientInterval) {
    clearInterval(ambientInterval);
    ambientInterval = null;
  }
  if (period === 'off' || muted) return;

  switch (period) {
    case 'day':
      // Gentle warm drone + occasional high bird tweets
      ambientNodes.push(createAmbientOsc(220, 'sine', 0.025, 3, 0.15));
      ambientNodes.push(createAmbientOsc(440, 'sine', 0.012, 8, 0.3));
      // Random bird tweets
      ambientInterval = setInterval(() => {
        if (muted) return;
        if (Math.random() < 0.3) {
          const f = 2000 + Math.random() * 1500;
          playTone(f, 'sine', 0.06 + Math.random() * 0.08, 0.04, 0, f + 400);
        }
      }, 4000);
      break;

    case 'night':
      // Low warm hum + cricket-like pulses
      ambientNodes.push(createAmbientOsc(110, 'sine', 0.02, 1, 0.08));
      ambientNodes.push(createAmbientOsc(165, 'triangle', 0.008, 2, 0.12));
      // Cricket chirps
      ambientInterval = setInterval(() => {
        if (muted) return;
        if (Math.random() < 0.25) {
          const base = 3800 + Math.random() * 800;
          playTone(base, 'sine', 0.02, 0.03);
          playTone(base, 'sine', 0.02, 0.025, 0.04);
          playTone(base, 'sine', 0.02, 0.02, 0.08);
        }
      }, 3000);
      break;

    case 'dawn':
      // Soft rising tone
      ambientNodes.push(createAmbientOsc(330, 'sine', 0.018, 5, 0.2));
      ambientNodes.push(createAmbientOsc(495, 'sine', 0.008, 3, 0.25));
      break;

    case 'dusk':
      // Mellow wind-like sound
      ambientNodes.push(createAmbientOsc(180, 'triangle', 0.015, 6, 0.1));
      ambientNodes.push(createAmbientOsc(270, 'sine', 0.01, 4, 0.18));
      break;
  }
}

export function stopAmbient() {
  stopAmbientNodes();
  if (ambientInterval) {
    clearInterval(ambientInterval);
    ambientInterval = null;
  }
}
