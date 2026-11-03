// Playback on an AudioBufferSource (sample-accurate time, seamless snippet loops) and an offline DSP pass that
// turns the track into a per-frame [rms, low, mid, high] envelope plus kick times, the fast clock on top of TRIBE.

export const ENV_FPS = 60;

export class Track {
  constructor() {
    this.ctx = new AudioContext();
    this.out = this.ctx.createGain();
    this.out.connect(this.ctx.destination);
    this.recDest = this.ctx.createMediaStreamDestination();
    this.out.connect(this.recDest);
    this.buffer = null;
    this.src = null;
    this.offset = 0;
    this.startedAt = 0;
    this.region = null; // {start, end, loop}
    this.onended = null;
  }

  get duration() { return this.buffer ? this.buffer.duration : 0; }
  get playing() { return !!this.src; }

  async load(arrayBuffer) {
    this.stop();
    this.buffer = await this.ctx.decodeAudioData(arrayBuffer);
    this.offset = 0;
    return this.buffer;
  }

  get time() {
    if (!this.src) return this.offset;
    let t = this.offset + (this.ctx.currentTime - this.startedAt);
    const r = this.region;
    if (r && r.loop && t >= r.end) t = r.start + ((t - r.end) % (r.end - r.start));
    return Math.min(t, this.duration);
  }

  // Play from `from` inside region {start, end, loop}. Without loop it stops at region.end.
  play(from, region) {
    if (!this.buffer) return;
    this.ctx.resume();
    this.stop();
    this.region = region;
    const { start, end, loop } = region;
    if (from < start || from >= end - 0.05) from = start;
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(this.out);
    if (loop) {
      src.loop = true;
      src.loopStart = start;
      src.loopEnd = end;
      src.start(0, from);
    } else {
      src.start(0, from, end - from);
    }
    src.onended = () => {
      if (this.src !== src) return;
      this.offset = this.time;
      this.src = null;
      this.onended?.();
    };
    this.src = src;
    this.offset = from;
    this.startedAt = this.ctx.currentTime;
  }

  pause() {
    if (!this.src) return;
    this.offset = this.time;
    this.stop();
  }

  stop() {
    if (!this.src) return;
    const s = this.src;
    this.src = null;
    s.onended = null;
    try { s.stop(); } catch { /* already stopped */ }
  }

  seek(t, region) {
    const was = this.playing;
    this.stop();
    this.offset = Math.max(0, Math.min(t, this.duration));
    if (was) this.play(this.offset, region);
  }
}

// Offline: low / mid / high bands through biquads, RMS per frame, normalized to the track's own 98th percentile,
// instant attack and ~120 ms release so hits read as pulses.
export async function envelope(buffer) {
  const sr = 22050;
  const len = Math.ceil(buffer.duration * sr);
  const off = new OfflineAudioContext(4, len, sr);
  const src = new AudioBufferSourceNode(off, { buffer });
  const merge = off.createChannelMerger(4);
  src.connect(merge, 0, 0);
  const bands = [
    { type: 'lowpass', frequency: 140, Q: 0.7 },
    { type: 'bandpass', frequency: 1200, Q: 0.6 },
    { type: 'highpass', frequency: 5000, Q: 0.7 },
  ];
  bands.forEach((b, i) => src.connect(new BiquadFilterNode(off, b)).connect(merge, 0, i + 1));
  merge.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();

  const hop = sr / ENV_FPS;
  const frames = Math.floor(len / hop);
  const env = new Float32Array(frames * 4);
  for (let c = 0; c < 4; c++) {
    const x = rendered.getChannelData(c);
    const col = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
      let s = 0;
      const a = Math.floor(f * hop), b = Math.floor((f + 1) * hop);
      for (let k = a; k < b; k++) s += x[k] * x[k];
      col[f] = Math.sqrt(s / (b - a));
    }
    const p98 = Float32Array.from(col).sort()[Math.floor(frames * 0.98)] || 1;
    const rel = Math.exp(-1 / (0.12 * ENV_FPS));
    let y = 0;
    for (let f = 0; f < frames; f++) {
      const v = Math.min(col[f] / p98, 1);
      y = v > y ? v : y * rel + v * (1 - rel);
      env[f * 4 + c] = y;
    }
  }

  // Kicks: a sharp rise in the low band, at most one per 220 ms.
  const kicks = [];
  let last = -1;
  for (let f = 3; f < frames; f++) {
    const lo = env[f * 4 + 1];
    const rise = lo - env[(f - 3) * 4 + 1];
    const t = f / ENV_FPS;
    if (lo > 0.4 && rise > 0.12 && t - last > 0.2) {
      kicks.push(t);
      last = t;
    }
  }
  return { env, frames, kicks: Float32Array.from(kicks) };
}

export function envAt(e, t) {
  const f = Math.max(0, Math.min(e.frames - 1, Math.floor(t * ENV_FPS)));
  return e.env.subarray(f * 4, f * 4 + 4);
}

// Seconds since the last kick at or before t (binary search).
export function sinceKick(e, t) {
  const k = e.kicks;
  let lo = 0, hi = k.length - 1, ans = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (k[m] <= t) { ans = m; lo = m + 1; } else hi = m - 1;
  }
  return ans < 0 ? 99 : t - k[ans];
}
