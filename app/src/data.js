// Loads the binaries written by tools/export_data.py and answers "what is the cortex doing at time t".

const N = 20484;
export const TEX_W = 256;
export const TEX_H = Math.ceil(N / TEX_W);

import { loadGlb, fitHead, samplePoints } from './headfit.js';

async function bin(name, Type) {
  const r = await fetch(`/data/${name}`);
  if (!r.ok) throw new Error(`missing data/${name}; run tools/export_data.py`);
  return new Type(await r.arrayBuffer());
}

// The head: the bundled Hunyuan3D head.glb (made from the demo face). `?head=scan` uses a private scan in
// data/local/ instead, when one is there (not in the repo).
async function loadHead(meta) {
  const local = new URLSearchParams(location.search).get('head') === 'scan'
    && (await fetch('/data/local/head_pts.f32', { method: 'HEAD' })).ok;
  if (local) {
    const [headPts, headVerts, headFaces] = await Promise.all([
      bin('local/head_pts.f32', Float32Array), bin('local/head_verts.f32', Float32Array), bin('local/head_faces.u32', Uint32Array),
    ]);
    return { headPts, headVerts, headFaces };
  }
  const glb = await (await fetch('/data/head.glb')).arrayBuffer();
  const m = fitHead(await loadGlb(glb), meta.bounds.brain_min, meta.bounds.brain_max);
  return { headPts: samplePoints(m.verts, m.faces, 140_000), headVerts: m.verts, headFaces: m.faces };
}

export async function loadData() {
  const meta = await (await fetch('/data/meta.json')).json();
  const [head, brainPos, brainIdx, brainW, roi, tribe] = await Promise.all([
    loadHead(meta), bin('brain_pos.f32', Float32Array), bin('brain_idx.u16', Uint16Array), bin('brain_w.u8', Uint8Array),
    bin('roi.u8', Uint8Array), fetch('/data/tribe_demo.tribe').then((r) => r.arrayBuffer()),
  ]);
  const data = { meta, ...head, brainPos, brainIdx, brainW, roi };
  data.cortex = parseTribe(tribe, data).cortex;
  return data;
}

// .tribe from the TribeV2Predict node: b"TRIB", uint32 LE header length, JSON header, steps x 20484 uint8 z-scores.
export function parseTribe(buf, data) {
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== 'TRIB') throw new Error('not a .tribe file');
  const n = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const preds = new Uint8Array(buf, 8 + n, header.steps * header.vertices);
  const meta = { tribe: header, groups: data.meta.groups };
  return { header, cortex: new Cortex(meta, preds, data.roi) };
}

const smooth = (f) => f * f * (3 - 2 * f);

// TRIBE v2 output: (steps, 20484) z-scores per vertex, quantized to uint8 over [-zClip, zClip].
export class Cortex {
  constructor(meta, preds, roi) {
    this.meta = meta;
    this.q = preds;
    this.roi = roi;
    this.steps = meta.tribe.steps;
    this.rate = meta.tribe.rate_hz;
    this.t0 = meta.tribe.t0;
    this.groups = meta.groups;
    const zc = meta.tribe.z_clip;
    this.lut = new Float32Array(256).map((_, i) => (i / 255) * 2 * zc - zc);
    this.z = new Float32Array(N);
    this.act = new Float32Array(N);
    this.curves = this.roiCurves();
  }

  // Mean z per ROI group for every TRIBE step: (steps, groups). Drives the HUD bars and the timeline.
  roiCurves() {
    const G = this.groups.length;
    const out = Array.from({ length: G }, () => new Float32Array(this.steps));
    const count = new Float32Array(G + 1);
    for (let v = 0; v < N; v++) count[this.roi[v]]++;
    for (let s = 0; s < this.steps; s++) {
      const sum = new Float32Array(G + 1);
      const row = s * N;
      for (let v = 0; v < N; v++) sum[this.roi[v]] += this.lut[this.q[row + v]];
      for (let g = 0; g < G; g++) out[g][s] = sum[g + 1] / Math.max(count[g + 1], 1);
    }
    return out;
  }

  // Fractional step for track time t, looping when the run is shorter than the track (another track on demo data).
  stepAt(t) {
    const span = this.steps / this.rate;
    let u = ((t - this.t0) % span + span) % span;
    return u * this.rate;
  }

  // Fill this.z and this.act (0..1 activation, the same curve as the offline renderer) for time t.
  update(t) {
    const s = this.stepAt(t);
    const i = Math.min(Math.floor(s), this.steps - 1);
    const j = Math.min(i + 1, this.steps - 1);
    const f = smooth(s - Math.floor(s));
    const a = i * N, b = j * N, lut = this.lut;
    for (let v = 0; v < N; v++) {
      const z = lut[this.q[a + v]] * (1 - f) + lut[this.q[b + v]] * f;
      this.z[v] = z;
      const x = (z + 0.1) / 1.7;
      this.act[v] = x < 0 ? 0 : x > 1 ? 1 : x;
    }
  }

  levels(t) {
    const s = this.stepAt(t);
    const i = Math.min(Math.floor(s), this.steps - 1);
    const j = Math.min(i + 1, this.steps - 1);
    const f = smooth(s - Math.floor(s));
    return this.curves.map((c) => c[i] * (1 - f) + c[j] * f);
  }
}
