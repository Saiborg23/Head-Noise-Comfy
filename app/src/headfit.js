// Turn a Hunyuan3D head mesh (GLB, unit-ish scale, y up) into the app frame: millimetres, y up, face toward +z,
// with the fsaverage5 brain fitted inside the cranium the same way the brain was fitted to the original head scan.

import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const SCALP = 13;  // mm of scalp + skull between the head surface and the cortex
const FACE_FRONT = 1;  // Hunyuan3D faces the input view toward +z; flip to -1 if a mesh comes back facing away

export async function loadGlb(buf) {
  const gltf = await new GLTFLoader().parseAsync(buf, '');
  let geo = null;
  gltf.scene.traverse((o) => { if (!geo && o.isMesh) { o.updateWorldMatrix(true, false); geo = o.geometry.clone().applyMatrix4(o.matrixWorld); } });
  if (!geo) throw new Error('GLB has no mesh');
  const pos = geo.attributes.position.array;
  const index = geo.index ? Uint32Array.from(geo.index.array) : Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
  return { pos: Float32Array.from(pos), index };
}

function extent(P, sel) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < P.length; i += 3) {
    if (sel && !sel(P, i)) continue;
    for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], P[i + k]); hi[k] = Math.max(hi[k], P[i + k]); }
  }
  return { lo, hi, size: hi.map((h, k) => h - lo[k]), mid: hi.map((h, k) => (h + lo[k]) / 2) };
}

// brainMin/brainMax: the cortex bounds in the app frame. Returns the mesh moved and scaled so the cranium ring
// (45 to 75 mm below the crown) wraps the brain with SCALP to spare, crown SCALP above the brain's top.
export function fitHead({ pos, index }, brainMin, brainMax) {
  const P = Float32Array.from(pos);
  for (let i = 0; i < P.length; i += 3) { P[i] *= FACE_FRONT; P[i + 2] *= FACE_FRONT; }
  const brainW = brainMax[0] - brainMin[0], brainD = brainMax[2] - brainMin[2];

  // First guess from overall width, then refine on the cranium band (it depends on the scale, so iterate).
  let s = (brainW + 2 * SCALP + 30) / extent(P).size[0];
  let band = null;
  for (let it = 0; it < 4; it++) {
    const top = extent(P).hi[1];
    band = extent(P, (Q, i) => Q[i + 1] < top - 45 / s && Q[i + 1] > top - 75 / s);
    if (!isFinite(band.size[0])) break;
    const k = Math.min((band.size[0] * s - 2 * SCALP) / brainW, (band.size[2] * s - 2 * SCALP) / brainD);
    s /= k;
  }
  const all = extent(P);
  const cx = band && isFinite(band.mid[0]) ? band.mid[0] : all.mid[0];
  const cz = band && isFinite(band.mid[2]) ? band.mid[2] : all.mid[2];
  const brainMidX = (brainMin[0] + brainMax[0]) / 2, brainMidZ = (brainMin[2] + brainMax[2]) / 2;
  const topY = brainMax[1] + SCALP;
  for (let i = 0; i < P.length; i += 3) {
    P[i] = (P[i] - cx) * s + brainMidX;
    P[i + 1] = (P[i + 1] - all.hi[1]) * s + topY;
    P[i + 2] = (P[i + 2] - cz) * s + brainMidZ + 4;  // brain sits slightly back of the skull's centre
  }
  return { verts: P, faces: index };
}

// Area-weighted random points on the surface, so the head reads as dust like the scan does.
export function samplePoints(verts, faces, n, seed = 7) {
  let r = seed;
  const rand = () => ((r = (r * 1664525 + 1013904223) >>> 0) / 4294967296);
  const T = faces.length / 3, cum = new Float64Array(T);
  let acc = 0;
  for (let t = 0; t < T; t++) {
    const a = faces[t * 3] * 3, b = faces[t * 3 + 1] * 3, c = faces[t * 3 + 2] * 3;
    const ux = verts[b] - verts[a], uy = verts[b + 1] - verts[a + 1], uz = verts[b + 2] - verts[a + 2];
    const vx = verts[c] - verts[a], vy = verts[c + 1] - verts[a + 1], vz = verts[c + 2] - verts[a + 2];
    acc += 0.5 * Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx);
    cum[t] = acc;
  }
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = rand() * acc;
    let lo = 0, hi = T - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < x) lo = m + 1; else hi = m; }
    let u = rand(), v = rand();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const a = faces[lo * 3] * 3, b = faces[lo * 3 + 1] * 3, c = faces[lo * 3 + 2] * 3;
    for (let k = 0; k < 3; k++) out[i * 3 + k] = verts[a + k] + u * (verts[b + k] - verts[a + k]) + v * (verts[c + k] - verts[a + k]);
  }
  return out;
}
