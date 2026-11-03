// Head Noise in Three.js: the head mesh as dust with the top sliced off, the fsaverage5 cortex inside driven by
// TRIBE v2 (slow clock) and the DSP envelope (fast clock), and sparks that rise out of the open cut.
// Colours and per-point weights match the offline renderer the project started from.

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { TEX_W, TEX_H } from './data.js';

const N = 20484;
const BG = new THREE.Vector3(0.012, 0.016, 0.03);
const DUST = new THREE.Vector3(0.55, 0.5, 0.6);
const CREAM = new THREE.Vector3(0.95, 0.89, 0.78);
const ACCENT = new THREE.Vector3(1.0, 0.22, 0.08);

// Shared: point size in pixels at 1080p for a point 700 mm from the camera, and depth fog that lets the far side recede.
const SIZE_FOG = /* glsl */ `
  uniform float uSize, uViewH;
  float pointPx(vec4 mv) { return uSize * (uViewH / 1080.0) * (700.0 / -mv.z); }
  float fog(vec4 mv) { return clamp(exp(-(-mv.z - 600.0) / 450.0), 0.0, 1.6); }
`;
const DISC = /* glsl */ `
  varying vec3 vC;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = dot(d, d) * 4.0;
    if (r > 1.0) discard;
    gl_FragColor = vec4(vC * (1.0 - r * 0.6), 1.0);
  }
`;

function pointsMaterial(vertexShader, uniforms) {
  return new THREE.ShaderMaterial({
    uniforms, vertexShader, fragmentShader: DISC,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, transparent: true,
  });
}

const HEAD_VS = /* glsl */ `
  ${SIZE_FOG}
  uniform float uCut, uGain;
  uniform vec3 uColor;
  varying vec3 vC;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float near = 1.0 - smoothstep(0.0, 3.0, uCut - position.y);  // points at the cut read as the section face
    vC = uColor * mix(0.085, 0.30, near) * uGain * fog(mv);
    gl_PointSize = position.y > uCut ? 0.0 : pointPx(mv);
  }
`;

const RING_VS = /* glsl */ `
  ${SIZE_FOG}
  uniform float uGain;
  uniform vec3 uColor;
  varying vec3 vC;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vC = uColor * uGain * fog(mv);
    gl_PointSize = pointPx(mv);
  }
`;

const BRAIN_VS = /* glsl */ `
  ${SIZE_FOG}
  uniform sampler2D uAct;   // per TRIBE vertex: r = activation 0..1, g = DSP pulse
  uniform vec3 uCenter, uCream, uAccent;
  uniform float uLift, uCut, uGain;
  attribute vec3 aIdx;      // 3 TRIBE vertices and barycentric weights (dust points between vertices)
  attribute vec3 aW;
  varying vec3 vC;
  vec2 act(float i) { int k = int(i + 0.5); return texelFetch(uAct, ivec2(k % ${TEX_W}, k / ${TEX_W}), 0).rg; }
  void main() {
    vec2 s = act(aIdx.x) * aW.x + act(aIdx.y) * aW.y + act(aIdx.z) * aW.z;
    float a = s.x, p = s.y;
    vec3 dir = normalize(position - uCenter);
    float top = smoothstep(uCut - 30.0, uCut + 10.0, position.y);
    vec3 pos = position + dir * (uLift * 14.0 * a * a * (1.0 + 1.2 * p));
    pos.y += top * uLift * 22.0 * a * p;  // hits push the exposed cortex up out of the cut
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float w = (0.10 + 0.9 * a * a) * (1.0 + 1.6 * p * (0.3 + a)) * (1.0 + 0.35 * step(uCut, position.y));
    vC = mix(uCream, uAccent, pow(a, 1.3)) * w * uGain * fog(mv);
    gl_PointSize = pointPx(mv);
  }
`;

const SPARK_VS = /* glsl */ `
  ${SIZE_FOG}
  uniform vec3 uCream, uAccent;
  attribute float aLife;    // remaining life 0..1
  attribute float aHeat;    // activation of the vertex it left
  varying vec3 vC;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float l = clamp(aLife, 0.0, 1.0);
    vC = mix(uCream, uAccent, aHeat) * (l * sqrt(l)) * 1.8 * fog(mv);
    gl_PointSize = aLife > 0.0 ? pointPx(mv) * (1.4 + 1.4 * l) : 0.0;
  }
`;

// Same tone curve as render.py: 1 - exp(-k x) over the additive buffer, then lifted onto the blue-black ground.
const ToneShader = {
  uniforms: { tDiffuse: { value: null }, uExposure: { value: 1.25 }, uBg: { value: BG } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float uExposure; uniform vec3 uBg; varying vec2 vUv;
    void main() {
      vec3 o = 1.0 - exp(-texture2D(tDiffuse, vUv).rgb * uExposure);
      gl_FragColor = vec4(o + uBg * (1.0 - o), 1.0);
    }`,
};

// Where the head mesh crosses the plane y = cut, resampled every `step` mm.
function sliceRing(verts, faces, y, step = 0.6) {
  const out = [];
  const P = (i) => [verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]];
  for (let f = 0; f < faces.length; f += 3) {
    const tri = [P(faces[f]), P(faces[f + 1]), P(faces[f + 2])];
    const hits = [];
    for (let e = 0; e < 3; e++) {
      const a = tri[e], b = tri[(e + 1) % 3];
      if ((a[1] - y) * (b[1] - y) < 0) {
        const u = (y - a[1]) / (b[1] - a[1]);
        hits.push([a[0] + (b[0] - a[0]) * u, y, a[2] + (b[2] - a[2]) * u]);
      }
    }
    if (hits.length !== 2) continue;
    const [p, q] = hits;
    const k = Math.max(2, Math.ceil(Math.hypot(q[0] - p[0], q[2] - p[2]) / step));
    for (let i = 0; i < k; i++) {
      const u = i / k;
      out.push(p[0] + (q[0] - p[0]) * u, y, p[2] + (q[2] - p[2]) * u);
    }
  }
  return new Float32Array(out);
}

class Sparks {
  constructor(n) {
    this.n = n;
    this.pos = new Float32Array(n * 3);
    this.vel = new Float32Array(n * 3);
    this.life = new Float32Array(n);
    this.max = new Float32Array(n).fill(1);
    this.lifeAttr = new Float32Array(n);
    this.heat = new Float32Array(n);
    this.cursor = 0;
  }

  spawn(x, y, z, vx, vy, vz, life, heat) {
    const i = this.cursor;
    this.cursor = (i + 1) % this.n;
    this.pos.set([x, y, z], i * 3);
    this.vel.set([vx, vy, vz], i * 3);
    this.life[i] = this.max[i] = life;
    this.heat[i] = heat;
  }

  step(dt, t) {
    const drag = Math.pow(0.4, dt);
    for (let i = 0; i < this.n; i++) {
      if (this.life[i] <= 0) { this.lifeAttr[i] = 0; continue; }
      this.life[i] -= dt;
      const j = i * 3, P = this.pos, V = this.vel;
      V[j] = V[j] * drag + Math.sin(P[j + 1] * 0.045 + t * 0.7 + i) * 26 * dt;
      V[j + 1] = V[j + 1] * Math.pow(0.85, dt) + 14 * dt;  // buoyant: keeps rising out of the head
      V[j + 2] = V[j + 2] * drag + Math.cos(P[j + 1] * 0.04 - t * 0.6 + i * 1.3) * 26 * dt;
      P[j] += V[j] * dt; P[j + 1] += V[j + 1] * dt; P[j + 2] += V[j + 2] * dt;
      this.lifeAttr[i] = Math.max(this.life[i], 0) / this.max[i];
    }
  }

  clear() { this.life.fill(0); this.lifeAttr.fill(0); }
}

export class Viewer {
  constructor(canvas, data) {
    this.canvas = canvas;
    this.data = data;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(1);
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setClearColor(0x000000, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, 16 / 9, 5, 5000);

    const b = data.meta.bounds;
    this.center = new THREE.Vector3(0, (b.brain_min[1] + b.brain_max[1]) / 2, 0);
    this.brainTop = b.brain_max[1];
    this.cut = 292;
    this.lift = 0.55;
    this.sparkAmount = 0.6;
    this.bloomBase = 0.55;
    this.auto = true;
    this.orbit = { az: -0.62, el: 0.42, dist: 780, target: new THREE.Vector3(0, 252, 6) };

    this.common = { uViewH: { value: 1080 }, uSize: { value: 1.6 } };
    this.buildHead();
    this.buildBrain();
    this.buildSparks();
    this.setCut(this.cut);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), this.bloomBase, 0.55, 0.0);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new ShaderPass(ToneShader));
    this.bindOrbit();
  }

  buildHead() {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.data.headPts, 3));
    this.headGeo = g;
    this.headMat = pointsMaterial(HEAD_VS, {
      ...this.common, uCut: { value: this.cut }, uGain: { value: 1 }, uColor: { value: DUST },
    });
    this.scene.add(new THREE.Points(g, this.headMat));

    this.ringGeo = new THREE.BufferGeometry();
    this.ringMat = pointsMaterial(RING_VS, { ...this.common, uGain: { value: 0.6 }, uColor: { value: CREAM } });
    this.scene.add(new THREE.Points(this.ringGeo, this.ringMat));
  }

  buildBrain() {
    const d = this.data;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(d.brainPos, 3));
    g.setAttribute('aIdx', new THREE.BufferAttribute(Float32Array.from(d.brainIdx), 3));
    g.setAttribute('aW', new THREE.BufferAttribute(d.brainW, 3, true));
    this.actData = new Float32Array(TEX_W * TEX_H * 2);
    this.actTex = new THREE.DataTexture(this.actData, TEX_W, TEX_H, THREE.RGFormat, THREE.FloatType);
    this.actTex.needsUpdate = true;
    this.brainMat = pointsMaterial(BRAIN_VS, {
      ...this.common, uAct: { value: this.actTex }, uCenter: { value: this.center },
      uCream: { value: CREAM }, uAccent: { value: ACCENT }, uLift: { value: this.lift },
      uCut: { value: this.cut }, uGain: { value: 0.5 },
    });
    const pts = new THREE.Points(g, this.brainMat);
    pts.frustumCulled = false;  // the shader moves points past the static bounds
    this.scene.add(pts);
    // Per-group pulse weights, same mapping as render.py: auditory <- kick band, visual <- hats, motor/insula <- mids.
    this.groupIx = Object.fromEntries(d.meta.groups.map((g, i) => [g, i + 1]));
    this.coef = new Float32Array(d.meta.groups.length + 1);
  }

  buildSparks() {
    this.sparks = new Sparks(9000);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.sparks.pos, 3).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aLife', new THREE.BufferAttribute(this.sparks.lifeAttr, 1).setUsage(THREE.DynamicDrawUsage));
    g.setAttribute('aHeat', new THREE.BufferAttribute(this.sparks.heat, 1).setUsage(THREE.DynamicDrawUsage));
    this.sparkGeo = g;
    this.sparkMat = pointsMaterial(SPARK_VS, { ...this.common, uCream: { value: CREAM }, uAccent: { value: ACCENT } });
    const pts = new THREE.Points(g, this.sparkMat);
    pts.frustumCulled = false;
    this.scene.add(pts);
    this.spawnDebt = 0;
  }

  setCut(y) {
    this.cut = y;
    this.headMat.uniforms.uCut.value = y;
    this.brainMat.uniforms.uCut.value = y;
    const ring = sliceRing(this.data.headVerts, this.data.headFaces, y);
    this.ringGeo.setAttribute('position', new THREE.BufferAttribute(ring, 3));
    this.ringGeo.computeBoundingSphere();
    // Spark sources: TRIBE vertices on the exposed crown, i.e. near or above the cut and facing up.
    const P = this.data.brainPos, c = this.center, lim = Math.min(y, this.brainTop) - 14;
    const src = [];
    for (let v = 0; v < N; v++) {
      const x = P[v * 3], yy = P[v * 3 + 1], z = P[v * 3 + 2];
      const up = (yy - c.y) / Math.hypot(x - c.x, yy - c.y, z - c.z);
      if (yy > lim && up > 0.15) src.push(v);
    }
    this.sparkSrc = Uint32Array.from(src);
  }

  // Swap the head for another surface in the app frame (e.g. the Hunyuan3D mesh): dust points + triangles for the cut.
  setHead(points, verts, faces) {
    this.data.headVerts = verts;
    this.data.headFaces = faces;
    this.headGeo.setAttribute('position', new THREE.BufferAttribute(points, 3));
    this.headGeo.computeBoundingSphere();
    this.setCut(this.cut);
  }

  setSize(w, h) {
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.common.uViewH.value = h;
    this.size = [w, h];
  }

  bindOrbit() {
    let drag = null;
    this.lastTouch = -1e9;
    this.canvas.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, az: this.orbit.az, el: this.orbit.el };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.orbit.az = drag.az - (e.clientX - drag.x) * 0.006;
      this.orbit.el = Math.max(-0.2, Math.min(1.35, drag.el + (e.clientY - drag.y) * 0.004));
      this.lastTouch = performance.now();
    });
    const end = () => { drag = null; };
    this.canvas.addEventListener('pointerup', end);
    this.canvas.addEventListener('pointercancel', end);
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbit.dist = Math.max(260, Math.min(1800, this.orbit.dist * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
  }

  // t: track seconds; env: [rms, low, mid, high]; since: seconds since the last kick; live: audio is running.
  frame(t, dt, env, since, live) {
    const d = this.data, cx = d.cortex;
    const [rms, low, mid, high] = env;
    cx.update(t);

    const G = this.groupIx, coef = this.coef;
    coef.fill(0);
    coef[G.auditory] = 1.1 * low; coef[G.visual] = 0.9 * high;
    coef[G.motor] = 0.6 * mid; coef[G.insula_emotion] = 0.4 * mid;
    const base = 0.15 * rms, A = this.actData, act = cx.act, roi = d.roi;
    for (let v = 0; v < N; v++) {
      A[v * 2] = act[v];
      A[v * 2 + 1] = base + coef[roi[v]];
    }
    this.actTex.needsUpdate = true;
    this.brainMat.uniforms.uLift.value = this.lift;

    // Only the brain and sparks react; the head, cut ring, glow and camera framing stay still.
    this.bloom.strength = this.bloomBase;

    this.emit(dt, low, since, live);
    this.sparks.step(live ? dt : 0, t);  // pause freezes the sparks mid-air
    for (const k of ['position', 'aLife', 'aHeat']) this.sparkGeo.attributes[k].needsUpdate = true;

    // Camera: own orbit plus a slow sway when auto is on and the view is untouched for 4 s.
    const o = this.orbit;
    const sway = this.auto && performance.now() - this.lastTouch > 4000;
    const az = o.az + (sway ? 0.38 * Math.sin((2 * Math.PI * t) / 25.3) : 0);
    const el = o.el + (sway ? 0.06 * Math.sin((2 * Math.PI * t) / 41.0) : 0);
    this.camera.position.set(
      o.target.x + o.dist * Math.cos(el) * Math.sin(az),
      o.target.y + o.dist * Math.sin(el),
      o.target.z + o.dist * Math.cos(el) * Math.cos(az),
    );
    this.camera.lookAt(o.target);
    this.composer.render(dt);
  }

  emit(dt, low, since, live) {
    if (!live || this.sparkAmount <= 0 || !this.sparkSrc.length) return;
    const kick = Math.exp(-since / 0.08);
    this.spawnDebt += dt * this.sparkAmount * (300 + 1800 * low + 9000 * kick);
    const P = this.data.brainPos, act = this.data.cortex.act, c = this.center, src = this.sparkSrc;
    let tries = 0;
    while (this.spawnDebt >= 1 && tries++ < 4000) {
      const v = src[(Math.random() * src.length) | 0];
      const a = act[v];
      if (Math.random() > a * Math.sqrt(a) + 0.02) continue;
      this.spawnDebt -= 1;
      const x = P[v * 3], y = P[v * 3 + 1], z = P[v * 3 + 2];
      const r = Math.hypot(x - c.x, y - c.y, z - c.z);
      const push = this.lift * 14 * a * a;
      const nx = (x - c.x) / r, ny = (y - c.y) / r, nz = (z - c.z) / r;
      const j = () => (Math.random() - 0.5) * 16;
      this.sparks.spawn(
        x + nx * push, y + ny * push, z + nz * push,
        nx * 22 + j(), 18 + 60 * a + 50 * low + j(), nz * 22 + j(),
        0.8 + Math.random() * 1.6, a,
      );
    }
    if (tries >= 4000) this.spawnDebt = 0;
  }
}
