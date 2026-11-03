import { loadData, parseTribe } from './data.js';
import { loadGlb, fitHead, samplePoints } from './headfit.js';
import { Track, envelope, envAt, sinceKick } from './audio.js';
import { Viewer } from './viewer.js';
import { drawHud } from './hud.js';
import { Timeline } from './timeline.js';

const $ = (s) => document.querySelector(s);
const FRAMES = { '16:9': [1920, 1080], '9:16': [1080, 1920], '1:1': [1080, 1080] };
const ZERO = new Float32Array(4);

const state = {
  audioFile: null, faceFile: null, faceURL: null, env: null, tribe: null,
  frame: '16:9', range: 'snippet', hud: true, recording: null, idleT: 12,
};
let region = { start: 12, end: 68, loop: true };

const parseTime = (s) => {
  const p = String(s).trim().split(':').map(Number);
  if (!p.length || p.some((n) => Number.isNaN(n))) return null;
  return p.length === 2 ? p[0] * 60 + p[1] : p[0];
};
const fmt = (t, dec = 0) => {
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(dec).padStart(dec ? dec + 3 : 2, '0')}`;
};

const data = await loadData();
const meta = data.meta;
const track = new Track();
const viewer = new Viewer($('#gl'), data);
const hudCanvas = $('#hud');
const hudCtx = hudCanvas.getContext('2d');
const timeline = new Timeline($('#timeline'), { onSeek: seek, onRegion: setRegion });
timeline.set(data.cortex.curves, meta.tribe.rate_hz, meta.track.duration);
timeline.region = region;
window.headNoise = { viewer, track, state, data };  // console handle for tuning

// ---- pipeline UI -------------------------------------------------------------------------------------------------

function step(name, cls, text) {
  const li = document.querySelector(`#steps li[data-step="${name}"]`);
  li.className = cls;
  li.querySelector('.v').textContent = text;
  li.querySelector('.v').classList.toggle('dim', cls === '' || cls === 'run');
}

function setupDrop(el, accept, onFile) {
  const input = el.querySelector('input');
  input.addEventListener('change', () => input.files[0] && onFile(input.files[0]));
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('over'); });
  el.addEventListener('dragleave', () => el.classList.remove('over'));
  el.addEventListener('drop', (e) => {
    e.preventDefault();
    el.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith(accept)) onFile(f);
  });
}

function setAudio(file) {
  state.audioFile = file;
  const el = $('#drop-audio');
  el.classList.add('filled');
  el.querySelector('.drop-main').textContent = file.name;
  el.querySelector('[data-role=status]').textContent = `${(file.size / 1e6).toFixed(1)} MB, press RUN`;
  $('#btn-run').disabled = false;
}

function setFace(file) {
  if (state.faceURL) URL.revokeObjectURL(state.faceURL);
  state.faceFile = file;
  state.faceURL = URL.createObjectURL(file);
  const el = $('#drop-face');
  el.classList.add('filled');
  const img = el.querySelector('img');
  img.src = state.faceURL;
  img.hidden = false;
  el.querySelector('.drop-main').textContent = file.name;
  el.querySelector('[data-role=status]').textContent = 'Hunyuan3D 2.1 turns it into the head at RUN';
}

setupDrop($('#drop-audio'), 'audio/', setAudio);
setupDrop($('#drop-face'), 'image/', setFace);

// The demo face (a synthetic portrait) and the demo track (an NCS release) are bundled; the track lives at
// app/public/data/<meta.track.audio>. Without it, any dropped track plays over the demo run.
async function loadDemo() {
  if (!state.faceFile) {
    const f = await fetch('/data/demo-face.jpg');
    if (f.ok) setFace(new File([await f.blob()], 'demo-face.jpg', { type: 'image/jpeg' }));
  }
  if (state.audioFile) return true;
  const r = await fetch(`/data/${meta.track.audio}`);
  if (!r.ok || !(r.headers.get('content-type') || '').startsWith('audio')) {
    $('#drop-audio [data-role=status]').textContent = 'no demo track bundled; drop any audio file';
    return false;
  }
  setAudio(new File([await r.blob()], meta.track.audio, { type: 'audio/mp4' }));
  return true;
}
$('#btn-demo').addEventListener('click', loadDemo);

$('#btn-run').addEventListener('click', run);

fetch('/api/health').then((r) => r.json()).then((h) => {
  $('#target').textContent = `on ${h.kind === 'cloud' ? 'Comfy Cloud' : h.base_url.replace(/^https?:\/\//, '')}`;
}).catch(() => { $('#target').textContent = 'backend offline'; });

// One platform step in the pipeline list, from the server's snapshot of that Comfy job.
function platformStep(name, s) {
  const pct = s.progress > 0 && s.progress < 1 ? ` ${Math.round(s.progress * 100)}%` : '';
  if (s.state === 'done') step(name, 'comfy', s.detail);
  else if (s.state === 'failed') step(name, 'fail', s.detail);
  else if (s.state === 'skipped') step(name, 'stub', s.detail);
  else step(name, 'run', s.state === 'running' ? `${s.detail}${pct}` : s.detail);
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url.split('/').pop()}: HTTP ${r.status}`);
  return r.arrayBuffer();
}

// Follow a run's SSE stream; resolves once the TRIBE predictions are loaded, swaps the head in whenever the mesh lands.
function follow(id) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/runs/${id}/events`);
    let gotPreds = false, gotMesh = false;
    es.onmessage = async (e) => {
      const snap = JSON.parse(e.data);
      platformStep('tribe', snap.steps.tribe);
      if (!gotMesh) platformStep('face', snap.steps.face);
      if (snap.files.includes('preds') && !gotPreds) {
        gotPreds = true;
        const buf = await fetchOk(`/api/runs/${id}/files/preds.tribe`);
        resolve(parseTribe(buf, data));
      }
      if (snap.files.includes('mesh') && !gotMesh) {
        gotMesh = true;
        try {
          const mesh = fitHead(await loadGlb(await fetchOk(`/api/runs/${id}/files/head.glb`)),
            meta.bounds.brain_min, meta.bounds.brain_max);
          viewer.setHead(samplePoints(mesh.verts, mesh.faces, 140_000), mesh.verts, mesh.faces);
          step('face', 'comfy', `Hunyuan3D 2.1, ${snap.steps.face.detail}`);
        } catch (err) {
          step('face', 'fail', `mesh: ${err.message}`);
        }
      }
      if (snap.done) {
        es.close();
        if (!gotPreds) reject(new Error(snap.steps.tribe.detail));
      }
    };
    es.onerror = () => { if (!gotPreds) { es.close(); reject(new Error('lost the backend')); } };
  });
}

async function run() {
  if (!state.audioFile || state.running) return;
  state.running = true;
  track.ctx.resume();  // inside the click, so playback can start when the jobs finish
  $('#btn-run').disabled = true;
  $('#btn-play').disabled = true;
  track.stop();
  for (const s of ['decode', 'dsp', 'tribe', 'face', 'viewer']) step(s, '', 'waiting');

  // The heavy steps run on Comfy; decode and the DSP envelope run here meanwhile.
  step('decode', 'run', 'decoding');
  const buf = await track.load(await state.audioFile.arrayBuffer());
  step('decode', 'done', `${fmt(buf.duration)}, ${buf.sampleRate / 1000} kHz`);

  const form = new FormData();
  form.append('audio', state.audioFile);
  form.append('duration', String(buf.duration));
  if (state.faceFile) form.append('face', state.faceFile);
  step('tribe', 'run', 'submitting');
  step('face', state.faceFile ? 'run' : 'stub', state.faceFile ? 'submitting' : 'no photo, default head');
  const submitted = fetch('/api/runs', { method: 'POST', body: form })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`backend ${r.status}`))));
  step('dsp', 'run', 'analysing');
  const env = await envelope(buf);
  step('dsp', 'done', `${env.kicks.length} hits, 60 fps`);

  let tribe;
  try {
    tribe = await follow((await submitted).id);
  } catch (err) {
    step('tribe', 'fail', `${err.message}; bundled run`);
    tribe = { header: { ...meta.tribe, ...meta.track, matched: Math.abs(buf.duration - meta.track.duration) < 0.75 }, cortex: data.cortex };
  }
  data.cortex = tribe.cortex;
  state.tribe = tribe.header;
  state.env = env;
  timeline.set(data.cortex.curves, tribe.header.rate_hz, buf.duration);
  const end = Math.min(region.end, buf.duration);
  setRegion({ start: Math.min(region.start, Math.max(0, end - 1)), end });
  $('#drop-audio [data-role=status]').textContent = `${fmt(buf.duration)} decoded`;
  step('viewer', 'done', 'live');
  $('#btn-play').disabled = false;
  $('#btn-rec').disabled = false;
  $('#btn-run').disabled = false;
  state.running = false;
  play();
}

// ---- transport -----------------------------------------------------------------------------------------------------

function play() {
  if (!state.env) return;
  track.play(track.time, region);
  $('#btn-play').textContent = 'PAUSE';
}

function pause() {
  track.pause();
  $('#btn-play').textContent = 'PLAY';
}

track.onended = () => {
  if (state.recording) stopRecording();
  $('#btn-play').textContent = 'PLAY';
};

// PLAY before anything has run starts the whole pipeline (with the demo track if none was dropped).
async function playPressed() {
  if (state.env) return track.playing ? pause() : play();
  track.ctx.resume();
  if (!state.audioFile && !(await loadDemo())) return;
  run();
}
$('#btn-play').addEventListener('click', playPressed);
addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || e.target.tagName === 'INPUT' || state.recording) return;
  e.preventDefault();
  playPressed();
});

function seek(t) {
  if (state.recording) return;
  if (state.env) track.seek(t, region);
  else state.idleT = t;
}

function setRegion(r) {
  if (state.recording) return;
  const dur = timeline.duration;
  region = { start: Math.max(0, r.start), end: Math.min(dur, r.end), loop: $('#in-loop').checked };
  timeline.region = region;
  $('#in-start').value = fmt(region.start);
  $('#in-end').value = fmt(region.end);
  if (track.playing) track.play(track.time, region);
}

for (const id of ['#in-start', '#in-end']) {
  $(id).addEventListener('change', () => {
    const s = parseTime($('#in-start').value), e = parseTime($('#in-end').value);
    if (s != null && e != null && e - s >= 1) setRegion({ start: s, end: e });
    else setRegion(region);
  });
}
$('#in-loop').addEventListener('change', () => setRegion(region));

// ---- view controls -------------------------------------------------------------------------------------------------

$('#s-cut').addEventListener('input', (e) => viewer.setCut(+e.target.value));
$('#s-lift').addEventListener('input', (e) => { viewer.lift = +e.target.value; });
$('#s-sparks').addEventListener('input', (e) => { viewer.sparkAmount = +e.target.value; });
$('#s-bloom').addEventListener('input', (e) => { viewer.bloomBase = +e.target.value; });
$('#c-auto').addEventListener('change', (e) => { viewer.auto = e.target.checked; });
$('#c-hud').addEventListener('change', (e) => { state.hud = e.target.checked; });

function segmented(sel, key, after) {
  const root = $(sel);
  root.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || state.recording) return;
    root.querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    state[key] = b.dataset.v;
    after?.();
  });
}
segmented('#seg-frame', 'frame', layout);
segmented('#seg-range', 'range');

// ---- layout: the stage always shows the export frame, letterboxed ----------------------------------------------------

function layout() {
  const box = $('#frame').getBoundingClientRect();
  const [ew, eh] = FRAMES[state.frame];
  const k = Math.min(box.width / ew, box.height / eh);
  const cw = Math.floor(ew * k), ch = Math.floor(eh * k);
  if (cw < 2 || ch < 2) return;
  for (const c of [$('#gl'), hudCanvas]) { c.style.width = `${cw}px`; c.style.height = `${ch}px`; }
  const dpr = Math.min(devicePixelRatio, 2);
  const [bw, bh] = state.recording ? [ew, eh] : [Math.round(cw * dpr), Math.round(ch * dpr)];
  if (!viewer.size || viewer.size[0] !== bw || viewer.size[1] !== bh) {
    viewer.setSize(bw, bh);
    hudCanvas.width = bw;
    hudCanvas.height = bh;
  }
}
new ResizeObserver(layout).observe($('#frame'));
layout();

// ---- recording -----------------------------------------------------------------------------------------------------

function pickMime() {
  return ['video/mp4;codecs=avc1.640033,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm']
    .find((m) => MediaRecorder.isTypeSupported(m));
}

$('#btn-rec').addEventListener('click', () => (state.recording ? stopRecording() : startRecording()));

function startRecording() {
  if (!state.env) return;
  const [w, h] = FRAMES[state.frame];
  const range = state.range === 'full'
    ? { start: 0, end: track.duration, loop: false }
    : { start: region.start, end: region.end, loop: false };
  const comp = document.createElement('canvas');
  comp.width = w;
  comp.height = h;
  const mime = pickMime();
  const stream = new MediaStream([
    ...comp.captureStream(60).getVideoTracks(),
    ...track.recDest.stream.getAudioTracks(),
  ]);
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 16_000_000, audioBitsPerSecond: 256_000 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  rec.onstop = () => {
    const blob = new Blob(chunks, { type: mime.split(';')[0] });
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const base = state.audioFile.name.replace(/\.[^.]+$/, '');
    const span = `${fmt(range.start).replace(':', '')}-${fmt(range.end).replace(':', '')}`;
    const name = `head-noise_${base}_${span}_${state.frame.replace(':', 'x')}.${ext}`;
    const url = URL.createObjectURL(blob);
    $('#rec-status').innerHTML = '';
    const a = Object.assign(document.createElement('a'), { href: url, download: name, textContent: `download ${name}` });
    $('#rec-status').append(a, ` (${(blob.size / 1e6).toFixed(1)} MB)`);
  };
  state.recording = { ctx: comp.getContext('2d'), rec, range };
  layout();
  viewer.sparks.clear();
  $('#btn-rec').textContent = 'STOP';
  $('#btn-rec').classList.add('rec');
  $('#btn-play').disabled = true;
  rec.start(1000);
  track.play(range.start, range);
}

function stopRecording() {
  const r = state.recording;
  if (!r) return;
  track.pause();
  r.rec.stop();
  state.recording = null;
  layout();
  $('#btn-rec').textContent = 'RECORD';
  $('#btn-rec').classList.remove('rec');
  $('#btn-play').disabled = false;
  $('#btn-play').textContent = 'PLAY';
}

// ---- frame loop ----------------------------------------------------------------------------------------------------

let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  const live = track.playing;
  let t;
  if (state.env) t = track.time;
  else t = state.idleT = (state.idleT + dt * 0.5) % meta.track.duration;  // before RUN: slow drift through the cached run

  const env = state.env ? envAt(state.env, t) : ZERO;
  const since = state.env ? sinceKick(state.env, t) : 99;
  viewer.frame(t, dt, env, since, live);

  const [W, H] = viewer.size;
  if (state.hud) {
    drawHud(hudCtx, W, H, {
      t, idle: !state.env, levels: data.cortex.levels(t), groups: meta.groups,
      tempo: state.tribe?.matched ? state.tribe.tempo : null,
      downbeats: state.tribe?.matched ? state.tribe.downbeats : null,
      title: !state.audioFile ? meta.track.title
        : state.tribe?.matched ? state.tribe.title : state.audioFile.name.replace(/\.[^.]+$/, ''),
    });
  } else hudCtx.clearRect(0, 0, W, H);

  const r = state.recording;
  if (r) {
    r.ctx.drawImage($('#gl'), 0, 0);
    if (state.hud) r.ctx.drawImage(hudCanvas, 0, 0);
    $('#rec-status').textContent = `recording ${fmt(t - r.range.start)} / ${fmt(r.range.end - r.range.start)}`;
  }

  $('#clock').textContent = state.env ? fmt(t, 1) : 'preview';
  timeline.t = t;
  timeline.draw();
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
