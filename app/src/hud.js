// Mono telemetry drawn on its own canvas so the recorder can composite it into the export.
// Same layout as render.py's hud(): title top left, clock top right, region bars bottom left, one accent on the leader.

const FONT = 'ui-monospace, "SF Mono", Menlo, monospace';
const CREAM = 'rgb(236,226,204)';
const DIM = 'rgb(120,116,128)';
const ACC = 'rgb(255,70,30)';
const TRACKC = 'rgb(40,40,52)';

const LABEL = {
  auditory: 'AUDITORY', visual: 'VISUAL', language: 'LANGUAGE', motor: 'MOTOR',
  default_mode: 'DEFAULT MODE', insula_emotion: 'INSULA',
};

export function drawHud(ctx, W, H, info) {
  ctx.clearRect(0, 0, W, H);
  const s = Math.max(11, Math.round(Math.min(H, W * 1.2) / 60));
  const m = s * 2;
  const f = `${s}px ${FONT}`, fb = `600 ${Math.round(s * 1.25)}px ${FONT}`;
  ctx.textBaseline = 'top';

  ctx.textAlign = 'left';
  ctx.font = fb; ctx.fillStyle = CREAM;
  ctx.fillText(info.brand || 'HEAD NOISE', m, m);
  ctx.font = f; ctx.fillStyle = DIM;
  ctx.fillText('TRIBE v2 / fsaverage5 / 20,484 vtx', m, m + s * 1.8);
  ctx.fillText('predicted, not measured', m, m + s * 3.2);

  ctx.textAlign = 'right';
  ctx.font = fb; ctx.fillStyle = CREAM;
  if (info.idle) {
    ctx.fillText('PREVIEW', W - m, m);
    ctx.font = f; ctx.fillStyle = DIM;
    ctx.fillText('no audio yet, press PLAY', W - m, m + s * 1.8);
  } else {
    ctx.fillText(`T+${info.t.toFixed(2).padStart(6, '0')}`, W - m, m);
  }
  ctx.font = f; ctx.fillStyle = DIM;
  const db = info.downbeats;
  if (!info.idle && db && info.tempo) {
    let bar = 0;
    while (bar < db.length && db[bar] <= info.t) bar++;
    ctx.fillText(`BAR ${String(bar).padStart(3, '0')}/${db.length}  ${Math.round(info.tempo)} BPM`, W - m, m + s * 1.8);
  }

  const lv = info.levels, groups = info.groups;
  let lead = 0;
  lv.forEach((v, i) => { if (v > lv[lead]) lead = i; });
  const rowH = s * 1.5;
  const y0 = H - m - groups.length * rowH;
  const x0 = m + s * 9, bw = s * 10;
  ctx.textAlign = 'left';
  groups.forEach((g, i) => {
    const y = y0 + i * rowH;
    const c = i === lead ? ACC : DIM;
    ctx.font = f; ctx.fillStyle = c;
    ctx.fillText(LABEL[g] || g.toUpperCase(), m, y);
    ctx.fillStyle = TRACKC;
    ctx.fillRect(x0, y + s * 0.55, bw, 1);
    const n = Math.max(0, Math.min(1, (lv[i] + 1.5) / 3)) * bw;
    ctx.fillStyle = c;
    ctx.fillRect(x0, y + s * 0.55 - Math.max(1, s / 8), n, Math.max(2, s / 4));
  });

  if (info.title) {
    ctx.textAlign = 'right'; ctx.font = f; ctx.fillStyle = DIM;
    ctx.fillText(info.title.toUpperCase(), W - m, H - m - s);
  }
}
