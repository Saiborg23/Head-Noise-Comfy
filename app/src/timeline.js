// Whole-track strip: the six TRIBE region curves, the snippet bracket (drag its edges) and the playhead (click to seek).

const HANDLE_PX = 7;

export class Timeline {
  constructor(canvas, { onSeek, onRegion }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onSeek = onSeek;
    this.onRegion = onRegion;
    this.duration = 1;
    this.curves = null;
    this.region = { start: 0, end: 1 };
    this.t = 0;
    this.drag = null;
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', () => { this.drag = null; });
    canvas.addEventListener('pointercancel', () => { this.drag = null; });
  }

  set(curves, stepsPerSec, duration) {
    this.curves = curves;
    this.rate = stepsPerSec;
    this.duration = duration;
  }

  x2t(clientX) {
    const r = this.canvas.getBoundingClientRect();
    return Math.max(0, Math.min(this.duration, ((clientX - r.left) / r.width) * this.duration));
  }

  t2x(t) { return (t / this.duration) * this.canvas.clientWidth; }

  down(e) {
    this.canvas.setPointerCapture(e.pointerId);
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    if (Math.abs(x - this.t2x(this.region.start)) < HANDLE_PX) this.drag = 'start';
    else if (Math.abs(x - this.t2x(this.region.end)) < HANDLE_PX) this.drag = 'end';
    else { this.drag = 'seek'; this.onSeek(this.x2t(e.clientX)); }
  }

  move(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const near = [this.region.start, this.region.end].some((t) => Math.abs(x - this.t2x(t)) < HANDLE_PX);
    this.canvas.style.cursor = this.drag === 'start' || this.drag === 'end' || near ? 'ew-resize' : 'pointer';
    if (!this.drag) return;
    const t = this.x2t(e.clientX);
    if (this.drag === 'seek') this.onSeek(t);
    else if (this.drag === 'start') this.onRegion({ start: Math.min(t, this.region.end - 1), end: this.region.end });
    else this.onRegion({ start: this.region.start, end: Math.max(t, this.region.start + 1) });
  }

  draw() {
    const c = this.canvas, dpr = this.dpr || Math.min(devicePixelRatio, 2);  // demo.js raises dpr for zoomed shots
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) {
      c.width = Math.round(W * dpr);
      c.height = Math.round(H * dpr);
    }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const xs = this.t2x(this.region.start), xe = this.t2x(this.region.end);
    g.fillStyle = 'rgba(236,226,204,0.05)';
    g.fillRect(xs, 0, xe - xs, H);

    if (this.curves) {
      const pad = 6;
      for (const curve of this.curves) {
        for (const inside of [false, true]) {
          g.save();
          g.beginPath();
          g.rect(inside ? xs : 0, 0, inside ? xe - xs : W, H);
          g.clip();
          g.strokeStyle = inside ? 'rgba(236,226,204,0.55)' : 'rgba(122,118,132,0.28)';
          g.lineWidth = 1;
          g.beginPath();
          for (let i = 0; i < curve.length; i++) {
            const x = this.t2x(i / this.rate);
            const y = pad + (1 - (Math.max(-1.6, Math.min(1.6, curve[i])) + 1.6) / 3.2) * (H - 2 * pad);
            i ? g.lineTo(x, y) : g.moveTo(x, y);
          }
          g.stroke();
          g.restore();
        }
      }
    }

    g.fillStyle = 'rgb(236,226,204)';
    for (const x of [xs, xe]) g.fillRect(Math.round(x) - 1, 0, 2, H);
    g.fillRect(xs, 0, 6, 1); g.fillRect(xs, H - 1, 6, 1);
    g.fillRect(xe - 6, 0, 6, 1); g.fillRect(xe - 6, H - 1, 6, 1);

    g.fillStyle = 'rgb(255,56,20)';
    g.fillRect(Math.round(this.t2x(this.t)), 0, 1, H);
  }
}
