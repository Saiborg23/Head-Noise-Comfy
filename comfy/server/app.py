"""Head Noise backend: takes a track and a face photo, runs them through ComfyUI with the Comfy SDK, and hands the
browser the TRIBE v2 predictions (.tribe) and the head mesh (.glb).

One codebase, three targets; only the environment changes:

    target              COMFY_BASE_URL                         COMFY_API_KEY     TRIBE step          mesh saved by
    local ComfyUI       http://127.0.0.1:8289 (comfy-api-proxy)  -               TribeV2Predict node  HeadNoiseSaveMesh
    platform deployment https://<deployment>.run.comfy.app       workspace key     TribeV2Predict node  HeadNoiseSaveMesh
    Comfy Cloud         unset (https://cloud.comfy.org)          Comfy API key     cached, by backend   core SaveGLB

Comfy Cloud runs core nodes only, so there the custom pack is not available: the face workflow saves with the core
SaveGLB node and the backend serves the cached TRIBE run itself. HEAD_NOISE_TARGET=cloud|custom overrides the guess.

Run (from the repo root):  .venv/bin/uvicorn --app-dir comfy/server app:app --port 8790
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

from comfy_sdk import AsyncComfy, ComfyError, JobFailed, Progress, StatusChange
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / "comfy" / "workflows"
TRIBE_CACHE = ROOT / "comfy" / "build" / "custom_nodes" / "comfyui-head-noise" / "cache"
RUNS_DIR = ROOT / "comfy" / "runs"
DIST = ROOT / "app" / "dist"

# Input and output node ids in the API workflows.
TRIBE_IN, TRIBE_OUT = "1", "2"
FACE_IN, FACE_OUT = "1", "15"
CACHE_PACING_S = 4.0

app = FastAPI(title="Head Noise")
RUNS: dict[str, "Run"] = {}


def base_url():
    return os.environ.get("COMFY_BASE_URL", "").strip() or "https://cloud.comfy.org"


def target():
    """Where jobs go, for the UI. Never includes credentials."""
    url = base_url()
    kind = os.environ.get("HEAD_NOISE_TARGET", "").strip().lower()
    if kind not in ("cloud", "custom"):
        kind = "cloud" if urlparse(url).hostname == "cloud.comfy.org" else "custom"
    return {"base_url": url.split("@")[-1], "kind": kind,
            "authenticated": bool(os.environ.get("COMFY_API_KEY", "").strip())}


def read_tribe(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b"TRIB":
        raise ValueError(f"{path} is not a .tribe file")
    n = int.from_bytes(raw[4:8], "little")
    return json.loads(raw[8:8 + n])


class Run:
    def __init__(self, rid, has_face):
        self.id = rid
        self.dir = RUNS_DIR / rid
        self.steps = {
            "tribe": {"state": "queued", "detail": "waiting", "progress": 0.0, "job": None},
            "face": {"state": "queued" if has_face else "skipped",
                     "detail": "waiting" if has_face else "no photo, default head", "progress": 0.0, "job": None},
        }
        self.files: dict[str, str] = {}
        self.meta: dict = {}
        self.done = False
        self.changed = asyncio.Event()

    def update(self, step, **kw):
        self.steps[step].update(kw)
        self.changed.set()

    def snapshot(self):
        return {"id": self.id, "steps": self.steps, "files": list(self.files), "meta": self.meta, "done": self.done,
                "target": target()}


async def run_job(client, run, step, workflow, in_node, in_field, upload, out_node, ext, dst):
    """Submit one workflow with one uploaded input, follow its events, download the output with extension `ext`."""
    wf = client.workflows.from_file(WORKFLOWS / workflow)
    wf.set_input(in_node, in_field, client.assets.from_file(upload))
    run.update(step, state="uploading", detail="uploading input")
    t0 = time.monotonic()
    try:
        job = await client.submit(wf, metadata={"app": "head-noise", "run": run.id, "step": step})
    except ComfyError as exc:
        # Deployments keep job labels; Comfy Cloud and the local proxy refuse them, and a refusal created no job.
        if exc.code not in ("invalid_request", "metadata_not_supported"):
            raise
        job = await client.submit(wf)
    run.update(step, state="running", detail="queued", job=job.id)
    async for ev in job.events():
        if isinstance(ev, Progress):
            msg = f"node {ev.nodes_done}/{ev.nodes_total}" if ev.nodes_total else (ev.message or "running")
            if ev.steps:
                msg += f", step {ev.step}/{ev.steps}"
            run.update(step, progress=float(ev.value), detail=msg)
        elif isinstance(ev, StatusChange):
            if ev.status == "queued" and ev.queue_position is not None:
                run.update(step, detail=f"queued, position {ev.queue_position}")
            elif ev.status == "running":
                run.update(step, detail="running")
            if ev.status in ("succeeded", "failed", "canceled", "expired"):
                break
    job = await job.result()  # authoritative; raises JobFailed
    outs = [o for o in job.get_outputs(out_node) if o.name.endswith(ext)] or [o for o in job.outputs if o.name.endswith(ext)]
    if not outs:
        raise RuntimeError(f"{workflow}: no {ext} output on node {out_node}")
    await outs[0].to_file(dst)
    run.update(step, state="done", progress=1.0, detail=f"done in {time.monotonic() - t0:.0f}s")
    return dst


async def tribe_from_cache(run, duration, dst):
    """For targets without the custom pack (Comfy Cloud): the cached run closest in length, served by the backend."""
    runs = sorted(TRIBE_CACHE.glob("*.tribe"))
    if not runs:
        raise RuntimeError(f"no cached TRIBE runs in {TRIBE_CACHE}")
    best = min(runs, key=lambda p: abs(read_tribe(p)["duration"] - (duration or 0)))
    t0 = time.monotonic()
    for i in range(1, 9):
        run.update("tribe", state="running", progress=i / 8, detail="cached run")
        await asyncio.sleep(CACHE_PACING_S / 8)
    shutil.copyfile(best, dst)
    run.update("tribe", state="done", progress=1.0, detail=f"cached run in {time.monotonic() - t0:.0f}s")


async def pipeline(run, audio_path, face_path, duration):
    cloud = target()["kind"] == "cloud"
    try:
        client = AsyncComfy(client_info="head-noise")
    except ComfyError as exc:  # e.g. no COMFY_API_KEY for Comfy Cloud; the cached TRIBE step can still run
        client, client_error = None, exc

    def need_client():
        if client is None:
            raise RuntimeError(str(client_error))

    async def tribe():
        dst = run.dir / "preds.tribe"
        if cloud:
            await tribe_from_cache(run, duration, dst)
        else:
            need_client()
            await run_job(client, run, "tribe", "tribe_v2_api.json", TRIBE_IN, "audio", audio_path, TRIBE_OUT, ".tribe", dst)
        h = read_tribe(dst)
        run.meta["tribe"] = {k: h.get(k) for k in ("model", "steps", "rate_hz", "vertices")}
        run.files["preds"] = dst.name
        run.update("tribe", detail=f"{h['steps']} x {h['vertices']:,} @ {h['rate_hz']:g} Hz, {run.steps['tribe']['detail']}")

    async def face():
        dst = run.dir / "head.glb"
        need_client()
        workflow = "face_to_3d_cloud_api.json" if cloud else "face_to_3d_api.json"
        await run_job(client, run, "face", workflow, FACE_IN, "image", face_path, FACE_OUT, ".glb", dst)
        run.files["mesh"] = dst.name

    async def guarded(step, fn):
        try:
            await fn()
        except JobFailed as exc:
            err = exc.error
            node = getattr(err, "class_type", None) or getattr(err, "node_id", None)
            run.update(step, state="failed", detail=f"{node}: {getattr(err, 'message', err)}" if node else str(exc))
        except Exception as exc:  # surfaced to the UI; the run keeps going with what it has
            run.update(step, state="failed", detail=str(exc) if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: {exc}")

    jobs = [guarded("tribe", tribe)]
    if face_path:
        jobs.append(guarded("face", face))
    if client is None:
        await asyncio.gather(*jobs)
    else:
        async with client:
            await asyncio.gather(*jobs)
    run.done = True
    run.changed.set()


@app.get("/api/health")
async def health():
    return target()


@app.post("/api/runs")
async def create_run(audio: UploadFile = File(...), face: UploadFile | None = File(None), duration: float | None = Form(None)):
    rid = uuid.uuid4().hex[:12]
    run = Run(rid, face is not None)
    run.dir.mkdir(parents=True, exist_ok=True)

    def save(up, stem):
        ext = Path(up.filename or "").suffix.lower() or ".bin"
        p = run.dir / f"{stem}{ext}"
        with p.open("wb") as f:
            shutil.copyfileobj(up.file, f)
        return p

    audio_path = save(audio, "track")
    face_path = save(face, "face") if face is not None else None
    RUNS[rid] = run
    run.task = asyncio.create_task(pipeline(run, audio_path, face_path, duration))  # keep a reference
    return run.snapshot()


@app.get("/api/runs/{rid}")
async def get_run(rid: str):
    if rid not in RUNS:
        raise HTTPException(404, "unknown run")
    return RUNS[rid].snapshot()


@app.get("/api/runs/{rid}/events")
async def run_events(rid: str):
    run = RUNS.get(rid)
    if not run:
        raise HTTPException(404, "unknown run")

    async def stream():
        while True:
            run.changed.clear()
            yield f"data: {json.dumps(run.snapshot())}\n\n"
            if run.done:
                return
            try:
                await asyncio.wait_for(run.changed.wait(), timeout=15)
            except TimeoutError:
                yield ": keepalive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


OUTPUT_NAMES = ("preds.tribe", "head.glb")


@app.get("/api/runs/{rid}/files/{name}")
async def run_file(rid: str, name: str):
    """A run's outputs. Read from disk, so they survive a backend restart; only the two output names are served."""
    if name not in OUTPUT_NAMES or not rid.isalnum():
        raise HTTPException(404, "no such file")
    path = RUNS_DIR / rid / name
    if not path.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(path)


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="app")
