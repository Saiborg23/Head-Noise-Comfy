"""Pack the brain geometry and one TRIBE v2 run into the flat files the Head Noise app and node pack read.

    app/public/data/brain_pos.f32   cortex points (20,484 fsaverage5 vertices + dust samples), app frame, mm
    app/public/data/brain_idx.u16   per point: the 3 TRIBE vertices it blends
    app/public/data/brain_w.u8      per point: their barycentric weights (0..255)
    app/public/data/roi.u8          per TRIBE vertex: region group id (0 = none, 1..6 = GROUPS)
    app/public/data/meta.json       groups, bounds, track info
    app/public/data/tribe_demo.tribe                   the TRIBE run, for the offline fallback
    comfy/build/custom_nodes/comfyui-head-noise/cache/ the same run, served by the TribeV2Predict node

App frame: millimetres, y up, face toward +z, cortex centred on x/z with its top at y = 327 (where the default head's
crown sits 13 mm above it). The brain scale is fixed so every head mesh is fitted around the same brain.

Usage (from the repo root, in the project venv):
    .venv/bin/python tools/export_data.py --preds path/to/preds.npy --audio path/to/track.wav \
        [--beats beats.json] [--name slug] [--title "Track title"]

`preds.npy` is TRIBE v2's (timesteps, 20484) output; `times.npy` next to it is used when present.
"""

import argparse
import json
import struct
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "public" / "data"
CACHE = ROOT / "comfy" / "build" / "custom_nodes" / "comfyui-head-noise" / "cache"

BRAIN_SCALE = 0.981929  # mm per fsaverage5 unit, sized to an adult cranium
BRAIN_TOP = 327.0
Z_CLIP = 3.0            # z-scores are clipped to +-3 and stored as uint8
MAGIC = b"TRIB"

# Destrieux regions grouped into the knobs the visuals use.
GROUPS = {
    "auditory": ["G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral"],
    "visual": ["Pole_occipital", "S_calcarine", "G_cuneus", "G_occipital_middle", "G_and_S_occipital_inf", "G_oc-temp_med-Lingual"],
    "language": ["G_front_inf-Opercular", "G_front_inf-Triangul", "S_temporal_sup", "G_pariet_inf-Supramar", "G_pariet_inf-Angular"],
    "motor": ["G_precentral", "S_central", "G_and_S_paracentral"],
    "default_mode": ["G_precuneus", "G_cingul-Post-dorsal", "G_cingul-Post-ventral", "G_and_S_cingul-Ant"],
    "insula_emotion": ["G_insular_short", "G_Ins_lg_and_S_cent_ins", "S_circular_insula_sup", "S_circular_insula_inf"],
}


def fsaverage5():
    from nilearn import datasets
    from nilearn.surface import load_surf_mesh

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    L, R = load_surf_mesh(fs["pial_left"]), load_surf_mesh(fs["pial_right"])
    verts = np.concatenate([L.coordinates, R.coordinates]).astype(np.float64)
    faces = np.concatenate([L.faces, R.faces + len(L.coordinates)])
    return verts, faces


def brain_points(extra, seed=3):
    """RAS -> app frame (x = -R, y = superior, z = anterior), plus `extra` area-weighted dust samples on the faces."""
    ras, faces = fsaverage5()
    v = np.stack([-ras[:, 0], ras[:, 2], ras[:, 1]], 1) * BRAIN_SCALE
    rng = np.random.default_rng(seed)
    area = np.linalg.norm(np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]), axis=1)
    fi = rng.choice(len(faces), extra, p=area / area.sum())
    w = rng.dirichlet([1, 1, 1], extra)
    pts = np.concatenate([v, np.einsum("nk,nkd->nd", w, v[faces[fi]])])
    lo, hi = pts[: len(v)].min(0), pts[: len(v)].max(0)
    pts -= [(lo[0] + hi[0]) / 2, hi[1] - BRAIN_TOP, (lo[2] + hi[2]) / 2]
    idx = np.concatenate([np.repeat(np.arange(len(v))[:, None], 3, 1), faces[fi]]).astype(np.uint16)
    wq = np.concatenate([np.tile([255, 0, 0], (len(v), 1)), np.round(w * 255)]).astype(np.uint8)
    return pts.astype(np.float32), idx, wq


def roi_ids():
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas["labels"]]
    full = np.concatenate([atlas["map_left"], atlas["map_right"]])
    roi = np.zeros(len(full), np.uint8)
    for gi, names in reversed(list(enumerate(GROUPS.values()))):  # first group wins where groups overlap
        roi[np.isin(full, [labels.index(n) for n in names if n in labels])] = gi + 1
    return roi


def load_preds(path, duration):
    """(times, z): z-scored per vertex over time. Step indices stored as times are rescaled to the track length."""
    p = np.load(path).astype(np.float32)
    tp = Path(path).with_name("times.npy")
    times = np.load(tp).astype(np.float32) if tp.exists() else np.arange(len(p), dtype=np.float32)
    if times[-1] > duration * 1.05:
        times = np.arange(len(p), dtype=np.float32) * (duration / len(p))
    return times, (p - p.mean(0)) / (p.std(0) + 1e-6)


def write_tribe(path, header, data):
    h = json.dumps(header, separators=(",", ":")).encode()
    Path(path).write_bytes(MAGIC + struct.pack("<I", len(h)) + h + data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="TRIBE v2 preds.npy (times.npy next to it is used if present)")
    ap.add_argument("--audio", required=True, help="the track the run was made from (for its duration)")
    ap.add_argument("--beats", help="optional JSON with tempo and downbeats (seconds) for the HUD bar counter")
    ap.add_argument("--name", help="slug for the run, defaults to the audio file's stem")
    ap.add_argument("--title", help="display title, defaults to the slug")
    ap.add_argument("--extra", type=int, default=60_000, help="dust points sampled on the cortex faces")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    name = a.name or Path(a.audio).stem

    pos, idx, wq = brain_points(a.extra)
    pos.tofile(DATA / "brain_pos.f32")
    idx.tofile(DATA / "brain_idx.u16")
    wq.tofile(DATA / "brain_w.u8")
    roi_ids().tofile(DATA / "roi.u8")

    duration = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", a.audio]).decode())
    times, z = load_preds(a.preds, duration)
    rate = float(len(times) / (times[-1] - times[0] + (times[1] - times[0])))
    q = np.round((np.clip(z, -Z_CLIP, Z_CLIP) + Z_CLIP) / (2 * Z_CLIP) * 255).astype(np.uint8)
    beats = json.loads(Path(a.beats).read_text()) if a.beats else {}
    header = {
        "version": 1, "model": "facebook/tribev2", "track": name, "title": a.title or name, "duration": duration,
        "steps": int(len(times)), "rate_hz": rate, "t0": float(times[0]), "vertices": int(q.shape[1]), "z_clip": Z_CLIP,
        "tempo": round(beats["tempo"], 3) if "tempo" in beats else None,
        "downbeats": [round(t, 4) for t in beats["downbeats"]] if "downbeats" in beats else None,
    }
    write_tribe(DATA / "tribe_demo.tribe", header, q.tobytes())
    write_tribe(CACHE / f"{name}.tribe", header, q.tobytes())

    bv = pos[:20484]
    meta = {
        "track": {k: header[k] for k in ("track", "title", "duration", "tempo", "downbeats")} | {"audio": f"{name}.m4a"},
        "tribe": {k: header[k] for k in ("steps", "rate_hz", "t0", "vertices", "z_clip")},
        "groups": list(GROUPS),
        "counts": {"brain": int(len(pos))},
        "bounds": {"brain_min": bv.min(0).round(2).tolist(), "brain_max": bv.max(0).round(2).tolist()},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {DATA} and {CACHE / (name + '.tribe')} ({len(times)} steps @ {rate:g} Hz)")


if __name__ == "__main__":
    main()
