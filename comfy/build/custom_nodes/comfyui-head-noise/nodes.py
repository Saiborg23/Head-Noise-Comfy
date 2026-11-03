"""Head Noise nodes for the Comfy developer platform.

    TribeV2Predict      AUDIO -> TRIBE v2 predicted cortex (fsaverage5, 20,484 vertices, 1 Hz) as a .tribe file
    HeadNoiseSaveMesh   MESH  -> GLB, reported as a plain `files` output so the Comfy API v2 hands it back as an asset

TRIBE v2 (facebook/tribev2) needs ~28 GB and minutes per track, so for the event demo the node serves a run made
offline with tools/tribe_probe.py, bundled in cache/. It picks the cached run whose length matches the audio.
Swapping in live inference only changes `_predict`; the file format and everything downstream stay the same.

.tribe layout: b"TRIB", uint32 LE header length, UTF-8 JSON header, then steps x vertices uint8 z-scores
(per vertex over time, clipped to +-z_clip). The web app reads it directly.
"""

import json
import os
import struct
import time
from pathlib import Path

import folder_paths
from comfy.utils import ProgressBar
from comfy_api.latest import IO, UI
from comfy_extras.nodes_save_3d import mesh_item_to_glb_bytes

CACHE = Path(__file__).parent / "cache"
MAGIC = b"TRIB"
MATCH_TOLERANCE_S = 0.75


def read_tribe(path):
    raw = Path(path).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path} is not a .tribe file")
    n = struct.unpack("<I", raw[4:8])[0]
    return json.loads(raw[8:8 + n]), raw[8 + n:]


def write_tribe(path, header, data):
    h = json.dumps(header, separators=(",", ":")).encode()
    Path(path).write_bytes(MAGIC + struct.pack("<I", len(h)) + h + data)


def _cached_runs():
    runs = []
    for p in sorted(CACHE.glob("*.tribe")):
        header, _ = read_tribe(p)
        runs.append((p, header))
    if not runs:
        raise RuntimeError(f"no TRIBE runs in {CACHE}")
    return runs


def _predict(duration):
    """The cached run closest in length to the audio; `matched` says whether it is this track's own run."""
    runs = _cached_runs()
    path, header = min(runs, key=lambda r: abs(r[1]["duration"] - duration))
    header = dict(header, matched=abs(header["duration"] - duration) <= MATCH_TOLERANCE_S, audio_duration=duration)
    return header, read_tribe(path)[1]


def _save_output(subfolder, prefix, ext, write):
    out_dir, name, counter, sub, _ = folder_paths.get_save_image_path(f"{subfolder}/{prefix}", folder_paths.get_output_directory())
    filename = f"{name}_{counter:05}_.{ext}"
    write(os.path.join(out_dir, filename))
    return {"filename": filename, "subfolder": sub, "type": "output"}


class TribeV2Predict(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TribeV2Predict",
            display_name="TRIBE v2 Predict",
            category="head_noise",
            description="Predicted fMRI response to the audio on the fsaverage5 cortex (20,484 vertices, 1 Hz).",
            is_output_node=True,
            inputs=[
                IO.Audio.Input("audio"),
                IO.Float.Input("pacing_seconds", default=4.0, min=0.0, max=120.0, step=0.5,
                               tooltip="Progress pacing while the cached run is served."),
            ],
            outputs=[IO.String.Output("preds_file", tooltip="Output-relative path of the .tribe file")],
        )

    @classmethod
    def execute(cls, audio, pacing_seconds) -> IO.NodeOutput:
        duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        header, data = _predict(duration)
        steps = header["steps"]
        bar = ProgressBar(steps)
        for _ in range(steps):
            if pacing_seconds:
                time.sleep(pacing_seconds / steps)
            bar.update(1)
        saved = _save_output("tribe", Path(header["track"]).stem, "tribe", lambda p: write_tribe(p, header, data))
        return IO.NodeOutput(f"{saved['subfolder']}/{saved['filename']}", ui={"files": [saved]})


class HeadNoiseSaveMesh(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="HeadNoiseSaveMesh",
            display_name="Save Head Mesh (GLB)",
            category="head_noise",
            description="Saves the first mesh of the batch as GLB under the `files` output key.",
            is_output_node=True,
            inputs=[
                IO.Mesh.Input("mesh"),
                IO.String.Input("filename_prefix", default="face"),
            ],
        )

    @classmethod
    def execute(cls, mesh, filename_prefix) -> IO.NodeOutput:
        glb = mesh_item_to_glb_bytes(mesh, 0, {})
        if glb is None:
            raise ValueError("the mesh is empty; try a lower VoxelToMesh threshold or a cleaner face photo")
        saved = _save_output("head_noise", filename_prefix, "glb", lambda p: Path(p).write_bytes(glb))
        return IO.NodeOutput(ui={"files": [saved]})


NODES = [TribeV2Predict, HeadNoiseSaveMesh]
