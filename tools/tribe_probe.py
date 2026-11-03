"""Run TRIBE v2 (facebook/tribev2) on audio files: the offline step that produced the cached run Head Noise ships.

Needs a separate environment with the tribev2 package (see https://github.com/facebookresearch/tribev2), access to
its gated text backbone on Hugging Face, and enough memory (it ran on a 48 GB Apple silicon Mac, CPU feature
extractors). Then pack the result for the app with tools/export_data.py.

Usage (from the tribev2 environment, in the repo root):
    python tools/tribe_probe.py NAME=path/to/audio.wav [NAME=...]

Writes data/tribe/<NAME>/:
    preds.npy       (timesteps, 20484) predicted BOLD on fsaverage5, left hemi then right
    times.npy       segment start times in seconds (1 per second)
    words.csv       transcribed words with their times
    rois.csv        z-scored mean per region group, one row per timestep
    pca.csv         top principal components over time
    curves.png      ROI curves plot
    brain_*.png     cortex snapshots (mean response, peak-visual moment)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "tribe"
CACHE = Path(os.environ.get("TRIBE_CACHE", Path.home() / ".cache" / "tribev2"))

# Destrieux region names grouped into the knobs the visuals will use.
ROI_GROUPS = {
    "auditory": ["G_temp_sup-G_T_transv", "S_temporal_transverse", "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral"],
    "visual": ["Pole_occipital", "S_calcarine", "G_cuneus", "G_occipital_middle", "G_and_S_occipital_inf", "G_oc-temp_med-Lingual"],
    "language": ["G_front_inf-Opercular", "G_front_inf-Triangul", "S_temporal_sup", "G_pariet_inf-Supramar", "G_pariet_inf-Angular"],
    "motor": ["G_precentral", "S_central", "G_and_S_paracentral"],
    "default_mode": ["G_precuneus", "G_cingul-Post-dorsal", "G_cingul-Post-ventral", "G_and_S_cingul-Ant"],
    "insula_emotion": ["G_insular_short", "G_Ins_lg_and_S_cent_ins", "S_circular_insula_sup", "S_circular_insula_inf"],
}

_real_run = subprocess.run


def _patched_run(cmd, *args, **kwargs):
    # tribev2 shells out to `uvx whisperx` with float16, which CPU CTranslate2 rejects,
    # and uvx would otherwise pick Homebrew's Python 3.15, which torch has no wheels for.
    if isinstance(cmd, list) and cmd[:2] == ["uvx", "whisperx"]:
        cmd = ["uvx", "--python", "3.11"] + cmd[1:]
        if "--device" in cmd and cmd[cmd.index("--device") + 1] == "cpu":
            cmd[cmd.index("--compute_type") + 1] = "int8"
        print("[probe] whisperx:", " ".join(cmd), flush=True)
    return _real_run(cmd, *args, **kwargs)


subprocess.run = _patched_run


def _patch_word_timing():
    """Fix tribev2's ExtractWordsFromAudio for chunked audio.

    Upstream transcribes the whole file once, then copies the full transcript onto every
    60 s chunk shifted by start + offset, so each word appears once per chunk at the wrong
    time and the timeline runs to ~2x the track length. Here each chunk gets only its own
    words at their real times, and transcripts are cached in CACHE instead of beside the audio.
    """
    import pandas as pd
    from tribev2.eventstransforms import ExtractWordsFromAudio

    def _run(self, events):
        if "Word" in events.type.unique():
            return events
        audio = events.loc[events.type == "Audio"]
        chunks = []
        for path in audio.filepath.unique():
            tsv = CACHE / "transcripts" / (Path(path).stem + ".tsv")
            if tsv.exists():
                words = pd.read_csv(tsv, sep="\t") if tsv.stat().st_size > 1 else pd.DataFrame()
            else:
                words = self._get_transcript_from_audio(Path(path), self.language)
                tsv.parent.mkdir(parents=True, exist_ok=True)
                words.to_csv(tsv, sep="\t", index=False)
            if words.empty:
                continue
            for ev in audio.loc[audio.filepath == path].itertuples():
                inside = (words.start >= ev.offset) & (words.start < ev.offset + ev.duration)
                chunk = words.loc[inside].copy()
                if chunk.empty:
                    continue
                for k, v in ev._asdict().items():
                    if k not in ("Index", "frequency", "filepath", "type", "start", "duration", "offset"):
                        chunk.loc[:, k] = v
                chunk["type"] = "Word"
                chunk["language"] = self.language
                chunk["start"] += ev.start - ev.offset
                chunks.append(chunk)
        return pd.concat([events, *chunks], ignore_index=True) if chunks else events

    ExtractWordsFromAudio._run = _run


def log(msg):
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def roi_masks():
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in atlas["labels"]]
    full = np.concatenate([atlas["map_left"], atlas["map_right"]])
    masks = {}
    for group, names in ROI_GROUPS.items():
        ids = [labels.index(n) for n in names if n in labels]
        masks[group] = np.isin(full, ids)
    return masks


def export(name, preds, times):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "preds.npy", preds.astype(np.float32))
    np.save(out / "times.npy", times)

    z = (preds - preds.mean(0)) / (preds.std(0) + 1e-6)
    masks = roi_masks()
    rois = {g: z[:, m].mean(1) for g, m in masks.items() if m.any()}
    header = "time," + ",".join(rois)
    np.savetxt(out / "rois.csv", np.column_stack([times] + list(rois.values())), delimiter=",", header=header, comments="", fmt="%.4f")

    u, s, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
    k = min(6, u.shape[1])
    pcs = u[:, :k] * s[:k]
    explained = (s[:k] ** 2) / (s**2).sum()
    np.savetxt(out / "pca.csv", np.column_stack([times, pcs]), delimiter=",",
               header="time," + ",".join(f"pc{i+1}" for i in range(k)), comments="", fmt="%.4f")
    log(f"{name}: PCA explained variance {np.round(explained, 3).tolist()}")

    fig, axes = plt.subplots(len(rois), 1, figsize=(14, 1.6 * len(rois)), sharex=True)
    for ax, (g, c) in zip(axes, rois.items()):
        ax.plot(times, c, lw=1.2)
        ax.set_ylabel(g, rotation=0, ha="right", fontsize=9)
        ax.axhline(0, color="0.8", lw=0.5)
    axes[-1].set_xlabel("seconds")
    fig.suptitle(f"TRIBE v2 predicted response: {name}")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=130)
    plt.close(fig)

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    n = preds.shape[1] // 2
    frames = {"mean": preds.mean(0)}
    if "visual" in rois:
        peak = int(np.argmax(rois["visual"]))
        frames[f"peak_visual_{times[peak]:.0f}s"] = z[peak]
    for label, v in frames.items():
        fig, axs = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(10, 4))
        for ax, hemi, data in [(axs[0], "left", v[:n]), (axs[1], "right", v[n:])]:
            plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], data, hemi=hemi, view="lateral",
                                        bg_map=fs[f"sulc_{hemi}"], axes=ax, colorbar=False, cmap="inferno")
        fig.savefig(out / f"brain_{label}.png", dpi=130)
        plt.close(fig)
    log(f"{name}: wrote {out}")


def main(jobs):
    from tribev2 import TribeModel

    _patch_word_timing()
    log("loading TRIBE v2")
    # The checkpoint config pins every feature extractor to cuda; there's none on a Mac.
    cpu = {f"data.{k}.device": "cpu" for k in
           ["text_feature", "audio_feature", "image_feature.image", "video_feature.image"]}
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=str(CACHE), config_update=cpu)
    for name, path in jobs:
        t0 = time.time()
        log(f"{name}: building events (whisperx transcription) for {path}")
        df = model.get_events_dataframe(audio_path=path)
        n_words = int((df.type == "Word").sum()) if "type" in df else 0
        log(f"{name}: {len(df)} events, {n_words} words; predicting")
        preds, segments = model.predict(events=df)
        # One row per TR (1 s), already lag-corrected; drop anything past the end of the audio.
        times = np.array([s.start for s in segments])
        audio_end = df.loc[df.type == "Audio", "stop"].max()
        keep = times < audio_end
        preds, times = preds[keep], times[keep]
        log(f"{name}: preds {preds.shape} ({(~keep).sum()} rows past audio end dropped) in {time.time() - t0:.0f}s")
        export(name, preds, times)
        words = df.loc[df.type == "Word", ["start", "duration", "text"]].sort_values("start")
        words.to_csv(OUT / name / "words.csv", index=False)


if __name__ == "__main__":
    jobs = [arg.split("=", 1) for arg in sys.argv[1:]]
    if not jobs:
        sys.exit(__doc__)
    main(jobs)
