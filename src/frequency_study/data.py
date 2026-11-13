"""Auditable two-source workload preprocessing and balanced scenario selection.

Input CSVs are *user supplied*. No published paper values are embedded as
results. Selection is deterministic for a given input, seed and quotas.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import GridConfig, exact_zoh, hard_indicators, net_disturbance_pu


DEFAULT_QUOTAS = {
    "self": {"train": (3780, 2520), "val": (810, 540), "test": (810, 540)},
    "mit": {"train": (2688, 1792), "val": (576, 384), "test": (576, 384)},
}


def rank_key(value, salt=""):
    return hashlib.sha256((salt + str(value)).encode("utf-8")).hexdigest()


@dataclass
class Window:
    source: str
    split: str
    identifier: str
    delta_load_mw: np.ndarray


def _numeric_group(frame, value_col):
    clean = frame[["time_s", value_col]].dropna().astype(float)
    clean = clean.groupby("time_s", as_index=False)[value_col].mean()
    clean = clean.sort_values("time_s")
    if len(clean) < 2 or not np.isfinite(clean.to_numpy()).all():
        raise ValueError("each trace needs at least two finite time/value rows")
    return clean.time_s.to_numpy(), clean[value_col].to_numpy()


def _windows(t, values, source, split, identity, cfg, mw_per_percent=0.60):
    # Adjacent 10-s windows may share an endpoint; no trace crosses a split.
    start = float(t[0])
    n = int(np.floor((t[-1] - start) / cfg.duration_s + 1e-9))
    for j in range(n):
        q = start + j * cfg.duration_s + cfg.times
        if q[-1] > t[-1] + 1e-8:
            break
        v = np.interp(q, t, values)
        delta = v - v[0]
        if source == "mit":
            delta = delta * mw_per_percent
        peak = np.max(np.abs(delta))
        if 0.10 <= peak <= 60.0:
            yield Window(source, split, f"{source}:{split}:{identity}:{j}", delta.astype(np.float64))


def self_windows(path, cfg=GridConfig()):
    """Chronological 70/15/15, 10-s boundary purge, then 10-s windows."""
    frame = pd.read_csv(path)
    required = {"time_s", "p_total_MW"}
    if not required.issubset(frame.columns):
        raise ValueError(f"self CSV requires {sorted(required)}")
    t, p = _numeric_group(frame, "p_total_MW")
    span = t[-1] - t[0]
    cuts = [t[0], t[0] + 0.70 * span, t[0] + 0.85 * span, t[-1]]
    # Exclude a total 10-s band centered on each chronological split boundary.
    half_purge = cfg.duration_s / 2
    for i, split in enumerate(("train", "val", "test")):
        lo = cuts[i] + (half_purge if i else 0.0)
        hi = cuts[i + 1] - (half_purge if i < 2 else 0.0)
        mask = (t >= lo) & (t <= hi)
        if mask.sum() >= 2:
            yield from _windows(t[mask], p[mask], "self", split, "self", cfg)


def mit_windows(path, cfg=GridConfig(), selection_seed=42):
    """Select 400 hash-ranked job IDs; allocate 280/60/60 before windowing."""
    frame = pd.read_csv(path, dtype={"job_id": str})
    required = {"job_id", "time_s", "utilization_percent"}
    if not required.issubset(frame.columns):
        raise ValueError(f"MIT CSV requires {sorted(required)}")
    jobs = sorted(frame.job_id.dropna().unique(), key=lambda x: rank_key(x, str(selection_seed)))
    if len(jobs) < 400:
        raise ValueError("paper protocol requires at least 400 distinct MIT jobs")
    chosen = jobs[:400]
    split_of = {j: ("train" if i < 280 else "val" if i < 340 else "test") for i, j in enumerate(chosen)}
    selected = frame[frame.job_id.isin(chosen)]
    groups = {job: group for job, group in selected.groupby("job_id", sort=False)}
    for job in chosen:
        group = groups[job]
        t, u = _numeric_group(group, "utilization_percent")
        if np.min(u) < 0 or np.max(u) > 100:
            raise ValueError(f"utilization percentage outside [0,100] for job {job}")
        options = list(_windows(t, u, "mit", split_of[job], job, cfg))
        options.sort(key=lambda w: rank_key(w.identifier, str(selection_seed)))
        yield from options[:8]


def select_scenarios(windows, quotas=DEFAULT_QUOTAS, seed=42, cfg=GridConfig()):
    """Draw up to 100 H,D pairs/window in batches of 10; balance by class.

    Raises if the input traces cannot meet requested quotas. Neither raw
    traces nor exact published selections are supplied with the manuscript.
    """
    rng = np.random.default_rng(seed)
    windows = list(windows)
    rng.shuffle(windows)
    bins = {(source, split, unsafe): [] for source in quotas for split in quotas[source] for unsafe in (0, 1)}
    for w in windows:
        key0, key1 = (w.source, w.split, 0), (w.source, w.split, 1)
        if len(bins[key0]) >= quotas[w.source][w.split][0] and len(bins[key1]) >= quotas[w.source][w.split][1]:
            continue
        u = net_disturbance_pu(w.delta_load_mw, cfg)
        for offset in range(0, 100, 10):
            h = rng.uniform(2.0, 8.0, 10)
            d = rng.uniform(15.0, 35.0, 10)
            y = exact_zoh(np.broadcast_to(u, (10, cfg.n_points)), h, d, cfg)
            _, _, unsafe = hard_indicators(y, cfg)
            for k in rng.permutation(10):
                key = (w.source, w.split, int(unsafe[k]))
                if len(bins[key]) < quotas[w.source][w.split][int(unsafe[k])]:
                    bins[key].append((u.copy(), float(h[k]), float(d[k]), y[k].copy(), w.identifier))
            if len(bins[key0]) >= quotas[w.source][w.split][0] and len(bins[key1]) >= quotas[w.source][w.split][1]:
                break
    missing = {str(k): quotas[k[0]][k[1]][k[2]] - len(v) for k, v in bins.items() if len(v) < quotas[k[0]][k[1]][k[2]]}
    if missing:
        raise ValueError(f"insufficient eligible workload/parameter candidates for quotas: {missing}")
    rows = []
    for key, group in bins.items():
        for u, h, d, y, window_id in group:
            rows.append((key[0], key[1], key[2], window_id, u, h, d, y))
    rng.shuffle(rows)
    return {
        "source": np.array([x[0] for x in rows]),
        "split": np.array([x[1] for x in rows]),
        "unsafe": np.array([x[2] for x in rows], dtype=np.int8),
        "window_id": np.array([x[3] for x in rows]),
        "u": np.stack([x[4] for x in rows]).astype(np.float32),
        "h": np.array([x[5] for x in rows], dtype=np.float32),
        "d": np.array([x[6] for x in rows], dtype=np.float32),
        "delta_f": np.stack([x[7] for x in rows]).astype(np.float32),
    }


def save_scenarios(path, arrays, cfg=GridConfig(), seed=42):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    manifest = {
        "seed": seed, "grid": vars(cfg), "counts": {
            f"{src}/{split}/{lab}": int(np.sum((arrays["source"] == src) & (arrays["split"] == split) & (arrays["unsafe"] == lab)))
            for src in ("self", "mit") for split in ("train", "val", "test") for lab in (0, 1)
        },
        "warning": "No scenario is a measured grid-frequency record; MIT MW values use an assumed 0.60 MW/percentage-point conversion.",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))


def load_scenarios(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
