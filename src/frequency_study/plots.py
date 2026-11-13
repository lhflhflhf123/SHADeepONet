"""Plots generated from model predictions, never from manuscript table values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import amplitude_quintiles, hd_grid
from .physics import GridConfig
from .train import load_checkpoint, predict_arrays, selection


def make_plots(arrays, pi_path, lstm_path, out_dir, device="cpu"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pi, ps = load_checkpoint(pi_path, device)
    lstm, ls = load_checkpoint(lstm_path, device)
    cfg = GridConfig()
    colors = {"PI-DeepONet": "#087e8b", "LSTM": "#e07a2f"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, source in zip(axes, ("self", "mit")):
        for name, model, state in (("PI-DeepONet", pi, ps), ("LSTM", lstm, ls)):
            q = amplitude_quintiles(model, state, arrays, source, device)
            ax.plot([v["median_peak_mw"] for v in q], [v["mean_per_window_rmse_mhz"] for v in q],
                    marker="o", label=name, color=colors[name])
        ax.set(title=source, xlabel="Median peak |load change| (MW)", ylabel="Mean per-window RMSE (mHz)")
        if source == "mit":
            ax.set_xscale("log")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.savefig(out / "amplitude_conditioned.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for row, (name, model, state) in enumerate((("PI-DeepONet", pi, ps), ("LSTM", lstm, ls))):
        for col, source in enumerate(("self", "mit")):
            cells = hd_grid(model, state, arrays, source, device)
            grid = np.array([c["rmse_mhz"] for c in cells]).reshape(5, 5)
            ax = axes[row, col]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
            ax.set(xticks=range(5), yticks=range(5), xticklabels=[15,20,25,30,35],
                   yticklabels=[2,3.5,5,6.5,8], xlabel="D (dimensionless)", ylabel="H (s)", title=f"{name}: {source}")
            for i in range(5):
                for j in range(5):
                    ax.text(j, i, f"{grid[i,j]:.1f}", ha="center", va="center", color="white", fontsize=7)
            fig.colorbar(im, ax=ax, label="Pooled RMSE (mHz)")
    fig.savefig(out / "hd_sensitivity.png", dpi=250)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for col, source in enumerate(("self", "mit")):
        ids = selection(arrays, "test", source)
        truth = arrays["delta_f"][ids]
        f_ref = 50 + truth.min(axis=1)
        r_ref = np.abs(np.diff(truth, axis=1) / cfg.dt).max(axis=1)
        for name, model, state in (("PI-DeepONet", pi, ps), ("LSTM", lstm, ls)):
            pred = predict_arrays(model, arrays, ids, state["mean"], state["std"], device)
            f_pred = 50 + pred.min(axis=1)
            r_pred = np.abs(np.diff(pred, axis=1) / cfg.dt).max(axis=1)
            axes[0, col].scatter(f_ref, f_pred, s=5, alpha=.25, label=name, color=colors[name])
            axes[1, col].scatter(r_ref, r_pred, s=5, alpha=.25, label=name, color=colors[name])
        for row, (label, threshold) in enumerate((("Minimum frequency (Hz)", 49.5), ("Peak |RoCoF| (Hz/s)", 1.0))):
            ax = axes[row, col]
            ax.axline((threshold, threshold), slope=1, color="black", linestyle="--", lw=.7)
            ax.axhline(threshold, color="gray", linestyle=":", lw=.7)
            ax.axvline(threshold, color="gray", linestyle=":", lw=.7)
            ax.set(title=f"{source}: {label}", xlabel=f"Reference {label}", ylabel=f"Predicted {label}")
    axes[0,0].legend(frameon=False, markerscale=3)
    fig.savefig(out / "indicator_scatter.png", dpi=250)
    plt.close(fig)

    # Example selection is deterministic and visibly identified; no cherry-picked
    # claim of typicality is attached to this qualitative plot.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, source in zip(axes, ("self", "mit")):
        ids = selection(arrays, "test", source)
        peaks = np.max(np.abs(arrays["u"][ids]), axis=1)
        mid = np.argsort(peaks)[len(peaks)//2]
        idx = ids[mid]
        ax.plot(cfg.times, 50 + arrays["delta_f"][idx], color="black", label="Exact ZOH")
        for name, model, state in (("PI-DeepONet", pi, ps), ("LSTM", lstm, ls)):
            pred = predict_arrays(model, arrays, np.array([idx]), state["mean"], state["std"], device)[0]
            ax.plot(cfg.times, 50 + pred, label=name, color=colors[name])
        ax.set(title=f"{source}: {arrays['window_id'][idx]}", xlabel="Time (s)", ylabel="Frequency (Hz)")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    fig.savefig(out / "illustrative_trajectories.png", dpi=250)
    plt.close(fig)
    return out
