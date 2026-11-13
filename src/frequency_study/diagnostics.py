"""Conditional error and in-support parameter-sensitivity diagnostics."""

import numpy as np

from .physics import GridConfig, exact_zoh
from .train import predict_arrays, selection


def amplitude_quintiles(model, state, arrays, source, device="cpu"):
    ids = selection(arrays, "test", source)
    peak_mw = np.max(np.abs(arrays["u"][ids]), axis=1) * 200.0
    pred = predict_arrays(model, arrays, ids, state["mean"], state["std"], device)
    order = np.argsort(peak_mw, kind="stable")
    result = []
    for n, bin_ids in enumerate(np.array_split(order, 5), 1):
        errors = pred[bin_ids] - arrays["delta_f"][ids][bin_ids]
        result.append({"quintile": n, "n": len(bin_ids), "median_peak_mw": float(np.median(peak_mw[bin_ids])),
                       "min_peak_mw": float(np.min(peak_mw[bin_ids])), "max_peak_mw": float(np.max(peak_mw[bin_ids])),
                       "pooled_rmse_mhz": float(np.sqrt(np.mean(errors ** 2)) * 1000),
                       "mean_per_window_rmse_mhz": float(np.mean(np.sqrt(np.mean(errors ** 2, axis=1))) * 1000)})
    return result


def hd_grid(model, state, arrays, source, device="cpu", h_values=(2, 3.5, 5, 6.5, 8),
            d_values=(15, 20, 25, 30, 35), cfg=GridConfig()):
    ids = selection(arrays, "test", source)
    grid = []
    for h in h_values:
        for d in d_values:
            temp = dict(arrays)
            temp["h"] = arrays["h"].copy()
            temp["d"] = arrays["d"].copy()
            temp["h"][ids] = h
            temp["d"][ids] = d
            truth = exact_zoh(arrays["u"][ids], h, d, cfg)
            pred = predict_arrays(model, temp, ids, state["mean"], state["std"], device)
            grid.append({"h_s": h, "d_dimensionless": d,
                         "rmse_mhz": float(np.sqrt(np.mean((pred - truth) ** 2)) * 1000)})
    return grid
