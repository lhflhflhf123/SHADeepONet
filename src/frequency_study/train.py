"""Five-seed train/evaluate protocol with domain-macro validation selection."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import load_scenarios
from .losses import EMALossWeights, component_losses, hard_indicators_torch
from .models import PIDeepONet, make_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selection(arrays, split, source="joint"):
    mask = arrays["split"] == split
    if source != "joint":
        mask &= arrays["source"] == source
    ids = np.where(mask)[0]
    if len(ids) == 0:
        raise ValueError(f"empty {split}/{source} partition")
    return ids


def fit_scaler(arrays, ids):
    values = np.log(np.stack((arrays["h"][ids], arrays["d"][ids]), axis=-1).astype(np.float64))
    return values.mean(0).astype(np.float32), np.maximum(values.std(0), 1e-6).astype(np.float32)


def make_loader(arrays, ids, mean, std, batch_size, shuffle, seed):
    eta = (np.log(np.stack((arrays["h"][ids], arrays["d"][ids]), axis=-1)) - mean) / std
    ds = TensorDataset(*(torch.from_numpy(x.astype(np.float32)) for x in (
        arrays["u"][ids], eta, arrays["h"][ids], arrays["d"][ids], arrays["delta_f"][ids])))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def predict_arrays(model, arrays, ids, mean, std, device="cpu", batch_size=512):
    model.eval()
    times = torch.linspace(0, 10, 81, device=device)
    loader = make_loader(arrays, ids, mean, std, batch_size, False, 0)
    predictions = []
    for u, eta, _, _, _ in loader:
        predictions.append(model(u.to(device), eta.to(device), times).cpu().numpy())
    return np.concatenate(predictions)


def metrics(pred, truth):
    pred = torch.from_numpy(pred)
    truth = torch.from_numpy(truth)
    fp, rp, up = hard_indicators_torch(pred)
    ft, rt, ut = hard_indicators_torch(truth)
    tp, fn, false_positive, tn = ((up & ut).sum().item(), ((~up) & ut).sum().item(),
                                  (up & (~ut)).sum().item(), ((~up) & (~ut)).sum().item())
    return {
        "rmse_mhz": float(((pred - truth).square().mean().sqrt() * 1000)),
        "minimum_mae_mhz": float(((fp - ft).abs().mean() * 1000)),
        "rocof_mae_hz_s": float((rp - rt).abs().mean()),
        "recall_percent": 100 * tp / (tp + fn) if tp + fn else None,
        "fpr_percent": 100 * false_positive / (false_positive + tn) if false_positive + tn else None,
        "tp": tp, "fn": fn, "fp": false_positive, "tn": tn,
    }


def validation_rmse(model, arrays, mean, std, source, device):
    domains = ("self", "mit") if source == "joint" else (source,)
    values = []
    for domain in domains:
        ids = selection(arrays, "val", domain)
        prediction = predict_arrays(model, arrays, ids, mean, std, device)
        values.append(float(np.sqrt(np.mean((prediction - arrays["delta_f"][ids]) ** 2))))
    return float(np.mean(values))  # unweighted domain macro-average, in Hz


def train_one(arrays, variant, source, seed, out_path, device="cpu", phase1_epochs=200,
              phase2_epochs=300, batch_size=128, patience=30):
    if source not in ("joint", "self", "mit"):
        raise ValueError("source must be joint, self or mit")
    set_seed(seed)
    train_ids = selection(arrays, "train", source)
    mean, std = fit_scaler(arrays, train_ids)
    loader = make_loader(arrays, train_ids, mean, std, batch_size, True, seed)
    model = make_model(variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(.9, .999))
    times = torch.linspace(0, 10, 81, device=device)
    history = []
    for phase, ceiling in ((1, phase1_epochs), (2, phase2_epochs)):
        if phase == 2 and not isinstance(model, PIDeepONet):
            break  # sequence baselines use supervised-only training
        if ceiling == 0:
            continue
        for group in optimizer.param_groups:
            group["lr"] = 1e-3 if phase == 1 else 1e-4
        ema = EMALossWeights(decay=.9, epsilon=1e-8) if phase == 2 and model.ema else None
        best, best_state, stale = float("inf"), None, 0
        for epoch in range(ceiling):
            model.train()
            total = 0.0
            for u, eta, h, d, truth in loader:
                u, eta, h, d, truth = [x.to(device) for x in (u, eta, h, d, truth)]
                optimizer.zero_grad(set_to_none=True)
                pred = model(u, eta, times)
                parts = {"d": (pred - truth).square().mean()} if phase == 1 else component_losses(pred, truth, u, h, d, model)
                loss = ema.combine(parts) if ema else sum(parts.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(u)
            score = validation_rmse(model, arrays, mean, std, source, device)
            history.append({"phase": phase, "epoch": epoch + 1, "train_loss": total / len(train_ids), "val_macro_rmse_hz": score})
            if score < best - 1e-10:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        # Phase II starts from the Phase I validation checkpoint, and its own
        # selected checkpoint is retained at the end of training.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "variant": variant, "source": source,
                "seed": seed, "mean": mean, "std": std, "history": history}, out_path)
    report = {"seed": seed, "variant": variant, "source_training": source, "best_validation_rmse_hz": best,
              "tests": {}}
    model.to(device)
    for domain in ("self", "mit"):
        ids = selection(arrays, "test", domain)
        pred = predict_arrays(model, arrays, ids, mean, std, device)
        report["tests"][domain] = metrics(pred, arrays["delta_f"][ids])
    out_path.with_suffix(".metrics.json").write_text(json.dumps(report, indent=2))
    return report


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(state["variant"])
    model.load_state_dict(state["model"])
    return model.to(device), state


def summarize_reports(reports):
    output = {}
    for domain in ("self", "mit"):
        output[domain] = {}
        keys = ("rmse_mhz", "minimum_mae_mhz", "rocof_mae_hz_s", "recall_percent", "fpr_percent")
        for key in keys:
            values = [r["tests"][domain][key] for r in reports if r["tests"][domain][key] is not None]
            output[domain][key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
    return output
