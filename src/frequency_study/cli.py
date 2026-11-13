"""Command line entry points for the manuscript protocol."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_QUOTAS, load_scenarios, mit_windows, save_scenarios, select_scenarios, self_windows
from .physics import GridConfig, exact_zoh, hard_indicators, net_disturbance_pu


def demo_dataset(path, seed=42):
    """Small deliberately synthetic smoke-test dataset, NOT paper data."""
    rng = np.random.default_rng(seed)
    cfg = GridConfig()
    rows = []
    for source in ("self", "mit"):
        for split, count in (("train", 40), ("val", 12), ("test", 12)):
            for i in range(count):
                start = rng.integers(1, 16)
                end = rng.integers(start + 3, 81)
                value = rng.uniform(-50, 60)
                delta = np.zeros(81)
                delta[start:end] = value
                if source == "mit":
                    delta += rng.uniform(-2, 2, 81)
                    delta[0] = 0
                u = net_disturbance_pu(delta, cfg)
                h, d = rng.uniform(2, 8), rng.uniform(15, 35)
                y = exact_zoh(u, h, d, cfg)
                _, _, unsafe = hard_indicators(y, cfg)
                rows.append((source, split, f"demo_{source}_{split}_{i}", u, h, d, y, int(unsafe)))
    arrays = {"source": np.array([x[0] for x in rows]), "split": np.array([x[1] for x in rows]),
              "window_id": np.array([x[2] for x in rows]), "u": np.stack([x[3] for x in rows]).astype(np.float32),
              "h": np.array([x[4] for x in rows], dtype=np.float32), "d": np.array([x[5] for x in rows], dtype=np.float32),
              "delta_f": np.stack([x[6] for x in rows]).astype(np.float32),
              "unsafe": np.array([x[7] for x in rows], dtype=np.int8)}
    save_scenarios(path, arrays, cfg, seed)
    print("Created synthetic SMOKE-TEST data only. It cannot reproduce manuscript results.")


def main():
    p = argparse.ArgumentParser(description="First-order frequency-response/PI-DeepONet study")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="prepare paper-protocol scenarios from supplied CSVs")
    prep.add_argument("--self-csv", required=True)
    prep.add_argument("--mit-csv", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--quotas", help="optional JSON with {source:{split:[safe,unsafe]}}")
    demo = sub.add_parser("demo", help="create small fake data for a code smoke test")
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int, default=42)
    tr = sub.add_parser("train", help="train model(s) using fixed prepared partitions")
    tr.add_argument("--data", required=True)
    tr.add_argument("--variant", required=True, choices=["data_only", "pi_baseline", "hard_ic", "structure", "hybrid", "risk_heads", "full_ema", "lstm", "fno"])
    tr.add_argument("--source", default="joint", choices=["joint", "self", "mit"])
    tr.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    tr.add_argument("--out-dir", required=True)
    tr.add_argument("--device", default="cpu")
    tr.add_argument("--phase1-epochs", type=int, default=200)
    tr.add_argument("--phase2-epochs", type=int, default=300)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--patience", type=int, default=30)
    ev = sub.add_parser("evaluate", help="evaluate an existing checkpoint")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--device", default="cpu")
    diag = sub.add_parser("diagnose", help="amplitude and in-support H,D diagnostics")
    diag.add_argument("--data", required=True)
    diag.add_argument("--checkpoint", required=True)
    diag.add_argument("--device", default="cpu")
    diag.add_argument("--out", required=True)
    plot = sub.add_parser("plot", help="create computed diagnostic figures from two checkpoints")
    plot.add_argument("--data", required=True)
    plot.add_argument("--pi-checkpoint", required=True)
    plot.add_argument("--lstm-checkpoint", required=True)
    plot.add_argument("--out-dir", required=True)
    plot.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.command == "prepare":
        quotas = DEFAULT_QUOTAS if not args.quotas else json.loads(Path(args.quotas).read_text())
        windows = list(self_windows(args.self_csv)) + list(mit_windows(args.mit_csv, selection_seed=args.seed))
        arrays = select_scenarios(windows, quotas, args.seed)
        save_scenarios(args.out, arrays, seed=args.seed)
        print(f"Saved {len(arrays['u'])} model-generated scenarios to {args.out}")
    elif args.command == "demo":
        demo_dataset(args.out, args.seed)
    elif args.command == "train":
        from .train import summarize_reports, train_one
        arrays = load_scenarios(args.data)
        out = Path(args.out_dir)
        reports = [train_one(arrays, args.variant, args.source, seed,
                             out / f"{args.variant}_{args.source}_seed{seed}.pt", args.device,
                             args.phase1_epochs, args.phase2_epochs, args.batch_size, args.patience)
                   for seed in args.seeds]
        (out / f"{args.variant}_{args.source}_summary.json").write_text(json.dumps(summarize_reports(reports), indent=2))
        print(json.dumps(summarize_reports(reports), indent=2))
    elif args.command in ("evaluate", "diagnose"):
        from .train import load_checkpoint, metrics, predict_arrays, selection
        arrays = load_scenarios(args.data)
        model, state = load_checkpoint(args.checkpoint, args.device)
        if args.command == "evaluate":
            for source in ("self", "mit"):
                ids = selection(arrays, "test", source)
                pred = predict_arrays(model, arrays, ids, state["mean"], state["std"], args.device)
                print(source, json.dumps(metrics(pred, arrays["delta_f"][ids]), indent=2))
        else:
            from .diagnostics import amplitude_quintiles, hd_grid
            result = {source: {"amplitude": amplitude_quintiles(model, state, arrays, source, args.device),
                               "hd_grid": hd_grid(model, state, arrays, source, args.device)}
                      for source in ("self", "mit")}
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2))
            print(f"Saved diagnostic results: {args.out}")
    elif args.command == "plot":
        from .plots import make_plots
        arrays = load_scenarios(args.data)
        print(f"Saved computed plots to {make_plots(arrays, args.pi_checkpoint, args.lstm_checkpoint, args.out_dir, args.device)}")


if __name__ == "__main__":
    main()
