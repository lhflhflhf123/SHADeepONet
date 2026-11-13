# Data-center workload -> aggregate frequency response

This is an independent, executable implementation of the **methods stated in the
November 13, 2026 manuscript**. It includes the exact first-order label generator,
two workload conversions and leakage-aware partitions, class-selected scenario
construction, PI-DeepONet ablations, LSTM/FNO baselines, training, hard-threshold
screening metrics and diagnostic plots. **It contains no original workload data,
trained weights or fabricated reproduction of manuscript tables.**

## 1. First-order model (usable without PyTorch)

For a positive load increase `delta_load_mw`, define `u = -delta_load_mw/200`
(pu on a 200-MVA base). With `x = (f-50)/50`, `H` in seconds, dimensionless
`D`, `dt=0.125 s`, the exact zero-order-hold (ZOH) step is

```text
a = exp(-D*dt/(2*H))
x[k+1] = a*x[k] + (1-a)*u[k]/D
delta_f[k] = 50*x[k]  # Hz, x[0]=0
```

The model **does not** contain governor, turbine, network or inverter control
states. A 10-s trace has 81 samples; `u[k]` is held on `[t[k], t[k+1])`.
The last input sample is not integrated. The reported RoCoF metric uses
`max(abs(diff(delta_f)/0.125))`, not the instantaneous derivative at a jump.
Unsafe means **either** sampled minimum `<49.5 Hz` **or** sampled maximum
absolute RoCoF `>1.0 Hz/s`; equality alone is safe.

```bash
PYTHONPATH=src python - <<'PY'
import numpy as np
from frequency_study.physics import GridConfig, exact_zoh, net_disturbance_pu, hard_indicators
c = GridConfig()
u = net_disturbance_pu(np.full(c.n_points, 30.0))
y = exact_zoh(u, h_s=4.0, damping=20.0)
print("first/last deviation (Hz):", y[0], y[-1])
print("minimum, peak RoCoF, unsafe:", hard_indicators(y))
PY
```

## 2. Install and required input

Use Python 3.10+ and install via `python -m pip install -e .`. PyTorch is
required for training, but the NumPy physics generator and its tests run
without it. A CUDA GPU is recommended for the full five-seed study.

Supply your own two CSV files:

| File | Required columns | Meaning |
| --- | --- | --- |
| Self-built | `time_s,p_total_MW` | Monotonic trace time in seconds; total facility power in MW (paper says synthetic 30-day, 1-s profile). |
| MIT-preprocessed | `job_id,time_s,utilization_percent` | Per-job time in seconds and GPU utilization in percentage **points** [0,100]. |

Repeated timestamps within each job are averaged. The code hash-ranks distinct
MIT job IDs, takes 400, splits them into 280/60/60 jobs and keeps up to eight
hash-selected, non-overlapping 10-s windows per job. Self-built input is split
chronologically 70/15/15 before extracting windows, with a total 10-s purge
centered on each boundary. Each window is interpolated to
81 samples. Input windows with peak absolute load change outside 0.10–60 MW
are discarded. MIT perturbations use the **assumed synthetic** rule
`delta_load_mw = 0.60*(utilization_percent - first_percent)`; this is not
facility-meter calibration. The input CSV must already contain the actual
selected source records. The paper does not provide a machine-readable
original 30-day task/power-generation process or full MIT object manifest.

```bash
frequency-study prepare --self-csv self_trace.csv --mit-csv mit_jobs.csv \
  --out data/scenarios.npz --seed 42
```

Within each source partition, the generator draws `H ~ U[2,8]` and
`D ~ U[15,35]` in candidate batches of 10 (at most 100 pairs per window),
generates exact ZOH labels, and selects scenarios to meet the manuscript's
60:40 safe/unsafe allocation. It fails with missing counts if the inputs cannot
meet the quotas. A `.manifest.json` next to the `.npz` records actual counts.
You may provide `--quotas quotas.json` to run a smaller experiment; that
changes the paper protocol.

For a quick **synthetic smoke test only**, no raw files are needed:

```bash
frequency-study demo --out data/demo.npz
```

The demo does not have the paper's sample counts or workload distributions.

## 3. Models and training

The full PI model has three encoders with three 48-unit `tanh` layers and a
48-dimensional linear output, plus a 48-unit `tanh` fusion layer, totaling
**27,745 parameters**. The disturbance RMS amplitude and signed shape,
paired odd projection and physical-time multiplication implement exact signed
amplitude equivariance and zero initial deviation. The training loss uses
the manuscript's smooth/change masks, discrete differential and integral
defects, bounded log-sum-exp indicator heads (`beta_f=1000 Hz^-1`,
`beta_r=500 s/Hz`), and optional EMA inverse-loss weights. The seven cumulative
variants are `data_only`, `pi_baseline`, `hard_ic`, `structure`, `hybrid`,
`risk_heads`, `full_ema`.

Default PI training uses Adam `(0.9,0.999)`, batch 128, phase I at most 200
epochs / learning rate `1e-3`, phase II at most 300 epochs / `1e-4`, gradient
clip 1.0, patience 30, and seeds 41–45. Source-only training uses its own
train/val partitions. Joint training selects the checkpoint by the **unweighted
mean of the two validation-domain trajectory RMSEs**, never test results.
Checkpoint, per-seed metrics and mean/sample-SD JSON are saved separately.

```bash
frequency-study train --data data/scenarios.npz --variant full_ema \
  --source joint --device cuda --out-dir runs/pi_joint
frequency-study train --data data/scenarios.npz --variant lstm \
  --source joint --device cuda --out-dir runs/lstm_joint
frequency-study train --data data/scenarios.npz --variant full_ema \
  --source self --device cuda --out-dir runs/pi_self
frequency-study train --data data/scenarios.npz --variant full_ema \
  --source mit --device cuda --out-dir runs/pi_mit
```

Use `--source self`/`mit` for the transfer arms. Run LSTM with the same arms to
form the two-model transfer comparison. `fno` and all PI variants are also
available via `--variant`. Because the manuscript only specifies LSTM/FNO
width/block counts, their remaining architectural choices here (time channel,
FNO projection and the baselines' supervised-only 200-epoch ceiling) are
**implementation choices**, not a claim to recover unavailable original code.

For a very quick test, use `--data data/demo.npz --seeds 41 --phase1-epochs 2
--phase2-epochs 2 --batch-size 16 --patience 2 --device cpu`.

## 4. Evaluation and plots

```bash
frequency-study evaluate --data data/scenarios.npz \
  --checkpoint runs/pi_joint/full_ema_joint_seed42.pt --device cuda
frequency-study diagnose --data data/scenarios.npz \
  --checkpoint runs/pi_joint/full_ema_joint_seed42.pt \
  --out runs/diagnostics.json --device cuda
frequency-study plot --data data/scenarios.npz \
  --pi-checkpoint runs/pi_joint/full_ema_joint_seed42.pt \
  --lstm-checkpoint runs/lstm_joint/lstm_joint_seed42.pt \
  --out-dir runs/figures --device cuda
```

`diagnose` computes both pooled and mean-per-window RMSE per amplitude quintile
and the 5x5 **within-support** `H,D` grid with freshly generated reference
labels and fixed model weights. The plots are calculated from the actual
checkpoint predictions and not populated from paper table entries. The
example trajectories select the median-amplitude test scenario, which need
not be the two selected Q3 scenarios of the manuscript.

Overall RMSE is pooled over all samples and time points. The predictor's
sampled hard extrema determine screening metrics; smooth heads train only.
The paper's reported numeric values can only be checked when the actual
original input traces, exact splits/selection manifest and trained runs are
provided. This package is a transparent reimplementation, not verification
that the manuscript's published figures were produced by these scripts.

## 5. Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src
```

The NumPy tests check the exact ZOH solution against an independent closed
form, load-sign convention, initial slope versus sampled RoCoF, strict OR
thresholds and the expected nonzero discrete regularizer defect. With
PyTorch installed they also test parameter count, hard initial condition,
signed amplitude equivariance, and smooth-head bounds.
