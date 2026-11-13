import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from frequency_study.data import Window, mit_windows, select_scenarios, self_windows
from frequency_study.physics import GridConfig


class PartitionTests(unittest.TestCase):
    def test_chronological_windows_do_not_cross_purged_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "self.csv"
            t = np.arange(0, 401, dtype=float)
            pd.DataFrame({"time_s": t, "p_total_MW": 100 + 2 * np.sin(t / 3)}).to_csv(path, index=False)
            windows = list(self_windows(path))
            self.assertTrue({w.split for w in windows} == {"train", "val", "test"})
            by_split = {name: [w.identifier for w in windows if w.split == name] for name in ("train", "val", "test")}
            self.assertEqual(len({w.identifier for w in windows}), len(windows))
            self.assertTrue(all(by_split.values()))

    def test_mit_job_partition_and_class_quotas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mit.csv"
            rows = [(f"j{job:03d}", t, float(t * 3)) for job in range(400) for t in range(11)]
            pd.DataFrame(rows, columns=["job_id", "time_s", "utilization_percent"]).to_csv(path, index=False)
            windows = list(mit_windows(path))
            job_split = {}
            for w in windows:
                job_id = w.identifier.split(":")[2]
                job_split.setdefault(job_id, set()).add(w.split)
            self.assertEqual({k: sum(w.split == k for w in windows) for k in ("train", "val", "test")},
                             {"train": 280, "val": 60, "test": 60})
            self.assertTrue(all(len(v) == 1 for v in job_split.values()))

            # One 50-MW step window produces safe and unsafe responses over H,D.
            w = Window("self", "train", "toy", np.r_[np.zeros(1), np.full(80, 50.0)])
            arrays = select_scenarios([w], {"self": {"train": (2, 2)}}, seed=42)
            self.assertEqual(len(arrays["u"]), 4)
            self.assertEqual(int(arrays["unsafe"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
