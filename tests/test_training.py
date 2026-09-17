import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import training
from training import RobustNormalizer, load_trained, sample_training_targets, train_expert


def _base(sample_id: str, value: float = 0.2) -> dict:
    count = 13
    rows = np.zeros((count, 10), dtype=np.float32)
    rows[:, 0] = value
    rows[:, 1] = value * 2
    rows[:, 2] = 1
    rows[2, :9] = 0
    rows[:, 9] = 1
    return {
        "frame_times": torch.arange(count, dtype=torch.float64) / 25,
        "trajectory_base": torch.from_numpy(rows),
        "events": [{"frame_number": 6, "event_type": "hit"}],
        "metadata": {"sample_id": sample_id, "base_spec": {
            "coordinate_precision": "consistent_float32",
            "max_derivative_gap_seconds": 0.2,
            "acceleration_mode": "velocity_midpoint",
        }},
    }


class TrainingTest(unittest.TestCase):
    def test_train_cli_help(self):
        result = subprocess.run(
            [sys.executable, "train.py", "--help"], cwd=Path(__file__).parents[1],
            text=True, capture_output=True, check=True,
        )
        self.assertIn("--expert", result.stdout)

    def test_normalizer_and_soft_targets_match_default_contract(self):
        rows = _base("a")["trajectory_base"].numpy()
        normalizer = RobustNormalizer.fit([rows])
        transformed = normalizer.transform(rows)
        self.assertEqual(normalizer.columns, ("x_norm", "y_norm", "dx", "dy", "speed",
                                              "acceleration", "angle_change", "curvature"))
        self.assertEqual(transformed.shape, (13, 10))
        targets = sample_training_targets(
            13, [{"frame_number": 6, "event_type": "hit"}],
            positive_radius=2, ignore_radius=2, negative_ratio=5, seed=20260709, sigma=1.2,
        )
        positives = [row for row in targets if row.kind == "positive"]
        self.assertEqual([row.center for row in positives], [4, 5, 6, 7, 8])
        self.assertAlmostEqual(positives[0].eventness, np.exp(-4 / (2 * 1.2 ** 2)))

    def test_training_reads_only_manifest_train_and_resume_loads_last(self):
        loaded = []

        def load_base(_root, sample_id):
            loaded.append(sample_id)
            if sample_id == "back_match_eval":
                raise AssertionError("评估样本不得进入训练")
            return _base(sample_id)

        fake_dataset = SimpleNamespace(
            load_manifest=lambda _root: {
                "train": ["train_a"],
                "eval": ["back_match_eval"],
                "sources": {"train_a": "2606_admin_back", "back_match_eval": "back_match"},
            },
            load_base=load_base,
            load_visual=lambda _root, _sample_id: torch.zeros(13, 3840, dtype=torch.float16),
        )
        config = {"train": {"epochs": 2, "batch_size": 16, "learning_rate": 5e-4,
                            "weight_decay": 0.01, "min_learning_rate": 5e-5}}
        with tempfile.TemporaryDirectory() as directory, patch(
            "training._dataset_api", return_value=fake_dataset
        ):
            output = Path(directory)
            save = training._atomic_save

            def interrupt_after_first_epoch(payload, path):
                save(payload, path)
                if path.name == "last.pt" and payload.get("completed_epoch") == 1:
                    raise RuntimeError("simulated interruption")

            with patch("training._atomic_save", side_effect=interrupt_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_expert("trajectory", Path("unused"), output, config, torch.device("cpu"))
            self.assertEqual(set(loaded), {"train_a"})
            self.assertFalse((output / "trajectory_expert.pt").exists())
            self.assertTrue((output / "last.pt").is_file())
            report = train_expert(
                "trajectory", Path("unused"), output, config, torch.device("cpu"), resume=True
            )
            model, payload = load_trained(output / "trajectory_expert.pt", torch.device("cpu"))
            self.assertEqual(payload["training"]["train_sources"], ["2606_admin_back"])
            self.assertEqual(payload["training"]["validation_sample_ids"], [])
            self.assertEqual(payload["training"]["test_sample_ids"], [])
            self.assertEqual(report["completed_epoch"], 2)
            self.assertTrue(model.training is False)
            reference_dir = output / "reference"
            train_expert("trajectory", Path("unused"), reference_dir, config, torch.device("cpu"))
            reference = torch.load(reference_dir / "trajectory_expert.pt", weights_only=True)
            for name, value in payload["model_state"].items():
                torch.testing.assert_close(value, reference["model_state"][name], rtol=0, atol=0)

    def test_visual_smoke_uses_frozen_cached_features(self):
        fake_dataset = SimpleNamespace(
            load_manifest=lambda _root: {
                "train": ["v"], "eval": [], "sources": {"v": "published"},
            },
            load_base=lambda _root, sample_id: _base(sample_id),
            load_visual=lambda _root, _sample_id: torch.zeros(13, 3840, dtype=torch.float16),
        )
        config = {"train": {"epochs": 1, "batch_size": 16}}
        with tempfile.TemporaryDirectory() as directory, patch(
            "training._dataset_api", return_value=fake_dataset
        ):
            report = train_expert(
                "visual", Path("unused"), Path(directory), config, torch.device("cpu")
            )
            payload = torch.load(Path(directory) / "visual_expert.pt", weights_only=True)
            self.assertEqual(payload["contract"]["window_radius"], 24)
            self.assertEqual(payload["contract"]["window_span_seconds"], 1.6)
            self.assertEqual(report["completed_epoch"], 1)


if __name__ == "__main__":
    unittest.main()
