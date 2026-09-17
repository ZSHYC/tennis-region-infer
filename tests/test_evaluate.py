import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import evaluate
from dataset import Event
from metrics import ScoreRow, evaluate_all, match_events, scores_to_events


class EvaluateTest(unittest.TestCase):
    def test_matching_maximizes_count_before_distance_and_nms_is_per_class(self):
        gt = [Event("s", 0, "hit"), Event("s", 3, "hit")]
        pred = [Event("s", 2, "hit"), Event("s", 5, "hit")]
        self.assertEqual(len(match_events(gt, pred, 2)), 2)
        rows = [ScoreRow("s", 0, 0.4, 0.7), ScoreRow("s", 5, 0.4, 0.0), ScoreRow("s", 6, 0.4, 0.0)]
        events = scores_to_events(rows, {"hit": 0.4, "bounce": 0.4}, 5)
        self.assertEqual([(e.frame_number, e.event_type) for e in events],
                         [(0, "bounce"), (0, "hit"), (6, "hit")])
        result = evaluate_all(gt, pred, {"s": np.arange(6, dtype=float)})
        self.assertEqual(result["frames"]["2"]["overall"]["tp"], 2)

    def test_evaluation_reads_only_eval_cache_and_uses_fixed_threshold(self):
        manifest = {"train": ["other"], "eval": ["01"],
                    "sources": {"other": "loveall", "01": "back_match_clipped"}}
        base = {"frame_times": torch.arange(8, dtype=torch.float64),
                "trajectory_base": torch.zeros(8, 10),
                "events": [{"frame_number": 2, "event_type": "hit"}]}
        payload = {"contract": {}, "normalizer": {}}
        scores = [[0.0, 0.0] for _ in range(8)]
        scores[2] = [0.4, 0.0]
        config = {"predict": {"batch_size": 8, "thresholds": {"hit": 0.4, "bounce": 0.4}, "nms_radius": 5}}
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(evaluate, "load_manifest", return_value=manifest), \
             patch.object(evaluate, "load_base", return_value=base) as read, \
             patch.object(evaluate, "load_expert", return_value=(object(), payload)), \
             patch.object(evaluate, "normalize", return_value=torch.zeros(8, 10)), \
             patch.object(evaluate, "dense_scores", return_value=scores):
            result = evaluate.evaluate(Path(directory), config, expert="trajectory", device="cpu")
        self.assertEqual([call.args[1] for call in read.call_args_list], ["01"])
        self.assertEqual(result["metrics"]["frames"]["5"]["overall"]["f1"], 1.0)
        self.assertFalse(result["threshold_search"])
        self.assertEqual(result["cache"]["raw_video_reads"], 0)


if __name__ == "__main__":
    unittest.main()
