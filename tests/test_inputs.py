import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from inputs import load_track, read_video_info, trajectory_rows


class _Capture:
    def __init__(self, _path):
        self.values = {1: 3, 3: 1280, 4: 720, 5: 25.0}

    def isOpened(self):
        return True

    def get(self, key):
        return self.values[key]

    def release(self):
        pass


class InputsTest(unittest.TestCase):
    def _csv(self, contents):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "track.csv"
        path.write_text(contents, encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return path

    @patch("inputs.subprocess.run")
    def test_video_info_sorts_irregular_packet_pts(self, run):
        run.return_value = subprocess.CompletedProcess(
            [], 0, json.dumps({"streams": [{"time_base": "1/1000"}], "packets": [{"pts": 80}, {"pts": 0}, {"pts": 35}]})
        )
        fake_cv2 = type("CV2", (), {
            "VideoCapture": _Capture, "CAP_PROP_FRAME_COUNT": 1, "CAP_PROP_FRAME_WIDTH": 3,
            "CAP_PROP_FRAME_HEIGHT": 4, "CAP_PROP_FPS": 5,
        })
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            info = read_video_info("clip.MOV")
        np.testing.assert_array_equal(info["frame_times"], np.array([0.0, 0.035, 0.08]))
        self.assertEqual((info["width"], info["height"], info["fps"]), (1280, 720, 25.0))

    def test_sparse_six_column_csv_is_aligned(self):
        path = self._csv(
            "frame_number,detected,x_orig,y_orig,width,height\n"
            "0,1,10,20,100,50\n2,0,,,100,50\n4,1,99,49,100,50\n"
        )
        track = load_track(path, 5, 100, 50)
        np.testing.assert_array_equal(track["detected"], [True, False, False, False, True])
        np.testing.assert_array_equal(track["x"], [10, 0, 0, 0, 99])
        self.assertEqual(track["x"].dtype, np.float32)

    def test_track_rejects_frame_and_coordinate_boundaries(self):
        for row in ("3,1,1,1,100,50", "1,1,100,1,100,50"):
            path = self._csv("frame_number,detected,x_orig,y_orig,width,height\n" + row + "\n")
            with self.assertRaises(ValueError):
                load_track(path, 3, 100, 50)

    def test_trajectory_uses_pts_resets_long_gap_and_has_no_stationary_noise(self):
        track = {
            "detected": np.ones(5, dtype=bool),
            "x": np.array([10, 20, 20, 50, 60], dtype=np.float32),
            "y": np.zeros(5, dtype=np.float32),
        }
        rows = trajectory_rows(track, np.array([0.0, 0.1, 0.15, 0.5, 0.6]), 100, 100)
        self.assertAlmostEqual(float(rows[1, 3]), 1.0)
        self.assertEqual(float(rows[2, 5]), 0.0)
        self.assertTrue(np.all(rows[2, 7:9] == 0))
        self.assertTrue(np.all(rows[3, 3:9] == 0))
        self.assertAlmostEqual(float(rows[4, 3]), 1.0, places=6)
        self.assertEqual(float(rows[4, 6]), 0.0)

    def test_rejects_visualized_and_non_raw_paths(self):
        for path in ("visualization/clip.mp4", "clip_visualized.MOV", "clip.avi"):
            with self.assertRaises(ValueError):
                read_video_info(path)


if __name__ == "__main__":
    unittest.main()
