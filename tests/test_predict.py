import unittest

import numpy as np
import torch

from predict import decode_events, fp32_inference, normalize, window_indices
from vision import four_tiles, image_batch


class PredictTest(unittest.TestCase):
    def test_nearest_pts_ties_padding_and_actual_offsets(self):
        times = torch.tensor([0.0, 0.125, 0.5], dtype=torch.float64)
        offsets = torch.tensor([-0.25, -0.0625, 0.0, 0.1875, 0.5], dtype=torch.float64)
        indices, valid = window_indices(times, torch.tensor([1]), offsets)
        self.assertEqual(indices.tolist(), [[0, 0, 1, 1, 2]])
        self.assertEqual(valid.tolist(), [[False, True, True, True, False]])
        self.assertEqual(((times[indices] - times[1]) * valid).tolist(), [[0, -0.125, 0, 0, 0]])

    def test_normalization_observation_mask_and_clip(self):
        rows = np.ones((3, 10), dtype=np.float32) * 9
        rows[:, 2] = [1, 0, 1]
        rows[:, 9] = [1, 1, 0]
        result = normalize(rows, {"center": [1] * 8, "scale": [2] * 8, "clip": 3})
        self.assertEqual(result[0, 0].item(), 3)
        self.assertEqual(result[1:, :2].tolist(), [[0, 0], [0, 0]])
        self.assertEqual(result[:, 9].tolist(), [1, 1, 0])
        self.assertTrue(np.array_equal(rows[:, 0], [9, 9, 9]))

    def test_nms_inclusive_threshold_radius_ties_and_missing_coordinates(self):
        scores = [[0.0, 0.0] for _ in range(13)]
        scores[0] = [0.4, 0.8]
        scores[5][0] = 0.4
        scores[6][0] = 0.4
        scores[12][0] = 0.3999
        track = {"detected": np.zeros(13, dtype=bool), "x": np.zeros(13), "y": np.zeros(13)}
        events = decode_events(scores, np.arange(13) * 0.03, track)
        self.assertEqual([(e["frame_number"], e["event_type"]) for e in events],
                         [(0, "bounce"), (0, "hit"), (6, "hit")])
        self.assertTrue(all(e["x"] is None and e["y"] is None for e in events))
        self.assertEqual(decode_events([[0.0, 0.0]], np.array([0.0]), track), [])

    def test_full_frame_and_tile_geometry(self):
        rgb = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape(720, 1280, 3)
        tiles = four_tiles(rgb)
        self.assertTrue(all(tile.shape == (396, 704, 3) for tile in tiles))
        self.assertTrue(np.array_equal(tiles[1][0, 0], rgb[0, 576]))
        batch = image_batch(np.zeros((2, 12, 20, 3), dtype=np.uint8))
        self.assertEqual(batch.shape, (2, 3, 256, 256))
        torch.testing.assert_close(batch[0, :, 0, 0], -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225]))

    def test_inference_context(self):
        with fp32_inference(torch.device("cpu")):
            self.assertTrue(torch.is_inference_mode_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())


if __name__ == "__main__":
    unittest.main()
