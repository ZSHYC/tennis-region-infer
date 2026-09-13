import unittest
from pathlib import Path

import torch

from model import EventModel, VisualTemporalModel


class ModelTest(unittest.TestCase):
    def test_trajectory_shapes_and_fully_missing_window(self):
        model = EventModel().eval()
        trajectory = torch.zeros(2, 9, 11)
        output = model(trajectory)
        self.assertEqual(output["eventness_logit"].shape, (2,))
        self.assertEqual(output["type_logits"].shape, (2, 2))
        self.assertTrue(all(torch.isfinite(value).all() for value in output.values()))
        self.assertEqual(model.proj.weight.shape, (64, 779))

    def test_visual_shapes_half_input_and_region_temporal_convolution(self):
        model = VisualTemporalModel().eval()
        seen = []
        hook = model.convs[0].register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].shape)
        )
        output = model(torch.zeros(2, 7, 3840, dtype=torch.float16))
        hook.remove()
        self.assertEqual(seen, [torch.Size((10, 64, 7))])
        self.assertEqual(output["eventness_logit"].shape, (2,))
        self.assertEqual(output["type_logits"].shape, (2, 2))
        self.assertTrue(all(value.dtype == torch.float32 for value in output.values()))
        self.assertNotIn("tile_positions", model.state_dict())

    def test_input_contracts(self):
        with self.assertRaisesRegex(ValueError, r"\[B,T,11\]"):
            EventModel()(torch.zeros(1, 3, 12))
        with self.assertRaisesRegex(ValueError, r"\[B,T,3840\]"):
            VisualTemporalModel()(torch.zeros(1, 3, 768))

    def test_published_weights_load_strictly_when_installed(self):
        model_dir = Path(__file__).parents[1] / "models"
        pairs = (
            (model_dir / "default-b0.pt", EventModel),
            (model_dir / "default-region-visual.pt", VisualTemporalModel),
        )
        if not all(path.exists() for path, _ in pairs):
            self.skipTest("发布权重尚未安装")
        for path, model_type in pairs:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model_type().load_state_dict(payload["model_state"], strict=True)


if __name__ == "__main__":
    unittest.main()
