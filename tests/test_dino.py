import unittest
from pathlib import Path

import torch

from dino import DinoV3


class DinoV3Test(unittest.TestCase):
    def test_contract_and_state_dict(self):
        model = DinoV3()
        self.assertEqual(len(model.state_dict()), 188)
        self.assertEqual(model.state_dict()["rope_embed.periods"].shape, (16,))
        with self.assertRaisesRegex(ValueError, r"\[N,3,256,256\]"):
            model(torch.zeros(1, 3, 224, 224))

    def test_published_weights_load_strictly_when_installed(self):
        weights = Path(__file__).parents[1] / "models" / "dinov3_vitb16.pth"
        if not weights.exists():
            self.skipTest("DINOv3 权重尚未安装")
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model = DinoV3().eval()
        model.load_state_dict(state, strict=True)
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 256, 256))
        self.assertEqual(output.shape, (1, 768))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
