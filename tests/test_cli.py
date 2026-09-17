"""临时合成缓存验证四个入口；不训练真实数据，不改发布权重。"""

import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from predict import load_experts


def test_synthetic_training_exports_loadable_experts_and_cache_evaluation(tmp_path):
    root = Path(__file__).parents[1]
    cache = tmp_path / "cache"
    for kind in ("base", "visual"):
        (cache / kind).mkdir(parents=True)
    manifest = {"train": ["synthetic_train"], "eval": ["synthetic_eval"],
                "sources": {"synthetic_train": "2606_admin_back", "synthetic_eval": "back_match_clipped"}}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for sid in ("synthetic_train", "synthetic_eval"):
        rows = torch.zeros(13, 10)
        rows[:, :2] = 0.5
        rows[:, 2] = rows[:, 9] = 1
        rows[2, :9] = 0
        signature = {"video": {"size": 123, "mtime_ns": 456}}
        base = {"frame_times": torch.arange(13, dtype=torch.float64) / 25,
                "trajectory_base": rows, "events": [{"frame_number": 6, "event_type": "hit"}],
                "metadata": {"frame_count": 13, "width": 100, "height": 100}}
        torch.save({"signature": signature, "data": base}, cache / "base" / f"{sid}.pt")
        torch.save({"signature": signature, "data": torch.zeros(13, 3840, dtype=torch.float16)},
                   cache / "visual" / f"{sid}.pt")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}

    def run(*args):
        return subprocess.run([sys.executable, *args], cwd=root, env=env, capture_output=True, text=True, check=True)

    for script in ("prepare.py", "train.py", "evaluate.py", "predict.py"):
        run(script, "--help")
    exported = tmp_path / "exported"
    exported.mkdir()
    for expert in ("trajectory", "visual"):
        out = tmp_path / expert
        run("train.py", "--expert", expert, "--cache-root", str(cache), "--output-dir", str(out),
            "--epochs", "1", "--batch-size", "16", "--device", "cpu")
        path = out / f"{expert}_expert.pt"
        payload = torch.load(path, weights_only=True)
        assert payload["training"]["train_sample_ids"] == ["synthetic_train"]
        (exported / path.name).write_bytes(path.read_bytes())
    load_experts(exported, torch.device("cpu"))
    output = tmp_path / "evaluation.json"
    run("evaluate.py", "--cache-root", str(cache), "--device", "cpu", "--output", str(output),
        "--trajectory-checkpoint", str(exported / "trajectory_expert.pt"),
        "--visual-checkpoint", str(exported / "visual_expert.pt"))
    report = json.loads(output.read_text())
    assert list(report["samples"]) == ["synthetic_eval"]
    assert report["threshold_search"] is False
    assert report["cache"]["raw_video_reads"] == 0
