import json
from pathlib import Path

import pytest
import torch

from dataset import discover_samples, load_base, load_config, load_events, load_manifest, load_visual


def _sample(root: Path, source: str, sample_id: str, nested_track: bool = False) -> None:
    base = root / source
    (base / "video").mkdir(parents=True, exist_ok=True)
    (base / "tracknetv5").mkdir(parents=True, exist_ok=True)
    (base / "GT").mkdir(parents=True, exist_ok=True)
    (base / "video" / f"{sample_id}.MP4").touch()
    track = base / "tracknetv5" / sample_id / f"{sample_id}_data.csv" if nested_track else base / "tracknetv5" / f"{sample_id}.csv"
    track.parent.mkdir(parents=True, exist_ok=True)
    track.touch()
    (base / "GT" / f"{sample_id}.csv").write_text(
        "frame_number,label\n1,near_hit\n3,far_bounce\n", encoding="utf-8"
    )


def test_discovery_only_uses_named_sources_and_handles_both_track_layouts(tmp_path: Path):
    _sample(tmp_path, "train", "train_1")
    _sample(tmp_path, "eval", "eval_1", nested_track=True)
    _sample(tmp_path, "excluded", "excluded_1")

    train = discover_samples(tmp_path, ["train"])
    evaluation = discover_samples(tmp_path, ["eval"])

    assert list(train) == ["train_1"]
    assert list(evaluation) == ["eval_1"]
    assert train["train_1"].source == "train"
    assert evaluation["eval_1"].trajectory.name == "eval_1_data.csv"

    gt = tmp_path / "eval" / "GT" / "eval_1.csv"
    nested_gt = tmp_path / "eval" / "GT" / "1"
    nested_gt.mkdir()
    gt.unlink()
    assert discover_samples(tmp_path, ["eval"])["eval_1"].annotations == nested_gt


def test_event_parsing_and_cache_public_contract(tmp_path: Path):
    _sample(tmp_path, "source", "sample")
    events = load_events(tmp_path / "source" / "GT" / "sample.csv", "sample")
    assert [(event.frame_number, event.event_type) for event in events] == [(1, "hit"), (3, "bounce")]

    cache = tmp_path / "cache"
    (cache / "base").mkdir(parents=True)
    (cache / "visual").mkdir()
    base = {
        "frame_times": torch.tensor([0.0, 0.04], dtype=torch.float64),
        "trajectory_base": torch.zeros(2, 10, dtype=torch.float32),
        "events": [{"frame_number": 1, "event_type": "hit"}],
        "metadata": {"frame_count": 2, "width": 1280, "height": 720},
    }
    signature = {"video": {"size": 10, "mtime_ns": 20}}
    torch.save({"signature": signature, "data": base}, cache / "base" / "sample.pt")
    torch.save(
        {"signature": signature, "data": torch.zeros(2, 3840, dtype=torch.float16)},
        cache / "visual" / "sample.pt",
    )
    (cache / "manifest.json").write_text(
        json.dumps({"train": ["sample"], "eval": [], "sources": {"sample": "source"}}),
        encoding="utf-8",
    )

    assert load_base(cache, "sample")["trajectory_base"].dtype == torch.float32
    assert load_visual(cache, "sample").dtype == torch.float16
    assert load_manifest(cache)["sources"] == {"sample": "source"}

    torch.save(
        {"signature": {"video": {"size": 11, "mtime_ns": 20}},
         "data": torch.zeros(2, 3840, dtype=torch.float16)},
        cache / "visual" / "sample.pt",
    )
    with pytest.raises(ValueError, match="视频签名不一致"):
        load_visual(cache, "sample")


def test_base_rejects_invalid_tensor_contract(tmp_path: Path):
    (tmp_path / "base").mkdir()
    broken = {
        "frame_times": torch.tensor([0.04, 0.0], dtype=torch.float64),
        "trajectory_base": torch.zeros(2, 10, dtype=torch.float32),
        "events": [], "metadata": {"frame_count": 2, "width": 1, "height": 1},
    }
    torch.save({"signature": {"video": {"size": 1, "mtime_ns": 1}}, "data": broken},
               tmp_path / "base" / "broken.pt")
    with pytest.raises(ValueError, match="张量合同无效"):
        load_base(tmp_path, "broken")


def test_config_requires_explicit_train_and_eval_sources(tmp_path: Path):
    pytest.importorskip("yaml")
    config = tmp_path / "config.yaml"
    config.write_text("data:\n  train_sources: [train]\n  eval_sources: [eval]\n", encoding="utf-8")
    assert load_config(config)["data"]["eval_sources"] == ["eval"]
