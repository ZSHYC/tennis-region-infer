"""训练、评估所需的数据发现和缓存读取。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Event:
    sample_id: str
    frame_number: int
    event_type: str
    score: float | None = None


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    source: str
    video: Path
    trajectory: Path
    annotations: Path


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 config.yaml 需要安装 PyYAML") from exc
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("配置必须包含 data")
    for key in ("train_sources", "eval_sources"):
        if not isinstance(config["data"].get(key), list) or not config["data"][key]:
            raise ValueError(f"data.{key} 必须是非空列表")
        if not all(isinstance(source, str) for source in config["data"][key]):
            raise ValueError(f"data.{key} 中的数据源必须加引号并写成字符串")
    return config


def load_manifest(cache_root: str | Path) -> dict:
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缓存清单不存在，请先运行 prepare.py: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train", "eval", "sources"):
        if key not in manifest:
            raise ValueError(f"缓存清单缺少 {key}: {path}")
    return manifest


def _load_cache(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缓存不存在，请先运行 prepare.py: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, dict) or "data" not in value:
        raise ValueError(f"缓存格式无效: {path}")
    return value


def load_base(cache_root: str | Path, sample_id: str) -> dict:
    data = _load_cache(Path(cache_root) / "base" / f"{sample_id}.pt")["data"]
    required = {"frame_times", "trajectory_base", "events", "metadata"}
    if not isinstance(data, dict) or not required <= data.keys():
        raise ValueError(f"base 缓存字段不完整: {sample_id}")
    times, trajectory = data["frame_times"], data["trajectory_base"]
    if (
        not isinstance(times, torch.Tensor) or times.dtype != torch.float64 or times.ndim != 1
        or not isinstance(trajectory, torch.Tensor) or trajectory.dtype != torch.float32
        or trajectory.ndim != 2 or trajectory.shape[1] != 10 or len(trajectory) != len(times)
        or (len(times) > 1 and not bool(torch.all(times[1:] > times[:-1])))
    ):
        raise ValueError(f"base 缓存张量合同无效: {sample_id}")
    return data


def load_visual(cache_root: str | Path, sample_id: str) -> torch.Tensor:
    root = Path(cache_root)
    visual_cache = _load_cache(root / "visual" / f"{sample_id}.pt")
    base_cache = _load_cache(root / "base" / f"{sample_id}.pt")
    visual_video = visual_cache.get("signature", {}).get("video")
    base_video = base_cache.get("signature", {}).get("video")
    if not visual_video or visual_video != base_video:
        raise ValueError(f"visual 与 base 的视频签名不一致，请重建视觉缓存: {sample_id}")
    visual = visual_cache["data"]
    frame_times = base_cache["data"].get("frame_times") if isinstance(base_cache["data"], dict) else None
    if (
        not isinstance(visual, torch.Tensor) or visual.dtype != torch.float16
        or visual.ndim != 2 or visual.shape[1] != 3840
        or not isinstance(frame_times, torch.Tensor) or len(visual) != len(frame_times)
    ):
        raise ValueError(f"visual 缓存必须是 [N,3840]: {sample_id}")
    return visual


_LABELS = {
    "near_hit": "hit", "far_hit": "hit", "hit": "hit",
    "near_bounce": "bounce", "far_bounce": "bounce", "bounce": "bounce",
    "near_net": "net", "far_net": "net", "net": "net",
}


def load_events(path: str | Path, sample_id: str) -> list[Event]:
    source = Path(path)
    rows: list[tuple[int, str, str]] = []
    if source.is_file():
        with source.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"frame_number", "label"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"标注 CSV 缺少字段 {sorted(missing)}: {source}")
            for line, row in enumerate(reader, 2):
                rows.append((int(row["frame_number"]), str(row["label"]), f"{source}:{line}"))
    elif source.is_dir():
        for file in sorted(source.glob("*.json")):
            try:
                frame = int(file.stem.split("_", 1)[0])
            except ValueError as exc:
                raise ValueError(f"无法从标注文件名解析帧号: {file}") from exc
            payload = json.loads(file.read_text(encoding="utf-8"))
            rows.extend((frame, str(shape.get("label", "")), str(file)) for shape in payload.get("shapes", []))
    else:
        raise FileNotFoundError(f"标注不存在: {source}")

    events: list[Event] = []
    seen: set[tuple[int, str]] = set()
    frame_types: dict[int, set[str]] = {}
    for frame, label, origin in rows:
        if frame < 0 or label not in _LABELS:
            raise ValueError(f"事件帧号或标签非法: {frame}, {label}: {origin}")
        event_type = _LABELS[label]
        key = (frame, event_type)
        if key in seen:
            raise ValueError(f"重复标准事件标签: {sample_id}:{frame}:{event_type}")
        seen.add(key)
        frame_types.setdefault(frame, set()).add(event_type)
        events.append(Event(sample_id, frame, event_type))
    conflicts = [frame for frame, types in frame_types.items() if {"hit", "bounce"} <= types]
    if conflicts:
        raise ValueError(f"同帧 hit/bounce 冲突: {sample_id}:{conflicts[0]}")
    return sorted(events, key=lambda event: (event.frame_number, event.event_type))


def discover_samples(data_root: str | Path, sources: list[str]) -> dict[str, SamplePaths]:
    root = Path(data_root)
    samples: dict[str, SamplePaths] = {}
    for source in sources:
        source_root = root / source
        videos = {
            path.stem: path for path in sorted((source_root / "video").iterdir())
            if path.is_file() and path.suffix.lower() in {".mp4", ".mov"}
            and "_visualized" not in path.stem.lower()
        }
        if not videos:
            raise ValueError(f"数据源没有原视频: {source_root / 'video'}")
        tracks: dict[str, Path] = {}
        for path in sorted((source_root / "tracknetv5").rglob("*.csv")):
            sample_id = path.stem.removesuffix("_data")
            if sample_id in videos:
                if sample_id in tracks:
                    raise ValueError(f"轨迹重复: {sample_id}: {tracks[sample_id]} 与 {path}")
                tracks[sample_id] = path
        annotation_candidates = {
            path.stem if path.is_file() else path.name: path
            for path in sorted((source_root / "GT").iterdir())
            if (path.is_file() and path.suffix.lower() == ".csv") or path.is_dir()
        }
        annotations: dict[str, Path] = {}
        for sample_id in videos:
            matches = [
                path for name, path in annotation_candidates.items()
                if name == sample_id or sample_id.endswith(name)
            ]
            if len(matches) == 1:
                annotations[sample_id] = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"GT 匹配不唯一: {sample_id}: {matches}")
        missing_tracks = sorted(videos.keys() - tracks.keys())
        missing_gt = sorted(videos.keys() - annotations.keys())
        if missing_tracks or missing_gt:
            raise ValueError(f"{source} 缺少配套文件: track={missing_tracks[:3]}, GT={missing_gt[:3]}")
        for sample_id, video in videos.items():
            if sample_id in samples:
                raise ValueError(f"跨数据源 sample_id 重复: {sample_id}")
            samples[sample_id] = SamplePaths(
                sample_id, source, video, tracks[sample_id], annotations[sample_id]
            )
    return dict(sorted(samples.items()))
