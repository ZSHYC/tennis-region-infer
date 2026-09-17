"""从发布数据生成训练和评估缓存。"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from dataset import SamplePaths, discover_samples, load_config, load_events
from dino import DinoV3
from inputs import load_track, read_video_info, trajectory_rows
from vision import extract_visual


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _tree_signature(path: Path) -> dict:
    if path.is_file():
        return {path.name: _file_signature(path)}
    return {
        str(file.relative_to(path)): _file_signature(file)
        for file in sorted(path.rglob("*.json"))
    }


def _base_signature(sample: SamplePaths) -> dict:
    return {
        "version": 1,
        "video": _file_signature(sample.video),
        "trajectory": _file_signature(sample.trajectory),
        "annotations": _tree_signature(sample.annotations),
        "trajectory_contract": "velocity_midpoint/consistent_float32/10d",
    }


def _visual_signature(sample: SamplePaths, weights: Path) -> dict:
    return {
        "version": 1,
        "video": _file_signature(sample.video),
        "weights": _file_signature(weights),
        "views": "global+four_55_percent_corners/256x256/float16",
    }


def _cache_hit(path: Path, signature: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (OSError, RuntimeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("signature") == signature and "data" in payload


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary.close()
    try:
        torch.save(payload, temporary.name)
        os.replace(temporary.name, path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    )
    try:
        json.dump(payload, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
        temporary.close()
        os.replace(temporary.name, path)
    finally:
        temporary.close()
        Path(temporary.name).unlink(missing_ok=True)


def _build_base(sample: SamplePaths) -> tuple[dict, dict]:
    started = time.perf_counter()
    info = read_video_info(sample.video)
    frame_count = len(info["frame_times"])
    track = load_track(sample.trajectory, frame_count, info["width"], info["height"])
    rows = trajectory_rows(track, info["frame_times"], info["width"], info["height"])
    events = load_events(sample.annotations, sample.sample_id)
    if events and events[-1].frame_number >= frame_count:
        raise ValueError(
            f"GT 帧号超出视频: {sample.sample_id}: {events[-1].frame_number} >= {frame_count}"
        )
    data = {
        "frame_times": torch.from_numpy(info["frame_times"]).to(torch.float64),
        "trajectory_base": torch.from_numpy(rows).to(torch.float32),
        "events": [
            {"frame_number": event.frame_number, "event_type": event.event_type}
            for event in events
        ],
        "metadata": {
            "frame_count": frame_count,
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "source": sample.source,
        },
    }
    return data, {"seconds": time.perf_counter() - started, "frames": frame_count}


def _load_dino(weights: Path, device: torch.device) -> DinoV3:
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 权重不存在: {weights}")
    model = DinoV3()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    return model.eval().requires_grad_(False).to(device)


def prepare(args: argparse.Namespace) -> dict:
    overall_started = time.perf_counter()
    config = load_config(args.config)
    scan_started = time.perf_counter()
    train = discover_samples(args.data_root, config["data"]["train_sources"])
    evaluation = discover_samples(args.data_root, config["data"]["eval_sources"])
    scan_seconds = time.perf_counter() - scan_started
    overlap = train.keys() & evaluation.keys()
    if overlap:
        raise ValueError(f"训练和评估 sample_id 重复: {sorted(overlap)[:3]}")
    all_samples = train | evaluation
    manifest = {
        "train": sorted(train),
        "eval": sorted(evaluation),
        "sources": {sample_id: sample.source for sample_id, sample in all_samples.items()},
    }
    cache_root = Path(args.cache_root)
    write_started = time.perf_counter()
    _atomic_json(cache_root / "manifest.json", manifest)
    manifest_write_seconds = time.perf_counter() - write_started

    selected = train if args.split == "train" else evaluation if args.split == "eval" else all_samples
    if args.sample_id:
        if args.sample_id not in selected:
            raise ValueError(f"sample-id 不在 {args.split} 范围内: {args.sample_id}")
        selected = {args.sample_id: selected[args.sample_id]}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("请求 CUDA，但当前环境不可用")
        torch.cuda.reset_peak_memory_stats(device)
    weights = Path(args.model_dir) / "dinov3_vitb16.pth"
    dino = None
    stats = {
        "samples": len(selected), "base_hits": 0, "base_misses": 0,
        "visual_hits": 0, "visual_misses": 0, "frames": 0,
        "base_seconds": 0.0, "visual_seconds": 0.0,
        "scan_seconds": scan_seconds, "model_load_seconds": 0.0,
        "cache_write_seconds": manifest_write_seconds,
    }
    started = time.perf_counter()
    for index, (sample_id, sample) in enumerate(selected.items(), 1):
        base_path = cache_root / "base" / f"{sample_id}.pt"
        base_signature = _base_signature(sample)
        if _cache_hit(base_path, base_signature):
            stats["base_hits"] += 1
            base = torch.load(base_path, map_location="cpu", weights_only=True, mmap=True)["data"]
        else:
            stats["base_misses"] += 1
            base, timing = _build_base(sample)
            stats["base_seconds"] += timing["seconds"]
            write_started = time.perf_counter()
            _atomic_torch_save(base_path, {"signature": base_signature, "data": base})
            stats["cache_write_seconds"] += time.perf_counter() - write_started
        frames = int(base["metadata"]["frame_count"])
        stats["frames"] += frames

        if not args.base_only:
            visual_path = cache_root / "visual" / f"{sample_id}.pt"
            visual_signature = _visual_signature(sample, weights)
            if _cache_hit(visual_path, visual_signature):
                stats["visual_hits"] += 1
            else:
                stats["visual_misses"] += 1
                if dino is None:
                    model_started = time.perf_counter()
                    dino = _load_dino(weights, device)
                    stats["model_load_seconds"] += time.perf_counter() - model_started
                visual_started = time.perf_counter()
                info = {
                    "frame_times": base["frame_times"].numpy(),
                    "width": base["metadata"]["width"],
                    "height": base["metadata"]["height"],
                }
                visual, _ = extract_visual(sample.video, info, dino, device, args.batch_size)
                stats["visual_seconds"] += time.perf_counter() - visual_started
                write_started = time.perf_counter()
                _atomic_torch_save(visual_path, {"signature": visual_signature, "data": visual})
                stats["cache_write_seconds"] += time.perf_counter() - write_started
        print(f"[{index}/{len(selected)}] {sample_id}")

    elapsed = time.perf_counter() - started
    stats["processing_seconds"] = elapsed
    stats["wall_seconds"] = time.perf_counter() - overall_started
    stats["frames_per_second"] = stats["frames"] / elapsed if elapsed else 0.0
    stats["cuda_peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备轨迹和五视图 DINOv3 缓存")
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dir", type=Path, default=here / "models")
    parser.add_argument("--base-only", action="store_true", help="只准备轨迹专家缓存")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size 必须为正数")
    return args


if __name__ == "__main__":
    prepare(parse_args())
