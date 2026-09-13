"""视频＋TrackNet CSV → 默认双专家的逐帧分数与 hit/bounce 事件。"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dino import DinoV3
from inputs import load_track, read_video_info, trajectory_rows
from model import TrajectoryExpert, VisualExpert
from vision import extract_visual

MODEL_DIR = Path(__file__).resolve().parent / "models"
EVENT_TYPES = ("hit", "bounce")


@contextmanager
def fp32_inference(device: torch.device):
    cuda = device.type == "cuda"
    if cuda:
        matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn


def normalize(rows: np.ndarray, normalizer: dict) -> torch.Tensor:
    indices = [0, 1, 3, 4, 5, 6, 7, 8]
    center = np.asarray(normalizer["center"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    active = (rows[:, 2] > 0.5) & (rows[:, 9] > 0.5)
    output = rows.copy()
    output[:, indices] = 0
    output[np.ix_(active, indices)] = np.clip(
        (rows[np.ix_(active, indices)] - center) / scale,
        -normalizer["clip"], normalizer["clip"],
    )
    return torch.from_numpy(output)


def window_indices(times: torch.Tensor, centers: torch.Tensor,
                   offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    targets = times[centers, None] + offsets[None, :]
    right = torch.searchsorted(times, targets).clamp(max=len(times) - 1)
    left = (right - 1).clamp(min=0)
    indices = torch.where((times[right] - targets).abs() < (times[left] - targets).abs(), right, left)
    tolerance = torch.finfo(times.dtype).eps * max(1.0, float(times.abs().max()), float(offsets.abs().max())) * 4
    valid = (targets >= times[0] - tolerance) & (targets <= times[-1] + tolerance)
    return indices, valid


def dense_scores(model: torch.nn.Module, rows: torch.Tensor, times: torch.Tensor,
                 contract: dict, device: torch.device, batch_size: int, *, trajectory: bool) -> list:
    offsets = torch.linspace(-contract["window_span_seconds"] / 2,
                             contract["window_span_seconds"] / 2,
                             2 * contract["window_radius"] + 1, dtype=torch.float64)
    scores = []
    with fp32_inference(device):
        for start in range(0, len(times), batch_size):
            centers = torch.arange(start, min(start + batch_size, len(times)))
            indices, valid = window_indices(times, centers, offsets)
            batch = rows[indices] * valid[..., None]
            if trajectory:
                relative = ((times[indices] - times[centers, None]) * valid).float()
                batch = torch.cat((batch, relative[..., None]), dim=-1)
            output = model(batch.to(device, non_blocking=True))
            values = output["eventness_logit"].sigmoid()[:, None] * output["type_logits"].softmax(dim=-1)
            scores.extend(values.float().cpu().tolist())
    return scores


def decode_events(scores: list, times: np.ndarray, track: dict) -> list[dict]:
    events = []
    for column, event_type in enumerate(EVENT_TYPES):
        kept = []
        for frame in sorted(range(len(scores)), key=lambda i: (-scores[i][column], i)):
            score = scores[frame][column]
            if score < 0.4 or any(abs(frame - previous) <= 5 for previous in kept):
                continue
            kept.append(frame)
            detected = bool(track["detected"][frame])
            events.append({
                "frame_number": frame, "timestamp_seconds": float(times[frame]),
                "event_type": event_type, "score": score,
                "x": float(track["x"][frame]) if detected else None,
                "y": float(track["y"][frame]) if detected else None,
            })
    return sorted(events, key=lambda event: (event["frame_number"], event["event_type"]))


def load_experts(model_dir: Path, device: torch.device) -> tuple:
    track = torch.load(model_dir / "trajectory_expert.pt", map_location="cpu", weights_only=True)
    visual = torch.load(model_dir / "visual_expert.pt", map_location="cpu", weights_only=True)
    for payload, kind, radius, span in (
        (track, "trajectory", 12, 0.4), (visual, "visual_tile_region_temporal", 24, 1.6),
    ):
        contract = payload["contract"]
        expected = {"model_kind": kind, "window_radius": radius, "window_span_seconds": span,
                    "event_types": list(EVENT_TYPES), "nms_radius": 5, "score_mode": "product"}
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError("权重不属于当前默认双专家推理合同")
    if track["contract"]["base_spec"].get("coordinate_precision") != "consistent_float32":
        raise ValueError("轨迹专家权重必须使用 consistent_float32 轨迹")
    track_model, visual_model = TrajectoryExpert(), VisualExpert()
    track_model.load_state_dict(track["model_state"], strict=True)
    visual_model.load_state_dict(visual["model_state"], strict=True)
    return (track_model.eval().requires_grad_(False).to(device),
            visual_model.eval().requires_grad_(False).to(device), track, visual)


def predict(video: Path, trajectory: Path, *, model_dir: Path = MODEL_DIR,
            device: str = "auto", batch_size: int = 512, dino_batch_size: int = 64) -> dict:
    if batch_size <= 0 or dino_batch_size <= 0:
        raise ValueError("batch size 必须为正数")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type not in {"cpu", "cuda"} or device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("只支持可用的 cpu/cuda 设备")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    info = read_video_info(video)
    track = load_track(trajectory, len(info["frame_times"]), info["width"], info["height"])
    base = trajectory_rows(track, info["frame_times"], info["width"], info["height"])
    input_seconds = perf_counter() - started
    stage = perf_counter()
    track_model, visual_model, track_checkpoint, visual_checkpoint = load_experts(model_dir, device)
    dino = DinoV3()
    dino.load_state_dict(torch.load(model_dir / "dinov3_vitb16.pth", map_location="cpu", weights_only=True), strict=True)
    dino.eval().requires_grad_(False).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_seconds = perf_counter() - stage
    stage = perf_counter()
    visual, extraction = extract_visual(video, info, dino, device, dino_batch_size)
    visual_seconds = perf_counter() - stage
    del dino
    stage = perf_counter()
    times = torch.from_numpy(info["frame_times"])
    track_scores = dense_scores(track_model, normalize(base, track_checkpoint["normalizer"]), times,
                                track_checkpoint["contract"], device, batch_size, trajectory=True)
    track_seconds = perf_counter() - stage
    stage = perf_counter()
    visual_scores = dense_scores(visual_model, visual, times, visual_checkpoint["contract"],
                                 device, batch_size, trajectory=False)
    visual_head_seconds = perf_counter() - stage
    stage = perf_counter()
    # 各路 float32 分数转 Python float 后等权相加。
    scores = [[0.5 * a + 0.5 * b for a, b in zip(left, right, strict=True)]
              for left, right in zip(track_scores, visual_scores, strict=True)]
    events = decode_events(scores, info["frame_times"], track)
    postprocess_seconds = perf_counter() - stage
    elapsed = perf_counter() - started
    return {
        "sample_id": Path(video).stem, "vision_mode": "fusion",
        "model": "trajectory-visual-fusion-v1",
        "video": str(Path(video).resolve()), "trajectory": str(Path(trajectory).resolve()),
        "width": info["width"], "height": info["height"], "fps": info["fps"],
        "scores": [{"frame_number": i, "hit_score": score[0], "bounce_score": score[1]}
                   for i, score in enumerate(scores)],
        "events": events,
        "fusion": {"weights": {"track": 0.5, "visual": 0.5},
                   "thresholds": {"hit": 0.4, "bounce": 0.4}, "nms_radius": 5},
        "cache": {"enabled": False, "hits": 0, "misses": 0},
        "timing": {"input_seconds": input_seconds, "model_load_seconds": model_seconds,
                   "decode_and_dino_seconds": visual_seconds, "track_seconds": track_seconds,
                   "visual_head_seconds": visual_head_seconds, "postprocess_seconds": postprocess_seconds,
                   "total_seconds": elapsed, "dense_frames": len(scores),
                   "frames_per_second": len(scores) / elapsed, "device": str(device),
                   "batch_size": batch_size, "dino_batch_size": dino_batch_size,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   **extraction},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="原始 mp4/mov 视频")
    parser.add_argument("--trajectory", type=Path, required=True, help="TrackNet 逐帧 CSV，可含缺失帧")
    parser.add_argument("--output", type=Path, required=True, help="分数、事件和耗时 JSON")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="三份本地权重所在目录")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--batch-size", type=int, default=512, help="事件头的中心帧 batch size")
    parser.add_argument("--dino-batch-size", type=int, default=64, help="DINO 每批图像数，全图/裁剪分别处理")
    args = parser.parse_args()
    payload = predict(args.video, args.trajectory, model_dir=args.model_dir, device=args.device,
                      batch_size=args.batch_size, dino_batch_size=args.dino_batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = perf_counter()
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({**payload["timing"], "output_write_seconds": perf_counter() - stage,
                      "events": len(payload["events"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
