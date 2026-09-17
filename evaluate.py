"""只读 back_match 缓存，以固定工作点评估单专家或等权融合。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dataset import Event, load_base, load_config, load_manifest, load_visual
from metrics import ScoreRow, evaluate_all, scores_to_events
from model import TrajectoryExpert, VisualExpert
from predict import MODEL_DIR, dense_scores, normalize


def load_expert(path: Path, kind: str, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    contract = payload["contract"]
    expected = "trajectory" if kind == "trajectory" else "visual_tile_region_temporal"
    radius, span = (12, 0.4) if kind == "trajectory" else (24, 1.6)
    if (contract.get("model_kind"), contract.get("window_radius"),
            contract.get("window_span_seconds"), contract.get("event_types"),
            contract.get("score_mode")) != (expected, radius, span, ["hit", "bounce"], "product"):
        raise ValueError(f"权重不符合当前 {kind} 输入合同: {path}")
    model = TrajectoryExpert() if kind == "trajectory" else VisualExpert()
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval().requires_grad_(False).to(device), payload


def evaluate(cache_root: Path, config: dict, *, expert: str = "fusion",
             trajectory_checkpoint: Path = MODEL_DIR / "trajectory_expert.pt",
             visual_checkpoint: Path = MODEL_DIR / "visual_expert.pt",
             device: str = "auto", sample_id: str | None = None) -> dict:
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("请求 CUDA，但当前环境不可用")
        torch.cuda.reset_peak_memory_stats(device)
    manifest = load_manifest(cache_root)
    ids = manifest["eval"]
    if not ids or any(manifest["sources"][s] != "back_match_clipped" for s in ids):
        raise ValueError("评估名单必须非空且只包含 back_match_clipped")
    if set(ids) & set(manifest["train"]):
        raise ValueError("训练和评估名单重叠")
    if sample_id is not None:
        if sample_id not in ids:
            raise ValueError("--sample-id 必须属于 back_match 评估名单")
        ids = [sample_id]
    batch_size = int(config["predict"]["batch_size"])
    thresholds = config["predict"]["thresholds"]
    nms_radius = int(config["predict"]["nms_radius"])
    if batch_size <= 0:
        raise ValueError("predict.batch_size 必须为正数")
    branches = {}
    for kind, path in (("trajectory", trajectory_checkpoint), ("visual", visual_checkpoint)):
        if expert in ("fusion", kind):
            branches[kind] = (*load_expert(path, kind, device), path)
    if not branches:
        raise ValueError("expert 必须为 fusion、trajectory 或 visual")
    setup_seconds = perf_counter() - started
    gt, predictions, times, samples = [], [], {}, {}
    read_seconds = forward_seconds = postprocess_seconds = 0.0
    for sid in ids:
        stage = perf_counter()
        base = load_base(cache_root, sid)
        visual = load_visual(cache_root, sid) if "visual" in branches else None
        frame_times = base["frame_times"]
        if visual is not None and len(visual) != len(frame_times):
            raise ValueError(f"视觉缓存与 PTS 帧数不一致: {sid}")
        read_seconds += perf_counter() - stage
        stage = perf_counter()
        scores = []
        for kind, (model, payload, _path) in branches.items():
            rows = normalize(base["trajectory_base"].numpy(), payload["normalizer"]) if kind == "trajectory" else visual
            scores.append(np.asarray(dense_scores(model, rows, frame_times, payload["contract"],
                                                  device, batch_size, trajectory=kind == "trajectory")))
        values = sum(scores) / len(scores)
        forward_seconds += perf_counter() - stage
        stage = perf_counter()
        rows = [ScoreRow(sid, frame, float(value[0]), float(value[1])) for frame, value in enumerate(values)]
        found = scores_to_events(rows, thresholds, nms_radius)
        truth = [Event(sid, int(event["frame_number"]), event["event_type"]) for event in base["events"]]
        axis = frame_times.numpy()
        samples[sid] = {"metrics": evaluate_all(truth, found, {sid: axis}),
                        "events": [asdict(event) for event in found]}
        gt.extend(truth)
        predictions.extend(found)
        times[sid] = axis
        postprocess_seconds += perf_counter() - stage
    metrics = evaluate_all(gt, predictions, times)
    elapsed = perf_counter() - started
    frames = sum(map(len, times.values()))
    return {
        "expert": expert, "split": "eval", "sources": ["back_match_clipped"],
        "sample_count": len(ids), "thresholds": thresholds, "nms_radius": nms_radius,
        "threshold_search": False, "metrics": metrics, "samples": samples,
        "checkpoints": {kind: str(path.resolve()) for kind, (_model, _payload, path) in branches.items()},
        "training_provenance": {kind: payload.get("training", "历史发布权重；并非按新划分重新训练")
                                for kind, (_model, payload, _path) in branches.items()},
        "cache": {"base_hit": len(ids), "visual_hit": len(ids) if "visual" in branches else 0,
                  "miss": 0, "raw_video_reads": 0},
        "timing": {"setup_seconds": setup_seconds, "cache_read_seconds": read_seconds,
                   "forward_seconds": forward_seconds, "postprocess_seconds": postprocess_seconds,
                   "total_seconds": elapsed, "dense_frames": frames, "frames_per_second": frames / elapsed,
                   "device": str(device), "cuda_peak_allocated_bytes":
                   torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expert", choices=("fusion", "trajectory", "visual"), default="fusion")
    parser.add_argument("--trajectory-checkpoint", type=Path, default=MODEL_DIR / "trajectory_expert.pt")
    parser.add_argument("--visual-checkpoint", type=Path, default=MODEL_DIR / "visual_expert.pt")
    parser.add_argument("--sample-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.cache_root, load_config(args.config), expert=args.expert,
                      trajectory_checkpoint=args.trajectory_checkpoint, visual_checkpoint=args.visual_checkpoint,
                      device=args.device, sample_id=args.sample_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "overall": report["metrics"]["frames"]["5"]["overall"],
                      "timing": report["timing"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
