"""当前 B0 / sigma18 架构的训练核心；只消费 prepare 生成的 cache。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from inputs import trajectory_rows
from model import TrajectoryExpert, VisualExpert
from predict import window_indices


FEATURE_NAMES = ("x_norm", "y_norm", "detected", "dx", "dy", "speed",
                 "acceleration", "angle_change", "curvature", "valid")
NORMALIZED_NAMES = ("x_norm", "y_norm", "dx", "dy", "speed", "acceleration",
                    "angle_change", "curvature")
EVENT_TYPES = ("hit", "bounce")
DEFAULT_TRAIN = {
    "seed": 20260710,
    "negative_seed": 20260709,
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "min_learning_rate": 5e-5,
    "negative_ratio": 5,
    "positive_radius": 2,
    "ignore_radius": 2,
    "positive_weight": 2.0,
    "type_weight": 0.5,
    "track_dropout": {"sample_probability": 0.5, "time_fraction": 0.2},
}


@dataclass(frozen=True)
class RobustNormalizer:
    columns: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    clip: float = 5.0

    @classmethod
    def fit(cls, videos: list[np.ndarray], min_scale: float = 1e-6,
            clip: float = 5.0) -> "RobustNormalizer":
        indices = [FEATURE_NAMES.index(name) for name in NORMALIZED_NAMES]
        selected = [rows[(rows[:, 2] > 0.5) & (rows[:, 9] > 0.5)][:, indices]
                    for rows in videos if np.any((rows[:, 2] > 0.5) & (rows[:, 9] > 0.5))]
        if not selected:
            raise ValueError("训练数据没有 detected 轨迹行")
        values = np.concatenate(selected)
        center = np.median(values, axis=0)
        scale = np.maximum(np.percentile(values, 75, axis=0)
                           - np.percentile(values, 25, axis=0), min_scale)
        return cls(NORMALIZED_NAMES, center.astype(np.float32), scale.astype(np.float32), clip)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        source = np.asarray(rows, dtype=np.float32)
        output = source.copy()
        indices = [FEATURE_NAMES.index(name) for name in self.columns]
        active = (source[:, 2] > 0.5) & (source[:, 9] > 0.5)
        output[:, indices] = 0
        output[np.ix_(active, indices)] = np.clip(
            (source[np.ix_(active, indices)] - self.center) / self.scale,
            -self.clip, self.clip,
        )
        return output

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "center": self.center.tolist(),
                "scale": self.scale.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, value: dict) -> "RobustNormalizer":
        return cls(tuple(value["columns"]), np.asarray(value["center"], dtype=np.float32),
                   np.asarray(value["scale"], dtype=np.float32), float(value["clip"]))


@dataclass(frozen=True)
class TargetRow:
    sample_id: str
    center: int
    kind: str
    eventness: float
    event_type: int
    type_mask: float


def sample_training_targets(frame_count: int, events: list[dict], positive_radius: int,
                            ignore_radius: int, negative_ratio: int, *, seed,
                            sigma: float, sample_id: str = "") -> list[TargetRow]:
    primary = [event for event in events if event["event_type"] in EVENT_TYPES]
    if frame_count <= 0 or min(positive_radius, ignore_radius, negative_ratio) < 0 or sigma <= 0:
        raise ValueError("采样参数无效")
    if any(not 0 <= int(event["frame_number"]) < frame_count for event in events):
        raise ValueError("事件帧号超出范围")
    positive = {center for event in primary for center in range(
        max(0, int(event["frame_number"]) - positive_radius),
        min(frame_count, int(event["frame_number"]) + positive_radius + 1))}
    ignored = set(positive)
    for event in events:
        if event["event_type"] in (*EVENT_TYPES, "net"):
            frame = int(event["frame_number"])
            ignored.update(range(max(0, frame - ignore_radius),
                                 min(frame_count, frame + ignore_radius + 1)))
    rows = []
    type_index = {name: index for index, name in enumerate(EVENT_TYPES)}
    for center in sorted(positive):
        distances = [(abs(center - int(event["frame_number"])), event) for event in primary]
        nearest_distance = min(distance for distance, _ in distances)
        nearest_types = {event["event_type"] for distance, event in distances
                         if distance == nearest_distance}
        rows.append(TargetRow(
            sample_id, center, "positive",
            math.exp(-(nearest_distance ** 2) / (2 * sigma ** 2)),
            type_index[sorted(nearest_types)[0]], float(len(nearest_types) == 1),
        ))
    candidates = [center for center in range(frame_count) if center not in ignored]
    rng = seed if isinstance(seed, random.Random) else random.Random(seed)
    for center in rng.sample(candidates, min(len(candidates), negative_ratio * len(positive))):
        rows.append(TargetRow(sample_id, center, "negative", 0.0, 0, 0.0))
    return sorted(rows, key=lambda row: row.center)


def _dataset_api():
    import dataset
    return dataset


def _train_config(config: dict, expert: str, epochs: int | None,
                  batch_size: int | None) -> dict:
    result = {**DEFAULT_TRAIN, **config.get("train", {})}
    if epochs is not None:
        result["epochs"] = epochs
    if batch_size is not None:
        result["batch_size"] = batch_size
    for name in ("epochs", "batch_size", "negative_ratio", "positive_radius", "ignore_radius"):
        result[name] = int(result[name])
    if result["epochs"] <= 0 or result["batch_size"] <= 0:
        raise ValueError("epochs 和 batch_size 必须为正数")
    return result


def _short_gap_durations(base: dict[str, dict]) -> np.ndarray:
    durations = []
    for cache in base.values():
        rows = np.asarray(cache["trajectory_base"])
        times = np.asarray(cache["frame_times"], dtype=np.float64)
        detected = rows[:, 2] > 0.5
        if len(times) < 3:
            continue
        median_dt = float(np.median(np.diff(times)))
        index = 0
        while index < len(detected):
            if detected[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(detected) and not detected[index + 1]:
                index += 1
            if start > 0 and index + 1 < len(detected):
                duration = float(times[index] - times[start] + median_dt)
                if duration <= 0.2:
                    durations.append(duration)
            index += 1
    return np.asarray(durations, dtype=np.float64)


def _drop_track(rows: np.ndarray, times: np.ndarray, durations: np.ndarray,
                fraction: float, generator: np.random.Generator) -> np.ndarray:
    rate = -math.log1p(-fraction) / float(durations.mean())
    scheduled = np.zeros(len(rows), dtype=bool)
    count = int(generator.poisson(rate * float(times[-1] - times[0] + durations.max())))
    if count:
        for start, duration in zip(
            generator.uniform(float(times[0] - durations.max()), float(times[-1]), count),
            generator.choice(durations, count), strict=True,
        ):
            scheduled[np.searchsorted(times, start, side="left"):
                      np.searchsorted(times, start + duration, side="left")] = True
    detected = (rows[:, 2] > 0.5) & ~scheduled
    rebuilt = trajectory_rows({"detected": detected, "x": rows[:, 0], "y": rows[:, 1]},
                              times, 1, 1)
    rebuilt[:, 9] = rows[:, 9]
    return rebuilt


def _contract(expert: str) -> dict:
    if expert == "trajectory":
        return {
            "model_kind": "trajectory", "window_radius": 12,
            "window_span_seconds": 0.4, "event_types": list(EVENT_TYPES),
            "nms_radius": 5, "score_mode": "product",
            "base_spec": {"feature_names": list(FEATURE_NAMES),
                          "max_derivative_gap_seconds": 0.2,
                          "acceleration_mode": "velocity_midpoint",
                          "coordinate_precision": "consistent_float32"},
        }
    return {
        "model_kind": "visual_tile_region_temporal", "window_radius": 24,
        "window_span_seconds": 1.6, "event_types": list(EVENT_TYPES),
        "nms_radius": 5, "score_mode": "product",
    }


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _cache_identity(cache_root: Path, train_ids: list[str], expert: str) -> list[dict]:
    identities = []
    kinds = ("base", "visual") if expert == "visual" else ("base",)
    for sample_id in train_ids:
        for kind in kinds:
            path = Path(cache_root) / kind / f"{sample_id}.pt"
            if path.is_file():
                stat = path.stat()
                identities.append({"sample_id": sample_id, "kind": kind,
                                   "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            else:
                identities.append({"sample_id": sample_id, "kind": kind, "missing": True})
    return identities


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(value: dict) -> None:
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def _batch(base: dict[str, dict], trajectory: dict[str, np.ndarray],
           visual: dict[str, torch.Tensor], targets: list[TargetRow],
           contract: dict, expert: str,
           augmented: dict[str, np.ndarray] | None, augment_rows: torch.Tensor | None) -> dict:
    offsets = torch.linspace(-contract["window_span_seconds"] / 2,
                             contract["window_span_seconds"] / 2,
                             2 * contract["window_radius"] + 1, dtype=torch.float64)
    windows = []
    for position, row in enumerate(targets):
        cache = base[row.sample_id]
        times = torch.as_tensor(cache["frame_times"], dtype=torch.float64)
        indices, valid = window_indices(times, torch.tensor([row.center]), offsets)
        if expert == "trajectory":
            source = (augmented[row.sample_id] if augmented is not None
                      and bool(augment_rows[position]) else trajectory[row.sample_id])
            values = torch.from_numpy(source)[indices[0]] * valid[0, :, None]
            relative = ((times[indices[0]] - times[row.center]) * valid[0]).float()
            values = torch.cat((values, relative[:, None]), dim=-1)
        else:
            values = visual[row.sample_id][indices[0]] * valid[0, :, None]
        windows.append(values)
    return {
        "input": torch.stack(windows),
        "eventness": torch.tensor([row.eventness for row in targets], dtype=torch.float32),
        "type": torch.tensor([row.event_type for row in targets]),
        "type_mask": torch.tensor([row.type_mask for row in targets], dtype=torch.float32),
    }


def train_expert(expert: str, cache_root: Path, output_dir: Path, config: dict,
                 device: torch.device, *, resume: bool = False,
                 epochs_override: int | None = None,
                 batch_size_override: int | None = None) -> dict:
    """训练一个专家；无 validation/best 选择，只导出最后一轮。"""
    if expert not in {"trajectory", "visual"}:
        raise ValueError("expert 只支持 trajectory 或 visual")
    settings = _train_config(config, expert, epochs_override, batch_size_override)
    dataset = _dataset_api()
    manifest = dataset.load_manifest(cache_root)
    train_ids = sorted(manifest["train"])
    eval_ids = set(manifest.get("eval", []))
    sources = manifest.get("sources", {})
    if not train_ids or len(train_ids) != len(set(train_ids)) or set(train_ids) & eval_ids:
        raise ValueError("manifest train 必须非空、唯一且不得包含 eval")
    train_sources = {str(sources.get(sample_id, "")) for sample_id in train_ids}
    forbidden_sources = {"back_match", "back_match_clipped", "2606_admin_back_clipped"}
    if any(source.lower() in forbidden_sources or "back_match" in source.lower()
           for source in train_sources):
        raise ValueError("训练集不得包含 back_match 或 2606_admin_back_clipped 来源")
    configured_sources = set(config.get("data", {}).get("train_sources", []))
    if configured_sources and not train_sources <= configured_sources:
        raise ValueError("manifest train 含 config.data.train_sources 之外的数据源")

    started = perf_counter()
    cache_started = perf_counter()
    base = {sample_id: dataset.load_base(cache_root, sample_id) for sample_id in train_ids}
    visual = ({sample_id: torch.as_tensor(dataset.load_visual(cache_root, sample_id))
               for sample_id in train_ids} if expert == "visual" else {})
    cache_read_seconds = perf_counter() - cache_started
    arrays = [np.asarray(base[sample_id]["trajectory_base"], dtype=np.float32)
              for sample_id in train_ids]
    normalizer = RobustNormalizer.fit(arrays)
    trajectory = {sample_id: normalizer.transform(np.asarray(base[sample_id]["trajectory_base"]))
                  for sample_id in train_ids}
    rng = random.Random(int(settings["negative_seed"]))
    sigma = 1.2 if expert == "trajectory" else 2.0
    targets = [row for sample_id in train_ids for row in sample_training_targets(
        len(base[sample_id]["frame_times"]), base[sample_id]["events"],
        settings["positive_radius"], settings["ignore_radius"], settings["negative_ratio"],
        seed=rng, sigma=sigma, sample_id=sample_id)]
    if not targets:
        raise ValueError("训练集没有 hit/bounce 目标")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / f"{expert}_expert.pt"
    last_path = output_dir / "last.pt"
    if not resume and (final_path.exists() or last_path.exists()):
        raise ValueError("输出已存在；新训练请换目录，继续训练请使用 --resume")
    if resume and not last_path.is_file():
        raise ValueError("--resume 要求 output-dir 中已有 last.pt")

    seed = int(settings["seed"])
    _set_determinism(seed)
    model = (TrajectoryExpert() if expert == "trajectory" else VisualExpert()).to(device)
    # 模型初始化消耗 RNG；训练随机流从固定 seed 重新开始，与原训练实现一致。
    _set_determinism(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, settings["epochs"]), eta_min=float(settings["min_learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    contract = _contract(expert)
    resume_contract = {"expert": expert, "train_sample_ids": train_ids,
                       "settings": settings, "contract": contract,
                       "normalizer": normalizer.to_dict(), "device": str(device),
                       "cache_identity": _cache_identity(cache_root, train_ids, expert)}
    history = []
    start_epoch = 1
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("training_state_version") != 1 or state.get("resume_contract") != resume_contract:
            raise ValueError("恢复合同不一致：配置、训练名单或 cache 合同已改变")
        model.load_state_dict(state["model_state"], strict=True)
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        history = state["history"]
        start_epoch = int(state["completed_epoch"]) + 1
        _restore_rng(state["rng_state"])

    dropout = settings["track_dropout"] if expert == "trajectory" else None
    durations = _short_gap_durations(base) if dropout else np.empty(0)
    if dropout and not len(durations):
        raise ValueError("train 没有可用于缺轨增强的内部短缺口")
    for epoch in range(start_epoch, settings["epochs"] + 1):
        epoch_started = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        augmented = None
        mixture = None
        if dropout:
            generator = np.random.default_rng([seed, epoch, 0])
            augmented = {sample_id: normalizer.transform(_drop_track(
                np.asarray(base[sample_id]["trajectory_base"], dtype=np.float32),
                np.asarray(base[sample_id]["frame_times"], dtype=np.float64), durations,
                float(dropout["time_fraction"]), generator)) for sample_id in sorted(train_ids)}
            mixture = torch.Generator().manual_seed(seed + epoch)
        indices = list(range(len(targets)))
        random.Random(seed + epoch).shuffle(indices)
        model.train()
        total = 0.0
        for offset in range(0, len(indices), settings["batch_size"]):
            selected = [targets[index] for index in indices[offset:offset + settings["batch_size"]]]
            use_augmented = (torch.rand(len(selected), generator=mixture)
                             < float(dropout["sample_probability"])) if dropout else None
            batch = _batch(base, trajectory, visual, selected, contract, expert,
                           augmented, use_augmented)
            values = batch["input"].to(device)
            eventness = batch["eventness"].to(device)
            types = batch["type"].to(device)
            type_mask = batch["type_mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(values)
                event_loss = F.binary_cross_entropy_with_logits(
                    output["eventness_logit"], eventness,
                    pos_weight=torch.tensor(float(settings["positive_weight"]), device=device))
                type_loss = (F.cross_entropy(output["type_logits"], types, reduction="none")
                             * type_mask).sum() / type_mask.sum().clamp_min(1)
                loss = event_loss + float(settings["type_weight"]) * type_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(selected)
        scheduler.step()
        elapsed = perf_counter() - epoch_started
        history.append({"epoch": epoch, "loss": total / len(targets),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "wall_seconds": elapsed,
                        "samples_per_second": len(targets) / elapsed if elapsed else 0.0,
                        "cuda_peak_allocated_bytes": (
                            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                        )})
        _atomic_save({
            "training_state_version": 1, "completed_epoch": epoch,
            "resume_contract": resume_contract, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "rng_state": _rng_state(), "history": history,
        }, last_path)

    completed_epoch = int(torch.load(last_path, map_location="cpu", weights_only=False)["completed_epoch"])
    training = {
        "expert": expert, "completed_epoch": completed_epoch, "selection": "final_epoch",
        "train_sample_ids": train_ids,
        "train_sources": sorted(train_sources),
        "validation_sample_ids": [], "test_sample_ids": [],
        "note": "无 validation/test；与历史发布 B0/sigma18 训练划分不同",
        "settings": settings, "sigma": sigma,
    }
    payload = {"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
               "contract": contract, "normalizer": normalizer.to_dict(), "training": training}
    _atomic_save(payload, final_path)
    return {"expert": expert, "completed_epoch": completed_epoch, "targets": len(targets),
            "output": str(final_path), "last": str(last_path),
            "wall_seconds": perf_counter() - started, "cache_read_seconds": cache_read_seconds,
            "cache_hits": len(train_ids) * (2 if expert == "visual" else 1),
            "cache_misses": 0, "history": history}


def load_trained(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    kind = payload["contract"]["model_kind"]
    model = TrajectoryExpert() if kind == "trajectory" else VisualExpert()
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval(), payload
