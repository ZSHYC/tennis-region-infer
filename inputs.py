"""从原视频和 TrackNet CSV 构造默认 B0 所需输入。"""

import csv
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def _raw_video(path: str | Path) -> Path:
    video = Path(path)
    if "visualization" in {part.lower() for part in video.parts} or "_visualized" in video.stem.lower():
        raise ValueError(f"可视化视频禁止作为模型输入: {video}")
    if video.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError(f"不支持的视频扩展名: {video.suffix}")
    return video


def read_video_info(path: str | Path) -> dict:
    """读取按 PTS 排序的帧时间和 OpenCV 视频元数据。"""
    import cv2

    video = _raw_video(path)
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base:packet=pts", "-of", "json", str(video),
    ]
    try:
        payload = json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True).stdout
        )
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError(f"ffprobe 无法读取视频 PTS: {video}") from exc
    streams = payload.get("streams", [])
    packets = payload.get("packets", [])
    if len(streams) != 1 or "time_base" not in streams[0]:
        raise ValueError(f"视频必须有且仅有一个可读取 time_base 的视频流: {video}")
    if not packets or any("pts" not in packet for packet in packets):
        raise ValueError(f"视频 packet 缺少 PTS: {video}")
    try:
        pts = np.asarray([int(packet["pts"]) for packet in packets], dtype=np.int64)
        time_base = float(Fraction(streams[0]["time_base"]))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"视频 PTS/time_base 非法: {video}") from exc
    if np.any(pts < 0) or len(np.unique(pts)) != len(pts) or not math.isfinite(time_base) or time_base <= 0:
        raise ValueError(f"视频 PTS 必须非负、唯一且 time_base 有效: {video}")

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV 无法打开视频: {video}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count != len(pts):
        raise ValueError(f"OpenCV 帧数与 packet PTS 数不一致: {frame_count} != {len(pts)}: {video}")
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"视频宽高/FPS 元数据无效: {video}")
    frame_times = np.sort(pts).astype(np.float64) * time_base
    if len(frame_times) > 1 and np.any(np.diff(frame_times) <= 0):
        raise ValueError(f"视频 PTS 时间轴必须严格递增: {video}")
    return {"frame_times": frame_times, "width": width, "height": height, "fps": fps}


def load_track(path: str | Path, frame_count: int, width: int, height: int) -> dict:
    """读取稀疏 TrackNet CSV，并对齐为逐帧轨迹。"""
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("frame_count/width/height 必须为正数")
    detected = np.zeros(frame_count, dtype=np.bool_)
    x = np.zeros(frame_count, dtype=np.float32)
    y = np.zeros(frame_count, dtype=np.float32)
    previous = -1
    row_count = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_number", "detected", "x_orig", "y_orig", "width", "height"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"轨迹 CSV 缺少字段: {missing}: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame_number"])
                detected_raw = int(row["detected"])
                csv_width, csv_height = int(row["width"]), int(row["height"])
                confidence = float(row.get("conf") or 0.0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"轨迹 CSV 数值非法: {path}:{line}") from exc
            if frame < 0 or frame >= frame_count or frame <= previous:
                raise ValueError(f"轨迹帧号越界或未严格递增: {path}:{line}")
            if detected_raw not in {0, 1}:
                raise ValueError(f"detected 必须为 0 或 1: {path}:{line}")
            if (csv_width, csv_height) != (width, height):
                raise ValueError(f"轨迹与视频分辨率不一致: {path}:{line}")
            try:
                x_value = float(row["x_orig"]) if detected_raw else 0.0
                y_value = float(row["y_orig"]) if detected_raw else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"轨迹坐标非法: {path}:{line}") from exc
            if not all(math.isfinite(value) for value in (x_value, y_value, confidence)):
                raise ValueError(f"轨迹数值必须有限: {path}:{line}")
            if detected_raw and not (0 <= x_value < width and 0 <= y_value < height):
                raise ValueError(f"检测坐标超出画面: {path}:{line}")
            detected[frame] = bool(detected_raw)
            x[frame], y[frame] = x_value, y_value
            previous, row_count = frame, row_count + 1
    if not row_count:
        raise ValueError(f"轨迹 CSV 为空: {path}")
    return {"detected": detected, "x": x, "y": y}


def trajectory_rows(track: dict, frame_times: np.ndarray, width: int, height: int) -> np.ndarray:
    """按默认 velocity_midpoint/consistent_float32 合同生成 10 维轨迹行。"""
    times = np.asarray(frame_times, dtype=np.float64)
    detected = np.asarray(track["detected"], dtype=np.bool_)
    x = np.asarray(track["x"])
    y = np.asarray(track["y"])
    count = len(times)
    if not (len(detected) == len(x) == len(y) == count):
        raise ValueError("轨迹数组与 PTS 长度必须一致")
    if width <= 0 or height <= 0 or (count > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("width/height 必须为正且 frame_times 严格递增")
    rows = np.zeros((count, 10), dtype=np.float32)
    rows[:, 9] = 1.0
    previous_index = None
    previous_speed = previous_elapsed = previous_angle = None
    for index in np.flatnonzero(detected):
        x_norm = float(np.float32(float(x[index]) / width))
        y_norm = float(np.float32(float(y[index]) / height))
        rows[index, :3] = (x_norm, y_norm, 1.0)
        if previous_index is not None:
            elapsed = float(times[index] - times[previous_index])
            if elapsed <= 0.2:
                dx = (x_norm - float(rows[previous_index, 0])) / elapsed
                dy = (y_norm - float(rows[previous_index, 1])) / elapsed
                speed = math.hypot(dx, dy)
                derivative_elapsed = elapsed if previous_elapsed is None else 0.5 * (previous_elapsed + elapsed)
                acceleration = 0.0 if previous_speed is None else (speed - previous_speed) / derivative_elapsed
                angle = None if speed == 0 else math.atan2(dy, dx)
                angle_change = 0.0
                if previous_angle is not None and angle is not None:
                    angle_change = abs((angle - previous_angle + math.pi) % (2 * math.pi) - math.pi)
                curvature_speed = speed if previous_speed is None else 0.5 * (previous_speed + speed)
                curvature = math.log1p((angle_change / derivative_elapsed) / max(curvature_speed, 1e-6))
                rows[index, 3:9] = dx, dy, speed, acceleration, angle_change / math.pi, curvature
                previous_speed, previous_elapsed, previous_angle = speed, elapsed, angle
            else:
                previous_speed = previous_elapsed = previous_angle = None
        previous_index = int(index)
    return rows
