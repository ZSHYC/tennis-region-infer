"""原视频的一次顺序解码与默认五视图 DINO 特征推理。"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.transforms.v2 import functional as VF


def image_batch(frames: np.ndarray) -> torch.Tensor:
    batch = torch.from_numpy(frames).permute(0, 3, 1, 2)
    chunk_size = max(1, (16 * 1024 * 1024) // np.prod(frames.shape[1:]))
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    resized = []
    for chunk in batch.split(int(chunk_size)):
        chunk = VF.to_dtype(chunk, torch.float32, scale=True)
        chunk = VF.resize(chunk, [256, 256], antialias=True)
        resized.append((chunk - mean) / std)
    return resized[0] if len(resized) == 1 else torch.cat(resized).contiguous(memory_format=torch.channels_last)


def four_tiles(rgb: np.ndarray) -> tuple[np.ndarray, ...]:
    height, width = rgb.shape[:2]
    h, w = (55 * height + 99) // 100, (55 * width + 99) // 100
    return (rgb[:h, :w], rgb[:h, width - w:], rgb[height - h:, :w], rgb[height - h:, width - w:])


def extract_visual(video: Path, info: dict, model: torch.nn.Module,
                   device: torch.device, batch_size: int) -> tuple[torch.Tensor, dict]:
    if batch_size <= 0:
        raise ValueError("DINO batch size 必须为正数")
    count = len(info["frame_times"])
    features = torch.empty(count, 5, 768, dtype=torch.float16)
    global_frames, tile_frames = [], []
    global_count = tile_count = forward_batches = decoded = 0
    capture = cv2.VideoCapture(str(video))

    def forward(frames: list, tile: bool) -> None:
        nonlocal global_count, tile_count, forward_batches
        batch = image_batch(np.stack(frames)).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = model(batch)
        if values.shape != (len(frames), 768) or not torch.isfinite(values).all():
            raise ValueError("DINO 必须输出有限的 [N,768] CLS 特征")
        values = values.to("cpu", dtype=torch.float16)
        if tile:
            indices = torch.arange(tile_count, tile_count + len(frames))
            features[indices // 4, indices % 4 + 1] = values
            tile_count += len(frames)
        else:
            features[global_count:global_count + len(frames), 0] = values
            global_count += len(frames)
        forward_batches += 1

    try:
        if not capture.isOpened():
            raise ValueError(f"无法打开原视频：{video}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded >= count or frame.shape[:2] != (info["height"], info["width"]):
                raise ValueError("解码帧数或分辨率与视频元数据不一致")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            global_frames.append(rgb)
            if len(global_frames) == batch_size:
                forward(global_frames, False)
                global_frames.clear()
            tile_frames.extend(four_tiles(rgb))
            while len(tile_frames) >= batch_size:
                forward(tile_frames[:batch_size], True)
                del tile_frames[:batch_size]
            decoded += 1
        if decoded != count:
            raise ValueError(f"解码帧数与 PTS 不一致：{decoded} != {count}")
        if global_frames:
            forward(global_frames, False)
        if tile_frames:
            forward(tile_frames, True)
    finally:
        capture.release()
    return features.reshape(count, 3840), {
        "decoded_frames": decoded, "decode_passes": 1,
        "dino_forward_batches": forward_batches, "dino_forward_images": decoded * 5,
        "feature_bytes": features.numel() * features.element_size(),
    }
