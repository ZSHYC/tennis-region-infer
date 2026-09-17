"""从已准备 cache 训练当前轨迹或视觉专家。"""

import argparse
import json
from pathlib import Path

import torch

import dataset
from training import train_expert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--expert", choices=("trajectory", "visual"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="每个专家使用独立目录；写入专家权重和 last.pt")
    parser.add_argument("--config", type=Path, default=here / "config.yaml")
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:0")
    parser.add_argument("--resume", action="store_true", help="从 output-dir/last.pt 恢复")
    parser.add_argument("--epochs", type=int, help="覆盖总训练轮数，主要用于 smoke")
    parser.add_argument("--batch-size", type=int, help="覆盖训练 batch size")
    args = parser.parse_args()
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    if device.type not in {"cpu", "cuda"} or device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device 只支持可用的 cpu/cuda 设备")
    report = train_expert(
        args.expert, args.cache_root, args.output_dir, dataset.load_config(args.config), device,
        resume=args.resume, epochs_override=args.epochs, batch_size_override=args.batch_size,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
