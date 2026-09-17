# 训练与评估

本次只迁移代码，没有重新训练两位专家。现有 `models/` 是历史 B0＋sigma18 发布权重。

## 数据范围

指定已经发布的 `data` 根目录即可。WSL 中路径为 `/home/zshyc/event/data`；`\\wsl.localhost\Ubuntu-22.04\home\zshyc\event\data` 是 Windows 访问同一目录的路径，不要直接作为 Linux shell 路径。

| 用途 | 目录 | 当前视频数 |
|---|---|---:|
| 训练 | `20260706_0712` | 26 |
| 训练 | `2606_admin_back` | 13 |
| 训练 | `loveall` | 173 |
| 训练 | `tracknet_tennis` | 95 |
| 训练 | `e2espot_tennis` | 3445 |
| 评估 | `back_match_clipped` | 10 |

训练共 3752 段，评估 10 段。每个来源已有 `video/`、`tracknetv5/`、`GT/`；GT 支持完整事件 CSV 或已有 LabelMe JSON。`2606_admin_back_clipped` 不被发现，admin 原目录的 13 段各使用一次。不运行切视频、图片编码、TrackNet 或可视化制作。

根 `config.yaml` 是来源与训练参数的入口。没有 validation，也不继承历史测试名单；不得利用 back_match 指标选择轮次、调学习率或搜索阈值，再称它为独立测试。

## 安装

Python 3.10+，先安装适合本机的 PyTorch/torchvision，再执行：

```bash
python -m pip install -r requirements.txt
ffprobe -version
```

准备视觉缓存需要 `models/dinov3_vitb16.pth`；沿用本项目已提供的 DINO 实现及许可，不要求安装原实验仓库。

## 准备一次缓存

```bash
python prepare.py --data-root /home/zshyc/event/data --cache-root cache --device cuda
```

先只验证一个样本可执行：

```bash
python prepare.py --data-root /home/zshyc/event/data --cache-root cache \
  --split train --sample-id tracknet_game8_clip6 --base-only --device cpu
```

`--base-only` 仅准备轨迹、PTS 和 GT，适用于只训练 B0。视觉训练需要去掉该参数准备五视图特征。准备单个样本不会将整个训练名单缩成一个视频；正式训练前需准备完整 train 缓存。

`manifest.json` 保存完整来源名单；base 与视觉缓存独立保存、逐视频原子写入，命中时不重新解码。训练和评估只读缓存，不读取原视频、CSV 或 GT 文件。原视频 `predict.py` 仍支持不持久化缓存的一次性推理。

## 后续由你启动训练

```bash
python train.py --expert trajectory --cache-root cache --output-dir outputs/trajectory --device cuda
python train.py --expert visual --cache-root cache --output-dir outputs/visual --device cuda
```

两位专家从头独立训练；视觉训练只读取冻结 DINO 特征，不训练 DINO。B0 窗口为 0.4 秒/25 位置，视觉为 1.6 秒/49 位置。保留 Gaussian eventness、固定负采样、双头损失和 B0 缺轨增强。

没有验证集，所以不输出按 back_match 选择的 best。固定轮数完成后输出 `trajectory_expert.pt` 或 `visual_expert.pt`，`last.pt` 用于恢复；两位专家使用不同输出目录。

```bash
python train.py --expert visual --cache-root cache --output-dir outputs/visual --device cuda --resume
```

恢复时保持同一数据、配置及总训练轮数。新方案改变了训练视频和选择规则，即使架构相同，也不保证得到历史 sigma18 参数或成绩。

## 只在 back_match 评估

先准备评估缓存，再指定新训练的两个文件：

```bash
python prepare.py --data-root /home/zshyc/event/data --cache-root cache --split eval --device cuda
python evaluate.py --cache-root cache \
  --trajectory-checkpoint outputs/trajectory/trajectory_expert.pt \
  --visual-checkpoint outputs/visual/visual_expert.pt \
  --output outputs/back-match-evaluation.json --device cuda
```

默认逐类分数 50/50 融合，阈值 0.4/0.4，同类别 NMS 半径 5 帧。报告事件级一对一匹配的 precision/recall/F1、FP/分钟、逐视频指标及耗时；不会搜索阈值或修改模型。`--expert trajectory` 或 `--expert visual` 可单独检查专家，使用相同配置工作点。

省略 checkpoint 参数会评估已有历史发布权重，报告会明确标注没有新方案训练信息；不能将其解释为排除 back_match 后的结果。仅验证入口可加 `--sample-id back_match_benchmark_01`，该结果不是完整 10 段评估。sample ID 使用视频文件名去掉扩展名。

0.4/0.4 是沿用的历史工作点，曾受旧方案 validation 影响；新方案固定复用它不等于从未参考过 back_match 的全新盲测协议。

## 使用新权重推理

训练输出与现有推理格式兼容。将两份新专家文件和 DINO 权重放入单独目录，指定 `--model-dir`，无需覆盖历史 `models/`：

```bash
python predict.py --video /path/to/video.mp4 --trajectory /path/to/track.csv \
  --model-dir /path/to/new-models --output outputs/prediction.json --device cuda
```

训练、评估和 CLI 推理使用根配置中的工作点。评估 GT 只计算指标，不参与预测。已有推理的坐标输出只是 TrackNet 当前帧坐标，缺轨时为 null。
