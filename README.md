# 网球事件纯推理

原始视频＋TrackNet CSV → 每帧 hit/bounce 分数及事件。完整实现 `tennisvar-lite`
在 2026-09-13 部署的默认方案：**B0 轨迹专家＋五视图区域先时间视觉专家，分数各占 50%**。
这是独立项目，运行时不引用研究仓库、旧缓存、标注或训练配置，不联网下载模型。

只有推理所需的网络、输入转换和事件后处理；没有训练、数据增强、标注、数据集制作、
阈值搜索、交叉验证、性能评估或可视化命令。`tests/` 仅检查推理实现，不计算模型成绩。

## 安装与运行

需要 Python 3.10+、系统 `ffprobe`（FFmpeg 提供）。在已安装合适 CPU/CUDA 版
PyTorch 和 torchvision 的环境中安装依赖：

```bash
cd /home/zshyc/tennis-region-infer
python -m pip install -r requirements.txt
python predict.py --help
python predict.py \
  --video /path/to/video.mp4 \
  --trajectory /path/to/tracknet.csv \
  --output outputs/prediction.json \
  --device auto
```

本机现成真实样本：

```bash
python predict.py \
  --video /home/zshyc/event/data/tracknet_tennis/video/tracknet_game8_clip6.mp4 \
  --trajectory /home/zshyc/event/data/tracknet_tennis/tracknetv5/tracknet_game8_clip6.csv \
  --output outputs/tracknet_game8_clip6.json \
  --device cpu --dino-batch-size 8
```

上面的路径只用于演示，换成自己的视频和 CSV 即可。也可从任意工作目录调用
`python /path/to/tennis-region-infer/predict.py ...`，默认权重相对脚本定位。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--video` | 必填 | 原始 `.mp4` / `.mov`，大小写均可 |
| `--trajectory` | 必填 | 对应视频原始分辨率的 TrackNet CSV |
| `--output` | 必填 | 包含分数、事件与耗时的 JSON 文件；已有同名文件会覆盖 |
| `--device` | `auto` | 有 CUDA 就使用，否则 CPU；也可强制 `cpu`、`cuda`、`cuda:0` |
| `--batch-size` | `512` | 两个事件头每批判断的中心帧数 |
| `--dino-batch-size` | `64` | DINO 每批图像数；内存或显存不足时减小，例如 `8` |
| `--model-dir` | 脚本旁 `models/` | 三份本地模型文件的目录 |

强制 CUDA 但设备不可用会报错。batch size 只控制资源用量，不重新选模型或阈值；
不同设备、PyTorch 版本或 batch size 可能产生浮点末位差异。

## 输入与输出

CSV 必须含 `frame_number,detected,x_orig,y_orig,width,height`；原仓库的 `conf` 列
也可保留。帧号从 0 开始且严格递增，允许跳帧：未提供的帧按没有轨迹观测处理。
`detected` 为 0/1，检测到的坐标须在原视频画面内，宽高必须与视频一致。
CSV 中的 FPS 不参与计算；窗口时间只取视频真实 PTS。

```csv
frame_number,detected,x_orig,y_orig,width,height
0,1,640.5,300.25,1280,720
1,0,,,1280,720
```

视频必须满足 packet PTS 非负、唯一、与解码帧数一致的原模型输入合同；不满足则报错。
拒绝 `visualization` 目录和带 `_visualized` 名字的视频，避免把叠加标注当作视觉输入。
本工具不负责生成 TrackNet 轨迹，也不需要任何 GT。

JSON 中 `scores` 包含全部视频帧的 `frame_number / hit_score / bounce_score`；
`events` 包含经过固定阈值与 NMS 的 `frame_number / timestamp_seconds / event_type / score / x / y`。
时间为对应原始 PTS 秒数，坐标为该帧 TrackNet 真实观测像素坐标；缺轨时 `x/y` 为 `null`。
`score` 是模型分数，不是经过概率校准的真实概率。没有事件时 `events` 为 `[]`，
仍保留全部逐帧分数。事件按帧号、类别排序，同一帧可以保留不同类别。

`timing` 分开报告输入转换、模型加载、视频解码与 DINO、轨迹头、视觉头和后处理耗时，
包括完整视频吞吐和 CUDA peak allocated memory；终端另报告文件写出耗时。
`decode_passes=1` 指像素顺序解码一次，元数据探测另行打开视频。
没有持久化 cache，故 cache 的 enabled 为 false、hit/miss 均为 0。

## 固定模型与权重

| 文件 | 内容 | 分发 |
|---|---|---|
| `models/default-b0.pt` | 原默认 B0 的全部参数、窗口合同和归一化常量 | 随 Git 提交，约 0.82 MiB |
| `models/default-region-visual.pt` | 原默认 region 视觉事件头全部参数与合同 | 随 Git 提交，约 0.82 MiB |
| `models/dinov3_vitb16.pth` | 原本机许可 DINOv3 ViT-B/16 LVD-1689M 骨干权重 | 已复制到本地，约 327 MiB，不进入普通 Git 历史 |

这三份都必须存在；不需要另装 DINO 官方源码。大权重原名为
`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`，移动部署时一并复制并按表中名字存放。
**仅克隆 Git 仓库还不能运行：须另行携带这份骨干权重。** 后续发布远端时可将它作为
Release 附件传递，当前项目没有自动下载或远端发布动作。使用和分发 DINO 代码及权重
须附带原文 [DINOv3 许可](DINOv3-LICENSE.md)。

网络代码只保留所需推理路径。`dino.py` 基于本地官方
[DINOv3 固定版本](https://github.com/facebookresearch/dinov3/tree/6876159a11b4df116f30f667f8c9888617df0751)
提取 ViT-B/16 的前向；全部 state_dict 键和参数形状保持一致。
事件头源自 `ZSHYC/event` 的 `b741b2978165bf24baf82e2f4dc782553b8327c3` 工作树，
对应部署文件 `default-b0.pt` 和 `default-region-visual.pt`。
部署文件去掉训练配置、数据划分和训练记录，没有裁剪参数或重新训练。

固定计算：轨迹 0.4 秒 / 25 位置 / 11 维；视频全图＋原图四角 55% 裁剪，
每图按原方法缩放至 256×256，DINO CLS 存为 float16 后输入视觉头，
视觉 1.6 秒 / 49 位置。先逐区域三层时间卷积，再区域 attention、双向 GRU 和双头。
两专家各自使用 `sigmoid(eventness) × softmax(type)`，逐类等权相加；
hit/bounce 阈值均为 0.4，NMS 半径为 5 帧，同分优先较早帧。
PTS 最近帧等距时取较早帧，窗口越界位置补零。B0 使用发布的 float32 坐标和
velocity midpoint 导数规则，只对有效检测行做发布常量归一化。

需要约 0.8 秒未来上下文，属于离线居中窗口推理。视频逐帧只解码一次，全部五视图
特征暂存在 CPU 内存，约 `帧数 × 7680` 字节；原图只保留当前 DINO batch，
不产生预处理数据集或磁盘特征缓存。

## 实现检查

```bash
python -m unittest discover -s tests -v
```

测试使用标准库 unittest，不安装额外测试框架。真实推理对齐结果另见
[交付核对记录](VERIFICATION.md)。代码组织参考
[tennis-event-infer](https://github.com/ZSHYC/tennis-event-infer)，模型与输入计算使用本项目当前默认版本。
