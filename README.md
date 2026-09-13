# 网球击球与落地事件推理

输入一段原始视频及对应的 TrackNet 轨迹 CSV，输出每帧的击球（`hit`）、落地（`bounce`）分数，以及经过阈值与非极大值抑制筛选的事件。

模型采用**轨迹专家与视觉专家等权融合**。轨迹专家读取球的位置和运动变化；视觉专家通过 DINOv3 编码全图与四个局部视图，分析跨帧画面。两路独立产生事件分数，再按各 50% 融合。模型标识为 `trajectory-visual-fusion-v1`。

项目提供完整的视频推理链路：读取真实时间戳、对齐轨迹、生成模型输入、提取视觉特征、运行专家网络和输出事件。所有权重从本地加载，无需标注、现成特征缓存或在线服务。项目不包含训练、数据增强、标注、数据集制作、阈值搜索或模型成绩评估功能。

## 文档导航

| 文档 | 内容 |
|---|---|
| 本页 | 安装、命令行、输入输出、设备与内存、部署及常见问题 |
| [模型结构详解](doc/MODEL.md) | 轨迹特征公式、时间采样、DINOv3、两专家逐层结构、张量形状和融合后处理 |
| [实现验证记录](doc/VERIFICATION.md) | 单元测试、数值一致性、真实视频运行证据及验证范围 |
| [DINOv3 许可](doc/DINOv3-LICENSE.md) | 视觉编码器代码和权重的许可原文 |

## 1. 环境准备

### 1.1 运行依赖

需要 Python 3.10 或更新版本，以及系统命令 `ffprobe`。Python 依赖仅为 NumPy、PyTorch、torchvision 和 OpenCV，版本范围见 [requirements.txt](requirements.txt)。

`ffprobe` 随 FFmpeg 提供，用于取得视频真实 PTS（显示时间戳）。它是系统程序，安装 Python 依赖不会自动安装它。安装 FFmpeg 后，在终端确认可调用：

```bash
python --version
ffprobe -version
```

在项目目录安装 Python 依赖：

```bash
cd /path/to/tennis-region-infer
python -m pip install -r requirements.txt
```

PyTorch 与 torchvision 需要相互兼容。GPU 推理还需要支持本机显卡和驱动的 CUDA 版 PyTorch；`requirements.txt` 不指定 CUDA 构建。已有可用环境时可直接使用，不需要重新创建环境。

检查 Python 依赖和设备：

```bash
python -c "import numpy, cv2, torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python predict.py --help
```

`CUDA: False` 不影响 CPU 推理。项目直接通过 `predict.py` 运行，不需要执行 `pip install .`，也没有单独安装的命令行程序。

### 1.2 放置三份模型权重

默认目录相对于 `predict.py` 定位，目录结构必须为：

```text
models/
├── trajectory_expert.pt
├── visual_expert.pt
└── dinov3_vitb16.pth
```

| 文件 | 作用 | 大小约值 | 是否随普通 Git 提交 |
|---|---|---:|---|
| `trajectory_expert.pt` | 轨迹专家参数、窗口约定、归一化常量 | 0.82 MiB | 是 |
| `visual_expert.pt` | 视觉专家时序网络参数及输入约定 | 0.82 MiB | 是 |
| `dinov3_vitb16.pth` | DINOv3 ViT-B/16 图像编码器参数 | 327 MiB | 否 |

本地交付目录已包含全部三份权重。**通过 Git 获取项目时，还须另行取得并放入 DINOv3 权重**；只有两个专家的 `.pt` 文件不能完成视频推理。

DINOv3 对应的官方权重名称为 `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`，放入本项目时命名为 `dinov3_vitb16.pth`。骨干前向实现已包含在项目中，不需要另行安装 DINOv3 源码。程序不自动下载、替换或补齐权重。DINOv3 代码及权重适用 [DINOv3 许可](doc/DINOv3-LICENSE.md)。

## 2. 执行推理

### 2.1 自动选择设备

```bash
python predict.py \
  --video /path/to/match.mp4 \
  --trajectory /path/to/match.csv \
  --output outputs/match.json \
  --device auto
```

有可用 CUDA 设备时选择 GPU，否则使用 CPU。输入视频与 CSV 的文件名不必相同，但内容必须对应同一段视频和坐标系。

程序完整处理视频后写出一个 JSON 文件，并在终端打印耗时、事件数量和输出路径。输出文件的父目录会自动创建；已有同名文件会被覆盖。

### 2.2 显式选择 CPU 或 GPU

CPU：

```bash
python predict.py \
  --video /path/to/match.mp4 \
  --trajectory /path/to/match.csv \
  --output outputs/match-cpu.json \
  --device cpu --dino-batch-size 8
```

GPU：

```bash
python predict.py \
  --video /path/to/match.mp4 \
  --trajectory /path/to/match.csv \
  --output outputs/match-gpu.json \
  --device cuda --dino-batch-size 8
```

指定 `cuda` 但 CUDA 不可用时直接报错；也可通过 `cuda:0` 等设备编号选择一张显卡。程序不会将一次任务拆分到多张 GPU。

### 2.3 参数完整说明

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--video` | 必填 | 原始 MP4/MOV 视频路径 |
| `--trajectory` | 必填 | TrackNet 轨迹 CSV 路径 |
| `--output` | 必填 | 完整推理结果 JSON 路径 |
| `--model-dir` | 脚本旁 `models/` | 三份权重所在的目录 |
| `--device` | `auto` | 自动选择，或显式使用 `cpu`、`cuda`、`cuda:0` |
| `--batch-size` | `512` | 事件专家每批处理的中心帧数量，须为正整数 |
| `--dino-batch-size` | `64` | 编码器每批处理的图像数量，全图和局部裁剪分别组批，须为正整数 |

两个 batch size 控制不同阶段。DINO 显存不足时减小 `--dino-batch-size`；事件专家前向显存不足时减小 `--batch-size`。它们不改变物理时间窗口、融合权重、类别阈值或 NMS 半径。

从其他工作目录也能调用完整脚本路径：

```bash
python /path/to/tennis-region-infer/predict.py \
  --video /data/match.mp4 \
  --trajectory /data/match.csv \
  --model-dir /data/tennis-models \
  --output /data/results/match.json
```

显式传入的相对路径按当前工作目录解释；未传 `--model-dir` 时，权重按脚本位置定位。

## 3. 输入数据要求

### 3.1 视频

- 支持 `.mp4`、`.mov`，扩展名大小写不敏感。
- 必须是原始画面；拒绝 `visualization` 目录以及文件名中包含 `_visualized` 的视频。
- 使用第一个视频流的 packet PTS 和 time base，排序得到逐帧显示时间。
- PTS 必须存在、非负、唯一；时间轴须严格递增，并与 OpenCV 元数据帧数及实际解码帧数一致。
- 视频分辨率必须有效且解码过程中保持一致，FPS 元数据须为有限正数。

窗口采样依据 PTS 秒数，不依据假定帧率或 CSV 中的 FPS。事件时间戳保留视频原始时间轴，不额外减去第一帧的时间戳。未满足时间轴约定的视频会报错，不使用估算帧率替代。

### 3.2 轨迹 CSV

必需字段如下，列的排列顺序不限：

| 字段 | 类型与含义 | 约束 |
|---|---|---|
| `frame_number` | 从 0 开始的帧号 | 非负、严格递增，且小于视频总帧数 |
| `detected` | 该帧是否真实观测到球 | 只能为 `0` 或 `1` |
| `x_orig` | 原视频像素横坐标 | 检测帧须有限且满足 `0 ≤ x < width` |
| `y_orig` | 原视频像素纵坐标 | 检测帧须有限且满足 `0 ≤ y < height` |
| `width` | 原视频宽度 | 每行与视频宽度一致 |
| `height` | 原视频高度 | 每行与视频高度一致 |

可选的 `conf` 列表示轨迹检测分数：如果存在，非空值须为有限数，空值按 0 读取；它不参与事件模型计算。其他列不会作为模型特征。

```csv
frame_number,detected,x_orig,y_orig,width,height,conf
0,1,640.5,300.25,1280,720,0.92
1,0,,,1280,720,
3,1,647.25,306.0,1280,720,0.88
```

上述示例没有第 2 帧记录，程序会将其补为无观测；第 1 帧显式写 `detected=0`，坐标可以为空。缺失的坐标不会插值为新的球轨迹。CSV 不能完全没有数据行，但所有帧均未检测到球的情况可以处理。

必须使用与视频相同分辨率的坐标，不能直接输入缩小后的网络图像坐标。本程序消费已有轨迹，不包含 TrackNet 球检测器。

## 4. 理解结果 JSON

### 4.1 顶层字段

| 字段 | 含义 |
|---|---|
| `sample_id` | 视频文件名去掉扩展名后的名称 |
| `model` | `trajectory-visual-fusion-v1` |
| `vision_mode` | `fusion`，表示双专家融合 |
| `video`、`trajectory` | 输入文件的绝对路径 |
| `width`、`height`、`fps` | 视频元数据；FPS 不用于替代 PTS |
| `scores` | 每一帧的融合后类别分数，按帧号排列 |
| `events` | 固定阈值和 NMS 筛选后的事件列表 |
| `fusion` | 两路融合权重、类别阈值和 NMS 半径 |
| `cache` | 缓存状态；当前为 `enabled=false`，hit/miss 均为 0 |
| `timing` | 分阶段耗时、吞吐、解码数量及显存使用 |

### 4.2 逐帧分数

`scores` 恰好覆盖 `0..N-1` 全部帧；每项包含 `frame_number`、`hit_score` 和 `bounce_score`。两个分数均在 `[0,1]` 内，但不要求相加等于 1：类别概率还乘以事件存在分数。

轨迹专家和视觉专家先分别计算类别分数：

$$
s_c=\sigma(l_{\mathrm{event}})\,\operatorname{softmax}(\mathbf{l}_{\mathrm{type}})_c
$$

再执行等权融合：

$$
s_c^{\mathrm{fusion}}=0.5\,s_c^{\mathrm{trajectory}}+0.5\,s_c^{\mathrm{visual}}
$$

其中 $c$ 表示击球或落地类别。JSON 中的分数已经融合完成，无需使用者再次平均或乘事件存在分数。

### 4.3 事件列表

`events` 中的每项包含：

| 字段 | 含义 |
|---|---|
| `frame_number` | 事件帧号，从 0 开始 |
| `timestamp_seconds` | 该帧在视频原始 PTS 时间轴上的秒数 |
| `event_type` | `hit` 或 `bounce` |
| `score` | 对应类别的融合分数，不是经过概率校准的真实概率 |
| `x`、`y` | 该帧的轨迹观测像素坐标；没有观测时为 `null` |

下面是一次实际推理的事件节选：

```json
{
  "frame_number": 19,
  "timestamp_seconds": 0.6333333333333333,
  "event_type": "bounce",
  "score": 0.5205889046192169,
  "x": 817.5,
  "y": 310.0
}
```

两类阈值均为 `0.4`，恰好等于阈值的候选也可保留。每类独立执行半径 5 帧的 NMS；同分优先保留较早帧，相距恰好 5 帧仍会被抑制。输出按帧号、类别排序。

不同类别之间不互相抑制，因此同一帧可以保留两个类别。没有候选被保留时，`events` 为 `[]`，但逐帧分数仍完整输出。缺轨帧也可能检测到事件；`x/y=null` 仅表示缺少真实轨迹观测，不能当作事件无效。

### 4.4 耗时和资源字段

| `timing` 字段 | 统计范围 |
|---|---|
| `input_seconds` | 视频元数据/PTS、CSV 读取与轨迹特征生成 |
| `model_load_seconds` | 三份模型加载及设备初始化 |
| `decode_and_dino_seconds` | 完整视频解码、图像转换和五视图 DINO 编码 |
| `track_seconds` | 轨迹归一化、组窗及轨迹专家推理 |
| `visual_head_seconds` | 视觉特征组窗及视觉专家推理 |
| `postprocess_seconds` | 分数融合、阈值及 NMS |
| `total_seconds` | 从输入读取到事件结果生成的内部总耗时，不含 JSON 序列化及写出 |
| `frames_per_second` | 视频帧数除以上述内部总耗时 |
| `decoded_frames`、`decode_passes` | 实际解码帧数与像素解码遍数，正常完成时为 N 和 1 |
| `dino_forward_images`、`dino_forward_batches` | 编码的图像数量和前向批次数；图像数为 `5N` |
| `feature_bytes` | 五视图特征数组占用的字节数，不是进程总内存 |
| `cuda_peak_allocated_bytes` | PyTorch 统计的 CUDA 峰值分配字节数；CPU 为 0 |

终端输出额外包含 `output_write_seconds`、事件数和输出文件路径。`decode_passes=1` 表示像素解码一次；读取元数据时还会单独打开视频。统计值包含真实视频输入链路，不能视为只运行事件网络的速度。

## 5. 模型能力与运行边界

| 组成 | 输入与计算 |
|---|---|
| 轨迹专家 | 0.4 秒窗口、25 个采样位置、11 维特征；时间卷积＋双向 GRU＋事件存在/类型双头 |
| 五视图视觉编码器 | 全图与四个原图角落裁剪，经 DINOv3 ViT-B/16 生成每帧 3840 维特征 |
| 视觉专家 | 1.6 秒窗口、49 个采样位置；逐区域时间卷积→区域 attention→双向 GRU→双头 |
| 事件输出 | 两路类别分数各占 50%，阈值 0.4，NMS 半径 5 帧 |

视觉裁剪使用固定图像区域，不依赖轨迹坐标；即使轨迹缺失，视觉专家也有完整的图像输入。但错误轨迹仍可能影响融合结果，视觉模型也不保证找回每个缺轨事件。

窗口以待判断帧为中心，视觉分支需要约 0.8 秒未来上下文。程序在读取完整视频并提取特征后输出结果，适用于离线视频处理，不是即时流式服务。

全部五视图特征临时保留在 CPU 内存，大小约为 `N × 7680` 字节。例如 100,000 帧仅该数组就需要约 732 MiB，此外还有原图批次、模型、窗口张量及输出分数等内存开销。减小 DINO batch size 可以减少批次内存，但不会缩小完整视频的特征数组。

CPU 与 GPU、不同软件版本或 batch size 可能产生浮点差异，阈值附近的结果也可能受影响。结构、精度路径和处理细节见[模型结构详解](doc/MODEL.md)。

## 6. Python 调用

在项目目录，或者已将项目目录加入 Python 模块搜索路径时：

```python
from pathlib import Path
from predict import predict

result = predict(
    Path("/data/match.mp4"),
    Path("/data/match.csv"),
    device="cpu",
    batch_size=512,
    dino_batch_size=8,
)
print(result["events"])
```

函数返回与命令行 JSON 相同结构的字典，不自动写文件。每次调用会重新加载模型并处理一个视频；同一调用中的两位专家处理同一完整时间轴。

## 7. 文件组织与部署

```text
tennis-region-infer/
├── README.md
├── requirements.txt
├── predict.py                 # 推理入口、时间窗口、融合与事件输出
├── inputs.py                  # 视频 PTS、轨迹 CSV 与运动特征
├── vision.py                  # 顺序解码、五视图输入与编码调度
├── dino.py                    # DINOv3 ViT-B/16 图像编码器
├── model.py                   # TrajectoryExpert 与 VisualExpert
├── models/                    # 三份本地权重
├── doc/
│   ├── MODEL.md               # 详细模型结构
│   ├── VERIFICATION.md        # 实现验证记录
│   └── DINOv3-LICENSE.md       # 许可原文
└── tests/                     # 推理实现单元测试
```

运行需要五个源码文件及三份权重；部署时同时保留依赖说明与许可文件。`outputs/` 和本地检查产物不属于模型依赖。源码中没有固定的数据集路径，移动项目后可通过绝对输入路径继续使用。

若使用 Git 管理部署版本，两份专家权重已纳入版本控制；DINOv3 大权重被 `.gitignore` 排除，需单独传递。只有源码仓库的副本不等于完整的模型部署副本。

## 8. 常见问题

| 现象 | 原因与处理 |
|---|---|
| 无法调用 `ffprobe` | 安装 FFmpeg，并确认 `ffprobe -version` 可执行 |
| 找不到模型文件 | 检查三份权重的文件名、位置和 `--model-dir` |
| 权重结构或模型约定不匹配 | 使用配套的专家权重与 DINOv3 ViT-B/16 权重，不混入其他网络权重 |
| CUDA 不可用 | 使用 `--device cpu`，或检查当前 PyTorch 构建及显卡驱动 |
| CUDA 显存不足 | 按报错阶段减小 `--dino-batch-size` 或 `--batch-size` |
| 主机内存不足 | 减小图像批次；若完整特征数组已过大，应使用更大内存环境 |
| CSV 帧号越界、重复或乱序 | 核对视频片段对应关系，使用从 0 开始、严格递增的有效帧号 |
| 轨迹与视频分辨率不一致 | 确认 `width/height`，并将坐标转换回原视频像素坐标系 |
| PTS 数量与帧数不一致 | 检查视频时间轴、封装和解码完整性；程序不会用估算 FPS 替代 |
| 事件坐标为 `null` | 对应帧没有轨迹观测；事件可由视觉信息支持 |
| 没有事件输出 | 查看 `scores`；可能没有候选达到固定阈值，不代表程序没有运行 |
| CPU 推理较慢 | 五视图 DINO 编码占主要计算量；可使用支持 CUDA 的环境 |

## 9. 实现检查

```bash
python -m unittest discover -s tests -v
```

测试使用 Python 标准库 unittest，检查推理输入约定、网络前向、时间采样和后处理，不运行训练或模型成绩评估。缺少可选本地权重时部分测试会跳过，因此部署完整性还应以三份权重均在位且真实推理成功为准。已完成的数值与真实视频检查见[实现验证记录](doc/VERIFICATION.md)。
