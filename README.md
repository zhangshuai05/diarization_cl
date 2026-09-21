# diarization_cl

Claude 侧说话人分离代码的里程碑存档。每个里程碑一个目录，各目录自带 README 与
`SHA256.json` 完整性清单。

## 里程碑

| 目录 | 日期 | 方法标识 | 内容 |
| --- | --- | --- | --- |
| （根目录） | 2026-09-08 | `claude-a2-v4-c1` | c1 v4：已部署基线，双人分离 + 逐段 ASR。 |
| `v12_meeting/` | 2026-09-21 | `claude-a2-v12-meeting` | v12：两级窗多说话人分离（2.0 秒窗定身份 + 0.6 秒窗找边界）+ 自动人数。 |

`v12_meeting/` 的状态是 `experimental`，**未替换 v7/v4 基线**。它相对 v7 在 5 个
三人文件上加权轮次准确率 +0.073，但自动人数在长会议上仍是首要缺口 —— 详见该目录
README 的「已知缺口」。

---

# c1 v4 里程碑：2026-09-08

用户确认的当前最佳 c1，方法标识 `claude-a2-v4-c1`。本目录为完整项目代码副本，原工作区与 DSH 测试台文件均保留。

## 内容

- `claude_a2.py`：当前已部署的完整 c1 算法及逐段 ASR，原样复制。
- `a2_v2.py`：对应离线算法核心，原样复制，作为研究版本存档。
- `denoise_frcrn.py`、`env_detect.py`：DSH FRCRN 降噪及自动环境判断，原样复制。
- `vendor_pkgs/`：DSH 降噪使用的本地 addict 辅助代码。
- `run.py`：服务器运行入口，使用显式模型配置，复现测试台的转码、环境检测、降噪、c1 处理顺序。
- `request.txt`：Python 依赖版本、系统组件和 CUDA 环境说明，可直接传给 pip。
- `models.example.json`：四个模型目录的配置示例。
- `verify.py`、`SHA256.json`：复制完整性检查及可选实际推理复测。
- `evidence/`：此前全量评估及网页接口结果，作为复现参考。

不包含模型权重、音频、Python 环境或 FFmpeg 二进制，也不需要 DSH 测试台的服务器、前端和 DSH 分离算法代码。第三方算法实现（如 FRCRN 网络）由 `request.txt` 中的 modelscope 等依赖提供。

## 服务器环境

推荐 Python 3.12；记录环境为 Python 3.12.10、PyTorch/torchaudio 2.11.0+cu128。GPU 服务器需兼容驱动。Linux 安装 FFmpeg 与 libsndfile，例如：

```bash
sudo apt-get install ffmpeg libsndfile1
python -m pip install -r request.txt
```

如需与记录环境相同的 CUDA 构建，先按 PyTorch 官方 cu128 源安装匹配的 torch、torchaudio，再安装其余依赖。离线服务器需要提前准备这些依赖的安装包；本目录没有打包 wheel。CPU 也可运行，但降噪较慢。Windows 需把 FFmpeg 加入 PATH。

## 模型配置

准备以下模型的完整目录（权重及配置文件），编辑 `models.example.json` 中的四个路径；可直接传入该配置文件，也可另存为 `models.json`。

| 配置键 | ModelScope 模型 ID |
|---|---|
| `vad` | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| `spk` | `iic/speech_campplus_sv_zh-cn_16k-common` |
| `asr` | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| `frcrn` | `iic/speech_frcrn_ans_cirm_16k` |

路径可为 Linux/Windows 绝对路径，也可相对于配置文件所在目录；不会相对于当前工作目录解释。`frcrn` 目录必须有 `pytorch_model.bin`。其余目录必须是 FunASR 可加载的完整快照。复现时使用同版本模型权重；模型 ID 本身不保证未来下载到相同权重，具体文件哈希见 `evidence/model_files.json`。

## 运行

```bash
python verify.py
python run.py /data/input.wav --models-config models.example.json -o output/result.json
```

默认 `--denoise auto`：与测试台一样先转为 16kHz、单声道、PCM16，检测噪声底，高于 -70dB 判为嘈杂并运行 FRCRN，然后 c1 分离和转写。支持 `--denoise on`、`--denoise off`。降噪失败会明确报错，避免误把未降噪结果当成同一条处理链。

Python 接入时，先调用 `run.load_models(config_path)`，再调用 `run.run(audio_path, models, denoise="auto")`；模型可复用。原样存档的算法文件含历史 Windows 路径常量，服务器请使用 `run.py`：该入口显式传入全部模型，并覆盖 FRCRN 的权重路径，不依赖原来的 C 盘或 D 盘目录。

## 复测

若另行准备两个原始样本，可进行实际推理检查：

```bash
python verify.py --models-config models.example.json --audio-dir /data/reference_audio
```

需要 `Note20260803112329.wav`（安静）与 `Note20260803111855.wav`（嘈杂）。校验段数、标签与边界（允许一步 0.15 秒的浮点差异），逐条报告转写文本差异。通过段落检查不表示文本逐字相同。不同设备或依赖版本可能有数值差异，校验失败应检查输出，不能直接视为代码缺失。

此前 23 文件文本匹配 282/373，带标签文本匹配 250/373；不是 DER，亦非全面超过 DSH 的结论。详情见 `evidence/`。迁移验证状态见 `evidence/packaging_check.md`；没有在真实 Linux 服务器上运行的结果时，不把本机异目录测试称为 Linux 实测。
