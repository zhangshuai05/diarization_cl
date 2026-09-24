# diarization_cl

单 mic 长会议系统的**说话人分离 + 说话人识别**。目标场景：安静、轮流发言、
36–40 分钟长音频。

## 快速上手（复现 DER 数字）

```bash
pip install -r requirements.txt

# 1) 数据与模型不在仓库内，先准备（见下节「外部依赖」）

# 2) 零推理先验口径（只读缓存，几十秒）
python -u -B scripts/evaluation/probe_fullref_truncation.py --skip-v12

# 3) 真实路径端到端（加载模型，约 30 分钟跑 12 场）
python -u -B scripts/evaluation/probe_e2e_unlocked2.py --split dev --min-win 2

# 4) 测试（必须在 streaming_asr/ 里跑：那里有 pytest.ini 设 testpaths/pythonpath；
#    在仓库根跑会把 milestones/ 与 backups/ 下的同名测试一起收集，报 import file mismatch）
cd streaming_asr && python -B -m pytest -q
```

## 当前结论（2026-09-24）

真实路径（真实 FSMN VAD 门，`v12.diarize()`，600s，collar=0.25）：

| 分 | 场次 | 基线 DER | 解锁 DER | 改善/恶化/不变 |
|---|---:|---:|---:|---|
| dev | 12 | 24.4% | **14.8%** | 3 / 0 / 9 |
| holdout | 8 | 19.8% | **10.2%** | 3 / 0 / 5 |

"解锁" = 拆掉 `claude_a2_v11.py` 里选 K 的**三道串联锁**（上限 def 时绑定为 4、
尺寸/份额门、孤窗令 `_silhouette` 返回 −1），三道必须一起改，单独拆任何一道都是
精确的 no-op。实现见 `scripts/evaluation/unlocked_selector.py`。

**必须带上的限定**：`min_win=2` 是在全部 20 场上选出的，**dev 与 holdout 都算样本内**
（AISHELL-4 的 test 划分只有 20 场，见下节）。

## 文档入口（按可靠性排序）

| 文件 | 内容 |
|---|---|
| `交接文档_0923.md` | 上一份交接文档。**§0/§4.6/§4.7 已被下面的实测记录更正** |
| `artifacts/v13_会话发现_20260923.md` | **2026-09-24 实测记录**，含三次自我更正 |
| `artifacts/v13_交付合并表.md` | 交付数字 + 证据索引（A 真实路径 / B 离线 / C 物理边界 / D 未验证 / E 产物索引） |
| `D:\yinp\AISHELL-4\DATA_SPEC.md` | AISHELL-4 数据规格（**在数据目录里，不在本仓库**） |

**新对话开场**：读发现记录与交付合并表，再从合并表 §D「尚未验证」挑一条。

## 外部依赖（不入库，需自行准备）

### 数据

| 项 | 路径 | 说明 |
|---|---|---|
| AISHELL-4 test | `D:\yinp\AISHELL-4\data\test\` | 20 场、8 通道、12.7 小时；本项目**只读 ch0** |
| 嵌入缓存 | `D:\yinp\AISHELL-4\_v13_spike\cache\` | 2.0s 窗 CAM++ 嵌入，探针的零推理前提 |
| VAD 区间缓存 | `...\_v13_spike\cache\vad\` | `cache_vad_intervals.py` 生成，只需一次 |

数据下载：OpenSLR SLR111 <https://www.openslr.org/111/>。

### 模型（走 ModelScope 缓存，不入库）

- **CAM++ 声纹**：`iic/speech_campplus_sv_zh-cn_16k-common`
- **FSMN VAD**：本机用的是**调过的本地副本**（`speech_noise_thres=0.7`），
  共享缓存是 0.6 且会把语音并成连续段。路径见 `streaming_asr/web_server.py` 的 `VAD_LOCAL`。
  ⚠ 本机 funasr 是魔改版：**不显式传 `max_end_silence_time` 时 `speech_noise_thres` 失效**。

### 环境

两个 conda 环境：`funasr`（主链路）、`3d-speaker`（CAM++）。细节见
`docs/环境清单.md`（含绝对路径与本机未装的备选模型）。

## 目录

| 目录 | 内容 |
|---|---|
| `streaming_asr/` | **算法主线**：v7–v12 泳道、`der.py`（DER 评分器）、`web_server.py`（8735 测试台） |
| `scripts/evaluation/` | **DER、对照与验收脚本**（本会话主战场，含 `unlocked_selector.py`） |
| `scripts/batch/`、`scripts/diagnostics/`、`scripts/audio/` | 批量推理、一次性诊断、音频处理 |
| `artifacts/` | 实测记录、交付表与 JSON 产物 |
| `docs/` | 历史计划、日志、环境清单 |
| `milestones/` | 版本归档（**只读，勿改**；`c1_v12_20260922` 是 v12 冻结快照） |

## 已知边界（写报告必须声明）

- **单通道天花板**：说话人重叠时单标签输出必然错一部分；本机无 TS-SEP。
  分离式提纯已否证（ClearVoice MossFormer2_SS 在轮替对话上退化成对半拆）。
- **参考里多是小说话人**：53 个"短发言者"时长占比 <5%，中位仅 2.0 秒；
  按"参考全人数"当已知 K 会造出假收益。见 `aishell4-tiny-reference-speakers` 相关记录。
- **`der.py` 口径**：union 分母（md-eval 口径）；**并人落在 `confusion`，不在 FA**。
- **DER 的两条口径差 69 个点**：参考是否截断到分析窗口。
  引用任何上界数字时必须标明（`probe_fullref_truncation.py` 有对照表）。
