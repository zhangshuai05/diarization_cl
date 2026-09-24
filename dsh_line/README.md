# dsh 线交付包（给 Claude 线查漏补缺）

> 本目录由 dsh 线工作区（`D:\Deepseek_herness\2026-08\diarization`）复制而来，2026-08-28。
> 规则：**只读参考**；dsh 线不会修改本目录之外桌面上的任何内容，本目录内容也请勿回写 dsh 线工作区——如需改动，复制到自己的目录再改。

## 目录内容

| 路径 | 说明 |
|---|---|
| `VAD修复交接.md` | **最优先看**：解决你"VAD 段数恒=4、调参无效"的根因与两步修复（dynamic_silence 硬重置 + 本地 config 阈值 0.7），附实测段数对比 |
| `实验报告.md` | dsh 线完整工作日志（VAD 坑、聚类策略矩阵、在线质心追踪、UIS-RNN 监督实验与诚实负结果、全部量化口径） |
| `补录音频清单.md` | 1m 双人补录规范（监督线数据杠杆） |
| `scripts/run_diar_spk.py` | 带 VAD 修复的分离主流水线（SEACO+CAM+++谱聚类，支持 --preset N / --outdir） |
| `scripts/vad_probe.py` | VAD 阈值探针（改本地 config + dynamic_silence=False 扫阈值） |
| `eval/` | 量化评估框架：`compare_seaco.py`（文件级真值对比）、`summarize_all.py`（距离×设备聚合）、`long_eval.py`（长会议评估）、`best_per_file.py`（天花板分析）、`inspect_transcribe.py`（真值解析）；数据：`ground_truth.json`（23 文件轮级真值）、`eval_summary.csv`（UIS/v1/funasr 三方法 55 行对比） |
| `sup/` | 监督线全套（含方向 A 修复）：伪时间戳数据构建、UIS-RNN PyTorch 实现（新说话人发射先验已修复）、训练（best-ckpt + 按帧 NLL）/过拟合检查、在线推理集成、评估 |

## 给 Claude 线的三个具体建议（详见 实验报告.md 对应章节）

1. **VAD 修复**直接套用（你的质心追踪卡点大概率解决）；副作用"干净文件自动定人数漂移"用 `preset_spk_num=2` 或你的时间连续性先验缓解。
2. **评估口径共用**：你现在的融合/质心结果可以用 `eval/` 这套 purity/good 口径对 23 文件真值量化（目前你的日志里缺量化对比，用户对比进度时这是 dsh 线的优势项）。
3. **UIS-RNN 结论参考**：负结果存在"新说话人发射先验 ±54nat"的建模缺陷（实验报告第十一节），你若要试监督路线，先修这个再训，别踩同一个坑。

## 路径注意

脚本内路径基于 dsh 线工作区（HERE 相对定位 + `MODELSCOPE_CACHE=C:\Users\EDY\.cache\modelscope`），复制到你的目录后需自行适配路径；本地 VAD 模型 config 修改按 `VAD修复交接.md` 在你自己的模型副本上做。
