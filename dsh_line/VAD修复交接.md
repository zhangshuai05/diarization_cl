# VAD 修复交接卡（给 Claude 线）

> 来源：dsh 线工作区（D:\Deepseek_herness\2026-08\diarization）。针对你日志第十节的痛点："VAD 段数始终=4，调参无效（嘈杂环境整段持续'有语音'）"。
>
> **更新（2026-08-28 晚）**：你的 A2 滑窗方案已绕开 VAD 段粗问题，本卡的紧急度下降。保留价值在两点：① 解释你当时"VAD 调参无效"的**根因**（避免以后在 ASR 侧 VAD 上再踩）；② 若将来仍用 VAD 切段（如实时流式方案），此修复是前提。可作参考而非必读。

## 根因（funasr 1.3.14 源码确认）

`funasr/models/fsmn_vad_streaming/model.py` 里有一个**动态静音调度**，默认开启（`dynamic_silence` 缺省 = `max_end_silence_time is None`）。它在**说话期间**执行：

```python
cache["stats"].speech_noise_thres = 0.5   # L964 附近，硬编码覆盖！
```

也就是：嘈杂环境下只要 VAD 一旦进入"有语音"状态，判定阈值被硬重置为 0.5，背景零星语音永远高于阈值 → 整段不切。**你调 `max_single_segment_time` / `speech_to_sil` / `max_end_silence` 都没用，因为阈值被这段代码覆盖了。**

## 修复（两步，缺一不可）

1. **改本地 VAD 模型 config.yaml**：`iic--speech_fsmn_vad_zh-cn-16k-common-pytorch\snapshots\master\config.yaml` 里
   `speech_noise_thres: 0.6` → `speech_noise_thres: 0.7`（运行时 kwargs 传不进模型，只能改 config；这也是我扫 0.6~0.95 后选的值）。
2. **关掉动态调度**：`AutoModel(..., vad_model=<本地 config 路径>, vad_kwargs={"dynamic_silence": False})`。

## 实测效果（同批 2m 文件，修复前后）

| 文件 | 修复前 VAD 段数 | 修复后 |
|---|---|---|
| 116@2m（Call20260718210657，44.5s/15 轮真值） | 1 | **16** |
| 118@2m（Note20260803111605，42.2s/15 轮真值） | 4 | **8~11** |

修复后段数接近真值轮数，质心追踪/离线聚类才有米下锅。

## 附带工具

`vad_probe.py`（在本交付目录）：改写本地 config.yaml 的 speech_noise_thres 并带 `dynamic_silence=False` 扫阈值的探针，可直接复用到你的环境（改路径）。

## 注意事项

- 阈值 0.7 是对这两台 40 元麦+游戏厅噪声调的；安静环境可回 0.6，你自己扫一遍更稳。
- 副作用：干净文件（1m）段切细后，funasr 内置谱聚类的"特征间隙自动定人数"会漂（116@1m 出现过 2→9 标签），预置人数 `preset_spk_num=2` 或质心追踪可缓解；详见 dsh 线实验报告第七节。
