# -*- coding: utf-8 -*-
"""按**生产门**重建探针用的窗嵌入缓存（主窗 / 细化窗 / 救援窗）。

## 为什么不能直接用 `_v13_spike/cache/<m>/w2_h1.npz`

那份是 **RTTM 门 + `skip_seconds=10`** 建的（`v12_aishell4.pure_windows` 的默认），
而生产路径（`v12.diarize`）与 `probe_ab_dev` 用的是 **VAD 门、无 skip**。
两者窗集不同，且差异按场次浮动（dev 12 场实测 −264 … +2 个窗），**不是常量偏移**，
所以那份缓存不能用于路径等价检查。实测 `L_R003S01C02`：探针 422 窗、
旧缓存门控 418 窗，差额恰好是 `skip=10` 吃掉的 t=5,7,8,9 四个窗。

## 三个门（**必须分开存**）

    主窗   | 2.0 / 1.0  | MIN_VAD_FRACTION = 0.5 | v11._extract_windows @2.0
    细化窗 | 0.6 / 0.15 | REFINE_MIN_VAD   = 0.3 | v12._refine_windows
    救援窗 | 0.6 / 0.15 | MIN_VAD_FRACTION = 0.5 | v11._extract_windows @0.6

⚠️ **细化窗与救援窗长宽完全相同，只有门限不同**（0.18s vs 0.30s）。
混用**不会报错**，只会让救援路径的窗集悄悄变成细化窗集。

`STAGES` 里的 `gate` **只用于声明与记录**：真正生效的门限由生产函数**在调用时**
读模块变量（`_extract_windows` 读 `v11.MIN_VAD_FRACTION`，`_refine_windows` 读
`v12.REFINE_MIN_VAD`）。所以 `extract()` 断言"声明值 == 当前模块值"——若有人在本
模块 import 之后改了那些常量，这里立刻报错，而不是把**导入时快照**写进 manifest、
让人以为窗集是按记录的门限建的。

## VAD 用真模型，不读 `vad/*.json`

`vad/*.json` 实测确实是真 FSMN VAD 的 600s 截断，但它存成了 **3 位小数**。
门限是 `overlap >= gate * window` 的**精确比较**，1ms 舍入就可能翻转边界窗——
而本缓存的全部意义就是与生产路径**逐窗一致**，所以多付一次 VAD 加载。

## 自检（本脚本以非零码退出的唯一条件）

    主窗窗数   必须等于 `probe_ab_dev` 报的 `n_windows`
    细化窗窗数 必须等于 `probe_ab_dev` 报的 `n_dense`
    场次必须存在于参照里（否则下面两条无从比对，也必须报）

这两个数只由 音频 + VAD 区间 + 窗参数 决定，**与嵌入无关**，所以是标定门限的硬判据。
（救援窗在 `probe_ab_dev` 里没有对应量——那条路径探针从未走过，故不设断言。）

## 写入是**合并**，不是覆盖

缺省 `--meetings` 时会带上既有 manifest 的全部场次（dev 12 场是逐次累加建成的），
所以无参数运行不会把 manifest 缩回 `TARGET_MEETINGS` 那三场。

    python -u -B scripts/evaluation/build_probe_cache.py
    python -u -B scripts/evaluation/build_probe_cache.py --meetings L_R003S01C02
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 只插 `streaming_asr/` + 本目录。**不要插仓库根**：那里有同名 `claude_a2.py`
# （DSH 测试台用），插进去会遮住 `streaming_asr/claude_a2.py`。
# 回归测试：`streaming_asr/tests/test_evaluator_path_hygiene.py`
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

CACHE = Path(r"D:\yinp\AISHELL-4\_v13_spike\cache")
DATA = Path(r"D:\yinp\AISHELL-4\data\test")
OUT = CACHE / "probe"
SECONDS = 600.0
PROBE_RESULTS = ROOT / "artifacts" / "v13_ab_dev_results.json"

# 三个门 = 生产路径里实际用的三个 (窗长, 步长, 门限)。取值全部来自模块常量，
# 不写字面量：生产改了常量，这里跟着变，不会悄悄漂移。
STAGES = {
    "main": {"window": v12.WIN_S, "hop": v12.HOP_S, "gate": v11.MIN_VAD_FRACTION},
    "refine": {"window": v12.REFINE_WIN_S, "hop": v12.REFINE_HOP_S,
               "gate": v12.REFINE_MIN_VAD},
    "rescue": {"window": v12.REFINE_WIN_S, "hop": v12.REFINE_HOP_S,
               "gate": v11.MIN_VAD_FRACTION},
}

# 第 2 步（路径等价检查）的三场：救援案例 / 细化案例 / 明确回归案例。
TARGET_MEETINGS = ["L_R003S01C02", "L_R003S03C02", "L_R003S04C02"]


def load_existing():
    """既有 manifest 里已建过的场次与其条目（用于合并，不覆盖式重建）。

    dev 12 场是**逐次累加**建出来的，而 `--meetings` 默认只列第 2 步那三场。
    若每次都新建 manifest 直接覆盖，无参数跑一次就会让 manifest 从 12 场缩回
    3 场，`probe_path_equiv` 随后读不到另外 9 场的 VAD 区间。
    """
    path = OUT / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("meetings", {})
    except (json.JSONDecodeError, OSError):
        return {}


def live_vad(models, wav, seconds=None):
    """生产口径的 VAD 区间：真 FSMN + `seconds` 截断（照抄 `probe_ab_dev:112-113`）。

    ⚠️ `seconds` **没有默认值**，必须显式传：早先 `--seconds` 只写进 manifest、
    没有传下来，于是 `--seconds 300` 会记录 "300" 却按 600 实算——manifest 元信息
    与实际窗集不一致，且自己也无法察觉这个偏移。用默认值 `=SECONDS` 会重新留下
    同一个缺口（本项目已确认 bug 族的"看起来可配、实际配了不生效"），故不设。
    """
    if seconds is None:
        raise ValueError("seconds 必须显式传入（不要依赖模块级 SECONDS）")
    segments = v11._vad_segs(models["vad"], wav)
    return [(s, min(e, seconds)) for s, e in segments if s < seconds]


def gate_in_effect(stage):
    """该阶段**当前**真正生效的门限。门限由生产函数在调用时读模块变量。"""
    return v12.REFINE_MIN_VAD if stage == "refine" else v11.MIN_VAD_FRACTION


def extract(stage, audio, rate, vad_segments, models):
    """按阶段调**生产函数本身**，不重写提取循环。

    主窗与救援窗共用 `v11._extract_windows`，靠 `_v11_window_params` 换常量区分；
    细化窗走 `v12._refine_windows`（它有自己的门限与 `REFINE_MIN_WINDOWS` 检查）。

    断言声明门限 == 生效门限：两者不同时 manifest 记的门限就是**导入时快照**，
    与实建窗集不符（本会话第 6 次"派发错但无人报错"的同族风险）。
    """
    assert gate_in_effect(stage) == STAGES[stage]["gate"], (
        f"{stage} 门限漂移：生效 {gate_in_effect(stage)} "
        f"!= 声明 {STAGES[stage]['gate']}")
    if stage == "refine":
        emb, times = v12._refine_windows(audio, rate, vad_segments, models["spk"])
        return emb, times
    with v12._v11_window_params(STAGES[stage]["window"], STAGES[stage]["hop"],
                                v12.MAX_SPEAKERS):
        emb, times, _errors = v11._extract_windows(audio, rate, vad_segments,
                                                   models["spk"])
    return emb, times


def build_meeting(meeting, models, seconds=None):
    """与 `live_vad` 同规矩：`seconds` 必须显式传（理由见其 docstring）。"""
    import soundfile as sf

    if seconds is None:
        raise ValueError("seconds 必须显式传入（不要依赖模块级 SECONDS）")
    wav = str(DATA / "wav" / f"{meeting}.flac")
    info = sf.info(wav)
    stop = min(int(seconds * info.samplerate), info.frames)
    audio, rate = sf.read(wav, dtype="float32", stop=stop)
    if audio.ndim > 1:
        audio = audio[:, 0]
    vad_segments = live_vad(models, wav, seconds)

    target = OUT / meeting
    target.mkdir(parents=True, exist_ok=True)
    counts = {}
    for stage in STAGES:
        emb, times = extract(stage, audio, rate, vad_segments, models)
        times = np.asarray(times, dtype=float)
        np.savez(target / f"{stage}.npz", emb=emb, times=times)
        counts[stage] = int(len(times))
    return counts, vad_segments


def probe_reference():
    """`probe_ab_dev` 报的两个窗数（主窗 / 密窗），作为门限标定的参照。"""
    if not PROBE_RESULTS.exists():
        return {}
    payload = json.loads(PROBE_RESULTS.read_text(encoding="utf-8"))
    return {row["meeting"]: {"n_windows": row.get("n_windows"),
                             "n_dense": row.get("n_dense")}
            for row in payload["results"] if not row.get("insufficient")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="要建的场次；缺省 = 既有 manifest 的全部场次 ∪ TARGET_MEETINGS")
    ap.add_argument("--seconds", type=float, default=SECONDS)
    args = ap.parse_args()

    # 合并而不是覆盖：缺省时把既有 manifest 的场次一并带上，避免无参数运行
    # 把 12 场缩成 3 场（见 `load_existing` 的说明）。
    existing = load_existing()
    if args.meetings is None:
        args.meetings = sorted(set(existing) | set(TARGET_MEETINGS))
        print(f"未指定 --meetings：沿用既有 {len(existing)} 场 + 目标场，"
              f"共 {len(args.meetings)} 场", flush=True)

    reference = probe_reference()
    if not reference:
        raise SystemExit(f"缺少 {PROBE_RESULTS}，无法做门限自检——先跑 probe_ab_dev")
    if abs(args.seconds - SECONDS) > 1e-9:
        # 自检的参照（`probe_ab_dev` 的 n_windows/n_dense）是 600 秒口径。
        # 换了截断长度窗数**本来就该不同**，此时自检必然失败、而失败原因会指向
        # "门限不一致"——那是误导。直接拒绝，不提供绕过开关：要改截断长度，
        # 先按同口径重跑 probe_ab_dev 生成新参照，否则这份缓存没有可比对的基准。
        raise SystemExit(
            f"--seconds {args.seconds} 与自检参照口径不符（参照是 {SECONDS}s）。"
            f"自检判据只对 {SECONDS}s 有效；请先按 {args.seconds}s 口径重跑 "
            "probe_ab_dev 后再建缓存。")

    from web_server import get_models
    models = get_models(lambda m: None, need_stream=False, need_offline=False)

    manifest = {"schema": "probe-window-cache-v1", "seconds": args.seconds,
                "vad": "live FSMN（web_server 的 VAD_LOCAL），按 seconds 截断",
                "metric": "cosine（CAM++ 归一化嵌入）",
                "stages": {name: dict(cfg) for name, cfg in STAGES.items()},
                # 先并入既有条目：未重建的场次条目原样保留，不被这次运行抹掉。
                "meetings": dict(existing)}
    failures = []

    for index, meeting in enumerate(args.meetings, 1):
        t0 = time.perf_counter()
        counts, vad_segments = build_meeting(meeting, models, args.seconds)
        ref = reference.get(meeting, {})
        print(f"[{index}/{len(args.meetings)}] {meeting}  "
              f"{time.perf_counter() - t0:6.1f}s  "
              f"主 {counts['main']}  细化 {counts['refine']}  救援 {counts['rescue']}  "
              f"| 探针 主 {ref.get('n_windows')} 密 {ref.get('n_dense')}", flush=True)

        if not ref:
            # 该场不在探针参照里 ⇒ 下面的自检全都 `continue`，缓存照写却零验证。
            # 必须显式记一笔，否则"有自检"是假的（同族：看起来有防护、实际没生效）。
            failures.append(f"{meeting}: 不在 {PROBE_RESULTS.name} 的参照里，"
                            "本场未做任何门限自检")
        for stage, key in (("main", "n_windows"), ("refine", "n_dense")):
            want = ref.get(key)
            got = counts[stage]
            if want is None:
                continue
            if got != want:
                failures.append(f"{meeting} {stage}: 缓存 {got} 窗 != 探针 {want} 窗")

        manifest["meetings"][meeting] = {
            "counts": counts,
            "probe_n_windows": ref.get("n_windows"),
            "probe_n_dense": ref.get("n_dense"),
            "vad_intervals": [[round(a, 4), round(b, 4)] for a, b in vad_segments],
        }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT}/manifest.json")

    if failures:
        print("\n★ 门限自检失败（缓存的窗集与生产路径不一致）：")
        for line in failures:
            print(f"  - {line}")
        raise SystemExit(1)
    print("门限自检通过：主窗 == probe n_windows，细化窗 == probe n_dense")


if __name__ == "__main__":
    main()
