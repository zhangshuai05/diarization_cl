# -*- coding: utf-8 -*-
"""`unlocked_selector` 的**句级主指标**（真实路径口径，零推理）。

为什么需要它：unlocked 的收益此前**只用 DER 说过**（`probe_e2e_unlocked2`），
而项目主指标是句级归属率。两把尺子对"多出来的簇"记账方式不同（DER 记为 FA，
主指标走匈牙利映射），**不能互相换算**，必须实测。

为什么便宜：`probe_path_equiv` 已验证「缓存 embedding + `real_path_labels`」
与 `v12.diarize()` 在句级**逐位一致**（26 个对比 `n_sentence_diff` 全 0）。
走真模型是 **21 分钟/场**，读缓存是秒级。

同时跑 base（生产 v12 选择器）与 `min_win` 多档，回答两件事：
  * 主指标上 unlocked 到底赚不赚；
  * `min_win=2` 是**全部 20 场选出来的**（in-sample），相邻档位是否同样好。

    python scripts/evaluation/probe_unlocked_sentence.py --min-wins 2,3,4
    python scripts/evaluation/probe_unlocked_sentence.py --selfcheck   # 对齐真模型读数
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "streaming_asr"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import claude_a2_v11 as v11  # noqa: E402
import claude_a2_v12 as v12  # noqa: E402

import probe_ab_dev as AB  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
from v12_aishell4 import split_meetings  # noqa: E402

from unlocked_selector import CAP, build_unlocked, pick_unlocked  # noqa: E402

E2E = ROOT / "artifacts" / "v13_e2e_unlocked2_dev_mw2_objective.json"

# 那个产物提交于 2026-09-24（134c254），**早于** K 上限修复 977fc6b（09-28），
# 此后**从未重跑**；而 977fc6b 自己写明「此前 v12 生产路径上的所有指标作废，需重跑」。
# 受影响的只有 base 臂的 `S_R004S01C01`（修复前 k=4，现 k=5）。
# fixed 臂不受影响：`unlocked()` 把 `_candidate_clusters` 整个替换了，
# 不经过那个漏传 `max_speakers` 的调用点。
# TODO(基线复跑后): 删除该白名单；复跑后的正确 base k 不应再被视为 stale。
STALE_BASE_K = {"S_R004S01C01": 4}


def candidates_for(E, min_win, mode, penalty):
    """`None` 表示 base（生产选择器）；否则返回 unlocked 选中的候选。

    `penalty` 只作用在 unlocked 这条支线上：base 必须恒用生产值，
    否则基线会被扫描值改掉，前后两腿不可比。
    """
    if min_win is None:
        cands, _ = v11._candidate_clusters(E, max_speakers=v12.MAX_SPEAKERS)
        selected = v11._select_candidate(cands)
        return {"k": int(selected["k"]), "labels": selected["labels"]}
    # `build_unlocked` 读的是模块变量 `v11.MULTI_K_PENALTY`（调用时求值）。
    saved = v11.MULTI_K_PENALTY
    v11.MULTI_K_PENALTY = penalty
    try:
        return pick_unlocked(build_unlocked(E, CAP, min_win), mode=mode)
    finally:
        v11.MULTI_K_PENALTY = saved


def run_meeting(meeting, min_wins, mode, penalty):
    E, times = PE.cached_stage(meeting, "main")
    rE, rtimes = PE.cached_stage(meeting, "refine")
    if len(E) < 2:
        return None
    sentences = AB.load_sentences(meeting)
    if not sentences:
        return None
    out = {}
    for mw in [None] + list(min_wins):
        chosen = candidates_for(E, mw, mode, penalty)
        labels = chosen["labels"]
        flabels, ftimes, level = AB.real_path_labels(np.asarray(labels), E, times, rE, rtimes)
        out["base" if mw is None else "mw%d" % mw] = {
            "k": int(chosen["k"]), "level": level,
            "n_clusters_after_refine": int(len(set(int(x) for x in flabels))),
            **AB.score(sentences, flabels, ftimes, AB._win(level))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--min-wins", default="2,3,4")
    ap.add_argument("--mode", default="objective",
                    choices=("objective", "largest_eligible", "largest"))
    ap.add_argument("--penalty", type=float, default=None,
                    help="MULTI_K_PENALTY 扫描；默认=生产值（只影响 unlocked 支线）")
    ap.add_argument("--selfcheck", action="store_true",
                    help="与真模型产物 v13_e2e_unlocked2_dev_mw2_objective.json 对齐 k")
    ap.add_argument("--root", default=None,
                    help="数据根（真值 TextGrid 所在）。train_L 等另一个 cohort 用这个；"
                         "需先用 build_probe_cache.py --root 建好窗缓存")
    ap.add_argument("--meetings", nargs="*", default=None,
                    help="显式场次列表。train_L 的场次名带日期前缀，"
                         "`split_meetings` 的分组函数会抛异常，必须走这条")
    ap.add_argument("--tag", default=None,
                    help="产物文件名里的标记；默认 = split（给了 --root 时用 root 目录名）")
    args = ap.parse_args()

    default_penalty = v11.MULTI_K_PENALTY
    penalty = default_penalty if args.penalty is None else args.penalty
    min_wins = [int(x) for x in args.min_wins.split(",") if x.strip()]
    if args.root:
        # `probe_ab_dev` 的真值根也在这里读（`load_sentences`）。
        AB.DATA = Path(args.root)
    tag = args.tag or (Path(args.root).name if args.root else args.split)
    meetings = args.meetings or split_meetings(AB.DATA, args.split)
    keys = ["base"] + ["mw%d" % w for w in min_wins]

    rows, skipped = {}, []
    for meeting in meetings:
        try:
            r = run_meeting(meeting, min_wins, args.mode, penalty)
        except FileNotFoundError as exc:      # 缺缓存 → 显式记录，不静默跳过
            skipped.append("%s: %s" % (meeting, exc))
            continue
        if r is None:
            skipped.append("%s: 窗数或句子数不足" % meeting)
            continue
        rows[meeting] = r

    if not rows:
        print("无可用场次：%s" % skipped)
        return

    print("mode=%s  cohort=%s  场次=%d  penalty=%g（生产 %g）  真路径口径（缓存重建，句级主指标）"
          % (args.mode, tag, len(rows), penalty, default_penalty))
    head = "%-16s" % "会议"
    for k in keys:
        head += "│ %-17s" % (k + " k/主指标")
    print(head + "│ Δmw2−base")
    for meeting in sorted(rows):
        r = rows[meeting]
        line = "%-16s" % meeting
        for k in keys:
            c = r[k]
            line += "│ k=%-2d %5.1f%% 弃%4.1f" % (
                c["k"], 100 * c["per_sentence"], 100 * c["abstain"])
        d = r["mw2"]["per_sentence"] - r["base"]["per_sentence"]
        print(line + "│ %+7.1f pts" % (100 * d))

    print()
    for k in keys:
        acc = np.mean([r[k]["per_sentence"] for r in rows.values()])
        levels = Counter(r[k]["level"] for r in rows.values())
        print("  %-6s 主指标均值 %.1f%%  弃权 %.1f%%  level=%s"
              % (k, 100 * acc, 100 * np.mean([r[k]["abstain"] for r in rows.values()]),
                 dict(levels)))
    base = np.mean([r["base"]["per_sentence"] for r in rows.values()])
    for k in keys[1:]:
        cur = np.mean([r[k]["per_sentence"] for r in rows.values()])
        diffs = [r[k]["per_sentence"] - r["base"]["per_sentence"] for r in rows.values()]
        print("  %-6s vs base: %+.1f pts  改善 %d / 恶化 %d / 持平 %d  (逐场Δ中位 %+.1f)"
              % (k, 100 * (cur - base),
                 sum(1 for d in diffs if d > 0.001), sum(1 for d in diffs if d < -0.001),
                 sum(1 for d in diffs if abs(d) <= 0.001), 100 * float(np.median(diffs))))

    if args.selfcheck and penalty != default_penalty:
        print("\n  selfcheck 跳过：penalty=%g 不是生产值，k 本就应该不同。" % penalty)
    if args.selfcheck and penalty == default_penalty:
        # 独立来源：真模型产物。缓存路径若复现不出它的 k，本脚本的口径就不成立。
        real = {r["meeting"]: r for r in json.loads(E2E.read_text(encoding="utf-8"))["rows"]}
        bad, stale = [], []
        for meeting, r in rows.items():
            if meeting not in real:
                continue
            if r["base"]["k"] != real[meeting]["base"]["k"]:
                (stale if STALE_BASE_K.get(meeting) == real[meeting]["base"]["k"]
                 else bad).append("%s base k %d≠%d" % (meeting, r["base"]["k"], real[meeting]["base"]["k"]))
            if "mw2" in r and r["mw2"]["k"] != real[meeting]["fixed"]["k"]:
                bad.append("%s mw2 k %d≠%d" % (meeting, r["mw2"]["k"], real[meeting]["fixed"]["k"]))
        print("\n  selfcheck vs 真模型产物: %s"
              % ("全部一致 ✓" if not bad else "不一致 " + "; ".join(bad)))
        if stale:
            print("  （已知陈旧，非本脚本问题：%s —— 产物未按 977fc6b 的要求重跑）"
                  % "; ".join(stale))

    penalty_suffix = "_penalty%g" % penalty
    out = ROOT / "artifacts" / ("v13_unlocked_sentence_%s%s.json" % (tag, penalty_suffix))
    out.write_text(json.dumps({"schema": "v13-unlocked-sentence-v1", "split": args.split,
                               "penalty": penalty, "penalty_source":
                               "default" if args.penalty is None else "explicit",
                               "cohort_tag": tag, "meetings": list(rows),
                               "mode": args.mode, "min_wins": min_wins, "cap": CAP,
                                "path": "cache+real_path_labels(句级已验证≡diarize)",
                                "production_penalty": default_penalty,
                               "skipped": skipped, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print("写入 %s" % out)
    if skipped:
        print("⚠ 跳过 %d 场: %s" % (len(skipped), skipped))


if __name__ == "__main__":
    main()
