# -*- coding: utf-8 -*-
"""DER 对拍门禁：把"缓存路径算出的 DER"钉死在"真模型路径的 DER"上。

对 dev 12 场跑两条臂，与 `artifacts/v13_e2e_unlocked2_dev_mw2_objective.json`
（真模型端到端产物）逐场比 DER，写 `artifacts/web_der_parity.json`：

* `base`  —— 生产选 K（λ = **运行时的** `v11.MULTI_K_PENALTY`，现为 0.005）。
* `fixed` —— 参照产物那一臂：`unlocked + min_win=2 + cap=10 + mode=objective`，
  λ **钉 0.015**（参照产物生成时的生产值）。

## 为什么分臂钉 λ（本门禁最容易假红的一处）

λ 是"多切一人罚多少"的常数。参照产物生成时生产值是 0.015，本线把生产值降到 0.005
（`claude_a2_v11.py` 有依据与限定）。两条臂读的是**同一个常量**，若两边都用 0.005，
`L_R003S03C02` 的 unlocked 会从参照的 k=3 跳到 k=6（实测 DER 0.1371 vs 参照 0.3395），
门禁必然假红。所以 base 钉运行时生产值、fixed 钉参照口径，并把两者的 λ 写进报告
`arms` 字段——**读数挂在哪个 λ 上，报告就得说出那个数**。

## 段表口径（判别式，不是猜的）

参照产物走的是 `probe_e2e_unlocked2.score`：`v12.diarize(...)["segments"]`，即
`run_claude_lane` 的 `SLIVER_WINDOWS`（<=2 窗）过滤**之前**的那一份。二分验证
（`L_R003S01C02`，base 臂）：

| 口径 | n_segments | DER |
|---|---|---|
| 过滤前（本门禁） | 68 | 0.7153 |
| `run_claude_lane` 过滤后 | 46 | 0.7546 |
| 参照产物 | 68 | 0.7153 |

差 3.9 点 ⇒ 过滤后口径会让 24 项里一片假红。`SEGMENT_CONVENTION` 记的就是这个选择，
每行的 `ours_n_segments` vs `ref_n_segments` 是它挂在产物里的证据。

参考段一律截到 600 s（`end = min(end, 600.0)`，由 `lanes.load_truth` 保证，与探针
`probe_e2e_unlocked2.main` 同口径）。不截断时 `L_R003S01C02` 差 0.08 点，刚好越过阈值。

## 判定

逐项 `|ours - ref| < 0.05` **个百分点**（严格小于）记 pass；24 项全 pass ⇒
`summary.status == "pass"`。阈值只在 `compare` 的参数里，**没有**为了对上而放宽的余地：
对不上就红，并把 `delta_detail_pts`（der/miss/FA/confusion 各自差多少点）逐行留下。

**规模也是判定的一部分**（spec §4.4）：`compare` 要求恰好 `EXPECTED_ROWS = 12 场 × 2 臂`
行、每场两臂齐全，不齐就 `status="fail"`，可读原因写进 `summary["shape_problems"]`——
参照产物被裁小/手术过时「跑得少 ⇒ 过得多」必须红，不许静默变成 `2/2 pass`。参照行缺
`base`/`fixed` 键也只记 fail：原来会 `KeyError` 崩栈退出，把盘上旧的 `pass` 留在原地冒充凭据。

`parity_status()` 是给页面用的**只读**入口：缺文件 / 坏 JSON / `status != "pass"`
一律返回 `{"status": "missing"|"fail", ...}`，**不抛异常**——页面拿不到对拍就画不了
DER，那是显示问题，不是崩溃理由。返回值另带 `stale`：报告记录的参照产物内容指纹
（`reference.sha256`）与当前参照对不上（或报告太旧、没记指纹）即 `true`——盘上的 pass
必须能说清「对着哪一版参照算的」才算凭据。

    python -m web.backend.verify_der          # 跑一次真实对拍（dev 12 场 × 2 臂）
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]

# 与 cache_path/lanes 同一套 `sys.path` 兜底：本模块可被 server 以 web/ 为根导入，
# 也可被测试以 backend 为根导入，或 `python -m web.backend.verify_der` 直接跑。
for _path in (_HERE, ROOT / "streaming_asr", ROOT / "scripts" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import lanes as L  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402

SCHEMA = "web-der-parity-v1"

# 逐项阈值：百分点，**严格小于**（0.05 本身不算通过）。
THRESHOLD_PTS = 0.05

# 期望规模（spec §4.4：dev 12 场 × base/fixed 两臂）——判定的一部分，见模块 docstring。
EXPECTED_MEETINGS = 12
EXPECTED_ARMS = ("base", "fixed")
EXPECTED_ROWS = EXPECTED_MEETINGS * len(EXPECTED_ARMS)

REF_PATH = ROOT / "artifacts" / "v13_e2e_unlocked2_dev_mw2_objective.json"
OUT_PATH = ROOT / "artifacts" / "web_der_parity.json"

# 参照产物那一臂的旋钮（`probe_e2e_unlocked2` 的默认值 + 生成时的生产 λ）。
FIXED_KNOBS = {"lam": 0.015, "min_win": 2, "cap": 10, "mode": "objective",
               "refine": True, "oracle": False, "pin_base": True}

# 参照产物用的段表：`v12.diarize` 的原始段（`run_claude_lane` 过滤前）。判别式见模块
# docstring；改这个常量等于换口径，必须先重跑一次对拍再改。
SEGMENT_CONVENTION = "v12.diarize raw segments (before run_claude_lane SLIVER_WINDOWS filter)"
SEGMENT_CONVENTION_NOTE = (
    "二进制判别（L_R003S01C02 base 臂）：过滤前 68 段/DER 0.7153 = 参照 0.7153；"
    "过滤后 46 段/DER 0.7546，差 3.9 点。参照侧由 probe_e2e_unlocked2.score 直接吃 "
    "v12.diarize(...)[\"segments\"]，即过滤前那一份。")
REFERENCE_TRUNCATION_S = 600.0   # 参考段截断（lanes.load_truth 与探针同口径）

DETAIL_KEYS = ("der", "miss_rate", "false_alarm_rate", "confusion_rate")


# ───────────────────── 判定（纯函数） ─────────────────────

def compare(rows, threshold_pts=THRESHOLD_PTS,
            expected=(EXPECTED_MEETINGS, EXPECTED_ARMS)):
    """逐行判 `|ours - ref| < threshold_pts`（百分点，严格小于），并断言期望规模。

    `rows` 里每项要有 `meeting` / `arm` / `ours` / `ref`（DER 取 0–1 小数）；
    `ours`/`ref` 缺失或非有限值时记 fail（原因写进 `reason`）——**不当 0 处理**。
    `expected=(场数, 臂)` 是规模防线：行数 / 场数 / 每场臂不齐一律 fail，原因写进
    `summary["shape_problems"]`（`expected=None` 只留给纯阈值单测）。
    不修改入参，返回 `{"rows": [...], "summary": {...}}`。
    """
    out = []
    for row in rows:
        ours, ref = row.get("ours"), row.get("ref")
        entry = {"meeting": row["meeting"], "arm": row["arm"], "ours": ours, "ref": ref}
        if not _finite(ref):
            entry.update({"delta_pts": None, "pass": False, "reason": "missing_ref"})
        elif not _finite(ours):
            entry.update({"delta_pts": None, "pass": False, "reason": "missing_ours"})
        else:
            delta = (ours - ref) * 100.0      # 百分点，带符号（正 = 我们更高）
            passed = abs(delta) < threshold_pts
            entry.update({"delta_pts": delta, "pass": passed,
                          "reason": "ok" if passed else "over_threshold"})
        out.append(entry)

    deltas = [entry["delta_pts"] for entry in out if entry["delta_pts"] is not None]
    failed = [{"meeting": entry["meeting"], "arm": entry["arm"],
               "delta_pts": entry["delta_pts"], "reason": entry["reason"]}
              for entry in out if not entry["pass"]]
    n_pass = sum(1 for entry in out if entry["pass"])
    shape_problems = _shape_problems(out, expected)
    summary = {
        "n": len(out),
        "pass": n_pass,
        "max_delta_pts": max((abs(value) for value in deltas), default=None),
        "expected_rows": None if expected is None else expected[0] * len(expected[1]),
        # 没有一项可比时不许报 pass（空对拍 != 对上了）；规模不齐时同理（少跑 != 过了）。
        "status": "pass" if out and n_pass == len(out) and not shape_problems else "fail",
        "failed": failed,
        "shape_problems": shape_problems,
    }
    return {"rows": out, "summary": summary}


def _shape_problems(entries, expected):
    """规模防线的可读原因：行数 / 场数 / 每场两臂是否齐全。`expected=None` 即关掉断言。"""
    if expected is None:
        return []
    n_meetings, arms = expected
    problems = []
    if len(entries) != n_meetings * len(arms):
        problems.append("规模不符：期望 %d 行（%d 场 × %s 两臂），实得 %d 行"
                        % (n_meetings * len(arms), n_meetings, "/".join(arms), len(entries)))
    meetings = sorted({entry["meeting"] for entry in entries})
    if len(meetings) != n_meetings:
        problems.append("场次不符：期望 %d 场，实得 %d 场（%s）"
                        % (n_meetings, len(meetings), "、".join(meetings) or "无"))
    for meeting in meetings:
        present = {entry["arm"] for entry in entries if entry["meeting"] == meeting}
        missing = [arm for arm in arms if arm not in present]
        if missing:
            problems.append("%s 缺臂：%s" % (meeting, "/".join(missing)))
    return problems


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        value == value and value not in (float("inf"), float("-inf"))


# ───────────────────── 参照产物 → 逐场/逐臂目标值 ─────────────────────

def _read_ref(ref_path):
    """读参照产物，返回 `(meta, {meeting: row})`。

    JSON 坏了显式抛（靶子本身读不动是调用方的错）；`rows` 空/不是列表则**不抛**——
    那是有载体但没靶子，报告给 `status="fail"`、一项都不跑（由 `compare([])` 判定）。
    """
    path = Path(ref_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    meta = {"path": str(path), "schema": payload.get("schema"),
            "split": payload.get("split"), "min_win": payload.get("min_win"),
            "cap": payload.get("cap"), "mode": payload.get("mode"),
            "sha256": _fingerprint(payload)}
    if not isinstance(rows, list) or not rows:
        return {**meta, "error": "参照产物没有 rows"}, {}
    usable = [row for row in rows if isinstance(row, dict) and isinstance(row.get("meeting"), str)]
    if len(usable) != len(rows):
        meta["error"] = "参照产物有 %d 行不可用（不是对象或缺 meeting）" % (len(rows) - len(usable))
    return meta, {row["meeting"]: row for row in usable}


def _fingerprint(payload):
    """参照产物的内容指纹（sha256 前 16 位）：排序键 + 紧凑分隔符，与换行/排版无关。"""
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _ref_fingerprint(ref_path):
    """盘上参照产物的指纹；不在 / 读不动（坏 JSON、编码、权限）一律给 None。"""
    try:
        payload = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _fingerprint(payload)


def _lanes_for_meeting(meeting):
    """一场的两条臂，一次 `run_lanes` 产出（base 钉生产 λ、unlocked 用 `FIXED_KNOBS` 的 λ）。

    返回 `{lane 名: payload}`；键与 `lanes.run_lanes` 的泳道名一致（`base` / `unlocked`）。
    """
    knob = {**L.KNOB_DEFAULTS, **FIXED_KNOBS}
    return {payload["lane"]: payload for payload in L.run_lanes(meeting, knob)}


def _row(meeting, arm, payload, ref_row, threshold_pts):
    """一条对拍记录：读数 + 段表指纹 + 逐项差（der/miss/FA/confusion，百分点）。

    `ref_row` 缺臂时为 `None`（参照没有该臂的键）：记 `ref=None` 交给 `compare` 判 fail，
    **不抛 KeyError**——门禁崩栈会让盘上旧的 pass 继续冒充凭据。
    """
    ref_row = ref_row if isinstance(ref_row, dict) else {}
    ours, ref = payload.get("der"), ref_row.get("der")
    row = {
        "meeting": meeting, "arm": arm, "lane": payload.get("lane"),
        "ours": ours, "ref": ref,
        "ours_k": payload.get("k"), "ref_k": ref_row.get("k"),
        "ours_level": payload.get("level"), "ref_level": ref_row.get("level"),
        "ours_n_segments": len(payload.get("segments") or []),
        "ref_n_segments": ref_row.get("n_segments"),
    }
    if _finite(ours) and _finite(ref):
        row["delta_pts"] = (ours - ref) * 100.0
        row["pass"] = abs(row["delta_pts"]) < threshold_pts
        row["reason"] = "ok" if row["pass"] else "over_threshold"
        row["delta_detail_pts"] = {key: (payload.get(key) - ref_row.get(key)) * 100.0
                                   for key in DETAIL_KEYS
                                   if _finite(payload.get(key)) and _finite(ref_row.get(key))}
    else:
        row.update({"delta_pts": None, "pass": False,
                    "reason": "missing_ref" if not _finite(ref) else "missing_ours",
                    "delta_detail_pts": {}})
    return row


# ───────────────────── 报告 ─────────────────────

def parity_report(ref_path=None, out_path=None):
    """跑真实对拍并写报告，返回报告 dict。

    参照产物决定跑哪些场次（它的 rows 即 dev 12 场），一场跑一次两条臂。
    """
    ref_path = Path(ref_path) if ref_path is not None else REF_PATH
    out_path = Path(out_path) if out_path is not None else OUT_PATH
    ref_meta, ref_rows = _read_ref(ref_path)

    started = time.perf_counter()
    rows = []
    for meeting in ref_rows:
        lap = time.perf_counter()
        lanes = _lanes_for_meeting(meeting)
        lap_s = time.perf_counter() - lap
        for arm, lane in (("base", "base"), ("fixed", "unlocked")):
            row = _row(meeting, arm, lanes[lane], ref_rows[meeting].get(arm), THRESHOLD_PTS)
            row["lap_s"] = lap_s
            rows.append(row)
        print("  %-16s base k=%-2s der=%-8s | fixed k=%-2s der=%-8s (%.1fs)"
              % (meeting, rows[-2]["ours_k"], _pct(rows[-2]["ours"]),
                 rows[-1]["ours_k"], _pct(rows[-1]["ours"]), lap_s), flush=True)
    elapsed_s = time.perf_counter() - started

    verdict = compare([{"meeting": row["meeting"], "arm": row["arm"],
                        "ours": row["ours"], "ref": row["ref"]} for row in rows],
                      threshold_pts=THRESHOLD_PTS)
    # `compare` 的逐行判定与 `_row` 是同一式子，但以 compare 为准（单一判定实现）。
    for row, entry in zip(rows, verdict["rows"]):
        row.update({key: entry[key] for key in ("delta_pts", "pass", "reason")})

    report = {
        "schema": SCHEMA,
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "threshold_pts": THRESHOLD_PTS,
        "segment_convention": SEGMENT_CONVENTION,
        "segment_convention_note": SEGMENT_CONVENTION_NOTE,
        "segment_convention_evidence": {
            "n_rows": len(rows),
            "n_rows_with_matching_segment_count":
                sum(1 for row in rows if row["ours_n_segments"] == row["ref_n_segments"]),
        },
        "reference_truncation_s": REFERENCE_TRUNCATION_S,
        "arms": {
            # λ 现读现写：base 钉的是**运行时生产值**（pin_base 的语义），不是写死的数。
            "base": {"lane": "base", "lam": round(float(v11.MULTI_K_PENALTY), 6),
                     "pinned": True, "source": "v11.MULTI_K_PENALTY"},
            "fixed": {"lane": "unlocked", "lam": FIXED_KNOBS["lam"],
                      "note": "参照产物口径", "min_win": FIXED_KNOBS["min_win"],
                      "cap": FIXED_KNOBS["cap"], "mode": FIXED_KNOBS["mode"]},
        },
        "reference": ref_meta,
        "elapsed_s": elapsed_s,
        "rows": rows,
        "summary": verdict["summary"],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parity_status(path=None):
    """页面用的只读入口：读 `artifacts/web_der_parity.json`，**任何情况都不抛异常**。

    返回值恒有 `status`：`"pass"`（对拍通过）/ `"fail"`（文件在但对拍没过，或文件读不动）
    / `"missing"`（文件不在）。`stale=True` 表示这份报告不是对着**当前**参照产物算的
    （指纹对不上，或报告太旧、没记指纹）——结论仍然成立，但不能再当这版参照的凭据。
    页面只在 `"pass"` 时渲染 DER 数字，`stale` 另挂警告。
    """
    path = Path(path) if path is not None else OUT_PATH
    base = {"path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {**base, "status": "missing"}
    except Exception as error:                      # 坏 JSON / 目录 / 编码 / 权限
        return {**base, "status": "fail", "error": "%s: %s" % (type(error).__name__, error)}

    summary = payload.get("summary") if isinstance(payload, dict) else None
    status = summary.get("status") if isinstance(summary, dict) else None
    if status not in ("pass", "fail"):
        return {**base, "status": "fail", "error": "报告缺 summary.status"}
    reference = payload.get("reference")
    reference = reference if isinstance(reference, dict) else {}
    recorded = reference.get("sha256")
    current = _ref_fingerprint(reference.get("path") or REF_PATH)
    return {**base, "status": status,
            "checked_at": payload.get("checked_at"),
            "threshold_pts": payload.get("threshold_pts"),
            "n_total": summary.get("n"), "n_pass": summary.get("pass"),
            "failed": summary.get("failed") or [],
            "reference_sha256": recorded,
            # 没记指纹（旧报告）也算 stale：盘上的 pass 得能对上「哪一版参照」才算凭据。
            "stale": recorded is None or recorded != current}


def _pct(value):
    return "—" if not _finite(value) else "%.4f" % value


def main(argv=None):
    ap = argparse.ArgumentParser(description="dev 12 场 × base/fixed 两臂的 DER 对拍门禁")
    ap.add_argument("--ref", default=None, help="参照产物，默认 %s" % REF_PATH.name)
    ap.add_argument("--out", default=None, help="报告输出，默认 %s" % OUT_PATH.name)
    ap.add_argument("--status", action="store_true", help="只读已落盘的报告，不跑对拍")
    args = ap.parse_args(argv)

    if args.status:
        print(json.dumps(parity_status(args.out), ensure_ascii=False, indent=2))
        return 0

    report = parity_report(args.ref, args.out)
    summary = report["summary"]
    print("\n%s  %d/%d 项通过  最大偏差 %s 点  阈值 %.2f 点  耗时 %.1fs"
          % (summary["status"].upper(), summary["pass"], summary["n"],
             "—" if summary["max_delta_pts"] is None else "%.4f" % summary["max_delta_pts"],
             report["threshold_pts"], report["elapsed_s"]))
    for problem in summary.get("shape_problems") or []:
        print("  规模 %s" % problem)
    for entry in summary["failed"]:
        row = next(r for r in report["rows"]
                   if r["meeting"] == entry["meeting"] and r["arm"] == entry["arm"])
        print("  FAIL %-16s %-5s ours=%s ref=%s delta=%s 点 %s"
              % (entry["meeting"], entry["arm"], _pct(row["ours"]), _pct(row["ref"]),
                 "—" if entry["delta_pts"] is None else "%+.4f" % entry["delta_pts"],
                 json.dumps(row["delta_detail_pts"], ensure_ascii=False)))
    print("报告 %s" % (args.out or OUT_PATH))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
