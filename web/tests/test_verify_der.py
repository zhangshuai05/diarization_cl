# -*- coding: utf-8 -*-
"""`verify_der` 的门禁判定、报告 schema 与安全读文件测试。

不碰 D:\\ 真缓存与真参照产物：参照读取与泳道跑法都注入假件（monkeypatch），
真实数据上的对拍由 `python -m web.backend.verify_der` 单独跑一次（耗时写进 T4 报告）。
"""
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import contract  # noqa: E402
import verify_der  # noqa: E402

import claude_a2_v11 as v11  # noqa: E402

DETAIL = ("miss_rate", "false_alarm_rate", "confusion_rate")


def fake_rows(delta_pts, arm="base"):
    """逐项造 `(ours, ref)`：ref 固定 0.2000，ours = ref + delta_pts/100。"""
    return [{"meeting": "M%d" % index, "arm": arm, "ref": 0.2,
             "ours": 0.2 + delta / 100.0}
            for index, delta in enumerate(delta_pts)]


# ───────────────────── compare：阈值判定（严格小于） ─────────────────────

def test_threshold_and_status():
    """brief Step 1 原文：0.05 不算通过（严格小于）。"""
    rep = verify_der.compare(fake_rows(delta_pts=[0.0, 0.04, 0.05]), threshold_pts=0.05,
                             expected=None)      # 纯阈值单测；规模防线另有用例
    assert rep["summary"]["status"] == "fail"
    assert rep["summary"]["n"] == 3 and rep["summary"]["pass"] == 2


def test_all_strictly_below_threshold_passes():
    rep = verify_der.compare(fake_rows([0.0, 0.049, -0.049]), threshold_pts=0.05, expected=None)
    assert rep["summary"]["status"] == "pass"
    assert rep["summary"]["n"] == 3 and rep["summary"]["pass"] == 3
    assert rep["summary"]["max_delta_pts"] == pytest.approx(0.049)


def test_delta_is_signed_in_points_and_absolute_for_judgement():
    rep = verify_der.compare([{"meeting": "M", "arm": "fixed", "ref": 0.30, "ours": 0.3004},
                              {"meeting": "N", "arm": "fixed", "ref": 0.30, "ours": 0.2994}],
                             threshold_pts=0.05, expected=None)
    rows = {row["meeting"]: row for row in rep["rows"]}
    assert rows["M"]["delta_pts"] == pytest.approx(0.04)    # 百分点，带符号（正 = 我们更高）
    assert rows["N"]["delta_pts"] == pytest.approx(-0.06)   # 负号也要算进绝对差
    assert rows["M"]["pass"] is True and rows["N"]["pass"] is False
    failed = rep["summary"]["failed"]
    assert [(item["meeting"], item["arm"], item["reason"]) for item in failed] == \
        [("N", "fixed", "over_threshold")]
    assert failed[0]["delta_pts"] == pytest.approx(-0.06)


def test_missing_numbers_count_as_fail_not_as_zero():
    """`der is None` 不能当 0：没算出来就是没通过，且要说明原因。"""
    rep = verify_der.compare([{"meeting": "M", "arm": "fixed", "ref": None, "ours": 0.1},
                              {"meeting": "N", "arm": "base", "ref": 0.1, "ours": None},
                              {"meeting": "O", "arm": "base", "ref": 0.1, "ours": float("nan")}],
                             threshold_pts=0.05)
    assert rep["summary"]["status"] == "fail"
    assert rep["summary"]["pass"] == 0
    assert rep["summary"]["max_delta_pts"] is None
    assert [row["reason"] for row in rep["rows"]] == ["missing_ref"] * 1 + \
        ["missing_ours"] * 2


def test_empty_rows_is_not_a_pass():
    """没有一项可比时不许报 pass——空对拍不等于对上了；且要说明规模不符。"""
    rep = verify_der.compare([])
    summary = rep["summary"]
    assert (summary["n"], summary["pass"], summary["max_delta_pts"]) == (0, 0, None)
    assert summary["status"] == "fail" and summary["failed"] == []
    assert summary["expected_rows"] == 24
    assert any("规模不符" in problem for problem in summary["shape_problems"])


def test_short_reference_fails_on_scale_even_when_every_row_matches():
    """终审 D：参照被裁小（2 行、逐项都在阈值内）也必须红——"跑得少 ⇒ 过得多"不许静默。"""
    rep = verify_der.compare(fake_rows([0.0, 0.0]))
    assert (rep["summary"]["n"], rep["summary"]["pass"]) == (2, 2)
    assert rep["summary"]["status"] == "fail"
    assert any("规模不符：期望 24 行" in problem for problem in rep["summary"]["shape_problems"])
    assert any("场次不符" in problem for problem in rep["summary"]["shape_problems"])


def test_full_scale_rows_pass_without_shape_problems():
    rows = [{"meeting": "M%02d" % index, "arm": arm, "ref": 0.2, "ours": 0.2}
            for index in range(12) for arm in ("base", "fixed")]
    rep = verify_der.compare(rows)
    assert rep["summary"]["status"] == "pass"
    assert rep["summary"]["expected_rows"] == 24 and rep["summary"]["shape_problems"] == []


def test_meeting_with_a_missing_arm_fails():
    """每场两臂必须齐：只有 base 的那场要在 shape_problems 里点名。"""
    rows = [{"meeting": "M%02d" % index, "arm": arm, "ref": 0.2, "ours": 0.2}
            for index in range(12) for arm in ("base", "fixed")
            if (index, arm) != (7, "fixed")]
    rep = verify_der.compare(rows)
    assert (rep["summary"]["n"], rep["summary"]["pass"]) == (23, 23)
    assert rep["summary"]["status"] == "fail"
    assert rep["summary"]["shape_problems"] == ["规模不符：期望 24 行（12 场 × base/fixed 两臂），"
                                                "实得 23 行", "M07 缺臂：fixed"]


def test_rows_are_not_mutated():
    rows = fake_rows([0.0])
    verify_der.compare(rows, threshold_pts=0.05)
    assert set(rows[0]) == {"meeting", "arm", "ref", "ours"}


# ───────────────────── parity_status：页面安全消费 ─────────────────────

def test_missing_file_reports_missing(tmp_path):
    """brief Step 1 原文。"""
    assert verify_der.parity_status(tmp_path / "nope.json")["status"] == "missing"


def test_missing_directory_reports_missing(tmp_path):
    assert verify_der.parity_status(tmp_path / "no_dir" / "x.json")["status"] == "missing"


def test_broken_json_reports_fail_without_raising(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{\"schema\": ", encoding="utf-8")
    status = verify_der.parity_status(path)
    assert status["status"] == "fail"
    assert "error" in status and status["path"] == str(path)


def test_json_without_summary_reports_fail(tmp_path):
    path = tmp_path / "nosummary.json"
    path.write_text(json.dumps({"schema": "web-der-parity-v1"}, ensure_ascii=False),
                    encoding="utf-8")
    assert verify_der.parity_status(path)["status"] == "fail"


def test_failed_report_reports_fail_and_lists_rows(tmp_path):
    path = tmp_path / "fail.json"
    path.write_text(json.dumps({
        "schema": "web-der-parity-v1", "checked_at": "2026-10-10T00:00:00",
        "threshold_pts": 0.05,
        "rows": [{"meeting": "M", "arm": "fixed", "ours": 0.1, "ref": 0.2,
                  "delta_pts": -10.0, "pass": False}],
        "summary": {"n": 1, "pass": 0, "max_delta_pts": 10.0, "status": "fail",
                    "failed": [{"meeting": "M", "arm": "fixed", "delta_pts": -10.0,
                                "reason": "over_threshold"}]},
    }, ensure_ascii=False), encoding="utf-8")
    status = verify_der.parity_status(path)
    assert status["status"] == "fail"
    assert status["n_total"] == 1 and status["n_pass"] == 0
    assert status["failed"] == [{"meeting": "M", "arm": "fixed", "delta_pts": -10.0,
                                 "reason": "over_threshold"}]


def test_passing_report_reports_counts_and_date(tmp_path):
    path = tmp_path / "pass.json"
    path.write_text(json.dumps({
        "schema": "web-der-parity-v1", "checked_at": "2026-10-10T00:00:00",
        "threshold_pts": 0.05, "rows": [],
        "summary": {"n": 24, "pass": 24, "max_delta_pts": 0.0, "status": "pass",
                    "failed": []},
    }, ensure_ascii=False), encoding="utf-8")
    status = verify_der.parity_status(path)
    assert status["status"] == "pass"
    assert (status["n_total"], status["n_pass"]) == (24, 24)
    assert status["checked_at"] == "2026-10-10T00:00:00"
    assert status["threshold_pts"] == 0.05


def test_status_default_path_is_the_artifact():
    """页面不带参数调用时读的就是交付产物那一份。"""
    status = verify_der.parity_status()
    assert status["status"] in ("missing", "fail", "pass")
    assert status["path"] == str(verify_der.OUT_PATH)


# ───────────────────── parity_report：臂、口径、落盘 ─────────────────────

def _payload(lane, der, k=2, level="refined", n_segments=50):
    values = {key: 0.1 for key in contract.LANE_PAYLOAD_KEYS}
    values.update({"lane": lane, "k": k, "level": level, "der": der,
                   "miss_rate": 0.01, "false_alarm_rate": 0.02, "confusion_rate": 0.07,
                   "segments": [{"start": float(i), "end": float(i) + 0.5, "spk": 0,
                                 "n_windows": 3, "margin": 0.1}
                                for i in range(n_segments)]})
    return {key: values[key] for key in contract.LANE_PAYLOAD_KEYS}


def fake_ref_doc(deltas_pts=None, n_meetings=12):
    """假参照：`deltas_pts[(meeting, arm)]` 给出该臂要比真值差多少点。"""
    deltas_pts = deltas_pts or {}
    rows = []
    for index in range(n_meetings):
        meeting = "M%02d" % index
        record = {"meeting": meeting, "eff_k": 3}
        for arm in ("base", "fixed"):
            ref = 0.2000
            record[arm] = {"der": ref, "miss_rate": 0.01, "false_alarm_rate": 0.02,
                           "confusion_rate": 0.07, "k": 2, "level": "refined",
                           "n_segments": 50}
        rows.append(record)
    return {"schema": "v13-e2e-unlocked2-v1", "split": "dev", "min_win": 2, "cap": 10,
            "mode": "objective", "rows": rows}


def _install_fakes(monkeypatch, ref_path, deltas_pts=None):
    """把参照读取与泳道跑法换成假件；返回假件收到的调用记录。"""
    seen = []
    deltas_pts = deltas_pts or {}

    def fake_lanes(meeting):
        seen.append(meeting)
        out = {}
        for lane, arm in (("base", "base"), ("unlocked", "fixed")):
            der = 0.2 + deltas_pts.get((meeting, arm), 0.0) / 100.0
            out[lane] = _payload(lane, der)
        return out

    ref_path.write_text(json.dumps(fake_ref_doc(deltas_pts), ensure_ascii=False),
                        encoding="utf-8")
    monkeypatch.setattr(verify_der, "_lanes_for_meeting", fake_lanes)
    return seen


def test_report_schema_arms_and_written_file(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path)

    report = verify_der.parity_report(ref_path, out_path)

    assert report["schema"] == "web-der-parity-v1"
    assert report["threshold_pts"] == 0.05
    assert report["reference_truncation_s"] == 600.0
    assert verify_der.SEGMENT_CONVENTION in report["segment_convention"]
    # 分臂 λ（裁定 1）：base = 运行时的生产常量（现为 0.005），fixed = 参照产物口径 0.015。
    assert report["arms"]["base"]["lam"] == v11.MULTI_K_PENALTY == 0.005
    assert report["arms"]["base"]["pinned"] is True
    assert report["arms"]["fixed"]["lam"] == 0.015
    assert report["arms"]["fixed"]["note"] == "参照产物口径"
    assert report["arms"]["fixed"]["min_win"] == 2
    assert report["arms"]["fixed"]["mode"] == "objective"
    assert report["elapsed_s"] >= 0.0
    assert json.loads(out_path.read_text(encoding="utf-8")) == report
    assert verify_der.parity_status(out_path)["status"] == "pass"


def test_report_runs_twelve_meetings_times_two_arms(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    seen = _install_fakes(monkeypatch, ref_path)

    report = verify_der.parity_report(ref_path, out_path)

    assert seen == ["M%02d" % i for i in range(12)]          # 一场只跑一次（两条臂一次产出）
    assert [(row["meeting"], row["arm"]) for row in report["rows"]] == [
        (meeting, arm) for meeting in seen for arm in ("base", "fixed")]
    assert report["summary"] == {"n": 24, "pass": 24, "max_delta_pts": 0.0,
                                 "expected_rows": 24, "status": "pass", "failed": [],
                                 "shape_problems": []}
    assert all(row["lap_s"] >= 0.0 for row in report["rows"])


def test_report_row_carries_per_component_delta_and_segment_counts(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path)

    report = verify_der.parity_report(ref_path, out_path)

    row = report["rows"][0]
    assert (row["ours"], row["ref"]) == (0.2, 0.2)
    assert row["ours_k"] == row["ref_k"] == 2
    assert row["ours_level"] == row["ref_level"] == "refined"
    assert (row["ours_n_segments"], row["ref_n_segments"]) == (50, 50)
    assert set(row["delta_detail_pts"]) == {"der"} | set(DETAIL)
    assert all(abs(value) < 1e-9 for value in row["delta_detail_pts"].values())
    assert report["segment_convention_evidence"] == {"n_rows": 24,
                                                     "n_rows_with_matching_segment_count": 24}


def test_report_fails_and_names_the_offending_arm(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path, deltas_pts={("M03", "fixed"): 0.4,
                                                      ("M07", "base"): -0.2})

    report = verify_der.parity_report(ref_path, out_path)

    assert report["summary"]["status"] == "fail"
    assert report["summary"]["n"] == 24 and report["summary"]["pass"] == 22
    assert report["summary"]["max_delta_pts"] == pytest.approx(0.4)
    assert report["summary"]["failed"] == [
        {"meeting": "M03", "arm": "fixed", "delta_pts": pytest.approx(0.4),
         "reason": "over_threshold"},
        {"meeting": "M07", "arm": "base", "delta_pts": pytest.approx(-0.2),
         "reason": "over_threshold"}]
    # 挂在盘上的那份同样是 fail（页面据此拒绝渲染 DER）
    assert verify_der.parity_status(out_path)["status"] == "fail"


def test_report_without_reference_rows_fails_instead_of_passing(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    ref_path.write_text(json.dumps({"schema": "v13-e2e-unlocked2-v1", "rows": []},
                                   ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(verify_der, "_lanes_for_meeting",
                        lambda meeting: pytest.fail("没有参照行时不该跑泳道"))

    report = verify_der.parity_report(ref_path, out_path)

    assert report["rows"] == [] and report["summary"]["status"] == "fail"


def test_reference_row_missing_an_arm_fails_instead_of_raising(monkeypatch, tmp_path):
    """终审 D 次生问题：参照行缺 `fixed` 键时不许 KeyError 崩栈（盘上旧 pass 会因此留在原地）。"""
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path)
    doc = fake_ref_doc()
    del doc["rows"][3]["fixed"]
    ref_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    report = verify_der.parity_report(ref_path, out_path)      # 不许抛 KeyError

    row = next(item for item in report["rows"]
               if (item["meeting"], item["arm"]) == ("M03", "fixed"))
    assert row["ref"] is None and row["pass"] is False
    assert report["summary"]["status"] == "fail"
    assert report["summary"]["failed"] == [{"meeting": "M03", "arm": "fixed",
                                             "delta_pts": None, "reason": "missing_ref"}]
    assert verify_der.parity_status(out_path)["status"] == "fail"   # 盘上那份也是 fail


def test_status_is_not_stale_for_a_report_written_against_the_current_reference(monkeypatch, tmp_path):
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path)

    report = verify_der.parity_report(ref_path, out_path)

    status = verify_der.parity_status(out_path)
    assert status["status"] == "pass" and status["stale"] is False
    assert status["reference_sha256"] == report["reference"]["sha256"]


def test_status_flags_a_pass_report_as_stale_once_the_reference_changed(monkeypatch, tmp_path):
    """参照产物变了（那行刷新 / 手工手术）之后，盘上的旧 pass 不许再冒充凭据。"""
    ref_path, out_path = tmp_path / "ref.json", tmp_path / "out.json"
    _install_fakes(monkeypatch, ref_path)
    report = verify_der.parity_report(ref_path, out_path)

    doc = fake_ref_doc()
    doc["rows"][0]["base"]["der"] = 0.5          # 只改参照，报告不动
    ref_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    status = verify_der.parity_status(out_path)
    assert status["status"] == "pass"             # 报告本身还是那次对拍的结论
    assert status["stale"] is True
    assert status["reference_sha256"] == report["reference"]["sha256"]


def test_status_flags_an_old_report_without_fingerprint_as_stale(tmp_path):
    """没有 `reference.sha256` 的旧报告没法证明自己是当前凭据 ⇒ stale。"""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "schema": "web-der-parity-v1", "checked_at": "2026-10-10T00:00:00",
        "threshold_pts": 0.05,
        "summary": {"n": 24, "pass": 24, "max_delta_pts": 0.0, "status": "pass",
                    "failed": []},
    }, ensure_ascii=False), encoding="utf-8")
    status = verify_der.parity_status(path)
    assert status["status"] == "pass" and status["stale"] is True
