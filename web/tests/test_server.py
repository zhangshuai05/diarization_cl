# -*- coding: utf-8 -*-
"""`web/server.py` 的 HTTP 契约：真 server 线程 + 随机端口 + `http.client`。

不跑真流水线（T1–T4 各有自己的验收）：算法层换成合成泳道，**HTTP 层是真的** ——
分块编码、终止块、keep-alive 连接复用都在真协议上验。终止块之后再写 socket 是
8736 踩过的坑，所以「读完一个流后复用同一条连接再发一个请求」是本文件的固定手法。

音频切片（T7）不在本文件的验收里：`clip_path` 未实现时端点退化成校验回执（JSON），
钳制结果同时挂在 `X-Clamped-Seconds` 头上，T7 换成 `audio/wav` 后这两条断言照样成立。
"""
import http.client
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parents[1]
for _path in (WEB, WEB / "backend"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import contract  # noqa: E402
import server  # noqa: E402

KEYS = contract.LANE_PAYLOAD_KEYS


# ─────────────────────────── 夹具与小工具 ───────────────────────────

@pytest.fixture
def http_server():
    """真 server：`ThreadingHTTPServer` 绑 127.0.0.1:0（随机端口），线程里 serve_forever。"""
    instance = server.Server(("127.0.0.1", 0), server.Handler)
    # poll_interval 调小：默认 0.5 s 的 shutdown 轮询会让每个用例白等半秒
    thread = threading.Thread(target=instance.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def _connect(srv):
    return http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=15)


def request(srv, path, method="GET", connection=None):
    """→ `(status, headers_dict, body: bytes)`；`connection` 给了就复用它（keep-alive）。"""
    own = connection is None
    conn = connection if connection is not None else _connect(srv)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        if own:
            conn.close()


def json_of(srv, path, method="GET"):
    status, headers, body = request(srv, path, method)
    return status, headers, json.loads(body.decode("utf-8"))


def ndjson_of(srv, path, method="GET", connection=None):
    status, headers, body = request(srv, path, method, connection)
    text = body.decode("utf-8")
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    return status, headers, lines


def lane_payload(lane, **overrides):
    """一条满足 `LANE_PAYLOAD_KEYS` 的合成泳道 payload（T3 的输出形状）。"""
    payload = dict.fromkeys(KEYS)
    payload.update({"lane": lane, "k": 2, "level": "2s-only", "overall": 0.5,
                    "merged_overall": 0.4, "segments": [{"start": 0.0, "end": 1.0, "spk": 0}]})
    payload.update(overrides)
    return payload


@pytest.fixture
def fake_meetings(monkeypatch):
    monkeypatch.setattr(server, "meetings", lambda: [{"id": "M1"}, {"id": "M2"}])
    return ["M1", "M2"]


def fake_lanes(monkeypatch, lanes=("base", "unlocked"), calls=None):
    """把算法层换成合成泳道；`calls` 给了就记 `(meeting, knobs)`。"""
    def generator(meeting, knobs):
        if calls is not None:
            calls.append((meeting, dict(knobs)))
        for lane in lanes:
            yield lane_payload(lane)

    monkeypatch.setattr(server, "run_lanes", generator)


CACHE_PRESENT = server.PE.PROBE_CACHE.is_dir()


# ─────────────────────────── /api/meetings ───────────────────────────

@pytest.mark.skipif(not CACHE_PRESENT, reason="探针缓存不在（D:\\ 数据）")
def test_meetings_catalog_fields_and_has_ref(http_server):
    """列表字段齐全；`has_ref` 与参照产物**逐场相等**（不是从 der 推出来的）。"""
    status, _, body = request(http_server, "/api/meetings")
    assert status == 200
    rows = json.loads(body.decode("utf-8"))["meetings"]
    assert rows, "缓存在场就该有场次"
    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)

    expected_keys = {"id", "cohort", "has_ref", "gt_speakers", "duration", "n_windows"}
    for row in rows:
        assert set(row) == expected_keys, row
        assert row["cohort"] in ("dev", "holdout", "train_L")
        assert isinstance(row["has_ref"], bool)
        assert row["gt_speakers"] >= 1 and row["duration"] > 0 and row["n_windows"] > 0

    artifact = json.loads((server.ROOT / "artifacts" / "v13_ab_dev_results.json")
                          .read_text(encoding="utf-8"))
    referenced = {item["meeting"] for item in artifact["results"]}
    ours = {row["id"] for row in rows}
    assert {row["id"] for row in rows if row["has_ref"]} == (referenced & ours)
    assert all(row["cohort"] == "dev" for row in rows if row["has_ref"])


def test_n_windows_prefers_manifest_counts_over_npz(monkeypatch, tmp_path):
    """主源（manifest `meetings.<id>.counts.main`）与兜底源（npz 形状）取值不同时，
    必须返回**主源**的值 —— 只断言 `> 0` 抓不到「主源恒 miss、静默走兜底」。

    桩：manifest 说 777，npz 形状是 4；旧实现在 entry 上直接取 `main`（恒 None），
    会返回 4，本用例必红。
    """
    import numpy as np

    meeting = "M_STUB"
    (tmp_path / meeting).mkdir()
    np.savez(tmp_path / meeting / "main.npz", emb=np.zeros((4, 2)))     # 兜底源 = 4
    (tmp_path / "manifest.json").write_text(
        json.dumps({"meetings": {meeting: {"counts": {"main": 777}}}}), encoding="utf-8")

    monkeypatch.setattr(server, "_CATALOG", None)
    monkeypatch.setattr(server.PE, "PROBE_CACHE", tmp_path)
    rows = server.meetings()
    assert len(rows) == 1 and rows[0]["id"] == meeting
    assert rows[0]["n_windows"] == 777, "读到 4 说明 counts.main 恒 miss、走了 npz 兜底"


# ─────────────────────────── /api/run：参数校验 ───────────────────────────

def test_meeting_whitelist_rejects_traversal_and_unknown(http_server, fake_meetings):
    for path in ("/api/run?meeting=../x", "/api/run?meeting=..%2F..%2Fetc",
                 "/api/run?meeting=NOT_A_MEETING", "/api/run", "/api/run?meeting="):
        status, _, payload = json_of(http_server, path)
        assert status == 400, path
        assert payload["t"] == "error" and payload["error"], path
    status, _, payload = json_of(http_server, "/api/audio?meeting=../x&seconds=60")
    assert status == 400 and payload["t"] == "error"


def test_knobs_reject_out_of_range_without_clamping(http_server, fake_meetings):
    for query in ("lam=0.031", "lam=-0.001", "lam=nan", "lam=inf", "lam=abc",
                  "min_win=1", "min_win=5", "min_win=2.5",
                  "cap=3", "cap=11", "cap=abc",
                  "mode=random", "mode=",
                  "refine=yes", "refine=2", "oracle=0.5", "pin_base=on"):
        status, _, payload = json_of(http_server, "/api/run?meeting=M1&" + query)
        assert status == 400, query
        assert payload["t"] == "error" and payload["error"], query


def test_knobs_are_passed_verbatim(http_server, fake_meetings, monkeypatch):
    calls = []
    fake_lanes(monkeypatch, calls=calls)

    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1&lam=0.03&min_win=4"
                                               "&cap=4&mode=largest_eligible&refine=0"
                                               "&oracle=1&pin_base=false")
    assert status == 200 and lines[-1]["t"] == "done"
    assert calls[-1] == ("M1", {"lam": 0.03, "min_win": 4, "cap": 4,
                                "mode": "largest_eligible", "refine": False,
                                "oracle": True, "pin_base": False})

    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M2&lam=true&min_win=3")
    assert status == 400                     # lam 不是数字：不静默取默认
    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M2&refine=TRUE&pin_base=1")
    assert status == 200 and lines[-1]["t"] == "done"
    assert calls[-1] == ("M2", dict(server.KNOB_DEFAULTS, refine=True, pin_base=True))


# ─────────────────────────── /api/truth ───────────────────────────

def _write_rttm(root, meeting, rows):
    """一行 RTTM：`SPEAKER <meeting> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>`。"""
    target = root / "TextGrid"
    target.mkdir(parents=True, exist_ok=True)
    body = "".join("SPEAKER %s 1 %.3f %.3f <NA> <NA> %s <NA> <NA>\n"
                    % (meeting, start, end - start, speaker)
                    for start, end, speaker in rows)
    (target / (meeting + ".rttm")).write_text(body, encoding="utf-8")


def test_truth_whitelist_is_the_same_as_run(http_server, fake_meetings):
    """`meeting` 走同一套白名单：不在目录里（含路径式取值）一律 400，不读文件。"""
    for path in ("/api/truth?meeting=../x", "/api/truth?meeting=..%2F..%2Fetc",
                 "/api/truth?meeting=NOT_A_MEETING", "/api/truth", "/api/truth?meeting="):
        status, _, payload = json_of(http_server, path)
        assert status == 400, path
        assert payload["t"] == "error" and payload["error"], path
    status, _, payload = json_of(http_server, "/api/truth?meeting=M1", method="POST")
    assert status == 405 and payload["t"] == "error"       # 和 /api/run 一样只 GET


def test_truth_shape_reference_truncated_to_the_run_window(http_server, fake_meetings,
                                                           monkeypatch, tmp_path):
    """形状 = `{reference, gt_k, has_ref}`；段表按 /api/run 的 600 s 窗口截断。"""
    root = tmp_path / "data"
    _write_rttm(root, "M1", [(0.0, 5.0, "S01"), (599.0, 700.0, "S02"),
                             (700.0, 720.0, "S03")])
    monkeypatch.setattr(server, "truth_roots", lambda: (root,))
    monkeypatch.setattr(server, "MEETINGS_ARTIFACT", tmp_path / "absent.json")

    status, _, payload = json_of(http_server, "/api/truth?meeting=M1")
    assert status == 200
    assert set(payload) == {"reference", "gt_k", "has_ref"}
    assert payload["reference"] == [
        {"start": 0.0, "end": 5.0, "speaker": "S01"},
        {"start": 599.0, "end": 600.0, "speaker": "S02"},   # 末端截到 600，不是 700
    ]                                                          # 600 之后开的段整条丢弃
    assert payload["gt_k"] == 2
    assert payload["has_ref"] is False


def test_truth_has_ref_reads_the_results_artifact(http_server, fake_meetings, monkeypatch,
                                                  tmp_path):
    """`has_ref` 与 /api/meetings 同一个源（`v13_ab_dev_results.json` 的 results）。"""
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path,))
    artifact = tmp_path / "ab.json"
    artifact.write_text(json.dumps({"results": [{"meeting": "M1"}]}), encoding="utf-8")
    monkeypatch.setattr(server, "MEETINGS_ARTIFACT", artifact)

    assert json_of(http_server, "/api/truth?meeting=M1")[2]["has_ref"] is True
    assert json_of(http_server, "/api/truth?meeting=M2")[2]["has_ref"] is False
    # 没有 rttm 的场次：空段表 + 0 人，不是 500
    status, _, payload = json_of(http_server, "/api/truth?meeting=M1")
    assert status == 200 and payload["reference"] == []


@pytest.mark.skipif(not CACHE_PRESENT, reason="探针缓存不在（D:\\ 数据）")
def test_truth_reference_equals_the_reference_api_run_uses(http_server):
    """真值轨与 /api/run 的参照**同源同口径**（采样 3 场，逐段逐位相等）。

    两边都读 `TextGrid/<meeting>.rttm` 并截到 600 s；`lanes.load_truth` 是 /api/run
    实际用的那份，这里的断言就是「不许另造一套截断口径」的回归锚。
    """
    import lanes

    for row in server.meetings()[:3]:
        status, _, payload = json_of(http_server, "/api/truth?meeting=" + row["id"])
        assert status == 200, row
        _, reference = lanes.load_truth(row["id"])
        assert payload["reference"] == reference, row
        assert payload["gt_k"] == len({s["speaker"] for s in reference}), row
        assert payload["gt_k"] == row["gt_speakers"], row


# ─────────────────────────── /api/run：lane 事件的 pairs ───────────────────────────

def test_lane_events_carry_pairs_from_the_meeting_reference(http_server, fake_meetings,
                                                            monkeypatch):
    """每条 lane 事件都带 `pairs`：口径 = 8736（`#N` + 最大重叠），不是前端自己算。"""
    fake_lanes(monkeypatch, lanes=("base",))
    monkeypatch.setattr(server, "reference_segments",
                        lambda meeting: [{"start": 0.0, "end": 1.0, "speaker": "S01"},
                                         {"start": 1.0, "end": 2.0, "speaker": "S02"}])

    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1")
    assert status == 200 and lines[-1]["t"] == "done"
    lane = lines[0]
    assert set(contract.LANE_EVENT_KEYS) <= set(lane)        # payload 16 键 + pairs
    assert lane["pairs"] == [{"pred": "#0", "ref": "S01", "overlap": 1.0,
                              "share": 1.0, "pred_seconds": 1.0}]
    assert "pairs" not in lane["metrics"]                    # 读数子对象里不塞行表


def test_lane_pairs_are_empty_without_reference_or_segments(http_server, fake_meetings,
                                                            monkeypatch):
    """无真值（50 场里 38 场）与 k=0（空段表）都要有 `pairs` 键，不是缺字段。"""
    fake_lanes(monkeypatch, lanes=("base",))
    monkeypatch.setattr(server, "reference_segments", lambda meeting: [])
    lane = ndjson_of(http_server, "/api/run?meeting=M1")[2][0]
    # 无真值：每簇一行、`ref` 为 None（页面靠它显示 —），不是空列表
    assert lane["pairs"] == [{"pred": "#0", "ref": None, "overlap": 0.0,
                              "share": 0.0, "pred_seconds": 1.0}]

    empty = lane_payload("base", segments=[])
    assert server.lane_event(empty, {"status": "missing"}, [])["pairs"] == []


def test_lane_event_missing_contract_key_is_still_loud(fake_meetings):
    """`pairs` 由编排层补，但 payload 缺真字段照样要抛（缺字段=bug，Ruling 6）。"""
    payload = lane_payload("base")
    payload.pop("der")
    with pytest.raises(RuntimeError, match="der"):
        server.lane_event(payload, {"status": "missing"}, [])
    event = server.lane_event(lane_payload("base"), {"status": "missing"},
                              [{"start": 0.0, "end": 1.0, "speaker": "S01"}])
    assert event["pairs"][0]["pred"] == "#0"


# ─────────────────────────── /api/audio：seconds 钳制 ───────────────────────────

@pytest.mark.parametrize("raw,expected", [("1e9", 3600.0), ("3601", 3600.0), ("1", 30.0),
                                          ("0", 30.0), ("-5", 30.0), ("600", 600.0),
                                          ("30", 30.0), ("3600", 3600.0)])
def test_audio_seconds_clamped(http_server, fake_meetings, raw, expected):
    status, headers, body = request(http_server, "/api/audio?meeting=M1&seconds=" + raw)
    assert status == 200
    assert headers["X-Clamped-Seconds"] == "%g" % expected
    if headers["Content-Type"].startswith("application/json"):
        assert json.loads(body.decode("utf-8"))["clamped_seconds"] == expected


def test_audio_seconds_default_and_bad_values(http_server, fake_meetings):
    status, headers, _ = request(http_server, "/api/audio?meeting=M1")
    assert status == 200 and headers["X-Clamped-Seconds"] == "600"

    for raw in ("nan", "inf", "-inf", "abc", ""):
        status, _, payload = json_of(http_server, "/api/audio?meeting=M1&seconds=" + raw)
        assert status == 400, raw
        assert payload["t"] == "error" and payload["error"], raw


# ─────────────────────────── /api/run：ndjson 流 ───────────────────────────

def test_run_streams_lane_events_then_done(http_server, fake_meetings, monkeypatch):
    fake_lanes(monkeypatch)
    status, headers, lines = ndjson_of(http_server, "/api/run?meeting=M1")
    assert status == 200
    assert headers["Content-Type"].startswith("application/x-ndjson")
    assert [line["t"] for line in lines] == ["lane", "lane", "done"]
    assert [line["lane"] for line in lines[:2]] == ["base", "unlocked"]

    parity = server.verify_der.parity_status()
    for line in lines:
        assert line["parity"]["status"] == parity["status"]
    for line in lines[:2]:
        assert set(KEYS) <= set(line)
        assert line["metrics"]["overall"] == 0.5
    assert lines[-1]["n_lanes"] == 2 and "elapsed_s" in lines[-1]


def test_run_error_event_then_terminator_and_connection_reuse(http_server, fake_meetings,
                                                              monkeypatch):
    """泳道中途抛异常：推 `error`、写终止块，**之后不再写 socket**（复用连接验证）。"""
    def exploding(meeting, knobs):
        yield lane_payload("base")
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "run_lanes", exploding)
    connection = _connect(http_server)
    try:
        status, headers, lines = ndjson_of(http_server, "/api/run?meeting=M1",
                                           connection=connection)
        assert status == 200
        assert [line["t"] for line in lines] == ["lane", "error"]
        assert "boom" in lines[-1]["error"]

        # 同一条连接上再发一个请求：上一个响应如果有尾巴，这里会解析错乱
        status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1", connection=connection)
        assert status == 200 and lines[-1]["t"] == "error"
    finally:
        connection.close()


def test_no_bytes_after_chunked_terminator(http_server, fake_meetings, monkeypatch):
    """终止块之后**一个字节都不许再写**（Ruling 3；8736 的坑）。

    用裸 socket + `Connection: close` 读到 EOF：分块帧自己解，终止块之后若有残留
    （keep-alive 下会把下一个响应读错），这里就直接断言到字节。
    """
    fake_lanes(monkeypatch, lanes=("base",))
    sock = socket.create_connection(("127.0.0.1", http_server.server_address[1]), timeout=15)
    try:
        sock.sendall(b"GET /api/run?meeting=M1 HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     b"Connection: close\r\n\r\n")
        raw = b""
        while True:
            piece = sock.recv(65536)
            if not piece:
                break
            raw += piece
    finally:
        sock.close()
    head, _, rest = raw.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200"), head[:80]
    frames = []
    while True:
        size_line, _, rest = rest.partition(b"\r\n")
        size = int(size_line, 16)               # 帧头坏了会在这里抛（不是静默放过）
        if size == 0:
            assert rest.startswith(b"\r\n"), "终止块本身不完整：%r" % rest[:20]
            assert rest[2:] == b"", "终止块之后还有 %d 字节：%r" % (len(rest) - 2, rest[2:60])
            break
        frames.append(rest[:size])
        assert rest[size:size + 2] == b"\r\n", "分块帧缺 CRLF"
        rest = rest[size + 2:]
    assert [json.loads(frame) for frame in frames][-1]["t"] == "done"


def test_run_missing_contract_key_is_loud(http_server, fake_meetings, monkeypatch):
    """缺字段=bug：不推半条 lane 事件，推 `error`（Ruling 6）。"""
    def broken(meeting, knobs):
        payload = lane_payload("base")
        payload.pop("der")
        yield payload

    monkeypatch.setattr(server, "run_lanes", broken)
    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1")
    assert status == 200
    assert [line["t"] for line in lines] == ["error"]
    assert "der" in lines[0]["error"] and "契约" in lines[0]["error"]


def test_run_is_serialized_409_then_lock_released(http_server, fake_meetings, monkeypatch):
    started = threading.Event()
    gate = threading.Event()

    def blocking(meeting, knobs):
        started.set()
        yield lane_payload("base")
        gate.wait(10)
        yield lane_payload("unlocked")

    monkeypatch.setattr(server, "run_lanes", blocking)
    first = {}

    def worker():
        first["result"] = request(http_server, "/api/run?meeting=M1")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert started.wait(5), "第一个 run 没起来"
        status, _, payload = json_of(http_server, "/api/run?meeting=M1")
        assert status == 409, payload
        assert payload["t"] == "error" and payload["error"]
    finally:
        gate.set()
        thread.join(timeout=10)
    assert not thread.is_alive()

    status, _, body = first["result"]
    assert status == 200
    lines = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    assert [line["t"] for line in lines] == ["lane", "lane", "done"]

    # 锁已释放：下一条请求能跑完（不被 409 卡住）
    fake_lanes(monkeypatch, lanes=("base",))
    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M2")
    assert status == 200 and lines[-1]["t"] == "done"


def test_run_lock_released_after_failure(http_server, fake_meetings, monkeypatch):
    def exploding(meeting, knobs):
        raise RuntimeError("nope")
        yield  # pragma: no cover —— 只为让函数是生成器

    monkeypatch.setattr(server, "run_lanes", exploding)
    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1")
    assert status == 200 and lines[-1]["t"] == "error"

    fake_lanes(monkeypatch, lanes=("base",))
    status, _, lines = ndjson_of(http_server, "/api/run?meeting=M1")
    assert status == 200 and lines[-1]["t"] == "done"


def test_emit_after_close_is_silent():
    """Ruling 3：`_closed` 置位后 `_emit` 静默返回，**不许碰 socket**。"""
    class Closed:
        _closed = True

        def __getattr__(self, name):
            raise AssertionError("终止块之后还在用 socket：%s" % name)

    assert server.Handler._emit(Closed(), {"t": "lane"}) is False


# ─────────────────────────── /api/verify 与静态 ───────────────────────────

@pytest.mark.parametrize("method", ["GET", "POST"])
def test_verify_returns_parity_status(http_server, method):
    status, _, lines = ndjson_of(http_server, "/api/verify", method=method)
    assert status == 200 and lines[-1]["t"] == "done"
    parity = lines[-1]["parity"]
    assert parity["status"] in ("pass", "fail", "missing")
    assert parity["status"] == server.verify_der.parity_status()["status"]
    if parity["status"] == "missing":
        assert "verify_der" in parity["error"], parity


def test_verify_missing_artifact_says_how_to_generate(http_server, monkeypatch):
    """Ruling 7：产物缺失时**不跑**对拍，回 `missing` + 生成命令。"""
    monkeypatch.setattr(server.verify_der, "parity_status",
                        lambda path=None: {"path": "X/web_der_parity.json", "status": "missing"})
    status, _, lines = ndjson_of(http_server, "/api/verify", method="POST")
    assert status == 200 and lines[-1]["t"] == "done"
    parity = lines[-1]["parity"]
    assert parity["status"] == "missing"
    assert "python -m web.backend.verify_der" in parity["error"]
    assert "X/web_der_parity.json" in parity["error"]


def test_meetings_empty_when_cache_absent(monkeypatch, tmp_path):
    """缓存不在（D 盘未挂）时目录为空，不是一个 500。"""
    monkeypatch.setattr(server, "_CATALOG", None)
    monkeypatch.setattr(server.PE, "PROBE_CACHE", tmp_path)
    assert server.meetings() == []


def test_root_serves_page_when_it_exists_else_404(http_server):
    page = WEB / "index.html"
    status, headers, body = request(http_server, "/")
    if page.is_file():
        assert status == 200 and headers["Content-Type"].startswith("text/html")
        assert body == page.read_bytes()
    else:
        assert status == 404
        assert b"index.html" in body                      # 信息可读，不抛异常


def test_unknown_path_is_404(http_server):
    for path in ("/nope", "/../etc/passwd", "/api/nope"):
        status, _, _ = request(http_server, path)
        assert status == 404, path
