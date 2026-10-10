# -*- coding: utf-8 -*-
"""`/api/audio`（T7）：真音频切片 —— 单声道 16 kHz、取第 0 通道、缓存、失败绝不装 200。

两条数据来源：

* **真实 dev 场次**（`AISHELL4_TEST`，默认 `D:\\yinp\\AISHELL-4\\data\\test`）：验收
  `soundfile.info` → `channels==1`、`samplerate==16000`、`abs(duration-seconds)<0.2`。
  音频不在本机时这些用例**跳过并说明**（不假装通过）。
* **现造的 8 通道 flac**（ffmpeg，只有 c0 有信号）：不依赖 D 盘就能钉死
  `pan=mono|c0=c0` 与 `-ac 1` 的差别、缓存命中、并发原子性、失败清理。
"""
import http.client
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

WEB = Path(__file__).resolve().parents[1]
for _path in (WEB, WEB / "backend"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import server  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
HAVE_FFMPEG = FFMPEG is not None
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 不在 PATH：无法切片（装 ffmpeg 后重来）")


# ─────────────────────────── 夹具与小工具 ───────────────────────────

@pytest.fixture
def http_server():
    """真 server + 随机端口（与 test_server.py 同一手法：HTTP 层是真的）。"""
    instance = server.Server(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=instance.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield instance
    finally:
        instance.shutdown()
        instance.server_close()
        thread.join(timeout=5)


def request(srv, path):
    """→ `(status, headers, body)`；每次一条新连接（音频是整包下载，不需要 keep-alive）。"""
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=60)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


def synth_source(root, meeting, seconds=4.0):
    """`<root>/wav/<meeting>.flac` = 8 通道 flac，**只有 c0** 是 440 Hz 正弦。

    这样 `-ac 1` 的下混 ≈ c0/6~8（RMS 差一个量级），而 `pan=mono|c0=c0` 与 c0 逐位相等
    —— `-ac 1` 与正取的差别在测试里是可判定的，不是靠注释。
    """
    wav_dir = root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    target = wav_dir / (meeting + ".flac")
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi",
                    "-i", "sine=frequency=440:sample_rate=16000:duration=%g" % seconds,
                    "-af", "pan=8c|c0=c0", "-ar", "16000", "-c:a", "flac", str(target)],
                   check=True)
    return target


def rms(samples):
    return float(np.sqrt(np.mean(np.square(samples))))


def dev_meeting():
    """真实一场 dev（本机有 flac）；没有音频数据 → None（调用方 skip 并说明）。"""
    for row in server.meetings():
        if row["cohort"] == "dev" and server.audio_source(row["id"]) is not None:
            return row["id"]
    return None


DEV_MEETING = dev_meeting()          # 导入时算一次（catalog 有缓存，不重复扫目录）
NO_DEV = pytest.mark.skipif(
    DEV_MEETING is None,
    reason="音频数据不可用：%s 下没有 dev 场次的 flac"
           "（设 AISHELL4_TEST 指向 AISHELL-4 test 根，或装好 D:\\ 数据）"
           % (server.truth_roots()[0],))


@pytest.fixture(scope="module")
def real_dev():
    if DEV_MEETING is None:
        pytest.skip("音频数据不可用：%s 下没有 dev 场次的 flac" % (server.truth_roots()[0],))
    return DEV_MEETING


class _Recorder:
    """假 handler：只记 `_file()` 往哪写、写了几块、每块多大（不碰真 socket）。"""

    def __init__(self):
        self.status = None
        self.headers = {}
        self.blocks = []
        self.flushed = False
        self._started = False
        self._closed = False
        self.wfile = SimpleNamespace(write=self.blocks.append,
                                     flush=self._flush)

    def _flush(self):
        self.flushed = True

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers[key] = value

    def end_headers(self):
        pass


# ─────────────────────────── clip_path：产物 ───────────────────────────

def test_clips_dir_is_the_web_cache():
    """缓存落在 `web/clips/`，且 `*.wav` 已被 .gitignore 忽略（产物不入库）。"""
    assert server.CLIP_DIR == WEB / "clips"
    assert "*.wav" in (server.ROOT / ".gitignore").read_text(encoding="utf-8")


@NO_DEV
def test_clip_is_mono_16k_and_seconds_long(real_dev, tmp_path, monkeypatch):
    """真 dev 场次：`soundfile.info` → 单声道、16 kHz、时长 ≈ seconds（±0.2 s）。

    缓存目录指到 tmp：每次都现切一次，防止「上一次的实现留下的缓存」把断言蒙过去。
    """
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")
    for seconds in (30.0, 45.0):
        path = server.clip_path(real_dev, seconds)
        assert path.is_file() and path.parent == server.CLIP_DIR
        info = sf.info(str(path))
        assert info.channels == 1, path
        assert info.samplerate == 16000, path
        assert abs(info.duration - seconds) < 0.2, (path, info.duration, seconds)


@NO_DEV
def test_clip_is_channel_zero_not_the_8ch_downmix(real_dev, tmp_path, monkeypatch):
    """真 dev 场次：切片与源 flac 的第 0 通道**逐样本相等**，且与 `-ac 1` 明显不同。

    8736 的注释点名的坑：8 通道直接下混到单声道是**另一条信号**（别的说话人也混进来了）。
    缓存目录指到 tmp（现切，不吃旧缓存）。
    """
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")
    path = server.clip_path(real_dev, 30.0)
    clip, rate = sf.read(str(path), dtype="float64")
    source, source_rate = sf.read(str(server.audio_source(real_dev)), dtype="float64",
                                  frames=len(clip))
    assert rate == source_rate == 16000 and source.shape[1] == 8
    c0 = source[:, 0]
    assert float(np.max(np.abs(clip - c0))) == 0.0, "切片不是源的第 0 通道"

    downmix = source.mean(axis=1)                       # `-ac 1` 会得到的信号
    assert float(np.max(np.abs(clip - downmix))) > 0.1 * float(np.max(np.abs(c0))), \
        "切片与 8 通道下混几乎相同：用错了切法（-ac 1）"


def test_missing_source_raises_clip_error(tmp_path, monkeypatch):
    """源 flac 不在本机 → `ClipError`（不是 None、不是空文件）。"""
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "empty",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")
    assert server.audio_source("M_ABSENT") is None
    with pytest.raises(server.ClipError, match="音频源"):
        server.clip_path("M_ABSENT", 30.0)
    assert not (tmp_path / "clips").exists()             # 连缓存目录都不该建


# ─────────────────────────── clip_path：切法与去重 ───────────────────────────

@needs_ffmpeg
def test_clip_takes_channel_zero_and_differs_from_ac1(tmp_path, monkeypatch):
    """现造的 8 通道 flac（只有 c0 有信号）：正取 = c0；`-ac 1` = 另一条（RMS 差数倍）。

    阈值来自实测：`pan` 与 c0 逐位相等（max diff = 0），`-ac 1` 的 RMS 只有 c0 的
    1/6.2（7.1→mono）。所以「用 -ac 1 实现」在 `> 4` 这条上必红。
    """
    source = synth_source(tmp_path / "data", "M_SYN", seconds=4.0)
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    path = server.clip_path("M_SYN", 4.0)
    clip, rate = sf.read(str(path), dtype="float64")
    raw, _ = sf.read(str(source), dtype="float64")

    mix = tmp_path / "ac1.wav"
    subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                    "-t", "4", "-i", str(source), "-ac", "1", "-ar", "16000", str(mix)],
                   check=True)
    downmix, _ = sf.read(str(mix), dtype="float64")

    assert rate == 16000 and clip.ndim == 1
    assert float(np.max(np.abs(clip - raw[:len(clip), 0]))) == 0.0, "不是第 0 通道"
    ratio = rms(clip) / rms(downmix)
    assert ratio > 4.0, "切片与 -ac 1 下混同量级（ratio=%.3f）：用错了切法" % ratio


@needs_ffmpeg
def test_clip_cache_hit_does_not_shell_out_again(tmp_path, monkeypatch):
    """缓存命中直接回：第二次调用**不再启动 ffmpeg**（文件也不重写）。"""
    synth_source(tmp_path / "data", "M_SYN", seconds=4.0)
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    first = server.clip_path("M_SYN", 4.0)
    mtime = first.stat().st_mtime_ns

    def boom(*args, **kwargs):
        raise AssertionError("缓存命中了还在切片：%s" % (args[0],))

    monkeypatch.setattr(server.subprocess, "run", boom)
    second = server.clip_path("M_SYN", 4.0)
    assert second == first and second.stat().st_mtime_ns == mtime

    info = sf.info(str(second))
    assert info.channels == 1 and info.samplerate == 16000


@needs_ffmpeg
def test_concurrent_requests_never_leave_half_a_wav(tmp_path, monkeypatch):
    """并发同键：锁内只切一次、`os.replace` 原子落盘 —— 不留 `.tmp.wav`、不出现半个 wav。"""
    synth_source(tmp_path / "data", "M_SYN", seconds=4.0)
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    calls = []
    real_run = subprocess.run

    def slow_run(*args, **kwargs):                       # 拉长临界区，让 6 个线程真重叠
        calls.append(args[0])
        time.sleep(0.25)
        return real_run(*args, **kwargs)

    monkeypatch.setattr(server.subprocess, "run", slow_run)

    threads = 6
    barrier = threading.Barrier(threads)
    results, errors = [], []

    def worker():
        try:
            barrier.wait(10)
            results.append(server.clip_path("M_SYN", 4.0))
        except Exception as exc:                         # noqa: BLE001 —— 线程里要带回主线程
            errors.append(exc)

    workers = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
    for item in workers:
        item.start()
    for item in workers:
        item.join(timeout=30)

    assert not errors and len(results) == threads
    assert len(set(results)) == 1, results
    assert len(calls) == 1, "同一个 (meeting, seconds) 切了 %d 次" % len(calls)
    assert list((tmp_path / "clips").glob("*.tmp.wav")) == []

    info = sf.info(str(results[0]))                      # 完整 wav：读得出头，不是半截
    assert info.channels == 1 and info.samplerate == 16000
    samples, _ = sf.read(str(results[0]))
    assert len(samples) == int(4 * 16000)


# ─────────────────────────── clip_path：失败路径 ───────────────────────────

def test_ffmpeg_not_in_path_raises_readable_error(tmp_path, monkeypatch):
    """ffmpeg 不在 PATH → `ClipError`（消息可读），不静默产出空文件。"""
    (tmp_path / "data" / "wav").mkdir(parents=True)
    (tmp_path / "data" / "wav" / "M_SYN.flac").write_bytes(b"x")     # 源在，但 ffmpeg 找不到
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")
    monkeypatch.setattr(server, "CLIP_FFMPEG", "definitely-not-a-real-binary-xyz")

    with pytest.raises(server.ClipError, match="ffmpeg"):
        server.clip_path("M_SYN", 4.0)
    assert list((tmp_path / "clips").glob("*.wav")) == []


def test_ffmpeg_failure_cleans_up_the_tmp_file(tmp_path, monkeypatch):
    """ffmpeg 写了一半再失败（8736 漏掉的路径）：`.tmp.wav` 必须清掉，目标不许出现。"""
    (tmp_path / "data" / "wav").mkdir(parents=True)
    (tmp_path / "data" / "wav" / "M_SYN.flac").write_bytes(b"not a flac")
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    def half_then_fail(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"half a wav")         # 模拟 ffmpeg 留下半截文件
        raise subprocess.CalledProcessError(1, cmd, stderr=b"[error] boom\n")

    monkeypatch.setattr(server.subprocess, "run", half_then_fail)
    with pytest.raises(server.ClipError, match="退出码 1"):
        server.clip_path("M_SYN", 4.0)
    assert list((tmp_path / "clips").iterdir()) == [], "失败后缓存目录里不该有残留"


@needs_ffmpeg
def test_real_ffmpeg_failure_raises_clip_error(tmp_path, monkeypatch):
    """真 ffmpeg 对着坏源文件返回非 0：抛 `ClipError`（含退出码与 stderr 末行）。"""
    (tmp_path / "data" / "wav").mkdir(parents=True)
    (tmp_path / "data" / "wav" / "M_BAD.flac").write_bytes(b"definitely not a flac")
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    with pytest.raises(server.ClipError, match="退出码"):
        server.clip_path("M_BAD", 4.0)
    assert list((tmp_path / "clips").glob("*.wav")) == []


# ─────────────────────────── HTTP：/api/audio ───────────────────────────

@NO_DEV
def test_http_audio_returns_wav_not_a_receipt(http_server, real_dev, tmp_path, monkeypatch):
    """`/api/audio` 回 `audio/wav` 字节 + `X-Clamped-Seconds`（真 dev 场次，1 → 30）。"""
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")   # 现切，不吃旧缓存
    status, headers, body = request(http_server, "/api/audio?meeting=%s&seconds=1" % real_dev)
    assert status == 200
    assert headers["Content-Type"] == "audio/wav"
    assert headers["X-Clamped-Seconds"] == "30"
    assert int(headers["Content-Length"]) == len(body) > 44

    info = sf.info(io.BytesIO(body))                     # 是能播的 wav，不是 JSON 回执
    assert info.channels == 1 and info.samplerate == 16000
    assert abs(info.duration - 30.0) < 0.2


def test_http_audio_bad_seconds_still_400(http_server, monkeypatch):
    """参数非法（`nan`/`inf`/非数）→ 400 JSON：这是**唯一**还允许的回执。"""
    monkeypatch.setattr(server, "meetings", lambda: [{"id": "M1"}])
    for raw in ("nan", "inf", "-inf", "abc", ""):
        status, _, body = request(http_server, "/api/audio?meeting=M1&seconds=" + raw)
        assert status == 400, raw
        assert body.decode("utf-8").startswith('{"t": "error"')


def test_http_audio_without_source_returns_receipt_not_empty_wav(http_server, tmp_path,
                                                                 monkeypatch):
    """本机没有这场 flac（如 train_L 未下载）：200 + 可读回执 + 钳制头。

    回执里 `clip` 必须是 `null` —— 空 wav 或 500 都会让页面以为有音频/整块坏掉；
    `app.js` 见到 JSON Content-Type 就走降级分支（T6 已实现，本任务不改前端）。
    """
    monkeypatch.setattr(server, "meetings", lambda: [{"id": "M_ABSENT"}])
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "empty",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    status, headers, body = request(http_server, "/api/audio?meeting=M_ABSENT&seconds=900")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert headers["X-Clamped-Seconds"] == "900"
    payload = json.loads(body.decode("utf-8"))
    assert payload["clip"] is None and payload["clamped_seconds"] == 900.0
    assert "音频源" in payload["note"]
    assert not (tmp_path / "clips").exists()


@needs_ffmpeg
def test_http_audio_ffmpeg_failure_is_500_not_a_silent_empty_wav(http_server, tmp_path,
                                                                 monkeypatch):
    """ffmpeg 返回非 0 → 500 + 可读错误（不是 200 空文件、也不是 404）。"""
    root = tmp_path / "data"
    (root / "wav").mkdir(parents=True)
    (root / "wav" / "M_BAD.flac").write_bytes(b"definitely not a flac")
    monkeypatch.setattr(server, "meetings", lambda: [{"id": "M_BAD"}])
    monkeypatch.setattr(server, "truth_roots", lambda: (root,))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")

    status, headers, body = request(http_server, "/api/audio?meeting=M_BAD&seconds=60")
    assert status == 500
    assert headers["Content-Type"].startswith("application/json")
    assert headers["X-Clamped-Seconds"] == "60"
    payload = json.loads(body.decode("utf-8"))
    assert "ffmpeg" in payload["error"] and "退出码" in payload["error"]
    assert list((tmp_path / "clips").glob("*.wav")) == []


@needs_ffmpeg
def test_http_audio_ffmpeg_missing_from_path_is_500(http_server, tmp_path, monkeypatch):
    """ffmpeg 不在 PATH → 500 + 告诉怎么修（`FileNotFoundError` 不许冒成 200/404）。"""
    synth_source(tmp_path / "data", "M_SYN", seconds=4.0)
    monkeypatch.setattr(server, "meetings", lambda: [{"id": "M_SYN"}])
    monkeypatch.setattr(server, "truth_roots", lambda: (tmp_path / "data",))
    monkeypatch.setattr(server, "CLIP_DIR", tmp_path / "clips")
    monkeypatch.setattr(server, "CLIP_FFMPEG", "definitely-not-a-real-binary-xyz")

    status, _, body = request(http_server, "/api/audio?meeting=M_SYN&seconds=30")
    assert status == 500
    assert "ffmpeg 不在 PATH" in json.loads(body.decode("utf-8"))["error"]


# ─────────────────────────── HTTP：分块写 ───────────────────────────

def test_file_is_written_in_blocks_without_reading_it_whole(tmp_path, monkeypatch):
    """`_file()` 用 `FILE_CHUNK` 的分块写（T5 的 `read_bytes()` 整读 = 已修缺陷）。

    把 `Path.read_bytes` 换成炸弹：整读一次这条就炸；再看每次 `wfile.write` 的块大小。
    """
    target = tmp_path / "big.wav"
    size = server.FILE_CHUNK * 3 + 12345
    target.write_bytes(b"\x00" * size)

    def no_whole_read(self, *args, **kwargs):
        raise AssertionError("整读了 %s：应改成 %d 字节一块的分块写"
                             % (self, server.FILE_CHUNK))

    monkeypatch.setattr(Path, "read_bytes", no_whole_read)

    recorder = _Recorder()
    server.Handler._file(recorder, target, "audio/wav", {"X-Clamped-Seconds": "600"})
    assert recorder.status == 200 and recorder._closed is True
    assert recorder.headers["Content-Type"] == "audio/wav"
    assert recorder.headers["Content-Length"] == str(size)
    assert recorder.headers["X-Clamped-Seconds"] == "600"
    assert sum(len(block) for block in recorder.blocks) == size
    assert len(recorder.blocks) == 4, [len(block) for block in recorder.blocks]
    assert max(len(block) for block in recorder.blocks) <= server.FILE_CHUNK
    assert recorder.flushed is True
