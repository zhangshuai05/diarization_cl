# -*- coding: utf-8 -*-
"""web 调试台服务（8737）：静态页 + `/api/*`；只做**参数校验 → 编排 → 流式返回**。

算法一律在 `web/backend/`（T1–T4），本模块不重算任何指标：

* `/api/meetings` —— 探针缓存目录（本机 50 场）+ 元信息（`has_ref` 读 `artifacts/v13_ab_dev_results.json`）。
* `/api/truth` —— 本场真值段表（`reference` + `gt_k` + `has_ref`）：时间轴的真值轨与映射表的
  分母；与 `/api/run` 的参照**同源同口径**（同一份 rttm、同一个 600 s 窗口）。
* `/api/run` —— `lanes.run_lanes(meeting, knobs)` 逐条推 ndjson（`lane` / `done` / `error`）；
  每条 `lane` 事件在 16 键 payload 之上带 `pairs`（映射表行，口径见 `metrics.pairs`）。
* `/api/audio` —— `seconds` 钳到 `[30, 3600]` 且拒 `nan/inf`（400）；切片走 `clip_path`：
  ffmpeg `pan=mono|c0=c0` 取第 0 通道（**不是** `-ac 1`）、缓存到 `web/clips/`、分块回 `audio/wav`。
* `/api/verify` —— `verify_der.parity_status()` 的**同步**回执（不在这里重跑 15 s 的对拍）。
* `/`、`/app.js`、`/style.css` —— `web/` 下的静态文件，不存在就 404（页面文件是 T6 的交付物）。

## 并发（Ruling 2）

`/api/run` 进程内**一次只准一个**：`_RUN_LOCK` 非阻塞获取，拿不到就 409、**不排队**——
λ 是模块常量，两个 run 并发会互踩（本机实测一条 1.3–1.9 s，409 重试很便宜）。
`lanes._KNOB_LOCK` 管单条泳道内 λ 交换的瞬时窗口，`_RUN_LOCK` 管整个流的时长，
两层不嵌套（`run_lanes` 在锁内算完再 yield）。

## 终止纪律（Ruling 3）

流式响应体只由 `_finish()` 写一次 `0\r\n\r\n`，**写完立刻** `_closed = True`；
`_emit` 第一件事就是看 `_closed`，置位后静默返回、不再碰 socket。8736 的坑是终止块
之后仍写 socket —— keep-alive 下第二个请求会读到上一个响应的尾巴。客户端掉线
（BrokenPipe/ConnectionReset）同样置 `_closed`，让编排循环自己退出、`finally` 关生成器。

## 契约

每个 `lane` 事件 = `contract.LANE_PAYLOAD_KEYS` 的 16 个键**原样平铺**（缺字段=bug，
这里直接抛错推 `error`）+ `parity`（页面据此决定 DER 显不显示）+ spec §5 的 `metrics`
子对象（同一份读数，T6 两种读法都成立）；`done` 带 `parity`。

跑法：`python web/server.py` → http://127.0.0.1:8737/
"""
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_HERE = Path(__file__).resolve().parent          # web/
ROOT = _HERE.parent                              # 仓库根
# 与 cache_path / lanes 同一套 `sys.path` 兜底：本模块按路径直接引后端与探针模块，
# 不依赖调用方先摆好路径（也不依赖 import 顺序）。
for _path in (_HERE / "backend", ROOT / "streaming_asr", ROOT / "scripts" / "evaluation"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import contract  # noqa: E402
import metrics  # noqa: E402
import probe_path_equiv as PE  # noqa: E402
import probe_sentence_attribution as P  # noqa: E402
import verify_der  # noqa: E402
from der import read_rttm  # noqa: E402
from lanes import DATA_ROOTS, KNOB_DEFAULTS, run_lanes  # noqa: E402

import v12_aishell4 as v12_aishell4  # noqa: E402

PORT = 8737
MEETINGS_ARTIFACT = ROOT / "artifacts" / "v13_ab_dev_results.json"

# 旋钮的合法域（Ruling 4：越界 400，**不静默钳制**）。
LAM_RANGE = (0.0, 0.03)
MIN_WIN_VALUES = (2, 3, 4)
CAP_VALUES = (4, 5, 6, 7, 8, 9, 10)
MODES = ("objective", "largest_eligible")
BOOL_WORDS = {"0": False, "1": True, "true": True, "false": False}

# `/api/audio` 的 seconds 相反：钳到闭区间（Ruling 4）。
AUDIO_SECONDS_DEFAULT = 600.0
AUDIO_SECONDS_RANGE = (30.0, 3600.0)

# 切片缓存与切法：`*.wav` 已被 .gitignore 忽略，不入库；一块 1 MiB 地回（大文件不整读）。
CLIP_DIR = _HERE / "clips"
CLIP_FFMPEG = "ffmpeg"
CLIP_TIMEOUT_S = 120                        # ffmpeg 挂死上限；超时按失败处理（清理临时文件）
FILE_CHUNK = 1 << 20
_CLIP_LOCK = threading.Lock()               # 同一时刻只准一个切片进程（8736 同款）

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ico": "image/x-icon",
}

KNOWN_API = ("/api/meetings", "/api/run", "/api/truth", "/api/audio", "/api/verify")

# 一次只准跑一个 run（Ruling 2）。`lanes._KNOB_LOCK` 不覆盖整条流，别混用。
_RUN_LOCK = threading.Lock()


class BadRequest(Exception):
    """参数不合法 → 400（消息直接回给页面，写清哪个参数、允许什么）。"""


class ClipError(Exception):
    """切片失败（源不在、ffmpeg 缺/非 0）→ `/api/audio` 回 500 + 可读错误。

    绝不静默回 200 空文件：页面看到 500 会降级到「只有播放头」，比空音频诚实。
    """


def _dumps(payload):
    """numpy 标量 → 原生类型；漏一个就会让整条流断在半路。"""
    def scalar(value):
        if hasattr(value, "item"):
            return value.item()
        raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)

    return json.dumps(payload, ensure_ascii=False, default=scalar)


# ─────────────────────────── 场次目录（白名单的唯一来源） ───────────────────────────

_CATALOG = None
_CATALOG_LOCK = threading.Lock()


def _read_json(path):
    """缺文件/坏 JSON → None（列表少两列，不是把整个页面打崩的理由）。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _meeting_cohort(meeting):
    """cohort 徽标：dev / holdout / train_L（与产物命名 `_dev` `_holdout` `_trainL` 同一套）。"""
    name = meeting.split("_", 1)[1] if meeting[:1].isdigit() else meeting
    try:
        group = v12_aishell4.meeting_group(name)
    except ValueError:
        return "train_L"                             # 带日期前缀的 train_L 场次
    if group in v12_aishell4.DEV_GROUPS:
        return "dev"
    if group in v12_aishell4.HOLDOUT_GROUPS:
        return "holdout"
    return "train_L"


def _n_windows_from_npz(path):
    """manifest 缺条目时的兜底：只读 npz 头的形状（mmap，不搬数据）。"""
    try:
        import numpy as np

        with np.load(path, mmap_mode="r") as data:
            return int(data["emb"].shape[0])
    except Exception:
        return None


def _build_meetings():
    """缓存目录 → 列表行 `{id, cohort, has_ref, gt_speakers, duration, n_windows}`。

    * `has_ref` **只**读 `artifacts/v13_ab_dev_results.json`（Ruling 5）——不用
      「der 是 None」这类推断：无参照是有真值文件但没进参照产物，两回事。
    * `gt_speakers` / `duration` 直接读 rttm 与 flac header，不解析 TextGrid。
    * `n_windows` 取缓存 manifest 的 `meetings.<id>.counts.main`（与参照产物的 `n_windows`
      同量，dev 12 场逐场实测相等）；manifest 缺条目时退回读 `main.npz` 的形状。
    """
    if not PE.PROBE_CACHE.is_dir():
        return []
    import soundfile as sf

    from der import read_rttm

    manifest = _read_json(PE.PROBE_CACHE / "manifest.json") or {}
    counts = manifest.get("meetings") or {}
    referenced = referenced_meetings()

    rows = []
    for entry in sorted(PE.PROBE_CACHE.iterdir()):
        if not entry.is_dir():
            continue
        meeting = entry.name
        gt_speakers, duration = None, None
        for root in truth_roots():
            rttm = root / "TextGrid" / ("%s.rttm" % meeting)
            if gt_speakers is None and rttm.is_file():
                gt_speakers = len({seg["speaker"] for seg in read_rttm(rttm)})
            flac = root / "wav" / ("%s.flac" % meeting)
            if duration is None and flac.is_file():
                duration = round(sf.info(flac).duration, 1)
        n_windows = ((counts.get(meeting) or {}).get("counts") or {}).get("main")
        if n_windows is None:
            n_windows = _n_windows_from_npz(entry / "main.npz")
        rows.append({"id": meeting, "cohort": _meeting_cohort(meeting),
                     "has_ref": meeting in referenced, "gt_speakers": gt_speakers,
                     "duration": duration, "n_windows": n_windows})
    return rows


def meetings():
    """`/api/meetings` 的列表 = `/api/run`、`/api/audio` 的白名单（同一次调用，Ruling 4）。

    目录构建只做一次（50 场的 rttm/flac header 在本机实测 0.05 s，但没必要每次重读）。
    """
    global _CATALOG
    with _CATALOG_LOCK:
        if _CATALOG is None:
            _CATALOG = _build_meetings()
        return [dict(row) for row in _CATALOG]


# ─────────────────────────── 真值段表（/api/truth 与 lane 的 pairs） ───────────────────────────

# 真值窗口：与 `probe_sentence_attribution.SECONDS` 同值（600 s）；`lanes.load_truth` 的参照
# 也是这么截的（`start < 600`、末端 min(end, 600)），这里不另造一套口径。
TRUTH_SECONDS = P.SECONDS


def truth_roots():
    """真值根目录（`<root>/TextGrid/<meeting>.rttm`）= `/api/run` 的同一对根。

    默认 = `lanes.DATA_ROOTS`（test + train_L，按 TextGrid 在哪定根）；`AISHELL4_TEST`
    可覆盖 test 根（与 8736 同名同默认），train_L 跟随它的父目录。注意：覆盖只影响本服务的
    真值读取（`/api/truth` 与 lane 的 `pairs`）与**音频源**（`/api/audio`）——
    泳道指标走 `lanes`（冻结文件）的默认根。
    """
    override = os.environ.get("AISHELL4_TEST")
    if not override:
        return DATA_ROOTS
    base = Path(override)
    return (base, base.parent / "train_L")


def reference_segments(meeting):
    """本场真值段表（600 s 窗口）：`read_rttm` + 截断，与 `lanes.load_truth` 的参照同口径。

    找不到 rttm 的场次返回空表——「无门限参照」是合法场次（50 场里 38 场如此），不是错误。
    """
    for root in truth_roots():
        path = root / "TextGrid" / ("%s.rttm" % meeting)
        if path.is_file():
            return [{"start": s["start"], "end": min(s["end"], TRUTH_SECONDS),
                     "speaker": s["speaker"]}
                    for s in read_rttm(path) if s["start"] < TRUTH_SECONDS]
    return []


def referenced_meetings():
    """`has_ref` 的唯一来源（Ruling 5）：参照产物的 results 场次名（与 `/api/meetings` 同源）。"""
    artifact = _read_json(MEETINGS_ARTIFACT) or {}
    return {item.get("meeting") for item in artifact.get("results") or []}


def lane_pairs(segments, reference):
    """T1 段表（`spk` 簇号）→ 映射表行（`metrics.pairs` 的输入口径）。

    簇名用 `#N`：与 8736 `bench_server.run` 把 `spk` 转 `f"#{int(spk)}"` 后再调 `pairs()`
    一字不差，行里的 `pred` 就是页面显示的那个名字。
    """
    predicted = [{"start": s["start"], "end": s["end"], "speaker": "#%d" % int(s["spk"])}
                 for s in (segments or [])]
    return metrics.pairs(predicted, reference or [])


# ─────────────────────────── 参数校验 ───────────────────────────

def _raw(query, key):
    """缺参数 → None；显式空值（`?x=`）→ `""`（`keep_blank_values`），按非法处理。"""
    values = query.get(key) or []
    return values[0] if values else None


def _parse_meeting(query):
    meeting = _raw(query, "meeting")
    if not meeting:
        raise BadRequest("缺少 meeting 参数（合法值见 /api/meetings）")
    known = {row["id"] for row in meetings()}
    if meeting not in known:
        raise BadRequest("未知场次 %r：不在 /api/meetings 白名单内（共 %d 场），"
                         "拒绝路径式取值" % (meeting, len(known)))
    return meeting


def _finite(raw, key):
    """字符串 → 有限 float；`nan`/`inf`/非数字一律 400（旋钮与 seconds 共用）。"""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise BadRequest("%s 不是数字：%r" % (key, raw))
    if not math.isfinite(value):
        raise BadRequest("%s 必须是有限数：%r" % (key, raw))
    return value


def _number(query, key, default, low, high):
    raw = _raw(query, key)
    if raw is None:
        return float(default)
    value = _finite(raw, key)
    if not low <= value <= high:
        raise BadRequest("%s=%s 越界：允许 [%g, %g]，不钳制" % (key, raw, low, high))
    return value


def _choice(query, key, default, allowed, cast=str):
    raw = _raw(query, key)
    if raw is None:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise BadRequest("%s 不是 %s：%r（允许 %s）"
                         % (key, cast.__name__, raw, list(allowed)))
    if value not in allowed:
        raise BadRequest("%s=%s 不在 %s 内，不钳制" % (key, raw, list(allowed)))
    return value


def _boolean(query, key, default):
    raw = _raw(query, key)
    if raw is None:
        return bool(default)
    value = BOOL_WORDS.get(raw.lower())
    if value is None:
        raise BadRequest("%s=%r 只接受 %s" % (key, raw, sorted(BOOL_WORDS)))
    return value


def parse_run(query):
    """`/api/run` 的 query → `(meeting, knobs)`；任何越界都是 `BadRequest`。"""
    meeting = _parse_meeting(query)
    knobs = {
        "lam": _number(query, "lam", KNOB_DEFAULTS["lam"], *LAM_RANGE),
        "min_win": _choice(query, "min_win", KNOB_DEFAULTS["min_win"], MIN_WIN_VALUES, int),
        "cap": _choice(query, "cap", KNOB_DEFAULTS["cap"], CAP_VALUES, int),
        "mode": _choice(query, "mode", KNOB_DEFAULTS["mode"], MODES),
        "refine": _boolean(query, "refine", KNOB_DEFAULTS["refine"]),
        "oracle": _boolean(query, "oracle", KNOB_DEFAULTS["oracle"]),
        "pin_base": _boolean(query, "pin_base", KNOB_DEFAULTS["pin_base"]),
    }
    return meeting, knobs


def parse_seconds(raw):
    """`/api/audio` 的 `seconds` → `(requested, clamped)`；缺参数用默认、不钳。

    与旋钮相反：这里**要**钳到 `[30, 3600]`，但 `nan`/`inf` 仍然 400（Ruling 4）。
    """
    if raw is None:
        return AUDIO_SECONDS_DEFAULT, AUDIO_SECONDS_DEFAULT
    seconds = _finite(raw, "seconds")
    low, high = AUDIO_SECONDS_RANGE
    return seconds, min(high, max(low, seconds))


def audio_source(meeting):
    """本机这场会议的音频源 `<root>/wav/<meeting>.flac`（与真值同一对根）；没有 → None。"""
    for root in truth_roots():
        path = root / "wav" / ("%s.flac" % meeting)
        if path.is_file():
            return path
    return None


def _ffmpeg_tail(exc):
    """ffmpeg 的 stderr 收尾一行（失败原因在最后一行；太长就截断，别把整页日志回给页面）。"""
    lines = (exc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    return (lines[-1] if lines else "没有 stderr")[:300]


def clip_path(meeting, seconds):
    """浏览器播放用的音频切片 → `web/clips/<meeting>_<seconds>.wav`；失败抛 `ClipError`。

    * **取第 0 通道**：`-af pan=mono|c0=c0`，不是 `-ac 1`——8 通道下混到单声道是另一条
      信号（别的说话人也混进来），8736 的注释写明了，T7 的测试盯着这条。
    * 缓存命中（含等锁期间别人切好了）直接回；未命中的在 `_CLIP_LOCK` 内切一次，
      `os.replace` 原子落盘——并发同键不会看到半个 wav。
    * 失败（源不在、ffmpeg 不在 PATH、ffmpeg 非 0、超时）清掉 `.tmp.wav` 再抛：8736 漏了
      这个临时文件（半截文件会留在缓存目录里），不要照抄。`timeout=CLIP_TIMEOUT_S` 保证
      ffmpeg 挂死时不会永久持有 `_CLIP_LOCK`（T7 审查遗留项）。
    """
    source = audio_source(meeting)
    if source is None:
        raise ClipError("本机没有 %s 的音频源：%s（装上音频或设 AISHELL4_TEST 后重试）"
                        % (meeting, "/".join(str(root / "wav") for root in truth_roots())))
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    target = CLIP_DIR / ("%s_%g.wav" % (meeting, seconds))
    if target.is_file():
        return target
    with _CLIP_LOCK:
        if target.is_file():                    # 等锁期间已经切好了：不用再起一个 ffmpeg
            return target
        temporary = target.with_suffix(".tmp.wav")
        try:
            try:
                subprocess.run(
                    [CLIP_FFMPEG, "-y", "-t", "%g" % seconds, "-i", str(source),
                     "-af", "pan=mono|c0=c0", "-ar", "16000", str(temporary)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    timeout=CLIP_TIMEOUT_S)
            except FileNotFoundError:
                raise ClipError("ffmpeg 不在 PATH：装好 ffmpeg 再切片（%s）" % CLIP_FFMPEG)
            except subprocess.TimeoutExpired:
                raise ClipError("ffmpeg 切 %s 前 %g 秒超过 %d 秒未完成：已中止，缓存未写入"
                                % (meeting, seconds, CLIP_TIMEOUT_S))
            except subprocess.CalledProcessError as exc:
                raise ClipError("ffmpeg 切 %s 前 %g 秒失败（退出码 %d）：%s"
                                % (meeting, seconds, exc.returncode, _ffmpeg_tail(exc)))
            os.replace(temporary, target)       # 原子：读者要么看到完整的 wav，要么没有
        finally:
            if temporary.exists():              # 成功时 replace 已经搬走，留下的都是半截
                try:
                    temporary.unlink()
                except OSError:
                    pass
    return target


# ─────────────────────────── 事件组装 ───────────────────────────

def lane_event(payload, parity, reference):
    """`run_lanes` 的 payload → 页面事件。缺契约字段就抛（缺字段=bug，Ruling 6）。

    `pairs`（映射表行）由编排层在这里补：泳道 payload 只带预测段表与读数，真值在编排层
    才注入——`lanes.py` 冻结、不产出 `pairs`，所以事件键 = payload 键 + `pairs`
    （`contract.LANE_EVENT_KEYS`）；取值口径见 `lane_pairs`。
    """
    event = {"t": "lane", **payload, "pairs": lane_pairs(payload.get("segments"), reference),
             "parity": parity}
    missing = [key for key in contract.LANE_EVENT_KEYS if key not in event]
    if missing:
        raise RuntimeError("泳道 payload 缺契约字段：%s" % ", ".join(missing))
    # spec §5 的形状：`metrics` = 除 lane/segments 外的 14 个读数（与平铺的是同一份）
    event["metrics"] = {key: payload[key] for key in contract.LANE_PAYLOAD_KEYS
                        if key not in ("lane", "segments")}
    return event


# ─────────────────────────── HTTP ───────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # 本地调试台不需要 access log
        pass

    # 一个 Handler 实例会被同一连接的多个请求复用（keep-alive），所以每进一个请求
    # 都要复位这两个标志：`_started` = 响应头已发出（不能再回 JSON 错误），
    # `_closed` = 本响应的 body 已结束（此后任何 `_emit` 静默返回）。
    _started = False
    _closed = False

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        # 本服务的 POST 不收 body；不读掉会让 keep-alive 的下一个请求错位。
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._dispatch("POST")

    def _dispatch(self, method):
        self._started = False
        self._closed = False
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if path in KNOWN_API and method != "GET" and not (path == "/api/verify"
                                                              and method == "POST"):
                self._json({"t": "error", "error": "405 端点 %s 不支持 %s" % (path, method)}, 405)
            elif path == "/api/meetings":
                self._json({"meetings": meetings()})
            elif path == "/api/run":
                self._run(query)
            elif path == "/api/truth":
                self._truth(query)
            elif path == "/api/audio":
                self._audio(query)
            elif path == "/api/verify":
                self._verify()
            elif path.startswith("/api/"):
                self._json({"t": "error", "error": "404 未知端点：%s" % path}, 404)
            else:
                self._static(path)
        except Exception as exc:                       # 400 已在各端点内处理；这里兜 500
            traceback.print_exc()
            if not self._started:
                try:
                    self._json({"t": "error", "error": "%s: %s" % (type(exc).__name__, exc)}, 500)
                except Exception:
                    traceback.print_exc()

    # --- 流式端点 ---------------------------------------------------------

    def _run(self, query):
        try:
            meeting, knobs = parse_run(query)
        except BadRequest as exc:
            self._json({"t": "error", "error": str(exc)}, 400)
            return
        if not _RUN_LOCK.acquire(blocking=False):
            # 忙 = 409，不排队（Ruling 2）：λ 是模块常量，两条 run 并发会互踩；
            # 重试成本低（本机一条 1.3–1.9 s），没必要排队把"忙"藏起来
            self._json({"t": "error", "error": "busy: 已有一个 /api/run 在跑"
                                              "（一次只准一个，避免 λ 互踩）；结束再试"}, 409)
            return

        generator = None
        try:
            # 真值段表读一次：/api/truth 与每条泳道的 pairs 用同一份（与 lanes 的参照同源）
            reference = reference_segments(meeting)
            self._ndjson_headers()
            parity = verify_der.parity_status()        # 快照一次：本次流的每条事件共用
            started = time.time()
            lanes = 0
            generator = run_lanes(meeting, knobs)
            for payload in generator:
                if not self._emit(lane_event(payload, parity, reference)):
                    break                              # 客户端走了：别再算、别写
                lanes += 1
            else:
                self._emit({"t": "done", "meeting": meeting, "n_lanes": lanes,
                            "elapsed_s": round(time.time() - started, 2), "parity": parity})
        except Exception as exc:
            traceback.print_exc()
            if self._started:
                self._emit({"t": "error", "error": "%s: %s" % (type(exc).__name__, exc)})
            else:
                self._json({"t": "error", "error": "%s: %s" % (type(exc).__name__, exc)}, 500)
        finally:
            if generator is not None:
                try:
                    generator.close()                  # 提前中断也走 run_lanes 的 λ 还原
                except Exception:
                    traceback.print_exc()
            _RUN_LOCK.release()
            if self._started:
                self._finish()

    def _truth(self, query):
        """`{reference, gt_k, has_ref}`（spec §5，Ruling 12）。

        `gt_k` = 返回的段表里**出现过的真值人**数（8736 同口径）——它与真值轨上看到的人一致；
        50 场实测里等于 `/api/meetings` 的 `gt_speakers`（600 s 窗口没截掉任何人）。
        """
        try:
            meeting = _parse_meeting(query)
        except BadRequest as exc:
            self._json({"t": "error", "error": str(exc)}, 400)
            return
        reference = reference_segments(meeting)
        self._json({"reference": reference,
                    "gt_k": len({segment["speaker"] for segment in reference}),
                    "has_ref": meeting in referenced_meetings()})

    def _verify(self):
        try:
            parity = dict(verify_der.parity_status())
            if parity.get("status") == "missing":
                # Ruling 7：不在这里跑 15 s 的对拍，只说清怎么生成
                parity["error"] = ("缺少对拍产物 %s：在仓库根跑 "
                                   "`python -m web.backend.verify_der`"
                                   "（dev 12 场 × 2 臂，约 15 s）" % parity.get("path"))
        except Exception as exc:
            parity = {"status": "fail", "error": "%s: %s" % (type(exc).__name__, exc)}
        self._ndjson_headers()
        try:
            self._emit({"t": "done", "parity": parity})
        finally:
            self._finish()

    def _audio(self, query):
        try:
            meeting = _parse_meeting(query)
            requested, seconds = parse_seconds(_raw(query, "seconds"))
        except BadRequest as exc:
            self._json({"t": "error", "error": str(exc)}, 400)
            return
        headers = {"X-Clamped-Seconds": "%g" % seconds}
        if audio_source(meeting) is None:
            # 本机没有这场音频（如 train_L 未下载）：回可读回执，不假装有音频、也不回空 wav。
            # 这是 /api/audio 唯一还允许的 JSON —— 非法参数 400 之外的降级路径。
            # 页面（web/app.js，T6）见到 application/json 就走「只有播放头」分支。
            self._json({"meeting": meeting, "requested_seconds": requested,
                        "clamped_seconds": seconds, "clip": None,
                        "note": "本机没有 %s 的音频源；切片未生成（时间轴仍可定位）" % meeting},
                       200, headers=headers)
            return
        try:
            clip = clip_path(meeting, seconds)
        except ClipError as exc:
            self._json({"t": "error", "error": str(exc)}, 500, headers=headers)
            return
        self._file(clip, "audio/wav", headers=headers)

    # --- 写回 -------------------------------------------------------------

    def _emit(self, payload):
        """写一行 ndjson。返回 False = 流已结束/客户端走了，调用方应停手（不再抛）。"""
        if self._closed:
            return False
        data = (_dumps(payload) + "\n").encode("utf-8")
        try:
            self.wfile.write(("%x\r\n" % len(data)).encode("ascii") + data + b"\r\n")
            self.wfile.flush()
        except OSError:                                # 掉线：静默收摊
            self._closed = True
            return False
        return True

    def _finish(self):
        """终止块**只写一次**，写完立刻置 `_closed`（Ruling 3；8736 的坑）。"""
        if self._closed:
            return
        self._closed = True
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            pass

    def _ndjson_headers(self):
        self._started = True
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _json(self, payload, status=200, headers=None):
        body = _dumps(payload).encode("utf-8")
        self._started = True
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self._closed = True

    def _file(self, path, content_type, headers=None):
        """分块写（`FILE_CHUNK` = 1 MiB/块）：3600 s 单声道 16 kHz 16-bit ≈ 115 MB，
        `read_bytes()` 整读进内存没必要（T5 的既有缺陷，T7 改成分块）。

        `Content-Length` 照旧给全量（文件是 `os.replace` 落盘的成品、不会再变），
        所以 keep-alive 与浏览器进度条都不受影响。
        """
        path = Path(path)
        size = path.stat().st_size                    # 先 stat：文件不在就还没发头，走 500
        self._started = True
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(FILE_CHUNK), b""):
                    self.wfile.write(block)
            self.wfile.flush()
        except OSError:
            self._closed = True                     # 掉线：静默收摊（与 _emit 同纪律）
            return
        self._closed = True

    def _static(self, path):
        """`web/` 下一层文件（T6 的 index.html / app.js / style.css）；没有就 404。"""
        name = path.lstrip("/") or "index.html"
        target = _HERE / name
        if ("/" in name or "\\" in name or name.startswith(".")
                or target.suffix not in STATIC_TYPES or not target.is_file()):
            self._json({"t": "error", "error": "404 找不到静态文件：%s（期望 %s）"
                                              % (path, target)}, 404)
            return
        self._file(target, STATIC_TYPES[target.suffix])


class Server(ThreadingHTTPServer):
    # 同 8736：Windows 上 SO_REUSEADDR 会让第二个实例静默绑上同一端口、分裂流量，
    # 宁可启动失败。daemon 线程让测试进程不被 handler 拖住。
    allow_reuse_address = False
    daemon_threads = True

    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    print("web 调试台：http://127.0.0.1:%d/  （场次见 /api/meetings）" % PORT, flush=True)
    try:
        Server(("127.0.0.1", PORT), Handler).serve_forever()
    except OSError as exc:
        raise SystemExit("端口 %d 起不来（可能已有一个实例）：%s" % (PORT, exc))


if __name__ == "__main__":
    main()
