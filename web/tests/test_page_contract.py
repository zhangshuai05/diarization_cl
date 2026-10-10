# -*- coding: utf-8 -*-
"""页面契约（T6）：`web/app.js` 的字面量与后端 contract 逐字比对 + 门禁逻辑锚定。

不跑 JS 引擎（T6 brief 要求）：按文本抽数组字面量 / 函数体，再断言关键分支与字面量。
覆盖的硬要求：

* `LANE_PAYLOAD_KEYS` / `TRUTH_DERIVED` 与 `web/backend/contract.py` 逐字相等；
* `merged` 列与 `oracle` 泳道的「读真值」徽标判定走 `TRUTH_DERIVED`，不是写死的 if；
* DER 门禁：`parity.status !== "pass"` → 字面量 `—（未对拍）`，数字渲染只在其后；
* 空读数（``None``）→ `—`，不是 0%；
* `pin_base` 指示：钉住「生产口径」/ 解钉「base 已非生产口径」；
* `/api/audio` 现在只回 JSON 回执：页面读回执并降级，不把 JSON 塞给播放器；
* 播放头 `left` 用**轨道自己的 rect** 换算（8736 有 72 px 错位的老 bug）；
* 零构建、无外部资源、无动效 / 渐变 / 阴影装饰。
"""
import re
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parents[1]
for _path in (WEB, WEB / "backend"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import contract  # noqa: E402

APP = WEB / "app.js"
HTML = WEB / "index.html"
CSS = WEB / "style.css"

# 页面布局必须存在的挂载点（spec §6.1）。
REQUIRED_IDS = ("statusbar", "parity-badge", "pin-badge", "knob-badge", "meetings",
                "meeting-count", "knobs", "lam", "min_win", "cap", "mode", "refine",
                "oracle", "pin_base", "run", "metrics-body", "merged-th", "tracklayer",
                "timeline-note", "audio", "mapping-body", "mapping-note", "log")


def _read(path):
    return path.read_text(encoding="utf-8")


def _array_literal(js, name):
    """抽 `const NAME = ["...", ...];` 的字符串列表（保留顺序）。"""
    match = re.search(r"const %s = \[(.*?)\];" % re.escape(name), js, re.S)
    assert match, "app.js 缺 %s 字面量" % name
    return tuple(re.findall(r'"([^"]+)"', match.group(1)))


def _function_body(js, name):
    """抽 `function name(...) { ... }` 的正文；字符串与 `//` 注释内的花括号不算配平。"""
    match = re.search(r"function\s+%s\s*\(" % re.escape(name), js)
    assert match, "app.js 缺函数 %s" % name
    start = js.index("{", match.end())
    depth, index, quote = 0, start, None
    while index < len(js):
        char = js[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "\"'`":
            quote = char
        elif char == "/" and js.startswith("//", index):
            newline = js.find("\n", index)
            index = len(js) if newline < 0 else newline
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return js[start + 1:index]
        index += 1
    raise AssertionError("函数 %s 的花括号不配平" % name)


# ─────────────────────────── 契约字面量 ───────────────────────────

def test_lane_payload_keys_literal_matches_backend():
    assert _array_literal(_read(APP), "LANE_PAYLOAD_KEYS") == contract.LANE_PAYLOAD_KEYS


def test_truth_derived_literal_matches_backend():
    assert _array_literal(_read(APP), "TRUTH_DERIVED") == contract.TRUTH_DERIVED


# ─────────────────────────── 读真值徽标（不写死） ───────────────────────────

def test_truth_badge_helpers_consult_the_contract_literal():
    js = _read(APP)
    lane_body = _function_body(js, "isTruthDerivedLane")
    field_body = _function_body(js, "isTruthDerivedField")
    assert "TRUTH_DERIVED.includes" in lane_body and "lane" in lane_body
    assert "TRUTH_DERIVED.includes" in field_body and "key" in field_body


def test_lane_badge_and_merged_header_use_the_helpers():
    js = _read(APP)
    badge_body = _function_body(js, "laneBadge")
    merged_body = _function_body(js, "renderMergedHeader")
    assert "isTruthDerivedLane" in badge_body and "读真值" in badge_body
    # merged 列名不写死：从 LANE_PAYLOAD_KEYS 里筛 merged*，再走 isTruthDerivedField
    assert "LANE_PAYLOAD_KEYS" in merged_body and "isTruthDerivedField" in merged_body
    assert "读真值" in merged_body


# ─────────────────────────── DER 门禁 ───────────────────────────

def test_der_gate_hides_numbers_without_pass():
    body = _function_body(_read(APP), "derDisplay")
    assert "parity" in body
    assert 'parity.status !== "pass"' in body
    assert "—（未对拍）" in body
    # 数字只允许出现在 pass 分支之后（字面量位置锚定）
    assert body.index("—（未对拍）") < body.index("toFixed")


def test_der_cell_does_not_leak_detail_when_unverified():
    body = _function_body(_read(APP), "derCell")
    assert "derDisplay" in body
    assert 'parity.status === "pass"' in body


def test_empty_reading_renders_dash_not_zero():
    body = _function_body(_read(APP), "pctOrDash")
    assert "null" in body and "undefined" in body
    assert "—" in body
    assert "|| 0" not in body


# ─────────────────────────── pin_base / 音频 / 播放头 ───────────────────────────

def test_pin_base_labels_exist():
    body = _function_body(_read(APP), "pinBaseText")
    assert "生产口径" in body
    assert "base 已非生产口径" in body


def test_audio_json_receipt_is_tolerated():
    body = _function_body(_read(APP), "loadAudio")
    assert "response.json()" in body          # T7 之前服务端回 JSON 回执
    assert "receipt" in body
    assert "audio/" in body                   # 只有真音频才塞给播放器
    assert "logLine" in body                  # 降级要在日志里给可读提示
    assert "disabled = true" in body          # 播放器不可用也不炸页面


def test_playhead_uses_track_rect_not_layer_width():
    body = _function_body(_read(APP), "playheadLeft")
    assert "getBoundingClientRect()" in body
    assert "trackRect.width" in body
    assert "layerRect.width" not in body      # 8736 的 72 px 错位根因


# ─────────────────────────── 静态骨架与视觉 ───────────────────────────

def test_app_dom_ids_exist_in_index():
    js, html = _read(APP), _read(HTML)
    used = set(re.findall(r'\$\("([^"]+)"\)', js))
    declared = set(re.findall(r'id="([^"]+)"', html))
    assert used <= declared, "app.js 引用了 index.html 没有的 id：%s" % sorted(used - declared)
    for required in REQUIRED_IDS:
        assert 'id="%s"' % required in html, "布局缺挂载点 #%s" % required


def test_zero_build_and_no_external_assets():
    html, css = _read(HTML), _read(CSS)
    assert re.search(r'<script src="app\.js"', html)
    assert re.search(r'<link rel="stylesheet" href="style\.css"', html)
    assert not re.search(r'(?:src|href)="(?:https?:)?//', html)
    assert "@import" not in css and "url(" not in css


def test_layout_columns_and_no_decorations():
    css = _read(CSS)
    assert "260px" in css and "320px" in css and "1fr" in css
    for banned in ("animation", "transition", "gradient", "box-shadow", "text-shadow",
                   "@keyframes"):
        assert banned not in css, "视觉要求禁止 %s" % banned
