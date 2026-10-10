"use strict";

/* AISHELL-4 分离调试台页面 / 零构建：原生 JS，直接读 T5 的 /api/*。
   state 只维护一份：场次、旋钮、已到的泳道、对拍状态。 */

/* ── 契约字面量（与 web/backend/contract.py 逐字一致，由 web/tests/test_page_contract.py 比对） ── */
const LANE_PAYLOAD_KEYS = [
  "lane", "k", "level",
  "overall", "clean", "overlap",
  "abstain", "unmapped", "wrong",
  "merged_overall", "merged_der",
  "der", "miss_rate", "false_alarm_rate", "confusion_rate",
  "segments",
];

/* 「读真值」名单：oracle 走泳道名分支、merged_* 走字段名分支（spec §6.2）。 */
const TRUTH_DERIVED = ["oracle", "merged_overall", "merged_der"];

/* 生产旋钮值：状态条的「旋钮非生产值」以它为基准。 */
const KNOB_DEFAULTS = { lam: 0.005, min_win: 2, cap: 10, mode: "objective", refine: true };

/* 后端真值截断窗口（probe_sentence_attribution.SECONDS），与 /api/audio 默认 seconds 同源。 */
const REFERENCE_WINDOW_S = 600;

const PREDICTION_COLORS = ["--p0", "--p1", "--p2", "--p3", "--p4", "--p5", "--p6", "--p7"];

const $ = (id) => document.getElementById(id);

const state = {
  meetings: [],
  meeting: null,
  parity: null,
  lanes: [],
  mappingLane: null,
  total: REFERENCE_WINDOW_S,
  playheadS: 0,
  audioReady: false,
  audioHinted: false,
};

/* ── 指标显示（空值一律 —，DER 受 parity 门禁） ────────────── */

function pctOrDash(value, digits) {
  if (value === null || value === undefined) return "—";
  return (value * 100).toFixed(digits === undefined ? 1 : digits) + "%";
}

/* DER 门禁（spec §6.3）：parity.status != "pass" 时不渲染任何 DER 数字。 */
function derDisplay(value, parity) {
  if (!parity || parity.status !== "pass") return "—（未对拍）";
  if (value === null || value === undefined) return "—";
  return (value * 100).toFixed(2) + "%";
}

function derCell(payload) {
  const der = derDisplay(payload.der, payload.parity);
  if (!payload.parity || payload.parity.status === "pass") {
    return der + ' <small>漏 ' + pctOrDash(payload.miss_rate, 1) + ' · 虚 '
      + pctOrDash(payload.false_alarm_rate, 1) + ' · 混 ' + pctOrDash(payload.confusion_rate, 1) + '</small>';
  }
  return der;
}

function pinBaseText(pinned) {
  return pinned ? "base 已钉住（生产口径）" : "base 已非生产口径";
}

/* ── 「读真值」徽标：判定只读 TRUTH_DERIVED，不写死 ─────────── */

function isTruthDerivedLane(lane) {
  return TRUTH_DERIVED.includes(lane);
}

function isTruthDerivedField(key) {
  return TRUTH_DERIVED.includes(key);
}

function laneBadge(lane) {
  return isTruthDerivedLane(lane) ? ' <span class="badge truth">读真值</span>' : "";
}

function renderMergedHeader() {
  const mergedKeys = LANE_PAYLOAD_KEYS.filter(function (key) { return key.indexOf("merged") === 0; });
  const badge = mergedKeys.some(isTruthDerivedField) ? ' <span class="badge truth">读真值</span>' : "";
  $("merged-th").innerHTML = "merged" + badge;
}

/* ── 状态条 ─────────────────────────────────────────────── */

function knobIsNonDefault() {
  return Number($("lam").value) !== KNOB_DEFAULTS.lam
    || Number($("min_win").value) !== KNOB_DEFAULTS.min_win
    || Number($("cap").value) !== KNOB_DEFAULTS.cap
    || $("mode").value !== KNOB_DEFAULTS.mode
    || $("refine").checked !== KNOB_DEFAULTS.refine;
}

function renderStatusBar() {
  const parity = state.parity;
  const badge = $("parity-badge");
  if (!parity) {
    badge.textContent = "DER 对拍 读不到";
    badge.className = "badge bad";
    badge.title = "未读到 artifacts/web_der_parity.json 的状态";
  } else if (parity.status === "pass") {
    const date = (parity.checked_at || "").slice(0, 10);
    badge.textContent = "DER 对拍 " + (parity.n_pass || 0) + "/" + (parity.n_total || 0) + " ✓ · " + date;
    badge.className = "badge ok";
    badge.title = "checked_at=" + parity.checked_at + " · 阈值 " + parity.threshold_pts + " 点";
  } else {
    badge.textContent = "DER 对拍未通过（" + parity.status + "）";
    badge.className = "badge bad";
    badge.title = parity.path || "";
    const failed = parity.failed || [];
    for (const item of failed) logLine("对拍未通过：" + JSON.stringify(item), "bad");
  }

  const pinned = $("pin_base").checked;
  $("pin-badge").textContent = pinBaseText(pinned);
  $("pin-badge").className = "badge" + (pinned ? "" : " warn");

  const changed = knobIsNonDefault();
  $("knob-badge").textContent = changed ? "旋钮非生产值 ⚠" : "旋钮=生产值";
  $("knob-badge").className = "badge" + (changed ? " warn" : "");
}

/* ── 小工具 ─────────────────────────────────────────────── */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtTime(seconds) {
  const value = Math.max(0, Math.round(seconds));
  return Math.floor(value / 60) + ":" + String(value % 60).padStart(2, "0");
}

function percent(value, total) {
  if (!total) return "0%";
  return (100 * value / total).toFixed(4) + "%";
}

function timelineTotal() {
  const meeting = state.meeting;
  if (!meeting || !meeting.duration) return REFERENCE_WINDOW_S;
  return Math.max(1, Math.min(meeting.duration, REFERENCE_WINDOW_S));
}

function predictionColor(spk) {
  const index = Number.isFinite(Number(spk)) ? Math.abs(Number(spk)) : 0;
  const name = PREDICTION_COLORS[index % PREDICTION_COLORS.length];
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#94a3b8";
}

function logLine(text, kind) {
  const list = $("log");
  if (!list) return;
  const item = document.createElement("li");
  if (kind) item.className = kind;
  item.textContent = text;
  list.append(item);
  while (list.children.length > 200) list.firstChild.remove();
  list.scrollTop = list.scrollHeight;
}

/* ── 场次列表 ───────────────────────────────────────────── */

async function loadMeetings() {
  const response = await fetch("/api/meetings");
  if (!response.ok) throw new Error("HTTP " + response.status);
  const data = await response.json();
  state.meetings = data.meetings || [];
  const list = $("meetings");
  list.textContent = "";
  for (const meeting of state.meetings) {
    const item = el("li");
    const button = el("button", "meeting");
    button.type = "button";
    button.setAttribute("aria-pressed", "false");
    button.innerHTML = '<span class="m-id">' + meeting.id + '</span>'
      + '<span class="m-meta">' + meeting.cohort
      + (meeting.has_ref ? " · 有参照" : " · 无参照")
      + " · " + meeting.gt_speakers + " 人 · " + fmtTime(meeting.duration) + '</span>';
    button.addEventListener("click", function () { selectMeeting(meeting, button); });
    item.append(button);
    list.append(item);
  }
  $("meeting-count").textContent = state.meetings.length + " 场";
  logLine("场次目录就绪：" + state.meetings.length + " 场");
}

function selectMeeting(meeting, button) {
  state.meeting = meeting;
  for (const other of document.querySelectorAll(".meeting")) {
    other.setAttribute("aria-pressed", other === button ? "true" : "false");
  }
  $("run").disabled = false;
  $("run-meta").textContent = meeting.id + " · 窗口 " + fmtTime(timelineTotal());
  logLine("选中 " + meeting.id + "（真值 " + meeting.gt_speakers + " 人 · "
    + (meeting.has_ref ? "有参照" : "无参照") + "）");
  renderTimeline();
}

/* 深链：/?meeting=<id>[&run=1] 打开即选中（可选即跑）——刷新/转发同一次运行。 */
function deepLink() {
  const query = new URLSearchParams(location.search);
  const wanted = query.get("meeting");
  if (!wanted) return;
  const button = [].slice.call(document.querySelectorAll(".meeting")).find(function (item) {
    return item.textContent.indexOf(wanted) >= 0;
  });
  if (!button) return;
  button.click();
  if (query.has("run")) run();
}

/* ── 运行与 NDJSON ──────────────────────────────────────── */

function knobsQuery() {
  const knobs = {
    lam: Number($("lam").value),
    min_win: Number($("min_win").value),
    cap: Number($("cap").value),
    mode: $("mode").value,
    refine: $("refine").checked,
    oracle: $("oracle").checked,
    pin_base: $("pin_base").checked,
  };
  return Object.keys(knobs).map(function (key) {
    return key + "=" + encodeURIComponent(String(knobs[key]));
  }).join("&");
}

async function* readNdjson(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

function acceptLane(event) {
  const missing = LANE_PAYLOAD_KEYS.filter(function (key) { return !(key in event); });
  if (missing.length) {
    logLine("泳道 " + event.lane + " 缺契约字段：" + missing.join(", "), "bad");
    return;
  }
  if (!state.mappingLane) state.mappingLane = event.lane;
  state.lanes.push({ lane: event.lane, payload: event, truthDerived: isTruthDerivedLane(event.lane) });
  state.parity = event.parity || state.parity;
  renderStatusBar();
  renderMetrics();
  renderTimeline();
  renderMapping();
  logLine("泳道 " + event.lane + "：k=" + event.k + " · " + event.level
    + " · DER " + derDisplay(event.der, event.parity));
}

async function run() {
  if (!state.meeting) return;
  const meeting = state.meeting;
  state.lanes = [];
  state.mappingLane = null;
  state.playheadS = 0;
  state.audioHinted = false;
  renderMetrics();
  renderTimeline();
  renderMapping();
  $("run").disabled = true;
  $("run").textContent = "运行中…";
  logLine("运行 " + meeting.id + "：" + knobsQuery());
  const started = performance.now();
  try {
    const response = await fetch("/api/run?meeting=" + encodeURIComponent(meeting.id) + "&" + knobsQuery());
    if (!response.ok) {
      const payload = await response.json().catch(function () { return null; });
      throw new Error(payload && payload.error ? payload.error : "HTTP " + response.status);
    }
    for await (const event of readNdjson(response)) {
      if (event.t === "lane") {
        acceptLane(event);
      } else if (event.t === "error") {
        logLine("运行错误：" + event.error, "bad");
      } else if (event.t === "done") {
        state.parity = event.parity || state.parity;
        renderStatusBar();
        logLine("完成：" + event.n_lanes + " 条泳道 · 服务端 " + event.elapsed_s + " s");
      }
    }
    await loadAudio(meeting.id, state.total);
  } catch (error) {
    logLine("运行失败：" + (error.message || error), "bad");
  } finally {
    $("run").disabled = false;
    $("run").textContent = "运行";
    logLine("本次耗时 " + ((performance.now() - started) / 1000).toFixed(2) + " s（含网络与渲染）");
  }
}

/* ── 指标表 ─────────────────────────────────────────────── */

function renderMetrics() {
  const body = $("metrics-body");
  body.textContent = "";
  if (!state.lanes.length) {
    const row = el("tr");
    row.append(el("td", "muted", "运行后每条泳道一行；空读数（后端 null）显示 —。"));
    row.firstChild.colSpan = 13;
    body.append(row);
    return;
  }
  const gtK = state.meeting && state.meeting.gt_speakers !== null ? state.meeting.gt_speakers : "—";
  for (const lane of state.lanes) {
    const payload = lane.payload;
    const visibleK = new Set(payload.segments.map(function (segment) { return segment.spk; })).size;
    const row = el("tr", state.mappingLane === lane.lane ? "selected" : "");
    row.addEventListener("click", function () {
      state.mappingLane = lane.lane;
      renderMetrics();
      renderMapping();
    });
    row.innerHTML =
      '<td>' + lane.lane + laneBadge(lane.lane) + '</td>'
      + '<td class="num">' + payload.k + '</td>'
      + '<td class="num">' + gtK + '</td>'
      + '<td class="num">' + visibleK + '</td>'
      + '<td class="num">' + pctOrDash(payload.overall, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.clean, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.overlap, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.abstain, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.unmapped, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.wrong, 2) + '</td>'
      + '<td class="num">' + pctOrDash(payload.merged_overall, 2) + ' / '
      + derDisplay(payload.merged_der, payload.parity) + '</td>'
      + '<td class="num">' + derCell(payload) + '</td>'
      + '<td>' + payload.level + '</td>';
    body.append(row);
  }
}

/* ── 时间轴 ─────────────────────────────────────────────── */

function drawRuler(ruler, total) {
  const step = total <= 120 ? 10 : total <= 480 ? 30 : total <= 1200 ? 60 : 120;
  for (let t = 0; t <= total + 0.001; t += step) {
    const tick = el("i", "tick");
    tick.style.left = percent(t, total);
    ruler.append(tick);
    const last = t > 0 && t + step > total;
    const label = el("b", "tick-label" + (last ? " end" : ""), fmtTime(t));
    label.style.left = percent(t, total);
    ruler.append(label);
  }
}

/* 播放头 left 用轨道自己的 rect 换算（8736 按整层宽度算是 72 px 错位的老 bug）。 */
function playheadLeft() {
  const layer = $("tracklayer");
  const track = layer.querySelector(".track");
  if (!track || !state.total) return 0;
  const layerRect = layer.getBoundingClientRect();
  const trackRect = track.getBoundingClientRect();
  return (trackRect.left - layerRect.left) + (state.playheadS / state.total) * trackRect.width;
}

function updatePlayhead(show) {
  const head = document.querySelector(".playhead");
  if (!head) return;
  head.style.left = playheadLeft() + "px";
  if (show || state.audioReady) head.hidden = false;
}

function renderTimeline() {
  const layer = $("tracklayer");
  layer.textContent = "";
  const meeting = state.meeting;
  if (!meeting) {
    layer.append(el("p", "muted", "先选一场会议，再点运行。"));
    $("timeline-meta").textContent = "";
    $("timeline-note").textContent = "选择场次后显示时间轴。";
    return;
  }
  const total = timelineTotal();
  state.total = total;
  $("timeline-meta").textContent = meeting.id + " · 参照窗口 " + fmtTime(total)
    + " · 真值 " + meeting.gt_speakers + " 人";

  const rulerRow = el("div", "row ruler-row");
  rulerRow.append(el("span", "lane-label muted", "秒"));
  const ruler = el("div", "track ruler");
  rulerRow.append(ruler);
  layer.append(rulerRow);
  drawRuler(ruler, total);

  /* 真值轨：T5 的 /api/run 只下发预测 segments，真值段表没有随 API 下发；
     这里不伪造段块，只标参照窗口，读数位置显示 —。 */
  const truthRow = el("div", "row");
  truthRow.append(el("span", "lane-label truth", "真值"));
  const truthTrack = el("div", "track");
  const truthBlock = el("div", "blk truth-blk");
  truthBlock.style.left = "0";
  truthBlock.style.width = percent(Math.min(total, REFERENCE_WINDOW_S), total);
  truthBlock.title = "真值段表未随 API 下发（T5 只发预测 segments）；这里只标参照窗口";
  truthTrack.append(truthBlock);
  truthRow.append(truthTrack);
  layer.append(truthRow);

  for (const lane of state.lanes) {
    const row = el("div", "row");
    row.append(el("span", "lane-label", lane.lane + (lane.truthDerived ? "（读真值）" : "")));
    const track = el("div", "track");
    track.dataset.lane = lane.lane;
    for (const segment of lane.payload.segments) {
      const piece = el("div", "blk");
      piece.style.left = percent(segment.start, total);
      piece.style.width = percent(Math.max(segment.end - segment.start, 0.001), total);
      piece.style.background = predictionColor(segment.spk);
      piece.title = "spk " + segment.spk + " · " + fmtTime(segment.start) + "–" + fmtTime(segment.end);
      track.append(piece);
    }
    row.append(track);
    layer.append(row);
  }
  if (!state.lanes.length) {
    layer.append(el("p", "muted", "运行后出现 base / unlocked（oracle 开启时多一条）泳道。"));
  }

  const head = el("div", "playhead");
  head.hidden = true;
  layer.append(head);
  $("timeline-note").textContent = "点击轨道定位（音频就绪时同步播放）；真值段表未随 API 下发，"
    + "真值轨仅标参照窗口。预测每条泳道一条轨，不按真值人分组。";
}

function seekFromClick(event) {
  const track = event.target.closest(".track");
  if (!track) return;
  const rect = track.getBoundingClientRect();
  const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
  state.playheadS = fraction * state.total;
  updatePlayhead(true);
  const audio = $("audio");
  if (state.audioReady && audio.duration) {
    audio.currentTime = Math.min(state.playheadS, audio.duration);
    audio.play();
  } else if (!state.audioHinted) {
    state.audioHinted = true;
    logLine("音频未就绪：点击只移动播放头（T7 后 /api/audio 返回真音频）", "warn");
  }
}

/* ── 音频（T7 之前 /api/audio 只回 JSON 回执，页面降级不炸） ── */

async function loadAudio(meeting, seconds) {
  const audio = $("audio");
  const url = "/api/audio?meeting=" + encodeURIComponent(meeting) + "&seconds=" + Math.round(seconds);
  try {
    const response = await fetch(url);
    if (!response.ok) {
      logLine("音频请求失败：HTTP " + response.status, "warn");
      return false;
    }
    const type = response.headers.get("Content-Type") || "";
    if (type.indexOf("audio/") === 0) {
      audio.src = url;
      audio.disabled = false;
      state.audioReady = true;
      logLine("音频就绪：点击时间轴可定位播放");
      return true;
    }
    const receipt = await response.json();
    audio.removeAttribute("src");
    audio.disabled = true;
    state.audioReady = false;
    logLine("音频未就绪（T7 未实现）：" + (receipt.note || "服务端未返回音频")
      + "；clamped_seconds=" + receipt.clamped_seconds
      + "，时间轴仍可定位（仅移动播放头）", "warn");
    return false;
  } catch (error) {
    state.audioReady = false;
    logLine("音频不可用：" + (error.message || error), "warn");
    return false;
  }
}

/* ── 映射表（当前选中泳道的预测簇） ─────────────────────── */

function currentLane() {
  if (!state.lanes.length) return null;
  return state.lanes.find(function (lane) { return lane.lane === state.mappingLane; }) || state.lanes[0];
}

function renderMapping() {
  const body = $("mapping-body");
  body.textContent = "";
  const lane = currentLane();
  if (!lane) {
    const row = el("tr");
    row.append(el("td", "muted", "运行后按泳道列出预测簇。"));
    row.firstChild.colSpan = 5;
    body.append(row);
    $("mapping-note").textContent = "运行后按选中泳道列出预测簇。";
    return;
  }
  const clusters = new Map();
  for (const segment of lane.payload.segments) {
    if (!clusters.has(segment.spk)) clusters.set(segment.spk, { spk: segment.spk, seconds: 0 });
    clusters.get(segment.spk).seconds += Math.max(0, segment.end - segment.start);
  }
  for (const cluster of clusters.values()) {
    const row = el("tr");
    row.innerHTML = '<td><span class="swatch" style="background:' + predictionColor(cluster.spk)
      + '"></span>spk ' + cluster.spk + '</td>'
      + '<td class="muted">—</td>'
      + '<td class="num">—</td>'
      + '<td class="num">—</td>'
      + '<td class="num">' + cluster.seconds.toFixed(1) + ' s</td>';
    body.append(row);
  }
  $("mapping-note").textContent = "泳道 " + lane.lane + "：共 " + clusters.size
    + " 个预测簇；重叠/纯度/匹配真值人需要真值段表——T5 的 API 未下发，故标 —；簇时长由预测段表算出。";
}

/* ── 旋钮与初始化 ───────────────────────────────────────── */

function bindKnobs() {
  $("lam").addEventListener("input", function () {
    $("lam-value").textContent = Number($("lam").value).toFixed(3);
    renderStatusBar();
  });
  for (const id of ["min_win", "cap", "mode", "refine"]) {
    $(id).addEventListener("change", renderStatusBar);
  }
  $("oracle").addEventListener("change", function () {
    renderStatusBar();
    if ($("oracle").checked) {
      logLine("oracle 开启：k=2..10 网格逐个跑，读数读真值，分钟级", "warn");
    }
  });
  $("pin_base").addEventListener("change", function () {
    renderStatusBar();
    logLine($("pin_base").checked
      ? "base 已钉住：改 λ 不影响 base（生产口径）"
      : "base 已解钉：下次运行 base 跟随 λ（base 已非生产口径）", "warn");
  });
  $("run").addEventListener("click", run);
  $("tracklayer").addEventListener("click", seekFromClick);
}

function bindAudio() {
  const audio = $("audio");
  audio.addEventListener("timeupdate", function () {
    state.playheadS = audio.currentTime;
    updatePlayhead(true);
  });
  audio.addEventListener("error", function () {
    state.audioReady = false;
    logLine("播放器加载失败：音频未就绪（不影响指标与时间轴）", "warn");
  });
}

async function fetchParity() {
  const response = await fetch("/api/verify");
  if (!response.ok) return null;
  for await (const event of readNdjson(response)) {
    if (event.t === "done" && event.parity) return event.parity;
  }
  return null;
}

function parityLine(parity) {
  if (parity.status === "pass") {
    return "DER 对拍 " + parity.n_pass + "/" + parity.n_total + " pass · " + parity.checked_at
      + "（阈值 " + parity.threshold_pts + " 点）";
  }
  return "DER 对拍未通过：" + parity.status + "（DER 列显示 —（未对拍））";
}

async function init() {
  renderMergedHeader();
  renderStatusBar();
  renderMetrics();
  renderTimeline();
  renderMapping();
  bindKnobs();
  bindAudio();
  logLine("页面就绪：读 /api/meetings 与 /api/verify");
  try {
    const parity = await fetchParity();
    if (parity) {
      state.parity = parity;
      renderStatusBar();
      logLine(parityLine(parity));
    }
  } catch (error) {
    logLine("对拍状态读取失败：" + (error.message || error), "warn");
  }
  try {
    await loadMeetings();
    deepLink();
  } catch (error) {
    logLine("场次目录读取失败：" + (error.message || error), "bad");
  }
}

init();
