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

/* 真值说话人一套配色（--s*），预测簇一套（--p*）：spec §6.2「真值一套、预测一套」。 */
const TRUTH_COLORS = ["--s0", "--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"];

/* 事件层的键 = 泳道 payload 键 + pairs（映射表行）：pairs 由 /api/run 的编排层注入
   （泳道 payload 本身不带真值），与 contract.LANE_EVENT_KEYS 同名同序。 */
const LANE_EVENT_KEYS = LANE_PAYLOAD_KEYS.concat(["pairs"]);

const $ = (id) => document.getElementById(id);

const state = {
  meetings: [],
  meeting: null,
  parity: null,
  lanes: [],
  mappingLane: null,
  truth: null,                      // /api/truth 的 reference（当前场次；未选/未到为 null）
  truthGtK: null,
  truthLoading: false,
  total: REFERENCE_WINDOW_S,
  playheadS: 0,
  audioReady: false,
  audioHinted: false,
  runSeq: 0,                        // 串台防护（缺陷 a）：在飞 run 的序号，换场次/重跑即作废
  runAbort: null,
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
  const cell = el("td", "num", derDisplay(payload.der, payload.parity));
  if (!payload.parity || payload.parity.status === "pass") {
    cell.append(el("small", "", "漏 " + pctOrDash(payload.miss_rate, 1) + " · 虚 "
      + pctOrDash(payload.false_alarm_rate, 1) + " · 混 " + pctOrDash(payload.confusion_rate, 1)));
  }
  return cell;
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
  return isTruthDerivedLane(lane) ? el("span", "badge truth", "读真值") : null;
}

function renderMergedHeader() {
  const mergedKeys = LANE_PAYLOAD_KEYS.filter(function (key) { return key.indexOf("merged") === 0; });
  const header = $("merged-th");
  header.textContent = "merged";
  if (mergedKeys.some(isTruthDerivedField)) header.append(el("span", "badge truth", "读真值"));
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
    // stale：报告不是对着当前参照产物算的（参照变了 / 报告太旧）——数字还在，但凭据过期。
    const stale = parity.stale ? " ⚠参照已变（须重跑）" : "";
    badge.textContent = "DER 对拍 " + (parity.n_pass || 0) + "/" + (parity.n_total || 0) + " ✓ · " + date + stale;
    badge.className = "badge" + (parity.stale ? " warn" : " ok");
    badge.title = "checked_at=" + parity.checked_at + " · 阈值 " + parity.threshold_pts + " 点"
      + (parity.stale ? " · 参照产物内容已变，重跑 python -m web.backend.verify_der" : "");
  } else {
    badge.textContent = "DER 对拍未通过（" + parity.status + "）";
    badge.className = "badge bad";
    badge.title = parity.path || "";
    for (const item of parity.failed || []) logFailureOnce(item);   // 去重（缺陷 d）
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

/* 真值说话人（/api/truth 的段表）：按首次出现顺序固定配色槽，同一个名字每次渲染同色。 */
function truthSpeakers() {
  const names = [];
  for (const segment of state.truth || []) {
    if (names.indexOf(segment.speaker) < 0) names.push(segment.speaker);
  }
  return names;
}

function truthColor(name) {
  const index = truthSpeakers().indexOf(name);
  // 不在真值轨里的名字（pairs 比段表先到）落到第 0 槽，不另造色。
  const slot = TRUTH_COLORS[(index < 0 ? 0 : index) % TRUTH_COLORS.length];
  return getComputedStyle(document.documentElement).getPropertyValue(slot).trim() || "#94a3b8";
}

/* `pairs[].pred` 是 `#N`（8736 口径）：配色槽按同一个 N 取，与时间轴的预测轨同色。 */
function predIndex(pred) {
  const value = Number(String(pred).replace("#", ""));
  return Number.isFinite(value) ? Math.abs(value) : 0;
}

/* 单元格工具：`null`/`undefined` 一律 —（与 pctOrDash 的空值惯例一致）。 */
function td(text, className) {
  return el("td", className, text === null || text === undefined ? "—" : String(text));
}

/* 匹配真值人：带真值配色的小色块 + 人名；无匹配给 —（不是空单元格）。 */
function refCell(name) {
  const cell = el("td");
  if (name === null || name === undefined) {
    cell.append(el("span", "muted", "—"));
    return cell;
  }
  const swatch = el("span", "swatch");
  swatch.style.background = truthColor(name);
  cell.append(swatch, String(name));
  return cell;
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

/* 失败清单去重（缺陷 d）：parity 非 pass 时 renderStatusBar 会被反复调用，
   同一项只写一次日志（换一批失败项才追加），不再每次 render 重刷整份清单。 */
const loggedFailures = new Set();

function logFailureOnce(item) {
  const text = JSON.stringify(item);
  if (loggedFailures.has(text)) return;
  loggedFailures.add(text);
  logLine("对拍未通过：" + text, "bad");
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
    button.dataset.id = meeting.id;              // 深链按它精确匹配（缺陷 c）
    button.setAttribute("aria-pressed", "false");
    // 服务端字符串一律 textContent / createElement（缺陷 b），不拼 HTML 字符串
    button.append(el("span", "m-id", meeting.id));
    button.append(el("span", "m-meta", meeting.cohort
      + (meeting.has_ref ? " · 有参照" : " · 无参照")
      + " · " + meeting.gt_speakers + " 人 · " + fmtTime(meeting.duration)));
    button.addEventListener("click", function () { selectMeeting(meeting, button); });
    item.append(button);
    list.append(item);
  }
  $("meeting-count").textContent = state.meetings.length + " 场";
  logLine("场次目录就绪：" + state.meetings.length + " 场");
}

function selectMeeting(meeting, button) {
  invalidateRun();                             // 缺陷 a：在飞的 run 立即作废并断开
  state.meeting = meeting;
  state.lanes = [];                            // 上一场的泳道不许留在新场次的时间轴上
  state.mappingLane = null;
  state.truth = null;                          // 真值轨等 /api/truth（按场次重取）
  state.truthGtK = null;
  state.truthLoading = true;
  for (const other of document.querySelectorAll(".meeting")) {
    other.setAttribute("aria-pressed", other === button ? "true" : "false");
  }
  $("run").disabled = false;
  $("run").textContent = "运行";            // 旧 run 的 finally 不会再碰按钮（token 已过期）
  $("run-meta").textContent = meeting.id + " · 窗口 " + fmtTime(timelineTotal());
  logLine("选中 " + meeting.id + "（真值 " + meeting.gt_speakers + " 人 · "
    + (meeting.has_ref ? "有参照" : "无参照") + "）");
  renderMetrics();
  renderTimeline();
  renderMapping();
  loadTruth(meeting);
}

/* 真值轨与映射表的数据源（spec §5 的 /api/truth）：选中场次就拉。
   回来时场次已经换了就丢弃（与 run 同一个串台思路，缺陷 a）。 */
async function loadTruth(meeting) {
  const requested = meeting.id;
  try {
    const response = await fetch("/api/truth?meeting=" + encodeURIComponent(requested));
    const payload = await response.json().catch(function () { return null; });
    if (!response.ok) {
      throw new Error(payload && payload.error ? payload.error : "HTTP " + response.status);
    }
    if (!state.meeting || state.meeting.id !== requested) return;
    state.truth = payload.reference || [];
    state.truthGtK = payload.gt_k;
    state.truthLoading = false;
    logLine("真值段表就绪：" + state.truth.length + " 段 · " + payload.gt_k + " 人 · "
      + (payload.has_ref ? "有参照" : "无参照"));
  } catch (error) {
    if (!state.meeting || state.meeting.id !== requested) return;
    state.truth = [];
    state.truthGtK = null;
    state.truthLoading = false;
    logLine("真值读取失败：" + (error.message || error), "warn");
  }
  if (state.meeting && state.meeting.id === requested) {
    renderTimeline();
    renderMapping();
  }
}

/* 深链：/?meeting=<id>[&run=1] 打开即选中（可选即跑）——刷新/转发同一次运行。
   `meeting` 与按钮上的 `data-id` **精确相等**（子串匹配会把 A 场开给 B 场，缺陷 c）。 */
function deepLink() {
  const query = new URLSearchParams(location.search);
  const wanted = query.get("meeting");
  if (!wanted) return;
  const button = [].slice.call(document.querySelectorAll(".meeting")).find(function (item) {
    return item.dataset.id === wanted;
  });
  if (!button) return;
  button.click();
  if (query.has("run")) run();                 // 只认 ?run，不新增自动行为
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
  const missing = LANE_EVENT_KEYS.filter(function (key) { return !(key in event); });
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

/* ── 串台防护（缺陷 a） ─────────────────────────────────────
   一次 run 领一个 (序号, 场次) token；换场次或重跑都会让旧 token 过期。旧流的事件在
   `run()` 里**进 acceptLane 之前**丢弃，同时把在飞的 fetch abort 掉（服务端 `_emit`
   看到写失败就提前收摊、还原 λ）。 */
function beginRun(meetingId) {
  if (state.runAbort) state.runAbort.abort();
  state.runAbort = new AbortController();
  state.runSeq += 1;
  return { seq: state.runSeq, meeting: meetingId, abort: state.runAbort };
}

function invalidateRun() {
  state.runSeq += 1;
  if (state.runAbort) {
    // 在飞的 run：记一行（这一场的结果不再要了），断掉 fetch，让服务端提前收摊
    if (state.meeting) logLine("已丢弃 " + state.meeting.id + " 的在飞结果（已换场次或重跑）", "warn");
    state.runAbort.abort();
    state.runAbort = null;
  }
}

function runTokenStale(token) {
  return token.seq !== state.runSeq || !state.meeting || state.meeting.id !== token.meeting;
}

async function run() {
  if (!state.meeting) return;
  const meeting = state.meeting;
  const token = beginRun(meeting.id);
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
    const response = await fetch("/api/run?meeting=" + encodeURIComponent(meeting.id) + "&" + knobsQuery(),
      { signal: token.abort.signal });
    if (!response.ok) {
      const payload = await response.json().catch(function () { return null; });
      throw new Error(payload && payload.error ? payload.error : "HTTP " + response.status);
    }
    for await (const event of readNdjson(response)) {
      if (runTokenStale(token)) return;   // 缺陷 a：旧流一律不画（作废时已记过日志）
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
    if (!runTokenStale(token)) await loadAudio(meeting.id, state.total);
  } catch (error) {
    if (error && error.name === "AbortError") return;    // 自己断开的：静默收摊，不当失败
    if (runTokenStale(token)) return;                    // 旧流的网络错误也不画、不刷日志
    logLine("运行失败：" + (error.message || error), "bad");
  } finally {
    if (!runTokenStale(token)) {                         // 别把新 run 的按钮/耗时复位掉
      $("run").disabled = false;
      $("run").textContent = "运行";
      logLine("本次耗时 " + ((performance.now() - started) / 1000).toFixed(2) + " s（含网络与渲染）");
    }
  }
}

/* ── 指标表 ─────────────────────────────────────────────── */

function renderMetrics() {
  const body = $("metrics-body");
  body.textContent = "";
  if (!state.lanes.length) {
    const cell = el("td", "muted", "运行后每条泳道一行；空读数（后端 null）显示 —。");
    cell.colSpan = 13;
    const row = el("tr");
    row.append(cell);
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
    // 服务端字符串（泳道名）只走 textContent（缺陷 b）
    const nameCell = el("td", "", lane.lane);
    const badge = laneBadge(lane.lane);
    if (badge) nameCell.append(" ", badge);
    row.append(nameCell);
    row.append(td(payload.k, "num"));
    row.append(td(gtK, "num"));
    row.append(td(visibleK, "num"));
    row.append(td(pctOrDash(payload.overall, 2), "num"));
    row.append(td(pctOrDash(payload.clean, 2), "num"));
    row.append(td(pctOrDash(payload.overlap, 2), "num"));
    row.append(td(pctOrDash(payload.abstain, 2), "num"));
    row.append(td(pctOrDash(payload.unmapped, 2), "num"));
    row.append(td(pctOrDash(payload.wrong, 2), "num"));
    row.append(td(pctOrDash(payload.merged_overall, 2) + " / "
      + derDisplay(payload.merged_der, payload.parity), "num"));
    row.append(derCell(payload));
    row.append(td(payload.level));
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
  const reference = state.truth || [];
  const truthNames = truthSpeakers();
  $("timeline-meta").textContent = meeting.id + " · 参照窗口 " + fmtTime(total)
    + " · 真值 " + (state.truth ? truthNames.length : meeting.gt_speakers) + " 人";

  const rulerRow = el("div", "row ruler-row");
  rulerRow.append(el("span", "lane-label muted", "秒"));
  const ruler = el("div", "track ruler");
  rulerRow.append(ruler);
  layer.append(rulerRow);
  drawRuler(ruler, total);

  /* 真值轨（spec §6.1 的第一条轨）：段表来自 /api/truth，与各泳道同一把尺子、真值一套配色。 */
  const truthRow = el("div", "row");
  truthRow.append(el("span", "lane-label truth", "真值"));
  const truthTrack = el("div", "track");
  truthTrack.dataset.lane = "truth";
  for (const segment of reference) {
    const piece = el("div", "blk");
    piece.style.left = percent(segment.start, total);
    piece.style.width = percent(Math.max(segment.end - segment.start, 0.001), total);
    piece.style.background = truthColor(segment.speaker);
    piece.title = segment.speaker + " · " + fmtTime(segment.start) + "–" + fmtTime(segment.end);
    truthTrack.append(piece);
  }
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
      piece.title = "#" + segment.spk + " · " + fmtTime(segment.start) + "–" + fmtTime(segment.end);
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
  const truthNote = state.truthLoading
    ? "真值轨读取中（/api/truth）"
    : (reference.length
      ? "真值轨 " + reference.length + " 段 · " + truthNames.length + " 人"
      : "该场无真值段表（/api/truth 返回空）");
  $("timeline-note").textContent = "点击轨道定位（音频就绪时同步播放）；" + truthNote
    + "，与泳道同一把尺子；预测每条泳道一条轨，不按真值人分组。";
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
    logLine("音频未就绪：点击只移动播放头（源 flac 缺失时 /api/audio 回降级回执）", "warn");
  }
}

/* ── 音频（/api/audio 回真 audio/wav；源 flac 缺失时回 JSON 降级回执，页面降级不炸） ── */

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
    logLine("音频未就绪（服务端回降级回执）：" + (receipt.note || "服务端未返回音频")
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
    const cell = el("td", "muted", "运行后按泳道列出预测簇。");
    cell.colSpan = 5;
    const row = el("tr");
    row.append(cell);
    body.append(row);
    $("mapping-note").textContent = "运行后按选中泳道列出预测簇。";
    return;
  }
  const truthK = state.truthGtK === null || state.truthGtK === undefined ? "—" : state.truthGtK;
  const pairs = lane.payload.pairs || [];
  let shown = pairs.length;
  if (pairs.length) {
    /* 服务端下发的映射表行（口径见 metrics.pairs）：一簇一行，值是最大重叠那份。 */
    for (const pair of pairs) {
      const row = el("tr");
      const predCell = el("td");
      const swatch = el("span", "swatch");
      swatch.style.background = predictionColor(predIndex(pair.pred));
      predCell.append(swatch, String(pair.pred));
      row.append(predCell);
      row.append(refCell(pair.ref));
      const matched = pair.ref !== null && pair.ref !== undefined;
      row.append(td(matched ? Number(pair.overlap).toFixed(1) + " s" : "—", "num"));
      row.append(td(matched ? (Number(pair.share) * 100).toFixed(1) + "%" : "—", "num"));
      row.append(td(Number(pair.pred_seconds).toFixed(1) + " s", "num"));
      body.append(row);
    }
  } else {
    /* 无 pairs（旧事件/真值未到）：仍按段表列簇与簇时长，真值三列记 —。 */
    const clusters = new Map();
    for (const segment of lane.payload.segments) {
      if (!clusters.has(segment.spk)) clusters.set(segment.spk, { spk: segment.spk, seconds: 0 });
      clusters.get(segment.spk).seconds += Math.max(0, segment.end - segment.start);
    }
    for (const cluster of clusters.values()) {
      const row = el("tr");
      const predCell = el("td");
      const swatch = el("span", "swatch");
      swatch.style.background = predictionColor(cluster.spk);
      predCell.append(swatch, "#" + cluster.spk);
      row.append(predCell);
      row.append(td(null));
      row.append(td(null, "num"));
      row.append(td(null, "num"));
      row.append(td(cluster.seconds.toFixed(1) + " s", "num"));
      body.append(row);
    }
    shown = clusters.size;
  }
  $("mapping-note").textContent = "泳道 " + lane.lane + "：" + shown + " 个预测簇 / 真值 "
    + truthK + " 人（gt_k）；匹配按重叠秒数取最大（与 DER 的 collar 口径不同），"
    + "未匹配到真值人的簇记 —；簇时长由预测段表算出。";
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
      + "（阈值 " + parity.threshold_pts + " 点）"
      + (parity.stale ? " ⚠ 参照产物已变，结论须重跑确认" : "");
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
