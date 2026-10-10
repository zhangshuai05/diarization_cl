# web/ —— AISHELL-4 分离调试台（8737）

给第一次进这个目录的人：选一场会议 → 跑三条选 K 路线（`base` / `unlocked` / `oracle`）
→ 在同一把 600 s 尺子上看句级指标、DER、真值/预测时间轴与簇映射表。

**不加载模型**：读数来自探针缓存里的窗嵌入，跑的是生产同一批函数（`web/backend/cache_path.py`）；
**不新增依赖**：numpy / scipy / soundfile 都是仓库既有（见根 `requirements.txt`），ffmpeg 只在切音频时用到。

## 1. 跑法

```bash
cd <仓库根>
python web/server.py          # → http://127.0.0.1:8737/
```

* 服务只绑 **127.0.0.1:8737**（`server.PORT`；`allow_reuse_address=False`，端口被占就启动失败，
  不静默分裂流量）。
* 页面 = `web/index.html` + `app.js` + `style.css`，由本服务静态提供；`/api/*` 的行为见
  `server.py` 模块 docstring。
* 依赖：仓库既有 `numpy` / `scipy` / `soundfile`。`ffmpeg` 需在 PATH 上——**只有** `/api/audio`
  切片用它；缺了其余功能照常，切片请求回可读错误。

数据根（本机路径）：

| 用途 | 位置 | 说明 |
|---|---|---|
| 真值（TextGrid / rttm / flac） | `AISHELL4_TEST`，默认 `D:\yinp\AISHELL-4\data\test` | `<root>/TextGrid/<meeting>.rttm`、`<root>/wav/<meeting>.flac`；train_L 根 = `<root>` 的父目录下 `train_L` |
| 探针缓存（算法输入） | `D:\yinp\AISHELL-4\_v13_spike\cache\probe` | `probe_path_equiv.PROBE_CACHE`，写死；`manifest.json` + 每场 `main.npz` / `refine.npz` / `rescue.npz`，**50 场** |
| 音频切片缓存 | `web/clips/<meeting>_<seconds>.wav` | 首次播放时由 ffmpeg 生成；已 gitignore；**无上限 / TTL**（见限制 7） |

⚠️ `AISHELL4_TEST` 只覆盖 `/api/truth` 与 `/api/audio` 的真值读取；**泳道指标不读这个环境变量**——
`lanes.DATA_ROOTS` 是写死的 test + train_L 两棵树。

## 2. 读数口径（页面上的数字从哪来）

**段表**：`web/backend/cache_path.py::segments_for` —— 读缓存的三阶段窗嵌入
（main 2.0/1.0 s、refine 0.6/0.15 s、rescue 0.6/0.15 s），喂进生产函数
（`v11._candidate_clusters` / `_select_candidate` / `_centroids`、`v12._refine_labels` /
`_segment_windows`）。即「**缓存 + 生产同一份段表**」，不是 `v12.diarize` 本体（不跑模型，
也不写第二套段边界公式）。每条泳道的 `level` 标出这条读数来自哪套窗：`2s-only` / `refined` / `rescued`。

**三条泳道**（`lanes.run_lanes`，每条 yield 一次）：

| 泳道 | 选 K 怎么选 | 默认 |
|---|---|---|
| `base` | 生产选择器（`v11._candidate_clusters` + `_select_candidate`）：合格门 = 最小簇 ≥ 4 窗且占比 ≥ 1.5%，取 objective（轮廓 − λ·(k−2)）峰值，轮廓 ≤ 0 退回 k=1；`cache_path` 显式传 `max_speakers=10` | `pin_base=on` 时 λ 钉在生产常量 0.005，旋钮动不了它 |
| `unlocked` | `unlocked_selector.unlocked(min_win, cap, mode)` 替换选择器（孤窗不参与合格判定与评分），λ = 旋钮值 | `min_win=2, cap=10, mode=objective, λ=0.005` |
| `oracle` | k=2..10 网格强制候选，取句级**双射**（`overall`）峰值，平局取小 k | 默认关；**读真值** |

**句级指标**（`metrics.sentence_metrics`，口径 = `probe_sentence_attribution.attribution`）：

* `overall` / `clean`（无重叠句）/ `overlap`（重叠句）三栏，页面按百分数显示。
* `overall = (n − 弃权 − 未映射 − 错答) / n`，用**精确计数**（残差 = 命中句数）；
  `弃权` / `未映射` / `错答` 三类 + 命中**四类互斥完备，相加恒为 1**。`错答 = n_wrong / n`
  也是精确计数，**不是**「1 − 对 − 弃权 − 未映射」的减法链（三处取整误差会全落到错答头上）。
* `clean` / `overlap` 取自 `attribution` 的 4 位取整值，只用于展示、不参与恒等式。
* 分母为 0 的场次（没有句子）比率给 `None`，页面显示 `—`，不是 0%。

**DER**：`der.py::der(reference, predicted, collar=0.25)`，帧级 union 分母；参考段截到 600 s。
段表口径 = **`v12.diarize` 的过滤前段**（`run_claude_lane` 的 `SLIVER_WINDOWS` 过滤**之前**
那一份）——对拍时二分验证过：`L_R003S01C02` base 臂，过滤前 68 段 / DER 0.7153 = 参照；
过滤后 46 段 / 0.7546，差 3.9 点 ⇒ 用过滤后口径 24 项会一片假红。

**DER 门禁**：`artifacts/web_der_parity.json` 的 `summary.status == "pass"` 才在页面渲染 DER，
否则显示 `—（未对拍）`。当前报告：

* `checked_at = 2026-10-10T16:44:37`，dev 12 场 × base/fixed 两臂 = **24/24 pass**，
  最大偏差 **0.0 点**（阈值：逐项严格 `< 0.05` 点），耗时 22.8 s。
* 重跑：`python -m web.backend.verify_der`（约 15–25 s）；只看状态：`python -m web.backend.verify_der --status`。
* **规模也是判定的一部分**：门禁要求恰好 12 场 × 2 臂 = 24 行、每场两臂齐全（参照被裁小、
  缺 `base`/`fixed` 键 ⇒ `status=fail` + `shape_problems` 说明），并把参照产物的内容指纹
  写进报告 `reference.sha256`；指纹对不上（参照被改过 / 报告太旧）时页面徽标带 **⚠参照已变**
  ——盘上的 pass 不许冒充当前凭据。

**`merged` 两列**：消重复簇（`truth_coverage` → `classify_dup` → `merge_dups`）后**重走生产段表**
再打分；**读真值**。

**`pairs`（映射表行）**：每个预测簇按区间**最大重叠秒数**配真值人（重叠秒 / 纯度 = 重叠 ÷ 簇时长 /
簇时长）；与 DER 的 collar 映射、句级双射都不是一回事——只看谁占得最多，无重叠则 `ref=None`
（页面显示 `—`，不瞎配）。

**「读真值」徽标**：`merged_overall` / `merged_der` / `oracle` 三个名字由 `contract.TRUTH_DERIVED`
定义，页面据此加徽标（不在页面写死）。它们是诊断上界，**不是可部署判据**。

## 3. 不变量

* `base` 钉住（`pin_base=on`，默认）= **生产口径**：λ 用运行前的 `v11.MULTI_K_PENALTY`（现 0.005）、
  选择器 / 合格门 / 上限都是生产那份；页面徽标「base 已钉住（生产口径）」。
* `pin_base=off` 后 `base` 跟随 λ / min_win / cap / mode / refine 旋钮、**不再代表生产**；
  页面徽标改「base 已非生产口径」。这是唯一「名字叫 base 但不是生产口径」的路径。
* λ 是模块常量：`lanes._lam` 在调用期间临时替换、**`try/finally` 无条件还原**；oracle 的
  `_KNOB_LOCK` 把整条流水线锁在单线程，`/api/run` 在进程内串行化（并发 409，不排队）。

## 4. 已知限制（写实，不美化）

1. **只有 dev 12 场被门禁对拍过**。缓存覆盖 50 场 = dev 12 + holdout 8 + train_L 30；
   `has_ref`（= 在 `artifacts/v13_ab_dev_results.json` 里）只有 12 场 True，其余 **38 场无参照**。
   DER 门禁只对 dev 12 的 base/fixed 两臂负责，但页面门禁是全局开关：holdout / train_L 的
   DER、merged 照显示，只是没被对拍覆盖。
2. **参照产物是 λ=0.015 时生成的**（`artifacts/v13_e2e_unlocked2_dev_mw2_objective.json`）
   ⇒ 对拍里 `fixed` 臂必须按 **0.015** 钉住才逐位对上；**页面默认的 `unlocked`（λ=0.005）
   那条没被对拍覆盖**（只保证选择器与段表口径相同）。生产现值已是 0.005。
   该参照产物在 **2026-10-09** 被刷新过一次：只重跑了 `S_R004S01C01` 的 base 臂
   （k 4→5、DER 24.82→15.11），另 11 行沿用旧产物、`fixed` 臂未变，文件里 `baseline_refresh`
   记着来源。刷之前算出的门禁报告在干净 checkout 上不可复现（23/24）——这是终审 Critical A，
   参照产物入库后才对得上。
3. `merged` / `oracle` **读真值**（见上），页面带「读真值」徽标；引用它们时必须说是诊断口径。
4. **`refine=False` 关不掉 `k==1` 的密窗救援**：旋钮语义是「细化」（换掉 `v12._refine_labels`），
   救援（`rescue.npz` → 重跑候选）是另一机制；关掉细化后仍可能出现 `level=rescued`，
   优化上限里仍含救援收益。
5. **本台 `oracle` 的 k 与 `docs/HANDOFF.md §4.11` 的 `bijectiveK` 在 4/12 场不同**
   （`L_R003S01C02` 6 vs 7、`M_R003S01C01` 2 vs 5、`M_R003S02C01` 4 vs 10、`M_R003S04C01` 7 vs 8）：
   本台用**生产段表 + 精确计数**，探针用平铺段表 + 4 位取整。页面内 k / overall / DER 自洽，
   但不要与 §4.11 的表逐场对数字。
6. **`/api/run` 串行化**：进程内一次只准一条流，拿不到锁直接 **409、不排队**——两个页面同时点
   「运行」，第二个必失败（本机一条 1–2 s，重试很便宜）。**`/api/audio` 在源 flac 缺失时降级为
   `200 + JSON 回执`**（不是 500、也不是空 wav）；页面读回执后只保留播放头。
7. **`web/clips/` 无上限、无 TTL**：每场每秒数一个 wav，3600 s 单声道 16 kHz ≈ 115 MB；
   长期跑记得手动清目录。
8. **实测耗时**（本机，`L_R003S01C02`）：oracle 关 **1.3–1.9 s**、开 **1.8–2.3 s**；
   oracle 单场 **0.2–0.5 s**。spec / plan 里「≈1 分钟/场」是过时数字，别再抄。
9. 8736（`streaming_asr/bench/`）与本台功能重叠，处置见文末。

## 5. 测试

```bash
# 仓库根：web 调试台全部测试
python -m pytest web/tests -q          # 142 passed

# 真跑一次 DER 对拍（dev 12 场 × 2 臂，约 15–25 s，会重写 artifacts/web_der_parity.json）
python -m web.backend.verify_der
```

`web/tests` 不碰 D 盘：缓存用 tmp_path 合成 npz、真值用内存构造；只有 `test_audio.py` 的
3 条真实音频用例在数据 / ffmpeg 不可用时**跳过并说明**。

算法的 154 passed 必须在 `streaming_asr/` 下跑（那里有 pytest.ini 管 testpaths / pythonpath）：

```bash
cd streaming_asr && python -m pytest -q tests ../test_c1_refinement.py   # 154 passed
```

## 6. 附：8736（`streaming_asr/bench/`）功能对照

8736 是旧测试台（`python bench_server.py` → 127.0.0.1:8736）。「必须留下」的五项能力在本台的对应：

- [x] **① 场次列表（真值人数 / 时长 / split）** —— `/api/meetings` → 左侧「场次」：50 场
  （8736 只列 test 20 场），每行 `gt_speakers`（读 rttm 去重）、`duration`（读 flac header）、
  `cohort`（dev / holdout / train_L，用 `v12_aishell4.DEV_GROUPS/HOLDOUT_GROUPS` 同一套判据）、
  `has_ref`、`n_windows`（缓存窗数）。
- [x] **② 音频播放 + 点击定位** —— `/api/audio` + `<audio>` + 轨道点击（`app.js::seekFromClick`）：
  与 8736 同一条通道（ffmpeg `pan=mono|c0=c0` 取第 0 通道、16 kHz、缓存一次）。差异：8736 固定
  600 s，本台 `seconds` 钳 `[30, 3600]`（默认 600）；源 flac 缺失时回 JSON 回执、页面降级为
  只有播放头。
- [x] **③ 真值 / 预测时间轴** —— 轨道图叠在一把 600 s 尺子上：真值一条轨、每条泳道一条预测轨，
  点击定位（音频就绪时同步播放）。预测轨不按真值人分组，与泳道读数同口径。
- [x] **④ 映射表（重叠秒 / 纯度 / 簇时长）** —— 右侧「映射表」：列 = 预测簇 / 匹配真值人 /
  重叠秒 / 纯度 / 簇时长，随选中泳道刷新；无真值 / 无重叠时明确显示 `—`（不瞎配）。
- [x] **⑤ 跑分进度流** —— `/api/run` 的 ndjson 流 + 右侧「进度」日志：逐条推 `lane`
  （含指标与映射表行）/ `done`（含 `parity`）/ `error`。差异：8736 会加载 ASR 模型并插转写进度，
  本台**不加载模型**（只做分离），进度与结果同流到达。

> **删除 `streaming_asr/bench/`（8736）等用户确认后再做**——控制器会问，不要自行删除。
> 在用户点头前，两份都留着。
