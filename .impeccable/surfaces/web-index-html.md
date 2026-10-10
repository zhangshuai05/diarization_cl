---
version: 1
slug: "web-index-html"
primary_target: "web/index.html"
related_targets: []
---

## THESIS

一个调试台只服务一个问题：**这场会议、这几种选 K 配置，差别在哪**。
它拒绝的类别默认是"仪表盘"：不做卡片网格、不做历史曲线、不做欢迎语——数字排成表、时间轴排成尺子，其余全部让位。

## OWN-WORLD

继承 `streaming_asr/bench/index.html` 的世界，不另起：暗底 `#0e1116`、面板 `#11161e`、1px 细线 `#1e2630`、数字等宽（`ui-monospace`）、正文灰阶三级。
颜色**只**承载一件事：区分说话人——真值一套（暖）、预测一套（冷），取自现有 `colorFor` 调色板；不可用色相表达状态。
组件语言：泳道行 = `[名字 72px][轨道]` 两列网格；轨道内块用绝对定位百分比；表头小号大写字母间距；旋钮是行内原生控件，不是自绘。

## STORY

用户打开就懂：左边选场次，顶部拧参数，中间这张表告诉我**哪条泳道在这一场上更好、好在哪一类**（对/弃权/未映射/错答、clean vs overlap、DER 的 miss/FA/conf）。
信任来自可见的口径：每条数字旁边标注口径与对拍状态，`base` 钉住生产值时标"生产口径"，`oracle` 标"读真值"。
动作是**换一场、拧一次、再看一遍**——不需要保存、不需要导出、不需要多点几下。

## FIRST VIEWPORT

顶到时间轴，不留滚动才能看全：
状态条（对拍状态 / base 是否钉住 / 旋钮是否偏离生产值）→ 三栏主体（左 260px 场次列表 · 中 自适应指标表 + 时间轴 · 右 320px 映射表与日志）→ 旋钮行嵌在中间栏顶部，与表头同一行视觉层级。
主要动作只有一个：**Run**（以及切换旋钮后自动重算），放在中栏顶部右侧、与指标表对齐，不浮层、不吸底。

## FORM

继承式扩展（impeccable `new-work` §3「Extend an existing surface」），世界与构图沿用 8736，只解决新目的、新层级、新状态与交互。
**未跑 concept-seed（extension 不需要 roll），因此没有 seed key**——按 reference 的规矩如实记录，不编造。

## FINISH

unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
