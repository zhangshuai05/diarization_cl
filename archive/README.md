# archive —— 退役产物

> 归档于 2026-10-08。**只读参考**。目录内文件不改内容，只是从仓库根移进来。
> 想复原任意一项：把它移回仓库根即可（`big_output/` 下的原路径就是根下同名）。

## 内容

| 子目录 | 内容 | 为什么在这 |
|---|---|---|
| `legacy_versions/` | `a2_v2_before_20260908.py`、`claude_a2_before_20260908.py` | c1 v4 之前的旧版源码快照 |
| `legacy_docs/9_plan/` | 九月工作计划 xlsx/docx、流程图 | 计划文档，无代码引用 |
| `legacy_docs/paper/` | `papers_for_c1.md` | 文献笔记，无代码引用 |
| `big_output/diar_result/`（56M, 11191 文件） | 2026-09 的推理输出、DER 明细、fusion 结果 | 大体积产物。**`docs/` 里 4 篇文档仍以 `diar_result/...` 路径引用它**，见下 |
| `big_output/cv_ss_test/`（687M） | 分离式提纯（MossFormer2_SS）实验 | 该方向**已否证**（轮替对话上退化成对半拆），产物留作证据 |
| `big_output/checkpoints/`（154M） | FRCRN_SE_16K 权重 | 模型权重，可从远端重取 |
| `big_output/diar_out/` | 早期 `pipeline_diarize.py` 输出 | 一次性产物 |

## 引用偏移（照抄，别当 bug）

移进 `big_output/` 后，下列**绝对路径引用仍指仓库根**，因此已失效：

- `docs/C1_优化复测_20260908.md:38,54`、`docs/交接文档_说话人分离_20260908.md:148`、
  `docs/交接文档_说话人分离_20260909.md:32,130`、`docs/工作日志.md:59,71,123,247,272`
  → 提到 `diar_result/...`。这些是**历史文档**，路径当作"当年位置"读。
- `scripts/audio/`、`scripts/batch/` 里 `OUT = os.path.join(BASE, "diar_result", ...)`
  → 这些脚本**下次运行会重建 `diar_result/`**（不是读不到而报错），需要时改 `BASE`。
- `archive/big_output/cv_ss_test/*.py` 里有硬编码 `C:\Users\EDY\Desktop\new\cv_ss_test\...`
  → 该实验已否证，**不打算再跑**，故未改。

`evidence/`（c1 里程碑证据）、`v12_meeting/`（v12 归档）、`milestones/`、`not_mind/`、
`dsh_line/`、`eval_dsh/` **均未移动**：前两个是被 `SHA256.json` / 活脚本引用的现行文件，
后四个分别是版本归档、无关目录、另一条线的只读交接包、以及被多处脚本读的真值目录。

## 顺带一提

`docs/音频位置与复现入口.md`（原根目录 `Audio_音频位置.txt`）提到的
`archive/_diag_segment_unit.py` **在本仓库中不存在**，原文件已不在——该行是旧路径残留。
