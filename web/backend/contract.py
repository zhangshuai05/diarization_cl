# -*- coding: utf-8 -*-
"""页面契约：泳道 payload 的字段名单（唯一来源，T3/T5/T6 都按这个名字用）。

T3 `lanes.run_lanes` 按这些键组装 payload，T5 `/api/run` 原样推 ndjson，
T6 `app.js` 里有一份同名字面量（由 T6 的契约测试逐字比对）。

`metrics.sentence_metrics` / `der_metrics` / `with_merged` 的返回键是这里的子集，
不多不少 —— `web/tests/test_metrics.py::test_returned_keys_are_in_contract` 锁住。
"""

# 顺序 = 页面指标表的阅读顺序：身份 → 句级 → 消重 → DER → 段表。
#
# * 比率字段全是 **0–1 小数**（页面再乘 100）。
# * `overall` / `clean` / `overlap` = 句级主指标（**按句平均**）的精确值；
#   `abstain` / `unmapped` / `wrong` = 全体的三类比率（精确计数）；
#   四者互斥完备、相加恒为 1（口径见 `metrics` 模块 docstring）。
# * `der` / `miss_rate` / `false_alarm_rate` / `confusion_rate` 同属一次 `der()` 的分解。
# * `segments` 是唯一的列表（T1 的生产段表），其余是标量或 None。
LANE_PAYLOAD_KEYS: tuple[str, ...] = (
    "lane", "k", "level",
    "overall", "clean", "overlap",
    "abstain", "unmapped", "wrong",
    "merged_overall", "merged_der",
    "der", "miss_rate", "false_alarm_rate", "confusion_rate",
    "segments",
)

# 「读真值」名单（页面必须给这些名字带徽标，不许写死在 app.js 里）：值由读真值算出，
# 不是纯预测能拿到的分数。两种名字都算「暴露给页面」：
#   * `merged_overall` / `merged_der` —— 字段名（在 `LANE_PAYLOAD_KEYS` 里），
#     由消重簇 recipe 经 `truth_coverage` 读真值算出（见 `metrics.with_merged`）；
#   * `oracle` —— **泳道名**，经 `payload["lane"]`（键名 `lane`）暴露给页面。
# 这条与 spec §6.2 同源，但把范围从 oracle 扩到 merged 两列（R1 审查发现）。
TRUTH_DERIVED: tuple[str, ...] = ("oracle", "merged_overall", "merged_der")
