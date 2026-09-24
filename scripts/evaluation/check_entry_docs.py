# -*- coding: utf-8 -*-
"""入口文档自洽体检：新克隆的第一读者能不能得到一致的说法。

为什么需要它：交接文档 / 发现记录 / 交付表 / README 四份文档都被反复追加更正，
而"更正"天然要**引用旧说法**。天真的字符串搜索会把引文当残留报假阳性
（本会话就报过一次：三处"残留"全是正当引文），所以这里的判据是
**命中点是否处在更正语境里**，而不是命中与否。

判据：
  1. 同一主张的数值是否在多份文档里一致（不一致 = 真问题）
  2. 旧说法是否出现在**没有**更正标记的段落里（有"已解 / 更正 / 推翻 / 是错的"等
     标记的，视为正当引文）

    python -u -B scripts/evaluation/check_entry_docs.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = [("README.md", ROOT / "README.md"),
        ("发现记录", ROOT / "artifacts/v13_会话发现_20260923.md"),
        ("交付合并表", ROOT / "artifacts/v13_交付合并表.md"),
        ("交接文档", ROOT / "交接文档_0923.md")]

# 被推翻的旧说法 -> 允许它们出现的语境标记
STALE = {
    "并人藏在 FA": ("更正", "错的", "旧"),
    "必须同时看 FA": ("更正", "错的", "旧"),
    "逐位相同、因此无效": ("更正", "错的", "推翻"),
    "尚未解决的自相矛盾": ("已解", "更正", "作废"),
    "无法从仓库侧追查": ("已解", "已复现", "更正"),
    "61 passed": ("过期", "已更正"),
}
CONTEXT = re.compile(r"已解|更正|推翻|错的|作废|过期|已复现|口径差")

# 同一主张必须一致的数值
CLAIMS = [("dev 基线", "24.4%"), ("dev 解锁", "14.8%"),
          ("holdout 基线", "19.8%"), ("holdout 解锁", "10.2%")]


def main():
    text = {name: p.read_text(encoding="utf-8", errors="replace")
            for name, p in DOCS if p.exists()}
    bad = 0

    print("=== 1. 旧说法是否出现在没有更正标记的段落 ===")
    for phrase, markers in STALE.items():
        for name, body in text.items():
            for m in re.finditer(re.escape(phrase), body):
                # 命中点前后 3 行内是否有更正标记
                start = body.rfind("\n", 0, max(0, m.start() - 300))
                window = body[max(0, start):m.end() + 300]
                if CONTEXT.search(window):
                    continue                     # 正当引文
                bad += 1
                line = body[:m.start()].count("\n") + 1
                print(f"  ✗ {name}:{line}  「{phrase}」无更正语境")
    if bad == 0:
        print("  全部处在更正语境里（引文），无裸残留")

    print("\n=== 2. 同一主张的数值一致性 ===")
    for label, value in CLAIMS:
        have = [name for name, body in text.items() if value in body]
        ok = len(have) >= 2
        if not ok:
            bad += 1
        print(f"  {'✓' if ok else '✗'} {label:<12}{value:<8} 出现在 {', '.join(have) or '（无）'}")

    print(f"\n  -> 问题 {bad} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
