# -*- coding: utf-8 -*-
"""
解析 Transcribe.xlsx 人工真值：归一化说话人标签（wyz->wzy），导出 JSON。
用法: python inspect_transcribe.py
"""
import openpyxl, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "Audio", "Transcribe.xlsx")
OUT = os.path.join(HERE, "output", "ground_truth.json")

wb = openpyxl.load_workbook(XLSX)
ws = wb["Sheet1"]
rows = list(ws.iter_rows(values_only=True))

ALIAS = {"wyz": "wzy"}  # 混写归一化

gt = []
for r in rows[2:]:
    dist, txt, fname, who, title = (list(r) + [None] * 5)[:5]
    if not fname or not isinstance(fname, str) or not fname.endswith(".wav"):
        continue
    turns = []
    if txt and isinstance(txt, str):
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^([A-Za-z]+)\s*[:：]\s*(.*)$", line)
            if m:
                spk = ALIAS.get(m.group(1).lower(), m.group(1).lower())
                text = m.group(2).strip()
                if text:
                    turns.append({"spk": spk, "text": text})
    gt.append({
        "file": fname,
        "distance": str(dist),
        "annotator": who,
        "n_turns": len(turns),
        "speakers": sorted(set(t["spk"] for t in turns)),
        "turns": turns,
    })

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

print(f"files with ground truth: {len(gt)}")
for g in gt:
    print(f"  {g['file']:<28} dist={g['distance']:<5} spks={g['speakers']} turns={g['n_turns']}")
print("saved:", OUT)
