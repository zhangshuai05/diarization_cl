"""Build the voice-algorithm engineer textbook PDF in one reproducible step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


BOOK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BOOK_DIR.parent
OUTPUT = PROJECT_DIR / "语音算法工程师教学.pdf"
FIGURE_SCRIPT = BOOK_DIR / "figures_teaching.py"
MATHTYPE_SCRIPT = BOOK_DIR / "math" / "generate_mathtype.py"
FORMULA_SOURCE = BOOK_DIR / "math" / "formula_sources.json"
ASSET_DIR = BOOK_DIR / "math" / "assets"

CONTENT_ORDER = [
    BOOK_DIR / "content" / "00_intro.md",
    *(BOOK_DIR / "content" / f"ch{i:02d}.md" for i in range(1, 14)),
    *(BOOK_DIR / "content" / f"appendix-{c}.md" for c in "abcdefg"),
]

PART_STARTS = {
    "ch01.md": "第一篇  地基：从岗位到声音",
    "ch05.md": "第二篇  ASR 主线：对齐、解码与流式",
    "ch09.md": "第三篇  说话人与多任务音频",
    "ch11.md": "第四篇  系统工程：压缩、部署与交付",
    "appendix-a.md": "附录  学习路线、工具与证据模板",
}


def _run(label: str, command: list[str]) -> None:
    print(f"[{label}] {' '.join(command)}")
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _check_sources() -> None:
    missing = [str(path) for path in CONTENT_ORDER if not path.exists()]
    if missing:
        raise FileNotFoundError("missing content file(s): " + ", ".join(missing))
    data = json.loads(FORMULA_SOURCE.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"formula source file is empty or malformed: {FORMULA_SOURCE}")
    missing_assets = [str(ASSET_DIR / f"{entry['id']}.png") for entry in data if not (ASSET_DIR / f"{entry['id']}.png").exists()]
    if missing_assets:
        raise FileNotFoundError("missing MathType PNG asset(s): " + ", ".join(missing_assets))


def build(output: Path = OUTPUT, *, regenerate: bool = True) -> Path:
    if regenerate:
        _run("figures", [sys.executable, str(FIGURE_SCRIPT)])
        _run("mathtype", [sys.executable, str(MATHTYPE_SCRIPT), "--source-file", str(FORMULA_SOURCE), "--output-dir", str(ASSET_DIR)])
    _check_sources()

    # The renderer uses a package-relative import (engine.renderer).  Adding
    # voice_book to sys.path keeps this script runnable both as a file and from
    # an IDE without requiring a project-wide install.
    if str(BOOK_DIR) not in sys.path:
        sys.path.insert(0, str(BOOK_DIR))
    from engine.renderer import BookDocTemplate, build_story
    from engine.style import register_fonts

    register_fonts()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = BookDocTemplate(
        str(output),
        title="语音算法工程师教学：从声学前端到端侧实时系统",
        author="study",
    )
    story = build_story(CONTENT_ORDER, PART_STARTS)
    doc.multiBuild(story)
    print(f"built {output} ({output.stat().st_size:,} bytes)")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT, help="output PDF path")
    parser.add_argument("--skip-regenerate", action="store_true", help="reuse existing figures and MathType assets")
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else (PROJECT_DIR / args.output).resolve()
    build(output, regenerate=not args.skip_regenerate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
