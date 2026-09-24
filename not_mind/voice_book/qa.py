"""Automated quality checks for ``语音算法工程师教学.pdf``.

The checker deliberately separates hard failures (a missing asset, a damaged
PDF, or a block outside the physical page) from visual-review warnings.  The
latter include sparse pages and possible orphan headings: a cover, a table of
contents, a chapter opener, or a final page may quite reasonably contain more
white space than a normal body page.

Run from any directory with::

    python voice_book/qa.py

Reports and rendered pages are written below ``voice_book/qa``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import unicodedata
from urllib.parse import unquote

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont


BOOK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BOOK_DIR.parent
DEFAULT_PDF = PROJECT_DIR / "语音算法工程师教学.pdf"
CONTENT_DIR = BOOK_DIR / "content"
QA_DIR = BOOK_DIR / "qa"
RENDER_DIR = QA_DIR / "rendered"

IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)")
EQUATION_OPEN_RE = re.compile(r"^:::equation\s+([A-Za-z0-9_-]+)\s*$")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")

# These are strong signals that a font/encoding was damaged during PDF
# generation.  Ordinary Chinese characters are intentionally not blacklisted.
MOJIBAKE_TOKENS = (
    "锟斤拷",
    "Ã",
    "Â",
    "â€",
    "ðŸ",
    "ï¿½",
    "璇煶",
    "鐩綍",
    "鈥",
    "銆",
    "绗",
)
TOFU_CHARS = {"�", "□", "■", "▯", "☐", "�"}


@dataclass
class Finding:
    severity: str
    kind: str
    message: str
    page: int | None = None
    detail: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceAudit:
    markdown_files: int = 0
    image_references: int = 0
    equation_references: int = 0
    unique_images: int = 0
    unique_equations: int = 0
    missing_assets: int = 0
    malformed_blocks: int = 0


@dataclass
class TextLine:
    text: str
    rect: tuple[float, float, float, float]
    size: float


def _safe_rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _normalise_title(value: str) -> str:
    """Normalise titles enough to match ReportLab-extracted text."""
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"[`*_]", "", value)
    # Punctuation and whitespace may be split into separate PDF spans.
    return "".join(ch.lower() for ch in value if not unicodedata.category(ch).startswith(("P", "Z")))


def _snippet(text: str, index: int, radius: int = 24) -> str:
    left = max(0, index - radius)
    right = min(len(text), index + radius + 1)
    return text[left:right].replace("\n", " ").strip()


def _rect_tuple(rect: fitz.Rect) -> tuple[float, float, float, float]:
    return (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))


def _rect_area(rect: fitz.Rect) -> float:
    return max(0.0, float(rect.width)) * max(0.0, float(rect.height))


def _union_area(rects: list[fitz.Rect]) -> float:
    """Return a modest exact union area for page-density estimation."""
    if not rects:
        return 0.0
    xs = sorted({x for r in rects for x in (max(0.0, r.x0), max(0.0, r.x1))})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue
        intervals = sorted(
            (max(0.0, r.y0), max(0.0, r.y1))
            for r in rects
            if r.x0 < x1 and r.x1 > x0 and r.y1 > r.y0
        )
        if not intervals:
            continue
        covered = 0.0
        sy, ey = intervals[0]
        for y0, y1 in intervals[1:]:
            if y0 <= ey:
                ey = max(ey, y1)
            else:
                covered += max(0.0, ey - sy)
                sy, ey = y0, y1
        covered += max(0.0, ey - sy)
        total += (x1 - x0) * covered
    return total


def _source_audit(findings: list[Finding]) -> tuple[SourceAudit, list[str], list[str]]:
    """Check every local image and MathType equation referenced by Markdown."""
    audit = SourceAudit()
    image_refs: list[str] = []
    equation_refs: list[str] = []
    heading_titles: list[str] = []

    if not CONTENT_DIR.exists():
        findings.append(Finding("error", "source", f"content directory is missing: {_safe_rel(CONTENT_DIR)}"))
        return audit, heading_titles, equation_refs

    markdown_files = sorted(CONTENT_DIR.glob("*.md"), key=lambda p: p.name.lower())
    audit.markdown_files = len(markdown_files)
    if not markdown_files:
        findings.append(Finding("error", "source", f"no Markdown files found under {_safe_rel(CONTENT_DIR)}"))

    for md_path in markdown_files:
        try:
            source = md_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            findings.append(Finding("error", "source_encoding", f"cannot decode {_safe_rel(md_path)} as UTF-8: {exc}"))
            continue

        for line_number, line in enumerate(source.replace("\r\n", "\n").split("\n"), 1):
            heading = HEADING_RE.match(line)
            if heading:
                heading_titles.append(heading.group(2).strip())

            image = IMAGE_RE.search(line)
            if image:
                ref = unquote(image.group(1).strip())
                image_refs.append(ref)
                # renderer.py resolves image paths from BOOK_DIR, so use the
                # same rule here instead of silently resolving from content/.
                if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", ref):
                    findings.append(Finding("error", "asset_reference", f"remote image is not supported by renderer: {ref}", detail=f"{_safe_rel(md_path)}:{line_number}"))
                    continue
                asset = (BOOK_DIR / ref).resolve()
                try:
                    asset.relative_to(BOOK_DIR.resolve())
                except ValueError:
                    findings.append(Finding("error", "asset_reference", f"image escapes voice_book: {ref}", detail=f"{_safe_rel(md_path)}:{line_number}"))
                    continue
                if not asset.exists():
                    audit.missing_assets += 1
                    findings.append(Finding("error", "missing_image", f"missing image asset: {ref}", detail=f"{_safe_rel(md_path)}:{line_number}"))
                else:
                    try:
                        with Image.open(asset) as image_obj:
                            image_obj.verify()
                    except Exception as exc:  # Pillow has several format-specific exceptions.
                        findings.append(Finding("error", "invalid_image", f"cannot read image asset: {ref}: {exc}", detail=f"{_safe_rel(md_path)}:{line_number}"))

            if line.startswith(":::"):
                if line.strip() == ":::":
                    # A closing marker is validated when its opening marker is
                    # encountered; an unmatched close is still useful to flag.
                    continue
                if line.startswith(":::equation"):
                    match = EQUATION_OPEN_RE.match(line)
                    if not match:
                        audit.malformed_blocks += 1
                        findings.append(Finding("error", "equation_syntax", f"invalid equation block header: {line.strip()}", detail=f"{_safe_rel(md_path)}:{line_number}"))
                        continue
                    equation_id = match.group(1)
                    equation_refs.append(equation_id)
                    # Locate the matching close marker.  This is intentionally
                    # a lightweight parser matching renderer.parse_markdown.
                    close = source.replace("\r\n", "\n").split("\n")[line_number:]
                    if ":::" not in [part.strip() for part in close]:
                        audit.malformed_blocks += 1
                        findings.append(Finding("error", "equation_syntax", f"equation block {equation_id!r} has no closing :::", detail=f"{_safe_rel(md_path)}:{line_number}"))
                    asset = (BOOK_DIR / "math" / "assets" / f"{equation_id}.png").resolve()
                    if not asset.exists():
                        audit.missing_assets += 1
                        findings.append(Finding("error", "missing_equation", f"missing MathType equation asset: math/assets/{equation_id}.png", detail=f"{_safe_rel(md_path)}:{line_number}"))
                    else:
                        try:
                            with Image.open(asset) as image_obj:
                                image_obj.verify()
                        except Exception as exc:
                            findings.append(Finding("error", "invalid_equation", f"cannot read MathType asset {equation_id}: {exc}", detail=f"{_safe_rel(md_path)}:{line_number}"))

    audit.image_references = len(image_refs)
    audit.equation_references = len(equation_refs)
    audit.unique_images = len(set(image_refs))
    audit.unique_equations = len(set(equation_refs))
    return audit, heading_titles, equation_refs


def _page_lines(page: fitz.Page) -> tuple[list[TextLine], list[fitz.Rect], list[fitz.Rect], str]:
    """Extract text lines, text block rectangles, image rectangles, and text."""
    data = page.get_text("dict", sort=True)
    lines: list[TextLine] = []
    text_rects: list[fitz.Rect] = []
    image_rects: list[fitz.Rect] = []
    for block in data.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        rect = fitz.Rect(bbox)
        if block.get("type") == 1:
            image_rects.append(rect)
            continue
        if block.get("type") != 0:
            continue
        text_rects.append(rect)
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(str(span.get("text", "")) for span in spans)
            if text.strip():
                size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                lines.append(TextLine(text=text, rect=tuple(float(v) for v in line.get("bbox", bbox)), size=size))
    return lines, text_rects, image_rects, page.get_text("text", sort=True)


def _find_heading(line: TextLine, heading_titles: list[str]) -> tuple[int, str] | None:
    nline = _normalise_title(line.text)
    if len(nline) < 3:
        return None
    for title in heading_titles:
        ntitle = _normalise_title(title)
        if not ntitle:
            continue
        # A wrapped heading is represented by one line in most cases; allow a
        # conservative prefix match only for large text to avoid body matches.
        if nline == ntitle or (line.size >= 11.0 and (nline.startswith(ntitle) or ntitle.startswith(nline))):
            level = 1 if title.startswith(("第", "附录", "导读")) else 2
            return level, title
    return None


def _is_toc_page(page_index: int, text: str) -> bool:
    # The current book uses one cover followed by a multi-page TOC.  The
    # first body heading starts after that front matter; do not mistake TOC
    # entries such as "13.3" for orphan section headings.
    if page_index < 6:
        return True
    if "目录" in text:
        return True
    # A second TOC page generally has many dotted leaders and little prose.
    return text.count("……") >= 3 or (text.count(".") >= 8 and len(text) < 900)


def _is_part_or_opener(text: str, heading_hits: list[tuple[int, str]]) -> bool:
    if any(level == 1 for level, _ in heading_hits):
        return True
    return bool(re.search(r"第[一二三四五六七八九十0-9]+篇|附录\s*[A-G]", text))


def _scan_pdf(
    pdf_path: Path,
    heading_titles: list[str],
    findings: list[Finding],
) -> dict:
    """Inspect structure, text, geometry, density, and orphan candidates."""
    stats: dict = {
        "page_count": 0,
        "toc_count": 0,
        "toc_entries": [],
        "text_characters": 0,
        "embedded_image_blocks": 0,
        "bad_text_pages": 0,
        "overflow_blocks": 0,
        "low_density_pages": 0,
        "orphan_candidates": 0,
        "pages": [],
    }
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        findings.append(Finding("error", "pdf_open", f"cannot open PDF {_safe_rel(pdf_path)}: {exc}"))
        return stats

    with document:
        stats["page_count"] = len(document)
        try:
            toc = document.get_toc(simple=True)
            stats["toc_count"] = len(toc)
            stats["toc_entries"] = [
                {"level": int(row[0]), "title": str(row[1]), "page": int(row[2])}
                for row in toc
                if len(row) >= 3
            ]
            for row in stats["toc_entries"]:
                if row["page"] < 1 or row["page"] > len(document):
                    findings.append(Finding("error", "toc", f"bookmark points outside document: {row}"))
        except Exception as exc:
            findings.append(Finding("error", "toc", f"cannot read PDF bookmarks/TOC: {exc}"))

        if not stats["page_count"]:
            findings.append(Finding("error", "pdf_empty", "PDF contains no pages"))
            return stats
        if not stats["toc_count"]:
            findings.append(Finding("warning", "toc", "PDF contains no outline/bookmark entries"))

        for page_index in range(len(document)):
            page_number = page_index + 1
            page = document[page_index]
            page_result = {
                "page": page_number,
                "width_pt": round(float(page.rect.width), 2),
                "height_pt": round(float(page.rect.height), 2),
                "characters": 0,
                "text_blocks": 0,
                "image_blocks": 0,
                "occupied_ratio": 0.0,
                "headings": [],
                "low_density": False,
            }
            try:
                lines, text_rects, image_rects, text = _page_lines(page)
            except Exception as exc:
                findings.append(Finding("error", "page_read", f"cannot inspect page {page_number}: {exc}", page=page_number))
                stats["pages"].append(page_result)
                continue

            page_result["characters"] = len(text)
            page_result["text_blocks"] = len(text_rects)
            page_result["image_blocks"] = len(image_rects)
            page_result["characters"] = len(text)
            stats["text_characters"] += len(text)
            stats["embedded_image_blocks"] += len(image_rects)

            # Scan for replacement glyphs, tofu squares, controls, and common
            # UTF-8/GBK mojibake.  Report each page once per class.
            page_bad = set()
            for bad in ("replacement", "tofu", "control", "mojibake"):
                if bad == "replacement" and ("�" in text or "￼" in text):
                    page_bad.add(bad)
                elif bad == "tofu" and any(ch in text for ch in TOFU_CHARS):
                    page_bad.add(bad)
                elif bad == "control" and any(unicodedata.category(ch) == "Cc" and ch not in "\t\r\n\f" for ch in text):
                    page_bad.add(bad)
                elif bad == "mojibake" and any(token in text for token in MOJIBAKE_TOKENS):
                    page_bad.add(bad)
            if "replacement" in page_bad or "mojibake" in page_bad:
                stats["bad_text_pages"] += 1
                samples = []
                for token in ("�", "￼", *MOJIBAKE_TOKENS):
                    pos = text.find(token)
                    if pos >= 0:
                        samples.append(_snippet(text, pos))
                findings.append(Finding("error", "bad_text", f"replacement glyph or mojibake detected on page {page_number}", page=page_number, detail=" | ".join(dict.fromkeys(samples))[:300]))
            if "tofu" in page_bad and "replacement" not in page_bad:
                findings.append(Finding("warning", "tofu", f"possible missing-glyph square on page {page_number}", page=page_number, detail=_snippet(text, next((text.find(ch) for ch in TOFU_CHARS if text.find(ch) >= 0), 0))))
            if "control" in page_bad:
                findings.append(Finding("warning", "control_char", f"unexpected control character in extracted text on page {page_number}", page=page_number))

            # Physical-page overflow checks for all text and image blocks.
            page_rect = page.rect
            all_rects = [("text", r) for r in text_rects] + [("image", r) for r in image_rects]
            for kind, rect in all_rects:
                if rect.width <= 0 or rect.height <= 0:
                    findings.append(Finding("error", "geometry", f"non-positive {kind} block on page {page_number}: {_rect_tuple(rect)}", page=page_number))
                    continue
                tolerance = 1.5
                if (rect.x0 < page_rect.x0 - tolerance or rect.y0 < page_rect.y0 - tolerance or
                        rect.x1 > page_rect.x1 + tolerance or rect.y1 > page_rect.y1 + tolerance):
                    stats["overflow_blocks"] += 1
                    findings.append(Finding("error", "overflow", f"{kind} block crosses physical page boundary on page {page_number}: {_rect_tuple(rect)}", page=page_number))

            # Heading matches support both chapter-opener exceptions and the
            # orphan-heading heuristic below.
            heading_hits: list[tuple[int, str, TextLine]] = []
            for line in lines:
                match = _find_heading(line, heading_titles)
                if match:
                    heading_hits.append((match[0], match[1], line))
                    page_result["headings"].append({"level": match[0], "title": match[1], "y1": round(line.rect[3], 1)})

            for level, title, line in heading_hits:
                # Ignore the normal cover/TOC/opening pages and only flag a
                # heading when virtually no substantive item follows it.
                if page_index == 0 or _is_toc_page(page_index, text) or level == 1:
                    continue
                below_lines = [other for other in lines if other.rect[1] >= line.rect[3] - 1.0]
                below_chars = sum(len(other.text.strip()) for other in below_lines if other is not line)
                below_images = sum(_rect_area(r) for r in image_rects if r.y0 >= line.rect[3] - 1.0)
                if line.rect[3] > page.rect.height * 0.78 and below_chars < 45 and below_images < page.rect.width * page.rect.height * 0.035:
                    stats["orphan_candidates"] += 1
                    findings.append(Finding("warning", "orphan_heading", f"小节标题疑似孤立在页底: {title}", page=page_number, detail=f"y1={line.rect[3]:.1f}pt, following_chars={below_chars}"))

            # Short table/caption labels ending in a colon are the most common
            # table-title form in this book.  They should have a table/list or
            # prose item immediately below them on the same page.
            for line in lines:
                stripped = line.text.strip()
                if not stripped or len(stripped) > 70 or not stripped.endswith((":", "：")):
                    continue
                if stripped.startswith(("http://", "https://")):
                    continue
                if line.rect[3] <= page.rect.height * 0.78 or _is_toc_page(page_index, text):
                    continue
                following = [other for other in lines if other.rect[1] >= line.rect[3] - 1.0 and other is not line]
                if not following:
                    stats["orphan_candidates"] += 1
                    findings.append(Finding("warning", "orphan_caption", f"表题/提示语疑似孤立在页底: {stripped}", page=page_number, detail=f"y1={line.rect[3]:.1f}pt"))

            # Density is a warning only.  Use the union of extracted text and
            # image rectangles so a figure-heavy page is not mistaken for a
            # blank page merely because it has few characters.
            occupied = _union_area(text_rects + image_rects)
            page_area = max(1.0, float(page.rect.width) * float(page.rect.height))
            occupied_ratio = occupied / page_area
            page_result["occupied_ratio"] = round(occupied_ratio, 4)
            exempt = page_index == 0 or page_index == len(document) - 1 or _is_toc_page(page_index, text) or _is_part_or_opener(text, [(h[0], h[1]) for h in heading_hits])
            if not exempt and occupied_ratio < 0.085 and len(text.strip()) < 300:
                page_result["low_density"] = True
                stats["low_density_pages"] += 1
                findings.append(Finding("warning", "low_density", f"疑似大面积空白页: occupied={occupied_ratio:.1%}, chars={len(text.strip())}", page=page_number))
            stats["pages"].append(page_result)
    return stats


def _render_pages(pdf_path: Path, render_dir: Path, dpi: int, findings: list[Finding]) -> dict:
    render_dir.mkdir(parents=True, exist_ok=True)
    # Only remove files produced by this script; leave any user notes or other
    # files in the QA directory untouched.
    for old in render_dir.glob("page-*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    paths: list[Path] = []
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        findings.append(Finding("error", "render", f"cannot render PDF: {exc}"))
        return {"rendered_pages": 0, "render_dir": _safe_rel(render_dir)}
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    with document:
        for index, page in enumerate(document):
            target = render_dir / f"page-{index + 1:04d}.png"
            try:
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(str(target))
                paths.append(target)
            except Exception as exc:
                findings.append(Finding("error", "render", f"failed to render page {index + 1}: {exc}", page=index + 1))
    return {"rendered_pages": len(paths), "render_dir": _safe_rel(render_dir), "paths": paths}


def _make_contact_sheet(paths: list[Path], output: Path, findings: list[Finding]) -> dict:
    if not paths:
        return {"contact_sheet": None}
    try:
        thumb_width = 190
        tile_width = thumb_width + 18
        tile_height = 300
        columns = 5
        rows = math.ceil(len(paths) / columns)
        sheet = Image.new("RGB", (columns * tile_width + 18, rows * tile_height + 18), "#edf1f5")
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for index, path in enumerate(paths):
            with Image.open(path) as original:
                thumb = original.convert("RGB")
                ratio = thumb_width / max(1, thumb.width)
                thumb = thumb.resize((thumb_width, max(1, round(thumb.height * ratio))), Image.Resampling.LANCZOS)
            col = index % columns
            row = index // columns
            x = 9 + col * tile_width
            y = 9 + row * tile_height
            sheet.paste(thumb, (x, y))
            draw.rectangle((x - 1, y - 1, x + thumb.width, y + thumb.height), outline="#9ba8b6", width=1)
            draw.text((x, y + thumb.height + 5), f"p.{index + 1}", fill="#243342", font=font)
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, format="PNG", optimize=True)
        return {"contact_sheet": _safe_rel(output), "width": sheet.width, "height": sheet.height, "columns": columns, "rows": rows}
    except Exception as exc:
        findings.append(Finding("error", "contact_sheet", f"failed to create contact sheet: {exc}"))
        return {"contact_sheet": None}


def _write_reports(report: dict, findings: list[Finding], report_json: Path, report_txt: Path) -> None:
    report["findings"] = [finding.as_dict() for finding in findings]
    report["errors"] = sum(1 for finding in findings if finding.severity == "error")
    report["warnings"] = sum(1 for finding in findings if finding.severity == "warning")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "语音算法工程师教学 PDF 质检报告",
        f"生成时间（UTC）: {report.get('generated_at', '')}",
        f"PDF: {report.get('pdf', '')}",
        f"页数: {report.get('pdf_stats', {}).get('page_count', 0)}",
        f"书签/目录条目数: {report.get('pdf_stats', {}).get('toc_count', 0)}",
        f"正文字符数: {report.get('pdf_stats', {}).get('text_characters', 0)}",
        f"嵌入图块数: {report.get('pdf_stats', {}).get('embedded_image_blocks', 0)}",
        f"坏字符页数: {report.get('pdf_stats', {}).get('bad_text_pages', 0)}",
        f"越界图/文字块数: {report.get('pdf_stats', {}).get('overflow_blocks', 0)}",
        f"低密度页数（警告）: {report.get('pdf_stats', {}).get('low_density_pages', 0)}",
        f"标题/表题孤立候选数（警告）: {report.get('pdf_stats', {}).get('orphan_candidates', 0)}",
        f"错误: {report['errors']}  警告: {report['warnings']}",
        "",
        "源文件资源审计:",
        json.dumps(report.get("source_audit", {}), ensure_ascii=False),
        "",
        "发现:",
    ]
    if findings:
        for finding in findings:
            page = f" [p.{finding.page}]" if finding.page else ""
            detail = f" ({finding.detail})" if finding.detail else ""
            lines.append(f"[{finding.severity.upper()}] {finding.kind}{page}: {finding.message}{detail}")
    else:
        lines.append("无发现。")
    contact = report.get("render", {}).get("contact_sheet")
    if contact:
        lines.extend(["", f"联系页: {contact}"])
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(pdf_path: Path, *, render: bool = True, dpi: int = 120) -> tuple[int, dict]:
    findings: list[Finding] = []
    source_audit, heading_titles, _ = _source_audit(findings)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pdf": _safe_rel(pdf_path),
        "source_audit": asdict(source_audit),
        "pdf_stats": {},
        "render": {},
    }
    if not pdf_path.exists():
        findings.append(Finding("error", "pdf_missing", f"PDF does not exist: {_safe_rel(pdf_path)}"))
    else:
        report["pdf_stats"] = _scan_pdf(pdf_path, heading_titles, findings)
        if report["pdf_stats"].get("page_count", 0) and source_audit.equation_references and report["pdf_stats"].get("embedded_image_blocks", 0) == 0:
            findings.append(Finding("error", "embedded_assets", "Markdown references equations/images but PDF contains no image blocks"))
        if render:
            rendered = _render_pages(pdf_path, RENDER_DIR, dpi, findings)
            paths = rendered.pop("paths", [])
            rendered.update(_make_contact_sheet(paths, QA_DIR / "contact_sheet.png", findings))
            report["render"] = rendered
    _write_reports(report, findings, QA_DIR / "report.json", QA_DIR / "report.txt")
    # Warnings are intentionally non-fatal.  This lets a reviewer inspect a
    # sparse chapter opener while still failing CI for correctness defects.
    return (1 if any(f.severity == "error" for f in findings) else 0), report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect and render the voice-engineer teaching PDF.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF path (default: 语音算法工程师教学.pdf next to voice_book)")
    parser.add_argument("--dpi", type=int, default=120, help="render DPI for page PNGs (default: 120)")
    parser.add_argument("--no-render", action="store_true", help="skip page PNG/contact-sheet generation")
    args = parser.parse_args(argv)
    pdf_path = args.pdf.expanduser()
    if not pdf_path.is_absolute():
        pdf_path = (Path.cwd() / pdf_path).resolve()
    else:
        pdf_path = pdf_path.resolve()
    if args.dpi < 36 or args.dpi > 300:
        parser.error("--dpi must be between 36 and 300")
    code, report = run(pdf_path, render=not args.no_render, dpi=args.dpi)
    print(f"QA complete: pages={report.get('pdf_stats', {}).get('page_count', 0)}, "
          f"toc={report.get('pdf_stats', {}).get('toc_count', 0)}, "
          f"errors={sum(1 for f in report.get('findings', []) if f.get('severity') == 'error')}, "
          f"warnings={sum(1 for f in report.get('findings', []) if f.get('severity') == 'warning')}")
    print(f"Report: {QA_DIR / 'report.txt'}")
    if not args.no_render:
        print(f"Rendered pages/contact sheet: {RENDER_DIR} / {QA_DIR / 'contact_sheet.png'}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
