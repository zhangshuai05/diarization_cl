"""Small, deterministic Markdown-to-ReportLab renderer for the new textbook."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
import re
import unicodedata

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, Image, NextPageTemplate, PageBreak,
    CondPageBreak, KeepTogether, PageTemplate, Paragraph, Spacer, Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics

from .style import (
    BODY_LEAD, CALLOUT_KINDS, FONT_BODY, FONT_BOLD, FONT_MONO, FRAME_W,
    MARGIN_B, MARGIN_L, MARGIN_R, MARGIN_T, PAGE_SIZE, PALETTE, STYLES,
)


BOOK_DIR = Path(__file__).resolve().parents[1]


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in text)


def _wrap_code_line(line: str, limit: int = 86) -> list[str]:
    if _display_width(line) <= limit:
        return [line]
    indent = re.match(r"\s*", line).group(0)
    remain = line
    out = []
    while _display_width(remain) > limit:
        width = 0
        cut = 0
        for i, ch in enumerate(remain):
            width += 2 if unicodedata.east_asian_width(ch) in "WFA" else 1
            if width > limit:
                break
            cut = i + 1
        search = remain[:cut]
        soft = max(search.rfind(","), search.rfind(" "), search.rfind("/"))
        if soft > max(10, len(indent) + 4):
            cut = soft + 1
        out.append(remain[:cut].rstrip())
        remain = indent + "  " + remain[cut:].lstrip()
    out.append(remain)
    return out


def inline_markup(text: str) -> str:
    """Escape text and translate a small safe inline Markdown subset."""
    out = []
    i = 0
    plain = []

    def flush():
        if plain:
            out.append(escape("".join(plain)))
            plain.clear()

    while i < len(text):
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j >= 0:
                flush(); out.append("<b>" + escape(text[i + 2:j]) + "</b>"); i = j + 2; continue
        if text[i] == "`":
            j = text.find("`", i + 1)
            if j >= 0:
                flush()
                # Keep identifiers such as SPEAKER_01 intact.  The helper
                # also supplies a CJK fallback for Chinese inside backticks.
                out.append("<nobr>" + _mixed_code_markup(text[i + 1:j]) + "</nobr>")
                i = j + 1; continue
        if text[i] == "[":
            m = re.match(r"\[([^\]]+)\]\(([^)]+)\)", text[i:])
            if m:
                flush(); label, url = m.group(1), m.group(2)
                if url.startswith(("http://", "https://")):
                    out.append(f'<link href="{escape(url, quote=True)}" color="#1F5C8B">{escape(label)}</link>')
                else:
                    out.append(escape(label))
                i += m.end(); continue
        plain.append(text[i]); i += 1
    flush()
    return "".join(out)


def _paragraph(text: str, style, bullet_text: str | None = None) -> Paragraph:
    return Paragraph(inline_markup(text.strip()), style, bulletText=bullet_text)


def _table_widths(rows: list[list[str]]) -> list[float]:
    cols = len(rows[0])
    maxima = []
    for c in range(cols):
        maxima.append(max(5, min(52, max(_display_width(re.sub(r"[`*\[\]()]", "", r[c])) for r in rows))))
    weights = [max(2.5, n ** 0.62) for n in maxima]
    special = next((i for i in range(cols) if any("SPEAKER_" in row[i] for row in rows)), None)
    if special is not None:
        # Keep two short speaker identifiers on one line (or break between
        # them), never in the middle of `_01`.
        minima = [10 * mm] * cols
        minima[special] = 48 * mm
        base = sum(minima)
    else:
        minimum = 15 * mm if cols <= 6 else 11 * mm
        minima = [minimum] * cols
        base = minimum * cols
    if base >= FRAME_W:
        return [FRAME_W / cols] * cols
    remaining = FRAME_W - base
    total = sum(weights)
    return [minimum + remaining * w / total for minimum, w in zip(minima, weights)]


def _make_table(rows: list[list[str]]) -> Table:
    cols = max(len(r) for r in rows)
    rows = [r + [""] * (cols - len(r)) for r in rows]
    data = []
    for ridx, row in enumerate(rows):
        style = STYLES["cell_head"] if ridx == 0 else STYLES["cell"]
        cell_style = ParagraphStyle(
            f"table_cell_{ridx}",
            parent=style,
            splitLongWords=0,
            # TableStyle TEXTCOLOR does not recolor text inside Paragraphs;
            # apply the header contrast to the Paragraph itself.
            textColor=colors.white if ridx == 0 else style.textColor,
        )
        data.append([Paragraph(inline_markup(cell.strip()), cell_style) for cell in row])
    table = Table(data, colWidths=_table_widths(rows), repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), PALETTE["accent"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), .35, PALETTE["rule"]),
    ]
    for r in range(2, len(rows), 2):
        commands.append(("BACKGROUND", (0, r), (-1, r), PALETTE["zebra"]))
    table.setStyle(TableStyle(commands))
    table.spaceBefore = 5
    table.spaceAfter = 9
    return table


def _code_table(lines: list[str]) -> Table:
    wrapped = []
    for line in lines:
        wrapped.extend(_wrap_code_line(line.rstrip("\n")))
    rows = []
    for line in wrapped or [""]:
        txt = _mixed_code_markup(line)
        rows.append([Paragraph(txt, STYLES["code"])])
    table = Table(rows, colWidths=[FRAME_W], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["code_bg"]),
        ("LINEBEFORE", (0, 0), (0, -1), 2.2, PALETTE["accent"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -2), 0),
    ]))
    table.spaceBefore = 5
    table.spaceAfter = 9
    return table


def _mixed_code_markup(line: str) -> str:
    """Keep ASCII code monospace while giving CJK/Unicode a real glyph font."""
    if not line:
        return "&#160;"
    runs: list[tuple[bool, str]] = []
    current_body = None
    current = []
    for ch in line:
        # Consolas has good ASCII coverage but no Chinese glyphs.  Treat full
        # width, non-ASCII and symbol characters as body-font runs.
        body = ord(ch) > 127 or unicodedata.east_asian_width(ch) in "WFA"
        if current_body is None:
            current_body = body
        if body != current_body:
            runs.append((current_body, "".join(current)))
            current = []
            current_body = body
        current.append(ch)
    if current:
        runs.append((current_body, "".join(current)))
    parts = []
    for body, value in runs:
        value = escape(value).replace(" ", "&#160;")
        if body:
            parts.append('<font face="Body">' + value + "</font>")
        else:
            parts.append(value)
    return "".join(parts) or "&#160;"


def _callout(kind: str, title: str, lines: list[str]) -> Table:
    border_key, bg_key, default_title = CALLOUT_KINDS.get(kind, CALLOUT_KINDS["tip"])
    flows = [Paragraph(inline_markup(title or default_title), STYLES["callout_title"])]
    chunks = []
    for line in lines + [""]:
        if line.strip():
            chunks.append(line.strip())
        elif chunks:
            text = " ".join(chunks)
            if re.match(r"^[-*]\s+", text):
                text = re.sub(r"^[-*]\s+", "", text)
                flows.append(_paragraph(text, STYLES["callout_body"], "•"))
            else:
                flows.append(Paragraph(inline_markup(text), STYLES["callout_body"]))
            chunks = []
    t = Table([[flows]], colWidths=[FRAME_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE[bg_key]),
        ("LINEBEFORE", (0, 0), (0, -1), 3, PALETTE[border_key]),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    t.spaceBefore = 6
    t.spaceAfter = 10
    return t


def _quote(lines: list[str]) -> Table:
    flows = [Paragraph(inline_markup(line), STYLES["callout_body"]) for line in lines if line.strip()]
    t = Table([[flows]], colWidths=[FRAME_W], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALETTE["code_bg"]),
        ("LINEBEFORE", (0, 0), (0, -1), 2, PALETTE["ink_faint"]),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    t.spaceAfter = 8
    return t


def _image_flow(alt: str, rel: str) -> list:
    path = BOOK_DIR / rel
    if not path.exists():
        raise FileNotFoundError(path)
    iw, ih = ImageReader(str(path)).getSize()
    max_h = 132 * mm
    max_w = 145 * mm if rel.endswith("cover_system.png") else FRAME_W
    scale = min(max_w / iw, max_h / ih)
    image = Image(str(path), width=iw * scale, height=ih * scale)
    image.hAlign = "CENTER"
    # A figure and its caption are one semantic unit.  Keeping them together
    # prevents a caption from becoming an orphan at the top of the next page.
    return [KeepTogether([Spacer(1, 5), image, Paragraph(inline_markup(alt), STYLES["caption"])])]


def _equation_flow(equation_id: str, source_lines: list[str]) -> list:
    """Embed a generated MathType asset without adding a competing text formula."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", equation_id):
        raise ValueError(f"invalid equation id: {equation_id!r}")
    path = BOOK_DIR / "math" / "assets" / f"{equation_id}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"missing MathType asset {path}; run "
            "python book/math/generate_mathtype.py"
        )
    # The PNG is intentionally supersampled for print/zoom quality.  Use the
    # original MathType GIF dimensions for physical layout so the equation's
    # size on the page remains stable across asset resolutions.
    dimension_path = path.with_suffix(".gif")
    if not dimension_path.exists():
        dimension_path = path
    iw, ih = ImageReader(str(dimension_path)).getSize()
    max_h = 29 * mm
    scale = min(1.0, FRAME_W / iw, max_h / ih)
    image = Image(str(path), width=iw * scale, height=ih * scale)
    image.hAlign = "CENTER"
    return [KeepTogether([Spacer(1, 4), image, Spacer(1, 3)])]


@dataclass
class RenderState:
    key_counter: int = 0
    after_h1: bool = False

    def key(self, level: int) -> str:
        self.key_counter += 1
        return f"h{level}_{self.key_counter}"


def parse_markdown(text: str, state: RenderState | None = None) -> list:
    state = state or RenderState()
    lines = text.replace("\r\n", "\n").split("\n")
    flow = []
    i = 0
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            txt = " ".join(line.strip() for line in paragraph_lines)
            style = STYLES["lead"] if state.after_h1 else STYLES["body"]
            flow.append(Paragraph(inline_markup(txt), style))
            state.after_h1 = False
            paragraph_lines = []

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush_paragraph(); i += 1; continue
        if line.startswith(":::equation"):
            flush_paragraph()
            match = re.match(r"^:::equation\s+([A-Za-z0-9_-]+)\s*$", line)
            if not match:
                raise ValueError(f"invalid equation block header: {line!r}")
            equation_id = match.group(1)
            body = []
            i += 1
            while i < len(lines) and lines[i].strip() != ":::":
                body.append(lines[i]); i += 1
            if i >= len(lines):
                raise ValueError(f"equation block {equation_id!r} has no closing :::")
            i += 1
            flow.extend(_equation_flow(equation_id, body))
            continue
        if line.startswith("```"):
            flush_paragraph(); code = []; i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            flow.append(_code_table(code)); continue
        if line.startswith(":::") and line.strip() != ":::":
            flush_paragraph(); m = re.match(r":::(\w+)\s*(.*)", line); kind, title = m.group(1), m.group(2); body=[]; i += 1
            while i < len(lines) and lines[i].strip() != ":::": body.append(lines[i]); i += 1
            i += 1; flow.append(_callout(kind, title, body)); continue
        if re.match(r"^\|.*\|\s*$", line) and i + 1 < len(lines) and re.match(r"^\|?\s*:?-+", lines[i + 1]):
            flush_paragraph(); rows=[]
            while i < len(lines) and re.match(r"^\|.*\|\s*$", lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells); i += 1
            if len(rows) >= 2 and all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in rows[1]): rows.pop(1)
            flow.append(_make_table(rows)); continue
        m = re.match(r"^(#{1,4})\s+(.+)$", line)
        if m:
            flush_paragraph(); level=len(m.group(1)); title=m.group(2).strip(); key=state.key(level)
            if level == 2 and title in {"本章小结", "思考题"}:
                # These are chapter-ending units.  Keep enough room for the
                # heading and more than one following line so a prompt does
                # not become an orphan at the foot of a page.
                flow.append(CondPageBreak(32 * mm))
            if level == 1:
                n = re.search(r"(?:第\s*)?(\d+|[A-G])", title)
                num = n.group(1) if n else ""
                if num.isdigit(): num = num.zfill(2)
                if num:
                    flow.append(Paragraph(num, STYLES["h1_num"]))
                p = Paragraph(inline_markup(title), STYLES["h1"]); state.after_h1=True
            else:
                p = Paragraph(inline_markup(title), STYLES[f"h{level}"])
            if level <= 2:
                p._toc_level = level - 1
                p._bookmark_key = key
                p._bookmark_text = re.sub(r"[`*_]", "", title)
            flow.append(p); i += 1; continue
        img = re.match(r"^!\[([^\]]+)\]\(([^)]+)\)\s*$", line)
        if img:
            flush_paragraph(); flow.extend(_image_flow(img.group(1), img.group(2))); i += 1; continue
        if line.startswith("> "):
            flush_paragraph(); q=[]
            while i < len(lines) and lines[i].startswith("> "): q.append(lines[i][2:]); i += 1
            flow.append(_quote(q)); continue
        li = re.match(r"^(\s*)([-*]|\d+\.)\s+(.+)$", line)
        if li:
            flush_paragraph(); indent = len(li.group(1).replace("\t", "    ")); marker=li.group(2); body=li.group(3)
            style = ParagraphStyle(f"list_{indent}_{marker}", parent=STYLES["bullet"], leftIndent=12 + indent*5, firstLineIndent=-10)
            bullet = "•" if marker in ("-", "*") else marker
            flow.append(Paragraph(inline_markup(body), style, bulletText=bullet)); i += 1; continue
        paragraph_lines.append(line); i += 1
    flush_paragraph()
    return flow


def _draw_main_page(canvas, doc):
    canvas.saveState()
    width, height = PAGE_SIZE
    canvas.setStrokeColor(PALETTE["rule"]); canvas.setLineWidth(.45)
    canvas.line(MARGIN_L, height - 14.7*mm, width - MARGIN_R, height - 14.7*mm)
    canvas.setFont(FONT_BODY, 7.8); canvas.setFillColor(PALETTE["ink_faint"])
    chapter = getattr(doc, "current_chapter", "语音算法工程师教学")
    if len(chapter) > 30: chapter = chapter[:29] + "…"
    if doc.page % 2:
        canvas.drawRightString(width - MARGIN_R, height - 11.7*mm, chapter)
        canvas.drawRightString(width - MARGIN_R, 10.5*mm, str(doc.page))
    else:
        canvas.drawString(MARGIN_L, height - 11.7*mm, chapter)
        canvas.drawString(MARGIN_L, 10.5*mm, str(doc.page))
    canvas.restoreState()


def _draw_cover_page(canvas, doc):
    """Reference-book-like cover bars, kept outside the story flow."""
    canvas.saveState()
    width, height = PAGE_SIZE
    canvas.setFillColor(PALETTE["accent"])
    canvas.rect(0, height - 13 * mm, width, 13 * mm, stroke=0, fill=1)
    canvas.rect(0, 0, width, 13 * mm, stroke=0, fill=1)
    canvas.restoreState()


def _draw_front_page(canvas, doc):
    canvas.saveState()
    width, height = PAGE_SIZE
    canvas.setStrokeColor(PALETTE["accent"])
    canvas.setLineWidth(.45)
    canvas.line(MARGIN_L, height - 18 * mm, width - MARGIN_R, height - 18 * mm)
    canvas.setFont(FONT_BODY, 7.8)
    canvas.setFillColor(PALETTE["ink_faint"])
    canvas.drawRightString(width - MARGIN_R, 10.5 * mm, str(doc.page))
    canvas.restoreState()


class BookDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=PAGE_SIZE, leftMargin=MARGIN_L, rightMargin=MARGIN_R, topMargin=MARGIN_T, bottomMargin=MARGIN_B, **kw)
        frame = Frame(MARGIN_L, MARGIN_B, FRAME_W, PAGE_SIZE[1]-MARGIN_T-MARGIN_B, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="body")
        self.addPageTemplates([
            PageTemplate(id="cover", frames=[frame], onPage=_draw_cover_page),
            PageTemplate(id="front", frames=[frame], onPage=_draw_front_page),
            PageTemplate(id="main", frames=[frame], onPageEnd=_draw_main_page),
        ])
        self.current_chapter = ""

    def afterFlowable(self, flowable):
        if not hasattr(flowable, "_toc_level"):
            return
        level = flowable._toc_level
        text = flowable._bookmark_text
        key = flowable._bookmark_key
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))
        if level == 0:
            self.current_chapter = text


def build_story(content_files: list[Path], part_starts: dict[str, str]) -> list:
    story = []
    cover_path = BOOK_DIR / "figures" / "cover_system.png"
    if not cover_path.exists():
        raise FileNotFoundError(f"missing cover figure {cover_path}")
    cover_iw, cover_ih = ImageReader(str(cover_path)).getSize()
    cover_scale = min((145 * mm) / cover_iw, (105 * mm) / cover_ih)
    cover_image = Image(str(cover_path), width=cover_iw * cover_scale,
                        height=cover_ih * cover_scale)
    cover_image.hAlign = "CENTER"
    story.extend([
        Spacer(1, 29 * mm),
        Paragraph("语音算法工程师教学", STYLES["cover_title"]),
        Paragraph("从声学前端到端侧实时系统", STYLES["cover_sub"]),
        HRFlowable(width="54%", thickness=1.5, color=PALETTE["accent"],
                   hAlign="CENTER", spaceAfter=13),
        cover_image,
        Spacer(1, 23 * mm),
        Paragraph(
            f"面向 ASR、3A、阵列、说话人与端云部署<br/>资料核验日期：{date.today().isoformat()}",
            STYLES["cover_meta"],
        ),
        NextPageTemplate("front"),
        PageBreak(),
        Paragraph("目录", STYLES["cover_title"]),
        Spacer(1, 6*mm),
    ])
    toc = TableOfContents()
    toc.levelStyles = [STYLES["toc1"], STYLES["toc2"]]
    toc.dotsMinLevel = 0
    story.append(toc)
    story.extend([NextPageTemplate("main"), PageBreak()])
    state = RenderState()
    for idx, path in enumerate(content_files):
        if idx:
            # A hard break after every chapter produced pages containing only
            # a short summary or one final question.  Start the next chapter
            # on the same page when there is enough room for a real opener;
            # otherwise CondPageBreak moves it cleanly to the next page.
            required = 118 * mm if path.name in part_starts else 86 * mm
            story.append(CondPageBreak(required))
        if path.name in part_starts:
            story.extend([Paragraph(part_starts[path.name], STYLES["part"]), HRFlowable(width="22%", thickness=1.6, color=PALETTE["accent"], hAlign="LEFT", spaceAfter=8)])
        story.extend(parse_markdown(path.read_text(encoding="utf-8"), state))
    return story
