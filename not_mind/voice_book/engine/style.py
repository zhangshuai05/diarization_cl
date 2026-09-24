"""Typography and colour system for the speech-engineer textbook."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PAGE_SIZE = A4
MARGIN_L = 22 * mm
MARGIN_R = 21 * mm
MARGIN_T = 19 * mm
MARGIN_B = 18 * mm
FRAME_W = PAGE_SIZE[0] - MARGIN_L - MARGIN_R

FONT_BODY = "Body"
FONT_BOLD = "BodyBold"
FONT_LIGHT = "BodyLight"
FONT_MONO = "Mono"
FONT_MONO_BOLD = "MonoBold"

_WINF = r"C:\Windows\Fonts"


def register_fonts():
    """注册中英文字体。微软雅黑覆盖中日韩字形，Consolas 用于代码。"""
    pdfmetrics.registerFont(TTFont(FONT_BODY, _WINF + r"\msyh.ttc"))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, _WINF + r"\msyhbd.ttc"))
    pdfmetrics.registerFont(TTFont(FONT_LIGHT, _WINF + r"\msyhl.ttc"))
    pdfmetrics.registerFont(TTFont(FONT_MONO, _WINF + r"\consola.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_MONO_BOLD, _WINF + r"\consolab.ttf"))
    pdfmetrics.registerFontFamily(
        FONT_BODY, normal=FONT_BODY, bold=FONT_BOLD, italic=FONT_BODY,
        boldItalic=FONT_BOLD)
    pdfmetrics.registerFontFamily(
        FONT_MONO, normal=FONT_MONO, bold=FONT_MONO_BOLD,
        italic=FONT_MONO, boldItalic=FONT_MONO_BOLD)


PALETTE = {
    "ink": colors.HexColor("#1A1A1A"),
    "ink_soft": colors.HexColor("#4A4A4A"),
    "ink_faint": colors.HexColor("#8A8A8A"),
    "rule": colors.HexColor("#D8D8D8"),
    "accent": colors.HexColor("#234776"),
    "accent_soft": colors.HexColor("#EAF0F7"),
    "code_bg": colors.HexColor("#F6F7F8"),
    "code_rule": colors.HexColor("#E2E4E6"),
    "tip": colors.HexColor("#1F7A5A"),
    "tip_bg": colors.HexColor("#EAF5F0"),
    "warn": colors.HexColor("#A8531B"),
    "warn_bg": colors.HexColor("#FBF0E7"),
    "key": colors.HexColor("#6B3FA0"),
    "key_bg": colors.HexColor("#F2EDF9"),
    "th_bg": colors.HexColor("#EFF2F4"),
    "zebra": colors.HexColor("#FAFBFC"),
}

BODY_SIZE = 10.8
BODY_LEAD = 18.2


def _p(name, **kw):
    base = dict(fontName=FONT_BODY, fontSize=BODY_SIZE, leading=BODY_LEAD,
                textColor=PALETTE["ink"], wordWrap="CJK", alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "body": _p("body", spaceAfter=6.2, firstLineIndent=21.6),
    "body_noindent": _p("body_noindent", spaceAfter=6.2, firstLineIndent=0),
    "body_tight": _p("body_tight", fontSize=10.5, leading=17.4, spaceAfter=3.5),
    "lead": _p("lead", fontSize=11.5, leading=19.5,
               textColor=PALETTE["ink_soft"], spaceAfter=11.5,
               firstLineIndent=0),

    "h1_num": _p("h1_num", fontName=FONT_LIGHT, fontSize=39, leading=43,
                  textColor=PALETTE["rule"], alignment=TA_RIGHT, spaceAfter=2),
    "h1": _p("h1", fontName=FONT_BOLD, fontSize=22, leading=30,
             textColor=PALETTE["ink"], alignment=TA_RIGHT, spaceAfter=4),
    "h1_sub": _p("h1_sub", fontSize=10.5, leading=16,
                 textColor=PALETTE["ink_faint"], alignment=TA_RIGHT,
                 spaceAfter=16),

    "h2": _p("h2", fontName=FONT_BOLD, fontSize=14.2, leading=21,
              textColor=PALETTE["accent"], alignment=TA_LEFT,
              spaceBefore=13, spaceAfter=5, keepWithNext=1),
    "h3": _p("h3", fontName=FONT_BOLD, fontSize=11.6, leading=18,
             textColor=PALETTE["ink"], alignment=TA_LEFT,
             spaceBefore=11, spaceAfter=4, keepWithNext=1),
    "h4": _p("h4", fontName=FONT_BOLD, fontSize=10.4, leading=17,
             textColor=PALETTE["ink_soft"], alignment=TA_LEFT,
             spaceBefore=8, spaceAfter=3, keepWithNext=1),

    "bullet": _p("bullet", fontSize=10.5, spaceAfter=4.5, leading=18.2),
    "code": _p("code", fontName=FONT_MONO, fontSize=8.8, leading=14,
               textColor=PALETTE["ink"], alignment=TA_LEFT, wordWrap=None),
    "cell": _p("cell", fontSize=9.4, leading=15, spaceAfter=0,
               alignment=TA_LEFT),
    "cell_head": _p("cell_head", fontName=FONT_BOLD, fontSize=9.4, leading=15,
                    spaceAfter=0, alignment=TA_LEFT),
    "cell_num": _p("cell_num", fontSize=9.4, leading=15, spaceAfter=0,
                   alignment=TA_RIGHT),
    "caption": _p("caption", fontSize=9.2, leading=15,
                  textColor=PALETTE["ink_faint"], alignment=TA_CENTER,
                  spaceBefore=4, spaceAfter=10),
    "formula_source": _p(
        "formula_source", fontName=FONT_MONO, fontSize=8.5, leading=12.5,
        textColor=PALETTE["ink_faint"], alignment=TA_CENTER,
        spaceBefore=3, spaceAfter=9, wordWrap=None,
    ),
    "callout_title": _p("callout_title", fontName=FONT_BOLD, fontSize=10.2,
                        leading=16, spaceAfter=3),
    "callout_body": _p("callout_body", fontSize=10.4, leading=17.4,
                       textColor=PALETTE["ink"], spaceAfter=4.5),
    "footnote": _p("footnote", fontSize=8.4, leading=13,
                    textColor=PALETTE["ink_soft"], spaceAfter=2),
    "source": _p("source", fontSize=8.7, leading=14,
                  textColor=PALETTE["ink_soft"], alignment=TA_LEFT,
                  firstLineIndent=0, spaceAfter=3),

    "toc1": _p("toc1", fontName=FONT_BOLD, fontSize=10.2, leading=16.2,
               spaceBefore=3.2, alignment=TA_LEFT),
    "toc2": _p("toc2", fontSize=9.2, leading=13.2, textColor=PALETTE["ink_soft"],
                alignment=TA_LEFT),
    "toc3": _p("toc3", fontSize=8.4, leading=12.2,
                textColor=PALETTE["ink_faint"], leftIndent=12,
                alignment=TA_LEFT),

    "cover_title": _p("cover_title", fontName=FONT_BOLD, fontSize=31,
                      leading=42, alignment=TA_CENTER, spaceAfter=8,
                      firstLineIndent=0),
    "cover_sub": _p("cover_sub", fontSize=13.5, leading=22,
                    textColor=PALETTE["accent"], alignment=TA_CENTER,
                    spaceAfter=20, firstLineIndent=0),
    "cover_meta": _p("cover_meta", fontSize=9.8, leading=17,
                     textColor=PALETTE["ink_faint"], alignment=TA_CENTER,
                     firstLineIndent=0),
    "part": _p("part", fontName=FONT_BOLD, fontSize=17, leading=26,
               textColor=PALETTE["accent"], alignment=TA_LEFT,
               spaceBefore=4, spaceAfter=6),
}

CALLOUT_KINDS = {
    "tip": ("tip", "tip_bg", "阅读提示"),
    "key": ("key", "key_bg", "核心判断"),
    "warn": ("warn", "warn_bg", "常见误区"),
    "lab": ("accent", "accent_soft", "动手实验"),
    "source": ("ink_faint", "code_bg", "资料来源"),
}
