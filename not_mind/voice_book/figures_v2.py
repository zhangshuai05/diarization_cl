# -*- coding: utf-8 -*-
"""figures_v2.py — 《语音算法工程师教学》插图重制版（VS Code 友好柔和深色）。

约定（与 figures_teaching.py 兼容）：
  - 输出目录不变：voice_book/figures/，文件名与 md 引用完全一致；
  - 每张图固定分区：主标题 / 副标题 / 内容区 / 底部总结，全部用 fig.text 或
    axes 内注释，禁止依赖 bbox_inches="tight" 兜底布局；
  - 配色：Catppuccin Mocha 系降饱和，黑底预览不刺眼；
  - 注释风格：半透明深色文字框 + 细箭头指向元素（Mel 不误导版风格，底色换深色）。

运行：python voice_book/figures_v2.py
"""
from __future__ import annotations

import math
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 200,
})

# ---------------- Catppuccin Mocha（降饱和，VS Code 深色主题友好） ----------------
BG     = "#1E1E2E"   # 页面背景（贴近 VS Code 暗色）
PANEL  = "#313244"   # 面板背景
PANEL2 = "#45475A"   # 面板强调 / 网格
INK    = "#CDD6F4"   # 主文字
MUTED  = "#A6ADC8"   # 次要文字
FAINT  = "#7F849C"   # 弱文字
GRID   = "#45475A"
BLUE   = "#89B4FA"
TEAL   = "#94E2D5"
GREEN  = "#A6E3A1"
YELLOW = "#F9E2AF"
ORANGE = "#FAB387"
PINK   = "#F5C2E7"
PURPLE = "#CBA6F7"
RED    = "#F38BA8"
CYAN   = "#89DCEB"   # sky

# 注释框（深色半透明底 + 亮色边，避免黑底下亮块刺眼）
ANN_FC = "#2A2A3E"


# ============================================================
# 工具层
# ============================================================
def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.35,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[save] {name}")


def new_fig(title, subtitle, size, note=None, note_color=INK):
    """固定分区：标题 / 副标题 / 内容 / 底部总结。返回 (fig, 内容区默认 rect)。"""
    fig = plt.figure(figsize=size, facecolor=BG)
    fig.text(0.012, 0.972, title, va="top", ha="left", color=INK,
             fontsize=15.5, fontweight="bold")
    if subtitle:
        fig.text(0.012, 0.928, subtitle, va="top", ha="left", color=MUTED, fontsize=9.8)
    if note:
        fig.text(0.5, 0.018, note, va="bottom", ha="center", color=note_color,
                 fontsize=10, fontweight="bold")
    return fig


def axes_grid(fig, rows=1, rect=None, hspace=0.42):
    """在内容区创建 rows 行子图。rect 为 [left, bottom, w, h]。"""
    if rect is None:
        rect = [0.052, 0.118, 0.915, 0.715]
    if rows == 1:
        return fig.add_axes(rect)
    left, bottom, width, height = rect
    h = (height - hspace * (rows - 1)) / rows
    axs = []
    for i in range(rows):
        axs.append(fig.add_axes([left, bottom + i * (h + hspace), width, h]))
    return axs  # 自下而上


def style(ax, xlab=None, ylab=None, grid=True, yticks=None, xticks=None):
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    if grid:
        ax.grid(color=GRID, alpha=0.45, linewidth=0.6, zorder=0)
    if xlab:
        ax.set_xlabel(xlab, color=MUTED, fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, color=MUTED, fontsize=9)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)


def ptitle(ax, s, color=INK, fontsize=11.5, y=1.0, pad=7):
    ax.set_title(s, loc="left", color=color, fontsize=fontsize, fontweight="bold",
                 pad=pad, y=y)


def box(ax, x, y, w, h, label, color, lw=1.8, fontsize=10.5, fill=None):
    """圆角盒子 + 居中粗体文字。fill=None 时用页面同色底（深色风格）。"""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.05",
                       facecolor=fill or BG, edgecolor=color, linewidth=lw,
                       zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color=INK, fontsize=fontsize, fontweight="bold", zorder=4)
    return p


def arrow(ax, a, b, color=MUTED, lw=1.5, style="-|>", ms=11):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms,
                                 linewidth=lw, color=color, zorder=3))


def note_box(ax, x, y, s, color=YELLOW, fontsize=9, ha="left", va="center",
             arrow_to=None, lw=1.0):
    """半透明深色注释框（可选细箭头指向 arrow_to）。返回注释对象。"""
    kw = dict(bbox=dict(boxstyle="round,pad=0.38", fc=ANN_FC, ec=color,
                        alpha=0.94, lw=lw), zorder=6)
    if arrow_to is not None:
        return ax.annotate(s, xy=arrow_to, xytext=(x, y), color=color,
                           fontsize=fontsize, ha=ha, va=va,
                           arrowprops=dict(arrowstyle="->", color=color, lw=0.9,
                                           shrinkA=2, shrinkB=1), **kw)
    return ax.text(x, y, s, color=color, fontsize=fontsize, ha=ha, va=va, **kw)


def mel_triangles(nfilt=26, nfft=512, sr=16000):
    """经典 mel 滤波器组（hamming, mel=2595*log10(1+f/700)）→ Hz 轴三角形。"""
    low_mel = 0.0
    high_mel = 2595 * numpy_log10(1 + (sr / 2) / 700)
    mel_pts = np.linspace(low_mel, high_mel, nfilt + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bins = np.floor((nfft + 1) * hz_pts / sr)
    tris = []
    for m in range(1, nfilt + 1):
        a, b, c = int(bins[m - 1]), int(bins[m]), int(bins[m + 1])
        ks = np.arange(a, c + 1)
        w = np.zeros_like(ks, dtype=float)
        for i, k in enumerate(ks):
            if k < b:
                w[i] = (k - a) / (b - a)
            elif k > b:
                w[i] = (c - k) / (c - b)
            else:
                w[i] = 1.0
        tris.append((a, b, c, ks, w))
    return tris, hz_pts, mel_pts, bins


def numpy_log10(v):
    return float(np.log10(v))


def signal():
    p = ROOT / "generated" / "demo_mono.wav"
    try:
        with wave.open(str(p), "rb") as f:
            sr = f.getframerate()
            n = f.getnframes()
            x = np.frombuffer(f.readframes(n), dtype=np.int16).astype(float) / 32768.0
        return sr, x
    except Exception:
        sr = 16000
        t = np.arange(sr * 2) / sr
        return sr, 0.2 * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 440 * t) * (t > 0.55)


def stft(x, sr, win_ms=25, hop_ms=10, nfft=512):
    n = max(8, round(sr * win_ms / 1000))
    hop = max(1, round(sr * hop_ms / 1000))
    w = np.hanning(n)
    frames = [np.abs(np.fft.rfft(x[i:i + n] * w, nfft)) ** 2
              for i in range(0, max(1, len(x) - n), hop)]
    z = np.array(frames).T if frames else np.zeros((nfft // 2 + 1, 1))
    return z, np.arange(z.shape[1]) * hop / sr, np.fft.rfftfreq(nfft, 1 / sr)


# ============================================================
# 示范图 1：fig_role_map —— 五类岗位共享一条能力链（流程链 + 输入/输出/失败）
# ============================================================
def fig_role_map():
    fig = new_fig(
        "五个岗位，共享一条能力链",
        "岗位不同 → 交付的证据不同；但都走同一条链：测量 → 推断 → 交付 → 评测",
        (10.6, 4.9),
        "每一格都要能回答：输入是什么？输出什么？失败了怎么发现？",
        note_color=BLUE,
    )
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.83])
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.4)

    cols = [GREEN, BLUE, PURPLE, ORANGE, TEAL]
    rows = [
        ("① 信号与声学", "输入：原始 / 多通道音频", "输出：波形、频谱、ERLE", "失败：回声、噪声、混叠"),
        ("② 识别与理解", "输入：增强后的特征", "输出：CER/WER、FAR/FRR", "失败：错字、漏唤醒"),
        ("③ 数据与评测", "输入：标注 + 场景切片", "输出：数据版本、失败切片", "失败：标签错、覆盖缺"),
        ("④ 模型与训练", "输入：特征 + 标签", "输出：checkpoint / 推理图", "失败：分布漂移"),
        ("⑤ 工程与部署", "输入：运行时 + 设备预算", "输出：RTF、P95、内存", "失败：超时、超预算"),
    ]
    w, h = 1.82, 1.72
    for i, ((name, inp, outp, fail), c) in enumerate(zip(rows, cols)):
        x = 0.22 + i * 2.02
        y = 2.35
        box(ax, x, y, w, h, "", c, lw=1.8)
        ax.text(x + w / 2, y + h - 0.34, name, ha="center", va="center",
                color=c, fontsize=11.5, fontweight="bold", zorder=4)
        for j, line in enumerate([inp, outp, fail]):
            ax.text(x + 0.14, y + h - 0.78 - j * 0.33, line, ha="left", va="center",
                    color=INK if j < 2 else RED, fontsize=8.2, zorder=4)
        if i < 4:
            arrow(ax, (x + w + 0.06, y + h / 2), (x + w + 0.24, y + h / 2), MUTED, 1.6)
    # 底部：验收证据示例
    note_box(ax, 0.22, 0.62, "验收证据各不相同：① ERLE/SNR　② CER/唤醒率　③ 数据版本　④ 可复现实验　⑤ 延迟与内存",
             color=MUTED, fontsize=9)
    save(fig, "fig_role_map.png")


# ============================================================
# 示范图 2：fig03_features —— 七步流程 + 26 个真实 Mel 三角形（修正 5 个示意三角）
# ============================================================
def fig03_features():
    fig = new_fig(
        "从波形到 Fbank / MFCC",
        "每一步都把输入变成更适合模型检查的表示（箭头 = 这一步的输入 → 输出）",
        (10.2, 5.6),
        "Mel 滤波器的 26 个三角形：在 Mel 轴上等距取点 → 映射回 Hz 轴 → 频率低处窄而密、高处宽而稀",
        note_color=PURPLE,
    )
    # ---- 上 panel：七步流程 ----
    ax = fig.add_axes([0.02, 0.56, 0.96, 0.275])
    ax.axis("off")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 2.9)
    steps = [
        ("波形", GREEN, "PCM 样本\n采样率 / 位深"),
        ("分帧\n加窗", BLUE, "25 ms 一帧\n帧移 10 ms"),
        ("FFT", BLUE, "257 个 bin\n每 bin 31.25 Hz"),
        ("Mel\n滤波", PURPLE, "加权求和\n26 个频带"),
        ("取 log", PURPLE, "20·log10\n接近人耳响度"),
        ("Fbank", ORANGE, "26 维特征\n喂给模型"),
        ("DCT\nMFCC", ORANGE, "去相关\n常用 13~40 维"),
    ]
    w, h = 1.28, 1.42
    for i, (lab, c, sub) in enumerate(steps):
        x = 0.22 + i * 1.40
        y = 0.95
        box(ax, x, y, w, h, lab, c, lw=1.8, fontsize=10)
        ax.text(x + w / 2, y - 0.24, sub, ha="center", va="center", color=MUTED,
                fontsize=7.6, zorder=4, linespacing=1.35)
        if i < 6:
            arrow(ax, (x + w + 0.07, y + h / 2), (x + w + 0.25, y + h / 2), MUTED, 1.5)
    # ---- 下 panel：26 个三角形 + 高亮第 13 个 ----
    ax2 = fig.add_axes([0.052, 0.145, 0.915, 0.36])
    ax2.set_facecolor(PANEL2)
    tris, hz_pts, mel_pts, bins = mel_triangles()
    f = np.linspace(0, 8000, 1601)
    m_show = 13  # 1-based
    for idx, (a, b, c_, ks, w) in enumerate(tris):
        f0, f1, f2 = a * 16000 / 512, b * 16000 / 512, c_ * 16000 / 512
        tri = np.maximum(0, np.minimum((f - f0) / max(f1 - f0, 1e-9),
                                       (f2 - f) / max(f2 - f1, 1e-9)))
        if idx == m_show - 1:
            ax2.plot(f, tri, color=ORANGE, lw=2.2, zorder=5)
            ax2.fill_between(f, 0, tri, color=ORANGE, alpha=0.30, zorder=4)
        else:
            ax2.plot(f, tri, color=GREEN, lw=0.7, alpha=0.75, zorder=3)
            ax2.fill_between(f, 0, tri, color=GREEN, alpha=0.07, zorder=2)
    apex_bin = int(bins[m_show])
    apex_hz = apex_bin * 16000 / 512  # bin 对应的频率（与"bin 53"一致）
    ax2.plot([apex_hz], [1.0], "o", color=ORANGE, ms=8, zorder=6)
    ax2.annotate(f"第 {m_show} 个三角形（主角）：顶点 = bin {apex_bin} = {apex_hz:.0f} Hz\n"
                 f"覆盖 bin {int(bins[m_show - 1])} ~ {int(bins[m_show + 1])}（"
                 f"{int(bins[m_show - 1]) * 16000 / 512:.0f} ~ {int(bins[m_show + 1]) * 16000 / 512:.0f} Hz）",
                 xy=(apex_hz, 1.0), xytext=(apex_hz + 480, 0.64),
                 fontsize=9, color=ORANGE, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.35", fc=ANN_FC, ec=ORANGE, alpha=0.94),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.1), zorder=7)
    note_box(ax2, 5450, 0.60, "等距发生在 Mel 轴：低频三角形窄（人耳分辨力高）\n高频三角形宽（人耳不敏感，少浪费分辨力）",
             color=GREEN, fontsize=8.6)
    ax2.set_xlim(-60, 8150)
    ax2.set_ylim(-0.04, 1.22)
    style(ax2, "频率 (Hz)", "权重 (0~1)", grid=True)
    ptitle(ax2, "26 个 Mel 三角形在 Hz 轴上的真实样子（等距是 Mel 轴上的性质）", fontsize=11)
    save(fig, "fig03_features.png")


# ============================================================
# 示范图 3：fig02_wave_spectrum —— 波形 / 整段 FFT / STFT（真实信号 + 注释）
# ============================================================
def fig02_wave_spectrum():
    sr, x = signal()
    x = x[: min(len(x), sr * 2)]
    t = np.arange(len(x)) / sr
    nfft = 1 << int(math.ceil(math.log2(len(x))))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    s /= max(s.max(), 1e-9)
    z, tt, ff = stft(x, sr)

    # 自动找特征：onset（能量首次超过均值）与 FFT 基频峰
    win = 400
    eng = np.array([np.sum(x[i:i + win] ** 2) for i in range(0, len(x) - win, win)])
    onset = (np.argmax(eng > eng.mean()) * win) / sr if eng.size else 0.0
    k_peak = int(np.argmax(s))
    f_peak = f[k_peak]

    fig = new_fig(
        "同一段语音：时域波形 / 整段 FFT / 时频图",
        f"16 kHz 采样 · 波形回答“什么时候响”，整段 FFT 回答“有哪些频率”，STFT 回答“什么频率在什么时候出现”",
        (9.6, 7.0),
        "波形看强弱，FFT 看频率，STFT 同时看频率和时间。",
        note_color=BLUE,
    )
    # 三面板（自下而上）：ax3 底部、ax2、ax1 顶部，房间距留给面板标题
    ax1 = fig.add_axes([0.065, 0.615, 0.90, 0.18])
    ax2 = fig.add_axes([0.065, 0.375, 0.90, 0.18])
    ax3 = fig.add_axes([0.065, 0.135, 0.90, 0.18])

    # ---- ① 波形 ----
    ax1.set_facecolor(PANEL)
    ax1.plot(t, x, color=TEAL, lw=0.55)
    ax1.axvline(onset, color=YELLOW, ls="--", lw=1.0)
    note_box(ax1, 0.02, 0.62, f"信号起点 {onset:.2f}s：\n能量跃升（波形可见地变高）", color=YELLOW,
             fontsize=8.4, arrow_to=(onset, 0.42))
    style(ax1, "时间 (s)", "振幅", grid=True)
    ptitle(ax1, "① 波形：声音什么时候变响", fontsize=10.5, pad=3)

    # ---- ② 整段 FFT ----
    ax2.set_facecolor(PANEL)
    ax2.plot(f, s, color=YELLOW, lw=0.9)
    ax2.fill_between(f, s, 0, color=YELLOW, alpha=0.14)
    for h, c in [(1, ORANGE), (2, ORANGE), (3, ORANGE)]:
        if h * k_peak < len(s):
            ax2.vlines(h * f_peak, 0, s[h * k_peak], color=ORANGE, ls=":", lw=1.0)
    note_box(ax2, 2900, 0.80, f"最高峰 {f_peak:.0f} Hz ≈ 基频\n点线 = 2、3 次谐波（{2*f_peak:.0f} / {3*f_peak:.0f} Hz）",
             color=ORANGE, fontsize=8.4, arrow_to=(f_peak + 30, 0.97))
    note_box(ax2, 2750, 0.26, "整段 FFT 丢失时间信息：\n只能看出“哪些频率存在”，看不出先后",
             color=RED, fontsize=8.4, arrow_to=(f_peak + 30, 0.45))
    ax2.set_xlim(0, 5000)
    ax2.set_ylim(0, 1.05)
    style(ax2, "频率 (Hz)", "相对能量", grid=True)
    ptitle(ax2, "② 整段 FFT：有哪些频率，但不知道何时出现", fontsize=10.5, pad=3)

    # ---- ③ STFT ----
    ax3.set_facecolor(PANEL)
    ax3.pcolormesh(tt, ff, 10 * np.log10(z + 1e-8), shading="auto", cmap="magma", zorder=2)
    ax3.axvline(onset, color=YELLOW, ls="--", lw=1.0)
    note_box(ax3, 1.42, 1620, f"{onset:.2f}s 处颜色由暗转亮 ↗\n能量带集中在 {f_peak:.0f} Hz 附近",
             color=YELLOW, fontsize=8.4, arrow_to=(onset + 0.08, 320))
    ax3.set_xlim(t[0], t[-1])
    ax3.set_ylim(0, 2000)
    style(ax3, "时间 (s)", "频率 (Hz)", grid=True)
    ptitle(ax3, "③ STFT：把时间切成小格（每帧 25 ms / 帧移 10 ms），保留变化过程", fontsize=10.5, pad=3)
    save(fig, "fig02_wave_spectrum.png")


def flow_chain(ax, items, y=2.0, w_scale=1.0, box_h=1.62, name_size=11, sub_size=8.2):
    """通用流程图：items = [(名称, 颜色, 小注行列表), ...]，盒子内右上名称 + 小注。"""
    n = len(items)
    step = 9.9 / n
    w = min(step - 0.34, 1.9) * w_scale
    total_w = n * step - 0.2
    x0 = (10.2 - total_w) / 2
    for i, (name, c, lines) in enumerate(items):
        x = x0 + i * step
        box(ax, x, y, w, box_h, "", c, lw=1.8)
        ax.text(x + w / 2, y + box_h - 0.36, name, ha="center", va="center",
                color=c, fontsize=name_size, fontweight="bold", zorder=4)
        for j, line in enumerate(lines):
            col = INK if not line.startswith("失败") else RED
            ax.text(x + 0.14, y + box_h - 0.80 - j * 0.31, line, ha="left",
                    va="center", color=col, fontsize=sub_size, zorder=4,
                    linespacing=1.2)
        if i < n - 1:
            arrow(ax, (x + w + 0.05, y + box_h / 2), (x + w + 0.23, y + box_h / 2), MUTED, 1.5)
    return x0


# ============================================================
# 批 1：流程链 / 数据类 5 张
# ============================================================

def fig02_stft_windows():
    """STFT 分帧：帧长 25ms、帧移 10ms、重叠 15ms（真实波形 + 三个窗口）。"""
    sr, x = signal()
    x = x[: min(len(x), sr * 2)]
    frame_ms, hop_ms = 25, 10
    n = round(sr * frame_ms / 1000)
    hop = round(sr * hop_ms / 1000)
    start = min(round(0.62 * sr), len(x) - n - 2 * hop)
    y = x[start: start + n + 2 * hop]
    y /= max(np.max(np.abs(y)), 1e-9)
    tm = np.arange(len(y)) / sr * 1000  # ms

    fig = new_fig(
        "STFT 第一步：沿着波形滑动窗口",
        f"每帧观察 {frame_ms} ms（16 kHz → {n} 个采样点），每隔 {hop_ms} ms 产生一帧（{hop} 点）",
        (10.0, 5.2),
        "25 ms = 10 ms + 15 ms　｜　16 kHz 下：400 点 = 160 点 + 240 点",
        note_color=BLUE,
    )
    ax = fig.add_axes([0.045, 0.145, 0.93, 0.66])
    ax.set_facecolor(PANEL)
    ax.plot(tm, y, color=INK, lw=0.8, zorder=2)
    ax.axhline(0, color=GRID, lw=0.8)
    win_cols = [BLUE, GREEN, BLUE]
    for i, s in enumerate([0, hop, 2 * hop]):
        ax.add_patch(Rectangle((s / sr * 1000, -1.15), frame_ms, 2.3,
                               facecolor=win_cols[i], alpha=0.16, edgecolor=win_cols[i],
                               lw=2.0, zorder=3))
        ax.text(s / sr * 1000 + frame_ms / 2, 0.95, f"第 {i + 1} 帧", color=win_cols[i],
                fontsize=10, ha="center", weight="bold", zorder=5)
    # 帧移 / 重叠标注
    ax.annotate("", xy=(hop / sr * 1000, 1.32), xytext=(0, 1.32),
                arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=1.6), zorder=5)
    ax.text(hop / sr * 1000 / 2, 1.42, f"帧移 {hop_ms} ms", color=YELLOW, fontsize=9.6,
            ha="center", zorder=5)
    ax.annotate("", xy=(frame_ms + hop / sr * 1000, -1.42), xytext=(frame_ms, -1.42),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.6, linestyle="--"), zorder=5)
    ax.text(frame_ms + hop / sr * 1000 / 2, -1.54, f"重叠 {frame_ms - hop_ms} ms", color=TEAL,
            fontsize=9.6, ha="center", zorder=5)
    note_box(ax, 2.5, 0.12,
             "窗长是取舍：短窗时间清楚、频率模糊；长窗频率清楚、时间模糊。\n"
             "工程上：要尽快出字用短窗；要分辨稳定音高 / 低频细节用长窗",
             color=BLUE, fontsize=8.6)
    ax.set_xlim(-2, (n + 2 * hop) / sr * 1000 + 2)
    ax.set_ylim(-1.75, 1.6)
    style(ax, "时间 (ms)", None, grid=True)
    ax.set_yticks([])
    ptitle(ax, "① 三个相邻分析窗（25 ms 宽）在波形上的位置", fontsize=11)
    save(fig, "fig02_stft_windows.png")


def fig_frontend_3a():
    """3A 前端：参考播放 → 采集 → AEC → ANS → AGC → VAD/KWS（双工路径）。"""
    fig = new_fig(
        "3A 不是三个孤立按钮，而是一条双工信号路径",
        "麦克风听到的是：用户声音 + 扬声器回放 + 房间反射 + 设备噪声；AEC/ANS/AGC 逐级改写 ASR 看到的证据",
        (10.8, 5.4),
        "排查顺序：先存原始近端 + 参考 → 分别试听 AEC/ANS/AGC 输出 → 最后看 VAD 的帧级决策",
        note_color=GREEN,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.6)
    flow_chain(ax, [
        ("参考播放", YELLOW, ["输入：TTS / 远端音频", "输出：扬声器播放信号", "失败：参考丢失 / 未同步"]),
        ("麦克风采集", BLUE, ["输入：近端麦克风 PCM", "输出：混合信号（含回声）", "失败：通道交错 / 增益漂移"]),
        ("AEC", GREEN, ["输入：近端 + 参考", "输出：近端语音（除回声）", "失败：双讲 / 路径变化"]),
        ("ANS", PURPLE, ["输入：噪声混合信号", "输出：降噪近端", "失败：削辅音 / 残留噪声"]),
        ("AGC", ORANGE, ["输入：动态范围不定", "输出：稳定响度", "失败：削波 / 增益突变"]),
        ("VAD / KWS", CYAN, ["输入：增强后每帧", "输出：语音段 / 关键词", "失败：误触发 / 漏唤醒"]),
    ], y=1.15, box_h=1.95, name_size=10.5, sub_size=7.8)
    note_box(ax, 0.25, 0.42, "ERLE 只是诊断量：它变大≠近端语音没被削\n听感、ASR、KWS、双讲、功耗的目标可能要求不同前端强度",
             color=RED, fontsize=8.8)
    note_box(ax, 6.7, 0.42, "快速打断要同时监听：唤醒词 + 用户语音 + 系统播报\n每条状态都必须有优先级与回退条件",
             color=TEAL, fontsize=8.8)
    save(fig, "fig_frontend_3a.png")


def fig_asr_families():
    """ASR 家族：对齐规则决定系统行为（四行对照 + 底部四个边界问题）。"""
    fig = new_fig(
        "ASR 模型家族：差异在对齐和历史依赖",
        "不要只背模型名字，先问：模型怎么把帧映射到输出？能看到多少未来？",
        (10.6, 5.6),
        "先问四个边界：训练标签？推理能看多远？结果能否修订？设备能承受多少缓存与计算？",
        note_color=PURPLE,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.2)
    rows = [
        ("DNN-HMM", "帧 → 隐状态", "依赖：状态转移 / 词典 + LM", "流式：自然（与解码器配合）", "失败：需要词表和语言模型", GREEN),
        ("CTC", "帧 → 字符 + blank", "依赖：帧间条件独立", "流式：适合（无未来依赖）", "失败：尖峰 ≠ 真实词边界", BLUE),
        ("AED", "整段编码 → 自回归逐字", "依赖：完整文本历史", "流式：难（需要右上下文）", "失败：缓存 / 修订复杂", PURPLE),
        ("RNN-T", "声学时间 × 标签历史网格", "依赖：已输出文本", "流式：边听边出（单调）", "失败：训练 / 解码实现复杂", ORANGE),
    ]
    x0, names_w = 0.28, 1.35
    hdr = ["模型", "对齐方式", "历史依赖", "流式能力", "典型失败"]
    for j, h in enumerate(hdr):
        ax.text(x0 + names_w + j * 1.9, 4.9, h, color=FAINT, fontsize=8.6,
                ha="left", va="center", weight="bold")
    for i, (name, align, dep, stream, fail, c) in enumerate(rows):
        y = 4.0 - i * 0.92
        box(ax, x0, y - 0.36, 1.25, 0.72, name, c, lw=1.8, fontsize=10.5)
        for j, txt in enumerate([align, dep, stream, fail]):
            col = RED if txt.startswith("失败") else INK
            ax.text(x0 + names_w + j * 1.9, y, txt, color=col, fontsize=8.8,
                    ha="left", va="center")
        ax.plot([x0, 10.3], [y - 0.52, y - 0.52], color=GRID, lw=0.6, alpha=0.6)
    note_box(ax, 0.28, 0.30, "Conformer 的局部卷积 + 全局注意力解决的是“表示能力”，\n"
             "不等于任何设备都能流式实时：chunk、缓存和右上下文要写出来",
             color=PURPLE, fontsize=8.8)
    save(fig, "fig_asr_families.png")


def fig_streaming_latency():
    """流式：同一系统至少有三条时间线（音频 / 处理 / 用户可见事件）。"""
    t_say, t_end, t_first, t_final = 0.8, 3.0, 4.2, 5.9
    fig = new_fig(
        "流式系统至少有三条时间线",
        "音频发生顺序 ≠ 网络到达顺序 ≠ 用户看到顺序；首字、句尾、会后重算是三个不同指标",
        (10.4, 5.6),
        "RTF < 1 只说明平均算得快，首字仍可能很慢（要积累右上下文）。",
        note_color=BLUE,
    )
    ax = fig.add_axes([0.05, 0.115, 0.92, 0.70])
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    segs = [
        ("音频时间", [("说“你  好”", 1.6, 2.0, GREEN), ("停顿 + 噪声", 3.6, 1.5, FAINT)]),
        ("处理管线", [("采集缓冲", 1.2, 0.9, MUTED), ("分帧 / 右上下文", 2.1, 1.0, BLUE),
                    ("前端", 3.1, 0.5, TEAL), ("模型 + 解码", 3.6, 1.5, PURPLE),
                    ("稳定化", 5.1, 1.3, ORANGE)]),
        ("用户可见", [(f"首字 partial 在 {t_first:.1f}s 上屏", t_first, 1.7, YELLOW),
                    (f"句尾 final 在 {t_final:.1f}s 定稿", t_final, 1.6, CYAN)]),
    ]
    for i, (lab, items) in enumerate(segs):
        y = 3.9 - i * 1.15
        ax.text(0.05, y + 0.28, lab, color=INK, fontsize=9.5, weight="bold",
                va="center", ha="left", zorder=5)
        for txt, x, w, c in items:
            ax.broken_barh([(x, w)], (y - 0.20, 0.40), facecolors=c, edgecolor=c,
                           alpha=0.85, zorder=3)
            ax.text(x + w / 2, y + 0.28, txt, color=c, fontsize=8.2, ha="center",
                    va="center", zorder=5)
    x0 = 10.2
    for y, x_start, x_end in [(3.0 - 0.20, 1.6, 3.6), (2.75 - 0.20, 1.2, 5.1)]:
        pass
    # 斜线：音频“你说完” → 用户可见 final（虚线示意跨时间线对应）
    ax.plot([t_end, t_final + 0.1], [3.0, 2.75], color=CYAN, ls=":", lw=1.0, alpha=0.8, zorder=4)
    note_box(ax, 6.4, 3.65, f"首字延迟 = partial 出现({t_first:.1f}s) − 用户开口({t_say:.1f}s)\n"
             f"≈ {t_first - t_say:.1f}s；句尾延迟 = final({t_final:.1f}s) − 说完({t_end:.1f}s) ≈ {t_final - t_end:.1f}s",
             color=YELLOW, fontsize=8.6)
    note_box(ax, 6.9, 0.55, "会后重算：用更完整上下文覆盖临时结果\n不占首字预算，但必须保留 revision 链",
             color=CYAN, fontsize=8.6)
    ax.set_yticks([])
    ax.tick_params(bottom=False, labelbottom=False)
    style(ax, "统一媒体时间（示意）", None, grid=False)
    save(fig, "fig_streaming_latency.png")


def fig_data_loop():
    """数据闭环：采集 → 清洗切分 → 训练 → 固定评测 → 失败切片 → 回灌。"""
    fig = new_fig(
        "评测不是终点，而是下一轮数据入口",
        "训练 pipeline 的第一项产物不是 checkpoint，而是能说明数据来源、保留原因与切分规则的版本",
        (10.8, 5.4),
        "失败样例要能回到原始音频；增强必须保留参数与随机种子",
        note_color=RED,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.7)
    flow_chain(ax, [
        ("采集", GREEN, ["记录：许可 / 时长分布", "输出：原始音频 + 元数据", "失败：格式 / 采样率混乱"]),
        ("清洗 / 切分", BLUE, ["文本归一化 / 去重 / 隔离", "输出：训练 / 开发 / 测试", "失败：说话人泄漏"]),
        ("训练", PURPLE, ["增强：参数 + 随机种子", "输出：模型 + 训练日志", "失败：漂移 / 过拟合"]),
        ("固定评测", ORANGE, ["固定集锁住录音变化", "输出：CER + 切片 + 区间", "失败：测试集当调参集"]),
        ("失败切片", RED, ["按场景 × 错误类型切", "输出：10 条可回放样例", "失败：只有平均分"]),
        ("回灌", CYAN, ["原始音频 + 配置 + 版本", "输出：下一轮训练建议", "失败：丢弃了失败证据"]),
    ], y=1.35, box_h=1.95, name_size=10.5, sub_size=7.8)
    note_box(ax, 0.25, 0.35, "切分必须按会议 / 说话人隔离：同一会议的片段\n不能随机撒到训练集与测试集两边",
             color=BLUE, fontsize=8.8)
    note_box(ax, 6.85, 0.35, "小测试集差异要报 bootstrap 区间；\n跨过“无改善”边界就写“暂未证明改善”",
             color=ORANGE, fontsize=8.8)
    save(fig, "fig_data_loop.png")


# ============================================================
# 批 2：CTC / 阵列 / 说话人 / 评测 5 张
# ============================================================

def fig04_ctc_collapse():
    """CTC：多条帧级路径坍缩到同一文本（blank 折叠 + 重复合并）。"""
    fig = new_fig(
        "CTC 为什么可以把重复和空白压掉",
        "每帧输出 blank 或字符；先合并连续重复，再去掉 blank——多条路径变成同一句话",
        (10.6, 5.4),
        "路径求和：P(y|x) = Σ_π ∏_t P(π_t|x)，π 为所有能折叠到 y 的帧级路径　｜　blank 不是真实词边界",
        note_color=PURPLE,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.9)
    paths = [
        (["blank", "你", "你", "blank", "好", "好", "blank"], 3.72, BLUE, "路径 A"),
        (["你", "blank", "你", "好", "blank", "好", "blank"], 2.42, GREEN, "路径 B"),
    ]
    FH = 0.60
    for cells, yc, c, pname in paths:
        ax.text(0.12, yc, pname, color=c, fontsize=9.5, weight="bold", ha="left", va="center")
        for j, v in enumerate(cells):
            cc = GRID if v == "blank" else c
            box(ax, 0.72 + j * 1.12, yc - FH / 2, 0.92, FH, v, cc, lw=1.5, fontsize=9)
            ax.text(1.18 + j * 1.12, yc - FH / 2 - 0.15, f"t{j + 1}", color=FAINT,
                    fontsize=7.6, ha="center")
    # 合流：两行路径 → 底部“你好”（箭头走 t3/t4 之间的盒缝，不穿过盒子）
    arrow(ax, (3.98, 3.41), (3.98, 1.98), PURPLE, 1.7)
    arrow(ax, (3.98, 2.70), (3.98, 1.98), PURPLE, 1.7)
    box(ax, 3.2, 1.08, 2.6, 0.88, "你好", PURPLE, lw=2.2, fontsize=15)
    note_box(ax, 8.55, 2.02, "不同路径（每帧独立打分）\n都可以折叠成同一个文本：\n“你 你 好 好” → “你好”",
             color=YELLOW, fontsize=8.6)
    note_box(ax, 0.30, 0.45, "训练不需要逐帧标签：把所有能坍缩到目标\n文本的路径概率求和即可（概率 = 路径和）",
             color=CYAN, fontsize=8.6)
    note_box(ax, 6.6, 3.04, "连续重复先合并，再去掉 blank：\n“你 你 好 好” → “你 好”",
             color=GREEN, fontsize=8.6)
    save(fig, "fig04_ctc_collapse.png")


def fig08_array_geometry():
    """阵列几何：间距决定可观测的空间变化（TDOA + 混叠上界 + 远场条件）。"""
    fig = new_fig(
        "麦克风阵列：间距决定能观察到的空间变化",
        "同一声源到达不同麦克风的时间不同 → TDOA；间距 d 与声速 c 一起决定空间混叠上界",
        (10.4, 5.4),
        "多通道必须真实同步：把一路 PCM 复制成 8 路，空间自由度不会增加",
        note_color=TEAL,
    )
    ax = fig.add_axes([0.045, 0.10, 0.93, 0.70])
    ax.set_facecolor(PANEL)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 8.6)
    ax.set_ylim(-1.6, 3.4)
    # 4 个麦克风（物理 d=5cm；图上间距为示意放大），声源方向 30°
    d_show = 0.55
    xs = np.array([0, d_show, 2 * d_show, 3 * d_show])
    theta = np.deg2rad(30)
    ax.scatter(xs, np.zeros(4), s=120, color=BLUE, zorder=5)
    ax.text(2.15, -0.30, "4 麦克风：物理间距 d = 5 cm（图上示意放大）",
            color=BLUE, fontsize=8.6, ha="left")
    # 声源方向（30°）+ 波前（垂直于方向、等间距的平行线）
    L = 7.2
    ux, uy = np.cos(theta), np.sin(theta)
    ax.plot([xs[1], xs[1] + ux * L], [0, uy * L], color=ORANGE, lw=2.0, zorder=4)
    ax.text(xs[1] + ux * L * 0.72, uy * L * 0.72 + 0.25, "声源方向 θ = 30°",
            color=ORANGE, fontsize=9)
    for s0 in np.arange(-4.5, 5.5, 1.0):   # 垂直方向的波前线
        px, py = 1.5 - uy * s0, 0 + ux * s0
        ax.plot([px - uy * 0.9, px + uy * 0.9], [py + ux * 0.9, py - ux * 0.9],
                color=ORANGE, lw=0.7, alpha=0.35, zorder=3)
    # 相邻麦克风间距标注 + TDOA
    ax.annotate("", xy=(xs[0], 0.62), xytext=(xs[1], 0.62),
                arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=1.5))
    ax.text(xs[1] / 2, 0.78, "d", color=YELLOW, ha="center", fontsize=10.5)
    note_box(ax, -1.15, 2.05, "相邻通道到达时间差：\nτ = d·sinθ / c（平面波）", color=YELLOW, fontsize=8.8,
             arrow_to=(xs[1], 0.35))
    note_box(ax, 5.55, 0.45, "空间混叠上界（设计检查起点）：\nf < c/(2d) ≈ 3430 Hz @ d=5cm", color=CYAN,
             fontsize=8.8, arrow_to=(2.0, 0.05))
    note_box(ax, 3.95, -0.85, "远场假设边界：r > 2D²/λ\n（近场需考虑波前曲率）", color=GREEN, fontsize=8.8)
    ax.set_xlabel("x（平面上）", color=MUTED, fontsize=9)
    ax.set_ylabel("y", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8, labelleft=False, labelbottom=False)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, alpha=0.35, linewidth=0.6)
    ptitle(ax, "① 声源方向 30° 时，4 个麦克风看到不同的到达时间", fontsize=11)
    save(fig, "fig08_array_geometry.png")


def fig08_beam_pattern():
    """Delay-and-Sum 方向响应：导向角错了，目标方向能量不能相干叠加。"""
    ang = np.linspace(-90, 90, 721)

    def resp(steer):
        d, f0, c_, n = 0.05, 1000.0, 343.0, 8
        idx = np.arange(n) - (n - 1) / 2
        tau = d * idx[:, None] * (np.sin(np.deg2rad(ang))[None, :] -
                                  np.sin(np.deg2rad(steer))) / c_
        a = np.abs(np.sum(np.exp(1j * 2 * np.pi * f0 * tau), axis=0) / n)
        return 20 * np.log10(np.maximum(a, 1e-4))

    r_ok, r_bad = resp(20), resp(-20)
    fig = new_fig(
        "波束形成：导向角错了，峰值就会偏",
        "Delay-and-Sum 用估计时延把各通道对齐后相加；导向角错 → 目标方向不能相干叠加，甚至被削弱",
        (10.4, 5.2),
        "单频理想化示例；真实会议要回到录音、SNR、CER、DER 与方向误差验收",
        note_color=ORANGE,
    )
    ax = fig.add_axes([0.055, 0.13, 0.92, 0.66])
    ax.set_facecolor(PANEL)
    ax.plot(ang, r_ok, color=GREEN, lw=2.0, label="正确导向 20°")
    ax.plot(ang, r_bad, color=ORANGE, lw=1.6, ls="--", label="错误导向 −20°")
    ax.axvline(20, color=GREEN, ls=":", lw=1.0)
    ax.axvline(-20, color=ORANGE, ls=":", lw=1.0, alpha=0.6)
    ax.axhline(0, color=GRID, lw=0.8)
    # 错误导向在 20° 处的损失
    i20 = int(np.argmin(abs(ang - 20)))
    ax.plot([20], [r_bad[i20]], "o", color=ORANGE, ms=7, zorder=6)
    note_box(ax, 44, r_bad[i20] + 1.2, f"错误导向 −20° 时，在真实方向 20° 处的\n响应 ≈ {r_bad[i20]:.0f} dB（目标能量被削弱）",
             color=ORANGE, fontsize=8.8, arrow_to=(24, r_bad[i20] + 0.5))
    note_box(ax, 38, -31.5, "y(t) = Σ_m w_m · x_m(t − τ_m)\nτ_m = 按估计方向计算的时延",
             color=BLUE, fontsize=8.8)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-38, 3)
    style(ax, "声源方向 (°)", "响应 (dB)", grid=True)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="lower center")
    save(fig, "fig08_beam_pattern.png")


def fig07_diarization_flow():
    """从 VAD 到 embedding 再到聚类：子窗级 embedding + 阈值聚类 + 错误示例。"""
    fig = new_fig(
        "说话人分段：先切子窗，再聚类",
        "整段只生成一个 embedding 会把说话人切换混在一起；在线算法只能看过去",
        (10.6, 6.6),
        "阈值太低 → 过度合并；阈值太高 → 簇过多；方向信息只能作辅助证据",
        note_color=ORANGE,
    )
    ax = fig.add_axes([0.02, 0.545, 0.96, 0.335])
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.9)
    flow_chain(ax, [
        ("VAD", GREEN, ["输入：每帧能量/特征", "输出：语音活动段", "失败：静音段误判"]),
        ("固定长度子窗", BLUE, ["输入：VAD 段内切分", "输出：~1s 小窗", "失败：跨切换边界"]),
        ("每窗一个 embedding", PURPLE, ["输入：窗口特征", "输出：声纹向量", "失败：漂移/混响影响"]),
        ("聚类", ORANGE, ["输入：向量序列", "输出：匿名簇标签", "失败：阈值不当"]),
        ("标签", CYAN, ["SPEAKER_00/01", "输出：谁在何时说话", "失败：不能直接实名"]),
    ], y=1.35, box_h=2.0, name_size=10.2, sub_size=7.6)

    ax2 = fig.add_axes([0.09, 0.135, 0.87, 0.33])
    ax2.set_facecolor(PANEL)
    ax2.set_xlim(0, 6.0)
    ax2.set_ylim(0, 4.4)
    ax2.set_yticks([3.0, 1.9, 0.8])
    ax2.set_yticklabels(["参考（真实）", "子窗聚类", "错误示例"], color=MUTED, fontsize=8.6)
    ref = [("A", 0.3, 1.5, BLUE), ("B", 1.9, 1.4, GREEN), ("A", 3.4, 1.6, BLUE)]
    for lab, x, w, c in ref:
        ax2.broken_barh([(x, w)], (2.80, 0.42), facecolors=c, alpha=0.9)
        ax2.text(x + w / 2, 3.30, f"SPK_{lab}", color=c, fontsize=8.6, ha="center")
    subs = [(0.3, 0.5, BLUE), (0.8, 0.5, BLUE), (1.3, 0.5, BLUE),
            (1.9, 0.5, GREEN), (2.4, 0.5, GREEN), (2.9, 0.5, GREEN),
            (3.4, 0.5, BLUE), (3.9, 0.5, BLUE), (4.4, 0.5, BLUE)]
    for x, w, c in subs:
        ax2.broken_barh([(x, w)], (1.72, 0.42), facecolors=c, alpha=0.55)
        ax2.text(x + w / 2, 2.28, "E", color=c, fontsize=7.6, ha="center")
    note_box(ax2, 5.15, 2.10, "每个子窗一个 embedding，\n与已有簇比较后合并", color=PURPLE, fontsize=8.4,
             arrow_to=(4.6, 1.93))
    ax2.broken_barh([(0.3, 4.7)], (0.62, 0.42), facecolors=PANEL2, alpha=0.9)
    ax2.add_patch(Rectangle((0.3, 0.62), 4.7, 0.42, fill=False, edgecolor=RED, lw=1.6, ls="--"))
    ax2.text(2.65, 1.13, "× 跨越切换只求一个向量 → 两个说话人混在一起", color=RED, fontsize=8.6, ha="center")
    ax2.set_xticks([0, 1, 2, 3, 4, 5, 6])
    style(ax2, "时间 (s)", None, grid=True)
    ptitle(ax2, "① 子窗聚类 vs 整段一个向量（错误示例）", fontsize=11)
    save(fig, "fig07_diarization_flow.png")


def fig_eval_matrix():
    """评测矩阵：场景切片 × 产品指标 + 图例。"""
    scenes = ["近讲", "远场", "噪声", "重叠", "口音"]
    metrics = ["CER/WER", "DER", "SI-SDR", "P95 延迟", "内存"]
    vals = np.array([[1, 1, 1, 1, 1],
                     [0, 1, 0, 1, 1],
                     [0, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 1]])
    fig = new_fig(
        "评测矩阵：场景 × 指标",
        "单一总分不能替代分场景证据；先找空格子，再决定下一轮采集",
        (10.4, 5.6),
        "每格都要指定：输入条件、指标口径、时间边界；SI-SDR 需干净参考；DER 需写明 collar/重叠口径",
        note_color=ORANGE,
    )
    ax = fig.add_axes([0.115, 0.215, 0.80, 0.575])
    ax.set_facecolor(PANEL)
    rows, cols = vals.shape
    for i in range(rows):
        for j in range(cols):
            on = bool(vals[i, j])
            ax.add_patch(Rectangle((j, rows - 1 - i), 1, 1, facecolor=BLUE if on else PANEL2,
                                   edgecolor=BG, lw=2.0, alpha=0.92 if on else 0.6))
            ax.text(j + 0.5, rows - 0.5 - i, "测" if on else "—", color=BG if on else FAINT,
                    fontsize=12, ha="center", va="center", weight="bold")
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols) + 0.5, scenes, color=INK, fontsize=9.5)
    ax.set_yticks(np.arange(rows) + 0.5, metrics[::-1], color=INK, fontsize=9.5)
    ax.tick_params(labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.92),
                       plt.Rectangle((0, 0), 1, 1, color=PANEL2, alpha=0.6)],
              labels=["该场景必须报告此指标", "该场景通常不适用（或裁剪后报告）"],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False,
              labelcolor=INK, fontsize=8.8, handlelength=1.2)
    save(fig, "fig_eval_matrix.png")


# ============================================================
# 批 3：部署 / 系统 / LLM / 全双工 / 路线 7 张
# ============================================================

def fig_deployment_stack():
    """从训练 checkpoint 到设备上的可测量延迟。"""
    fig = new_fig(
        "从 checkpoint 到设备：每一层都要测",
        "导出、量化、运行时、服务与设备各有自己的失败方式；“能跑”不等于“能实时跑”",
        (10.8, 5.4),
        "精度、内存、峰值延迟与吞吐一起验收；固定：输入、线程、批大小、版本、热身次数",
        note_color=TEAL,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.7)
    flow_chain(ax, [
        ("训练权重", PURPLE, ["框架专用格式", "输出：checkpoint", "失败：算子不兼容"]),
        ("导出 ONNX", BLUE, ["交换 / 审计边界", "输出：计算图", "失败：动态轴/版本"]),
        ("量化 / 剪枝", ORANGE, ["x ≈ s·(q − z)", "输出：INT8 模型", "失败：校准集代表性"]),
        ("运行时", GREEN, ["ONNX RT / TensorRT", "输出：engine", "失败：绑定 GPU/驱动"]),
        ("服务管线", TEAL, ["Triton 批处理/调度", "输出：推理服务", "失败：状态/背压未设计"]),
        ("设备", CYAN, ["ARM / DSP / NPU", "输出：端侧结果", "失败：内存/线程/首帧编译"]),
    ], y=1.35, box_h=1.95, name_size=10.5, sub_size=7.8)
    note_box(ax, 0.25, 0.38, "静态量化需要代表性校准数据：校准集没覆盖\n远场、口音、长尾词 → 只在校准分布上正常",
             color=ORANGE, fontsize=8.8)
    note_box(ax, 6.9, 0.38, "同一 INT8 模型在不同内核/指令集下速度可能完全不同\n（CMSIS-NN 针对 SIMD/DSP/MVE 提供不同实现）",
             color=BLUE, fontsize=8.8)
    save(fig, "fig_deployment_stack.png")


def fig_edge_budget():
    """轻量化：同时满足误差、内存和时延边界（全部归一化到 FP32=100）。"""
    x = np.arange(4)
    labels = ["FP32\n基线", "INT8\n量化", "剪枝 + INT8", "专用算子"]
    mem = [100, 52, 34, 25]
    lat = [100, 72, 54, 38]
    err = [0, 3, 7, 12]      # 相对基线 CER 增幅 %
    w = 0.26
    fig = new_fig(
        "端侧优化：内存、延迟、误差一起看",
        "把 FP32 基线归一化为 100；误差一列用相对基线 CER 增幅（%）表示",
        (10.4, 5.2),
        "目标是三者同时落在预算边界内；数字必须绑定：校准集、目标硬件、输入形状与热身次数",
        note_color=ORANGE,
    )
    ax = fig.add_axes([0.055, 0.135, 0.92, 0.66])
    ax.set_facecolor(PANEL)
    ax.bar(x - w, mem, w, color=BLUE, label="内存（相对 %）")
    ax.bar(x, lat, w, color=ORANGE, label="延迟（相对 %）")
    ax.bar(x + w, err, w, color=RED, label="CER 增幅（相对 %）")
    ax.axhline(90, color=YELLOW, ls="--", lw=1.2)
    ax.text(3.5, 92, "预算边界 90%", color=YELLOW, fontsize=8.6, ha="right")
    for xi, (m, l, e) in enumerate(zip(mem, lat, err)):
        for xo, v in [(xi - w, m), (xi, l), (xi + w, e)]:
            ax.text(xo, v + 2, f"{v:.0f}", color=INK, fontsize=8, ha="center")
    ax.set_xticks(x, labels, color=INK, fontsize=9)
    ax.set_ylim(0, 118)
    style(ax, "方案", "相对基线 100", grid=True)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9, loc="upper right")
    save(fig, "fig_edge_budget.png")


def fig09_architecture():
    """可回放的语音系统架构：证据仓库 + 版本链。"""
    fig = new_fig(
        "可回放的语音系统架构",
        "原始音频只追加（证据仓库）；每个模块输出都是带来源的版本",
        (10.6, 5.4),
        "每个结果都能回到：哪段音频、哪版配置、哪次模型；调试 = 回放同一段输入比较两个版本",
        note_color=GREEN,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 5.0)
    layers = [
        ("采集 / 时间戳", GREEN, "样本序号 + 采样率 + 通道"),
        ("VAD / DOA / 增强", BLUE, "中间信号 + 参数版本"),
        ("ASR / diarization", PURPLE, "文本 + 时间范围 + 模型版本"),
        ("事件 / 纠错 / UI", ORANGE, "修订链 + 状态"),
    ]
    for i, (lab, c, meta) in enumerate(layers):
        y = 4.45 - i * 0.92
        box(ax, 0.5, y - 0.32, 4.0, 0.72, "", c, lw=1.8)
        ax.text(2.5, y + 0.06, lab, color=c, fontsize=10.5, weight="bold", ha="center", va="center", zorder=4)
        ax.text(2.5, y - 0.42, meta, color=MUTED, fontsize=7.8, ha="center", va="center", zorder=4)
        if i < 3:
            arrow(ax, (2.5, y - 0.48), (2.5, y - 0.82), MUTED, 1.4)
    # 右侧：证据仓库 + 版本链
    box(ax, 6.0, 2.35, 1.85, 1.1, "原始音频\n只追加", GREEN, lw=2.0, fontsize=10)
    box(ax, 8.35, 2.35, 1.85, 1.1, "版本链\n来源+配置+模型", PURPLE, lw=2.0, fontsize=9)
    arrow(ax, (5.35, 3.9), (6.9, 3.35), GREEN, 1.5)
    arrow(ax, (7.95, 3.35), (8.4, 3.9), PURPLE, 1.5)
    note_box(ax, 5.95, 1.55, "字符串可以被修订，\n音频时间和来源不能靠猜。\n事件必须带：时间范围、样本序号、\n模块名、模型版本、配置摘要、状态",
             color=YELLOW, fontsize=8.4)
    note_box(ax, 0.5, 0.50, "云端 / 端侧混合：明确哪些状态留在设备、\n哪些音频可上传；回退是产品行为（超时→上一版 partial）",
             color=TEAL, fontsize=8.6)
    save(fig, "fig09_architecture.png")


def fig10_version_chain():
    """版本链：每个结果保留来源、配置、模型与可回放音频时间段。"""
    fig = new_fig(
        "版本链：每个结果都有可追溯的来源",
        "原始音频 → 处理版本 → 模型版本 → 纠错版本；每层都指向同一段可回放音频",
        (10.8, 5.2),
        "来源、配置、模型、时间段和可回放证据一起交付",
        note_color=PURPLE,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.0)
    nodes = [
        ("原始音频", GREEN, "证据仓库\n样本 + 时间戳"),
        ("增强 v1", TEAL, "配置：AEC/ANS 参数"),
        ("ASR v3", BLUE, "模型 + 解码配置"),
        ("说话人 v2", PURPLE, "阈值 + 模型版本"),
        ("纠错 v1", ORANGE, "词表 + 提示词版本"),
        ("摘要 v1", PINK, "LLM 版本 + prompt"),
    ]
    for i, (lab, c, meta) in enumerate(nodes):
        x = 0.25 + i * 1.72
        box(ax, x, 2.5, 1.42, 0.9, lab, c, lw=1.8, fontsize=9.5)
        ax.text(x + 0.71, 2.16, meta, color=MUTED, fontsize=7.6, ha="center", va="center", zorder=4,
                linespacing=1.3)
        if i < 5:
            arrow(ax, (x + 1.48, 2.95), (x + 1.66, 2.95), MUTED, 1.5)
    # 公共音频时间轴
    ax.annotate("", xy=(8.55, 4.62), xytext=(0.25, 4.62),
                arrowprops=dict(arrowstyle="<->", color=YELLOW, lw=1.5))
    ax.text(4.4, 4.78, "同一段音频时间范围（00:00:00.00 – 00:00:12.30）", color=YELLOW,
            fontsize=8.8, ha="center")
    for i in range(6):
        x = 0.25 + i * 1.72 + 0.71
        ax.plot([x, x], [4.62, 3.42], color=YELLOW, ls=":", lw=0.9, alpha=0.8)
    note_box(ax, 0.25, 0.75, "回放 = 取同一段输入 + 同一版配置 → 比较两个版本的事件；\n而不是“再跑一次看看”",
             color=GREEN, fontsize=8.8)
    note_box(ax, 6.9, 0.75, "改动要能回答：输入变了吗？权重变了吗？\n解码约束变了吗？部署运行时变了吗？",
             color=BLUE, fontsize=8.8)
    save(fig, "fig10_version_chain.png")


def fig_llm_guardrails():
    """ASR + LLM：只能在证据边界内改写（可审计护栏）。"""
    fig = new_fig(
        "ASR + LLM：只能在证据边界内改写",
        "LLM 拥有语言证据，没有新的声学证据——可以提出候选，不能把流畅当成听清",
        (10.8, 5.8),
        "评测 = CER/WER + 专名/数字准确率 + 保护字段改坏率 + 无证据新增率 + 时间戳保留率",
        note_color=PURPLE,
    )
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.2)
    steps = [
        ("ASR 原文 + 时间戳", BLUE, ["输入：音频 → 文本", "输出：词序列 + 置信度"]),
        ("候选证据", GREEN, ["n-best / token 置信度", "允许使用的领域词表"]),
        ("约束规则", PURPLE, ["数字 / 专名 / 人名 / 金额", "地址 / 否定词 = 保护字段"]),
        ("可审计输出", ORANGE, ["修改区间 + 修改类型", "所用证据 + 不确定标记"]),
    ]
    for i, (lab, c, lines) in enumerate(steps):
        x = 0.35 + i * 2.55
        box(ax, x, 3.4, 2.0, 1.15, lab, c, lw=1.8, fontsize=10)
        ax.text(x + 1.0, 2.98, lines[0], color=MUTED, fontsize=7.8, ha="center", zorder=4)
        ax.text(x + 1.0, 2.68, lines[1], color=MUTED, fontsize=7.8, ha="center", zorder=4)
        if i < 3:
            arrow(ax, (x + 2.08, 3.98), (x + 2.4, 3.98), MUTED, 1.5)
    # 示例：成功与失败
    note_box(ax, 0.35, 1.85, "低置信度“十五万” → 无证据的新数字“十四万”：\n句子更顺了，但数字可能被改错 → 保留原文 / 转人工",
             color=RED, fontsize=8.8)
    note_box(ax, 6.0, 1.85, "每个保留/替换的 token 指回原 ASR 时间范围；\n纯新增内容标“无直接声学对齐”",
             color=GREEN, fontsize=8.8)
    note_box(ax, 0.35, 0.55, "输出不能只有新文本：还要有修改区间、类型、证据、\n不确定标记，以及 ASR / 纠错 / prompt / 词表版本",
             color=YELLOW, fontsize=8.6)
    save(fig, "fig_llm_guardrails.png")


def fig_duplex_streams():
    """播放流、采集流与打断状态在同一时间轴。"""
    fig = new_fig(
        "全双工：播放、采集、判断共享一条时间轴",
        "播报流必须作为 AEC 远端参考；AEC 之后的 VAD/KWS 才决定是否出现真实打断候选",
        (10.6, 5.8),
        "验收：打断响应 P50/P95、每小时误打断 / 漏打断、播报残留、双讲损伤、取消后时间轴连续",
        note_color=TEAL,
    )
    ax = fig.add_axes([0.07, 0.12, 0.90, 0.70])
    ax.set_facecolor(PANEL)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.7)
    ax.set_yticks([4.05, 2.95, 1.85, 0.75])
    ax.set_yticklabels(["TTS 播放", "麦克风采集", "AEC/VAD/KWS", "系统状态"], color=MUTED, fontsize=8.6)
    ax.broken_barh([(0.6, 1.9), (2.6, 1.1), (3.8, 0.6), (4.5, 4.1)], (3.85, 0.42), facecolors=ORANGE)
    ax.text(1.55, 4.42, "播放会话", color=ORANGE, fontsize=8.2, ha="center")
    ax.broken_barh([(0.6, 8.0)], (2.75, 0.42), facecolors=BLUE, alpha=0.75)
    ax.text(1.2, 3.32, "持续监听", color=BLUE, fontsize=8.2, ha="center")
    ax.broken_barh([(0.6, 2.2), (2.8, 0.5), (3.3, 0.5), (3.8, 0.5), (4.3, 0.6)], (1.65, 0.42),
                   facecolors=[GREEN, FAINT, YELLOW, FAINT, YELLOW])
    ax.text(1.7, 2.22, "VAD：用户语音", color=GREEN, fontsize=8.2, ha="center")
    ax.text(4.6, 2.22, "打断候选（语义确认）", color=YELLOW, fontsize=8.2, ha="center")
    states = [("LISTENING", 0.6, 2.6, GREEN), ("PLAYING", 2.6, 1.6, ORANGE),
              ("INTERRUPT_PENDING", 4.2, 1.4, YELLOW), ("LISTENING_AFTER_CANCEL", 5.6, 3.4, TEAL)]
    for lab, x, w, c in states:
        ax.broken_barh([(x, w)], (0.54, 0.42), facecolors=c, alpha=0.9)
        ax.text(x + w / 2, 1.12, lab, color=c, fontsize=8.2, ha="center")
    ax.axvline(4.2, color=YELLOW, ls="--", lw=1.1)
    note_box(ax, 6.9, 4.05, "风扇噪声让 VAD 触发 ≠ 应取消播报；\n用户清晰说“停”也不该等完整句尾 → 分语义层判定",
             color=RED, fontsize=8.6)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    style(ax, "统一媒体时间（示意）", None, grid=True)
    save(fig, "fig_duplex_streams.png")


def fig_roadmap():
    """24 周学习路线：6 个阶段阶梯 + 最小产物 + 通过条件。"""
    stages = [
        ("1–4 周", "信号基础", "可检查 WAV/特征的脚本", "能解释每个时间/频率参数", GREEN),
        ("5–8 周", "ASR + 指标", "离线 ASR 基线 + 错误分析", "固定集可复现，错误拆成 S/D/I", BLUE),
        ("9–12 周", "流式 revision", "带时间戳的事件协议", "首字 / 句尾 P95 可测", PURPLE),
        ("13–16 周", "阵列 / 说话人", "可控回声/噪声/阵列实验", "中间信号和边界条件齐全", ORANGE),
        ("17–20 周", "压缩与部署", "压缩前后对照表", "目标设备内存和时延达标", TEAL),
        ("21–24 周", "完整作品", "技术方案 + 回放集 + 演示", "另一人能凭文档复现", CYAN),
    ]
    fig = new_fig(
        "学习路线：先做可回放的最小系统，再逐步加复杂度",
        "每个阶段都以一个可回放的最小系统为出口；固定输入、可回放和指标闭环不能省略",
        (10.8, 6.0),
        "每阶段留下三种证据：代码、实验、报告；再补失败样例与下一步计划",
        note_color=CYAN,
    )
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.2)
    # 阶梯：每级高度递增
    h0, dh = 0.55, 0.42
    for i, (weeks, name, prod, passc, c) in enumerate(stages):
        x = 0.25 + i * 1.76
        y = 1.05 + i * 0.38
        h = h0 + i * dh * 0.22
        box(ax, x, y, 1.55, h, "", c, lw=1.8)
        ax.text(x + 0.775, y + h - 0.30, name, color=c, fontsize=10.2, weight="bold",
                ha="center", va="center", zorder=4)
        ax.text(x + 0.775, y + h - 0.62, weeks, color=MUTED, fontsize=8, ha="center", zorder=4)
        ax.text(x + 0.775, y - 0.18, "产出：", color=FAINT, fontsize=7.4, ha="center", zorder=4)
        ax.text(x + 0.775, y - 0.42, prod, color=INK, fontsize=7.4, ha="center", zorder=4,
                wrap=True, linespacing=1.25)
        ax.text(x + 0.775, y - 0.78, "通过：", color=FAINT, fontsize=7.4, ha="center", zorder=4)
        ax.text(x + 0.775, y - 1.02, passc, color=INK, fontsize=7.4, ha="center", zorder=4,
                wrap=True, linespacing=1.25)
        if i < 5:
            arrow(ax, (x + 1.60, y + h / 2), (x + 1.74, y + h / 2), MUTED, 1.4)
    note_box(ax, 0.25, 0.12, "数学补给按需出现：遇到 RMS/卷积/概率/矩阵/傅里叶再补附录 F；\n不应延后的是记录输入、版本和指标的习惯",
             color=CYAN, fontsize=8.6)
    save(fig, "fig_roadmap.png")


def main():
    for f in [fig_role_map, fig03_features, fig02_wave_spectrum,
              fig02_stft_windows, fig_frontend_3a, fig_asr_families,
              fig_streaming_latency, fig_data_loop,
              fig04_ctc_collapse, fig08_array_geometry, fig08_beam_pattern,
              fig07_diarization_flow, fig_eval_matrix,
              fig_deployment_stack, fig_edge_budget, fig09_architecture,
              fig10_version_chain, fig_llm_guardrails, fig_duplex_streams,
              fig_roadmap]:
        f()
    print("done.")


if __name__ == "__main__":
    main()
