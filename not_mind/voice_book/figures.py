"""Create the reproducible figures used by the textbook.

The figures are intentionally small, reproducible matplotlib programs.  The
signal examples use the checked-in generated WAV when available and otherwise
fall back to a deterministic synthetic signal.
"""
from pathlib import Path
import math
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
COL = {"ink": "#1A1A1A", "soft": "#4A4A4A", "accent": "#1F5C8B", "green": "#1F7A5A", "orange": "#A8531B", "purple": "#6B3FA0", "pale": "#E8F0F6", "grid": "#D8D8D8"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=COL["grid"], alpha=.45, linewidth=.6)
    ax.tick_params(labelsize=8)


def box(ax, x, y, w, h, text, color, fontsize=11):
    p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02", facecolor="white", edgecolor=color, linewidth=1.8)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, color=COL["ink"], wrap=True)


def fig01_core():
    fig, ax = plt.subplots(figsize=(9.2, 5.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.text(5, 6.55, "可信的语音系统 = 音频 + 模型 + 评测", ha="center", fontsize=16, weight="bold", color=COL["ink"])
    for x, t, c in zip([.7, 3.65, 6.6], ["耳朵\n大脑\n尺子", "音频\n模型\n评测", "可复现输入\n可替换模块\n可归因指标"], [COL["green"], COL["accent"], COL["purple"]]):
        box(ax, x, 3.2, 2.5, 2.1, t, c)
    for a, b in [(3.2, 3.65), (6.15, 6.6)]: ax.annotate("", (b, 4.25), (a, 4.25), arrowprops={"arrowstyle": "->", "color": COL["soft"], "lw": 1.5})
    ax.text(5, 2.35, "同一个对象的三种表达层次", ha="center", fontsize=10, color=COL["soft"])
    ax.text(5, .85, "证据  →  解释  →  判断", ha="center", fontsize=13, color=COL["accent"], weight="bold")
    save(fig, "fig01_core_formula.png")


def fig01_structure():
    fig, ax = plt.subplots(figsize=(9.3, 5.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    parts = [("第一篇\n建立地基", ["1 声音", "2 表示", "3 特征"], .2, COL["green"]), ("第二篇\nASR 主线", ["4 对齐", "5 打分", "6 流式"], 2.7, COL["accent"]), ("第三篇\n说话人与\n空间", ["7 分段", "8 阵列"], 5.2, COL["purple"]), ("第四篇\n系统工程", ["9 事件", "10 纠错"], 7.7, COL["orange"])]
    for title, items, x, c in parts:
        box(ax, x, 3.5, 2.0, 1.7, title, c, 9.2)
        for i, item in enumerate(items):
            ax.text(x + 1.0, 2.8 - i*.45, item, ha="center", fontsize=9, color=COL["soft"])
    for x in [2.2, 4.7, 7.2]: ax.annotate("", (x + .35, 4.35), (x, 4.35), arrowprops={"arrowstyle": "->", "color": COL["soft"], "lw": 1.3})
    ax.text(5, .45, "每一篇都回到：输入是否可复现，模块是否可替换，指标是否可归因", ha="center", fontsize=10, color=COL["ink"])
    save(fig, "fig01_structure.png")


def load_signal():
    p = ROOT / "generated" / "demo_mono.wav"
    try:
        with wave.open(str(p), "rb") as f:
            sr = f.getframerate(); n = f.getnframes(); x = np.frombuffer(f.readframes(n), dtype=np.int16).astype(float) / 32768.0
        return sr, x
    except Exception:
        sr = 16000; t = np.arange(sr * 3) / sr
        return sr, .35*np.sin(2*np.pi*220*t) + .12*np.sin(2*np.pi*660*t) * (t > .9)


def stft(x, sr, win_ms):
    n = int(sr * win_ms / 1000); hop = max(1, n // 4); nfft = 1 << int(math.ceil(math.log2(n)))
    frames = []
    w = np.hanning(n)
    for i in range(0, max(1, len(x)-n), hop): frames.append(np.abs(np.fft.rfft(x[i:i+n] * w, nfft))**2)
    z = np.array(frames).T if frames else np.zeros((nfft//2+1, 1))
    return z, np.arange(z.shape[1]) * hop / sr, np.fft.rfftfreq(nfft, 1/sr)


def fig02_wave_spectrum():
    sr, x = load_signal(); x = x[:min(len(x), sr*2)]; t = np.arange(len(x))/sr
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 7.0),
                             gridspec_kw={"height_ratios": [1, 1, 1.35]},
                             constrained_layout=True)
    a, b, c = axes
    a.plot(t, x, color=COL["accent"], lw=.65)
    a.set_title("① 波形：看声音什么时候变响", loc="left", fontsize=13, weight="bold")
    a.text(.99, .88, "能看强弱\n不容易看清音高", transform=a.transAxes,
           ha="right", va="top", fontsize=10, color=COL["soft"],
           bbox={"boxstyle": "round,pad=.3", "fc": "#EAF0F7", "ec": COL["accent"]})
    a.set_xlabel("时间 (s)"); a.set_ylabel("振幅"); clean(a)

    # A whole-utterance FFT intentionally removes the time axis.  Showing it
    # between waveform and STFT makes the reason for framing visible.
    nfft = 1 << int(math.ceil(math.log2(len(x))))
    freq = np.fft.rfftfreq(nfft, 1 / sr)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    spectrum = spectrum / max(spectrum.max(), 1e-9)
    b.plot(freq, spectrum, color=COL["orange"], lw=1.0)
    b.fill_between(freq, spectrum, color=COL["orange"], alpha=.12)
    b.set_xlim(0, min(5000, sr / 2)); b.set_ylim(0, 1.05)
    b.set_title("② 整段 FFT：看得到“有什么”，看不到“什么时候”", loc="left", fontsize=13, weight="bold")
    b.text(.99, .88, "时间被压成\n一条频率曲线", transform=b.transAxes,
           ha="right", va="top", fontsize=10, color=COL["soft"],
           bbox={"boxstyle": "round,pad=.3", "fc": "#FBF0E7", "ec": COL["orange"]})
    b.set_xlabel("频率 (Hz)"); b.set_ylabel("相对能量"); clean(b)

    z, tt, ff = stft(x, sr, 25)
    c.pcolormesh(tt, ff, 10*np.log10(z + 1e-8), shading="auto", cmap="magma")
    # Speech energy in this teaching signal is concentrated below 2 kHz.
    # Showing that band at full height makes the time-varying harmonics legible.
    c.set_ylim(0, min(2000, sr / 2))
    c.set_title("③ STFT：把时间切成小格，保留变化过程", loc="left", fontsize=13, weight="bold")
    c.text(.99, .92, "横轴=时间  |  纵轴=频率\n越亮=这一刻能量越强", transform=c.transAxes,
           ha="right", va="top", fontsize=10, color="white",
           bbox={"boxstyle": "round,pad=.35", "fc": "#1A1A1A", "alpha": .72, "ec": "white"})
    c.set_xlabel("时间 (s)"); c.set_ylabel("频率 (Hz)"); clean(c)
    save(fig, "fig02_wave_spectrum.png")


def fig02_stft_windows():
    sr, x = load_signal()
    frame_ms, shift_ms = 25, 10
    overlap_ms = frame_ms - shift_ms
    frame_samples = round(sr * frame_ms / 1000)
    shift_samples = round(sr * shift_ms / 1000)
    overlap_samples = frame_samples - shift_samples

    # A short, energetic excerpt makes frame boundaries visible at textbook size.
    start_sample = min(round(.64 * sr), max(0, len(x) - round(.09 * sr)))
    excerpt = x[start_sample:start_sample + round(.09 * sr)]
    t_ms = np.arange(len(excerpt)) / sr * 1000
    if len(excerpt):
        excerpt = excerpt / max(np.max(np.abs(excerpt)), 1e-9)

    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor="#0B0D10")
    fig.subplots_adjust(left=.065, right=.76, top=.82, bottom=.27)
    ax.set_facecolor("#0B0D10")
    ax.plot(t_ms, excerpt, color="#F3F4F6", lw=.7, zorder=2)
    ax.axhline(0, color="#6B7280", lw=.6, alpha=.65)

    starts = [5, 15, 25]
    frame_colors = ["#E5E7EB", "#38A8DF", "#E5E7EB"]
    for idx, (start, color) in enumerate(zip(starts, frame_colors), 1):
        ax.add_patch(patches.Rectangle(
            (start, -1.08), frame_ms, 2.16,
            facecolor=color, alpha=.10 if idx != 2 else .20,
            edgecolor=color, linewidth=2.0, zorder=3))
        ax.text(start + frame_ms / 2, .88, f"第 {idx} 帧",
                ha="center", va="center", fontsize=10.5,
                color="#FFFFFF", weight="bold", zorder=5)

    # The overlap is the intersection of two adjacent frames.
    overlap_start = starts[1]
    ax.add_patch(patches.Rectangle(
        (overlap_start, -1.08), overlap_ms, 2.16,
        facecolor="#1D9BD1", alpha=.28, edgecolor="none", zorder=1))

    arrow = {"arrowstyle": "<->", "lw": 1.7, "shrinkA": 0, "shrinkB": 0}
    ax.annotate("", (starts[1], 1.22), (starts[0], 1.22),
                arrowprops={**arrow, "color": "#FFFFFF"})
    ax.text((starts[0] + starts[1]) / 2, 1.30, "帧移 10 ms",
            ha="center", va="bottom", color="#FFFFFF", fontsize=10)
    ax.annotate("", (starts[0] + frame_ms, -1.25), (starts[1], -1.25),
                arrowprops={**arrow, "color": "#38A8DF"})
    ax.text((starts[1] + starts[0] + frame_ms) / 2, -1.34, "重叠 15 ms",
            ha="center", va="top", color="#49BDF2", fontsize=10)
    ax.annotate("", (starts[1] + frame_ms, -1.56), (starts[1], -1.56),
                arrowprops={**arrow, "color": "#F4C95D"})
    ax.text(starts[1] + frame_ms / 2, -1.66, "帧长 25 ms",
            ha="center", va="top", color="#F4C95D", fontsize=10, weight="bold")

    ax.set_xlim(0, 72); ax.set_ylim(-1.95, 1.52)
    ax.set_xlabel("时间 (ms)", color="#D1D5DB", fontsize=10)
    ax.set_yticks([]); ax.tick_params(axis="x", colors="#D1D5DB", labelsize=9)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")

    fig.suptitle("STFT 第一步：沿着波形滑动窗口，逐帧做 FFT",
                 x=.41, y=.94, color="#FFFFFF", fontsize=15, weight="bold")
    fig.text(.785, .72, "每隔 10 ms\n产生一帧", color="#FFFFFF", fontsize=12, weight="bold")
    fig.text(.785, .52, "每帧观察\n25 ms 声音", color="#F4C95D", fontsize=12, weight="bold")
    fig.text(.785, .32, "相邻两帧\n共享 15 ms", color="#49BDF2", fontsize=12, weight="bold")
    fig.text(.50, .105,
             f"帧长 = 帧移 + 重叠 = {shift_ms} ms + {overlap_ms} ms = {frame_ms} ms",
             ha="center", color="#FFFFFF", fontsize=12.5, weight="bold")
    fig.text(.50, .045,
             f"16 kHz 下：{frame_samples} 点 = {shift_samples} 点 + {overlap_samples} 点",
             ha="center", color="#A7D8F0", fontsize=10.5)
    fig.savefig(OUT / "fig02_stft_windows.png", dpi=240, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def mel_filterbank(sr=16000, nfft=512, nmel=24):
    def hz2mel(f): return 2595*np.log10(1+f/700)
    def mel2hz(m): return 700*(10**(m/2595)-1)
    edges = mel2hz(np.linspace(hz2mel(80), hz2mel(7600), nmel+2)); bins = np.floor((nfft+1)*edges/sr).astype(int); freqs = np.fft.rfftfreq(nfft, 1/sr); filters=[]
    for i in range(1, nmel+1):
        f = np.zeros(len(freqs)); l, c, r = bins[i-1], bins[i], bins[i+1]
        if c>l: f[l:c] = (np.arange(l,c)-l)/max(1,c-l)
        if r>c: f[c:r] = (r-np.arange(c,r))/max(1,r-c)
        filters.append(f)
    return freqs, np.array(filters)


def fig03_features():
    fig = plt.figure(figsize=(9.4, 5.2)); gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5]); ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 1)
    labels = ["波形", "分帧\n加窗", "FFT", "Mel\n滤波器", "log\n能量", "Fbank", "DCT\n→ MFCC"]
    colors = [COL["green"], COL["accent"], COL["accent"], COL["purple"], COL["purple"], COL["orange"], COL["orange"]]
    for i, lab in enumerate(labels):
        x = .08 + i*1.42; box(ax, x, .25, 1.08, .46, lab, colors[i], 8.4)
        if i < len(labels)-1: ax.annotate("", (x+1.55, .48), (x+1.22, .48), arrowprops={"arrowstyle":"->", "color":COL["soft"]})
    freqs, filters = mel_filterbank(); bx = fig.add_subplot(gs[1]);
    for i in range(0, len(filters), 3): bx.plot(freqs, filters[i], lw=1, alpha=.72)
    bx.set_xlim(0, 8000); bx.set_ylim(0, 1.05); bx.set_xlabel("频率 (Hz)"); bx.set_ylabel("权重"); bx.set_title("Mel 三角滤波器组：低频更密，高频更宽", loc="left", fontsize=11); clean(bx)
    save(fig, "fig03_features.png")


def fig04_ctc():
    fig, ax = plt.subplots(figsize=(9.2, 3.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.text(.2, 3.55, "路径 A", fontsize=10, weight="bold"); ax.text(.2, 2.45, "路径 B", fontsize=10, weight="bold")
    for y, vals in [(3.3, ["空白","空白","你","你","空白","好","空白"]), (2.2, ["你","空白","空白","好","好","好","空白"])]:
        for i, v in enumerate(vals):
            fc = COL["pale"] if v == "空白" else "#EAF5F0" if v == "你" else "#FBF0E7"
            ax.add_patch(patches.Rectangle((1.0+i*1.0, y-.24), .82, .48, facecolor=fc, edgecolor=COL["grid"]))
            ax.text(1.41+i*1.0, y, v, ha="center", va="center", fontsize=12)
        ax.annotate("", (8.5, y), (8.1, y), arrowprops={"arrowstyle":"->", "color":COL["soft"]})
    ax.text(8.8, 2.75, "去重复\n去空白", ha="center", va="center", fontsize=10, color=COL["accent"])
    ax.annotate("", (9.5, 2.0), (9.5, 1.3), arrowprops={"arrowstyle":"->", "color":COL["soft"]})
    ax.text(9.5, .8, "你好", ha="center", fontsize=15, weight="bold", color=COL["accent"])
    ax.text(4.5, .3, "多条合法帧级路径可以坍缩到同一文本，CTC 对它们求概率和", ha="center", fontsize=9, color=COL["soft"])
    save(fig, "fig04_ctc_collapse.png")


def fig04_families():
    fig, ax = plt.subplots(figsize=(9.2, 4.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    rows = [("CTC", "帧 → blank/字 → 坍缩", COL["accent"]), ("AED", "整段编码 ← 注意力 → 逐字解码", COL["green"]), ("RNN-T", "帧 × 历史文本网格上单调前进", COL["purple"]), ("Paraformer", "CIF 估字数与边界 → 并行解码", COL["orange"])]
    for i, (name, desc, c) in enumerate(rows):
        y = 3.9 - i*.95; ax.text(.45, y, name, fontsize=11, weight="bold", color=c); ax.plot([1.5, 9.3], [y, y], color=COL["grid"], lw=1); ax.text(1.8, y+.12, desc, fontsize=10, color=COL["ink"])
        ax.scatter([2.0, 4.0, 6.0, 8.0], [y-.2]*4, s=45, color=c, alpha=.75)
    ax.text(5, .35, "比较维度：对齐约束、历史文本依赖、流式性、训练/解码代价", ha="center", fontsize=10, color=COL["soft"])
    save(fig, "fig04_alignment_families.png")


def fig05_cer():
    fig, ax = plt.subplots(figsize=(9.2, 2.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
    ref = list("今天开会讨论接口"); hyp = list("今天开会讨论接通")
    ax.text(.25, 2.45, "参考", fontsize=11, weight="bold"); ax.text(.25, 1.35, "识别", fontsize=11, weight="bold")
    for i, ch in enumerate(ref):
        ax.text(1.2+i*.55, 2.45, ch, ha="center", fontsize=13); ax.plot([1.2+i*.55, 1.2+i*.55], [1.7, 2.1], color=COL["grid"])
    for i, ch in enumerate(hyp): ax.text(1.2+i*.55, 1.35, ch, ha="center", fontsize=13, color=COL["accent"])
    ax.add_patch(patches.Rectangle((1.2+7*.55-.26, 1.05), .52, 1.7, facecolor="none", edgecolor=COL["orange"], linewidth=2)); ax.text(5, .45, "S = 替换，D = 删除，I = 插入；CER = (S + D + I) / N", ha="center", fontsize=10, color=COL["soft"])
    save(fig, "fig05_cer_errors.png")


def fig06_revision():
    fig, ax = plt.subplots(figsize=(9.2, 3.3)); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.set_yticks([1,2,3]); ax.set_yticklabels(["asr_partial", "asr_final", "text_final"]); ax.set_xlabel("音频时间 / revision"); clean(ax)
    ax.plot([0.5, 2.6, 4.5, 6.5], [1,1,1,1], "o-", color=COL["accent"], label="临时文本"); ax.plot([5.2, 7.0], [2,2], "o-", color=COL["green"], label="ASR 定稿"); ax.plot([7.6, 9.2], [3,3], "o-", color=COL["orange"], label="规则/LLM 新版本")
    ax.annotate("revision 3 覆盖 revision 2", (4.5,1), (2.6,1.8), arrowprops={"arrowstyle":"->", "color":COL["purple"]}, fontsize=9, color=COL["purple"]); ax.legend(loc="lower right", fontsize=8, frameon=False); ax.set_title("结果可以修订，身份和时间范围不能靠字符串猜", loc="left", fontsize=11); save(fig, "fig06_revision_timeline.png")


def fig06_latency():
    fig, ax = plt.subplots(figsize=(9.2, 3.2)); ax.set_xlim(0, 3.4); ax.set_ylim(0, 3); ax.set_yticks([.55,1.35,2.15]); ax.set_yticklabels(["首字 partial", "ASR final", "text final"]); ax.set_xlabel("秒（目标边界）"); clean(ax)
    ax.barh([.55,1.35,2.15], [1.5,2.0,3.0], height=.42, color=[COL["accent"], COL["green"], COL["orange"]], alpha=.82); ax.axvline(1.5, color=COL["accent"], ls="--", lw=1); ax.axvline(2.0, color=COL["green"], ls="--", lw=1); ax.axvline(3.0, color=COL["orange"], ls="--", lw=1); ax.set_title("端到端延迟预算（目标边界，不代表未经实测的占比）", loc="left", fontsize=11); save(fig, "fig06_latency_budget.png")


def fig07_flow():
    fig, ax = plt.subplots(figsize=(9.2, 4.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    for i, (x, t, c) in enumerate([(0.3,"VAD",COL["green"]),(2.1,"定长\n子窗",COL["accent"]),(4.0,"embedding",COL["purple"]),(6.0,"聚类",COL["orange"]),(8.0,"标签",COL["accent"])]):
        box(ax,x,3.1,1.45,0.9,t,c,9)
        if i<4: ax.annotate("", (x+1.9,3.55), (x+1.45,3.55), arrowprops={"arrowstyle":"->","color":COL["soft"]})
    ax.text(1.25,1.9,"正确：一个 VAD 段切成多个子窗",ha="center",fontsize=9,color=COL["green"]); ax.plot([.4,2.1,3.8,5.5],[1.45,1.45,1.45,1.45],"o-",color=COL["green"]); ax.text(7.1,1.45,"各子窗分别聚类",fontsize=9,color=COL["green"])
    ax.text(1.25,.65,"错误：整段只求一个向量，跨越切换会混合两个人",ha="center",fontsize=9,color=COL["orange"]); ax.add_patch(patches.Rectangle((.4,.1),5.1,.55,facecolor="#FBF0E7",edgecolor=COL["orange"])); ax.text(2.95,.42,"人 A  ─────  人 B",ha="center",fontsize=9,color=COL["orange"]); ax.text(6.0,.42,"一个混合 embedding",fontsize=9,color=COL["orange"]); save(fig,"fig07_diarization_flow.png")


def fig07_der():
    fig, ax = plt.subplots(figsize=(9.2, 3.0)); ax.set_xlim(0,10); ax.set_ylim(0,3); ax.set_yticks([.55,1.45,2.35]); ax.set_yticklabels(["参考", "预测", "错误类型"]); ax.set_xlabel("时间"); clean(ax)
    ax.broken_barh([(0.5,2.1),(3.3,2.2),(6.4,2.6)],(.25,.55),facecolors=COL["green"]); ax.broken_barh([(0.5,1.4),(2.1,1.3),(3.3,1.3),(4.6,1.2),(6.4,1.4),(8.1,.9)],(1.15,.55),facecolors=[COL["green"],COL["orange"],COL["accent"],COL["purple"],COL["green"],COL["orange"]]); ax.text(1.2,2.35,"漏检",color=COL["orange"],fontsize=9); ax.text(3.5,2.35,"误检",color=COL["orange"],fontsize=9); ax.text(6.4,2.35,"混淆",color=COL["purple"],fontsize=9); ax.text(5,.05,"DER = 漏检 + 误检 + 混淆 / 参考说话总时长",ha="center",fontsize=9,color=COL["soft"]); save(fig,"fig07_der_components.png")


def fig08_geometry():
    fig, ax = plt.subplots(figsize=(8.8,4.5)); ax.set_aspect("equal"); ax.set_xlim(-1,8); ax.set_ylim(-2.5,3.4); ax.set_xlabel("阵列坐标 x (m)"); ax.set_ylabel("y (m)"); clean(ax)
    xs=np.array([0,.05,.1,.15]); ax.scatter(xs,np.zeros(4),s=55,color=COL["accent"],label="麦克风");
    ax.plot([xs[0],xs[-1]],[0,0],color=COL["soft"],lw=1); ax.annotate("d",(.05,.12),(.05,.38),ha="center",color=COL["soft"]); ax.plot([.075,4.8],[0,1.7],color=COL["orange"],lw=1.5); ax.plot([.075,4.8],[0,0],color=COL["grid"],lw=1); ax.text(4.65,1.9,"声源方向",color=COL["orange"],fontsize=10); ax.annotate("θ",(.7,.1),(.9,.5),arrowprops={"arrowstyle":"->","color":COL["purple"]},color=COL["purple"],fontsize=12); ax.text(1.2,-1.1,"不同通道的到达时间差 = TDOA",fontsize=10,color=COL["accent"]); ax.legend(frameon=False,fontsize=8); save(fig,"fig08_array_geometry.png")


def fig08_beam():
    angles=np.linspace(-90,90,721); d=.05; f=1000; c=343; n=8; idx=np.arange(n)-(n-1)/2
    def resp(steer):
        tau=d*idx[:, None]*(np.sin(np.deg2rad(angles))[None, :]-np.sin(np.deg2rad(steer)))/c
        a=np.abs(np.sum(np.exp(1j*2*np.pi*f*tau),axis=0)/n); return 20*np.log10(np.maximum(a,1e-4))
    fig, ax=plt.subplots(figsize=(9.2,3.8)); ax.plot(angles,resp(20),color=COL["green"],label="正确 steering = 20°"); ax.plot(angles,resp(-20),color=COL["orange"],label="错误 steering = -20°"); ax.axvline(20,color=COL["green"],ls="--",lw=1); ax.set_ylim(-35,1); ax.set_xlabel("声源方向 (°)"); ax.set_ylabel("归一化响应 (dB)"); ax.set_title("Delay-and-Sum 方向响应：导向误差会把峰值移走",loc="left",fontsize=11); clean(ax); ax.legend(frameon=False,fontsize=8); save(fig,"fig08_beam_pattern.png")


def fig09_arch():
    fig, ax=plt.subplots(figsize=(9.4,4.9)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    rows=[("同步多通道音频",.85,COL["accent"]),("采集 / 时间戳",2.1,COL["green"]),("VAD / DOA / 增强",3.35,COL["purple"]),("ASR + diarization",4.6,COL["orange"]),("事件 / 纠错 / UI",5.85,COL["accent"])]
    for t,y,c in rows: box(ax,3.0,y-.32,4.0,.64,t,c,10)
    for y1,y2 in zip([1.15,2.4,3.65,4.9],[1.75,3.0,4.25,5.5]): ax.annotate("",(5,y2-.32),(5,y1+.32),arrowprops={"arrowstyle":"->","color":COL["soft"]})
    ax.text(8.3,3.2,"原始音频\n（只追加）",ha="center",va="center",bbox={"boxstyle":"round,pad=.35","facecolor":"#FAFBFC","edgecolor":COL["grid"]},fontsize=9); ax.annotate("",(7.05,3.35),(7.8,3.35),arrowprops={"arrowstyle":"->","color":COL["soft"]}); ax.text(1.25,3.2,"会后重处理\n全局对齐",ha="center",va="center",bbox={"boxstyle":"round,pad=.35","facecolor":"#FAFBFC","edgecolor":COL["grid"]},fontsize=9); ax.annotate("",(2.95,3.35),(1.9,3.35),arrowprops={"arrowstyle":"->","color":COL["soft"]}); save(fig,"fig09_architecture.png")


def fig09_timeline():
    fig, ax=plt.subplots(figsize=(9.2,3.4)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.set_yticks([.7,1.7,2.7]); ax.set_yticklabels(["音频时间", "模块处理", "网络到达"]); ax.set_xlabel("时间"); clean(ax)
    ax.plot([.5,8.8],[.7,.7],color=COL["accent"],lw=3); ax.scatter([1.2,3.6,6.8],[.7,.7,.7],color=COL["accent"],s=35)
    ax.broken_barh([(1.2,1.1),(3.6,.55),(6.8,.8)],(1.45,.5),facecolors=COL["green"]); ax.broken_barh([(2.0,.35),(3.0,.3),(7.3,.35)],(2.45,.5),facecolors=COL["orange"])
    ax.text(5,3.45,"网络到达顺序可能与音频发生顺序不同",ha="center",fontsize=10,color=COL["orange"]); ax.text(5,.12,"按样本序号和音频时间戳合并，不能按消息抵达顺序拼接",ha="center",fontsize=9,color=COL["soft"]); save(fig,"fig09_event_timeline.png")


def fig10_chain():
    fig, ax=plt.subplots(figsize=(9.3,2.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3)
    labels=["原始音频","音频处理\n版本","ASR\n版本","说话人\n版本","文本纠错\n版本","纪要\n版本"]; cols=[COL["accent"],COL["green"],COL["purple"],COL["orange"],COL["accent"],COL["green"]]
    for i,(t,c) in enumerate(zip(labels,cols)):
        x=.15+i*1.62; box(ax,x,.95,1.25,.95,t,c,9)
        if i<5: ax.annotate("",(x+1.55,1.42),(x+1.25,1.42),arrowprops={"arrowstyle":"->","color":COL["soft"]})
    ax.text(5,.35,"每个版本保留来源、配置、模型和可回放的音频时间段",ha="center",fontsize=10,color=COL["soft"]); save(fig,"fig10_version_chain.png")


def fig_cover_system():
    """Minimal cover mark: evidence, inference, constraints and measurement."""
    # The cover already carries the book title; keep this diagram title-free
    # so it does not compete with or overlap the cover typography.
    fig, ax = plt.subplots(figsize=(8.8, 5.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    cx, cy = 5, 2.8
    circ = patches.Circle((cx, cy), .82, facecolor="#F4F6F8", edgecolor=COL["accent"], linewidth=2)
    ax.add_patch(circ); ax.text(cx, cy + .12, "语音", ha="center", va="center", fontsize=15, weight="bold", color=COL["accent"])
    ax.text(cx, cy - .23, "工程系统", ha="center", va="center", fontsize=9, color=COL["soft"])
    nodes = [(1.35, 4.0, "麦克风\n音频证据", COL["green"]), (5, 1.0, "模型\n推断", COL["accent"]), (8.65, 4.0, "端云约束\n算力与时延", COL["orange"]), (5, 5.05, "评测\n可复现", COL["purple"])]
    for x, y, label, color in nodes:
        p = patches.FancyBboxPatch((x - 1.0, y - .42), 2.0, .84, boxstyle="round,pad=.02,rounding_size=.03", facecolor="white", edgecolor=color, linewidth=1.8)
        ax.add_patch(p); ax.text(x, y, label, ha="center", va="center", fontsize=10, color=COL["ink"])
        ax.annotate("", (cx + (x-cx)*.42, cy + (y-cy)*.42), (x - (x-cx)*.32, y - (y-cy)*.32), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    ax.text(5, .25, "先测量，再建模；先定义边界，再谈指标", ha="center", fontsize=10, color=COL["soft"])
    save(fig, "cover_system.png")


def fig_role_map():
    fig, ax = plt.subplots(figsize=(9.5, 5.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    ax.text(5, 6.65, "五类岗位，其实共享一条能力链", ha="center", fontsize=14, weight="bold", color=COL["ink"])
    cols = [("信号与声学", ["采样 / STFT", "AEC / ANS / AGC", "阵列 / DOA"], COL["green"]),
            ("识别与理解", ["ASR / KWS", "LID / diarization", "TTS / 音频检测"], COL["accent"]),
            ("数据与评测", ["清洗 / 标注", "CER / DER / SI-SDR", "噪声与口音切片"], COL["purple"]),
            ("模型与训练", ["CTC / AED / RNN-T", "自监督 / 适配", "蒸馏 / 剪枝 / 量化"], COL["orange"]),
            ("工程与落地", ["Python + C/C++", "ONNX / TensorRT", "ARM / DSP / NPU"], COL["accent"])]
    xs = [.15, 2.15, 4.15, 6.15, 8.15]
    for x, (title, items, color) in zip(xs, cols):
        p = patches.FancyBboxPatch((x, 3.0), 1.7, 2.55, boxstyle="round,pad=.03,rounding_size=.03", facecolor="white", edgecolor=color, linewidth=1.7)
        ax.add_patch(p); ax.text(x+.85, 5.2, title, ha="center", va="center", fontsize=10, weight="bold", color=color)
        for j, item in enumerate(items): ax.text(x+.85, 4.45-j*.55, item, ha="center", va="center", fontsize=8.6, color=COL["ink"])
        if x < xs[-1]: ax.annotate("", (x+1.98, 4.28), (x+1.73, 4.28), arrowprops={"arrowstyle": "->", "color": COL["soft"], "lw": 1.2})
    ax.text(5, 1.95, "岗位不同，验收方式不同；但都要回答同一问题：输入变坏时，系统如何知道自己变坏了？", ha="center", fontsize=9.5, color=COL["soft"])
    ax.text(5, 1.18, "测量  →  建模  →  约束  →  评测  →  回放", ha="center", fontsize=12, weight="bold", color=COL["accent"])
    save(fig, "fig_role_map.png")


def fig_frontend_3a():
    fig, ax = plt.subplots(figsize=(9.4, 4.8)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.62, "3A 不是三个孤立按钮，而是一条双工信号路径", ha="center", fontsize=13, weight="bold", color=COL["ink"])
    boxes = [("远端播放\n参考信号", .35, 3.95, 1.45, .8, COL["orange"]), ("麦克风\n近端信号", .35, 2.0, 1.45, .8, COL["accent"]), ("AEC\n估计并消除回声", 2.35, 2.95, 1.62, .95, COL["green"]), ("ANS\n抑制残余噪声", 4.42, 2.95, 1.62, .95, COL["purple"]), ("AGC\n控制响度", 6.49, 2.95, 1.32, .95, COL["orange"]), ("VAD\n判断是否有语音", 8.18, 2.95, 1.25, .95, COL["accent"]), ("KWS\n检测目标词", 8.18, 1.62, 1.25, .78, COL["purple"])]
    for label, x, y, w, h, c in boxes:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.02,rounding_size=.03", facecolor="white", edgecolor=c, linewidth=1.7); ax.add_patch(p)
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.8, color=COL["ink"])
    ax.annotate("", (2.5, 3.42), (1.92, 4.25), arrowprops={"arrowstyle":"->", "color":COL["orange"], "lw":1.4})
    ax.annotate("", (2.5, 3.2), (1.92, 2.4), arrowprops={"arrowstyle":"->", "color":COL["accent"], "lw":1.4})
    for x in [4.02, 6.09, 7.81]: ax.annotate("", (x+.40, 3.42), (x, 3.42), arrowprops={"arrowstyle":"->", "color":COL["soft"], "lw":1.3})
    ax.annotate("", (8.80, 2.40), (8.80, 2.95), arrowprops={"arrowstyle":"->", "color":COL["purple"], "lw":1.2})
    ax.text(1.1, .85, "回声路径变化", ha="center", fontsize=9, color=COL["orange"]); ax.text(3.0, .85, "残余误差", ha="center", fontsize=9, color=COL["purple"]); ax.text(5.0, .85, "增益与削波", ha="center", fontsize=9, color=COL["orange"]); ax.text(7.25, .85, "VAD 判停", ha="center", fontsize=9, color=COL["accent"]); ax.text(8.82, .85, "KWS 唤醒/打断", ha="center", fontsize=8.7, color=COL["purple"])
    ax.plot([.5, 9.65], [1.22, 1.22], color=COL["grid"], lw=.8)
    ax.text(5, .25, "每一级都要记录输入、输出、延迟和失败切片；只看最终 CER 会丢掉定位线索", ha="center", fontsize=9.3, color=COL["soft"])
    save(fig, "fig_frontend_3a.png")


def fig_asr_families():
    fig, ax = plt.subplots(figsize=(9.4, 5.1)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)
    ax.text(5, 5.8, "ASR 模型家族：对齐准则与编码器是两条轴", ha="center", fontsize=13, weight="bold", color=COL["ink"])
    rows = [("DNN-HMM", "帧 → 状态 → WFST", "词典和语言模型显式存在", COL["green"]), ("CTC", "帧 → blank/字 → 坍缩", "条件独立，解码简单", COL["accent"]), ("AED", "整段编码 → 注意力解码", "全局文本历史，非天然流式", COL["purple"]), ("RNN-T", "声学时间 × 标签历史", "单调输出，适合流式", COL["orange"])]
    for i, (name, path, trade, c) in enumerate(rows):
        y = 4.85 - i*.82
        ax.text(.55, y, name, fontsize=10.2, weight="bold", color=c)
        ax.plot([1.85, 7.35], [y-.04, y-.04], color=COL["grid"], lw=1)
        ax.scatter([2.15, 3.85, 5.55, 7.05], [y-.04]*4, s=32, color=c, alpha=.82)
        ax.text(2.15, y+.21, path, fontsize=8.8, color=COL["ink"])
        ax.text(9.65, y-.04, trade, ha="right", va="center", fontsize=8.4, color=COL["soft"])
    p = patches.FancyBboxPatch((1.35, .58), 7.3, .72, boxstyle="round,pad=.03,rounding_size=.03", facecolor="#F6F7F8", edgecolor=COL["accent"], linewidth=1.4)
    ax.add_patch(p)
    ax.text(5, .94, "Conformer 是编码器架构：可与 CTC、AED 或 RNN-T 等训练/解码准则组合", ha="center", va="center", fontsize=9.2, color=COL["accent"], weight="bold")
    ax.text(5, .27, "没有“最先进”的抽象答案：场景、延迟、数据和设备共同决定模型选择", ha="center", fontsize=9.3, color=COL["soft"])
    save(fig, "fig_asr_families.png")


def fig_streaming_latency():
    fig, ax = plt.subplots(figsize=(9.4, 4.0)); ax.set_xlim(0, 5.8); ax.set_ylim(0, 4); ax.set_yticks([.75, 1.8, 2.85]); ax.set_yticklabels(["首字 partial", "ASR final", "会后定稿"]); ax.set_xlabel("从语音发生到用户看到结果的秒数"); clean(ax)
    vals = [(0.0, 0.35, COL["green"], "采集"), (0.35, .22, COL["accent"], "分帧"), (.57, .55, COL["purple"], "前端"), (1.12, .78, COL["orange"], "模型"), (1.9, .32, COL["accent"], "解码")]
    for start, width, c, label in vals: ax.broken_barh([(start, width)], (.52, .42), facecolors=c); ax.text(start+width/2, .73, label, ha="center", va="center", fontsize=8, color="white")
    vals2 = [(0.0, .35, COL["green"], "采集"), (.35, .22, COL["accent"], "分帧"), (.57, .55, COL["purple"], "前端"), (1.12, 1.28, COL["orange"], "模型 + 稳定化")]
    for start, width, c, label in vals2: ax.broken_barh([(start, width)], (1.58, .42), facecolors=c); ax.text(start+width/2, 1.79, label, ha="center", va="center", fontsize=8, color="white")
    vals3 = [(0.0, 2.4, COL["green"], "可回放音频"), (2.4, 1.05, COL["purple"], "全局重算"), (3.45, 1.35, COL["orange"], "纠错 / 纪要")]
    for start, width, c, label in vals3: ax.broken_barh([(start, width)], (2.63, .42), facecolors=c); ax.text(start+width/2, 2.84, label, ha="center", va="center", fontsize=8, color="white")
    ax.axvline(2.22, color=COL["ink"], ls="--", lw=.9); ax.text(2.22, 3.58, "边界要先定义\n再测量", ha="center", fontsize=9, color=COL["ink"])
    ax.set_title("同一系统至少有三条时间线：首字、定稿、会后重算（示意值）", loc="left", fontsize=11)
    save(fig, "fig_streaming_latency.png")


def fig_data_loop():
    fig, ax = plt.subplots(figsize=(9.4, 4.7)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    labels = [("采集与许可", 1.0, 3.9, COL["green"]), ("清洗 / 切分", 3.0, 3.9, COL["accent"]), ("训练 / 微调", 5.0, 3.9, COL["purple"]), ("固定评测", 7.0, 3.9, COL["orange"]), ("失败切片", 5.0, 1.35, COL["orange"]), ("重采样 / 增强", 2.5, 1.35, COL["green"])]
    for label,x,y,c in labels:
        p=patches.FancyBboxPatch((x-.8,y-.38),1.6,.76,boxstyle="round,pad=.02,rounding_size=.03",facecolor="white",edgecolor=c,linewidth=1.6); ax.add_patch(p); ax.text(x,y,label,ha="center",va="center",fontsize=9,color=COL["ink"])
    for a,b in [((1.8,3.9),(2.2,3.9)),((3.8,3.9),(4.2,3.9)),((5.8,3.9),(6.2,3.9)),((7.8,3.9),(8.7,3.9)),((7.0,3.52),(5.5,1.74)),((4.3,1.35),(3.3,1.35)),((2.0,1.74),(1.55,3.52))]: ax.annotate("",b,a,arrowprops={"arrowstyle":"->","color":COL["soft"],"lw":1.2})
    ax.text(5, .35, "评测不是终点：它产生下一轮最值得补的数据和最该查的模块", ha="center", fontsize=10, color=COL["soft"])
    save(fig, "fig_data_loop.png")


def fig_deployment_stack():
    fig, ax = plt.subplots(figsize=(9.3, 5.1)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2)
    ax.text(5, 5.8, "从训练 checkpoint 到设备上的可测量延迟", ha="center", fontsize=13, weight="bold", color=COL["ink"])
    layers=[("PyTorch / TensorFlow checkpoint", COL["purple"]),("导出：ONNX / 算子审计", COL["accent"]),("压缩：蒸馏 · 剪枝 · INT8", COL["orange"]),("运行时：ONNX Runtime / TensorRT", COL["green"]),("服务：Triton / C++ pipeline", COL["accent"]),("设备：GPU · ARM · DSP · NPU", COL["green"])]
    for i,(label,c) in enumerate(layers):
        y=4.85-i*.72; w=5.8-(i%2)*.55; x=(10-w)/2; p=patches.FancyBboxPatch((x,y-.25),w,.5,boxstyle="round,pad=.02,rounding_size=.03",facecolor="white",edgecolor=c,linewidth=1.6); ax.add_patch(p); ax.text(5,y,label,ha="center",va="center",fontsize=9.1,color=COL["ink"])
        if i<len(layers)-1: ax.annotate("",(5,y-.46),(5,y-.28),arrowprops={"arrowstyle":"->","color":COL["soft"],"lw":1.1})
    ax.text(1.1, 1.0, "每层都要测：准确率\n内存 · 峰值延迟\n吞吐 · 功耗", ha="center", va="center", fontsize=9, color=COL["soft"])
    ax.text(8.9, 1.0, "能跑 ≠ 可交付\n先固定输入与版本\n再比较硬件", ha="center", va="center", fontsize=9, color=COL["soft"])
    save(fig, "fig_deployment_stack.png")


def fig_eval_matrix():
    fig, ax = plt.subplots(figsize=(9.4, 4.9)); fig.subplots_adjust(bottom=.21); ax.set_xlim(-.35, 4.35); ax.set_ylim(-.35, 4.35); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels(["近讲", "远场", "噪声", "重叠", "口音 / 方言"]); ax.set_yticklabels(["CER/WER", "DER", "SI-SDR", "P95 延迟", "功耗 / 内存"]); ax.grid(True, color=COL["grid"], linewidth=.8); ax.tick_params(labelsize=8); ax.set_title("评测矩阵：场景切片 × 产品指标（示意覆盖）", loc="left", fontsize=11)
    vals = np.array([[1,1,1,1,1],[0,1,0,1,1],[0,1,1,1,0],[1,1,1,1,1],[1,1,1,0,1]])
    for y in range(5):
        for x in range(5):
            if vals[y,x]:
                c=[COL["green"],COL["accent"],COL["purple"],COL["orange"]][(x+y)%4]
                ax.add_patch(patches.Circle((x,y),.18,color=c,alpha=.85)); ax.text(x,y,"测",ha="center",va="center",fontsize=7,color="white",weight="bold")
    fig.text(.5, .035, "同一个总分不能替代分场景、分模块、分时间边界的证据", ha="center", fontsize=9.5, color=COL["soft"])
    save(fig, "fig_eval_matrix.png")


def fig_speech_tasks():
    fig, ax = plt.subplots(figsize=(9.3, 4.5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4)
    ax.text(5, 5.02, "相邻任务共享前端，但输出和证据不同", ha="center", fontsize=13, weight="bold", color=COL["ink"])
    center=(5,2.45); ax.add_patch(patches.Circle(center,.72,facecolor="#F4F6F8",edgecolor=COL["accent"],linewidth=1.8)); ax.text(5,2.45,"共享音频\n特征与时间轴",ha="center",va="center",fontsize=9.2,color=COL["ink"])
    nodes=[("ASR\n说了什么",1.25,3.85,COL["accent"]),("KWS\n是否出现词",3.25,1.0,COL["green"]),("LID\n哪种语言",6.75,1.0,COL["purple"]),("分段 / 说话人\n谁在何时说",8.75,3.85,COL["orange"])]
    for label,x,y,c in nodes:
        p=patches.FancyBboxPatch((x-.85,y-.4),1.7,.8,boxstyle="round,pad=.02,rounding_size=.03",facecolor="white",edgecolor=c,linewidth=1.6); ax.add_patch(p); ax.text(x,y,label,ha="center",va="center",fontsize=8.8,color=COL["ink"]); ax.annotate("",(5+(x-5)*.5,2.45+(y-2.45)*.5),(x-(x-5)*.28,y-(y-2.45)*.28),arrowprops={"arrowstyle":"->","color":c,"lw":1.2})
    ax.text(5, .25, "把任务边界写进标签、阈值、回退策略和评测，而不是只换一个模型名", ha="center", fontsize=9.3, color=COL["soft"])
    save(fig, "fig_speech_tasks.png")


def fig_llm_guardrails():
    """Show an auditable, evidence-constrained ASR correction interface."""
    fig, ax = plt.subplots(figsize=(9.4, 4.7)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.text(5, 5.62, "ASR + LLM：校对员只能在证据边界内改写", ha="center", fontsize=13, weight="bold", color=COL["ink"])
    items = [
        ("ASR 原始转写\nraw_text + 时间戳", .35, 2.85, 1.72, .95, COL["accent"]),
        ("候选证据\nn-best / 置信度\n领域词表", .35, 1.35, 1.72, 1.05, COL["green"]),
        ("受约束纠错\n保护数字、否定、专名\n只允许可解释替换", 3.08, 2.3, 2.35, 1.45, COL["purple"]),
        ("可审计输出\ncorrected_text\n差异 + 版本 + 对齐", 6.65, 2.85, 2.3, 1.2, COL["orange"]),
    ]
    for label, x, y, w, h, c in items:
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03,rounding_size=.03", facecolor="white", edgecolor=c, linewidth=1.7)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9.0, color=COL["ink"])
    ax.annotate("", (3.0, 3.05), (2.12, 3.30), arrowprops={"arrowstyle":"->", "color":COL["soft"], "lw":1.3})
    ax.annotate("", (3.0, 2.75), (2.12, 1.85), arrowprops={"arrowstyle":"->", "color":COL["green"], "lw":1.3})
    ax.annotate("", (6.55, 3.15), (5.55, 3.15), arrowprops={"arrowstyle":"->", "color":COL["soft"], "lw":1.3})
    p = patches.FancyBboxPatch((6.35, .95), 2.95, .9, boxstyle="round,pad=.03,rounding_size=.03", facecolor="#FBF0E7", edgecolor=COL["orange"], linewidth=1.4)
    ax.add_patch(p); ax.text(7.82, 1.40, "无证据新增\n保留原文 / 人工复核", ha="center", va="center", fontsize=8.7, color=COL["orange"], weight="bold")
    ax.annotate("", (7.82, 2.0), (7.82, 2.82), arrowprops={"arrowstyle":"->", "color":COL["orange"], "lw":1.1})
    ax.text(5, .34, "评测：CER/WER + 专名/数字保护率 + 无证据新增率 + 时间戳保留率", ha="center", fontsize=9.4, color=COL["soft"])
    save(fig, "fig_llm_guardrails.png")


def fig_duplex_streams():
    """Show playback, capture and interruption decisions on one time axis."""
    fig, ax = plt.subplots(figsize=(9.4, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4)
    ax.set_yticks([4.45, 3.35, 2.25, 1.15]); ax.set_yticklabels(["TTS 播放下行", "麦克风上行", "AEC / VAD / KWS", "播放状态"])
    ax.set_xlabel("统一媒体时间（示意）"); clean(ax)
    ax.set_title("播放流、采集流与打断状态必须共享一条时间轴", loc="left", fontsize=11)
    ax.broken_barh([(0.5, 4.7), (6.4, 2.3)], (4.1, .46), facecolors=COL["orange"])
    ax.text(2.85, 4.33, "TTS 播报", ha="center", va="center", fontsize=8.5, color="white")
    ax.text(7.55, 4.33, "取消后续播报", ha="center", va="center", fontsize=8.4, color="white")
    ax.broken_barh([(0.5, 2.1), (2.6, 1.7), (4.3, 1.25), (5.55, 2.8)], (3.0, .46), facecolors=[COL["green"], COL["accent"], COL["purple"], COL["green"]])
    for x, label in [(1.55, "静音"), (3.45, "用户说话"), (4.93, "停止"), (6.95, "继续监听")]:
        ax.text(x, 3.23, label, ha="center", va="center", fontsize=8.2, color="white")
    ax.plot([.5, 8.35], [2.47, 2.47], color=COL["grid"], lw=1.0)
    ax.scatter([3.1, 4.5], [2.47, 2.47], s=42, color=[COL["purple"], COL["orange"]], zorder=3)
    ax.text(3.1, 2.68, "VAD 触发", ha="center", fontsize=8.3, color=COL["purple"])
    ax.text(4.5, 2.68, "KWS/语义确认", ha="center", fontsize=8.3, color=COL["orange"])
    states = [(0.5, 2.6, COL["accent"], "PLAYING"), (3.1, 1.4, COL["orange"], "INTERRUPT_PENDING"), (4.5, 3.0, COL["green"], "LISTENING_AFTER_CANCEL")]
    for start, width, color, label in states:
        ax.broken_barh([(start, width)], (.92, .46), facecolors=color)
        ax.text(start + width/2, 1.15, label, ha="center", va="center", fontsize=7.8, color="white")
    ax.axvline(4.5, color=COL["ink"], ls="--", lw=.9)
    ax.text(4.5, 4.96, "实际取消时刻", ha="center", fontsize=8.8, color=COL["ink"])
    ax.text(5, .25, "VAD 只回答“有无语音”；是否打断还要结合 AEC 残差、KWS/语义和播放状态", ha="center", fontsize=9.2, color=COL["soft"])
    save(fig, "fig_duplex_streams.png")


def fig_edge_budget():
    fig, ax = plt.subplots(figsize=(9.3, 3.8)); x=np.arange(4); labels=["FP32 基线","INT8","蒸馏 + INT8","端侧专用算子"]; mem=[100,52,34,25]; lat=[100,72,54,38];
    w=.34; ax.bar(x-w/2,mem,w,label="相对内存",color=COL["accent"]); ax.bar(x+w/2,lat,w,label="相对延迟",color=COL["orange"]); ax.set_xticks(x,labels); ax.set_ylim(0,115); ax.set_ylabel("基线 = 100（示意值）"); ax.set_title("轻量化的目标是同时满足误差、内存和时延边界",loc="left",fontsize=11); clean(ax); ax.legend(frameon=False,fontsize=8,ncol=2,loc="upper right")
    for xi, v in zip(x-w/2,mem): ax.text(xi,v+3,f"{v}",ha="center",fontsize=8)
    for xi, v in zip(x+w/2,lat): ax.text(xi,v+3,f"{v}",ha="center",fontsize=8)
    save(fig,"fig_edge_budget.png")


def fig_roadmap():
    fig, ax = plt.subplots(figsize=(9.4, 4.8)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6)
    ax.text(6,5.55,"先做可回放的最小系统，再逐步增加模型复杂度",ha="center",fontsize=13,weight="bold",color=COL["ink"])
    phases=[("0-4 周\n信号地基",COL["green"]),("5-8 周\nASR 与评测",COL["accent"]),("9-12 周\n流式与说话人",COL["purple"]),("13-16 周\n3A 与阵列",COL["orange"]),("17-20 周\n端云部署",COL["accent"]),("21-24 周\n完整作品集",COL["green"])]
    for i,(label,c) in enumerate(phases):
        x=.35+i*1.93; p=patches.FancyBboxPatch((x,3.1),1.55,1.05,boxstyle="round,pad=.02,rounding_size=.03",facecolor="white",edgecolor=c,linewidth=1.7); ax.add_patch(p); ax.text(x+.775,3.63,label,ha="center",va="center",fontsize=8.9,color=COL["ink"]); 
        if i<5: ax.annotate("",(x+1.84,3.63),(x+1.58,3.63),arrowprops={"arrowstyle":"->","color":COL["soft"],"lw":1.2})
    ax.text(6,2.25,"每阶段留下三种证据",ha="center",fontsize=10,color=COL["soft"])
    for i,(label,c) in enumerate([("可运行代码",COL["green"]),("可复现实验",COL["accent"]),("可解释报告",COL["purple"])]):
        x=1.0+i*3.7; ax.add_patch(patches.FancyBboxPatch((x,1.0),3.0,.65,boxstyle="round,pad=.02,rounding_size=.03",facecolor="#FAFBFC",edgecolor=c,linewidth=1.2)); ax.text(x+1.5,1.33,label,ha="center",va="center",fontsize=9,color=COL["ink"])
    ax.text(6,.32,"作品集不靠“跑过模型”证明，而靠输入、指标、失败案例和部署边界证明",ha="center",fontsize=9.3,color=COL["soft"])
    save(fig,"fig_roadmap.png")


def main():
    fig01_core(); fig01_structure(); fig02_wave_spectrum(); fig02_stft_windows(); fig03_features(); fig04_ctc(); fig04_families(); fig05_cer(); fig06_revision(); fig06_latency(); fig07_flow(); fig07_der(); fig08_geometry(); fig08_beam(); fig09_arch(); fig09_timeline(); fig10_chain()
    fig_cover_system(); fig_role_map(); fig_frontend_3a(); fig_asr_families(); fig_streaming_latency(); fig_data_loop(); fig_deployment_stack(); fig_eval_matrix(); fig_speech_tasks(); fig_llm_guardrails(); fig_duplex_streams(); fig_edge_budget(); fig_roadmap()
    print(f"generated 30 figures in {OUT}")


if __name__ == "__main__": main()
