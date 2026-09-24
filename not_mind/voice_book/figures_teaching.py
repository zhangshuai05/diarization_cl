"""Teaching-first replacements for every figure in the voice textbook."""
from pathlib import Path
import math
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
})

BG = "#0B1220"
PANEL = "#111C2E"
INK = "#F8FAFC"
MUTED = "#A8B5C7"
GRID = "#334155"
BLUE = "#38BDF8"
CYAN = "#22D3EE"
GREEN = "#34D399"
YELLOW = "#FBBF24"
ORANGE = "#FB923C"
PINK = "#F472B6"
PURPLE = "#A78BFA"
RED = "#FB7185"


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def canvas(title, subtitle=None, size=(9.4, 4.8)):
    fig = plt.figure(figsize=size, facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_title(title, loc="left", color=INK, fontsize=15, weight="bold", pad=14)
    if subtitle:
        fig.text(.02, .925, subtitle, color=MUTED, fontsize=9.5)
    return fig, ax


def style(ax, xlab=None, ylab=None, grid=True):
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(GRID)
    if grid: ax.grid(color=GRID, alpha=.55, linewidth=.6)
    if xlab: ax.set_xlabel(xlab, color=MUTED)
    if ylab: ax.set_ylabel(ylab, color=MUTED)
    ax.title.set_color(INK)


def text(ax, x, y, s, color=INK, size=10, **kw):
    ax.text(x, y, s, color=color, fontsize=size, **kw)


def box(ax, x, y, w, h, label, color=BLUE, size=10, fill=PANEL):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.025,rounding_size=.04",
                       facecolor=fill, edgecolor=color, linewidth=1.8)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", color=INK, fontsize=size, weight="bold")
    return p


def arrow(ax, a, b, color=MUTED, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color))


def signal():
    p = ROOT / "generated" / "demo_mono.wav"
    try:
        with wave.open(str(p), "rb") as f:
            sr = f.getframerate(); n = f.getnframes()
            x = np.frombuffer(f.readframes(n), dtype=np.int16).astype(float) / 32768
        return sr, x
    except Exception:
        sr = 16000; t = np.arange(sr*2)/sr
        return sr, .2*np.sin(2*np.pi*220*t) + .08*np.sin(2*np.pi*440*t)*(t>.55)


def stft(x, sr, win_ms=25, hop_ms=10):
    n = max(8, round(sr*win_ms/1000)); hop = max(1, round(sr*hop_ms/1000))
    nfft = 1 << int(math.ceil(math.log2(n))); w = np.hanning(n)
    frames = [np.abs(np.fft.rfft(x[i:i+n]*w, nfft))**2 for i in range(0, max(1, len(x)-n), hop)]
    z = np.array(frames).T if frames else np.zeros((nfft//2+1, 1))
    return z, np.arange(z.shape[1])*hop/sr, np.fft.rfftfreq(nfft, 1/sr)


def fig01_core():
    fig, ax = canvas("语音系统先回答三个问题", "输入是什么？模型推断什么？如何证明它工作？", (9.2, 5.0)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    nodes=[("证据\n音频、标注、时间戳",1.2,3.4,GREEN),("推断\n模型和算法",5,3.4,BLUE),("验证\n指标、切片、回放",8.8,3.4,ORANGE)]
    for lab,x,y,c in nodes: box(ax,x-.95,y-.55,1.9,1.1,lab,c,10)
    arrow(ax,(2.2,3.4),(4.0,3.4),GREEN); arrow(ax,(6.0,3.4),(7.8,3.4),BLUE)
    text(ax,5,1.65,"最小闭环：测量  →  建模  →  约束  →  评测  →  回放",BLUE,13,ha="center",weight="bold")
    text(ax,5,.55,"没有输入边界和证据链，单个漂亮的指标没有工程意义。",MUTED,10,ha="center")
    save(fig,"fig01_core_formula.png")


def fig01_structure():
    fig, ax = canvas("一本教材的阅读路线", "每一章都从可观察的信号出发，再进入模型和部署", (9.3, 4.7)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5.8)
    steps=[("01\n声音与采样",GREEN),("02\n特征与 STFT",BLUE),("03\nASR 与对齐",PURPLE),("04\n说话人与阵列",ORANGE),("05\n系统与交付",CYAN)]
    for i,(lab,c) in enumerate(steps):
        x=.3+i*1.94; box(ax,x,2.3,1.55,1.1,lab,c,9)
        if i<4: arrow(ax,(x+1.55,2.85),(x+1.86,2.85),MUTED)
    for x,lab,c in [(.9,"能读懂波形",GREEN),(2.84,"能解释参数",BLUE),(4.78,"能定位错误",PURPLE),(6.72,"能分析边界",ORANGE),(8.66,"能复现结果",CYAN)]:
        text(ax,x,1.55,lab,c,8.8,ha="center")
    text(ax,5,.65,"先把现象画清楚，再谈模型为什么有效。",INK,12,ha="center",weight="bold")
    save(fig,"fig01_structure.png")


def fig02_wave_spectrum():
    sr,x=signal(); x=x[:min(len(x),sr*2)]; t=np.arange(len(x))/sr
    fig=plt.figure(figsize=(9.2,7.0),facecolor=BG); gs=fig.add_gridspec(3,1,hspace=.5)
    a=fig.add_subplot(gs[0]); a.set_facecolor(PANEL); a.plot(t,x,color=CYAN,lw=.55); style(a,"时间 (s)","振幅"); a.set_title("① 波形：声音什么时候变响",loc="left",color=INK,weight="bold")
    b=fig.add_subplot(gs[1]); b.set_facecolor(PANEL); nfft=1<<int(math.ceil(math.log2(len(x)))); f=np.fft.rfftfreq(nfft,1/sr); s=np.abs(np.fft.rfft(x*np.hanning(len(x)),nfft)); s/=max(s.max(),1e-9); b.plot(f,s,color=YELLOW); b.fill_between(f,s,0,color=YELLOW,alpha=.12); b.set_xlim(0,5000); b.set_ylim(0,1.05); style(b,"频率 (Hz)","相对能量"); b.set_title("② 整段 FFT：有哪些频率，但不知道何时出现",loc="left",color=INK,weight="bold")
    z,tt,ff=stft(x,sr); c=fig.add_subplot(gs[2]); c.set_facecolor(PANEL); c.pcolormesh(tt,ff,10*np.log10(z+1e-8),shading="auto",cmap="magma"); c.set_ylim(0,2000); style(c,"时间 (s)","频率 (Hz)"); c.set_title("③ STFT：把时间切成小格，保留变化过程",loc="left",color=INK,weight="bold")
    fig.text(.5,.02,"波形看强弱，FFT 看频率，STFT 同时看频率和时间。",color=INK,ha="center",fontsize=11,weight="bold")
    save(fig,"fig02_wave_spectrum.png")


def fig02_stft_windows():
    sr,x=signal(); x=x[:min(len(x),sr*2)]; frame_ms,hop_ms=25,10; n=round(sr*frame_ms/1000); hop=round(sr*hop_ms/1000)
    start=min(round(.62*sr),len(x)-n); y=x[start:start+n+hop*2]; y/=max(np.max(np.abs(y)),1e-9); tm=np.arange(len(y))/sr*1000
    fig,ax=canvas("STFT 第一步：沿着波形滑动窗口", None, (9.2,4.9)); ax.set_facecolor(PANEL); ax.plot(tm,y,color=INK,lw=.8); ax.axhline(0,color=GRID)
    starts=[0,10,20];
    for i,s in enumerate(starts):
        ax.add_patch(Rectangle((s,-1.1),frame_ms,2.2,facecolor=[BLUE,GREEN,BLUE][i],alpha=.18,edgecolor=[BLUE,GREEN,BLUE][i],lw=2)); text(ax,s+frame_ms/2,.88,f"第 {i+1} 帧",INK,10,ha="center",weight="bold")
    ax.annotate("帧移 10 ms",xy=(10,1.25),xytext=(0,1.25),arrowprops=dict(arrowstyle="<->",color=YELLOW,lw=1.7),ha="center",color=YELLOW,fontsize=10)
    ax.annotate("重叠 15 ms",xy=(25,-1.35),xytext=(10,-1.35),arrowprops=dict(arrowstyle="<->",color=CYAN,lw=1.7),ha="center",color=CYAN,fontsize=10)
    ax.set_xlim(0,52); ax.set_ylim(-1.7,1.55); style(ax,"时间 (ms)",None,False); ax.set_yticks([])
    fig.text(.5,.06,"25 ms = 10 ms + 15 ms",color=INK,ha="center",fontsize=13,weight="bold")
    fig.text(.5,.025,"16 kHz 下：400 点 = 160 点 + 240 点",color=MUTED,ha="center",fontsize=10)
    save(fig,"fig02_stft_windows.png")


def fig03_features():
    fig,ax=canvas("从波形到 Fbank / MFCC", "每一步都把输入变成更适合模型检查的表示", (9.3,4.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    labels=[("波形",GREEN),("分帧\n加窗",BLUE),("FFT",BLUE),("Mel\n滤波",PURPLE),("取 log",PURPLE),("Fbank",ORANGE),("DCT\nMFCC",ORANGE)]
    for i,(lab,c) in enumerate(labels):
        x=.2+i*1.4; box(ax,x,3.2,1.05,1.0,lab,c,9)
        if i<6: arrow(ax,(x+1.08,3.7),(x+1.33,3.7),MUTED)
    ax2=fig.add_axes([.12,.12,.76,.25]); ax2.set_facecolor(PANEL); f=np.linspace(0,8000,400)
    for center,w,c in [(250,280,GREEN),(700,450,BLUE),(1400,700,PURPLE),(2800,1300,ORANGE),(5200,2400,PINK)]:
        tri=np.maximum(1-np.abs(f-center)/w,0); ax2.plot(f,tri,color=c,lw=2)
    style(ax2,"频率 (Hz)","权重"); ax2.set_ylim(0,1.05); ax2.set_title("Mel 滤波器：低频更密，高频更宽",loc="left",color=INK,fontsize=10)
    save(fig,"fig03_features.png")


def fig04_ctc():
    fig,ax=canvas("CTC 为什么可以把重复和空白压掉", "先对齐每个时间帧，再折叠 blank 与连续重复", (9.2,4.4)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    vals=["blank","你","你","blank","好","好","blank"]
    for i,v in enumerate(vals):
        c=GRID if v=="blank" else BLUE if v=="你" else GREEN; box(ax,1+i*1.05,3.2,.82,.7,v,c,8)
    arrow(ax,(1,2.7),(8.2,2.7),MUTED)
    text(ax,4.6,2.35,"去掉 blank，再合并连续重复",YELLOW,11,ha="center",weight="bold")
    for i,v in enumerate(["你","好"]): box(ax,3.3+i*1.7,1.15,1.25,.75,v,[BLUE,GREEN][i],14)
    text(ax,5,.35,"路径可能很多，但它们最终可以折叠成同一个文本。",MUTED,10,ha="center")
    save(fig,"fig04_ctc_collapse.png")


def fig04_families():
    fig,ax=canvas("ASR 模型家族：差异在对齐和解码", "不要只背模型名字，先问它如何处理时间", (9.3,4.7)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5.8)
    rows=[("CTC","每帧独立打分 → blank 折叠",BLUE),("AED","整段编码 → 注意力逐字解码",GREEN),("RNN-T","声学时间 × 历史文本",PURPLE),("Paraformer","估计字数与边界 → 并行解码",ORANGE)]
    for i,(name,desc,c) in enumerate(rows):
        y=4.65-i*.95; text(ax,.55,y,name,c,11,weight="bold"); ax.plot([1.65,7.2],[y-.03,y-.03],color=GRID); text(ax,1.9,y+.2,desc,INK,9.5); 
        for j in range(4): ax.scatter(2.1+j*1.45,y-.25,s=45,color=c,alpha=.8)
    text(ax,5,.35,"工程选择取决于流式约束、历史依赖、数据量和设备预算。",MUTED,10,ha="center")
    save(fig,"fig04_alignment_families.png")


def fig05_cer():
    fig,ax=canvas("CER：把识别结果对齐到参考文本", "S = 替换，D = 删除，I = 插入", (9.2,3.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,4.6)
    ref=list("今天开会讨论"); hyp=list("今天开会讨")
    text(ax,.35,3.45,"参考",MUTED,11,weight="bold"); text(ax,.35,2.1,"识别",BLUE,11,weight="bold")
    for i,ch in enumerate(ref): text(ax,1.4+i*.75,3.45,ch,INK,14,ha="center"); ax.plot([1.4+i*.75,1.4+i*.75],[2.55,3.05],color=GRID)
    for i,ch in enumerate(hyp): text(ax,1.4+i*.75,2.1,ch,BLUE,14,ha="center")
    ax.add_patch(Rectangle((1.4+5*.75-.3,1.8),.6,1.9,fill=False,edgecolor=ORANGE,lw=2)); text(ax,5.1,1.1,"D：参考里有字，识别里没有",ORANGE,10,ha="center")
    text(ax,5,.35,"CER = (S + D + I) / 参考字符数",INK,12,ha="center",weight="bold")
    save(fig,"fig05_cer_errors.png")


def fig06_revision():
    fig,ax=canvas("流式识别：结果会更新，但时间范围不能乱", "partial → final → corrected", (9.2,4.0)); ax.set_facecolor(PANEL); ax.set_xlim(0,10); ax.set_ylim(0,4); style(ax,"音频时间 / revision",None)
    for y,lab,c,pts in [(3.0,"partial",BLUE,[1.0,2.8,4.6]),(2.0,"final",GREEN,[3.7,5.4,7.1]),(1.0,"corrected",ORANGE,[6.3,8.0,9.1])]:
        text(ax,.2,y,lab,c,9,weight="bold"); ax.plot(pts,[y]*len(pts),"o-",color=c); 
    ax.annotate("revision 3 覆盖 revision 2",(4.6,3),(2.6,3.65),arrowprops=dict(arrowstyle="->",color=PINK),color=PINK,fontsize=9)
    text(ax,5,.35,"可修订的是文本版本；说话人、时间戳和证据仍要保留。",MUTED,9.5,ha="center")
    save(fig,"fig06_revision_timeline.png")


def fig06_latency():
    fig,ax=canvas("端到端延迟预算：先定义用户看到什么", "同一系统至少有三条时间线", (9.2,3.8)); ax.set_facecolor(PANEL); ax.set_xlim(0,3.5); ax.set_yticks([.65,1.55,2.45]); ax.set_yticklabels(["首字 partial","ASR final","会后重算"],color=MUTED); style(ax,"秒",None)
    vals=[1.2,2.0,3.0]; cols=[BLUE,GREEN,ORANGE]
    for y,v,c in zip([.65,1.55,2.45],vals,cols): ax.barh(y,v,.45,color=c,alpha=.9); text(ax,v+.06,y,f"{v:.1f}s",INK,9,va="center")
    text(ax,1.75,.18,"数字是目标边界，不是未经实测的占比。",MUTED,9.5,ha="center")
    save(fig,"fig06_latency_budget.png")


def fig07_flow():
    fig,ax=canvas("说话人分段：先切窗，再聚类", "整段音频只生成一个 embedding 会混合不同说话人", (9.2,4.3)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    labels=[("VAD",GREEN),("固定长度\n子窗",BLUE),("每窗一个\nembedding",PURPLE),("聚类",ORANGE),("标签",CYAN)]
    for i,(lab,c) in enumerate(labels):
        x=.35+i*1.9; box(ax,x,3.25,1.35,.85,lab,c,9)
        if i<4: arrow(ax,(x+1.4,3.68),(x+1.8,3.68),MUTED)
    ax.plot([.55,2.2,3.85,5.5],[1.5]*4,"o-",color=GREEN); text(ax,3,1.85,"每个子窗分别聚类",GREEN,10,ha="center")
    ax.add_patch(Rectangle((.55,.35),4.6,.55,facecolor="#321A22",edgecolor=RED)); text(ax,2.85,.62,"错误：跨越说话人切换，只求一个向量",RED,9,ha="center")
    save(fig,"fig07_diarization_flow.png")


def fig07_der():
    fig,ax=canvas("DER：把时间轴上的三类错误分开", "漏检 + 误检 + 混淆", (9.2,3.4)); ax.set_facecolor(PANEL); ax.set_xlim(0,10); ax.set_ylim(0,3); ax.set_yticks([.65,1.55]); ax.set_yticklabels(["参考","预测"],color=MUTED); style(ax,"时间 (s)",None)
    ax.broken_barh([(0.5,2.1),(3.4,2.0),(6.3,2.7)],(.35,.45),facecolors=GREEN)
    ax.broken_barh([(0.5,1.4),(2.0,1.2),(3.4,1.3),(4.7,1.1),(6.3,1.4),(8.0,1.0)],(1.25,.45),facecolors=[GREEN,ORANGE,BLUE,PURPLE,GREEN,ORANGE])
    text(ax,2.3,2.4,"误检",ORANGE,9,ha="center"); text(ax,5.3,2.4,"混淆",PURPLE,9,ha="center"); text(ax,8.4,2.4,"漏检",ORANGE,9,ha="center")
    text(ax,5,.08,"DER = 漏检 + 误检 + 混淆 / 参考说话总时长",INK,10,ha="center",weight="bold")
    save(fig,"fig07_der_components.png")


def fig08_geometry():
    fig,ax=canvas("麦克风阵列：间距决定能观察到的空间变化", "同一声源到达不同麦克风的时间不同 → TDOA", (8.9,4.6)); ax.set_aspect("equal"); ax.set_xlim(-.6,7); ax.set_ylim(-2,4); style(ax,"x (m)","y (m)")
    xs=np.array([0,.05,.10,.15]); ax.scatter(xs,[0]*4,s=75,color=BLUE); 
    text(ax,.08,-.38,"4 个麦克风，间距 d = 5 cm",BLUE,8.5,ha="left")
    ax.plot([.075,5.7],[0,2.8],color=ORANGE,lw=2); text(ax,5.75,2.95,"声源方向",ORANGE,10)
    ax.annotate("d",(.05,.1),(.05,.6),arrowprops=dict(arrowstyle="<->",color=YELLOW),color=YELLOW,ha="center",fontsize=12)
    text(ax,2.0,-1.35,"相邻通道的到达时间差 = TDOA",CYAN,11,ha="center",weight="bold")
    save(fig,"fig08_array_geometry.png")


def fig08_beam():
    ang=np.linspace(-90,90,721); fig,ax=canvas("波束形成：导向角错了，峰值就会偏", "Delay-and-Sum 的方向响应示意", (9.2,4.0)); ax.set_facecolor(PANEL)
    def resp(steer):
        d=.05; f=1000; c=343; n=8; idx=np.arange(n)-(n-1)/2; tau=d*idx[:,None]*(np.sin(np.deg2rad(ang))[None,:]-np.sin(np.deg2rad(steer)))/c; a=np.abs(np.sum(np.exp(1j*2*np.pi*f*tau),axis=0)/n); return 20*np.log10(np.maximum(a,1e-4))
    ax.plot(ang,resp(20),color=GREEN,label="正确导向 20°"); ax.plot(ang,resp(-20),color=ORANGE,label="错误导向 -20°"); ax.axvline(20,color=GREEN,ls="--"); style(ax,"声源方向 (°)","响应 (dB)"); ax.set_ylim(-35,1); ax.legend(frameon=False,labelcolor=INK)
    save(fig,"fig08_beam_pattern.png")


def fig09_arch():
    fig,ax=canvas("可回放的语音系统架构", "音频只追加；处理结果带版本和时间范围", (9.3,4.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    layers=[("采集 / 时间戳",GREEN),("VAD / DOA / 增强",BLUE),("ASR / diarization",PURPLE),("事件 / 纠错 / UI",ORANGE)]
    for i,(lab,c) in enumerate(layers):
        y=4.7-i*.95; box(ax,3.2,y-.3,3.6,.6,lab,c,10)
        if i<3: arrow(ax,(5,y-.38),(5,y-.6),MUTED)
    box(ax,.55,2.2,1.8,.9,"会后\n全局对齐",ORANGE,9); box(ax,7.65,2.2,1.8,.9,"原始音频\n只追加",GREEN,9); arrow(ax,(2.4,2.65),(3.15,3.0),ORANGE); arrow(ax,(6.85,3.0),(7.6,2.65),GREEN)
    text(ax,5,.55,"每个结果都能回到：哪段音频、哪版配置、哪次模型。",MUTED,10,ha="center")
    save(fig,"fig09_architecture.png")


def fig09_timeline():
    fig,ax=canvas("音频顺序和网络到达顺序不是一回事", "合并事件时按样本序号和音频时间戳，不按消息抵达顺序", (9.2,3.9)); ax.set_facecolor(PANEL); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.set_yticks([.75,1.75,2.75]); ax.set_yticklabels(["音频时间","模块处理","网络到达"],color=MUTED); style(ax,"时间",None)
    ax.plot([.5,9.2],[.75,.75],color=BLUE,lw=3); ax.scatter([1.5,4,7.5],[.75]*3,color=BLUE,s=40)
    ax.broken_barh([(1.5,1.1),(4,.6),(7.5,.9)],(1.5,.45),facecolors=GREEN)
    ax.broken_barh([(2.8,.45),(3.5,.35),(8.1,.4)],(2.5,.45),facecolors=ORANGE)
    text(ax,5,3.45,"网络可能乱序到达",ORANGE,11,ha="center",weight="bold"); text(ax,5,.12,"排序键：stream_id + sample_index + audio_timestamp",MUTED,9.5,ha="center")
    save(fig,"fig09_event_timeline.png")


def fig10_chain():
    fig,ax=canvas("版本链：每个结果都保留来源", "原始音频 → 处理版本 → 模型版本 → 纠错版本", (9.3,3.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,3.4)
    labs=[("原始音频",BLUE),("增强 v1",GREEN),("ASR v3",PURPLE),("说话人 v2",ORANGE),("纠错 v1",CYAN),("摘要 v1",PINK)]
    for i,(lab,c) in enumerate(labs):
        x=.25+i*1.62; box(ax,x,1.2,1.25,.85,lab,c,9)
        if i<5: arrow(ax,(x+1.28,1.62),(x+1.55,1.62),MUTED)
    text(ax,5,.45,"来源、配置、模型、时间段和可回放证据一起交付。",MUTED,10,ha="center")
    save(fig,"fig10_version_chain.png")


def fig_cover_system():
    fig,ax=canvas("语音算法工程师教学", "从声音、模型到可交付系统", (8.8,5.4)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,6)
    box(ax,3.9,2.35,2.2,1.15,"可复现\n语音系统",BLUE,14)
    for x,y,lab,c in [(1.35,4.25,"音频\n证据",GREEN),(5,5.15,"模型\n推断",PURPLE),(8.65,4.25,"指标\n评测",ORANGE),(5,.9,"部署\n约束",CYAN)]:
        box(ax,x-.85,y-.42,1.7,.84,lab,c,10); arrow(ax,(x,y),(5,2.9),c)
    text(ax,5,.25,"先测量，再建模；先定义边界，再谈性能。",MUTED,10,ha="center")
    save(fig,"cover_system.png")


def simple_flow(name,title,items,colors,subtitle=""):
    fig,ax=canvas(title,subtitle,(9.3,4.2)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    step=9.0/len(items)
    for i,lab in enumerate(items):
        x=.35+i*step; box(ax,x,2.15,step-.25,.9,lab,colors[i%len(colors)],8.8)
        if i<len(items)-1: arrow(ax,(x+step-.2,2.6),(x+step-.05,2.6),MUTED)
    text(ax,5,.7,"每一步都要能说清：输入、输出、失败边界。",MUTED,10,ha="center")
    save(fig,name)


def fig_role_map():
    simple_flow("fig_role_map.png","五类岗位共享一条能力链",["信号与声学","识别与理解","数据与评测","模型与训练","工程与部署"],[GREEN,BLUE,PURPLE,ORANGE,CYAN],"岗位不同，证据和验收方式不同")
def fig_frontend_3a():
    simple_flow("fig_frontend_3a.png","3A 前端：回声、噪声、增益共同影响 VAD/KWS",["参考播放","麦克风采集","AEC","ANS","AGC","VAD / KWS"],[ORANGE,BLUE,GREEN,PURPLE,YELLOW,CYAN],"只看最终 CER，会丢掉定位线索")
def fig_asr_families():
    simple_flow("fig_asr_families.png","ASR 家族：对齐规则决定系统行为",["DNN-HMM","CTC","AED","RNN-T"],[GREEN,BLUE,PURPLE,ORANGE],"流式约束越强，历史依赖和解码策略越重要")
def fig_streaming_latency():
    simple_flow("fig_streaming_latency.png","流式系统至少有三条时间线",["采集","分帧","前端","模型","解码","稳定化"],[GREEN,BLUE,PURPLE,ORANGE,CYAN,PINK],"首字、最终结果、会后重算不是同一个延迟")
def fig_data_loop():
    simple_flow("fig_data_loop.png","评测不是终点，而是下一轮数据入口",["采集","清洗 / 切分","训练","固定评测","失败切片","回灌"],[GREEN,BLUE,PURPLE,ORANGE,RED,CYAN],"失败样例要能回到原始音频")
def fig_deployment_stack():
    simple_flow("fig_deployment_stack.png","从 checkpoint 到设备：每一层都要测",["训练权重","导出 ONNX","量化 / 剪枝","运行时","服务管线","设备"],[PURPLE,BLUE,ORANGE,GREEN,CYAN,PINK],"精度、内存、峰值延迟和吞吐一起验收")
def fig_speech_tasks():
    simple_flow("fig_speech_tasks.png","相邻任务共享前端，但输出和证据不同",["共享音频 / 特征","ASR","KWS","LID","说话人分段"],[BLUE,GREEN,ORANGE,PURPLE,CYAN],"任务边界写进标签、阈值和评测")
def fig_roadmap():
    simple_flow("fig_roadmap.png","学习路线：先做可回放的最小系统",["信号基础","ASR + 指标","流式 revision","阵列 / 说话人","部署","完整作品"],[GREEN,BLUE,PURPLE,ORANGE,CYAN,PINK],"每阶段留下代码、实验、报告和失败样例")


def fig_eval_matrix():
    fig,ax=canvas("评测矩阵：场景 × 指标", "单一总分不能替代分场景证据", (9.3,4.7));
    scenes=["近讲","远场","噪声","重叠","口音"]; metrics=["CER/WER","DER","SI-SDR","P95 延迟","内存"]
    vals=np.array([[1,1,1,1,1],[0,1,0,1,1],[0,1,1,1,0],[1,1,1,1,1],[1,1,1,0,1]])
    ax.imshow(vals,cmap="Blues",vmin=0,vmax=1,alpha=.85); ax.set_xticks(range(5),scenes,color=MUTED); ax.set_yticks(range(5),metrics,color=MUTED); ax.tick_params(labelsize=8); ax.grid(color=BG,lw=1); 
    for y in range(5):
        for x in range(5): text(ax,x,y,"测" if vals[y,x] else "-",INK,10,ha="center",va="center",weight="bold")
    style(ax); text(ax,2,-.9,"每个场景都要指定：输入条件、指标口径、时间边界。",MUTED,10,ha="center")
    save(fig,"fig_eval_matrix.png")


def fig_llm_guardrails():
    fig,ax=canvas("ASR + LLM：只能在证据边界内改写", "原文、候选证据、约束规则和差异都要可审计", (9.3,4.5)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    labels=[("ASR 原文\n+ 时间戳",BLUE),("候选证据\n置信度 / 词表",GREEN),("约束规则\n数字、专名、否定",PURPLE),("可审计输出\n差异 + 版本",ORANGE)]
    for i,(lab,c) in enumerate(labels): x=.35+i*2.4; box(ax,x,2.35,1.85,.95,lab,c,9); 
    for i in range(3): arrow(ax,(2.25+i*2.4,2.82),(2.65+i*2.4,2.82),MUTED)
    text(ax,5,1.2,"无证据的新内容：保留原文，转人工复核。",RED,11,ha="center",weight="bold")
    text(ax,5,.45,"评测：CER/WER + 专名保护率 + 无证据新增率",MUTED,9.5,ha="center")
    save(fig,"fig_llm_guardrails.png")


def fig_duplex_streams():
    fig,ax=canvas("全双工：播放、采集、判断共享一条时间轴", "VAD 只判断有没有声音；是否打断还要看 AEC、KWS 和播放状态", (9.3,4.5)); ax.set_facecolor(PANEL); ax.set_xlim(0,10); ax.set_ylim(0,5); ax.set_yticks([4.2,3.1,2.0,0.9]); ax.set_yticklabels(["TTS 播放","麦克风","AEC / VAD / KWS","状态"],color=MUTED); style(ax,"统一媒体时间",None)
    ax.broken_barh([(0.5,4.4),(6.2,2.5)],(3.85,.45),facecolors=ORANGE); ax.broken_barh([(0.5,2.2),(2.7,1.5),(4.2,1.2),(5.4,2.8)],(2.75,.45),facecolors=[GREEN,BLUE,PURPLE,GREEN]); ax.scatter([3.2,4.5],[2.2,2.2],color=[PURPLE,ORANGE],s=40); ax.broken_barh([(0.5,2.7),(3.2,1.3),(4.5,3.2)],(.55,.45),facecolors=[BLUE,ORANGE,GREEN]); text(ax,4.5,4.7,"打断发生在 4.5 s",YELLOW,9,ha="center")
    save(fig,"fig_duplex_streams.png")


def fig_edge_budget():
    fig,ax=canvas("端侧优化：内存、延迟、精度一起看", "把 FP32 基线归一化为 100", (9.3,4.0)); ax.set_facecolor(PANEL); x=np.arange(4); labels=["FP32","INT8","剪枝 + INT8","专用算子"]; mem=[100,52,34,25]; lat=[100,72,54,38]; w=.34
    ax.bar(x-w/2,mem,w,color=BLUE,label="内存"); ax.bar(x+w/2,lat,w,color=ORANGE,label="延迟"); ax.set_xticks(x,labels,color=MUTED); ax.set_ylim(0,115); style(ax,"方案","相对基线",False); ax.legend(frameon=False,labelcolor=INK)
    save(fig,"fig_edge_budget.png")


def main():
    funcs=[fig01_core,fig01_structure,fig02_wave_spectrum,fig02_stft_windows,fig03_features,fig04_ctc,fig04_families,fig05_cer,fig06_revision,fig06_latency,fig07_flow,fig07_der,fig08_geometry,fig08_beam,fig09_arch,fig09_timeline,fig10_chain,fig_cover_system,fig_role_map,fig_frontend_3a,fig_asr_families,fig_streaming_latency,fig_data_loop,fig_deployment_stack,fig_eval_matrix,fig_speech_tasks,fig_llm_guardrails,fig_duplex_streams,fig_edge_budget,fig_roadmap]
    for f in funcs: f()
    print(f"generated {len(funcs)} teaching figures in {OUT}")


if __name__ == "__main__": main()
