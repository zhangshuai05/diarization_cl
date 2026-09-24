# -*- coding: utf-8 -*-
"""
第 3 章 lab：可控回声与噪声
======================
生成近端语音 x（input/near_end.wav，TTS 合成），叠加"回声"（延迟+衰减的参考信号副本），
再叠加可控噪声；单变量扫描（回声延迟 / 回声增益 / 噪声类型），记录：
    波形(SVG) / 输入SNR / VAD段数(能量法+FunASR fsmn) / ASR输出(paraformer-large)
教学提醒：AEC 诊断数字(SNR/ERLE 等) ≠ 识别质量，别拿前端指标直接当 ASR 质量。

用法（在 shiyan/ 下，Windows python3.12，FunASR 环境）:
    python aec_lab.py             # 全矩阵 9 格
    python aec_lab.py --quick     # 快速子集(4格)
    python aec_lab.py --skip-asr  # 不加载 FunASR，只出前端指标
"""
import argparse
import csv
import os
import sys
import time
import wave

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SR = 16000
RNG = np.random.default_rng(42)          # 固定种子，可复现
OUT_DIR = os.path.join(BASE, "results")


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------- 音频 IO（stdlib wave，16bit PCM） ----------------

def load_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, "需要 16k wav"
        assert w.getsampwidth() == 2
        n = w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:                      # 取单声道
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        return x


def save_wav(path, x):
    y = np.clip(x, -1.0, 1.0)                         # 写盘前防过载削顶（文档注明口径）
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


# ---------------- 信号构造 ----------------

def add_echo(x, delay_ms, gain):
    """回声 = 近端副本延迟 delay_ms 后按 gain 衰减；返回 近端+回声。"""
    n = int(round(delay_ms / 1000.0 * SR))
    y = np.zeros_like(x)
    if n == 0:
        y = x.copy()
    else:
        y[:n] = x[:n]
        y[n:] = x[: len(x) - n] * gain + x[n:]
    return y


def make_noise(kind, n):
    if kind == "white":
        return RNG.standard_normal(n).astype(np.float32)
    if kind == "pink":
        # 白噪声频谱乘 1/sqrt(f) 的 FFT 成形 -> 粉噪
        x = RNG.standard_normal(n).astype(np.float32)
        X = np.fft.rfft(x)
        f = np.arange(1, len(X) + 1)
        X = X / np.sqrt(np.maximum(f, 1.0))
        return np.fft.irfft(X, n).astype(np.float32)
    raise ValueError(kind)


def snr_db(clean, dist):
    """SNR = 10log10(P_clean / P_disturbance)，全程口径（含静音，文档注明）。
    无干扰(纯近端)返回 "inf"。"""
    pc = float(np.mean(clean ** 2))
    pd = float(np.mean(dist ** 2))
    if pd < 1e-12:
        return float("inf")
    return 10.0 * np.log10(pc / pd)


# ---------------- 能量法 VAD（透明、可控的教学口径） ----------------

def energy_vad_segments(x, thr_db=-30.0, frame_ms=20, hop_ms=10,
                        max_gap_ms=250, min_len_ms=120):
    """帧对数能量阈值 + 静音间隙合并 + 短段剔除。返回 [(s_sec, e_sec), ...]"""
    fl, hl = int(frame_ms * SR / 1000), int(hop_ms * SR / 1000)
    n_frames = 1 + (len(x) - fl) // hl
    if n_frames <= 0:
        return []
    db = np.empty(n_frames)
    for i in range(n_frames):
        seg = x[i * hl: i * hl + fl]
        db[i] = 10.0 * np.log10(float(np.mean(seg ** 2)) + 1e-12)
    voiced = db > thr_db
    segs, start = [], None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append((start * hl, i * hl))
            start = None
    if start is not None:
        segs.append((start * hl, len(x)))
    merged = []
    for s, e in segs:
        if merged and (s - merged[-1][1]) < max_gap_ms * SR / 1000:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    min_len = min_len_ms * SR / 1000
    return [(s / SR, e / SR) for s, e in merged if (e - s) >= min_len]


# ---------------- FunASR（fsmn VAD + paraformer ASR + 标点） ----------------

_chain = None


def get_chain():
    global _chain
    if _chain is None:
        from funasr import AutoModel
        _chain = AutoModel(
            model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            disable_update=True,
        )
    return _chain


def transcribe(path):
    """返回 (asr_text, fsmn_vad_seg_count)。段数口径 = 解码输出的时间戳段数。"""
    res = get_chain().generate(input=path)
    r = res[0]
    ts = r.get("timestamp")
    return r.get("text", ""), len(ts) if ts else 1


# ---------------- 波形 SVG（免 matplotlib 依赖） ----------------

def wave_svg(x, y, path, title):
    """两轨波形：上=近端(灰) 下=混音(红)。min/max 包络降采样。"""
    n = min(len(x), len(y))
    step = max(1, n // 1400)
    n2 = n - n % step
    env = lambda s: (np.min(s[:n2].reshape(-1, step), axis=1),
                     np.max(s[:n2].reshape(-1, step), axis=1))
    W, H, PAD = 1500, 180, 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
             f'<rect width="{W}" height="{H}" fill="#fff"/>',
             f'<text x="{PAD}" y="16" font-size="13">{title}</text>']
    for sig, color, y0 in ((x, "#999999", 50), (y, "#cc3333", 140)):
        lo, hi = env(sig)
        scale = 42.0 / (float(np.max(np.abs(sig))) + 1e-9)
        pts = []
        for i in range(len(lo)):
            xp = PAD + i * (W - 2 * PAD) / len(lo)
            pts.append(f"{xp:.0f},{y0 - lo[i] * scale:.1f}")
        for i in reversed(range(len(hi))):
            xp = PAD + i * (W - 2 * PAD) / len(hi)
            pts.append(f"{xp:.0f},{y0 - hi[i] * scale:.1f}")
        parts.append(f'<polygon points="{" ".join(pts)}" fill="{color}" opacity="0.85"/>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))


# ---------------- 主流程 ----------------

def build_cases():
    """单变量扫描：分别改变 回声延迟 / 回声增益 / 噪声类型（lab 要求）。"""
    cases = []
    cases.append(dict(delay=0, gain=-99, noise="none", noise_snr=None,
                      note="基线:近端原声"))
    for d in (50, 150, 600):                      # 回声延迟 ms（短/中/长）
        cases.append(dict(delay=d, gain=-10.0, noise="none", noise_snr=None,
                          note=f"回声延迟 {d}ms"))
    for g in (-20.0, -10.0, -3.0):                # 回声增益 dB
        cases.append(dict(delay=150, gain=g, noise="none", noise_snr=None,
                          note=f"回声增益 {g}dB"))
    for k in ("white", "pink"):                   # 噪声类型，SNR +15dB 固定
        cases.append(dict(delay=150, gain=-10.0, noise=k, noise_snr=15.0,
                          note=f"{k} 噪声"))
    return cases


def run(args):
    os.makedirs(os.path.join(OUT_DIR, "wavs"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "waves"), exist_ok=True)
    near_path = os.path.join(BASE, "input", "near_end.wav")
    x = load_wav(near_path)
    log(f"近端语音: {near_path}  时长 {len(x) / SR:.1f}s")

    all_cases = build_cases()
    if args.quick:
        idx = [0, 3, 5, 8]                        # 基线/600ms/-3dB/pink
        cases = [all_cases[i] for i in idx]
    else:
        cases = all_cases
    log(f"用例数: {len(cases)}" + (" (quick)" if args.quick else ""))
    if args.skip_asr:
        log("--skip-asr: 不加载 FunASR")

    csv_path = os.path.join(OUT_DIR, "cases.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["case", "note", "delay_ms", "gain_db", "noise",
                    "noise_snr_db", "snr_db", "vad_seg_energy", "vad_seg_fsmn",
                    "asr_text"])
        for i, c in enumerate(cases):
            t0 = time.time()
            gain = 0.0 if c["gain"] <= -90 else 10 ** (c["gain"] / 20.0)
            y = add_echo(x, c["delay"], gain)
            if c["noise"] != "none":
                noise = make_noise(c["noise"], len(y))
                target = float(np.mean(x ** 2)) / 10 ** (c["noise_snr"] / 10.0)
                pn = float(np.mean(noise ** 2)) + 1e-12
                y = y + noise.astype(np.float32) * np.sqrt(target / pn)
            y = y.astype(np.float32)
            snr = snr_db(x, y - x)
            segs = energy_vad_segments(y)

            tag = f"case_{i:02d}"
            mix_wav = os.path.join(OUT_DIR, "wavs", f"{tag}_mix.wav")
            save_wav(mix_wav, y)
            asr_text, nfs = "", ""
            if not args.skip_asr:
                asr_text, nfs = transcribe(mix_wav)

            w.writerow([tag, c["note"], c["delay"], c["gain"], c["noise"],
                        c["noise_snr"], f"{snr:.1f}", len(segs), nfs, asr_text])
            f.flush()
            wave_svg(x, y, os.path.join(OUT_DIR, "waves", f"{tag}.svg"),
                     f"{tag} {c['note']} | SNR={snr:.1f}dB | VAD段数(能量)={len(segs)}")
            extra = "" if args.skip_asr else f" VAD(fsmn)={nfs} ASR={asr_text[:50]}"
            log(f"[{tag}] {c['note']}  SNR={snr:.1f}dB  VAD(能量)={len(segs)}段{extra}"
                f"  ({time.time() - t0:.0f}s)")

    log(f"完成 -> {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-asr", action="store_true")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(args)
