// dsp.js —— 3A/VAD 教学演示的 DSP 核心（无 DOM，浏览器 <script> 与 node require 通用）
// 链路：近端 x → 混入回声(延迟+衰减)+噪声 → AEC(NLMS) → ANS(谱减) → AGC → VAD(能量法)
// 与 shiyan/aec_lab.py 口径一致：16k、VAD 帧20ms/hop10ms/阈值-30dB/合并250ms/最短120ms
(function (root) {
  "use strict";

  const SR = 16000;

  // ---------- 基础工具 ----------
  function hann(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
    return w;
  }

  // radix-2 复数 FFT/DIFT（n 为 2 的幂），in-place；inverse=true 为逆变换（调用方补 1/n）
  function fft(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (inverse ? 2 : -2) * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
          const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

  // ---------- 干扰合成（与 aec_lab.py 同口径；r=远端/参考信号）----------
  function makeMix(x, r, delayMs, gainDb, noiseKind, noiseSnrDb) {
    const gain = gainDb <= -90 ? 0 : Math.pow(10, gainDb / 20);
    const echo = makeEcho(r, delayMs, gain);       // 回声 = 远端参考信号的延迟衰减副本
    const y = new Float32Array(x.length);
    y.set(x);
    for (let i = 0; i < echo.length; i++) y[i] += echo[i];
    if (noiseKind !== "none" && noiseSnrDb != null) {
      const noise = makeNoise(noiseKind, x.length);
      let px = 0, pn = 0;
      for (let i = 0; i < x.length; i++) { px += x[i] * x[i]; pn += noise[i] * noise[i]; }
      const target = (px / x.length) / Math.pow(10, noiseSnrDb / 10);
      const scale = Math.sqrt(target / (pn / x.length + 1e-12));
      for (let i = 0; i < y.length; i++) y[i] += noise[i] * scale;
    }
    return { echo, mix: y };
  }
  // ---------- 原始工具（makeMix 依赖）----------
  function makeEcho(x, delayMs, gainLin) {
    // 纯回声：参考信号经 delayMs 延迟、gainLin 衰减后的副本
    const d = Math.round(delayMs / 1000 * SR);
    const y = new Float32Array(x.length);
    if (d === 0) {
      for (let n = 0; n < x.length; n++) y[n] = x[n] * gainLin;
      return y;
    }
    for (let n = d; n < x.length; n++) y[n] = x[n - d] * gainLin;
    return y;
  }

  function makeNoise(kind, n) {
    let x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.random() * 2 - 1;
    if (kind === "pink") {
      const N = nextPow2(n);
      const re = new Float32Array(N), im = new Float32Array(N);
      re.set(x);
      fft(re, im);
      for (let k = 0; k <= N / 2; k++) {
        const g = 1 / Math.sqrt(Math.max(k, 0.25));
        re[k] *= g; im[k] *= g;
        if (k > 0 && k < N / 2) { re[N - k] *= g; im[N - k] *= g; }
      }
      fft(re, im, true); // 逆变换
      for (let i = 0; i < N; i++) x[i] = re[i] / N;
    }
    return x;
  }

  // ---------- AEC：归一化最小均方 (NLMS)，参考=扬声器信号 ----------
  // y: 近端麦输入(含回声+噪声)，ref: 参考信号(回声源)。收敛后 out ≈ 近端本体+噪声
  function nlmsAec(y, ref, L, mu) {
    const n = y.length;
    const w = new Float32Array(L);
    const out = new Float32Array(n);
    const buf = new Float32Array(L);
    for (let i = 0; i < n; i++) {
      buf.copyWithin(1, 0, L - 1);
      buf[0] = ref[i] || 0;
      let est = 0, pb = 1e-8;
      for (let k = 0; k < L; k++) { est += w[k] * buf[k]; pb += buf[k] * buf[k]; }
      const e = y[i] - est;
      out[i] = e;
      const g = (mu * e) / pb;
      for (let k = 0; k < L; k++) w[k] += g * buf[k];
    }
    return out;
  }

  // ---------- ANS：谱减（STFT，帧512/hop128/Hann，噪声谱=逐bin中位数谱估计）----------
  function spectralSubtract(x, strength, frameN, hop) {
    frameN = frameN || 512; hop = hop || 128;
    const N = x.length;
    const nF = Math.max(1, Math.floor((N - frameN) / hop) + 1);
    const win = hann(frameN);
    const F = frameN;
    // 稀疏采样前 240 帧求逐 bin 中位数谱（对平稳噪声稳健）
    const nEst = Math.min(nF, 240);
    const frames = [];
    for (let f = 0; f < nEst; f++) {
      const s = f * hop;
      const re = new Float32Array(F), im = new Float32Array(F);
      for (let i = 0; i < F; i++) re[i] = x[Math.min(s + i, N - 1)] * win[i];
      fft(re, im);
      const mag = new Float32Array(F / 2);
      for (let k = 0; k < F / 2; k++) mag[k] = Math.hypot(re[k], im[k]);
      frames.push(mag);
    }
    const noiseAvg = new Float32Array(F / 2);
    const vals = new Float32Array(nEst);
    for (let k = 0; k < F / 2; k++) {
      for (let f = 0; f < nEst; f++) vals[f] = frames[f][k];
      vals.sort();
      noiseAvg[k] = vals[Math.floor(nEst / 2)];
    }
    // 逐帧谱减 + OLA 重建
    const out = new Float32Array(N);
    const winSum = new Float32Array(N);
    for (let f = 0; f < nF; f++) {
      const s = f * hop;
      const re = new Float32Array(F), im = new Float32Array(F);
      for (let i = 0; i < F; i++) re[i] = x[Math.min(s + i, N - 1)] * win[i];
      fft(re, im);
      for (let k = 0; k < F / 2; k++) {
        const mag = Math.hypot(re[k], im[k]);
        // 谱减 α=2.5(over-subtraction) + spectral floor(0.12*mag 保底)
        const nm = Math.max(mag - strength * noiseAvg[k] * 2.5, 0.12 * mag);
        const scale = nm / (mag + 1e-12);
        re[k] *= scale; im[k] *= scale;
      }
      for (let k = 1; k < F / 2; k++) { re[F - k] = re[k]; im[F - k] = -im[k]; }
      fft(re, im, true); // 逆变换(未归一, 下式除 F)
      for (let i = 0; i < F; i++) {
        out[Math.min(s + i, N - 1)] += re[i] * win[i] / F;
        winSum[Math.min(s + i, N - 1)] += win[i] * win[i];
      }
    }
    for (let i = 0; i < N; i++) out[i] /= (winSum[i] + 1e-6);
    return out;
  }

  // ---------- AGC：目标 RMS 全局增益 + 限幅 ----------
  function agc(x, targetRms, maxGain) {
    maxGain = maxGain || 100; // 上限保护，默认放得很宽
    let rms = 0;
    for (let i = 0; i < x.length; i++) rms += x[i] * x[i];
    rms = Math.sqrt(rms / x.length) + 1e-9;
    let g = targetRms / rms;
    g = Math.min(g, maxGain);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = Math.max(-1, Math.min(1, x[i] * g));
    return { out, gain: g };
  }

  // ---------- VAD：能量法（与 aec_lab.py 同口径）----------
  function vadSegments(x, thrDb, frameMs, hopMs, maxGapMs, minLenMs) {
    frameMs = frameMs || 20; hopMs = hopMs || 10;
    maxGapMs = maxGapMs || 250; minLenMs = minLenMs || 120;
    thrDb = thrDb == null ? -30 : thrDb;
    const fl = Math.round(frameMs * SR / 1000), hl = Math.round(hopMs * SR / 1000);
    const nF = Math.max(0, Math.floor((x.length - fl) / hl) + 1);
    if (nF <= 0) return [];
    const db = new Float32Array(nF);
    for (let f = 0; f < nF; f++) {
      let e = 1e-12, s = f * hl;
      for (let i = 0; i < fl; i += 2) { const v = x[s + i]; e += v * v; }
      db[f] = 10 * Math.log10(e / (fl / 2));
    }
    const segs = [];
    let start = -1;
    for (let f = 0; f < nF; f++) {
      if (db[f] > thrDb && start < 0) start = f;
      else if (db[f] <= thrDb && start >= 0) { segs.push([start * hl, f * hl]); start = -1; }
    }
    if (start >= 0) segs.push([start * hl, x.length]);
    const merged = [];
    for (const [s, e] of segs) {
      if (merged.length && s - merged[merged.length - 1][1] < maxGapMs * SR / 1000) {
        merged[merged.length - 1][1] = e;
      } else merged.push([s, e]);
    }
    const minLen = minLenMs * SR / 1000;
    return merged.filter(([s, e]) => e - s >= minLen).map(([s, e]) => [s / SR, e / SR]);
  }

  // ---------- 指标 ----------
  function rmsDb(x) {
    let e = 0;
    for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    return 10 * Math.log10(e / x.length + 1e-12);
  }
  function snrDb2(clean, mixed) {
    let pc = 0, pd = 0;
    const n = Math.min(clean.length, mixed.length);
    for (let i = 0; i < n; i++) {
      pc += clean[i] * clean[i];
      const d = mixed[i] - clean[i];
      pd += d * d;
    }
    if (pd < 1e-12) return Infinity;
    return 10 * Math.log10((pc / n) / (pd / n));
  }

  // ---------- 频谱（供绘图）----------
  function avgSpectrum(x, frameN) {
    frameN = frameN || 1024;
    const F = nextPow2(frameN);
    const win = hann(F);
    const nF = Math.max(1, Math.floor(x.length / (F / 2)) - 1);
    const acc = new Float32Array(F / 2);
    let cnt = 0;
    const hop = F / 2;
    for (let f = 0; f < nF; f += 4) {
      const s = f * hop;
      const re = new Float32Array(F), im = new Float32Array(F);
      for (let i = 0; i < F; i++) re[i] = (x[Math.min(s + i, x.length - 1)] || 0) * win[i];
      fft(re, im);
      for (let k = 0; k < F / 2; k++) acc[k] += Math.hypot(re[k], im[k]);
      cnt++;
    }
    for (let k = 0; k < acc.length; k++) acc[k] /= Math.max(cnt, 1);
    return acc;
  }

  // ---------- 全链路 ----------
  function computeAll(x, r, p) {
    const { echo, mix } = makeMix(x, r, p.delayMs, p.gainDb, p.noiseKind, p.noiseSnrDb);
    const aecOut = p.aecOn ? nlmsAec(mix, r, p.aecLen, p.aecMu) : mix.slice();
    const ansOut = p.ansOn ? spectralSubtract(aecOut, p.ansStrength) : aecOut.slice();
    const agcRes = p.agcOn ? agc(ansOut, Math.pow(10, p.agcTarget / 20)) : { out: ansOut.slice(), gain: 1 };
    const segments = p.vadOn ? vadSegments(agcRes.out, p.vadThr) : [];
    return {
      echo, mix, aec: aecOut, ans: ansOut, agc: agcRes.out,
      segments,
      metrics: {
        snrMix: snrDb2(x, mix),
        snrAec: aecOut === mix ? null : snrDb2(x, aecOut),
        snrAns: ansOut === aecOut ? null : snrDb2(x, ansOut),
        rmsAgc: rmsDb(agcRes.out),
        vadSegs: segments.length,
        agcGain: agcRes.gain,
      },
    };
  }

  const api = {
    SR, hann, fft, nextPow2, makeEcho, makeNoise, makeMix,
    nlmsAec, spectralSubtract, agc, vadSegments, snrDb2, rmsDb, avgSpectrum, computeAll,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DSP = api;
})(typeof window !== "undefined" ? window : globalThis);
