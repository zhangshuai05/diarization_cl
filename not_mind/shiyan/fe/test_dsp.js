// test_dsp.js —— DSP 核心的数值回归检查（node test_dsp.js）
// 验证：FFT 往返、回声合成、NLMS 收敛、谱减无 NaN、AGC 电平、VAD 分段、全链路
const assert = require("assert");
const D = require("./dsp.js");
const SR = D.SR;

function rand(n, seed) {
  let s = seed || 1;
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    a[i] = s / 0x3fffffff - 1;
  }
  return a;
}
function tone(n, f, amp) {
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = amp * Math.sin(2 * Math.PI * f * i / SR);
  return a;
}
function rms(x) {
  if (!x.length) return 0;
  let e = 0;
  for (let i = 0; i < x.length; i++) e += x[i] * x[i];
  return Math.sqrt(e / x.length);
}

// 1. FFT 往返
{
  const N = 1024, x = rand(N, 7);
  const re = Float32Array.from(x), im = new Float32Array(N);
  D.fft(re, im); D.fft(re, im, true);
  let err = 0;
  for (let i = 0; i < N; i++) err = Math.max(err, Math.abs(re[i] / N - x[i]));
  assert(err < 1e-4, `FFT 往返误差 ${err}`);
  console.log(`[1] FFT 往返 OK (max err=${err.toExponential(2)})`);
}

// 2. 回声合成：远端 r 延迟/增益叠加，近端 x 不动
{
  const x = tone(24000, 440, 1);
  const r = tone(24000, 180, 1);
  const { mix } = D.makeMix(x, r, 100, -10, "none", null);
  const d = 1600; // 100ms
  assert(Math.abs(mix[d] - (x[d] + r[0] * 0.3162)) < 0.02, "回声叠加点数值");
  assert(Math.abs(mix[0] - x[0]) < 1e-6, "延迟前不叠加(近端原样)");
  assert(Math.abs(mix[d + 100] - (x[d + 100] + r[100] * 0.3162)) < 0.05, "回声持续叠加");
  console.log("[2] 回声合成 OK (100ms/-10dB，近端保留+远端回波叠加)");
}

// 3. NLMS 收敛：参考=r，AEC 消除回声后应接近近端 x
{
  const x = rand(48000, 11);
  const r = rand(48000, 12);
  for (let i = 0; i < x.length; i++) x[i] *= 0.5;
  const { mix } = D.makeMix(x, r, 100, -6, "none", null);
  const out = D.nlmsAec(mix, r, 2048, 0.3); // μ=0.3：短音频内收敛快，稳态残差(教学点：excess MSE)
  // 残差 = out - x（应≈0），与回声能量混入前的"干扰"对比
  let pe = 0, pc = 0, px = 0;
  for (let i = 0; i < x.length; i++) { const d0 = out[i] - x[i]; pe += d0 * d0; const d1 = mix[i] - x[i]; pc += d1 * d1; px += x[i] * x[i]; }
  const echoDb = 10 * Math.log10((pc / px) + 1e-12);
  const resDb = 10 * Math.log10((pe / px) + 1e-12);
  assert(resDb < echoDb - 5, `NLMS 未充分消回声: 残差${resDb.toFixed(1)}dB vs 回声${echoDb.toFixed(1)}dB`);
  console.log(`[3] NLMS 收敛 OK (回声 -${(-echoDb).toFixed(1)}dB, 消后残差 ${resDb.toFixed(1)}dB ≈ 收敛)`);
}

// 4. 谱减：白噪声上无 NaN 且能量下降
{
  const x = rand(48000, 3);
  const out = D.spectralSubtract(x, 0.9);
  assert(!out.some(Number.isNaN) && !out.some(v => !isFinite(v)), "谱减出现 NaN");
  const dB = 20 * Math.log10((rms(out) + 1e-12) / (rms(x) + 1e-12));
  assert(dB < -5, `谱减降噪不足: ${dB}dB`);
  console.log(`[4] 谱减 OK (纯噪声输出抑制 ${dB.toFixed(1)}dB, 无 NaN)`);
}

// 5. AGC：输出 RMS 逼近目标
{
  const x = tone(SR * 2, 200, 0.05);
  const { out, gain } = D.agc(x, Math.pow(10, -20 / 20));
  const r = rms(out);
  assert(Math.abs(20 * Math.log10(r) - -20) < 0.5, `AGC 目标未达: ${20 * Math.log10(r)}dB`);
  console.log(`[5] AGC OK (输出 ${(20 * Math.log10(r)).toFixed(2)} ≈ 目标 -20dB, 增益 ${(20 * Math.log10(gain)).toFixed(1)}dB)`);
}

// 6. VAD：构造 1-2s/3-4s 两段语音应得 2 段
{
  const n = SR * 5;
  const x = new Float32Array(n);
  for (let i = SR; i < 2 * SR; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * 220 * i / SR);
  for (let i = 3 * SR; i < 4 * SR; i++) x[i] = 0.5 * Math.sin(2 * Math.PI * 220 * i / SR);
  const segs = D.vadSegments(x, -30);
  assert.strictEqual(segs.length, 2, `VAD 段数 ${segs.length}`);
  assert(Math.abs(segs[0][0] - 1) < 0.1 && Math.abs(segs[1][1] - 4) < 0.1, "VAD 起止偏差大");
  console.log(`[6] VAD OK (段=[${segs.map(s => s.map(t => t.toFixed(2)).join("-")).join(", ")}])`);
}

// 7. 全链路 computeAll（窄带/纯音信号下 NLMS 欠定，故用宽带信号当近端/远端）
{
  const x = rand(SR * 3, 5);
  const r = rand(SR * 3, 6);
  for (let i = 0; i < x.length; i++) { x[i] *= 0.2; r[i] *= 0.2; }
  const res = D.computeAll(x, r, {
    delayMs: 100, gainDb: -10, noiseKind: "white", noiseSnrDb: 15,
    aecOn: true, aecLen: 2048, // 必须 ≥ 回声延迟(1600 采样) 才能收敛，教学点
    aecMu: 0.3,
    ansOn: true, ansStrength: 0.8,
    agcOn: true, agcTarget: -20,
    vadOn: true, vadThr: -30,
  });
  for (const k of ["echo", "mix", "aec", "ans", "agc"]) {
    assert(!res[k].some(Number.isNaN), `${k} 含 NaN`);
    assert(res[k].length === x.length, `${k} 长度不符`);
  }
  // AEC 核心功能验证：输出中"回声成分"(与延迟参考 r 的相关系数)应显著下降
  // (双讲场景下 SNR 数字可能不升反降——excess MSE，真实产品靠 DTD 处理，教学点)
  const d = 1600;
  const corr = (a, b) => {
    let ma = 0, mb = 0; const n = a.length - d;
    for (let i = d; i < a.length; i++) { ma += a[i]; mb += b[i - d]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = d; i < a.length; i++) {
      const xa = a[i] - ma, xb = b[i - d] - mb;
      num += xa * xb; da += xa * xa; db += xb * xb;
    }
    return num / Math.sqrt(da * db);
  };
  const cMix = corr(res.mix, r), cAec = corr(res.aec, r);
  assert(cAec < cMix * 0.5, `AEC 后回声相关未下降: mix=${cMix.toFixed(3)} aec=${cAec.toFixed(3)}`);
  console.log(`[7] 全链路 OK (回声相关 ${cMix.toFixed(3)}→${cAec.toFixed(3)}; SNR 输入${res.metrics.snrMix.toFixed(1)}→AEC${res.metrics.snrAec?.toFixed(1)}dB; VAD段数${res.metrics.vadSegs})`);
}

console.log("\n全部通过 ✓");
