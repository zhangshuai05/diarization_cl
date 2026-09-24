# -*- coding: utf-8 -*-
"""
FunASR 说话人分离流水线：FSMN-VAD + CAM++ 声纹嵌入 + 谱聚类 + SEACO ASR + 标点。
输出 sentence_info 带 spk 标签（完整 JSON + 可读 txt）。

用法:
  python run_diar_spk.py <wav1> [wav2 ...]                  # 自动估计说话人数(特征间隙+0.78合并)
  python run_diar_spk.py --preset 2 <wav1> [wav2 ...]       # 预置说话人数
  python run_diar_spk.py --preset 2 --outdir output/p2 ...  # 指定输出目录
"""
import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", r"C:\Users\EDY\.cache\modelscope")

VAD_MODEL = os.path.join(
    HERE, "iic--speech_fsmn_vad_zh-cn-16k-common-pytorch", "snapshots", "master")
SEACO_MODEL = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
PUNC_MODEL = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
SPK_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"


def build_model():
    from funasr import AutoModel
    return AutoModel(
        model=SEACO_MODEL, model_revision="master",
        vad_model=VAD_MODEL,
        vad_kwargs={"dynamic_silence": False},
        punc_model=PUNC_MODEL,
        spk_model=SPK_MODEL,
        disable_update=True, disable_pbar=True,
    )


def process(model, wav, outdir, preset):
    t0 = time.time()
    print(f"\n===== transcribing: {wav} (preset_spk_num={preset})", flush=True)
    gen_kwargs = dict(
        batch_size=1, sentence_timestamp=True,
        merge_vad=True, merge_length_s=15,
    )
    if preset:
        gen_kwargs["preset_spk_num"] = preset
    res = model.generate(input=wav, **gen_kwargs)
    out = res[0]
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(wav).rsplit(".", 1)[0]

    json_path = os.path.join(outdir, base + "_diar.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    si = out.get("sentence_info") or []
    txt_path = os.path.join(outdir, base + "_diar.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in si:
            f.write("[{:>8.2f} - {:>8.2f}] spk{}  {}\n".format(
                s.get("start", 0.0), s.get("end", 0.0),
                s.get("spk", "?"), s.get("text", "")))

    spks = sorted(set(str(s.get("spk", "?")) for s in si)) if si else []
    print(f"saved: {json_path} / {txt_path} ({len(si)} sentences)")
    print("speaker labels in file:", spks)
    print(f"elapsed: {time.time()-t0:.1f}s", flush=True)


def main(wavs, outdir, preset):
    t0 = time.time()
    print("loading VAD + SEACO + punc + CAM++ ...", flush=True)
    model = build_model()
    print(f"models loaded in {time.time()-t0:.1f}s", flush=True)
    for wav in wavs:
        try:
            process(model, wav, outdir, preset)
        except Exception as e:
            print(f"[FAILED] {wav}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    preset = None
    outdir = None
    wavs = []
    i = 0
    while i < len(args):
        if args[i] == "--preset" and i + 1 < len(args):
            preset = int(args[i + 1]); i += 2
        elif args[i] == "--outdir" and i + 1 < len(args):
            outdir = args[i + 1]; i += 2
        else:
            wavs.append(args[i]); i += 1
    if not wavs:
        wavs = [os.path.join(HERE, "Audio", "Device116", "Call20260718210228.wav")]
    if outdir is None:
        outdir = os.path.join(HERE, "output", f"p{preset}" if preset else "auto")
    if not os.path.isabs(outdir):
        outdir = os.path.join(HERE, outdir)
    main(wavs, outdir, preset)
