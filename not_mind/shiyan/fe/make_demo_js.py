# -*- coding: utf-8 -*-
"""把 input/near_end.wav / far_end.wav 转成 fe/demo_audio.js（近端）与 fe/demo_audio_far.js（远端/参考信号）。
前端无网络加载用；用法：python make_demo_js.py"""
import base64
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

def dump(src, dest, var):
    with open(src, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(f"// 演示音频 base64(wav)：由 make_demo_js.py 生成，勿手改\n")
        f.write(f'window.{var} = "{b64}";\n')
    print(f"OK -> {dest} ({os.path.getsize(dest)} bytes)")

dump(os.path.join(ROOT, "input", "near_end.wav"), os.path.join(BASE, "demo_audio.js"), "DEMO_AUDIO_B64")
dump(os.path.join(ROOT, "input", "far_end.wav"), os.path.join(BASE, "demo_audio_far.js"), "DEMO_FAR_B64")
