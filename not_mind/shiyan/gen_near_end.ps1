# 用 Windows 系统 TTS (SAPI) 合成"近端语音" -> input/near_end.wav
# 16 kHz / 16bit / mono，与 FunASR 链路对齐。教学简化：合成语音代替真人录音。
param(
    [string]$Text = "您好，欢迎光临。请问今天需要办理什么业务？好的，这边帮您查询一下，请稍等。您这张卡已经激活成功了，请收好，祝您生活愉快。"
)

$ErrorActionPreference = "Stop"
$outDir = Join-Path $PSScriptRoot "input"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outWav = Join-Path $outDir "near_end.wav"

Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 优先选一个中文语音；找不到则退回默认语音并警告
$zh = $synth.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -like "zh*" } | Select-Object -First 1
if ($zh) {
    $synth.SelectVoice($zh.VoiceInfo.Name)
    Write-Host "TTS voice: $($zh.VoiceInfo.Name)"
} else {
    Write-Host "WARN: no zh-CN voice installed, using default: $($synth.Voice.Name)"
}

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($outWav, $fmt)
$synth.Speak($Text)
$synth.Dispose()

$len = (Get-Item $outWav).Length
Write-Host "OK -> $outWav ($len bytes)"
