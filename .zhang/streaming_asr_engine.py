import os
import wave
import numpy as np
from typing import List, Dict, Optional
from funasr import AutoModel


class StreamingSession:
    """
    流式会议 ASR 引擎核心状态机 (Task 2)
    职责：
    1. 接收 16kHz 16bit 单声道 PCM 音频流，实时双写落盘到 session_audio.wav
    2. 驱动 paraformer-zh-streaming 实时输出斜体草稿事件 (asr_partial)
    3. 检测句尾静音/超长换气，触发局部 Flush，锁死物理时间戳输出确认事件 (asr_final)
    """

    SAMPLE_RATE = 16000
    BYTES_PER_SAMPLE = 2  # int16 = 2 bytes

    def __init__(
        self,
        session_wav_path: str = "session_audio.wav",
        asr_model: Optional[AutoModel] = None,
        silence_rms_threshold: float = 0.008,  # 静音能量阈值 (归一化 float32 [-1, 1])
        end_silence_chunks: int = 2,           # 连续多少个静音/空输出 Chunk 触发断句 (2 * 600ms = 1.2s)
        max_utt_duration_ms: int = 15000,      # 单句最长软上限 (15秒遇到微小停顿即强制截断)
        device: str = "cuda:0",
    ):
        self.session_wav_path = session_wav_path
        self.silence_rms_threshold = silence_rms_threshold
        self.end_silence_chunks = end_silence_chunks
        self.max_utt_duration_ms = max_utt_duration_ms

        # 1. FunASR 流式分块参数: [0, 10, 5] 对应 10 * 60ms = 600ms 步长 (9600 个采样点)
        self.chunk_size = [0, 10, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1
        self.chunk_stride_samples = self.chunk_size[1] * 960  # 9600 samples = 600ms

        # 2. 加载或复用 paraformer-zh-streaming 模型 (支持多会话共享同一个模型实例)
        if asr_model is not None:
            self.asr_model = asr_model
        else:
            print("[StreamingSession] 正在加载 paraformer-zh-streaming 模型...")
            self.asr_model = AutoModel(
                model="paraformer-zh-streaming",
                device=device,
                disable_update=True,
                disable_pbar=True,
            )

        # 3. 初始化实时双写 WAV 文件
        os.makedirs(os.path.dirname(os.path.abspath(session_wav_path)), exist_ok=True)
        self._wav_file = wave.open(self.session_wav_path, "wb")
        self._wav_file.setnchannels(1)
        self._wav_file.setsampwidth(self.BYTES_PER_SAMPLE)
        self._wav_file.setframerate(self.SAMPLE_RATE)

        # 4. 全局物理采样点时钟与 PCM 待处理缓冲区
        self.total_samples_written = 0    # 已写入 WAV 的总采样点数
        self.total_samples_processed = 0  # 已送入 ASR 引擎推理的总采样点数
        self._pcm_buffer = np.zeros(0, dtype=np.float32)

        # 5. 流式 ASR 上下文缓存与单句状态机 (Utterance State Machine)
        self.asr_cache = {}
        self.utt_counter = 0
        self.current_utt: Optional[Dict] = None
        self.silence_chunk_count = 0
        self.completed_utterances: List[Dict] = []

    @property
    def processed_ms(self) -> int:
        """基于已推理采样点数的严格物理时钟（毫秒），绝不使用系统时间"""
        return int(self.total_samples_processed * 1000 // self.SAMPLE_RATE)

    def _new_utt_id(self) -> str:
        self.utt_counter += 1
        return f"utt_{self.utt_counter:04d}"

    @staticmethod
    def _calc_rms(samples: np.ndarray) -> float:
        """计算音频块均方根能量 (RMS)"""
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))

    def feed_pcm_chunk(self, pcm_bytes: bytes, is_final: bool = False) -> List[Dict]:
        """
        核心流式入口：喂入任意长度的 16kHz 16bit 单声道 PCM 字节流
        返回当前触发的事件列表（可能包含 asr_partial 和/或 asr_final）
        """
        events: List[Dict] = []

        # Step 1: 实时双写落盘到 WAV 文件，并转换到 [-1.0, 1.0] float32 数组
        if pcm_bytes:
            self._wav_file.writeframes(pcm_bytes)
            self._wav_file._patchheader()  # 实时刷新 WAV 头，确保随时读取均合法
            int16_data = np.frombuffer(pcm_bytes, dtype=np.int16)
            self.total_samples_written += len(int16_data)
            float32_data = int16_data.astype(np.float32) / 32768.0
            self._pcm_buffer = np.concatenate([self._pcm_buffer, float32_data])

        # Step 2: 按 600ms (9600 samples) 标准步长循环消费缓冲区
        while len(self._pcm_buffer) >= self.chunk_stride_samples:
            chunk = self._pcm_buffer[: self.chunk_stride_samples]
            self._pcm_buffer = self._pcm_buffer[self.chunk_stride_samples :]

            # 判断这是否是整场会议的最后一个标准块
            is_last_chunk = is_final and (len(self._pcm_buffer) == 0)
            chunk_events = self._process_single_chunk(chunk, is_stream_end=is_last_chunk)
            events.extend(chunk_events)

        # Step 3: 如果是会议结束 (is_final=True)，处理不足 600ms 的尾部残余音频或未封口的句子
        if is_final:
            if len(self._pcm_buffer) > 0:
                tail_chunk = self._pcm_buffer
                self._pcm_buffer = np.zeros(0, dtype=np.float32)
                events.extend(self._process_single_chunk(tail_chunk, is_stream_end=True))
            elif self.current_utt is not None:
                # 缓冲区已空但还有一句话处于 partial 态，执行强制收尾 Flush
                final_ev = self._finalize_current_utterance(flush_asr=True)
                if final_ev:
                    events.append(final_ev)

            self.close_wav()

        return events

    def _process_single_chunk(self, chunk: np.ndarray, is_stream_end: bool = False) -> List[Dict]:
        """处理单个 600ms 音频块，驱动 partial -> final 状态流转"""
        events: List[Dict] = []
        chunk_start_ms = self.processed_ms
        self.total_samples_processed += len(chunk)
        chunk_end_ms = self.processed_ms

        rms = self._calc_rms(chunk)
        is_Active_speech = rms >= self.silence_rms_threshold

        # 情况 A：当前处于空闲静音态，且本块依然是静音 -> 直接跳过 ASR 推理，节省算力并防幻觉
        if self.current_utt is None and not is_Active_speech and not is_stream_end:
            return events

        # 情况 B：检测到有效能量开口，且当前没有活跃句子 -> 创建新句子 utt_id
        if self.current_utt is None and is_Active_speech:
            self.current_utt = {
                "utt_id": self._new_utt_id(),
                # 向前留 180ms 前导余量，防止句首辅音被截断
                "start_ms": max(0, chunk_start_ms - 180),
                "end_ms": None,
                "raw_text": "",
            }
            self.silence_chunk_count = 0

        # 执行流式 Paraformer 推理
        res = self.asr_model.generate(
            input=chunk,
            cache=self.asr_cache,
            is_final=is_stream_end,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
        )
        delta_text = res[0].get("text", "") if res else ""

        # 如果原本没创建句子但 ASR 吐出了字，兜底创建 current_utt
        if delta_text and self.current_utt is None:
            self.current_utt = {
                "utt_id": self._new_utt_id(),
                "start_ms": max(0, chunk_start_ms - 180),
                "end_ms": None,
                "raw_text": "",
            }

        # 更新当前句状态
        if self.current_utt is not None:
            if delta_text:
                self.current_utt["raw_text"] += delta_text
                self.silence_chunk_count = 0
                # 触发【斜体草稿事件 asr_partial】
                events.append({
                    "event": "asr_partial",
                    "utt_id": self.current_utt["utt_id"],
                    "start_ms": self.current_utt["start_ms"],
                    "current_ms": chunk_end_ms,
                    "text": self.current_utt["raw_text"],
                })
            else:
                # 本块未产出新字：如果能量也低于阈值，累加静音计数
                if not is_Active_speech:
                    self.silence_chunk_count += 1

            # 判定是否满足【句尾封口条件】
            utt_duration_ms = chunk_end_ms - self.current_utt["start_ms"]
            hit_normal_pause = self.silence_chunk_count >= self.end_silence_chunks
            hit_max_duration_pause = (
                utt_duration_ms >= self.max_utt_duration_ms
                and (not is_Active_speech or not delta_text)
            )

            if hit_normal_pause or hit_max_duration_pause or is_stream_end:
                # 若非流自然结束，执行一次局部 Flush 刷出 lookahead 缓冲里的尾字
                final_ev = self._finalize_current_utterance(flush_asr=(not is_stream_end))
                if final_ev:
                    events.append(final_ev)

        return events

    def _finalize_current_utterance(self, flush_asr: bool = True) -> Optional[Dict]:
        """对当前句执行尾字 Flush、锁死 [start_ms, end_ms] 并重置 ASR Cache"""
        if self.current_utt is None:
            return None

        # 传入极短静音帧 + is_final=True，把 Paraformer decoder 缓存里的尾字逼出来
        if flush_asr and self.asr_cache:
            dummy_tail = np.zeros(1600, dtype=np.float32)  # 100ms 虚拟静音仅用于 flush 解码器
            res = self.asr_model.generate(
                input=dummy_tail,
                cache=self.asr_cache,
                is_final=True,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
            )
            tail_text = res[0].get("text", "") if res else ""
            if tail_text:
                self.current_utt["raw_text"] += tail_text

        # 重置流式缓存，确保下一句从干净状态开始
        self.asr_cache = {}
        self.silence_chunk_count = 0

        raw_text = self.current_utt["raw_text"].strip()
        if not raw_text:
            # 过滤纯噪音触发的空句子，回退计数器
            self.utt_counter -= 1
            self.current_utt = None
            return None

        # 锁死物理结束时间戳（扣除末尾静音等待时间，但保证大于 start_ms）
        deduct_ms = max(0, (self.silence_chunk_count - 1) * 600)
        locked_end_ms = max(self.current_utt["start_ms"] + 300, self.processed_ms - deduct_ms)
        self.current_utt["end_ms"] = locked_end_ms

        final_utt = {
            "utt_id": self.current_utt["utt_id"],
            "start_ms": self.current_utt["start_ms"],
            "end_ms": self.current_utt["end_ms"],
            "raw_text": raw_text,
        }
        self.completed_utterances.append(final_utt)
        self.current_utt = None

        return {
            "event": "asr_final",
            **final_utt,
        }

    def close_wav(self):
        """安全关闭会话录音文件"""
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception:
                pass
            self._wav_file = None


# =====================================================================
# 本地模拟流式自测与 Task 1 联调入口
# =====================================================================
if __name__ == "__main__":
    import sys
    import time
    import soundfile as sf

    # 1. 指定测试音频路径（支持命令行传入: python streaming_asr_engine.py your_test.wav）
    test_wav = sys.argv[1] if len(sys.argv) > 1 else "test_meeting.wav"
    if not os.path.exists(test_wav):
        print(f"[提示] 未找到测试音频 '{test_wav}'。")
        print("用法: python streaming_asr_engine.py <16kHz单声道测试音频.wav>")
        sys.exit(0)

    # 2. 读取测试音频并转换为 16kHz 16bit 单声道 PCM 字节流
    audio_data, sr = sf.read(test_wav, dtype="int16")
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]  # 多声道取第一声道
    if sr != 16000:
        raise ValueError(f"测试音频采样率必须为 16000Hz，当前为 {sr}Hz")

    output_session_wav = "session_audio.wav"
    session = StreamingSession(session_wav_path=output_session_wav)

    # 每次模拟推流 300ms (4800 个采样点)，测试内部缓冲与 600ms 步长对齐
    step_samples = 4800
    total_samples = len(audio_data)

    print("\n" + "=" * 65)
    print("🎙️  开始模拟实时流式会议推流 (斜体草稿 -> 句尾锁定)")
    print("=" * 65)

    for idx in range(0, total_samples, step_samples):
        pcm_slice = audio_data[idx : idx + step_samples].tobytes()
        is_end = (idx + step_samples) >= total_samples

        events = session.feed_pcm_chunk(pcm_slice, is_final=is_end)

        for ev in events:
            if ev["event"] == "asr_partial":
                # 用 \r 在同一行动态刷新，模拟前端【灰色斜体实时跳字】
                sys.stdout.write(
                    f"\r\033[3;90m[草稿 {ev['utt_id']} | {ev['start_ms']}ms~] {ev['text']}\033[0m"
                )
                sys.stdout.flush()
            elif ev["event"] == "asr_final":
                # 换行定格打印，模拟【句尾时间锁死】
                sys.stdout.write("\r" + " " * 80 + "\r")  # 清除当前草稿行
                print(
                    f"✅ [锁定 {ev['utt_id']}] [{ev['start_ms']:06d}ms -> {ev['end_ms']:06d}ms]: {ev['raw_text']}"
                )

    print("\n" + "=" * 65)
    print(f"💾 会话录音已完整双写保存至: {output_session_wav}")
    print(f"📊 共计锁定 {len(session.completed_utterances)} 个有效单句 (Utterances)")
    print("=" * 65)

    # 3. 自动尝试联动 Task 1 (offline_diarizer.py) 进行会后说话人回填测试
    try:
        from offline_diarizer import run_diarization, align_utterances_with_speakers

        print("\n⏳ 正在调用 Task 1 (offline_diarizer) 执行会后全局说话人识别与回填...")
        t0 = time.time()
        spk_segments = run_diarization(output_session_wav)
        utt_to_spk = align_utterances_with_speakers(session.completed_utterances, spk_segments)
        cost_s = time.time() - t0

        print(f"🎉 说话人回填完成 (耗时 {cost_s:.2f}s)！最终会议纪要如下：\n")
        for utt in session.completed_utterances:
            uid = utt["utt_id"]
            spk = utt_to_spk.get(uid, "Speaker_Unknown")
            s_sec = utt["start_ms"] / 1000.0
            e_sec = utt["end_ms"] / 1000.0
            print(f"[{s_sec:06.2f}s - {e_sec:06.2f}s] 【{spk}】 ({uid}): {utt['raw_text']}")

    except ImportError:
        print("\n[提示] 尚未检测到同目录下的 `offline_diarizer.py` (Task 1)。")
        print("等 Agent 完成 Task 1 后，再次运行本脚本即可自动查看【流式转写 + 离线说话人回填】的完整联调结果！")