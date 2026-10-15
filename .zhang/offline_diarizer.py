import os
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sklearn.cluster import SpectralClustering


class OverlapAwareDiarizer:
    """
    抗重叠污染的离线会议说话人日志引擎 (基于 ModelScope 3D-Speaker CAM++ & fsmn-vad)
    专门解决：多人抢话、插话导致的声纹混合特征污染与聚类崩盘问题
    """

    SAMPLE_RATE = 16000

    def __init__(
        self,
        win_sec: float = 0.9,            # 缩短声纹提取窗长至 0.9s (降低单窗内含2人语音的概率)
        step_sec: float = 0.45,          # 步长 0.45s，提高时间边界分辨率
        overlap_margin: float = 0.065,   # Top1 与 Top2 说话人余弦相似度差值小于此阈值时，判定为重叠语音
        device: str = "cuda:0",
    ):
        self.win_samples = int(win_sec * self.SAMPLE_RATE)
        self.step_samples = int(step_sec * self.SAMPLE_RATE)
        self.overlap_margin = overlap_margin

        print("[OfflineDiarizer] 正在加载离线 VAD 与 3D-Speaker (CAM++) 声纹模型...")
        self.vad_model = AutoModel(
            model="fsmn-vad", device=device, disable_update=True, disable_pbar=True
        )
        # 使用 ModelScope 官方 3D-Speaker CAM++ 声纹特征提取器
        self.sv_pipeline = pipeline(
            task=Tasks.speaker_verification,
            model="iic/speech_campplus_sv_zh-cn_16k-common",
            device=device,
        )

    def _extract_subsegments_and_embeddings(self, wav_path: str):
        """Step 1: VAD 切出有效语音段 -> 按 0.9s/0.45s 细粒度滑窗提取 L2 归一化声纹向量"""
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.SAMPLE_RATE:
            raise ValueError(f"音频采样率需为 16000Hz，当前为 {sr}Hz")

        vad_res = self.vad_model.generate(input=wav_path)
        vad_segments = vad_res[0].get("value", []) if vad_res else []

        subsegments = []
        embeddings = []

        for seg_start_ms, seg_end_ms in vad_segments:
            s_idx = int(seg_start_ms * self.SAMPLE_RATE // 1000)
            e_idx = int(seg_end_ms * self.SAMPLE_RATE // 1000)
            seg_len = e_idx - s_idx
            if seg_len < int(0.4 * self.SAMPLE_RATE):  # 忽略短于 0.4s 的极碎杂音
                continue

            # 细粒度滑窗切分
            cur = s_idx
            while cur < e_idx:
                end = min(cur + self.win_samples, e_idx)
                # 若尾部剩余不足 0.5s 且前面已有窗口，合并到上一个窗口或向前对齐
                if (end - cur) < int(0.5 * self.SAMPLE_RATE) and cur > s_idx:
                    cur = max(s_idx, end - self.win_samples)

                wav_slice = audio[cur:end]
                emb_out = self.sv_pipeline([wav_slice], output_emb=True)
                emb = np.array(emb_out["embs"][0], dtype=np.float32)
                # L2 归一化，使内积等于余弦相似度
                emb = emb / (np.linalg.norm(emb) + 1e-8)

                subsegments.append({
                    "start_ms": int(cur * 1000 // self.SAMPLE_RATE),
                    "end_ms": int(end * 1000 // self.SAMPLE_RATE),
                })
                embeddings.append(emb)

                if end == e_idx:
                    break
                cur += self.step_samples

        return subsegments, np.array(embeddings)

    @staticmethod
    def _estimate_num_speakers(affinity_matrix: np.ndarray, max_spk: int = 8) -> int:
        """基于相似度矩阵特征值间隔 (Eigen-gap) 自动估计会议人数"""
        if len(affinity_matrix) <= 2:
            return 1
        # 对余弦相似度进行阈值去噪强化
        W = np.maximum(0.0, affinity_matrix)
        np.fill_diagonal(W, 0.0)
        threshold = np.percentile(W, 70)
        W[W < threshold] = 0.0

        degree = np.sum(W, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-8))
        L_norm = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt

        eigenvalues = np.sort(np.linalg.eigvalsh(L_norm))
        max_k = min(max_spk, len(eigenvalues) - 1)
        if max_k <= 1:
            return 1

        # 检查若全图高度相似（单人会议），直接返回 1
        if np.mean(affinity_matrix) > 0.68:
            return 1

        gaps = np.diff(eigenvalues[1 : max_k + 1])
        best_k = int(np.argmax(gaps) + 2)
        return best_k

    def run(self, wav_path: str, oracle_num_speakers: Optional[int] = None) -> List[Dict]:
        """执行抗重叠全局聚类，输出带重叠检测的说话人时间片"""
        subsegments, embs = self._extract_subsegments_and_embeddings(wav_path)
        n_segs = len(subsegments)
        if n_segs == 0:
            return []
        if n_segs == 1:
            return [{**subsegments[0], "speaker": "Speaker_1", "overlap_speaker": None}]

        # 1. 计算全局余弦相似度矩阵
        sim_matrix = np.dot(embs, embs.T)

        # 2. 剔除“边缘混合帧”先做核心样本聚类（每个样本与邻居的平均相似度太低或处于模糊区则降权）
        num_spk = oracle_num_speakers or self._estimate_num_speakers(sim_matrix)
        if num_spk == 1:
            return [
                {**seg, "speaker": "Speaker_1", "overlap_speaker": None}
                for seg in subsegments
            ]

        # 将余弦相似度映射到 [0, 1] 并锐化，斩断重叠语音造成的弱连接桥
        affinity = np.clip((sim_matrix + 1.0) / 2.0, 0.0, 1.0) ** 3
        cluster_model = SpectralClustering(
            n_clusters=num_spk,
            affinity="precomputed",
            assign_labels="discretize",
            random_state=42,
        )
        init_labels = cluster_model.fit_predict(affinity)

        # 3. 计算每个说话人的【高纯度核心质心 (Core Centroids)】
        # 只取每个簇内部相似度最高的前 70% 纯净单人切片算质心，滤除重叠污染帧
        centroids = np.zeros((num_spk, embs.shape[1]), dtype=np.float32)
        for k in range(num_spk):
            idx_k = np.where(init_labels == k)[0]
            if len(idx_k) == 0:
                continue
            raw_center = np.mean(embs[idx_k], axis=0)
            raw_center /= np.linalg.norm(raw_center) + 1e-8
            sims_to_center = np.dot(embs[idx_k], raw_center)
            cutoff = np.percentile(sims_to_center, 30)  # 淘汰底部的 30% 可疑混合切片
            pure_idx = idx_k[sims_to_center >= cutoff]
            pure_center = np.mean(embs[pure_idx], axis=0)
            centroids[k] = pure_center / (np.linalg.norm(pure_center) + 1e-8)

        # 4. 重新计算每个切片对各纯净质心的得分，识别【主说话人】与【重叠次说话人】
        scores = np.dot(embs, centroids.T)  # shape: [n_segs, num_spk]
        results = []
        for i, seg in enumerate(subsegments):
            sorted_spk = np.argsort(scores[i])[::-1]
            top1_spk = int(sorted_spk[0])
            top2_spk = int(sorted_spk[1])
            top1_score = scores[i, top1_spk]
            top2_score = scores[i, top2_spk]

            # 若 Top1 和 Top2 得分都很高 (>0.45) 且极其接近 (差值 < overlap_margin) -> 判定为两人重叠语音
            is_overlap = (top2_score > 0.45) and ((top1_score - top2_score) < self.overlap_margin)

            results.append({
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "speaker": f"Speaker_{top1_spk + 1}",
                "overlap_speaker": f"Speaker_{top2_spk + 1}" if is_overlap else None,
            })

        return results


# 全局单例缓存，避免重复加载模型
_DIARIZER_INSTANCE: Optional[OverlapAwareDiarizer] = None


def run_diarization(wav_path: str, num_speakers: Optional[int] = None) -> List[Dict]:
    """Task 1 标准接口：输入整场 WAV，输出说话人时间片段列表"""
    global _DIARIZER_INSTANCE
    if _DIARIZER_INSTANCE is None:
        _DIARIZER_INSTANCE = OverlapAwareDiarizer()
    return _DIARIZER_INSTANCE.run(wav_path, oracle_num_speakers=num_speakers)


def align_utterances_with_speakers(
    utterances: List[Dict], spk_segments: List[Dict]
) -> Dict[str, str]:
    """
    Task 1 标准对齐接口（支持重叠说话人标注）：
    计算每句 Utterance [start_ms, end_ms] 内各说话人的累计活跃时长。
    若单句内存在显著的第二说话人（占比 > 25% 或命中重叠窗口），标注为 "Speaker_X (+Speaker_Y)"
    """
    mapping: Dict[str, str] = {}
    if not spk_segments:
        return {u["utt_id"]: "Speaker_1" for u in utterances}

    for utt in utterances:
        uid = utt["utt_id"]
        u_start, u_end = utt["start_ms"], utt["end_ms"]
        spk_durations: Dict[str, int] = {}

        for seg in spk_segments:
            overlap = max(0, min(u_end, seg["end_ms"]) - max(u_start, seg["start_ms"]))
            if overlap > 0:
                main_spk = seg["speaker"]
                spk_durations[main_spk] = spk_durations.get(main_spk, 0) + overlap
                # 如果该时间片本身被检测出双人重叠，给次说话人也计入权重
                if seg.get("overlap_speaker"):
                    ov_spk = seg["overlap_speaker"]
                    spk_durations[ov_spk] = spk_durations.get(ov_spk, 0) + int(overlap * 0.6)

        if spk_durations:
            sorted_spks = sorted(spk_durations.items(), key=lambda x: x[1], reverse=True)
            primary_spk, primary_dur = sorted_spks[0]
            total_dur = sum(spk_durations.values())

            # 若第二说话人在该句内的声纹时长占比超过 28%，说明这句发生了抢话/重叠语音
            if len(sorted_spks) > 1 and (sorted_spks[1][1] / max(1, primary_dur)) >= 0.35:
                secondary_spk = sorted_spks[1][0]
                mapping[uid] = f"{primary_spk} (+{secondary_spk} 重叠)"
            else:
                mapping[uid] = primary_spk
        else:
            # 若无直接交集，取时间距离最近的片段归属
            u_mid = (u_start + u_end) // 2
            nearest_seg = min(
                spk_segments, key=lambda s: abs(((s["start_ms"] + s["end_ms"]) // 2) - u_mid)
            )
            mapping[uid] = nearest_seg["speaker"]

    return mapping