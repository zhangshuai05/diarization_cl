# c1 说话人分离项目 · 论文清单（Claude 借鉴版）

> 生成日期：2026-09-09。本文件为纯资料文档，供 Claude 在开发 `claude-a2-v4-c1` 时参考；**不构成对代码的任何改动**。
> 所有条目均已核实发表出处（ISCA Archive / DBLP / OpenAlex / arXiv / 出版社 DOI）。引用数为 OpenAlex 口径，Google Scholar 通常更高。
> 标记说明：【在用】= 你当前流水线正在使用的模型/工具；【需标注数据】= 方法需要带标注的训练数据，只能理解、不宜直接上线；【趋势参考】= 预印本/新方向，未定论。

---

## 一、第一梯队：c1 核心改进的五大来源（推荐阅读顺序）

### 1. Turn-to-Diarize（ICASSP 2022）⭐ 对应你 c1 的"换人边界细化"
- **Title**: Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection
- **Authors**: Wei Xia, Han Lu, Quan Wang, Anshuman Tripathi, Yiling Huang, Ignacio López Moreno, Haşim Sak（Google）
- **Link**: <https://doi.org/10.1109/ICASSP43922.2022.9746531>（arXiv:2109.11641）· 49 引
- **借鉴点**：你 c1 的"0.6s 窗 / 0.15s 步长质心比较"本质是**启发式换人检测**。此文的结论是：换人/说话人轮转（Speaker Turn Detection, TDT）可以被**建模为可学习目标**，且能用 **ASR 输出（词/句时间戳）约束换人边界**。可落地思路：把细化路径改为"候选换人点 + 打分"（句子边界、标点、停顿先验），或研究其"说话人轮转数 ≤2 的约束"对你双人通话的适用性。它还支持流式——你的通话场景是流式的，值得关注。

### 2. TS-VAD（INTERSPEECH 2020）⭐ 对应你"段级质心"的升级路线
- **Title**: Target-Speaker Voice Activity Detection: A Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario
- **Authors**: Ivan Medennikov, Maxim Korenevsky, Tatiana Prisyach, Yuri Khokhlov, Mariya Korenevskaya, Ivan Sorokin, Tatiana Timofeeva, Anton Mitrofanov, Andrei Andrusenko, Ivan Podluzhny, Aleksandr Laptev, Aleksei Romanenko（STC 语音）
- **Link**: <https://doi.org/10.21437/Interspeech.2020-1602> · 197 引 · pp.274–278
- **借鉴点**：三段式 = ① 聚类得到每说话人 profile（i-vector/嵌入）→ ② 带说话人 profile 的**逐帧二分打分** → ③ 迭代重估。c1 的"段级质心比较"是它第②步的简化版。可落地思路：质心细化升级为"帧级打分 + 后处理平滑"，并注意它对**重叠说话人**天然有优势（你的质心法在抢话时会失效）。⚠️ 需要说话人 profile 标注/初始化，注意与你的"无标注"约束的差异。

### 3. VBx（Computer Speech & Language 2022）⭐ 可直接 A/B 的聚类式 SOTA
- **Title**: Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: Theory, implementation and analysis on standard tasks
- **Authors**: Federico Landini, Ján Profant, Mireia Díez, Lukáš Burget（Brno UT Speech@FIT + Phonexia）
- **Link**: <https://doi.org/10.1016/j.csl.2021.101254> · 15 引（OpenAlex 低估，行业标准工具）
- **借鉴点**："嵌入序列 → HMM + 变分贝叶斯**帧级重分割**"的两段式，比"聚类 + 启发式滑窗细化"更 principled，还有**说话人数自估计**。可落地思路：与当前 c1 的 AHC/细化做同数据 A/B（你的 `c1_comparison` 流程可直接复用），是"聚类法天花板"的参照物。官方开源：`https://github.com/BUTSpeechFIT/VBx`。

### 4. 双人电话语音的对照实验综述（CSL 2023）⭐ 与你数据同构的选型依据
- **Title**: An experimental review of speaker diarization methods with application to two-speaker conversational telephone speech recordings
- **Authors**: Luca Serafini, Samuele Cornell, Giovanni Morrone, Enrico Zovato, Alessio Brutti, Stefano Squartini（UNIVPM + PerVoice + FBK）
- **Link**: <https://doi.org/10.1016/j.csl.2023.101534> · 18 引
- **借鉴点**：在**双人电话对话**（与你的 Device116 通话、Device118 录音同构）上系统对比了聚类式（x-vector/ECAPA + AHC、VBx 等）、EEND、TS-VAD 等方法。可落地思路：让 Claude 依据此文的方法矩阵，给出 c1 当前设计（ASR 段级质心 + 启发式细化）在"无标注"约束下的**合理性推导**，并引用其结论支撑未来选型（例如：电话场景下聚类式 + 简单重分割往往是性价比最优）。

### 5. ECAPA-TDNN Embeddings for Speaker Diarization（INTERSPEECH 2021）⭐ 对应你的短答塌缩
- **Title**: ECAPA-TDNN Embeddings for Speaker Diarization
- **Authors**: Nauman Dawalatabad, Mirco Ravanelli, François Grondin, Jenthe Thienpondt, Brecht Desplanques, Hwidong Na
- **Link**: <https://doi.org/10.21437/Interspeech.2021-941> · 97 引 · pp.3560–3564
- **借鉴点**：专门研究**短段嵌入**在 diarization 中的鲁棒性（分段长度、嵌入归一化、批内数据增强——把同一信号的多种污染版本拼接同批训练）。可落地思路：短答（"支付宝""哦""谢谢"）被并掉的原因之一就是短段质心不可靠；可据此增加"长度加权质心 / 短段阈值 / 嵌入归一化"处理，并检查 CAM++ 段级向量输出的归一化方式。

---

## 二、按 c1 模块对照（第二梯队）

### 2.1 前置：环境检测与降噪
| 论文 | 出处 | 说明 |
|---|---|---|
| [FRCRN: Boosting Feature Representation using Frequency Recurrence for Monaural Speech Enhancement](https://arxiv.org/abs/2206.07293)【在用】 | ICASSP 2022（DNS Challenge 冠军） | 你正在用的 `speech_frcrn_ans_cirm_16k`；理解 CIRM 复比掩码与频带递归，可指导降噪开关阈值（-70dB 噪声底）与降噪强度 |
| [3D-Speaker: A Large-Scale Multi-Device, Multi-Distance, Multi-Dialect Corpus](https://huggingface.co/papers/2306.15354)【在用/语料】 | arXiv 2023 | 你的 CAM++ 出自 3D-Speaker 项目；语料覆盖多距离/多方言，解释为什么 1m/1.5m/2m 分档测试合理 |
| [Fun-ASR Technical Report](https://arxiv.org/abs/2509.12508)【在用/工具】 | arXiv 2025（技术报告） | 你整条 FunASR 工具链（VAD/ASR/spk 模式/标点）的官方说明；非正式论文，仅作工程参考 |

### 2.2 说话人嵌入
| 论文 | 出处 | 说明 |
|---|---|---|
| [CAM++ (Context-Aware Masking)](https://doi.org/10.21437/Interspeech.2023-1513)【在用】 | INTERSPEECH 2023 · 138 引 | 你正在用的嵌入模型原始论文；上下文感知掩码本身就是应对**变长/短段**设计的——调 `spk` 参数前先读它 |
| [ECAPA-TDNN](https://doi.org/10.21437/interspeech.2020-2650) | INTERSPEECH 2020 · **1502 引** | 嵌入事实标准；若换嵌入做 A/B，首选 |
| [X-Vectors: Robust DNN Embeddings](https://doi.org/10.1109/icassp.2018.8461375) | ICASSP 2018 · **2711 引** | 一切现代聚类式 diarization 的起点；理解 VBx/聚类法的术语体系必读 |

### 2.3 聚类与边界（EEND 路线，理解上限用）
| 论文 | 出处 | 说明 |
|---|---|---|
| [EEND with Self-Attention](https://doi.org/10.1109/asru46091.2019.9003959) | ASRU 2019 · 210 引 | EEND 开山之作；理解"端到端 vs 聚类"分水岭 |
| [Encoder-Decoder Based Attractors (EEND-EDA)](https://doi.org/10.1109/TASLP.2022.3162080) | **IEEE TASLP** 2022 · 61 引 | 解决"说话人数不确定"，EEND 路线期刊代表作；对应你"塌缩救援/人数估计"问题的学术解 |
| [Advances in Integration of EEND and Clustering-Based Diarization (EEND-VC)](https://doi.org/10.21437/Interspeech.2021-1004) | INTERSPEECH 2021 · pp.3565–3569 | Kinoshita, Delcroix, Tawara（NTT）；EEND+聚类混合，处理重叠 + 长录音。⚠️（更正：此前口头提到的"End-to-End Speaker Diarization with Vector Clustering"标题有误，应以本条目为准） |

### 2.4 场景特化：远场 / 双人 / 域失配
| 论文 | 出处 | 说明 |
|---|---|---|
| [NPU Speaker Verification System for INTERSPEECH 2020 FFSVC](https://doi.org/10.21437/Interspeech.2020-2688) | INTERSPEECH 2020 · pp.3471–3475 | Li Zhang, Jian Wu, Lei Xie（西工大 ASLP，与你是同一谱系）；task1/task3 第一/第二。直接对应你的 **1m/1.5m/2m + 短句** 问题：ResNet-BAM、域对抗训练（DAT）、数据增广、数据选择——四条线索都可借鉴 |
| [Multi-Level Transfer Learning from Near-Field to Far-Field Speaker Verification](https://doi.org/10.21437/Interspeech.2021-1980) | INTERSPEECH 2021 · pp.1094–1098 | Zhang, Wang, Lee, Xie, Li（同门，NUS 合作）；教师-学生 + 对比损失学**域不变嵌入**——若升级嵌入做远场鲁棒性的直接起点 |
| [Leveraging Speaker Embeddings in EEND for Two-Speaker Scenarios](https://doi.org/10.21437/Odyssey.2024-16) | **Odyssey 2024**（ISCA 说话人识别工作坊） | Álvarez-Trejos, Labrador, Lozano-Díez（UAM 马德里）。⚠️（更正：此前口头误标为 Serafini 团队，实际作者以本条为准）；两说话人场景 = 你的通话场景，EEND+嵌入的天花板参照 |

### 2.5 管线与时间戳
| 论文 | 出处 | 说明 |
|---|---|---|
| [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://doi.org/10.21437/interspeech.2023-78) | INTERSPEECH 2023 · 274 引 | Bain, Huh, Han, Zisserman（牛津 VGG）；**强制对齐**（wav2vec2 CTC）给出词级时间戳——你的 sentence timestamp 精度不足是换人边界细化的痛点，其"VAD→ASR→对齐"架构值得抄 |

### 2.6 重叠语音（可选；⚠️ 需标注数据，建议只做理解与对照）
| 论文 | 出处 | 说明 |
|---|---|---|
| [SOT: Serialized Output Training for End-to-End Overlapped Speech Recognition](https://doi.org/10.21437/Interspeech.2020-999) | INTERSPEECH 2020 · 62 引 | Kanda 等（微软）；重叠语音的 ASR 序列化输出——若通话出现抢话、漏识别，这是学术正解 |
| 分离经典（若上 TS-SEP）：[PIT](https://arxiv.org/abs/1607.00325)（ICASSP 2017）、[Conv-TasNet](https://doi.org/10.1109/TASLP.2019.2915167)（TASLP 2019）、[DPRNN](https://doi.org/10.1109/ICASSP40776.2020.9054266)（ICASSP 2020）、[SepFormer](https://arxiv.org/abs/2010.13154)（ICASSP 2021） | 顶级会议/期刊 | 对应你环境清单里"缺失的 TS-SEP"；FunASR `speech_tsse_*` 目标说话人提取的学术源头为 ICASSP 2018 [SpeakerBeam](https://doi.org/10.1109/ICASSP.2018.8462661) 一类工作 |

### 2.7 评估
| 论文 | 出处 | 说明 |
|---|---|---|
| [The Second DIHARD Diarization Challenge](https://doi.org/10.21437/interspeech.2019-1268) | INTERSPEECH 2019 | DER / 漏检 / 虚警的标准定义与评测流程（pyannote.metrics 实现）；你目前用文本代理指标，Claude 可据此设计"边界 ±0.5s"的补充指标 |
| [A Review of Speaker Diarization: Recent Advances with Deep Learning](https://doi.org/10.1016/j.csl.2021.101317) | Computer Speech & Language 2022（arXiv:2101.09624） | 当前最权威综述；用于建立全局术语与分类框架 |

### 2.8 趋势参考（可选，2024–2025）
| 论文 | 出处 | 说明 |
|---|---|---|
| [DiarizationLM: Speaker Diarization Post-Processing with LLMs](https://arxiv.org/abs/2401.03506) | arXiv 2024（微软） | LLM 后处理修正 diarization 输出；对你们"标签映射错误"类问题有启发，但需大模型部署 |
| [LLM-based speaker diarization correction](https://doi.org/10.1016/j.specom.2025.103224) | Speech Communication 2025 | 同上方向的期刊正式版 |
| [SEAL: Speaker Error Correction using Acoustic-conditioned LLMs](https://arxiv.org/abs/2501.08421) | arXiv 2025 | 利用声学条件 + LLM 修正说话人标签 |

---

## 三、给 Claude 的落地清单（5 篇 × 1 个动作）

1. **Turn-to-Diarize** → 把"0.6s/0.15s 启发式细化"改造成"候选换人点（ASR 句边界/标点/停顿）+ 打分"的算法框架，并与现版做同数据对比。
2. **TS-VAD** → 梳理"聚类 profile → 逐帧打分 → 重估"的升级路线图，说明 c1 质心法在**重叠语音**下的理论失效点。
3. **VBx** → 在 23 个原始音频上跑一次 VBx 官方实现，与 c1 的 282/373（文本匹配）、250/373（带标签）做同口径对比；若提升有限，说明原因（代理指标、中文、远场）。
4. **Serafini CSL 2023** → 产出"为什么 c1 当前设计是合理的（无标注约束下）"的推导，包含选型对比表。
5. **ECAPA-diar** → 实现/评估"长度加权质心 + 嵌入归一化"两个小改动，看 12329（支付宝/哦/谢谢）类短答是否回升。

## 四、诚实边界（写进 Claude 的系统提示）

- **需标注数据的方法**（EEND 全家、TS-VAD 训练、SOT、SepFormer 等）：本项目无标注能力，**只作为理解与对照**，不作为上线方案。
- **评估**：截至目前 c1 只有文本代理指标（SequenceMatcher ≥ 0.35，282/373 与 250/373），**不是 DER**，且样本参与了诊断、不代表独立泛化；任何"超越 DSH"的结论均不成立，直到用 DER/独立测试验证。
- **引用数**：OpenAlex 口径；Google Scholar 通常高 1.5–3 倍。

## 五、BibTeX（核心条目，作者/DOI 均已核对）

```bibtex
@inproceedings{xia2022turntodiarize,
  title     = {Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection},
  author    = {Wei Xia and Han Lu and Quan Wang and Anshuman Tripathi and Yiling Huang and Ignacio L{\'o}pez Moreno and Ha{\c{s}}im Sak},
  booktitle = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2022},
  doi       = {10.1109/ICASSP43922.2022.9746531}
}

@inproceedings{medennikov2020tsvad,
  title     = {Target-Speaker Voice Activity Detection: A Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario},
  author    = {Ivan Medennikov and Maxim Korenevsky and Tatiana Prisyach and Yuri Khokhlov and Mariya Korenevskaya and Ivan Sorokin and Tatiana Timofeeva and Anton Mitrofanov and Andrei Andrusenko and Ivan Podluzhny and Aleksandr Laptev and Aleksei Romanenko},
  booktitle = {Interspeech 2020},
  pages     = {274--278},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-1602}
}

@article{landini2022vbx,
  title     = {Bayesian {HMM} clustering of x-vector sequences ({VBx}) in speaker diarization: Theory, implementation and analysis on standard tasks},
  author    = {Federico Landini and J{\'a}n Profant and Mireia D{\'i}ez and Luk{\'a}{\v{s}} Burget},
  journal   = {Computer Speech \& Language},
  volume    = {71},
  pages     = {101254},
  year      = {2022},
  doi       = {10.1016/j.csl.2021.101254}
}

@article{serafini2023review,
  title     = {An experimental review of speaker diarization methods with application to two-speaker conversational telephone speech recordings},
  author    = {Luca Serafini and Samuele Cornell and Giovanni Morrone and Enrico Zovato and Alessio Brutti and Stefano Squartini},
  journal   = {Computer Speech \& Language},
  volume    = {78},
  pages     = {101534},
  year      = {2023},
  doi       = {10.1016/j.csl.2023.101534}
}

@inproceedings{dawalatabad2021ecapa,
  title     = {{ECAPA-TDNN} Embeddings for Speaker Diarization},
  author    = {Nauman Dawalatabad and Mirco Ravanelli and Fran{\c{c}}ois Grondin and Jenthe Thienpondt and Brecht Desplanques and Hwidong Na},
  booktitle = {Interspeech 2021},
  pages     = {3560--3564},
  year      = {2021},
  doi       = {10.21437/Interspeech.2021-941}
}

@inproceedings{wang2023campp,
  title     = {{CAM++}: A Fast and Efficient Network for Speaker Verification Using Context-Aware Masking},
  author    = {Hui Wang and Siqi Zheng and Yafeng Chen and Luyao Cheng and Qian Chen},
  booktitle = {Interspeech 2023},
  pages     = {5301--5305},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-1513}
}

@inproceedings{desplanques2020ecapa,
  title     = {{ECAPA-TDNN}: Emphasized Channel Attention, Propagation and Aggregation in {TDNN} Based Speaker Verification},
  author    = {Brecht Desplanques and Jenthe Thienpondt and Kris Demuynck},
  booktitle = {Interspeech 2020},
  pages     = {3830--3834},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-2650}
}

@inproceedings{snyder2018xvectors,
  title     = {{X-vectors}: Robust {DNN} Embeddings for Speaker Recognition},
  author    = {David Snyder and Daniel Garcia-Romero and Gregory Sell and Daniel Povey and Sanjeev Khudanpur},
  booktitle = {2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {5329--5333},
  year      = {2018},
  doi       = {10.1109/ICASSP.2018.8461375}
}

@inproceedings{fujita2019eend,
  title     = {End-to-End Neural Speaker Diarization with Self-Attention},
  author    = {Yusuke Fujita and Naoyuki Kanda and Shota Horiguchi and Kenji Nagamatsu},
  booktitle = {2019 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},
  pages     = {296--303},
  year      = {2019},
  doi       = {10.1109/ASRU46091.2019.9003959}
}

@article{horiguchi2022eda,
  title     = {Encoder-Decoder Based Attractors for End-to-End Neural Diarization},
  author    = {Shota Horiguchi and Yusuke Fujita and Shinji Watanabe and Yawen Xue and Paola Garcia},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume    = {30},
  pages     = {1493--1507},
  year      = {2022},
  doi       = {10.1109/TASLP.2022.3162080}
}

@inproceedings{kinoshita2021eendvc,
  title     = {Advances in Integration of End-to-End Neural and Clustering-Based Diarization for Real Conversational Speech},
  author    = {Keisuke Kinoshita and Marc Delcroix and Naohiro Tawara},
  booktitle = {Interspeech 2021},
  pages     = {3565--3569},
  year      = {2021},
  doi       = {10.21437/Interspeech.2021-1004}
}

@inproceedings{bain2023whisperx,
  title     = {{WhisperX}: Time-Accurate Speech Transcription of Long-Form Audio},
  author    = {Max Bain and Jaesung Huh and Tengda Han and Andrew Zisserman},
  booktitle = {Interspeech 2023},
  pages     = {4483--4487},
  year      = {2023},
  doi       = {10.21437/Interspeech.2023-78}
}

@inproceedings{kanda2020sot,
  title     = {Serialized Output Training for End-to-End Overlapped Speech Recognition},
  author    = {Naoyuki Kanda and Yashesh Gaur and Xiaofei Wang and Zhong Meng and Takuya Yoshioka},
  booktitle = {Interspeech 2020},
  pages     = {2797--2801},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-999}
}

@inproceedings{ryant2019dihard2,
  title     = {The Second {DIHARD} Diarization Challenge: Dataset, Task, and Baselines},
  author    = {Neville Ryant and Kenneth Church and Christopher Cieri and Alejandrina Cristia and Jun Du and Sriram Ganapathy and Mark Liberman},
  booktitle = {Interspeech 2019},
  pages     = {756--760},
  year      = {2019},
  doi       = {10.21437/Interspeech.2019-1268}
}

@inproceedings{zhang2020ffsvc,
  title     = {{NPU} Speaker Verification System for {INTERSPEECH} 2020 Far-Field Speaker Verification Challenge},
  author    = {Li Zhang and Jian Wu and Lei Xie},
  booktitle = {Interspeech 2020},
  pages     = {3471--3475},
  year      = {2020},
  doi       = {10.21437/Interspeech.2020-2688}
}

@inproceedings{zhang2021transfer,
  title     = {Multi-Level Transfer Learning from Near-Field to Far-Field Speaker Verification},
  author    = {Li Zhang and Qing Wang and Kong Aik Lee and Lei Xie and Haizhou Li},
  booktitle = {Interspeech 2021},
  pages     = {1094--1098},
  year      = {2021},
  doi       = {10.21437/Interspeech.2021-1980}
}

@inproceedings{alvareztrejos2024twospk,
  title     = {Leveraging Speaker Embeddings in End-to-End Neural Diarization for Two-Speaker Scenarios},
  author    = {Juan Ignacio {\'A}lvarez-Trejos and Beltr{\'a}n Labrador and Alicia Lozano-D{\'i}ez},
  booktitle = {Odyssey 2024: The Speaker and Language Recognition Workshop},
  year      = {2024},
  doi       = {10.21437/Odyssey.2024-16}
}

@article{park2022review,
  title     = {A review of speaker diarization: Recent advances with deep learning},
  author    = {Tae Jin Park and others},
  journal   = {Computer Speech \& Language},
  volume    = {72},
  pages     = {101317},
  year      = {2022},
  doi       = {10.1016/j.csl.2021.101317}
}

@article{luo2019convtasnet,
  title     = {{Conv-TasNet}: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation},
  author    = {Yi Luo and Nima Mesgarani},
  journal   = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume    = {27},
  number    = {8},
  pages     = {1256--1266},
  year      = {2019},
  doi       = {10.1109/TASLP.2019.2915167}
}

@inproceedings{yu2017pit,
  title     = {Permutation Invariant Training of Deep Models for Speaker-Independent Multi-talker Speech Separation},
  author    = {Dong Yu and Morten Kolb{\ae}k and Zheng-Hua Tan and Jesper Jensen},
  booktitle = {2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {241--245},
  year      = {2017},
  eprint    = {1607.00325},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}

@inproceedings{luo2020dprnn,
  title     = {Dual-Path {RNN}: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation},
  author    = {Yi Luo and Zhuo Chen and Takuya Yoshioka},
  booktitle = {ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages     = {46--50},
  year      = {2020},
  doi       = {10.1109/ICASSP40776.2020.9054266}
}

@inproceedings{subakan2021sepformer,
  title     = {Attention is All You Need in Speech Separation},
  author    = {Cem Subakan and Mirco Ravanelli and Samuele Cornell and Mirko Bronzi and Jianyuan Zhong},
  booktitle = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2021},
  eprint    = {2010.13154},
  archivePrefix = {arXiv},
  primaryClass = {eess.AS}
}
```

> 注：个别条目（park2022review）为预校作者简写，正式投稿前请按 DOI 页面核对完整作者列表；`pages` 字段为常见出版页码，若与最终版本不符请以出版社为准。
