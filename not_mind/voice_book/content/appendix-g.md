# 附录 G 网上调研与参考资料

本附录记录本书写作时使用的公开资料，并区分“已打开核验”和“仅取得入口、仍待复核”。访问日期为 2026-08-07。论文中的 benchmark 数字只在论文给出的数据、模型和评测设置下成立；官方文档中的版本、硬件支持和 API 会变化，使用前应重新打开链接核验。

## 论文与数据

1. [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)。论文摘要说明卷积用于局部模式、Transformer 用于全局交互，并给出 LibriSpeech 结果；本书用它解释架构能力与部署边界的区别。
2. [wav2vec 2.0](https://arxiv.org/abs/2006.11477)。论文讨论在潜空间遮盖并做对比学习，以及少量标注下的表现；本书用它说明自监督降低标注需求但不消除领域偏移。
3. [HuBERT](https://arxiv.org/abs/2106.07447)。论文描述离线聚类和 masked prediction；本书用它说明隐藏单元、聚类一致性和下游微调。
4. [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)。Whisper 论文报告大规模多语种、多任务弱监督；本书把它作为鲁棒性和二次开发的讨论对象，不把零样本结果当成端侧保证。
5. [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)。RNN Transducer 的原始论文；本书用它解释单调流式输出和二维时间/标签历史。
6. [WeNet: Production oriented Streaming and Non-streaming End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2102.01547)。论文摘要描述 dynamic chunk attention、hybrid CTC/attention 和两遍解码；本书用它解释“生产导向”的流式折中。
7. [WenetSpeech](https://arxiv.org/abs/2110.03370)。论文给出多领域中文语料的高质量、弱标注和未标注规模，以及互联网/会议测试集；本书保留其数据规模的原始语境。
8. [GigaSpeech](https://arxiv.org/abs/2106.06909)。论文描述多领域英文语料、强弱标注和过滤/对齐流程；本书用它说明数据规模与质量控制必须同时报告。
9. [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)。Graves 等人在 ICML 2006 的原始论文；本书用 blank、路径坍缩和无分段标签训练的基本思想。

## 官方工程文档

10. [WebRTC Audio Processing Module](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/g3doc/audio_processing_module.md)。官方文档把 AEC、NS、AGC 作为 APM 的典型语音增强效果，并说明可嵌入 WebRTC 管线或独立使用。
11. [ONNX Runtime: Quantize ONNX Models](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)。官方文档说明 8-bit 线性量化、scale/zero point、QOperator/QDQ、动态/静态量化和预处理。
12. [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)。官方入口说明 TensorRT 面向 NVIDIA GPU 推理优化，支持多种精度、动态形状和性能测量；engine 必须绑定目标硬件和版本核验。
13. [NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)。官方文档说明模型仓库、协议、调度器、批处理、后端和模型管理 API；本书用它区分服务层和流式状态层。
14. [Arm CMSIS-NN](https://arm-software.github.io/CMSIS-NN/latest/)。官方手册说明面向 Cortex-M 的高效 kernel、不同指令集实现和 int8/int16 量化规范。
15. [Hugging Face Automatic Speech Recognition](https://huggingface.co/docs/transformers/tasks/asr)。官方任务文档提供数据加载、预处理、训练、评估和推理的统一入口；本书提醒默认 recipe 不等于流式和端侧方案。
16. [Kaldi documentation](https://kaldi-asr.org/doc/)。官方文档作为经典特征、声学模型和解码工具的入口；具体 recipe 和依赖版本需锁定。

## 课程组织参考（未实时打开核验）

17. [Microsoft: AI For Beginners](https://github.com/microsoft/AI-For-Beginners)。本书只借鉴“概念 → 小实验 → 复盘问题”的课程节奏，用它帮助读者从理解走到动手；没有把该仓库的正文、代码、图像或练习答案当作本书素材。当前写作环境未能实时打开 GitHub 仓库页面，因此这里不对其当前默认分支、提交版本或许可证作断言。再次分发本书前，必须重新打开该 URL，核对仓库版本、许可证及所引用资源的单独许可。

## 资料使用声明

用户提供的岗位汇总文件是本书的需求来源，不是公开技术标准。岗位中的公司、地点、薪资、发布日期和“加分项”会变化；本书只把其中反复出现的能力转成学习与验收任务，不对招聘结果作承诺。项目仓库、模型卡和数据许可可能因代理、登录或版本变动而无法在当前环境直接访问，遇到这种情况应以官方发行页、许可证和本地实测为准。
