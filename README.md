# Awesome RGB-T Fusion ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg) ![-RGBT-red](https://user-images.githubusercontent.com/38373305/205479612-e61d11b4-6c3e-4eb0-8d19-2e7170eb7783.svg)
A collection of deep learning based RGB-T-Fusion methods, codes, and datasets.  
The main directions involved are Multispectral Pedestrian Detection, RGB-T Aerial Object Detection, RGB-T Semantic Segmentation, RGB-T Salient Object Detection, RGB-T Crowd Counting, RGB-T Fusion Tracking.  

Welcome to add valuable papers and codes, feel free to star and contact me. Keep updating....🚀

<h1>🔥 News</h1>
<div style="height: 100px; overflow: auto;">
<li>
    [10/2026]: We have updated this repository with new papers accepted at prestigious conferences.
</li>

</div>
<!-- <div style="margin-top: 5px; font-size: small; margin-bottom: 0px;">⬆ scrollable</div> -->

<!-- <details>
<summary>Some News: 🆕</summary>

🚀 **2025.08.18 A major update to this repository.** <br>

💎 **2025.07.04 Add one our paper in RGB-T Aerial Object Detection.** <br>

👀 **2025.04.01 Add one dataset in RGB-T-Aerial-Object-Detection (RGBT-Tiny).** <br>

💎 **2025.02.17 Add one dataset-SMOD and some papers in Multispectral Pedestrian Detection.** <br>

👀 **2025.02.12 Add one dataset-MFAD in Multispectral Pedestrian Detection.** <br>

👀 **2024.12.23 Add one TPAMI paper in RGB-T Salient Object Detection.** <br>

💎 **2024.10.31 Add one our paper in Pixel-level Fusion for Detection.** <br>

👀 **2024.10.19 Add one dataset in Multispectral Pedestrian Detection.** <br>

💎 **2024.06.24 Add one our paper and one CVPR paper.** <br>

👀 **2024.05.23 Add one dataset in RGB-T Aerial Object Detection.** <br>

💎 **2024.04.23 Add more papers about RGB-T Salient Object Detection.** <br>

👀 **2024.03.17 Add one CVPR paper.** <br>

💎 **2024.03.15 Add new content about RGB-T Semantic segmentation.** <br>

👀 **2024.03.12 Add one our paper and one CVPR paper.** <br>

</details> -->

## Contents  

1. [Multispectral Pedestrian Detection](#Multispectral-Pedestrian-Detection)
2. [RGB-T Aerial Object Detection](#RGB-T-Aerial-Object-Detection)
3. [RGB-T Semantic Segmentation](#RGB-T-Semantic-Segmentation)
4. [RGB-T Salient Object Detection](#RGB-T-Salient-Object-Detection)
5. [RGB-T Crowd Counting](#RGB-T-Crowd-Counting)
6. [RGB-T Fusion Tracking](#RGB-T-Fusion-Tracking)

--------------------------------------------------------------------------------------
# Multispectral Pedestrian Detection
## Datasets and Tools
|Dataset                        | Years |Modality     |Images| Classes|
|-------------------------------|-------|-------------|------|--------|
|[KAIST](https://soonminhwang.github.io/rgbt-ped-detection/)|2015|RGB, LWIR|95K|1|
|[CVC-14](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/)|2016|RGB, LWIR|8K|1|
|[FLIR](https://www.flir.cn/oem/adas/adas-dataset-form/)|2018|RGB, LWIR|20K|5|
[FLIR-aligned](https://github.com/zonaqiu/FLIR-align)|2020|RGB, LWIR|5K|3|
|[Utokyo](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)|2017|RGB, NIR, MIR, FIR|7K|5|
|[LLVIP](https://bupt-ai-cz.github.io/LLVIP/)|2021| RGB, LWIR|15K|1|
|[M<sup>3</sup>FD](https://github.com/dlut-dimt/TarDAL)|2022|RGB, LWIR|4K|6|
|[Multi-Spectral Stereo](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset)|2023|RGB, NIR, LWIR|195K|4|
|[SMOD](https://www.kaggle.com/datasets/zizhaochen6/sjtu-multispectral-object-detection-smod-dataset)|2024|RGB, LWIR|8K|4|
|[MMPD](https://github.com/jin-s13/MMPD-Dataset)|2024|RGB, LWIR|260K|1|
|[InfraParis](https://ensta-u2is-ai.github.io/infraParis/)|2024| RGB, LWIR, Depth|7K|19|
|[MFAD](https://github.com/hukefy/EI2Det)|2025|RGB, LWIR|12K|6|
|[MMVIP](https://github.com/yyppptjr/MMVIP)|2026|RGB, LWIR|128K|2|
|[RGBT-Ground](https://github.com/crazyxiaoxi/RGBTVG)|2026|RGB, LWIR|40K|5|

<!-- |[MMPD](https://github.com/jin-s13/MMPD-Dataset)|2024|RGB, LWIR|1200+|1| -->
<!-- [Multi-Spectral Stereo](https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset), -->


### KAIST Tools
- Improved KAIST Testing Annotations provided by Liu et al.[download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
- Sanitized KAIST Training Annotations provided by Li et al.[download](https://github.com/Li-Chengyang/MSDS-RCNN) 
- Improved KAIST Training Annotations provided by Zhang et al.[download](https://github.com/luzhang16/AR-CNN) 
- Evalutaion codes.[download](https://github.com/CalayZhou/MBNet/tree/master/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)
- Evaluation [Website](https://eval.ai/web/challenges/challenge-page/1247/leaderboard/3137).

## Papers

### Vision-Language Model Perception 
1. MLLM Makes Strong Backbone for Multi-Modal Object Detection, Neurips 2026, Yifeng Yang et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11133673)]
2. Does Seeing More Mean Knowing More? Mono-Anchored Advantage Normalization for Multi-Source Visual Reasoning, Neurips 2026, Fanhu Zeng et al., [[PDF](https://arxiv.org/pdf/2605.25437)][[Code](https://github.com/AI9Stars/MARS)]
3. RGBT-GroundBench: Visual Grounding Beyond RGB in Complex Real-World Scenarios, ECCV 2026, Tianyi Zhao et al., [[PDF](https://link.springer.com/chapter/10.1007/978-3-032-37422-6_24)][[Code](https://github.com/crazyxiaoxi/RGBT-GroundBench)]

### Pretrained Generalist Model
1. LoME: LoRA-Driven Multimodal Extractor for  RGB-X Vision Tasks, TCSVT 2026, Yusi Zhang et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11133673)][[Code](https://github.com/zyszxhy/LoME)]
2. M-SpecGene: Generalized Foundation Model for RGBT Multispectral Vision, ICCV 2025, Kailai Zhou et al., [[PDF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_M-SpecGene_Generalized_Foundation_Model_for_RGBT_Multispectral_Vision_ICCV_2025_paper.html)][[Code](https://github.com/CalayZhou/M-SpecGene)]
3. UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning, ACM MM 2025, Maoxun Yuan et al., [[PDF](https://arxiv.org/abs/2404.17360)][[Code](https://github.com/PoTsui99/UniRGB-IR)]
4. When Pedestrian Detection Meets Multi-Modal Learning: Generalist Model and Benchmark Dataset, ECCV 2024, Yi Zhang et al., [[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06487.pdf)][[Code](https://github.com/BubblyYi/MMPedestron)]
5. Mixed Patch Visible-Infrared Modality Agnostic Object Detection, WACV 2025, Heitor R. Medeiros et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10943701)][[Code](https://github.com/heitorrapela/MiPa)]

### Modality Bias
1. AIR-Det: Asymmetric Multimodal Object Detection via Main-Conditioned Residual Learning, ACMMM2026, Ziang Gan et al., [[PDF]()]
2. Distribution-Aligned Multimodal Fusion for Robust Object Detection, CVPR 2026, Xiaohui Hao et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Hao_Distribution-Aligned_Multimodal_Fusion_for_Robust_Object_Detection_CVPR_2026_paper.pdf)]
3. UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning, ACM MM 2025, Maoxun Yuan et al., [[PDF](https://arxiv.org/abs/2404.17360)][[Code](https://github.com/PoTsui99/UniRGB-IR)]
4. Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge, ECCV 2024, Medeiros, Heitor Rapela et al., [[PDF](https://eccv.ecva.net/virtual/2024/poster/816)][[Code](https://github.com/heitorrapela/ModTr)]
5. AMFD: Distillation via Adaptive Multimodal Fusion for Multispectral Pedestrian Detection, Arxiv 2024, Zizhao Chen et al., [[PDF](https://arxiv.org/pdf/2405.12944)][[Code](https://github.com/bigD233/AMFD)]
6. D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection, CVPR 2024, Dinh Phat Do et al., [[PDF](https://arxiv.org/pdf/2403.09359.pdf)][[Code](https://github.com/EdwardDo69/D3T)]
7. Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection, CVPR 2024, Taeheon Kim et al. [[PDF](https://arxiv.org/pdf/2403.01300.pdf)][[Code](https://github.com/ssbin0914/Causal-Mode-Multiplexer)]
8. HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information, WACV 2024, Medeiros, Heitor Rapela et al., [[PDF](https://openaccess.thecvf.com/content/WACV2024/papers/Medeiros_HalluciDet_Hallucinating_RGB_Modality_for_Person_Detection_Through_Privileged_Information_WACV_2024_paper.pdf)][[Code](https://github.com/heitorrapela/HalluciDet)]
9. Unsupervised Domain Adaptation for Multispectral Pedestrian Detection, CVPR 2019 Workshop , Dayan Guan et al.[[PDF](https://arxiv.org/abs/1904.03692)][[Code](https://github.com/dayanguan/unsupervised_multispectral_pedestrian_detectio)]
10. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019, Y. Cao et al. Information Fusion 2019, [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)][[Code](https://github.com/Huaqing-lucky/unsupervised_multispectral_pedestrian_detection)]
11. Learning crossmodal deep representations for robust pedestrian detection, CVPR 2017, D. Xu et al.[[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)][[Code](https://github.com/danxuhk/CMT-CNN)]

### Pixel-level Fusion for Detection
1. PCDFusion: Proposal-Context-Detail Bayesian Rendering for Infrared-Visible Image Fusion, Neurips 2026, Rui Liu et al., [[PDF](https://neurips.cc/virtual/2026/poster/152664)]
2. RoES: Rotational Equivariant Selective-frequency Fusion for Multimodal Images, ACMMM 2026, Jiabao Wang, et al., [[PDF](https://arxiv.org/pdf/2609.12497)][[Code](https://github.com/BryceLosky/RoES-Fusion)]
3. P2Fusion: Prompt-based Progressive Infrared-Visible Image Fusion Via Dual-Prior Distillation, ECCV 2026, Yi Shi et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/5454)][[Code](https://github.com/YiShi99/P2Fusion)]
4. RegionFuse: Region-Adaptive Pixel Distribution Learning for Infrared and Visible Image Fusion, CVPR 2026, Jianghan Xia et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/39120)][[Code](https://github.com/DarkIceField/RegionFuse)]
5. Fusion in Your Way: Aligning Image Fusion with Heterogeneous Demands via Direct Preference Optimization, CVPR 2026, Zengyi Yang et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Su_Fusion_in_Your_Way_Aligning_Image_Fusion_with_Heterogeneous_Demands_CVPR_2026_paper.pdf)][[Code](https://github.com/suweijian1996/DPOFusion)]
6. Customized Fusion: A Closed-Loop Dynamic Network for Adaptive Multi-Task-Aware Infrared-Visible Image Fusion, CVPR 2026, Yiming Sun et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/oral/40295)][[Code](https://github.com/YR0211/CLDyN.git)]
7. CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion, AAAI 2026, Yiming Sun et al., [[PDF](https://arxiv.org/pdf/2601.08619)][[Code](https://github.com/Sevryy/CtrlFuse)]
8. DDFD: Diffusion-Based Denoising Fusion for Object Detection in Infrared-Visible Images, ACM MM 2025, Min Dang et al., [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3755183)]
9. LUT-Fuse: Towards Extremely Fast Infrared and Visible Image Fusion via Distillation to Learnable Look-Up Tables, ICCV 2025, Haodong Zhu et al., [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Yi_LUT-Fuse_Towards_Extremely_Fast_Infrared_and_Visible_Image_Fusion_via_ICCV_2025_paper.pdf)][[Code](https://github.com/zyb5/LUT-Fuse)]
10. DCEvo: Discriminative Cross-Dimensional Evolutionary Learning for Infrared and Visible Image Fusion, CVPR 2025, Jinyuan Liu et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_DCEvo_Discriminative_Cross-Dimensional_Evolutionary_Learning_for_Infrared_and_Visible_Image_CVPR_2025_paper.pdf)][[Code](https://github.com/Beate-Suy-Zhang/DCEvo)]
11. Task-driven Image Fusion with Learnable Fusion Loss, CVPR 2025, Haowen Bai et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Bai_Task-driven_Image_Fusion_with_Learnable_Fusion_Loss_CVPR_2025_paper.pdf)][[Code](https://github.com/HaowenBai/TDFusion)]
12. DANet: A Dual-Branch Framework With Diffusion-Integrated Autoencoder for Infrared–Visible Image Fusion, TIM 2025,  Chengyi Pan et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10919151)][[Code](https://github.com/Pancy9476/DANet)]
13. RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement, ICIP 2025, Raman Jha et al., [[PDF](https://arxiv.org/abs/2505.24705)][[Code](https://github.com/jhakrraman/rt-xnet)]
14. SFDFusion: An Efficient Spatial-Frequency Domain Fusion Network for Infrared and Visible Image Fusion, ECAI 2024, KunHu et al. [[PDF](https://arxiv.org/pdf/2410.22837)][[Code](https://github.com/lqz2/SFDFusion)]
15. E2E-MFD: Towards End-to-End Synchronous Multimodal Fusion Detection, Neurips 2024, Jiaqing Zhang et al. [[PDF](https://arxiv.org/abs/2403.09323)][[Code]( https://github.com/icey-zhang/EfficientMFD)]
16. Multi-modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion, ICCV 2023, Yiming Sun et al.[[PDF](https://arxiv.org/abs/2302.01392)][[Code]( https://github.com/SunYM2020/MoE-Fusion)]
17. MetaFusion : Infrared and Visible Image Fusion via Meta-Feature Embedding from Object Detection, CVPR 2023, Wenda Zhao et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.pdf)][[Code](https://github.com/wdzhao123/MetaFusion)]
18. Locality guided cross-modal feature aggregation and pixel-level fusion for multispectral pedestrian detection, Information Fusion 2022, Yanpeng Cao et al. [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253522000549)]
19. Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection, CVPR 2022, Jinyuan Liu et al.[[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.pdf)][[Code](https://github.com/dlut-dimt/TarDAL)]
20. DetFusion: A Detection-driven Infrared and Visible Image Fusion Network, ACM Multimedia 2022, Yiming Sun et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3503161.3547902)][[Code](https://github.com/SunYM2020/DetFusion)]

### Illumination Aware
1. Efficient Multispectral Object Detection with attentive feature aggregation leveraging zero-shot implicit illumination guidance, Information Fusion 2025, Zhongxia Xiong et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253525000120)]
2. EI<sup>2</sup>Det: Edge-Guided Illumination-Aware Interactive Learning for Visible-Infrared Object Detection, TCSVT 2025, Ke Hu et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10877920)] [[Code](https://github.com/hukefy/EI2Det)]
3. RGB-X Object Detection via Scene-Specific Fusion Modules, WACV 2024, Sri Aditya Deevi et al. [[PDF](https://arxiv.org/abs/2310.19372)] [[Code](https://github.com/dsriaditya999/RGBXFusion)]
4. Illumination-Guided RGBT Object Detection With Inter- and Intra-Modality Fusion, IEEE Transactions on Instrumentation and Measurement 2023, Yan Zhang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10057437/)][[Code](https://github.com/NNNNerd/Triple-I-Net-TINet)]
5. IGT: Illumination-guided RGB-T object detection with transformers, Knowledge-Based Systems 2023, Keyu Chen et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705123001739)]
6. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
7. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
8. Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection, Information Fusion 2019, Dayan Guan et al.[[PDF](https://arxiv.org/abs/1802.09972)][[Code](https://github.com/dayanguan/illumination-aware_multispectral_pedestrian_detection/)]
9.  Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection, Pattern Recognition 2018, Chengyang Li et al.[[PDF](https://www.sciencedirect.com/science/article/pii/S0031320318303030)][[Code](https://github.com/Li-Chengyang/IAF-RCNN)]


### Feature Alignment
1.  Contextually-Guided State Space Fusion for Misaligned Multi-Spectral Object Detection, ACMMM 2025, Guyue Jin et al., [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3754550)]
2.  CF-Deformable DETR: An End-to-End Alignment-Free Model for Weakly Aligned Visible-Infrared Object Detection,  IJCAI 2024, Haolong Fu et al. [[PDF](https://www.ijcai.org/proceedings/2024/0084.pdf)][[Code](https://github.com/116508/CF-Deformable-DETR)]
3. C<sup>2</sup>Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection, TGRS 2024, Maoxun Yuan et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10472947)][[Code](https://github.com/yuanmaoxun/C2Former)]
4. Cross-Modality Proposal-guided Feature Mining for Unregistered RGB-Thermal Pedestrian Detection, TMM 2024, Chao Tian et al. [[PDF](https://ieeexplore.ieee.org/document/10382506)]
5. Multi-Modal Feature Pyramid Transformer for RGB-Infrared Object Detection, TITS 2023, Yaohui Zhu et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10105844)]
6. Attentive Alignment Network for Multispectral Pedestrian Detection, ACM MM 2023, Nuo Chen et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3613444)]
7. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI2022, Jung Uk Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
8. Mlpd: Multi-label pedestrian detector in multispectral domain, RAL 2021, Jiwon Kim et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9496129)][[Code](https://github.com/sejong-rcv/MLPD-Multi-Label-Pedestrian-Detection)]
9. Weakly Aligned Feature Fusion for Multimodal Object Detection, TNNLS 2021, Lu Zhang et al. [[PDF](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)]
10. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
11. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al.[[PDF](https://arxiv.org/abs/1901.02645)][[Code](https://github.com/luzhang16/AR-CNN)]
12. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.[[PDF](https://arxiv.org/abs/1808.04818)][[Code](https://github.com/Li-Chengyang/MSDS-RCNN)]

### Modality Translation and  Knowledge Distillation 
1. InfraNet: Quality-Aware RGB Guidance for Infrared Object Detection, ECCV 2026, Zichao Feng et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/3873)]
2. Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing
Infrared, CVPR 2026, Yafei Zhang et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Missing_No_More_Dictionary-Guided_Cross-Modal_Image_Fusion_under_Missing_Infrared_CVPR_2026_paper.pdf)][[Code](https://github.com/harukiv/DCMIF)]
3. SAM-Guided Semantic Knowledge Fusion for Visible-Infrared Object Detection, ACM MM 2025,  Ting Li et al., [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3755718)][[Code]( https://github.com/liting1018/SemFusion)]
4. Multimodal Decomposed Distillation with Instance Alignment and Uncertainty Compensation for Thermal Object Detection, ACM MM 2025, Yanfeng Liu et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3755841)][[Code](https://github.com/lyf0801/DecomKD)]
5. Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning, ICCV 2025, Tianyi Zhao et al. [[PDF](https://arxiv.org/abs/2503.11780)][[Code](https://github.com/Zhao-Tian-yi/M2D-LIF)]
6. Pseudo Visible Feature Fine-Grained Fusion for Thermal Object Detection, CVPR 2025, Ting Li et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Pseudo_Visible_Feature_Fine-Grained_Fusion_for_Thermal_Object_Detection_CVPR_2025_paper.pdf)][[Code](https://github.com/liting1018/PFGF)]
7. Modality Translation for Object Detection Adaptation Without Forgetting Prior Knowledge, ECCV 2024, Heitor R. Medeiros et al.,[[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/12401.pdf)][[Code](https://github.com/heitorrapela/ModTr)]
8. TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation, ACM MM 2023, Zeyu Wang et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3581783.3613849)][[Code](https://github.com/zeyuwang-zju/TIRDet)]
9. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI 2022, Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
10. Robust Thermal Infrared Pedestrian Detection By Associating Visible Pedestrian Knowledge, ICASSP 2022, Sungjune Park et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9746886)]
11. Low-cost Multispectral Scene Analysis with Modality Distillation, Zhang Heng et al. [[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_Low-Cost_Multispectral_Scene_Analysis_With_Modality_Distillation_WACV_2022_paper.pdf)]
12. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
13. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, TCSCT 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]

### Fusion Architecture
1. DETR is Secretly a Multispectral Detector: Zero-Parameter Adaptation via Semantic Alignment, ECCV 2026, Xiangyang Li et al., [[PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/10379.pdf)][[Code](https://github.com/UserXiangYang/ZPA-MDETR)]
2. Efficient RGB-T Object Detection Via Sparse Cross-Modality Fusion, ECCV 2026, Chao Tian et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/4023)]
3. Bridging Multimodal Fusion and Expert Routing via Spectral Reliability Descriptors for Robust Object Detection, ACMMM 2026, Yefeng Wu, [[PDF](https://arxiv.org/pdf/2606.01173)]
4. Multispectral State-Space Feature Fusion: Bridging Shared and Cross-Parametric Interactions for Object Detection, Jifeng Shen et al. Information Fusion 2026, [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253525009571)][[Code](https://github.com/61s61min/MS2Fusion.git)]
5. IRDFusion: Iterative relation-map difference guided feature fusion for multispectral object detection, Jifeng Shen et al. Pattern Recognition 2026 [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320326001548)][[Code](https://github.com/61s61min/IRDFusion)]
6. Removal Then Selection: A Coarse-to-Fine Fusion Perspective for RGB-Infrared Object Detection, TITS 2025, Tianyi Zhao et al. [[PDF](https://ieeexplore.ieee.org/document/11278552)][[Code]( https://github.com/Zhao-Tian-yi/RSDet.git)][[知乎](https://zhuanlan.zhihu.com/p/679289837)]
7. WaveMamba: Wavelet-Driven Mamba Fusion for RGB-Infrared Object Detection, ICCV 2025,  Haodong Zhu et al., [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_WaveMamba_Wavelet-Driven_Mamba_Fusion_for_RGB-Infrared_Object_Detection_ICCV_2025_paper.pdf)][[Code](https://github.com/EhanDong/WaveMamba)]
8. Rethinking Early-Fusion Strategies for Improved Multispectral Object Detection, TIV 2025, Xue Zhang et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10681477)][[Code](https://github.com/XueZ-phd/Efficient-RGB-T-Early-Fusion-Detection)]
9. Specificity-Guided Cross-Modal Feature Reconstruction for RGB-Infrared Object Detection, TITS 2024,  Xiaoyu Sun et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10759534)][[Code](https://github.com/SXYSUOSUO/SCFR)]
10. DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion, ECCV 2024, Junjie Guo et al., [[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03984.pdf)][[Code](https://github.com/gjj45/DAMSDet)]
11. TFDet: Target-Aware Fusion for RGB-T Pedestrian Detection, TNNLS 2024, Xue Zhang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10645696)][[Code](https://github.com/XueZ-phd/TFDet.git)]
12. Frequency Mining and Complementary Fusion Network for RGB-Infrared Object Detection, GRSL 2024, Yangfeixiao Liu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10644032)]
13. M2FNet: Mask-guided Multi-level Fusion for RGB-T Pedestrian Detection, TMM 2024, Xiangyang Li et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10478590)]
14. ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection, Pattern Recognition 2024, Shen Jifeng et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320323006118)][[Code](https://github.com/chanchanchan97/ICAFusion)]
15. Improving RGB-infrared object detection with cascade alignment-guided transformer, Information Fusion 2024, Maoxun Yuan et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253524000241)]
16. Multispectral Object Detection via Cross-Modal Conflict-Aware Learning, ACM MM 2023, Xiao He et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3612651)][[Code](https://github.com/hexiao0275/CALNet-Dronevehicle)]
17. Stabilizing Multispectral Pedestrian Detection with Evidential Hybrid Fusion, TCSVT 2023, Li Qing et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10225383)]
18. Multimodal Object Detection by Channel Switching and Spatial Attention, CVPRW 2023, Yue Cao et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf)]
19. Multi-Modal Feature Pyramid Transformer for RGB-Infrared Object Detection, TITS 2023, Yaohui Zhu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10105844)]
20. Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection, TMM 2023, Ruimin Li et al. [[PDF](https://ieeexplore.ieee.org/document/10114594)][[Code](https://github.com/RimXidian/MCHE-CF-for-Multispectral-Pedestrian-Detection)]
21. HAFNet: Hierarchical Attentive Fusion Network for Multispectral Pedestrian Detection, Remote Sensing 2023, Peiran Peng et al. [[PDF](https://www.mdpi.com/2072-4292/15/8/2041)]
22. Multimodal Object Detection via Probabilistic Ensembling, ECCV2022, Yi-Ting Chen et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)][[Code](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)]
23. Learning a Dynamic Cross-Modal Network for Multispectral Pedestrian Detection, ACM Multimedia 2022, Jin Xie et al. [[PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3547895)]
24. Confidence-aware Fusion using Dempster-Shafer Theory for Multispectral Pedestrian Detection, TMM 2022, Qing Li et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9739079)]
25. Attention-Guided Multi-modal and Multi-scale Fusion for Multispectral Pedestrian Detection, PRCV 2022, Wei Bao et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-18907-4_30)]
26. Improving RGB-Infrared Pedestrian Detection by Reducing Cross-Modality Redundancy, ICIP2022, Qingwang Wang et al.  [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]
27. Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving, IEEE Transactions on Intelligent Transportation Systems, Kinjal Dasgupta et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9706418)]
28. Adopting the YOLOv4 Architecture for Low-LatencyMultispectral Pedestrian Detection in Autonomous Driving, Sensors 2022, Kamil Roszyk et al. [[PDF](https://www.mdpi.com/1424-8220/22/3/1082)]
29. Deep Active Learning from Multispectral Data Through Cross-Modality Prediction Inconsistency, ICIP2021, Heng Zhang et al.[[PDF](https://ieeexplore.ieee.org/document/9506322)]
30. Attention Fusion for One-Stage Multispectral Pedestrian Detection, Sensors 2021, Zhiwei Cao et al. [[PDF](https://www.mdpi.com/1424-8220/21/12/4184)]
31. Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, Jung Uk Kim et al. [[PDF](https://ieeexplore.ieee.org/document/9419080)]
32. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]
33. Guided Attentive Feature Fusion for Multispectral Pedestrian Detection, WACV 2021, Heng Zhang et al. [[PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)]
34. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
35. Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al. [[PDF](https://hal.archives-ouvertes.fr/hal-02872132/file/icip2020.pdf)]
36. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
37. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
38. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al. [[PDF](https://arxiv.org/abs/1901.02645)][[Code](https://github.com/luzhang16/AR-CNN)]
39. Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pesdestrian Detecion, ISPRS Journal of Photogrammetry and Remote Sensing 2019, Yanpeng Cao et al.[[PDF](https://arxiv.org/abs/1902.05291)][[Code](https://github.com/dayanguan/realtime_multispectral_pedestrian_detection)]
40. Cross-modality interactive attention network for multispectral pedestrian detection, Information Fusion 2019, Lu Zhang et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253518304111)][[Code](https://github.com/luzhang16/CIAN)]
41. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019,  Cao, Yanpeng et al.[[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)]
42. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.[[PDF](https://arxiv.org/abs/1808.04818)][[Code](https://github.com/Li-Chengyang/MSDS-RCNN)][[Project Link](https://li-chengyang.github.io/home/MSDS-RCNN/)]
43. Unified Multi-spectral Pedestrian Detection Based on Probabilistic Fusion Networks, Pattern Recognition 2018, Kihong Park et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300906)]
44. Multispectral Deep Neural Networks for Pedestrian Detection, BMVC 2016, Jingjing Liu et al.[[PDF](https://arxiv.org/abs/1611.02644)][[Code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn)]
45. Multispectral Pedestrian Detection Benchmark Dataset and Baseline, 2015, Soonmin Hwang et al.[[PDF](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)][[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

### Adversarial Attack and Defense
1. MEC-Patch: Visible-Infrared Cross-Modal Adversarial Attack Driven by Intrinsic Material Emissivity Laws, Zhixiang Huang et al., ACM MM 2026, [[PDF](https://arxiv.org/pdf/2608.05634)][[Code](https://github.com/star0521/MEC-Patch)]
2. Physical Adversarial Clothing Evades Visible-Thermal Detectors Via Non-Overlapping RGB-T Pattern, CVPR 2026, Xiaopei Zhu et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/39343)][[Code](https://github.com/thu-ml/RGBT-Clothing)]
3. CDUPatch: Color-Driven Universal Adversarial Patch Attack for Dual-Modal Visible-Infrared Detectors,  Jiahuan Long et al., ACM MM 2025, [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3755188)]
4. PAIF: Perception-Aware Infrared-Visible Image Fusion for Attack-Tolerant Semantic Segmentation, Zhu Liu et al., ACM MM 2023, [[PDF](https://dl.acm.org/doi/abs/10.1145/3581783.3611928)][[Code](https://github.com/LiuZhu-CV/PAIF)]

--------------------------------------------------------------------------------------

# RGB-T Aerial Object Detection
## Datasets

|Dataset                        | Years |Modality     |Images| Classes|Alignment|
|-------------------------------|-------|-------------|------|--------|---------|
|[DroneVehicle](https://github.com/VisDrone/DroneVehicle)|2021|RGB, LWIR|56K|5|Partially|
|[VEDAI](https://downloads.greyc.fr/vedai/)|2014|RGB, NIR|1K|9|Strictly|
|[NII-CU](https://www.nii-cu-multispectral.org/)|2022|RGB, LWIR|5K|1|Strictly|
|[DVTOD](https://github.com/VDT-2048/DVTOD)|2024|RGB, LWIR|2K|3|Misaligned|
|[VTSaR](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10833840)|2024|RGB, LWIR|40K|1|Strictly|
|[RGBT-Tiny](https://github.com/XinyiYing/RGBT-Tiny)|2025|RGB, LWIR|93K|7|Partially|
|[ATR-UMOD](https://github.com/supercpy/ATR-UMOD)|2025|RGB, LWIR|13K|11|Partially|
|[MODA](https://github.com/shuaihao-han/MODA)|2026|RGB, NIR|13K|8|Strictly|
|[Tri-Modal Dataset](https://github.com/radlab-sketch/trimodal-uav-det.git)|2026|RGB, LWIR, Event|10K|1|Partially|

## Papers
1. DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection, CVPR 2026, Chaolang Li et al.,  [[PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.pdf)]
2. Unifying Heterogeneous Multi-Modal Remote Sensing Detection Via Language-Pivoted Pretraining, ICML 2026,Yuxuan Li et al., [[PDF](https://arxiv.org/pdf/2603.01758)][[Code](https://github.com/zcablii/SM3Det)]
3. Bridging Multimodal Fusion and Expert Routing via Spectral Reliability Descriptors for Robust Object Detection, ACMMM 2026, Yefeng Wu, [[PDF](https://arxiv.org/pdf/2606.01173)]
4. IRAA: Information Redistribution Adversarial Attacks for Multimodal Object Detection, ACMMM 2026, Sixu Liu et al., [[PDF]()]
5. AIR-Det: Asymmetric Multimodal Object Detection via Main-Conditioned Residual Learning, ACMMM 2026, Ziang Gan et al., [[PDF]()]
6. Beyond RGB: Mining Simulated IR Priors for RGB-only UAV Detection in Unlabeled Real Scenarios, ACMMM 2026, Qiulu Li et al., [[PDF]()]
7. Tri-Modal Fusion Transformers for UAV-based Object Detection, CVPR 2026, Craig Iaboni et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/39677)][[Code](https://github.com/radlab-sketch/trimodal-uav-det.git)]
8. MODA: The First Challenging Benchmark for Multispectral Object Detection in Aerial Images, AAAI 2026, Shuaihao Han et al., [[PDF](https://arxiv.org/abs/2512.09489)][[Code](https://github.com/shuaihao-han/MODA)]
9. SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection, AAAI 2026, Yuxuan Li et al., [[PDF](https://arxiv.org/pdf/2412.20665)][[Code](https://github.com/zcablii/SM3Det)]
10. IGIANet: Illumination Guided Implicit Alignment Network for Infrared–Visible
UAVDetection, AAAI 2026, Xiangqi Chen et al., [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/37298)]
1. Contextually-Guided State Space Fusion for Misaligned Multi-Spectral Object Detection, ACMMM 2025, Guyue Jin et al., [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3754550)]
2. Fusion Meets Diverse Conditions: A High-diversity Benchmark and Baseline for UAV-based Multimodal Object Detection with Condition Cues, ICCV 2025, Chen Chen et al., [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Fusion_Meets_Diverse_Conditions_A_High-diversity_Benchmark_and_Baseline_for_ICCV_2025_paper.pdf)]
3. Rethinking Multi-modal Object Detection from the Perspective of Mono-Modality Feature Learning, ICCV 2025, Tianyi Zhao et al., [[PDF](https://arxiv.org/abs/2503.11780)][[Code](https://github.com/Zhao-Tian-yi/M2D-LIF)]
4.  Detection-Friendly Nonuniformity Correction: A Union Framework for Infrared UAV Target Detection, CVPR 2025, Houzhang Fang et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Fang_Detection-Friendly_Nonuniformity_Correction_A_Union_Framework_for_Infrared_UAV_Target_CVPR_2025_paper.pdf)][[Code](https://github.com/IVPLaboratory/UniCD)]
5. SFFR: Spatial-Frequency Feature Reconstruction for Multispectral Aerial Object Detection, TGRS 2025, Xin Zuo et al. [[PDF](https://ieeexplore.ieee.org/document/11240218/)][[Code](https://github.com/qchenyu1027/SFFR)]
6. Reflectance-Guided Progressive Feature Alignment Network for All-Day UAV Object Detection, TGRS 2025, Zhicheng Zhao et al., [[PDF](https://ieeexplore.ieee.org/document/11017749)][[Code](https://github.com/uavdet/RGFNet)]
7.  Visible-Infrared Image Alignment For Unmanned Aerial Vehicles: Benchmark and New Baseline, TGRS 2025, Zhinan Gao et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10854690)][[Code](https://github.com/gaozhinanjiu/UAVmatch)]
8.  LAST-Net: Local Adaptivity Spatial Transformer Network for Multiobject Detection in UAV Remote Sensing Thermal Infrared Imagery, TGRS 2025, Min Li et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10972092)]
9.  Visible-Thermal Tiny Object Detection: A Benchmark Dataset and Baselines, TPAMI 2025, Xinyi Ying et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10914515)][[Code](https://github.com/XinyiYing/RGBT-Tiny)]
10. Aerial image object detection based on RGB-Infrared multi-branch progressive fusion, TGRS 2025, Kewei Liu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10933976)]
11. Dual Dynamic Cross-modal Interaction Network for Multimodal Remote Sensing Object Detection, TGRS 2025, Wei Bao et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10843244)][[Code](https://github.com/Drew-bw/DDCINet)]
12. Unified multimodal fusion transformer for few shot object detection for remote sensing images, Information Fusion 2024, Abdullah Azeem et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253524002860)]
13. SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection, Yuxuan Li et al., Arxiv 2025, [[PDF](https://arxiv.org/pdf/2412.20665)][[Code](https://github.com/zcablii/SM3Det)]
14. CCLDet: A Cross-Modality and Cross-Domain Low-Light Detector, Xiping Shang et al., TITS 2025, [[PDF](https://ieeexplore.ieee.org/abstract/document/10844996)]
15. Dual-Dynamic Cross-Modal Interaction Network for Multimodal Remote Sensing Object Detection, TGRS 2025, Wei Bao et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10843244)][[Code](https://github.com/Drew-bw/DDCINet)]
16. DAMSDet: Dynamic Adaptive Multispectral Detection Transformer with Competitive Query Selection and Adaptive Feature Fusion, ECCV 2024, Junjie Guo et al., [[PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/03984.pdf)][[Code](https://github.com/gjj45/DAMSDet)]
17. Cross Teaching-Enhanced Multi-spectral Remote Sensing Object Detection with Transformer, TGRS 2024, Jiahe Zhu et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10770223)]
18. Cross-Modal Oriented Object Detection of UAV Aerial Images Based on Image Feature, TGRS 2024, Huiying Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10440361)]
19. CMDistill: Cross-modal Distillation Framework for UAV Image Object Detection, JSTAR 2024, Xiaozhong Tong et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10715640)]
20. Multimodal Feature-Guided Pretraining for RGB-T Perception, JSTAR 2024, Junlin Ouyang et al., [[PDF](https://ieeexplore.ieee.org/document/10663834)]
21. Mask-Guided Mamba Fusion for Drone-based Visible-Infrared Vehicle Detection, TGRS 2024, Simiao Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10659747)]
22. Low-rank Multimodal Remote Sensing Object Detection with Frequency Filtering Experts, TGRS 2024, Xu Sun et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10643097)][[Code](https://github.com/cq100/LF-MDet)]
23. Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection, CVPR 2024, Chen Chen et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.pdf)]
24. Misaligned Visible-Thermal Object Detection: A Drone-based Benchmark and Baseline, TIV 2024, Kechen Song, [[PDF](https://ieeexplore.ieee.org/abstract/document/10522939/authors)][[Code](https://github.com/VDT-2048/DVTOD)]
25. C<sup>2</sup>Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection, TGRS 2024, Maoxun Yuan et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10472947)][[Code](https://github.com/yuanmaoxun/C2Former)]
26. ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection, Pattern Recognition 2024, Shen Jifeng et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10225383)][[Code](https://github.com/chanchanchan97/ICAFusion)]
27. Improving RGB-infrared object detection with cascade alignment-guided transformer, Information Fusion 2024, Maoxun Yuan et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253524000241)]
28. Multispectral Object Detection via Cross-Modal Conflict-Aware Learning, ACM MM 2023, Xiao He et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3612651)][[Code](https://github.com/hexiao-cs/CALNet-Dronevehicle)]
29. LRAF-Net: Long-Range Attention Fusion Network for Visible–Infrared Object Detection, TNNLS 2023, Haolong Fu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10144688)]
30. Modality Balancing Mechanism for RGB-Infrared Object Detection in Aerial Image, PRCV 2023, Weibo Cai er al., [[PDF](https://link.springer.com/chapter/10.1007/978-981-99-8555-5_7)]
31. RGB-Infrared Multi-Modal Remote Sensing Object Detection Using CNN and Transformer Based Feature Fusion, IGARSS 2023, Tao Tian et al., [[PDF](https://ieeexplore.ieee.org/document/10282173)]
32. GF-Detection: Fusion with GAN of Infrared and Visible Images for Vehicle Detection at Nighttime, Remote Sensing 2022, Peng Gao et al. [[PDF](https://www.mdpi.com/2072-4292/14/12/2771)]
33. Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery, Pattern Recognition, Qingyun Fang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322002679)]
34. Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection, ECCV 2022, Maoxun Yuan et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690501.pdf)]
35. Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning, TCSVT 2022, Yiming Sun [[PDF](https://ieeexplore.ieee.org/abstract/document/9759286)][[Code](https://github.com/SunYM2020/UA-CMDet)]
36. Improving RGB-Infrared Object Detection by Reducing Cross-Modality Redundancy, Remote Sensing 2022, Qingwang Wang et al. [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]

--------------------------------------------------------------------------------------

# RGB-T Semantic Segmentation
## Datasets
|Dataset                        | Years |Modality     |Images| Classes|
|-------------------------------|-------|-------------|------|--------|
|[MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)|2017|RGB, LWIR|1.5K|8|
|[PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)|2020|RGB, LWIR|900|5|
|[SemanticRT](https://github.com/jiwei0921/SemanticRT)|2023|RGB, LWIR|11K|13|
|[FMB](https://github.com/JinyuanLiu-CV/SegMiF)|2023|RGB, LWIR|1.5K|14|
|[IDD-AW (Adverse Weather)](https://iddaw.github.io/)|2023|RGB, NIR|5K|18|
|[Caltech Aerial RGBT](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset)|2024|RGB, LWIR|4K|10|

## Papers
1. Towards Unsupervised Multi-modal Semantic Segmentation, ECCV 2026, Haitian Zhang et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/3704)]
2. UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning, ACM MM 2025, Maoxun Yuan et al., [[PDF](https://arxiv.org/abs/2404.17360)][[Code](https://github.com/PoTsui99/UniRGB-IR)]
3.  StitchFusion: Weaving Any Visual Modalities to Enhance Multimodal Semantic Segmentation, ACMMM 2025, Bingyu Li et al., [[PDF](https://dl.acm.org/doi/pdf/10.1145/3746027.3755110)][[Code](https://github.com/LiBingyu01/StitchFusion)]
4.  AMDANet: Attention-Driven Multi-Perspective Discrepancy Alignment for RGB-Infrared Image Fusion and Segmentation, ICCV 2025,  Haifeng Zhong et al., [[PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhong_AMDANet_Attention-Driven_Multi-Perspective_Discrepancy_Alignment_for_RGB-Infrared_Image_Fusion_and_ICCV_2025_paper.pdf)][[Code](https://github.com/Zhonghaifeng6/AMDANet)]
5. Every SAMDrop Counts: Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond, CVPR 2025, Guanyao Wu et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_Every_SAM_Drop_Counts_Embracing_Semantic_Priors_for_Multi-Modality_Image_CVPR_2025_paper.pdf)][[Code](https://github.com/RollingPlain/SAGE_IVIF)]
6. Task-driven Image Fusion with Learnable Fusion Loss, CVPR 2025, Haowen Bai et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Bai_Task-driven_Image_Fusion_with_Learnable_Fusion_Loss_CVPR_2025_paper.pdf)][[Code](https://github.com/HaowenBai/TDFusion)]
7. Knowledge Distillation SegFormer-Based Network  for RGB-T Semantic Segmentation, TCYB 2024, Wujie Zhou et al,. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10817074)][[Code](https://github.com/purple-ting/KDSNet)]
8. Caltech Aerial RGB-Thermal Dataset in the Wild, ECCV 2024, Connor Lee et al,. [[PDF](extension://ngbkcglbmlglgldjfcnhaijeecaccgfi/https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08026.pdf)][[Code](https://github.com/aerorobotics/caltech-aerial-rgbt-dataset)]
9. Context-Aware Interaction Network for RGB-T Semantic Segmentation, TMM 2024, Ying Lv et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10379106)][[Code](https://github.com/YingLv1106/CAINet)]
10. Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation, Jinyuan Liu et al., [[PDF](https://arxiv.org/pdf/2308.02097)][[Code](https://github.com/JinyuanLiu-CV/SegMiF)]
11. CACFNet: Cross-Modal Attention Cascaded Fusion Network for RGB-T Urban Scene Parsing, TIV 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10251592)]
12. On Exploring Shape and Semantic Enhancements for RGB-X Semantic Segmentation, TIV 2023, Yuanjian Yang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10185113)][[Code](https://github.com/HenonBamboo/SASEM)]
13. Complementarity-aware cross-modal feature fusion network for RGB-T semantic segmentation, PR 2023, Wei Wu et al.,  [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322003624)]
14. MMSMCNet: Modal Memory Sharing and Morphological Complementary Networks for RGB-T Urban Scene Semantic Segmentation, TCSVT 2023, Wujie Zhou et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10123009)][[Code](https://github.com/2021nihao/MMSMCNet)]
15. SGFNet: Semantic-Guided Fusion Network for RGB-Thermal Semantic Segmentation, TCSVT 2023, Yike Wang et al., [[PDF](https://ieeexplore.ieee.org/document/10138593)][[Code](https://github.com/kw717/SGFNet)]
16. DBCNet: Dynamic Bilateral Cross-Fusion Network for RGB-T Urban Scene Understanding in Intelligent Vehicles, TCYB 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10217340)]
17. Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks, RAL 2023, Mingjian Liang et al., [[PDF](https://ieeexplore.ieee.org/document/10113725)][[Code](https://github.com/freeformrobotics/eaefnet)]
18. Embedded Control Gate Fusion and Attention Residual Learning for RGB–Thermal Urban Scene Parsing, TITS 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/10041960)]
19. UTFNet: Uncertainty-Guided Trustworthy Fusion Network for RGB-Thermal Semantic Segmentation, GRSL 2023, Qingwang Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10273407)][[Code](https://github.com/KustTeamWQW/UTFNet)]
20. Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning, arxiv 2023, Shaohua Dong et al., [[PDF](https://arxiv.org/pdf/2312.00360.pdf)][[Code](https://github.com/shaohuadong2021/dplnet?tab=readme-ov-file)]
21. A RGB-Thermal Image Segmentation Method Based on Parameter Sharing and Attention Fusion for Safe Autonomous Driving, TITS 2023, Guofa Li et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10337777)]
22. SFAF-MA: Spatial Feature Aggregation and Fusion With Modality Adaptation for RGB-Thermal Semantic Segmentation, TIM 2023, Xunjie He et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10103760)][[Code](https://github.com/hexunjie/SFAF-MA)]
23. Edge-aware guidance fusion network for RGB–thermal scene parsing, AAAI 2022, Wujie Zhou et al., [[PDF](https://arxiv.org/abs/2112.05144)][[Code](https://github.com/ShaohuaDong2021/EGFNet)]
24. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers, TITS 2022, Jiaming Zhang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10231003)][[Code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)]
25. RGB-T Semantic Segmentation with Location, Activation, and Sharpening, TCSVT 2022, Gongyang Li et al., [[PDF](https://ieeexplore.ieee.org/document/9749834)][[Code](https://github.com/MathLee/LASNet)]
26. A Feature Divide-and-Conquer Network for RGB-T Semantic Segmentation, TCSVT 2022, Shenlu Zhao et al., [[PDF](https://ieeexplore.ieee.org/document/9987529)]
27. CCAFFMNet: Dual-spectral semantic segmentation network with channel-coordinate attention feature fusion module, Neurocomputing 2022, [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231221017331)]
28. CGFNet: cross-guided fusion network for RGB-thermal semantic segmentation, The Visual Computer 2022, Yanping Fu et al., [[PDF](https://link.springer.com/article/10.1007/s00371-022-02559-2)]
29. MTANet: Multitask-Aware Network with Hierarchical Multimodal Fusion for RGB-T Urban Scene Understanding, TIV 2022, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9900351)][[Code](https://github.com/ShaohuaDong2021/MTANet)]
30. GCNet: Grid-Like Context-Aware Network for RGB-Thermal Semantic Segmentation, Neurocomputing 2022, Jinfu Liu et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231222009006)]
31. ABMDRNet: Adaptive-weighted Bi-directional Modality Difference Reduction Network for RGB-T Semantic Segmentation, CVPR 2021, Qiang Zhang et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_ABMDRNet_Adaptive-Weighted_Bi-Directional_Modality_Difference_Reduction_Network_for_RGB-T_Semantic_CVPR_2021_paper.pdf)]
32. GMNet: Graded-Feature Multilabel-Learning Network for RGB-Thermal Urban Scene Semantic Segmentation, TIP 2021, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9531449)][[Code](https://github.com/Jinfu0913/GMNet)]
33. MFFENet: Multiscale Feature Fusion and Enhancement Network for RGBThermal Urban Road Scene Parsing, TMM 2021, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9447924)]
34. FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation, IROS 2021, Fuqin Deng et al., [[PDF](https://arxiv.org/pdf/2110.08988.pdf)][[Code](https://github.com/matrixgame2018/FEANet)]
35. HeatNet: Bridging the Day-Night Domain Gap in Semantic Segmentation with Thermal Images, IROS 2021, Johan Vertens et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9341192)]
36. Robust semantic segmentation based on RGB-thermal in variable lighting scenes, Measurement 2021, Zhifeng Guo et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0263224121010903)]
37. FuseSeg: Semantic segmentation of urban scenes based on RGB and thermal data fusion, TASE 2020, Yuxiang Sun et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9108585/)]
38. PST900: RGB-Thermal Calibration, Dataset and Segmentation Network, ICRA 2020, Shreyas S. Shivakumar et al., [[PDF](https://ieeexplore.ieee.org/document/9196831)][[Code](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)]
39. RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes, RAL 2019, Yuxiang Sun et al., [[PDF](https://ieeexplore.ieee.org/document/8666745)][[Code](https://github.com/yuxiangsun/RTFNet)]
40. MFNet: Towards Real-Time Semantic Segmentation for Autonomous Vehicles with Multi-Spectral Scenes, IROS 2019, Qishen Ha et al., [[PDF](https://ieeexplore.ieee.org/document/8206396)][[Code](https://github.com/haqishen/MFNet-pytorch)]


--------------------------------------------------------------------------------------

# RGB-T Salient Object Detection
## Datasets
VT821 Dataset [[PDF](https://link.springer.com/content/pdf/10.1007%2F978-981-13-1702-6_36.pdf)][[link](https://drive.google.com/file/d/0B4fH4G1f-jjNR3NtQUkwWjFFREk/view?resourcekey=0-Kgoo3x0YJW83oNSHm5-LEw)], VT1000 Dataset [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296)][[link](https://drive.google.com/file/d/1NCPFNeiy1n6uY74L0FDInN27p6N_VCSd/view)], VT5000 Dataset [[PDF](https://arxiv.org/pdf/2007.03262.pdf)][[link]( https://pan.baidu.com/s/1ksuUr3cr6_-fZAsSUp0n0w)[9yqv]], VI-RGBT1500 Dataset[[PDF](https://ieeexplore.ieee.org/document/10003255)][[link](https://github.com/huanglm-me/VI-RGBT1500?tab=readme-ov-file)], UVT2000 Dataset[[PDF](https://ieeexplore.ieee.org/abstract/document/10551543)][[link](https://github.com/Angknpng/SACNet)]
## Papers
1. Liquid Fusion of Heterogeneous Representations Towards General Salient Object Detection, ECCV 2026, Ke Chen et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/4978)][[Code](https://github.com/cke520/LFNet)]
2. RA-SOD: Reliability-Aware RGB-T Salient Object Detection Under Modality Degradation, ECCV 2026, Hongbo Gao et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/3933)][[Code](https://github.com/zaoxienian/RA-SOD)]
3. Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection, CVPR 2026, Mianzhao Wang et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/39877)]
4. UniRGB-IR: A Unified Framework for Visible-Infrared Downstream Tasks via Adapter Tuning, ACM MM 2025, Maoxun Yuan et al., [[PDF](https://arxiv.org/abs/2404.17360)][[Code](https://github.com/PoTsui99/UniRGB-IR)]
5. Divide-and-Conquer: Confluent Triple-Flow Network for RGB-T Salient Object Detection, TPAMI 2024, Hao Tang et al., [[PDF](https://ieeexplore.ieee.org/document/10778650)][[Code](https://github.com/CSer-Tang-hao/ConTriNet_RGBT-SOD)]
6. Alignment-Free RGBT Salient Object Detection: Semantics-guided Asymmetric Correlation Network and A Unified Benchmark, TMM 2024, Kunpeng Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10551543)][[Code](https://github.com/Angknpng/SACNet)]
7. VST++: Efficient and Stronger Visual Saliency Transformer, TPAMI 2024, Nian Liu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10497889)][[Code](https://github.com/nnizhang/VST)]
8. UniTR: A Unified TRansformer-based Framework for Co-object and Multi-modal Saliency Detection, TMM 2024, Ruohao Guo et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10444934)]
9. Learning Adaptive Fusion Bank for Multi-modal Salient Object Detection, TCSVT 2024, Kunpeng Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10464332)]
10. TMNet: Triple-modal interaction encoder and multi-scale fusion decoder network for V-D-T salient object detection, PR 2024, Bin Wan et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320323007719)]
11. Weighted Guided Optional Fusion Network for RGB-T Salient Object Detection, TOMM 2024, Jie Wang et al.,  [[PDF](https://dl.acm.org/doi/abs/10.1145/3624984)][[Code](https://github.com/WJ-CV/WGOFNet)]
12. Position-Aware Relation Learning for RGB-Thermal Salient Object Detection, TIP 2023, Heng Zhou et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10113883)]
13. WaveNet: Wavelet Network With Knowledge Distillation for RGB-T Salient Object Detection, TIP 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10127616)][[Code](https://github.com/nowander/WaveNet)]
14. LSNet: Lightweight Spatial Boosting Network for Detecting Salient Objects in RGB-Thermal Images, TIP 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/10042233)][[Code](https://github.com/zyrant/LSNet)]
15. CAVER: Cross-Modal View-Mixed Transformer for Bi-Modal Salient Object Detection, TIP 2023, Youwei Pang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10015667)][[Code](https://github.com/lartpang/caver)]
16. Glass Segmentation With RGB-Thermal Image Pairs, TIP 2023, Dong Huo et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10076416)][[Code](https://github.com/Dong-Huo/RGB-T-Glass-Segmentation)]
17. MFFNet: Multi-modal Feature Fusion Network for V-D-T Salient Object Detection, TMM 2023, Bin Wan et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10171982)]
18. LFTransNet: Light Field Salient Object Detection via a Learnable Weight Descriptor, TCSVT 2023, Zhengyi Liu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10138590)][[Code](https://github.com/liuzywen/LFTransNet)]
19. Multiple Graph Affinity Interactive Network and a Variable Illumination Dataset for RGBT Image Salient Object Detection, TCSVT 2023, Kechen Song et al., [[PDF](https://ieeexplore.ieee.org/document/10003255)][[Code](https://github.com/huanglm-me/VI-RGBT1500)]
20. Cross-Modality Double Bidirectional Interaction and Fusion Network for RGB-T Salient Object Detection, TCSVT 2023, Zhengxuan Xie et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10032588)]
21. DBCNet: Dynamic Bilateral Cross-Fusion Network for RGB-T Urban Scene Understanding in Intelligent Vehicles, TCYB 2023, [[PDF](https://ieeexplore.ieee.org/abstract/document/10217340)]
22. Frequency-aware feature aggregation network with dual-task consistency for RGB-T salient object detection, PR 2023, Heng Zhou et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320323007409)]
23. Cross-modal co-feedback cellular automata for RGB-T saliency detection, PR 2023, Yu Pang et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322006185)]
24. An interactively reinforced paradigm for joint infrared-visible image fusion and saliency object detection, Information Fusion 2023, Di Wang et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253523001446)][[Code](https://github.com/wdhudiekou/IRFS.)] 
25. Thermal images-aware guided early fusion network for cross-illumination RGB-T salient object detection, EAAI 2023, Han Wang et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0952197622006303)][[Code](https://github.com/VDT-2048/TAGFNet)]
26. Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks, RAL 2023, Mingjian Liang et al., [[PDF](https://ieeexplore.ieee.org/document/10113725)][[Code](https://github.com/freeformrobotics/eaefnet)]
27. MENet: Lightweight multimodality enhancement network for detecting salient objects in RGB-thermal images, Neurocomputing 2023, Junyi Wu et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231223000358)]
28. Feature aggregation with transformer for RGB-T salient object detection, Neurocomputing 2023, Ping Zhang et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231223004526)][[Code](https://github.com/ELOESZHANG/FANet)] 
29. Texture-Guided Saliency Distilling for Unsupervised Salient Object Detection, CVPR2023， Huajun Zhou et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Texture-Guided_Saliency_Distilling_for_Unsupervised_Salient_Object_Detection_CVPR_2023_paper.pdf)][[Code](www.github.com/moothes/A2S-v2)]
30. Saliency Prototype for RGB-D and RGB-T Salient Object Detection, ACM MM 2023, Zihao Zhang et al., [[PDF](https://dl.acm.org/doi/abs/10.1145/3581783.3612466)][[Code](https://github.com/ZZ2490/SPNet)]
31. ADNet: An Asymmetric Dual-Stream Network for RGB-T Salient Object Detection, ACM MM 2023, Yaqun Fang et al., [[PDF](https://dl.acm.org/doi/abs/10.1145/3595916.3626440)]
32. Scribble-Supervised RGB-T Salient Object Detection, ICME 2023, Zhengyi Liu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10219673)][[Code](https://github.com/liuzywen/RGBTScribble-ICME2023)]
33. Feature Enhancement and Fusion for RGB-T Salient Object Detection, ICIP 2023, Fengming Sun et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10222404)]
34. Weakly Alignment-free RGBT Salient Object Detection with Deep Correlation Network, TIP 2022, Zhengzheng Tu et al.,  [[PDF](https://ieeexplore.ieee.org/abstract/document/9779787)][[Code](https://github.com/lz118/Deep-Correlation-Network)]
35. Cross-modal co-feedback cellular automata for RGB-T saliency detection, PR 2022, Yu Pang et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322006185)]
36. RGBT Salient Object Detection: A Large-Scale Dataset and Benchmark, TMM 2022, Zhengzheng Tu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9767629)]
37. Does Thermal really always matter for RGB-T salient object detection, TMM 2022, Runmin Cong et al., [[PDF](https://ieeexplore.ieee.org/document/9926193)][[Code](https://github.com/rmcong/TNet_TMM2022)]
38. TCNet: Co-Salient Object Detection via Parallel Interaction of Transformers and CNNs, TCSVT 2023, Yanliang Ge et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9968016)][[Code](https://github.com/zhangqiao970914/TCNet)]
39. Modality-Induced Transfer-Fusion Network for RGB-D and RGB-T Salient Object Detection, TCSVT 2022, Gang Chen et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9925217)]
40. Cross-Collaborative Fusion-Encoder Network for Robust RGB-Thermal Salient Object Detection, TCSVT 2022, Guibiao Liao et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9801871)][[Code](https://git.openi.org.cn/OpenVision/CCFENet)]
41. CGMDRNet: Cross-Guided Modality Difference Reduction Network for RGB-T Salient Object Detection, TCSVT 2022, Gang Chen et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9756028)]
42. RGB-T Semantic Segmentation With Location, Activation, and Sharpening, TCSVT 2022, Gongyang Li et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9900351)][[Code](https://github.com/MathLee/LASNet)]
43. HRTransNet: HRFormer-Driven Two-Modality Salient Object Detection, TCSVT 2022, Bin Tang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9869666)][[Code](https://github.com/liuzywen/HRTransNet)]
44. Asymmetric cross-modal activation network for RGB-T salient object detection, KBS 2022, Chang Xu et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705122011406)][[Code]( www.github.com/xanxuso/ACMANet)]
45. Three-stream interaction decoder network for RGB-thermal salient object detection, KBS 2022, Fushuo Huo et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705122011005)][[Code](https://github.com/huofushuo/TIDNet)]
46. Real-time One-stream Semantic-guided Refinement Network for RGB-Thermal Salient Object Detection, TIM 2022, Fushuo Huo et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9803225)][[Code](https://github.com/huofushuo/OSRNet)]
47. Multi-modal Interactive Attention and Dual Progressive Decoding Network for RGB-D/T Salient Object Detection, Neurocomputing 2022, Yanhua Liang et al.,  [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231222002971)][[Code](https://github.com/Liangyh18/MIA_DPD)]
48. PSNet: Parallel symmetric network for RGB-T salient object detection, Neurocomputing 2022, Hongbo Bi et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0925231222011559)]
49. Mirror Complementary Transformer Network for RGB-thermal Salient Object Detection, arxiv 2022, [[PDF](https://arxiv.org/abs/2207.03558)][[Code](https://github.com/jxr326/SwinMCNet)]
50. Bimodal Information Fusion Network for Salient Object Detection based on Transformer, PRML 2022, Zhuo Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9882262)]
51. Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection, TIP 2021, Zhengzheng Tu et al., [[PDF](https://ieeexplore.ieee.org/document/9454273)][[Code](https://github.com/lz118/Multi-interactive-Dual-decoder)]
52. ECFFNet: Effective and Consistent Feature Fusion Network for RGB-T Salient Object Detection, TCSVT 2021, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9420662)]
53. SwinNet: Swin Transformer drives edge-aware RGB-D and RGB-T salient object detection, TCSVT 2021, [[PDF](https://ieeexplore.ieee.org/document/9611276)][[Code](https://github.com/liuzywen/SwinNet)]
54. Unified Information Fusion Network for Multi-Modal RGB-D and RGB-T Salient Object Detection, TCSVT 2021, Wei Gao et al., [[PDF](https://ieeexplore.ieee.org/document/9439490)]
55. Multi-graph Fusion and Learning for RGBT Image Saliency Detection, TCSVT 2021, Liming Huang et al., [[PDF](https://ieeexplore.ieee.org/document/9389777)]
56. CGFNet: Cross-Guided Fusion Network for RGB-T Salient Object Detection, TCSVT 2021, Jie Wang et al., [[PDF](https://ieeexplore.ieee.org/document/9493207)][[Code](https://github.com/wangjie0825/CGFNet.git)]
57. Efficient Context-Guided Stacked Refinement Network for RGB-T Salient Object Detection, TCSVT 2021, Fushuo Huo et al., [[PDF](https://ieeexplore.ieee.org/document/9505635)][[Code](https://github.com/huofushuo/CSRNet)]
58. RGB-T Salient Object Detection via Fusing Multi-Level CNN Features, TIP 2020, Qiang Zhang et al., [[PDF](https://ieeexplore.ieee.org/document/8935533)][[Code](https://github.com/nexiakele/RGB-T-Salient-Object-Detection-via-Fusing-Multi-level-CNN-Features)]
59. Revisiting Feature Fusion for RGB-T Salient Object Detection, TCSCT 2020, Qiang Zhang et al., [[PDF](https://ieeexplore.ieee.org/document/9161021)][[Code](https://github.com/nexiakele/Revisiting-Feature-Fusion-for-RGB-T-Salient-Object-Detection)]
60. Deep Domain Adaptation Based Multi-Spectral Salient Object Detection, TMM 2020, Shaoyue Song et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9308922)]
61. Abiotic Stress Prediction from RGB-T Images of Banana Plantlets, ECCV 2020, Sagi Levanon et al., [[PDF](https://link.springer.com/chapter/10.1007/978-3-030-65414-6_20)]
62. Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation, AAAI 2020, Shaoyue Song et al.[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/6879)]
63. Deep Domain Adaptation Based Multi-spectral Salient Object Detection, TMM 2020, Shaoyue Song et al., [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9308922)]
64. RGB-T Image Saliency Detection via Collaborative Graph Learning, TMM 2019, Zhengzheng Tu et al., [[PDF](https://ieeexplore.ieee.org/document/8744296)][[Code](https://github.com/lz118/RGBT-Salient-Object-Detection)]
65. RGBT Salient Object Detection: Benchmark and A Novel Cooperative Ranking Approach, TCSVT 2019, Jin Tang et al., [[PDF](https://ieeexplore.ieee.org/document/8891733)][[Code](https://github.com/lz118/RGBT-Salient-Object-Detection)]
66. M3S-NIR: Multi-Modal Multi-Scale Noise-Insensitive Ranking for RGB-T Saliency Detection, MIPR 2019, Zhengzheng Tu et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/8695412)][[Code](https://github.com/lz118/RGBT-Salient-Object-Detection)]
67. RGB-T Saliency Detection Benchmark: Dataset, Baselines, Analysis and a Novel Approach, IGTA 2018, Guizhao Wang et al., [[PDF](https://link.springer.com/chapter/10.1007/978-981-13-1702-6_36)][[Code](https://github.com/lz118/RGBT-Salient-Object-Detection)]
68. Learning Multiscale Deep Features and SVM Regressors for Adaptive RGB-T Saliency Detection, ISCID 2017, Yunpeng Ma et al., [[PDF](https://ieeexplore.ieee.org/document/8275796)]

--------------------------------------------------------------------------------------

# RGB-T Crowd Counting
## Datasets
RGBT-CC[[link](http://lingboliu.com/RGBT_Crowd_Counting.html)], DroneRGBT [[link](https://github.com/VisDrone/DroneRGBT)]
## Papers
1. Closing the Modality Gap without Pixel-wise Registration: Geometric-Semantic Tri-Optimization for Misaligned Infrared-Visible Image Segmentation, ACMMM 2026, Tingyu Shi et al.,  [[PDF]()][[Code](https://github.com/tingyu-shi/Tri-FRS)]
2. Free Lunch Enhancements for Multi-modal Crowd Counting, CVPR 2025, Haoliang Meng et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Meng_Free_Lunch_Enhancements_for_Multi-modal_Crowd_Counting_CVPR_2025_paper.pdf)][[Code](https://github.com/HenryCilence/Free-Lunch-Multimodal-Counting)]

3. CCANet: A Collaborative Cross-modal Attention Network for RGB-D Crowd Counting, TMM2023, Yanbo Liu et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10086642)]
4. MC3Net: Multimodality Cross-Guided Compensation Coordination Network for RGB-T Crowd Counting, TITS 2023, Wujie Zhou et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10285562)]
5.  RGB-T Multi-Modal Crowd Counting Based on Transformer, BMVC 2022, Zhengyi Liu et al. [[PDF](https://bmvc2022.mpi-inf.mpg.de/0427.pdf)]
6. Spatio-channel Attention Blocks for Cross-modal Crowd Counting, ACCV2022, Youjia Zhang et al. [[PDF](https://openaccess.thecvf.com/content/ACCV2022/html/Zhang_Spatio-channel_Attention_Blocks_for_Cross-modal_Crowd_Counting_ACCV_2022_paper.pdf)]
7. DEFNet: Dual-Branch Enhanced Feature Fusion Network for RGB-T Crowd Counting, TITS 2022, Zhou, Wujie et al. [[PDF](https://ieeexplore.ieee.org/document/9889192)]
8. MAFNet: A Multi-Attention Fusion Network for RGB-T Crowd Counting, arxiv2022, Pengyu Chen et al. [[PDF](https://arxiv.org/pdf/2208.06761.pdf)]
9. Multimodal Crowd Counting with Mutual Attention Transformers, ICME 2022, Wu, Zhengtao et al.  [[PDF](https://ieeexplore.ieee.org/abstract/document/9859777)]
10. Conditional RGB-T Fusion for Effective Crowd Counting, ICIP 2022, Esha Pahwa et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9897548)]
11. Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting, CVPR2021, Lingbo Liu et al. [[PDF](https://arxiv.org/pdf/2012.04529.pdf)][[Code](https://github.com/chen-judge/RGBTCrowdCounting)]
12. RGB-T Crowd Counting from Drone: A Benchmark and MMCCN Network, ACCV2020, Tao Peng et al. [[PDF](https://openaccess.thecvf.com/content/ACCV2020/papers/Peng_RGB-T_Crowd_Counting_from_Drone_A_Benchmark_and_MMCCN_Network_ACCV_2020_paper.pdf)][[Code](https://github.com/VisDrone/DroneRGBT)]

--------------------------------------------------------------------------------------

# RGB-T Fusion Tracking
## Datasets
GTOT [[PDF](https://ieeexplore.ieee.org/document/7577747)][[link](https://github.com/mmic-lcl/Datasets-and-benchmark-code)], RGBT234 Dataset [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302808)][[link](https://sites.google.com/view/ahutracking001/)], LasHeR Dataset [[PDF](https://arxiv.org/abs/2104.13202)][[link](https://github.com/mmic-lcl/Datasets-and-benchmark-code)], UniRTL Dataset [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320324007350)][[Code](https://github.com/Liamzh0331/Unismot)]
## Papers
1. Progressive Multi-cue Alignment for Unaligned RGBT Tracking, CVPR 2026, Jiandong Jin et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/38417)]
2. Local-to-global Cross-modal Coordination for Self-supervised RGB-T Tracking, ECCV 2026, Yueying Zhang et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/4228)][[Code](https://github.com/zyyyya/LGCTrack)]
3. RT-RMOT: A Dataset and Framework for RGB-Thermal Referring Multi-Object Tracking, ECCV 2026, Yanqiu Yu et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/5378)]
4. GAP-Track: Bridging The Resolution Gap for Cross-Resolution RGBT Tracking, ECCV 2026, Shiyu Zhang et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/3545)][[Code](https://github.com/shijiahahaha/GAPTrack)]
5. OCTA-SOT: Online Cross-Modal Trajectory Adjustment for RGBT Anti-UAV Single Object Tracking under Spatio-Temporal Misalignment, ECCV 2026, Xiaokang Liu et al., [[PDF](https://eccv.ecva.net/virtual/2026/poster/4855)][[Code](https://github.com/xkliu-eps/AntiUAVRGBTTracking)]
6. SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker, CVPR 2026, Junbin Su et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/40039)][[Code](https://github.com/AutoLab-SAI-SJTU/SEATrack)]
7. Adaptive Depth Lightweight RGB-T Tracking with Holistic Token Routing, CVPR 2026, Tian Ding et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/39098)]
8. RAGTrack: Language-aware RGBT Tracking with Retrieval-Augmented Generation, CVPR 2026, Hao Li et al., [[PDF](https://cvpr.thecvf.com/virtual/2026/poster/37117)]
9. Unaligned UAV RGBT Tracking: A Largescale Benchmark and a Novel Approach, AAAI 2026, Yun Xiao et al., [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/38079)]
10. UniRTL: A universal RGBT and low-light benchmark for object tracking, Pattern Recognition 2025, Lian Zhang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320324007350)][[Code](https://github.com/Liamzh0331/Unismot)]
11. MTNet: Learning Modality-aware Representation with Transformer for RGBT Tracking, ICME 2023, Ruichao Hou et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219799)]
12. Visual Prompt Multi-Modal Tracking, CVPR 2023, Jiawen Zhu et al. [[PDF](https://arxiv.org/abs/2303.10826)][[Code](https://github.com/jiawen-zhu/ViPT)]
13. Efficient RGB-T Tracking via Cross-Modality Distillation, CVPR 2023, Zhang, Tianlu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10205202)]
14. Bridging Search Region Interaction with Template for RGB-T Tracking, Hui, CVPR2023, Tianrui et al.[[PDF](https://ieeexplore.ieee.org/document/10203113)][[Code](https://github.com/RyanHTR/TBSI)]
15. Jointly Modeling Motion and Appearance Cues for Robust RGB-T Tracking, TIP2021, Zhang, Pengyu et al. [[PDF](https://ieeexplore.ieee.org/document/9364880)] TIP 2022, Tu, Zhengzheng et al., [[PDF](https://ieeexplore.ieee.org/document/9617143)]
16. RGBT tracking via reliable feature configuration, SCIS 2022, Tu, Zhengzheng et al. [[PDF](https://link.springer.com/article/10.1007/s11432-020-3160-5)]
17. Attribute-Based Progressive Fusion Network for RGBT Tracking, AAAI 2022, Xiao Yun et al. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20187/19946)][[Code](https://github.com/yangmengmeng1997/APFNet)]
18. Dense Feature Aggregation and Pruning for RGBT Tracking, ACM Multimedia 2022, Yabin Zhu et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3343031.3350928)]
19. Prompting for Multi-Modal Tracking, ACM Multimedia 2022, Jinyu Yang et al. [[PDF](https://arxiv.org/abs/2207.14571)]
20. Learning Adaptive Attribute-Driven Representation for Real-Time RGB-T Tracking, IJCV 2021, Zhang, Pengyu et al. [[PDF](https://link.springer.com/article/10.1007/s11263-021-01495-3)]
21. Quality-Aware Feature Aggregation Network for Robust RGBT Tracking, TIV 2021, Zhu, Yabin, [[PDF](https://ieeexplore.ieee.org/abstract/document/9035457)]
22. Challenge-Aware RGBT Tracking, ECCV 2020, Li Chenglong et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_14)]
23. Object fusion tracking based on visible and infrared images: A comprehensive review, Information Fusion 2020, Zhang, Xingchen et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253520302657)]
24. RGB-T object tracking: Benchmark and baseline, Pattern Recognition 2019, Li, Chenglong et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320319302808)]
25. Cross-Modal Pattern-Propagation for RGB-T Tracking, CVPR2020, Chaoqun Wang et al., [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Modal_Pattern-Propagation_for_RGB-T_Tracking_CVPR_2020_paper.pdf)]


--------------------------------------------------------------------------------------
