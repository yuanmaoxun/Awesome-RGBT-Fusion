# Awesome RGB-T Fusion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![-RGBT-red](https://user-images.githubusercontent.com/38373305/205479612-e61d11b4-6c3e-4eb0-8d19-2e7170eb7783.svg)
A collection of deep learning based RGB-T-Fusion methods, codes, and datasets.  
The main directions involved are Multispectral Pedestrian Detection, RGB-T Vehicle Detection, RGB-T Semantic Segmentation, RGB-T Crowd Counting, RGB-T Fusion Tracking.  
Feel free to star and fork! Keep updating....🚀

## Some News: 🆕

👀 **2024.03.17 Add one CVPR paper.** <br>

💎 **2024.03.15 Add new content about RGB-T Semantic segmentation.** <br>

👀 **2024.03.12 Add one our paper and one CVPR paper.** <br>

--------------------------------------------------------------------------------------
## Contents  

1. [Multispectral Pedestrian Detection](#Multispectral-Pedestrian-Detection)
2. [RGB-T Vehicle Detection](#RGB-T-Vehicle-Detection)
3. [RGB-T Semantic Segmentation](#RGB-T-Semantic-Segmentation)
4. [RGB-T Salient Object Detection](#RGB-T-Salient-Object-Detection)
5. [RGB-T Crowd Counting](#RGB-T-Crowd-Counting)
6. [RGB-T Fusion Tracking](#RGB-T-Fusion-Tracking)

--------------------------------------------------------------------------------------
# Multispectral Pedestrian Detection
## Datasets and Annotations
[KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/), [CVC-14 dataset](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/), [FLIR dataset](https://www.flir.cn/oem/adas/adas-dataset-form/), [FLIR-aligned dataset](https://github.com/zonaqiu/FLIR-align), [Utokyo](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/), [M<sup>3</sup>FD dataset](https://github.com/dlut-dimt/TarDAL)
- Improved KAIST Testing Annotations provided by Liu et al.[Link to download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
- Sanitized KAIST Training Annotations provided by Li et al.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN) 
- Improved KAIST Training Annotations provided by Zhang et al.[Link to download](https://github.com/luzhang16/AR-CNN) 
## Tools
- Evalutaion codes.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[Link to download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)
## Papers
### Fusion Architecture

1. Causal Mode Multiplexer: A Novel Framework for Unbiased Multispectral Pedestrian Detection, CVPR 2024, Taeheon Kim et al. [[PDF](https://arxiv.org/pdf/2403.01300.pdf)]
2. Removal and Selection: Improving RGB-Infrared Object Detection via Coarse-to-Fine Fusion, arxiv 2024, Tianyi Zhao et al. [[PDF](https://arxiv.org/abs/2401.10731)][[Code]( https://github.com/Zhao-Tian-yi/RSDet.git)][[知乎](https://zhuanlan.zhihu.com/p/679289837)]
3. ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection, Pattern Recognition 2024, Shen Jifeng et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320323006118)][[Code](https://github.com/chanchanchan97/ICAFusion)]
4. Improving RGB-infrared object detection with cascade alignment-guided transformer, Information Fusion 2024, Maoxun Yuan et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253524000241)]
5. Multispectral Object Detection via Cross-Modal Conflict-Aware Learning, ACM MM 2023, Xiao He et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3612651)][[Code](https://github.com/hexiao0275/CALNet-Dronevehicle)]
6. Stabilizing Multispectral Pedestrian Detection with Evidential Hybrid Fusion, TCSVT 2023, Li Qing et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10225383)]
7. Multimodal Object Detection by Channel Switching and Spatial Attention, CVPRW 2023, Yue Cao et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf)]
8. Multi-Modal Feature Pyramid Transformer for RGB-Infrared Object Detection, TITS 2023, Yaohui Zhu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10105844)]
9.  Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection, TMM 2023, Ruimin Li et al. [[PDF](https://ieeexplore.ieee.org/document/10114594)][[Code](https://github.com/RimXidian/MCHE-CF-for-Multispectral-Pedestrian-Detection)]
10. HAFNet: Hierarchical Attentive Fusion Network for Multispectral Pedestrian Detection, Remote Sensing 2023, Peiran Peng et al. [[PDF](https://www.mdpi.com/2072-4292/15/8/2041)]
11. Multimodal Object Detection via Probabilistic Ensembling, ECCV2022, Yi-Ting Chen et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)][[Code](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)]
12. Learning a Dynamic Cross-Modal Network for Multispectral Pedestrian Detection, ACM Multimedia 2022, Jin Xie et al. [[PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3547895)]
13. Confidence-aware Fusion using Dempster-Shafer Theory for Multispectral Pedestrian Detection, TMM 2022, Qing Li et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9739079)]
14. Attention-Guided Multi-modal and Multi-scale Fusion for Multispectral Pedestrian Detection, PRCV 2022, Wei Bao et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-18907-4_30)]
15. Improving RGB-Infrared Pedestrian Detection by Reducing Cross-Modality Redundancy, ICIP2022, Qingwang Wang et al.  [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]
16. Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving, IEEE Transactions on Intelligent Transportation Systems, Kinjal Dasgupta et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9706418)]
17. Adopting the YOLOv4 Architecture for Low-LatencyMultispectral Pedestrian Detection in Autonomous Driving, Sensors 2022, Kamil Roszyk et al. [[PDF](https://www.mdpi.com/1424-8220/22/3/1082)]
18. Deep Active Learning from Multispectral Data Through Cross-Modality Prediction Inconsistency, ICIP2021, Heng Zhang et al.[[PDF](https://ieeexplore.ieee.org/document/9506322)]
19. Attention Fusion for One-Stage Multispectral Pedestrian Detection, Sensors 2021, Zhiwei Cao et al. [[PDF](https://www.mdpi.com/1424-8220/21/12/4184)]
20. Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, Jung Uk Kim et al. [[PDF](https://ieeexplore.ieee.org/document/9419080)]
21. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]
22. Guided Attentive Feature Fusion for Multispectral Pedestrian Detection, WACV 2021, Heng Zhang et al. [[PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)]
23. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
24. Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al. [[PDF](https://hal.archives-ouvertes.fr/hal-02872132/file/icip2020.pdf)]
25. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
26. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
27. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al. [[PDF](https://arxiv.org/abs/1901.02645)][[Code](https://github.com/luzhang16/AR-CNN)]
28. Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pesdestrian Detecion, ISPRS Journal of Photogrammetry and Remote Sensing 2019, Yanpeng Cao et al.[[PDF](https://arxiv.org/abs/1902.05291)][[Code](https://github.com/dayanguan/realtime_multispectral_pedestrian_detection)]
29. Cross-modality interactive attention network for multispectral pedestrian detection, Information Fusion 2019, Lu Zhang et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253518304111)][[Code](https://github.com/luzhang16/CIAN)]
30. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019,  Cao, Yanpeng et al.[[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)]
31. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.[[PDF](https://arxiv.org/abs/1808.04818)][[Code](https://github.com/Li-Chengyang/MSDS-RCNN)][[Project Link](https://li-chengyang.github.io/home/MSDS-RCNN/)]
32. Unified Multi-spectral Pedestrian Detection Based on Probabilistic Fusion Networks, Pattern Recognition 2018, Kihong Park et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300906)]
33. Multispectral Deep Neural Networks for Pedestrian Detection, BMVC 2016, Jingjing Liu et al.[[PDF](https://arxiv.org/abs/1611.02644)][[Code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn)]
34. Multispectral Pedestrian Detection Benchmark Dataset and Baseline, 2015, Soonmin Hwang et al.[[PDF](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)][[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

### Pixel-level Fusion for Detection
1. EfficientMFD: Towards More Efficient Multimodal Synchronous Fusion Detection, arxiv 2024, Jiaqing Zhang et al. [[PDF](https://arxiv.org/abs/2403.09323)][[Code]( https://github.com/icey-zhang/EfficientMFD)]
2. Multi-modal Gated Mixture of Local-to-Global Experts for Dynamic Image Fusion, ICCV 2023, Yiming Sun et al.[[PDF](https://arxiv.org/abs/2302.01392)][[Code]( https://github.com/SunYM2020/MoE-Fusion)]
3. MetaFusion : Infrared and Visible Image Fusion via Meta-Feature Embedding from Object Detection, CVPR 2023, Wenda Zhao et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_MetaFusion_Infrared_and_Visible_Image_Fusion_via_Meta-Feature_Embedding_From_CVPR_2023_paper.pdf)][[Code](https://github.com/wdzhao123/MetaFusion)]
4. Locality guided cross-modal feature aggregation and pixel-level fusion for multispectral pedestrian detection, Information Fusion 2022, Yanpeng Cao et al. [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253522000549)]
5. Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection, CVPR 2022, Jinyuan Liu et al.[[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Target-Aware_Dual_Adversarial_Learning_and_a_Multi-Scenario_Multi-Modality_Benchmark_To_CVPR_2022_paper.pdf)][[Code](https://github.com/dlut-dimt/TarDAL)]
6. DetFusion: A Detection-driven Infrared and Visible Image Fusion Network, ACM Multimedia 2022, Yiming Sun et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3503161.3547902)][[Code](https://github.com/SunYM2020/DetFusion)]

### Illumination Aware
1. Illumination-Guided RGBT Object Detection With Inter- and Intra-Modality Fusion, IEEE Transactions on Instrumentation and Measurement 2023, Yan Zhang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10057437/)][[Code](https://github.com/NNNNerd/Triple-I-Net-TINet)]
2. IGT: Illumination-guided RGB-T object detection with transformers, Knowledge-Based Systems 2023, Keyu Chen et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705123001739)]
3. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
4. Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection, Information Fusion 2019, Dayan Guan et al.[[PDF](https://arxiv.org/abs/1802.09972)][[Code](https://github.com/dayanguan/illumination-aware_multispectral_pedestrian_detection/)]
5. Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection, Pattern Recognition 2018, Chengyang Li et al.[[PDF](https://arxiv.org/pdf/1802.09972.pdf)][[Code](https://github.com/Li-Chengyang/IAF-RCNN)]

### Feature Alignment
1. Cross-Modality Proposal-guided Feature Mining for Unregistered RGB-Thermal Pedestrian Detection, TMM 2024, Chao Tian et al. [[PDF](https://ieeexplore.ieee.org/document/10382506)]
2. Attentive Alignment Network for Multispectral Pedestrian Detection, ACM MM 2023, Nuo Chen et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3613444)]
3. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI2022, Jung Uk Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
4. Mlpd: Multi-label pedestrian detector in multispectral domain, IEEE Robotics and Automation Letters 2021, Jiwon Kim et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9496129)]
5. Weakly Aligned Feature Fusion for Multimodal Object Detection, ITNNLS 2021, Lu Zhang et al. [[PDF](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)]
6. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
7. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al.
[[PDF](https://arxiv.org/abs/1901.02645)]
[[Code](https://github.com/luzhang16/AR-CNN)]
1. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.
[[PDF](https://arxiv.org/abs/1808.04818)]
[[Code](https://github.com/Li-Chengyang/MSDS-RCNN)]

### Single Modality
1. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI 2022, Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
2. Robust Thermal Infrared Pedestrian Detection By Associating Visible Pedestrian Knowledge, ICASSP 2022, Sungjune Park et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9746886)]
3. Low-cost Multispectral Scene Analysis with Modality Distillation, Zhang Heng et al. [[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_Low-Cost_Multispectral_Scene_Analysis_With_Modality_Distillation_WACV_2022_paper.pdf)]
4. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
4. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]

### Domain Adaptation
1. D3T: Distinctive Dual-Domain Teacher Zigzagging Across RGB-Thermal Gap for Domain-Adaptive Object Detection, CVPR 2024, Dinh Phat Do et al., [[PDF](https://arxiv.org/pdf/2403.09359.pdf)][[Code](https://github.com/EdwardDo69/D3T)]
2. Unsupervised Domain Adaptation for Multispectral Pedestrian Detection, CVPR 2019 Workshop , Dayan Guan et al.
[[PDF](https://arxiv.org/abs/1904.03692)]
[[Code](https://github.com/dayanguan/unsupervised_multispectral_pedestrian_detectio)]
1. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019, Y. Cao et al. Information Fusion 2019, [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)]
[[Code](https://github.com/Huaqing-lucky/unsupervised_multispectral_pedestrian_detection)]
1. Learning crossmodal deep representations for robust pedestrian detection, CVPR 2017, D. Xu et al.[[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)][[Code](https://github.com/danxuhk/CMT-CNN)]

--------------------------------------------------------------------------------------

# RGB-T Vehicle Detection
## Datasets
DroneVehicle: partially aligned [[link](https://github.com/VisDrone/DroneVehicle)]

VEDAI: strictly aligned [[link](https://downloads.greyc.fr/vedai/)]

## Papers
1. ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection, Pattern Recognition 2024, Shen Jifeng et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10225383)][[Code](https://github.com/chanchanchan97/ICAFusion)]
2. Improving RGB-infrared object detection with cascade alignment-guided transformer, Information Fusion 2024, Maoxun Yuan et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253524000241)]
2. Multispectral Object Detection via Cross-Modal Conflict-Aware Learning, ACM MM 2023, Xiao He et al. [[PDF](https://dl.acm.org/doi/10.1145/3581783.3612651)][[Code](https://github.com/hexiao-cs/CALNet-Dronevehicle)]
3. LRAF-Net: Long-Range Attention Fusion Network for Visible–Infrared Object Detection, TNNLS 2023, Haolong Fu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10144688)]
4. GF-Detection: Fusion with GAN of Infrared and Visible Images for Vehicle Detection at Nighttime, Remote Sensing 2022, Peng Gao et al. [[PDF](https://www.mdpi.com/2072-4292/14/12/2771)]
5. Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery, Pattern Recognition, Qingyun Fang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322002679)]
6. Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection, ECCV 2022, Maoxun Yuan et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690501.pdf)]
7. Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning, TCSVT 2022, Yiming Sun [[PDF](https://ieeexplore.ieee.org/abstract/document/9759286)][[Code](https://github.com/SunYM2020/UA-CMDet)]
8. Improving RGB-Infrared Object Detection by Reducing Cross-Modality Redundancy, Remote Sensing 2022, Qingwang Wang et al. [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]

--------------------------------------------------------------------------------------

# RGB-T-Semantic-segmentation
## Datasets
[Utokyo dataset](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/), [PST900 dataset](https://github.com/ShreyasSkandanS/pst900_thermal_rgb), [SemanticRT dataset](https://github.com/jiwei0921/SemanticRT)
## Papers
1. Context-Aware Interaction Network for RGB-T Semantic Segmentation, TMM 2024, Ying Lv et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10379106)][[Code](https://github.com/YingLv1106/CAINet)]
2. Complementarity-aware cross-modal feature fusion network for RGB-T semantic segmentation, PR 2023, Wei Wu et al.,  [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322003624)]
3. MMSMCNet: Modal Memory Sharing and Morphological Complementary Networks for RGB-T Urban Scene Semantic Segmentation, TCSVT 2023, Wujie Zhou et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10123009)][[Code](https://github.com/2021nihao/MMSMCNet)]
4. SGFNet: Semantic-Guided Fusion Network for RGB-Thermal Semantic Segmentation, TCSVT 2023, Yike Wang et al., [[PDF](https://ieeexplore.ieee.org/document/10138593)][[Code](https://github.com/kw717/SGFNet)]
5. DBCNet: Dynamic Bilateral Cross-Fusion Network for RGB-T Urban Scene Understanding in Intelligent Vehicles, TCYB 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10217340)]
6. Explicit Attention-Enhanced Fusion for RGB-Thermal Perception Tasks, RAL 2023, Mingjian Liang et al., [[PDF](https://ieeexplore.ieee.org/document/10113725)][[Code](https://github.com/freeformrobotics/eaefnet)]
7. Embedded Control Gate Fusion and Attention Residual Learning for RGB–Thermal Urban Scene Parsing, TITS 2023, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/10041960)]
8. UTFNet: Uncertainty-Guided Trustworthy Fusion Network for RGB-Thermal Semantic Segmentation, GRSL 2023, Qingwang Wang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10273407)][[Code](https://github.com/KustTeamWQW/UTFNet)]
9. Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning, arxiv 2023, Shaohua Dong et al., [[PDF](https://arxiv.org/pdf/2312.00360.pdf)][[Code](https://github.com/shaohuadong2021/dplnet?tab=readme-ov-file)]
10. Edge-aware guidance fusion network for RGB–thermal scene parsing, AAAI 2022, Wujie Zhou et al., [[PDF](https://arxiv.org/abs/2112.05144)][[Code](https://github.com/ShaohuaDong2021/EGFNet)]
11. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with Transformers, TITS 2022, Jiaming Zhang et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/10231003)][[Code](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)]
12. RGB-T Semantic Segmentation with Location, Activation, and Sharpening, TCSVT 2022, Gongyang Li et al., [[PDF](https://ieeexplore.ieee.org/document/9749834)][[Code](https://github.com/ShaohuaDong2021/MTANet)]
13. A Feature Divide-and-Conquer Network for RGB-T Semantic Segmentation, TCSVT 2022, Shenlu Zhao et al., [[PDF](https://ieeexplore.ieee.org/document/9987529)]
14. MTANet: Multitask-Aware Network with Hierarchical Multimodal Fusion for RGB-T Urban Scene Understanding, TIV 2022, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9900351)][[Code](https://github.com/MathLee/LASNet)]
15. ABMDRNet: Adaptive-weighted Bi-directional Modality Difference Reduction Network for RGB-T Semantic Segmentation, CVPR 2021, Qiang Zhang et al., [[PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_ABMDRNet_Adaptive-Weighted_Bi-Directional_Modality_Difference_Reduction_Network_for_RGB-T_Semantic_CVPR_2021_paper.pdf)]
16. GMNet: Graded-Feature Multilabel-Learning Network for RGB-Thermal Urban Scene Semantic Segmentation, TIP 2021, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9531449)][[Code](https://github.com/Jinfu0913/GMNet)]
17. MFFENet: Multiscale Feature Fusion and Enhancement Network for RGBThermal Urban Road Scene Parsing, TMM 2021, Wujie Zhou et al., [[PDF](https://ieeexplore.ieee.org/document/9447924)]
18. FEANet: Feature-Enhanced Attention Network for RGB-Thermal Real-time Semantic Segmentation, IROS 2021, Fuqin Deng et al., [[PDF](https://arxiv.org/pdf/2110.08988.pdf)][[Code](https://github.com/matrixgame2018/FEANet)]
19. HeatNet: Bridging the Day-Night Domain Gap in Semantic Segmentation with Thermal Images, IROS 2021, Johan Vertens et al., [[PDF](https://ieeexplore.ieee.org/abstract/document/9341192)]
20. PST900: RGB-Thermal Calibration, Dataset and Segmentation Network, ICRA 2020, Shreyas S. Shivakumar et al., [[PDF](https://ieeexplore.ieee.org/document/9196831)][[Code](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)]
21. RTFNet: RGB-Thermal Fusion Network for Semantic Segmentation of Urban Scenes, RAL 2019, Yuxiang Sun et al., [[PDF](https://ieeexplore.ieee.org/document/8666745)][[Code](https://github.com/yuxiangsun/RTFNet)]
22. MFNet: Towards Real-Time Semantic Segmentation for Autonomous Vehicles with Multi-Spectral Scenes, IROS 2019, Qishen Ha et al., [[PDF](https://ieeexplore.ieee.org/document/8206396)][[Code](https://github.com/haqishen/MFNet-pytorch)]


--------------------------------------------------------------------------------------

# RGB-T Salient Object Detection
## Datasets
VT821 Dataset [[PDF](https://link.springer.com/content/pdf/10.1007%2F978-981-13-1702-6_36.pdf)][[link](https://drive.google.com/file/d/0B4fH4G1f-jjNR3NtQUkwWjFFREk/view?resourcekey=0-Kgoo3x0YJW83oNSHm5-LEw)], VT1000 Dataset [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296)][[link](https://drive.google.com/file/d/1NCPFNeiy1n6uY74L0FDInN27p6N_VCSd/view)], VT5000 Dataset [[PDF](https://arxiv.org/pdf/2007.03262.pdf)][[link]( https://pan.baidu.com/s/1ksuUr3cr6_-fZAsSUp0n0w)[9yqv]]
## Papers
### Domain Adaptation
1. Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation, AAAI 2020, Shaoyue Song et al.[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/6879)]
2. Deep Domain Adaptation Based Multi-spectral Salient Object Detection, TMM 2020, Shaoyue Song et al.[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9308922)]
### Fusion Architecture
1. Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection, TIP 2021, Wu, Zhengtao et al.[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9454273)]

--------------------------------------------------------------------------------------

# RGB-T Crowd Counting
## Datasets
RGBT-CC[[link](http://lingboliu.com/RGBT_Crowd_Counting.html)], DroneRGBT [[link](https://pan.baidu.com/s/1GSuLVk4siUZwlhDhOeVB-Q)[vvqt]]
## Papers
### Domain Adaptation
1. RGB-T Crowd Counting from Drone: A Benchmark and MMCCN Network, ACCV2020, Tao Peng et al. [[PDF](https://openaccess.thecvf.com/content/ACCV2020/papers/Peng_RGB-T_Crowd_Counting_from_Drone_A_Benchmark_and_MMCCN_Network_ACCV_2020_paper.pdf)][[Code](https://github.com/VisDrone/DroneRGBT)]
### Fusion Architecture
1. CCANet: A Collaborative Cross-modal Attention Network for RGB-D Crowd Counting, TMM2023, Yanbo Liu et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10086642)]
2. MC3Net: Multimodality Cross-Guided Compensation Coordination Network for RGB-T Crowd Counting, TITS 2023, Wujie Zhou et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10285562)]
3.  RGB-T Multi-Modal Crowd Counting Based on Transformer, BMVC 2022, Zhengyi Liu et al. [[PDF](https://bmvc2022.mpi-inf.mpg.de/0427.pdf)]
4. Spatio-channel Attention Blocks for Cross-modal Crowd Counting, ACCV2022, Youjia Zhang et al. [[PDF](https://openaccess.thecvf.com/content/ACCV2022/html/Zhang_Spatio-channel_Attention_Blocks_for_Cross-modal_Crowd_Counting_ACCV_2022_paper.pdf)]
5. DEFNet: Dual-Branch Enhanced Feature Fusion Network for RGB-T Crowd Counting, TITS 2022, Zhou, Wujie et al. [[PDF](https://ieeexplore.ieee.org/document/9889192)]
6. MAFNet: A Multi-Attention Fusion Network for RGB-T Crowd Counting, arxiv2022, Pengyu Chen et al. [[PDF](https://arxiv.org/pdf/2208.06761.pdf)]
7. Multimodal Crowd Counting with Mutual Attention Transformers, ICME 2022, Wu, Zhengtao et al.  [[PDF](https://ieeexplore.ieee.org/abstract/document/9859777)]
8. Conditional RGB-T Fusion for Effective Crowd Counting, ICIP 2022, Esha Pahwa et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9897548)]
9. Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting, CVPR2021, Lingbo Liu et al. [[PDF](https://arxiv.org/pdf/2012.04529.pdf)][[Code](https://github.com/chen-judge/RGBTCrowdCounting)]

--------------------------------------------------------------------------------------

# RGB-T Fusion Tracking
## Datasets
GTOT [[PDF](https://ieeexplore.ieee.org/document/7577747)][[link](https://github.com/mmic-lcl/Datasets-and-benchmark-code)], RGBT234 Dataset [[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320319302808)][[link](https://sites.google.com/view/ahutracking001/)], LasHeR Dataset [[PDF](https://arxiv.org/abs/2104.13202)][[link](https://github.com/mmic-lcl/Datasets-and-benchmark-code)]
## Papers
1. MTNet: Learning Modality-aware Representation with Transformer for RGBT Tracking, ICME 2023, Ruichao Hou et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10219799)]
2. Visual Prompt Multi-Modal Tracking, CVPR 2023, Jiawen Zhu et al. [[PDF](https://arxiv.org/abs/2303.10826)][[Code](https://github.com/jiawen-zhu/ViPT)]
3. Efficient RGB-T Tracking via Cross-Modality Distillation, CVPR 2023, Zhang, Tianlu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10205202)]
4. Bridging Search Region Interaction with Template for RGB-T Tracking, Hui, CVPR2023, Tianrui et al.[[PDF](https://ieeexplore.ieee.org/document/10203113)][[Code](https://github.com/RyanHTR/TBSI)]
5. Jointly Modeling Motion and Appearance Cues for Robust RGB-T Tracking, TIP2021, Zhang, Pengyu et al. [[PDF](https://ieeexplore.ieee.org/document/9364880)] TIP 2022, Tu, Zhengzheng et al., [[PDF](https://ieeexplore.ieee.org/document/9617143)]
6. RGBT tracking via reliable feature configuration, SCIS 2022, Tu, Zhengzheng et al. [[PDF](https://link.springer.com/article/10.1007/s11432-020-3160-5)]
7. Attribute-Based Progressive Fusion Network for RGBT Tracking, AAAI 2022, Xiao Yun et al. [[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/20187/19946)][[Code](https://github.com/yangmengmeng1997/APFNet)]
8. Dense Feature Aggregation and Pruning for RGBT Tracking, ACM Multimedia 2022, Yabin Zhu et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3343031.3350928)]
9. Prompting for Multi-Modal Tracking, ACM Multimedia 2022, Jinyu Yang et al. [[PDF](https://arxiv.org/abs/2207.14571)]
10. Learning Adaptive Attribute-Driven Representation for Real-Time RGB-T Tracking, IJCV 2021, Zhang, Pengyu et al. [[PDF](https://link.springer.com/article/10.1007/s11263-021-01495-3)]
11. Quality-Aware Feature Aggregation Network for Robust RGBT Tracking, TIV 2021, Zhu, Yabin, [[PDF](https://ieeexplore.ieee.org/abstract/document/9035457)]
12. Challenge-Aware RGBT Tracking, ECCV 2020, Li Chenglong et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_14)]
13. Object fusion tracking based on visible and infrared images: A comprehensive review, Information Fusion 2020, Zhang, Xingchen et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253520302657)]
14. RGB-T object tracking: Benchmark and baseline, Pattern Recognition 2019, Li, Chenglong et al., [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320319302808)]
15. Cross-Modal Pattern-Propagation for RGB-T Tracking, CVPR2020, Chaoqun Wang et al., [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Modal_Pattern-Propagation_for_RGB-T_Tracking_CVPR_2020_paper.pdf)]


--------------------------------------------------------------------------------------