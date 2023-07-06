# Awesome RGB-T Feature Fusion [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
A collection of RGB-T-Feature-Fusion methods (deep learning methods mainly), codes, and datasets.  
The main directions involved are Multispectral Pedestrian, RGB-T Vehicle Detection, RGB-T Crowd Counting, RGB-T Fusion Tracking.  
Feel free to star and fork! We will continue to update this repository!  
## Contents  

1. [Multispectral Pedestrian](#Multispectral-Pedestrian)
2. [RGB-T Vehicle Detection](#RGB-T-Vehicle-Detection)
3. [RGB-T Crowd Counting](#RGB-T-Crowd-Counting)
4. [RGB-T Salient Object Detection](#RGB-T-Salient-Object-Detection)
5. [RGB-T Fusion Tracking](#RGB-T-Fusion-Tracking)

# Multispectral-Pedestrian
## Datasets and Annotations
[KAIST dataset](https://soonminhwang.github.io/rgbt-ped-detection/), [CVC-14 dataset](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/)
, [FLIR dataset](https://www.flir.cn/oem/adas/adas-dataset-form/), [LLVIP dataset](https://bupt-ai-cz.github.io/LLVIP/), [M<sup>3</sup>FD dataset](https://github.com/dlut-dimt/TarDAL)
 - Improved KAIST Testing Annotations provided by Liu et al.[Link to download](https://docs.google.com/forms/d/e/1FAIpQLSe65WXae7J_KziHK9cmX_lP_hiDXe7Dsl6uBTRL0AWGML0MZg/viewform?usp=pp_url&entry.1637202210&entry.1381600926&entry.718112205&entry.233811498) 
  - Sanitized KAIST Training Annotations provided by Li et al.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN) 
 - Improved KAIST Training Annotations provided by Zhang et al.[Link to download](https://github.com/luzhang16/AR-CNN) 
## Tools
- Evalutaion codes.[Link to download](https://github.com/Li-Chengyang/MSDS-RCNN/tree/master/lib/datasets/KAISTdevkit-matlab-wrapper)
- Annotation: vbb format->xml format.[Link to download](https://github.com/SoonminHwang/rgbt-ped-detection/tree/master/data/scripts)
## Papers
### Fusion Architecture
1. Multimodal Object Detection by Channel Switching and Spatial Attention, CVPRW 2023, Yue Cao et al. [[PDF](https://openaccess.thecvf.com/content/CVPR2023W/PBVS/papers/Cao_Multimodal_Object_Detection_by_Channel_Switching_and_Spatial_Attention_CVPRW_2023_paper.pdf)]
2. Illumination-Guided RGBT Object Detection With Inter- and Intra-Modality Fusion, IEEE Transactions on Instrumentation and Measurement, Yan Zhang et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10057437/)]
3. Multi-Modal Feature Pyramid Transformer for RGB-Infrared Object Detection, TITS 2023, Yaohui Zhu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10105844)]
4. Multiscale Cross-modal Homogeneity Enhancement and Confidence-aware Fusion for Multispectral Pedestrian Detection, TMM 2023, Ruimin Li et al. [[PDF](https://ieeexplore.ieee.org/document/10114594)]
5. HAFNet: Hierarchical Attentive Fusion Network for Multispectral Pedestrian Detection, Remote Sensing 2023, Peiran Peng et al. [[PDF](https://www.mdpi.com/2072-4292/15/8/2041)]
6. DetFusion: A Detection-driven Infrared and Visible Image Fusion Network, ACM Multimedia 2022, Yiming Sun et al. [[PDF](https://dl.acm.org/doi/pdf/10.1145/3503161.3547902)]
7. Multimodal Object Detection via Probabilistic Ensembling, ECCV2022, Yi-Ting Chen et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690139.pdf)]
8. Learning a Dynamic Cross-Modal Network for Multispectral Pedestrian Detection, ACM Multimedia 2022, Jin Xie et al. [[PDF](https://dl.acm.org/doi/abs/10.1145/3503161.3547895)]
9.  Confidence-aware Fusion using Dempster-Shafer Theory for Multispectral Pedestrian Detection, TMM 2022, Qing Li et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9739079)]
10. Attention-Guided Multi-modal and Multi-scale Fusion for Multispectral Pedestrian Detection, PRCV 2022, Wei Bao et al. [[PDF](https://link.springer.com/chapter/10.1007/978-3-031-18907-4_30)]
11. Improving RGB-Infrared Pedestrian Detection by Reducing Cross-Modality Redundancy, ICIP2022, Qingwang Wang et al.  [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]
12. Spatio-contextual deep network-based multimodal pedestrian detection for autonomous driving, IEEE Transactions on Intelligent Transportation Systems, Kinjal Dasgupta et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9706418)]
13. Adopting the YOLOv4 Architecture for Low-LatencyMultispectral Pedestrian Detection in Autonomous Driving, Sensors 2022, Kamil Roszyk et al. [[PDF](https://www.mdpi.com/1424-8220/22/3/1082)]
14. Deep Active Learning from Multispectral Data Through Cross-Modality Prediction Inconsistency, ICIP2021, Heng Zhang et al.[[PDF](https://ieeexplore.ieee.org/document/9506322)]
15. Attention Fusion for One-Stage Multispectral Pedestrian Detection, Sensors 2021, Zhiwei Cao et al. [[PDF](https://www.mdpi.com/1424-8220/21/12/4184)]
16. Uncertainty-Guided Cross-Modal Learning for Robust Multispectral Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, Jung Uk Kim et al. [[PDF](https://ieeexplore.ieee.org/document/9419080)]
17. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]
18. Guided Attentive Feature Fusion for Multispectral Pedestrian Detection, WACV 2021, Heng Zhang et al. [[PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)]
19. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
20. Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks, ICIP 2020, Heng Zhang et al. [[PDF](https://hal.archives-ouvertes.fr/hal-02872132/file/icip2020.pdf)]
21. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
22. Anchor-free Small-scale Multispectral Pedestrian Detection, BMVC 2020, Alexander Wolpert et al. [[PDF](https://arxiv.org/abs/2008.08418)][[Code](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)]
23. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al. [[PDF](https://arxiv.org/abs/1901.02645)][[Code](https://github.com/luzhang16/AR-CNN)]
24. Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pesdestrian Detecion, ISPRS Journal of Photogrammetry and Remote Sensing 2019, Yanpeng Cao et al.[[PDF](https://arxiv.org/abs/1902.05291)][[Code](https://github.com/dayanguan/realtime_multispectral_pedestrian_detection)]
25. Cross-modality interactive attention network for multispectral pedestrian detection, Information Fusion 2019, Lu Zhang et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S1566253518304111)][[Code](https://github.com/luzhang16/CIAN)]
26. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019,  Cao, Yanpeng et al.[[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)]
27. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.[[PDF](https://arxiv.org/abs/1808.04818)][[Code](https://github.com/Li-Chengyang/MSDS-RCNN)][[Project Link](https://li-chengyang.github.io/home/MSDS-RCNN/)]
28. Unified Multi-spectral Pedestrian Detection Based on Probabilistic Fusion Networks, Pattern Recognition 2018, Kihong Park et al.[[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0031320318300906)]
29. Multispectral Deep Neural Networks for Pedestrian Detection, BMVC 2016, Jingjing Liu et al.[[PDF](https://arxiv.org/abs/1611.02644)][[Code](https://github.com/denny1108/multispectral-pedestrian-py-faster-rcnn)]
30. Multispectral Pedestrian Detection Benchmark Dataset and Baseline, 2015, Soonmin Hwang et al.[[PDF](https://soonminhwang.github.io/rgbt-ped-detection/misc/CVPR15_Pedestrian_Benchmark.pdf)][[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

### Illumination Aware
1. IGT: Illumination-guided RGB-T object detection with transformers, Knowledge-Based Systems, Keyu Chen et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0950705123001739)]
2. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
3. Fusion of Multispectral Data Through Illumination-aware Deep Neural Networks for Pedestrian Detection, Information Fusion 2019, Dayan Guan et al.[[PDF](https://arxiv.org/abs/1802.09972)][[Code](https://github.com/dayanguan/illumination-aware_multispectral_pedestrian_detection/)]
4. Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection, Pattern Recognition 2018, Chengyang Li et al.[[PDF](https://arxiv.org/pdf/1802.09972.pdf)][[Code](https://github.com/Li-Chengyang/IAF-RCNN)]

### Feature Alignment
1. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI2022, Jung Uk Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
2. Mlpd: Multi-label pedestrian detector in multispectral domain, IEEE Robotics and Automation Letters 2021, Jiwon Kim et al. [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9496129)]
3. Weakly Aligned Feature Fusion for Multimodal Object Detection, ITNNLS 2021, Lu Zhang et al. [[PDF](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)]
3. Improving Multispectral Pedestrian Detection by Addressing Modality Imbalance Problems, ECCV 2020, Kailai Zhou et al. [[PDF](https://arxiv.org/pdf/2008.03043.pdf)][[Code](https://github.com/CalayZhou/MBNet)]
4. Weakly Aligned Cross-Modal Learning for Multispectral Pedestrian Detection, ICCV 2019, Lu Zhang et al.
[[PDF](https://arxiv.org/abs/1901.02645)]
[[Code](https://github.com/luzhang16/AR-CNN)]
5. Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation, BMVC 2018, Chengyang Li et al.
[[PDF](https://arxiv.org/abs/1808.04818)]
[[Code](https://github.com/Li-Chengyang/MSDS-RCNN)]

### Single Modality
1. Towards Versatile Pedestrian Detector with Multisensory-Matching and Multispectral Recalling Memory, AAAI 2022, Kim et al. [[PDF](https://www.aaai.org/AAAI22Papers/AAAI-8768.KimJ.pdf)]
2. Robust Thermal Infrared Pedestrian Detection By Associating Visible Pedestrian Knowledge, ICASSP 2022, Sungjune Park et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/9746886)]
3. Low-cost Multispectral Scene Analysis with Modality Distillation, Zhang Heng et al. [[PDF](https://openaccess.thecvf.com/content/WACV2022/papers/Zhang_Low-Cost_Multispectral_Scene_Analysis_With_Modality_Distillation_WACV_2022_paper.pdf)]
4. Task-conditioned Domain Adaptation for Pedestrian Detection in Thermal Imagery, ECCV 2020, My Kieu et al. [[PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670545.pdf)][[Code](https://github.com/mrkieumy/task-conditioned)]
4. Deep Cross-modal Representation Learning and Distillation for Illumination-invariant Pedestrian Detection, IEEE Transactions on Circuits and Systems for Video Technology 2021, T. Liu et al. [[PDF](https://ieeexplore.ieee.org/document/9357413/)]

### Unsupervised Domain Adaptation
1. Unsupervised Domain Adaptation for Multispectral Pedestrian Detection, CVPR 2019 Workshop , Dayan Guan et al.
[[PDF](https://arxiv.org/abs/1904.03692)]
[[Code](https://github.com/dayanguan/unsupervised_multispectral_pedestrian_detectio)]
2. Pedestrian detection with unsupervised multispectral feature learning using deep neural networks, Information Fusion 2019, Y. Cao et al. Information Fusion 2019, [[PDF](https://www.sciencedirect.com/science/article/pii/S1566253517305948)]
[[Code](https://github.com/Huaqing-lucky/unsupervised_multispectral_pedestrian_detection)]
3. Learning crossmodal deep representations for robust pedestrian detection, CVPR 2017, D. Xu et al.[[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Learning_Cross-Modal_Deep_CVPR_2017_paper.pdf)][[Code](https://github.com/danxuhk/CMT-CNN)]

# RGB-T Vehicle Detection
## Datasets
DroneVehicle[[link](https://github.com/VisDrone/DroneVehicle)], Multispectral Datasets for Detection and Segmentation[[link](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)]
## papers
1. LRAF-Net: Long-Range Attention Fusion Network for Visible–Infrared Object Detection, TNNLS 2023, Haolong Fu et al. [[PDF](https://ieeexplore.ieee.org/abstract/document/10144688)]
2. GF-Detection: Fusion with GAN of Infrared and Visible Images for Vehicle Detection at Nighttime, Remote Sensing 2022, Peng Gao et al. [[PDF](https://www.mdpi.com/2072-4292/14/12/2771)]
3. Cross-modality attentive feature fusion for object detection in multispectral remote sensing imagery, Pattern Recognition, Qingyun Fang et al. [[PDF](https://www.sciencedirect.com/science/article/pii/S0031320322002679)]
4. Translation, Scale and Rotation: Cross-Modal Alignment Meets RGB-Infrared Vehicle Detection, ECCV 2022, Maoxun Yuan et al. [[PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690501.pdf)]
5. Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning, TCSVT 2022, Yiming Sun [[PDF](https://ieeexplore.ieee.org/abstract/document/9759286)]
6. Improving RGB-Infrared Object Detection by Reducing Cross-Modality Redundancy, Remote Sensing 2022, Qingwang Wang et al. [[PDF](https://www.mdpi.com/2072-4292/14/9/2020)]


# RGB-T Crowd Counting
## Datasets
RGBT-CC[[link](http://lingboliu.com/RGBT_Crowd_Counting.html)], DroneCrowd [[link](https://github.com/VisDrone/DroneCrowd)]
## papers
### Domain Adaptation
1. RGB-T Crowd Counting from Drone: A Benchmark and MMCCN Network, ACCV2020, Tao Peng et al. [[PDF](https://openaccess.thecvf.com/content/ACCV2020/papers/Peng_RGB-T_Crowd_Counting_from_Drone_A_Benchmark_and_MMCCN_Network_ACCV_2020_paper.pdf)][[Code](https://github.com/VisDrone/DroneRGBT)]
### Fusion Architecture
1. MAFNet: A Multi-Attention Fusion Network for RGB-T Crowd Counting, arxiv2022, Pengyu Chen et al. [[PDF](https://arxiv.org/pdf/2208.06761.pdf)]
2. Multimodal Crowd Counting with Mutual Attention Transformers, ICME 2022, Wu, Zhengtao et al.  [[PDF](https://ieeexplore.ieee.org/abstract/document/9859777)]
3. Cross-Modal Collaborative Representation Learning and a Large-Scale RGBT Benchmark for Crowd Counting, CVPR2021, Lingbo Liu et al. [[PDF](https://arxiv.org/pdf/2012.04529.pdf)][[Code](https://github.com/chen-judge/RGBTCrowdCounting)]

# RGB-T Salient Object Detection
## Datasets
VT821 Dataset [[PDF](https://link.springer.com/content/pdf/10.1007%2F978-981-13-1702-6_36.pdf)][[link](https://drive.google.com/file/d/0B4fH4G1f-jjNR3NtQUkwWjFFREk/view?resourcekey=0-Kgoo3x0YJW83oNSHm5-LEw)], VT1000 Dataset [[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8744296)][[link](https://drive.google.com/file/d/1NCPFNeiy1n6uY74L0FDInN27p6N_VCSd/view)], VT5000 Dataset [[PDF](https://arxiv.org/pdf/2007.03262.pdf)][[link]( https://pan.baidu.com/s/1ksuUr3cr6_-fZAsSUp0n0w)[9yqv]]
## papers
### Domain Adaptation
1. Multi-Spectral Salient Object Detection by Adversarial Domain Adaptation, AAAI 2020, Shaoyue Song et al.[[PDF](https://ojs.aaai.org/index.php/AAAI/article/view/6879)]
2. Deep Domain Adaptation Based Multi-spectral Salient Object Detection, TMM 2020, Shaoyue Song et al.[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9308922)]
### Fusion Architecture
Multi-Interactive Dual-Decoder for RGB-Thermal Salient Object Detection, TIP 2021, Wu, Zhengtao et al.[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9454273)]
# RGB-T Fusion Tracking
## papers
1. Visual Prompt Multi-Modal Tracking, CVPR 2023, Jiawen Zhu et al. [[PDF](https://arxiv.org/abs/2303.10826)][[Code](https://github.com/jiawen-zhu/ViPT)]
2. Prompting for Multi-Modal Tracking, ACM Multimedia 2022, Jinyu Yang et al. [[PDF](https://arxiv.org/abs/2207.14571)]