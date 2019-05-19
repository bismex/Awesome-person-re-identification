# Awesome Person Re-identification (Person Re-ID) [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This is a repository for organizing articles related to person re-identification. Most papers are linked to the pdf address provided by "arXiv" or "Openaccess". However, some papers require an academic license to browse. For example, IEEE, springer, and elsevier journal, etc.

---

## Table of Contents

- [1. Statistics](#statistics)
- [2. Survey](#survey)
- [3. CVPR2019](#cvpr-2019)
- [4. Unsupervised Person Re-ID (+ Transfer / Semi-supervised learning)](#unsupervised-person-re-id)
- [5. Supervised Person Re-ID](#supervised-person-re-id)
- [6. Person search](#person-search)
- [7. Others](#others)
- [8. Datasets](#datasets)

---

## Statistics


| Conference  | Link | #Total | Unsupervised reID | Supervised reID | Person search | Others | Datasets |
|---           |---   |---|---|---|---|---|---|
| CVPR2019 | will be updated soon | 29 | ? | ? | ? | ? | ? |
| ECCV2018 | [Click](http://openaccess.thecvf.com/ECCV2018.py)  | 19 | 4 | 8 | 4 | 3 | 0 |
| CVPR2018 | [Click](http://openaccess.thecvf.com/CVPR2018.py)  | 31 | 5 | 23 | 0 | 2 | 1 |
| ICCV2017 | [Click](http://openaccess.thecvf.com/ICCV2017.py)  | 16 | 7 | 6 | 1 | 2 | 0 |
| CVPR2017 | [Click](http://openaccess.thecvf.com/CVPR2017.py)  | 16 | 2 | 9 | 1 | 3 | 1 |

---

## Survey

#### *"Person Re-identification: Past, Present and Future"*, arXiv 2016 [[paper](https://arxiv.org/pdf/1610.02984.pdf)]

#### *"A survey of approaches and trends in person re-identification"*, Image and Vision Computing 2014 [[paper](https://ac.els-cdn.com/S0262885614000262/1-s2.0-S0262885614000262-main.pdf?_tid=de6eee6c-08e6-486c-9d7a-18d2e0c30091&acdnat=1539565884_afc3e4f2e7068a620c9fbfde6129d35d)]

#### *"Appearance Descriptors for Person Re-identification: a Comprehensive Review"*, arXiv 2013 [[paper](https://arxiv.org/abs/1307.5748)]

---


## CVPR 2019
### (not categorized yet)

#### *"Joint Discriminative and Generative Learning for Person Re-identification"*
#### *"Densely Semantically Aligned Person Re-Identification"*
#### *"Generalizable Person Re-identification by Domain-Invariant Mapping Network"*
#### *"Re-Identification with Consistent Attentive Siamese Networks"*
#### *"Distilled Person Re-identification: Towards a More Scalable System"*
#### *"Weakly Supervised Person Re-Identification"*
#### *"Patch Based Discriminative Feature Learning for Unsupervised Person Re-identification"*
#### *"Unsupervised Person Re-identification by Soft Multilabel Learning"*
#### *"Invariance Matters: Exemplar Memory for Domain Adaptive Person Re-identification"*
#### *"Re-ranking via Metric Fusion for Object Retrieval and Person Re-identification"*
#### *"Progressive Pose Attention Transfer for Person Image Generation"*
#### *"Unsupervised Person Image Generation with Semantic Parsing Transformation"*
#### *"Learning to Reduce Dual-level Discrepancy for Infrared-Visible Person Re-identification"*
#### *"Text Guided Person Image Synthesis"*
#### *"Re-Identification Supervised 3D Texture Generation"*
#### *"Learning Context Graph for Person Search"*
#### *"Query-guided End-to-End Person Search"*
#### *"Multi-person Articulated Tracking with Spatial and Temporal Embeddings"*
#### *"Dissecting Person Re-identification from the Viewpoint of Viewpoint"*
#### *"Towards Rich Feature Discovery with Class Activation Maps Augmentation for Person Re-Identification"*
#### *"AANet: Attribute Attentio Network for Person Re-Identification"*
#### *"VRSTC: Occlusion-Free Video Person Re-Identification"*
#### *"Adaptive Transfer Network for Cross-Domain Person Re-Identification"*
#### *"Pyramidal Person Re-IDentification via Multi-Loss Dynamic Training"*
#### *"Interaction-and-Aggregation Network for Person Re-identification"*
#### *"Skin-based identification from multispectral image data using CNNs"*
#### *"Feature Distance Adversarial Network for Vehicle Re-Identification"*
#### *"Part-regularized Near-Duplicate Vehicle Re-identification"*
#### *"CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification"*

---

## Unsupervised Person Re-ID 
### (+Transfer learning)
### (+Semi-supervised learning)

### [ECCV2018]


#### *"Domain Adaptation through Synthesis for Unsupervised Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper.pdf)]
- Performance
  - Rank-1 (Market1501) : 65.7 (Single query)

#### *"Unsupervised Person Re-identification by Deep Learning Tracklet Association"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Minxian_Li_Unsupervised_Person_Re-identification_ECCV_2018_paper.pdf)]
- Performance
  - Rank-1 (Market1501) : 63.7 (ResNet-50. Images resized to 256x128.)
- Summary
  - Sparse space-time tracklet (SSTT) sampling
    - 1) Sparse temporal sampling
    - 2) Sparse spatial sampling
  - Tracklet association unsupervised deep learning (TAUDL) in an end-to-end manner.
    - 1) Per-camera tracklet discrimination learning
    - 2) Cross-camera tracklet association learning

#### *"Generalizing A Person Retrieval Model Hetero- and Homogeneously"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zhun_Zhong_Generalizing_A_Person_ECCV_2018_paper.pdf)] [[Github](https://github.com/zhunzhong07/HHL)]
- Performance
  - Rank-1 (Market1501) : 62.2(Single query, source domain : DukeMTMC) / 56.8(Single query, source domain : CUHK03)

#### *"Robust Anchor Embedding for Unsupervised Video Person Re-Identification in the Wild"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Mang_YE_Robust_Anchor_Embedding_ECCV_2018_paper.pdf)]

### [CVPR2018]

#### *"Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.pdf)] [[Github](https://github.com/ahangchen/TFusion)]
- Performance
  - Rank-1 (Market1501) : 60.7(source domain : CUHK01) / 59.2(source domain : VIPeR) / 58.2(source domain : GRID)

#### *"Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Transferable_Joint_Attribute-Identity_CVPR_2018_paper.pdf)]
- Performance
  - Rank-1 (Market1501) : 58.2(source domain : DukeMTMC)

#### *"Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Deng_Image-Image_Domain_Adaptation_CVPR_2018_paper.pdf)] [[Github](https://github.com/Simon4Yan/Learning-via-Translation)]
- Rank-1 (Market1501) : 58.1

#### *"Disentangled Person Image Generation"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ma_Disentangled_Person_Image_CVPR_2018_paper.pdf)]
- Performance
  - Rank-1 (Market1501) : 35.5

#### *"Exploit the Unknown Gradually: One-Shot Video-Based Person Re-Identification by Stepwise Learning"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Exploit_the_Unknown_CVPR_2018_paper.pdf)] [[Github](https://github.com/Yu-Wu/Exploit-Unknown-Gradually)] [[Homepage](https://yu-wu.net/publication/cvpr2018-oneshot-reid/)]


### [ICCV2017]

#### *"Cross-view Asymmetric Metric Learning for Unsupervised Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Yu_Cross-View_Asymmetric_Metric_ICCV_2017_paper.pdf)] [[Github](https://github.com/KovenYu/CAMEL)]
- Performance
  - Rank-1 (Market1501) : 54.5(Multiple query)

#### *"Efficient Online Local Metric Adaptation via Negative Samples for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Efficient_Online_Local_ICCV_2017_paper.pdf)]
- Performance
  - Rank-1 (Market1501) : 51.5(Multiple query) / 40.9(Single query)

#### *"SHaPE: A Novel Graph Theoretic Algorithm for Making Consensus-based Decisions in Person Re-identification Systems"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Barman_SHaPE_A_Novel_ICCV_2017_paper.pdf)]

#### *"Stepwise Metric Promotion for Unsupervised Video Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Stepwise_Metric_Promotion_ICCV_2017_paper.pdf)]

#### *"Group Re-Identification via Unsupervised Transfer of Sparse Features Encoding"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lisanti_Group_Re-Identification_via_ICCV_2017_paper.pdf)]

#### *"Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Unlabeled_Samples_Generated_ICCV_2017_paper.pdf)] [[Github](https://github.com/layumi/Person-reID_GAN)]

#### *"Dynamic Label Graph Matching for Unsupervised Video Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ye_Dynamic_Label_Graph_ICCV_2017_paper.pdf)] [[Github](https://github.com/mangye16/dgm_re-id)]


### [CVPR2017]

#### *"One-Shot Metric Learning for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bak_One-Shot_Metric_Learning_CVPR_2017_paper.pdf)]

#### *"Unsupervised Adaptive Re-Identification in Open World Dynamic Camera Networks"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Panda_Unsupervised_Adaptive_Re-Identification_CVPR_2017_paper.pdf)]


### [arXiv]

#### *"Unsupervised Person Re-identification: Clustering and Fine-tuning"* [[paper](https://arxiv.org/pdf/1705.10444.pdf)] [[Github](https://github.com/hehefan/Unsupervised-Person-Re-identification-Clustering-and-Fine-tuning)]
- Performance
  - Rank-1 (Market1501) : 41.9 (Single query, source domain : CUHK03)



---

## Supervised Person Re-ID

### [ECCV2018] 

#### *"Maximum Margin Metric Learning Over Discriminative Nullspace for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/T_M_Feroz_Ali_Maximum_Margin_Metric_ECCV_2018_paper.pdf)]

#### *"Person Re-identification with Deep Similarity-Guided Graph Neural Network"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yantao_Shen_Person_Re-identification_with_ECCV_2018_paper.pdf)]

#### *"Pose-Normalized Image Generation for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xuelin_Qian_Pose-Normalized_Image_Generation_ECCV_2018_paper.pdf)]

#### *"Improving Deep Visual Representation for Person Re-identification by Global and Local Image-language Association"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Dapeng_Chen_Improving_Deep_Visual_ECCV_2018_paper.pdf)]

#### *"Hard-Aware Point-to-Set Deep Metric for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Rui_Yu_Hard-Aware_Point-to-Set_Deep_ECCV_2018_paper.pdf)]

#### *"Part-Aligned Bilinear Representations for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper.pdf)] [[Github](https://github.com/yuminsuh/part_bilinear_reid)]
- Summary
  - Two stream network (Appearance map extractor, Part map extractor-OpenPose)
  - Aggregator : bilinear pooling (better than concat + Ave. pool + linear)

#### *"Mancs: A Multi-task Attentional Network with Curriculum Sampling for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Cheng_Wang_Mancs_A_Multi-task_ECCV_2018_paper.pdf)]

#### *"Beyond Part Models: Person Retrieval with Refined Part Pooling (and A Strong Convolutional Baseline)"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yifan_Sun_Beyond_Part_Models_ECCV_2018_paper.pdf)]

### [CVPR2018]

#### *"Diversity Regularized Spatiotemporal Attention for Video-based Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Diversity_Regularized_Spatiotemporal_CVPR_2018_paper.pdf)]

#### *"A Pose-Sensitive Embedding for Person Re-Identification with Expanded Cross Neighborhood Re-Ranking"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.pdf)]

#### *"Human Semantic Parsing for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Kalayeh_Human_Semantic_Parsing_CVPR_2018_paper.pdf)]

#### *"Video Person Re-identification with Competitive Snippet-similarity Aggregation and Co-attentive Snippet Embedding"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Video_Person_Re-Identification_CVPR_2018_paper.pdf)]

#### *"Mask-guided Contrastive Attention Model for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.pdf)]

#### *"Person Re-identification with Cascaded Pairwise Convolutions"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Person_Re-Identification_With_CVPR_2018_paper.pdf)]

#### *"Multi-Level Factorisation Net for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chang_Multi-Level_Factorisation_Net_CVPR_2018_paper.pdf)]

#### *"Attention-Aware Compositional Network for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Attention-Aware_Compositional_Network_CVPR_2018_paper.pdf)]

#### *"Deep Group-shuffling Random Walk for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Deep_Group-Shuffling_Random_CVPR_2018_paper.pdf)]

#### *"Harmonious Attention Network for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Harmonious_Attention_Network_CVPR_2018_paper.pdf)]

#### *"Efficient and Deep Person Re-Identification using Multi-Level Similarity"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Guo_Efficient_and_Deep_CVPR_2018_paper.pdf)]

#### *"Pose Transferrable Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Pose_Transferrable_Person_CVPR_2018_paper.pdf)]

#### *"Adversarially Occluded Samples for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Huang_Adversarially_Occluded_Samples_CVPR_2018_paper.pdf)]

#### *"Camera Style Adaptation for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhong_Camera_Style_Adaptation_CVPR_2018_paper.pdf)]

#### *"Dual Attention Matching Network for Context-Aware Feature Sequence based Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Si_Dual_Attention_Matching_CVPR_2018_paper.pdf)]

#### *"Easy Identification from Better Constraints: Multi-Shot Person Re-Identification from Reference Constraints"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Easy_Identification_From_CVPR_2018_paper.pdf)]

#### *"Eliminating Background-bias for Robust Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf)]

#### *"Features for Multi-Target Multi-Camera Tracking and Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)]

#### *"Multi-shot Pedestrian Re-identification via Sequential Decision Making"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Multi-Shot_Pedestrian_Re-Identification_CVPR_2018_paper.pdf)]

#### *"End-to-End Deep Kronecker-Product Matching for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_End-to-End_Deep_Kronecker-Product_CVPR_2018_paper.pdf)]

#### *"Deep Spatial Feature Reconstruction for Partial Person Re-identification: Alignment-free Approach"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Deep_Spatial_Feature_CVPR_2018_paper.pdf)]

#### *"Resource Aware Person Re-identification across Multiple Resolutions"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Resource_Aware_Person_CVPR_2018_paper.pdf)]

#### *"Group Consistent Similarity Learning via Deep CRF for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Group_Consistent_Similarity_CVPR_2018_paper.pdf)]

### [ICCV2017]

#### *"A Two Stream Siamese Convolutional Neural Network For Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chung_A_Two_Stream_ICCV_2017_paper.pdf)]

#### *"Learning View-Invariant Features for Person Identification in Temporally Synchronized Videos Taken by Wearable Cameras"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zheng_Learning_View-Invariant_Features_ICCV_2017_paper.pdf)]

#### *"Deeply-Learned Part-Aligned Representations for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhao_Deeply-Learned_Part-Aligned_Representations_ICCV_2017_paper.pdf)]

#### *"Pose-driven Deep Convolutional Model for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Su_Pose-Driven_Deep_Convolutional_ICCV_2017_paper.pdf)]

#### *"Jointly Attentive Spatial-Temporal Pooling Networks for Video-based Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Xu_Jointly_Attentive_Spatial-Temporal_ICCV_2017_paper.pdf)]

#### *"Multi-scale Deep Learning Architectures for Person Re-identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qian_Multi-Scale_Deep_Learning_ICCV_2017_paper.pdf)]

### [CVPR2017]

#### *"Learning Deep Context-Aware Features Over Body and Latent Parts for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Learning_Deep_Context-Aware_CVPR_2017_paper.pdf)]

#### *"Beyond Triplet Loss: A Deep Quadruplet Network for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Beyond_Triplet_Loss_CVPR_2017_paper.pdf)]

#### *"Spindle Net: Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhao_Spindle_Net_Person_CVPR_2017_paper.pdf)]

#### *"Re-Ranking Person Re-Identification With k-Reciprocal Encoding"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf)]

#### *"Scalable Person Re-Identification on Supervised Smoothed Manifold"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Scalable_Person_Re-Identification_CVPR_2017_paper.pdf)]

#### *"Point to Set Similarity Based Deep Feature Learning for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_Point_to_Set_CVPR_2017_paper.pdf)]

#### *"Fast Person Re-Identification via Cross-Camera Semantic Binary Transformation"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Fast_Person_Re-Identification_CVPR_2017_paper.pdf)]

#### *"See the Forest for the Trees: Joint Spatial and Temporal Recurrent Neural Networks for Video-Based Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_See_the_Forest_CVPR_2017_paper.pdf)]

#### *"Consistent-Aware Deep Learning for Person Re-Identification in a Camera Network"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Consistent-Aware_Deep_Learning_CVPR_2017_paper.pdf)]



---

## Person Search

### [ECCV2018]

#### *"RCAA: Relational Context-Aware Agents for Person Search"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaojun_Chang_RCAA_Relational_Context-Aware_ECCV_2018_paper.pdf)]

#### *"Person Search in Videos with One Portrait Through Visual and Temporal Links"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Qingqiu_Huang_Person_Search_in_ECCV_2018_paper.pdf)]

#### *"Person Search by Multi-Scale Matching"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xu_Lan_Person_Search_by_ECCV_2018_paper.pdf)]

#### *"Person Search via A Mask-Guided Two-Stream CNN Model"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Di_Chen_Person_Search_via_ECCV_2018_paper.pdf)]

### [ICCV2017]

#### *"Neural Person Search Machines"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Neural_Person_Search_ICCV_2017_paper.pdf)]

### [CVPR2017]

#### *"Person Search with Natural Language Description"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Person_Search_With_CVPR_2017_paper.pdf)]


---

## Others

### [ECCV2018]

#### *"Integrating Egocentric Videos in Top-view Surveillance Videos: Joint Identification and Temporal Alignment"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Shervin_Ardeshir_Integrating_Egocentric_Videos_ECCV_2018_paper.pdf)]

#### *"Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Karianakis_Reinforced_Temporal_Attention_ECCV_2018_paper.pdf)]

#### *"Adversarial Open-World Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiang_Li_Adversarial_Open-World_Person_ECCV_2018_paper.pdf)]

### [CVPR2018]

#### *"Viewpoint-aware Attentive Multi-view Inference for Vehicle Re-identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Viewpoint-Aware_Attentive_Multi-View_CVPR_2018_paper.pdf)]

#### *"Exploiting Transitivity for Learning Person Re-identification Models on a Budget"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Roy_Exploiting_Transitivity_for_CVPR_2018_paper.pdf)]


### [ICCV2017] 

#### *"Orientation Invariant Feature Embedding and Spatial Temporal Regularization for Vehicle Re-identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Orientation_Invariant_Feature_ICCV_2017_paper.pdf)]

#### *"RGB-Infrared Cross-Modality Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_RGB-Infrared_Cross-Modality_Person_ICCV_2017_paper.pdf)]

### [CVPR2017] 

#### *"Joint Detection and Identification Feature Learning for Person Search"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf)]

#### *"Multiple People Tracking by Lifted Multicut and Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tang_Multiple_People_Tracking_CVPR_2017_paper.pdf)]

#### *"Pose-Aware Person Recognition"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kumar_Pose-Aware_Person_Recognition_CVPR_2017_paper.pdf)]


### [Gait based] 

These are papers that search for "gait" and "person re-identification" in google scholar.

#### *"Person Re-identification by Video Ranking"*, ECCV2014 [[paper](https://link.springer.com/content/pdf/10.1007%2F978-3-319-10593-2_45.pdf)]

#### *"Person Re-identification using View-dependent Score-level Fusion of Gait and Color Features"*, ICPR 2012 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6460721)]

#### *"Enhancing Person Re-identification by Integrating Gait Biometric"*, Neurocomputing 2015 [[paper](https://ac.els-cdn.com/S0925231215006256/1-s2.0-S0925231215006256-main.pdf?_tid=6d48323a-1f67-4707-baf2-776f9c8d665f&acdnat=1539565898_188b1a8907ebd5c7021ff30d21bf3530)]

#### *"A Hierarchical Method Combining Gait and Phase of Motion with Spatiotemporal Model for Person Re-identification"*, Pattern Recognition Letters 2012 [[paper](https://ac.els-cdn.com/S0167865512000359/1-s2.0-S0167865512000359-main.pdf?_tid=003cfe6b-e4c8-44aa-80df-da83ab9aa943&acdnat=1539567285_9d4a214d72f8814b74f326c9dda43f57)]

#### *"Person Re-Identification by Discriminative Selection in Video Ranking"*, TPAMI 2016 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7393860)]

#### *"A Spatio-temporal Appearance Representation for Video-based Pedestrian Re-identification"*, ICCV 2015 [[paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Liu_A_Spatio-Temporal_Appearance_ICCV_2015_paper.pdf)]

#### *"Gait-Assisted Person Re-identification in Wide Area Surveillance"*, ACCV 2014 [[paper](https://link.springer.com/content/pdf/10.1007%2F978-3-319-16634-6_46.pdf)]

#### *"Person Re-identiﬁcation by Unsupervised Video Matching"*, Pattern Recognition 2017 [[paper](https://reader.elsevier.com/reader/sd/pii/S0031320316303764?token=F4EC31F9496FAE898967B4C2E36762A0E0F5AF343083EA28318A6DEF29415E18D4A740DA25A16A7B22B36C7704288BA6)]

#### *"Swiss-System Based Cascade Ranking for Gait-Based Person Re-identification"*, AAAI 2015 [[paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9417/9479)]

#### *"Person Re-identification using Height-based Gait in Colour Depth Camera"*, ICIP 2013 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6738689)]

#### *"Review of person re-identification techniques"*, IET Computer vision 2013 [[paper](http://digital-library.theiet.org/docserver/fulltext/iet-cvi/8/6/IET-CVI.2013.0180.pdf?expires=1539569257&id=id&accname=guest&checksum=768BDC38DF1D86E8455FA599A2083BDA)]

#### *"Person Re-identification in Appearance Impaired Scenarios"*, arXiv 2016 [[paper](https://arxiv.org/pdf/1604.00367.pdf)]

#### *"Learning Compact Appearance Representation for Video-based Person Re-Identification"*, arXiv 2017 [[paper](https://arxiv.org/pdf/1702.06294.pdf)]

#### *"Recurrent Attention Models for Depth-Based Person Identification"*, CVPR 2016 [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Haque_Recurrent_Attention_Models_CVPR_2016_paper.pdf)]

#### *"Towards view-point invariant Person Re-identification via fusion of Anthropometric and Gait Features from Kinect measurements"*, VISAPP 2017 [[paper](http://vislab.isr.ist.utl.pt/wp-content/uploads/2017/03/nambiar_VISAPP2017.pdf)]

#### *"Person Re-identification in Frontal Gait Sequences via Histogram of Optic Flow Energy Image"*, ICACIVS 2016 [[paper](https://link.springer.com/content/pdf/10.1007%2F978-3-319-48680-2_23.pdf)]

#### *"Learning Bidirectional Temporal Cues for Video-based Person Re-Identification"*, TCSVT 2016 [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7954700)]


---


## Datasets

### [CVPR2018] 

#### *"Person Transfer GAN to Bridge Domain Gap for Person Re-Identification"* [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wei_Person_Transfer_GAN_CVPR_2018_paper.pdf)]
- Rank-1 (Market1501) : 38.6(source domain : DukeMTMC)

### [CVPR2017] 

#### *"Person Re-Identification in the Wild"* [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zheng_Person_Re-Identification_in_CVPR_2017_paper.pdf)]

### [[Market-1501 Leaderboard](https://jingdongwang2017.github.io/Projects/ReID/Datasets/result_market1501.html)]

### [[Collection](http://robustsystems.coe.neu.edu/sites/robustsystems.coe.neu.edu/files/systems/projectpages/reiddataset.html)]

---

## Reference 

https://github.com/handong1587/handong1587.github.io/blob/master/_posts/deep_learning/2015-10-09-re-id.md


