---
title: "Grounding DINO 1.5: Advance the “Edge” of Open-Set Object Detection"
source: "https://ar5iv.labs.arxiv.org/html/2405.10300"
author:
published:
created: 2026-05-28
description: "This paper introduces Grounding DINO 1.5, a suite of advanced open-set object detection models developed by IDEA Research, which aims to advance the “Edge”111We use “edge” for its dual meaning both as in pushing the bo…"
tags:
  - "clippings"
---
Tianhe Ren,  Qing Jiang <sup>∗</sup>,  Shilong Liu <sup>∗</sup>,  Zhaoyang Zeng <sup>∗</sup>,   
Wenlong Liu,  Han Gao,  Hongjie Huang,  Zhengyu Ma,  Xiaoke Jiang,   
Yihao Chen,  Yuda Xiong,  Hao Zhang,  Feng Li,  Peijun Tang,  Kent Yu,  Lei Zhang  
  
International Digital Economy Academy (IDEA), IDEA Research  
[https://deepdataspace.com/home](https://deepdataspace.com/home) Equal contributions. List order is random.Project lead and corresponding author.

###### Abstract

This paper introduces Grounding DINO 1.5, a suite of advanced open-set object detection models developed by IDEA Research, which aims to advance the “Edge” <sup>1</sup> of open-set object detection. The suite encompasses two models: Grounding DINO 1.5 Pro, a high-performance model designed for *stronger* generalization capability across a wide range of scenarios, and Grounding DINO 1.5 Edge, an efficient model optimized for *faster* speed demanded in many applications requiring edge deployment. The Grounding DINO 1.5 Pro model advances its predecessor by scaling up the model architecture, integrating an enhanced vision backbone, and expanding the training dataset to over 20 million images with grounding annotations, thereby achieving a richer semantic understanding. The Grounding DINO 1.5 Edge model, while designed for efficiency with reduced feature scales, maintains robust detection capabilities by being trained on the same comprehensive dataset. Empirical results demonstrate the effectiveness of Grounding DINO 1.5, with the Grounding DINO 1.5 Pro model attaining a 54.3 AP on the COCO detection benchmark and a 55.7 AP on the LVIS-minival zero-shot transfer benchmark, setting new records for open-set object detection. Furthermore, the Grounding DINO 1.5 Edge model, when optimized with TensorRT, achieves a speed of 75.2 FPS while attaining a zero-shot performance of 36.2 AP on the LVIS-minival benchmark, making it more suitable for edge computing scenarios. Model examples and demos with API will be released at [https://github.com/IDEA-Research/Grounding-DINO-1.5-API](https://github.com/IDEA-Research/Grounding-DINO-1.5-API).

## 1 Introduction

In this paper, we introduce Grounding DINO 1.5, a series of powerful and practical open-set object detection models developed by IDEA Research. Object detection is a fundamental task in computer vision, with recent efforts focusing on developing generic detectors capable of performing detection across a wide variety of real-world applications. A key strategy for improving model generalization across diverse object categories is the integration of language modality, which has received increasingly more attention and has undergone extensive development in recent research.

Grounding DINO [^18] represents a significant advancement in this area. Building on the Transformer-based DINO [^33] architecture, it incorporates linguistic information to enable open-set object detection in various scenarios. Following GLIP [^16], Grounding DINO redefines object detection as a phrase grounding task, facilitating large-scale pre-training using both detection and grounding datasets. This approach, coupled with self-training on pseudo-labeled grounding data derived from an almost unlimited pool of image-text pairs, enhances the model’s applicability to open-world settings due to its robust architecture and semantic-rich dataset.

Building upon the success of Grounding DINO, Grounding DINO 1.5 extends the model’s capabilities in two key areas: *stronger* detection performance and *faster* inference speed, designated as the Grounding DINO 1.5 Pro and Grounding DINO 1.5 Edge models, respectively.

The Grounding DINO 1.5 Pro model significantly expands both the model’s capacity and the dataset size, with the goal of creating a more potent and versatile open-set object detection model. Specifically, we have upscaled the model by incorporating the pre-trained ViT-L [^6] architecture and developed a data engine capable of producing over 20 million images with grounding annotations from diverse sources, thereby enriching the model’s semantic comprehension.

By contrast, the Grounding DINO 1.5 Edge model is tailored for edge devices, focusing on computational efficiency without compromising detection quality. We develop an efficient feature enhancer that leverages only high-level image features, removing the need of multi-scale features. This streamlined approach maintains the model’s strong context-aware detection capabilities, after training on the same 20 million images as used for the Pro model.

Extensive results from our experiments validate the superiority of Grounding DINO 1.5. Specifically, Grounding DINO 1.5 Pro achieves a 54.3 AP on the COCO detection zero-shot transfer benchmark and simultaneously achieves a 55.7 AP and a 47.6 AP on the LVIS-minival and LVIS-val zero-shot transfer benchmarks, respectively, setting new records on these benchmarks. Moreover, under TensorRT optimization, the Grounding DINO 1.5 Edge model reaches a speed of 75.2 FPS and achieves a zero-shot performance of 36.2 AP on the LVIS-minival benchmark, making it more suitable for edge computing scenarios.

## 2 Model Training

### 2.1 Model Architecture

We present the overall framework of Grounding DINO 1.5 series in Figure 1. This framework retains the dual-encoder-single-decoder structure of Grounding DINO and extends it into Grounding DINO 1.5 for both the Pro and Edge models, which are introduced in Sections 2.1.1 and 2.1.2, respectively.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x1.png)

Figure 1: Grounding DINO 1.5 series overall framework.

#### 2.1.1 Grounding DINO 1.5 Pro

Grounding DINO 1.5 Pro preserves the core architecture of Grounding DINO while incorporating a larger Vision Transformer backbone. We adopt the pre-trained ViT-L [^6] model as our primary vision backbone for its superior performance on downstream tasks and its pure Transformer design, which lays a solid foundation for optimizing the training and inference processes.

Following the methodologies of Grounding DINO [^18] and GLIP [^16], Grounding DINO 1.5 Pro employs a deep early fusion strategy during feature extraction. This involves cross-attention mechanisms between language and image features before the decoding phase, facilitating a more integrated information fusion.

We also compared the strategies of early fusion and later fusion. We observe that models trained with early fusion architecture design tend to yield a higher detection recall and better bounding box precision accuracy. However, this approach can also lead to increased model hallucinations, such as predicting objects not present in the input images. In contrast to early fusion, the models with a late fusion design, which integrate language and image modalities only in the loss calculation phase, generally demonstrate robustness against hallucinations but may lead to lower detection recall. This is primarily due to the increased challenge of vision and language alignment as late fusion keeps features from different modalities separately until the loss phase.

Consequently, to simultaneously enhance the model’s prediction capability and maintain its robustness during inference, we have retained the early fusion design while introducing a more comprehensive training sampling strategy, which increases the proportion of negative samples during training. Such improvements facilitate achieving a balance between the advantages and drawbacks of the early fusion architecture.

#### 2.1.2 Grounding DINO 1.5 Edge

Deploying Grounding DINO on edge devices is highly desired by many applications, including autonomous driving, medical image processing, computational photography, etc. However, there is a large gap between the computational cost required by an open-set detection model and the limited resources available on edge devices. Grounding DINO uses multi-scale image features and a heavy computational feature enhancer for faster training and better performance, which is impractical for real-time scenarios in real-world applications.

To overcome this obstacle, we propose a novel efficient feature enhancer, as shown in Fig.2. Recognizing that lower-level image features lack semantic information and introduce excessive computational costs, as demonstrated in Lite-DETR [^15], we limit cross-modality fusion to high-level image features (P5 level) only. This approach greatly reduces the number of tokens that need to be processed, significantly cutting the computational demands of the feature enhancer. To facilitate easier deployment on edge-side GPUs, we replace deformable self-attention with vanilla self-attention, and introduce a cross-scale feature fusion module [^37] to integrate low-level image features (from P3 and P4 levels). Such a design effectively balances feature enhancement and computational efficiency.

In our edge-optimized model, Grounding DINO 1.5 Edge, we replace the original feature enhancer with our newly proposed efficient one and employ EfficientViT-L1 [^1] as the image backbone for rapid multi-scale feature extraction. We deploy the model on the NVIDIA Orin NX platform, achieving an inference speed of over 10 FPS at an input size of 640 $\times$ 640. The visualization of the model predictions on NVIDIA Orin NX is shown in Figure 16, which demonstrates the effectiveness of our modifications in real-world edge computing environments.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x2.png)

Figure 2: Comparison of the Origin Feature Enhancer and the New Efficient Feature Enhancer.

### 2.2 Training Dataset

To train a robust open-set detector, it is crucial to construct a high-quality grounding dataset that is sufficiently rich in categories and encompasses a wide range of detection scenarios. Grounding DINO 1.5 is pre-trained on over 20M grounding images, termed Grounding-20M, which are collected from publicly available sources. We have carefully developed a series of annotation pipelines and post-processing rules to guarantee the high quality of the grounding annotations.

## 3 Model Evaluation

We compare Grounding DINO 1.5 with other related works on both the zero-shot and fine-tuning settings. The best and the second best results are indicated in bold and with underline.

### 3.1 Zero-Shot Transfer of Grounding DINO 1.5 Pro

Following the implementation of Grounding DINO [^18], we evaluate our model’s zero-shot transfer performance on the COCO [^17] dataset, which comprises 80 common categories and the LVIS [^7] datasets, which includes 1203 more diverse and long-tailed categories. We report the performance of fixed AP [^4] on both the LVIS-val and LVIS-minival splits. As shown in Table 1, our model shows a significant performance improvement compared to the previous Grounding DINO models. For instance, on the COCO zero-shot transfer benchmark, Grounding DINO 1.5 Pro achieves a 54.3 AP, improving upon Grounding DINO Swin-L by 1.8 AP. On the LVIS-minival and LVIS-val zero-shot transfer benchmarks, Grounding DINO 1.5 Pro achieves a 55.7 AP and a 47.6 AP, outperforming the previous best model, DetCLIPv3, by 6.9 AP and 6.2 AP, respectively. Furthermore, compared with the Grounding DINO Swin-T model, our model demonstrates a remarkable improvement of 28.3 AP (55.7 AP vs. 27.4 AP) on the LVIS-minival zero-shot transfer benchmark.

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Pre-training data</td><td>COCO</td><td colspan="4">LVIS <sup>minival</sup></td><td colspan="4">LVIS <sup>val</sup></td><td>ODinW35</td><td>ODinW13</td></tr><tr><td></td><td></td><td></td><td>AP <sub>all</sub></td><td>AP <sub>all</sub></td><td>AP <sub>r</sub></td><td>AP <sub>c</sub></td><td>AP <sub>f</sub></td><td>AP <sub>all</sub></td><td>AP <sub>r</sub></td><td>AP <sub>c</sub></td><td>AP <sub>f</sub></td><td>AP <sub>avg</sub></td><td>AP <sub>avg</sub></td></tr><tr><td colspan="14">Supervised Models (Pre-training data includes COCO, LVIS, etc.)</td></tr><tr><td>GLIPv2 <sup><a href="#fn:35">35</a></sup></td><td>Swin-H <sup><a href="#fn:32">32</a></sup></td><td>FourODs,COCO,GoldG,CC15M,SBU</td><td>60.6</td><td>50.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>55.5</td></tr><tr><td>Grounding DINO <sup><a href="#fn:18">18</a></sup></td><td>Swin-L <sup><a href="#fn:19">19</a></sup></td><td>O365,OID,GoldG,Cap4M,COCO,RefC</td><td>60.7</td><td>33.9</td><td>22.2</td><td>30.7</td><td>38.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>APE (B) <sup><a href="#fn:24">24</a></sup></td><td>ViT-L</td><td>COCO,LVIS,O365,OID,VG</td><td>57.7</td><td>62.5</td><td>-</td><td>-</td><td>-</td><td>57.0</td><td>-</td><td>-</td><td>-</td><td>29.4</td><td>59.8</td></tr><tr><td>APE (D) <sup><a href="#fn:24">24</a></sup></td><td>ViT-L <sup><a href="#fn:6">6</a></sup></td><td>COCO,LVIS,O365,OID,VG,RefC,SA-1B,GoldG,PhraseCut</td><td>58.3</td><td>64.7</td><td>-</td><td>-</td><td>-</td><td>59.6</td><td>-</td><td>-</td><td>-</td><td>28.8</td><td>57.9</td></tr><tr><td>GLEE-Pro <sup><a href="#fn:27">27</a></sup></td><td>ViT-L <sup><a href="#fn:6">6</a></sup></td><td>GLEE-merged-10M (COCO,LVIS,etc)</td><td>62.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>55.7</td><td>49.2</td><td>-</td><td>-</td><td>-</td><td>53.4</td></tr><tr><td colspan="14">Zero-shot Transfer Models</td></tr><tr><td>OWL-ViT <sup><a href="#fn:22">22</a></sup></td><td>ViT-L <sup><a href="#fn:5">5</a></sup></td><td>O365,OID,VG,LiT</td><td>42.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>34.6</td><td>31.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MDETR <sup><a href="#fn:11">11</a></sup></td><td>ResNet101 <sup><a href="#fn:8">8</a></sup></td><td>COCO,GoldG</td><td>-</td><td>22.5</td><td>7.4</td><td>22.7</td><td>25.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>GLIP <sup><a href="#fn:16">16</a></sup></td><td>Swin-L</td><td>FourODs,GoldG,Cap24M</td><td>49.8</td><td>37.3</td><td>28.2</td><td>34.3</td><td>41.5</td><td>26.9</td><td>17.1</td><td>23.3</td><td>35.4</td><td>-</td><td>52.1</td></tr><tr><td>Grounding DINO <sup><a href="#fn:18">18</a></sup></td><td>Swin-T</td><td>O365,GoldG,Cap4M</td><td>48.4</td><td>27.4</td><td>18.1</td><td>23.3</td><td>32.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>22.3</td><td>49.8</td></tr><tr><td>Grounding DINO <sup><a href="#fn:18">18</a></sup></td><td>Swin-L</td><td>O365,OID,GoldG</td><td>52.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>26.1</td><td>56.9</td></tr><tr><td>OpenSeeD <sup><a href="#fn:34">34</a></sup></td><td>Swin-L</td><td>COCO,O365</td><td>-</td><td>23.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>15.2</td><td>-</td></tr><tr><td>UniDetector <sup><a href="#fn:26">26</a></sup></td><td>ResNet50 <sup><a href="#fn:8">8</a></sup></td><td>COCO,O365,OID</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>19.8</td><td>18.0</td><td>19.2</td><td>21.2</td><td>-</td><td>47.3</td></tr><tr><td>OmDet-Turbo-B <sup><a href="#fn:36">36</a></sup></td><td>ConvNeXt-B <sup><a href="#fn:20">20</a></sup></td><td>O365,GoldG,PhraseCut,Hake,HOI-A</td><td>53.4</td><td>34.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>30.1</td><td>54.7</td></tr><tr><td>OWL-ST <sup><a href="#fn:21">21</a></sup></td><td>CLIP L/14 <sup><a href="#fn:23">23</a></sup></td><td>WebLI2B</td><td>-</td><td>40.9</td><td>41.5</td><td>-</td><td>-</td><td>35.2</td><td>36.2</td><td>-</td><td>-</td><td>24.4</td><td>53.0</td></tr><tr><td>MQ-GLIP <sup><a href="#fn:28">28</a></sup></td><td>Swin-L</td><td>O365</td><td>-</td><td>43.4</td><td>34.5</td><td>41.2</td><td>46.9</td><td>34.7</td><td>26.9</td><td>32.0</td><td>41.3</td><td>23.9</td><td>54.1</td></tr><tr><td>DetCLIP <sup><a href="#fn:30">30</a></sup></td><td>Swin-L</td><td>O365,GoldG,YFCC1M</td><td>-</td><td>38.6</td><td>36.0</td><td>38.3</td><td>39.3</td><td>28.4</td><td>25.0</td><td>27.0</td><td>31.6</td><td>-</td><td>-</td></tr><tr><td>DetCLIPv2 <sup><a href="#fn:29">29</a></sup></td><td>Swin-L</td><td>O365,GoldG,CC15M</td><td>-</td><td>44.7</td><td>43.1</td><td>46.3</td><td>43.7</td><td>36.6</td><td>33.3</td><td>36.2</td><td>38.5</td><td>-</td><td>-</td></tr><tr><td>DetCLIPv3 <sup><a href="#fn:31">31</a></sup></td><td>Swin-L</td><td>O365,V3Det,GoldG,GranuCap50M</td><td>-</td><td>48.8</td><td>49.9</td><td>49.7</td><td>47.8</td><td>41.4</td><td>41.4</td><td>40.5</td><td>42.3</td><td>-</td><td>-</td></tr><tr><td>YOLO-World <sup><a href="#fn:3">3</a></sup></td><td>YOLOv8-L <sup><a href="#fn:10">10</a></sup></td><td>O365,GoldG,CC3M</td><td>45.1</td><td>35.4</td><td>27.6</td><td>34.1</td><td>38.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DINOv <sup><a href="#fn:14">14</a></sup></td><td>Swin-L</td><td>COCO,SA-1B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>15.7</td><td>-</td></tr><tr><td>T-Rex2 (visual) <sup><a href="#fn:9">9</a></sup></td><td>Swin-L</td><td>O365,OID,HierText,CrowdHuman,SA-1B</td><td>46.5</td><td>47.6</td><td>45.4</td><td>46.0</td><td>49.5</td><td>45.3</td><td>43.8</td><td>42.0</td><td>49.5</td><td>27.8</td><td>-</td></tr><tr><td>T-Rex2 (text) <sup><a href="#fn:9">9</a></sup></td><td>Swin-L</td><td>O365,OID,GoldG,CC3M,SBU,LAION</td><td>52.2</td><td>54.9</td><td>49.2</td><td>54.8</td><td>56.1</td><td>45.8</td><td>42.7</td><td>43.2</td><td>50.2</td><td>22.0</td><td>-</td></tr><tr><td>Grounding DINO 1.5 Pro (zero-shot)</td><td>ViT-L <sup><a href="#fn:6">6</a></sup></td><td>Grounding-20M</td><td>54.3</td><td>55.7</td><td>56.1</td><td>57.5</td><td>54.1</td><td>47.6</td><td>44.6</td><td>47.9</td><td>48.7</td><td>30.2</td><td>58.7</td></tr></tbody></table>

Table 1: Performance of Grounding DINO 1.5 Pro on the COCO, LVIS, ODinW35 [^16] and ODinW13 [^16] benchmarks compared to previous methods. Gray numbers indicate that the training dataset includes images or annotations from COCO or LVIS datasets.

| Method | Backbone | PascalVOC | AerialDrone | Aquarium | Rabbits | EgoHands | Mushrooms | Packages | Raccoon | Shellfish | Vehicles | Pistols | Pothole | Thermal | AP <sub>avg</sub> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLIP | Swin-L | 61.7 | 7.1 | 26.9 | 75.0 | 45.5 | 49.0 | 62.8 | 63.3 | 68.9 | 57.3 | 68.6 | 25.7 | 66.0 | 52.1 |
| GLIPv2 | Swin-H | 66.3 | 10.9 | 30.4 | 74.6 | 55.1 | 52.1 | 71.3 | 63.8 | 66.2 | 57.2 | 66.4 | 33.8 | 73.3 | 55.5 |
| Grounding DINO | Swin-L | 66.0 | 12.6 | 28.1 | 72.8 | 52.1 | 73.0 | 63.9 | 67.9 | 64.8 | 62.7 | 71.7 | 31.4 | 78.4 | 56.9 |
| OmDet-Turbo-B | ConvNeXt-B | 63.7 | 16.2 | 28.5 | 70.6 | 55.7 | 71.5 | 65.6 | 63.6 | 39.7 | 61.9 | 65.5 | 30.2 | 78.4 | 54.7 |
| OWL-ST | CLIP L/14 | 53.9 | 19.9 | 32.3 | 84.9 | 47.1 | 76.6 | 70.9 | 63.8 | 35.0 | 58.5 | 62.6 | 27.5 | 55.6 | 53.0 |
| MQ-GLIP-L | Swin-L | 64.7 | 17.4 | 30.3 | 71.8 | 57.2 | 63.9 | 53.0 | 58.1 | 63.0 | 63.2 | 74.4 | 27.0 | 58.7 | 54.1 |
| APE (B) | ViT-L | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | 59.8 |
| APE (D) | ViT-L | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | 57.9 |
| GLEE-Pro-Scale | ViT-L | 69.1 | 13.7 | 34.7 | 75.6 | 38.9 | 57.8 | 50.6 | 65.6 | 62.7 | 67.3 | 69.0 | 30.7 | 59.1 | 53.4 |
| T-Rex2 (visual prompt) | Swin-L | 65.8 | 16.0 | 27.0 | 70.0 | 61.9 | 83.7 | 58.9 | 67.1 | 53.0 | 66.4 | 69.0 | 24.1 | 61.4 | 55.7 |
| Grounding DINO 1.5 Pro | ViT-L | 67.1 | 19.0 | 38.5 | 65.7 | 61.8 | 82.1 | 58.1 | 72.5 | 62.0 | 64.3 | 71.9 | 29.0 | 71.4 | 58.7 |

Table 2: Detailed zero-shot performance comparison between Grounding DINO 1.5 Pro and related works on ODinW13 benchmark.

We further evaluate the generalization capability of our model in multiple real-world scenarios using the ODinW (Object Detection in the Wild) [^16] benchmark, which encompasses 35 datasets covering a wide range of application domains. We observe that within the ODinW benchmark, several datasets exhibit significant quality issues in terms of the annotated category names. To mitigate such problems, during testing, we performed prompt engineering to refine category names on datasets where their performance was particularly poor to better align their category names with actual testing scenarios. Grounding DINO 1.5 Pro achieves an average of 58.7 AP over 13 datasets on the ODinW13 benchmark and set a new record on the ODinW35 benchmark with an average of 30.2 AP over 35 datasets, improving upon Grounding DINO by 4.2 AP. The comprehensive per-dataset performance of Grounding DINO 1.5 Pro on ODinW13 are presented in Table 2. Furthermore, the detailed per-dataset performance of Grounding DINO 1.5 Pro on ODinW35 is available in Appendix Section 7.1.

### 3.2 Fine-tuning Results on Downstream Datasets

We further investigate the transferability of Grounding DINO 1.5 Pro by fine-tuning it on various downstream datasets. As shown in Table 3, on the LVIS dataset, the fine-tuned Grounding DINO 1.5 Pro model achieves a 68.1 AP and a 63.5 AP on the LVIS-minival and LVIS-val splits, respectively, which represent an enhancement of 12.4 AP and 15.9 AP over the Grounding DINO 1.5 Pro zero-shot setting.

<table><tbody><tr><td>Method</td><td>Backbone</td><td colspan="4">LVIS <sup>minival</sup></td><td colspan="4">LVIS <sup>val</sup></td><td>ODinW35</td><td>ODinW13</td></tr><tr><td></td><td></td><td>AP <sub>all</sub></td><td>AP <sub>r</sub></td><td>AP <sub>c</sub></td><td>AP <sub>f</sub></td><td>AP <sub>all</sub></td><td>AP <sub>r</sub></td><td>AP <sub>c</sub></td><td>AP <sub>f</sub></td><td>AP <sub>avg</sub></td><td>AP <sub>avg</sub></td></tr><tr><td>GLIP</td><td>Swin-L</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>68.9</td></tr><tr><td>GLEE-Pro</td><td>ViT-L</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>69.0</td></tr><tr><td>GLIPv2</td><td>Swin-H</td><td>59.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>70.4</td></tr><tr><td>OWL-ST+FT †</td><td>CLIP L/14</td><td>54.4</td><td>46.1</td><td>-</td><td>-</td><td>49.4</td><td>44.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DetCLIPv2</td><td>Swin-L</td><td>60.1</td><td>58.3</td><td>61.7</td><td>59.1</td><td>53.1</td><td>49.0</td><td>53.2</td><td>54.9</td><td>-</td><td>70.4</td></tr><tr><td>DetCLIPv3</td><td>Swin-L</td><td>60.5</td><td>60.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>72.1</td></tr><tr><td>DetCLIPv3 †</td><td>Swin-L</td><td>60.8</td><td>56.7</td><td>63.2</td><td>59.4</td><td>54.1</td><td>45.8</td><td>55.4</td><td>56.4</td><td>-</td><td>-</td></tr><tr><td>Grounding DINO 1.5 Pro (zero-shot)</td><td>ViT-L</td><td>55.7</td><td>56.1</td><td>57.5</td><td>54.1</td><td>47.6</td><td>44.6</td><td>47.9</td><td>48.7</td><td>30.2</td><td>58.7</td></tr><tr><td>Grounding DINO 1.5 Pro</td><td>ViT-L</td><td>68.1 (+12.4)</td><td>68.7</td><td>69.5</td><td>66.6</td><td>63.5 (+15.9)</td><td>64.0</td><td>63.8</td><td>63.0</td><td>70.6 (+40.4)</td><td>72.4 (+13.7)</td></tr></tbody></table>

Table 3: Fine-tuning performance of Grounding DINO 1.5 Pro on the LVIS-minival, LVIS-val, ODinW35 and ODinW13 benchmarks. The fixed AP [^4] on LVIS-minival and val splits are reported. †indicates results of fine-tuning with LVIS base categories only.

After fine-tuning on the ODinW35 dataset, the Grounding DINO 1.5 Pro model sets new records by achieving an average of 70.6 AP across 35 datasets on the ODinW35 benchmark and 72.4 AP over 13 datasets on the ODinW13 benchmark, respectively. This represents a significant improvement over the zero-shot setting, with respective improvements of 40.5 AP and 13.7 AP. The detailed fine-tuning performance of Grounding DINO 1.5 Pro on the ODinW13 benchmark is reported in Table 4.

| Method | Backbone | PascalVOC | AerialDrone | Aquarium | Rabbits | EgoHands | Mushrooms | Packages | Raccoon | Shellfish | Vehicles | Pistols | Pothole | Thermal | AP <sub>all</sub> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLIP | Swin-L | 69.6 | 32.6 | 56.6 | 76.4 | 79.4 | 88.1 | 67.1 | 69.4 | 65.8 | 71.6 | 75.7 | 60.3 | 83.1 | 68.9 |
| GLEE-Pro | ViT-L | 72.6 | 36.5 | 58.1 | 80.5 | 74.1 | 92.0 | 67.0 | 76.5 | 66.4 | 70.5 | 66.4 | 55.7 | 80.6 | 69.0 |
| GLIPv2 | Swin-H | 74.4 | 36.3 | 58.7 | 77.1 | 79.3 | 88.1 | 74.3 | 73.1 | 70.0 | 72.2 | 72.5 | 58.3 | 81.4 | 70.4 |
| DetCLIPv2 | Swin-L | 74.4 | 44.1 | 54.7 | 80.9 | 79.9 | 90.0 | 74.1 | 69.4 | 61.2 | 68.1 | 80.3 | 57.1 | 81.1 | 70.4 |
| Grounding DINO | Swin-T | 73.6 | 36.6 | 57.7 | 78.7 | 79.2 | 92.8 | 74.7 | 74.7 | 61.2 | 69.6 | 75.9 | 60.4 | 85.9 | 70.9 |
| MQ-GLIP-L | Swin-L | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- | 71.3 |
| DetCLIPv3 | Swin-L | 76.4 | 51.2 | 57.5 | 79.9 | 80.2 | 90.4 | 75.1 | 70.9 | 63.6 | 69.8 | 82.7 | 56.2 | 83.8 | 72.1 |
| Grounding DINO 1.5 Pro | ViT-L | 77.6 | 37.0 | 60.2 | 75.1 | 78.6 | 89.2 | 72.1 | 81.8 | 70.8 | 74.6 | 77.6 | 62.3 | 84.0 | 72.4 |

Table 4: Detailed fine-tuning performance of Grounding DINO 1.5 Pro on the ODinW13 benchmark.

### 3.3 Main Results of Grounding DINO 1.5 Edge

After pre-training on Grounding-20M, we directly evaluate Grounding DINO 1.5 Edge on the COCO dataset and LVIS dataset in a zero-shot manner. Following previous works [^18] [^3] [^35], we evaluate on both LVIS-val and LVIS-minival splits and report the fixed AP [^4] for comparison. The main results are shown in Table 5. Compared with current real-time open-set detectors in an end-to-end test setting, which do not use language cache, Grounding DINO 1.5 Edge achieves a zero-shot AP of 45.0 on COCO. Regarding the zero-shot performance on LVIS-minival, Grounding DINO 1.5 Edge achieves remarkable performance, an AP score of 36.2, which surpasses all other state-of-the-art algorithms (OmDet-Turbo-T 30.3 AP, YOLO-Worldv2-L 32.9 AP, YOLO-Worldv2-M 30.0 AP, YOLO-Worldv2-S 22.7 AP). Notably, deploying Grounding DINO 1.5 Edge model optimized with TensorRT on NVIDIA Orin NX achieves an inference speed of over 10 FPS at an input size of 640 $\times$ 640.

<table><tbody><tr><td>Method</td><td>Backbone</td><td>Pre-training data</td><td>test size</td><td>COCO</td><td colspan="4">LVIS <sup>minival</sup></td><td colspan="4">LVIS <sup>val</sup></td><td>FPS(A100/TensorRT)</td><td>FPS(Orin NX)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>AP <sub>all</sub></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>r</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>r</ci></apply></apply> <annotation>AP_{\mathrm{r}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>c</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>c</ci></apply></apply> <annotation>AP_{\mathrm{c}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>f</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>f</ci></apply></apply> <annotation>AP_{\mathrm{f}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>all</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>all</ci></apply></apply> <annotation>AP_{\mathrm{all}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>r</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>r</ci></apply></apply> <annotation>AP_{\mathrm{r}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>c</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>c</ci></apply></apply> <annotation>AP_{\mathrm{c}}</annotation></semantics></math></td><td><math><semantics><mrow><mi>A</mi> <mo></mo><msub><mi>P</mi> <mi>f</mi></msub></mrow> <apply><ci>𝐴</ci> <apply><csymbol>subscript</csymbol> <ci>𝑃</ci> <ci>f</ci></apply></apply> <annotation>AP_{\mathrm{f}}</annotation></semantics></math></td><td></td><td></td></tr><tr><td colspan="14">End-to-End Open-Set Object Detection</td><td></td></tr><tr><td>GLIP-T</td><td>Swin-T</td><td>O365,GoldG,Cap4M</td><td>800 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1333</td><td>46.3</td><td>26.0</td><td>20.8</td><td>21.4</td><td>31.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Grounding DINO-T</td><td>Swin-T</td><td>O365,GoldG,Cap4M</td><td>800 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1333</td><td>48.4</td><td>27.4</td><td>18.1</td><td>23.3</td><td>32.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>9.4 / 42.6</td><td>1.1</td></tr><tr><td colspan="14">Real-time End-to-End Open-Set Object Detection</td><td></td></tr><tr><td>YOLO-Worldv2-S†</td><td>YOLOv8-S</td><td>O365,GoldG</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>-</td><td>22.7</td><td>16.3</td><td>20.8</td><td>25.5</td><td>17.3</td><td>11.3</td><td>14.9</td><td>22.7</td><td>47.4 / -</td><td>-</td></tr><tr><td>YOLO-Worldv2-M†</td><td>YOLOv8-M</td><td>O365,GoldG</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>-</td><td>30.0</td><td>25.0</td><td>27.2</td><td>33.4</td><td>23.5</td><td>17.1</td><td>20.0</td><td>30.1</td><td>42.7 / -</td><td>-</td></tr><tr><td>YOLO-Worldv2-L†</td><td>YOLOv8-L</td><td>O365,GoldG</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>-</td><td>33.0</td><td>22.6</td><td>32.0</td><td>35.8</td><td>26.0</td><td>18.6</td><td>23.0</td><td>32.6</td><td>37.4 / -</td><td>-</td></tr><tr><td>YOLO-Worldv2-L†</td><td>YOLOv8-L</td><td>O365,GoldG,CC3M-Lite</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>-</td><td>32.9</td><td>25.3</td><td>31.1</td><td>35.8</td><td>26.1</td><td>20.6</td><td>22.6</td><td>32.3</td><td>37.4 / -</td><td>-</td></tr><tr><td>OmDet-Turbo-T <sup>‡</sup></td><td>Swin-T</td><td>O365,GoldG</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>42.5</td><td>30.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>21.5 / 140.0</td><td>-</td></tr><tr><td>Grounding DINO 1.5 Edge</td><td>EfficientViT-L1</td><td>Grounding-20M</td><td>640 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 640</td><td>42.9</td><td>33.5</td><td>28.0</td><td>34.3</td><td>33.9</td><td>27.3</td><td>26.3</td><td>25.7</td><td>29.6</td><td>21.7 / 111.6</td><td>10.7</td></tr><tr><td>Grounding DINO 1.5 Edge</td><td>EfficientViT-L1</td><td>Grounding-20M</td><td>800 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1333</td><td>45.0</td><td>36.2</td><td>33.2</td><td>36.6</td><td>36.3</td><td>29.3</td><td>28.1</td><td>27.6</td><td>31.6</td><td>18.5 / 75.2</td><td>5.5</td></tr></tbody></table>

Table 5: Zero-shot Results of Grounding DINO 1.5 Edge on COCO and LVIS. Speed measurement is performed on an A100 GPU, expressed in frames per second (FPS). The format used is PyTorch speed / TensorRT FP32 speed. And FPS on NVIDIA Orin NX is also reported. †indicates results of YOLO-World are reproduced by the latest official codes. <sup>‡</sup> indicates it uses language cache, which does not calculate the latency of the text encoder.

## 4 Case Analysis and Qualitative Visualization

In this section, we visualize the detection results of Grounding DINO 1.5 models in real-world scenarios. The images and text prompts are primarily sourced from the COCO [^17], LVIS [^7], V3Det [^25], OpenImages [^13], CC3M [^2] and SA-1B [^12]. We are deeply grateful for their contributions, which have significantly benefited the community.

### 4.1 Common Object Detection

The visualizations presented in Figures 3 and 4 demonstrate the robust performance of Grounding DINO 1.5 Pro for common object detection scenarios. Our model’s proficiency is evident not only in its handling of typical cases but also in its ability to accurately detect objects under challenging conditions.

The model adeptly identifies objects in monochromatic images, where color cues are minimal, as illustrated by the first example in Figure 3. This showcases the model’s reliance on shape and texture to distinguish objects. The detection of blurry objects, as seen in the second example of Figure 3, is a testament to the model’s robustness against common image degradations, maintaining high accuracy even when visual clarity is compromised.

The ability to detect small and partially occluded objects is crucial for many applications. Grounding DINO 1.5 Pro’s success in these scenarios, as shown in the last image in Figure 3, indicates its fine-grained understanding and the nuanced integration of multi-modal information in autonomous driving scenes. The visualizations also highlight the model’s versatility in handling objects of varying sizes and shapes, from petite to sprawling, each accurately localized and identified.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x3.png)

Figure 3: Model predictions on common objects with Grounding DINO 1.5 Pro (part 1).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x4.png)

Figure 4: Model predictions on common objects with Grounding DINO 1.5 Pro (part 2).

### 4.2 Long-tailed Object Detection

This subsection delves into the capability of Grounding DINO 1.5 Pro in detecting long-tailed objects, which are less frequently encountered categories that pose unique challenges for object detection models. The examples provided in Figure 5 highlight the model’s nuanced capability of understanding and detecting such uncommon objects. The model demonstrates its ability to recognize a diverse set of objects, including those that are not commonly found in everyday settings.

For instance, the second image within the figure illustrates the model’s capacity to identify a fungus, which is a category that requires specialized knowledge to discern. The third image showcases the model’s fine-grained detection capabilities, accurately pinpointing a popsicle amidst a variety of potential distractors. The model’s contextual understanding is evident in its ability to detect a taco in the last image of the third line, an object that may be challenging to recognize without the appropriate contextual cues.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x5.png)

Figure 5: Model predictions on long-tailed categories with Grounding DINO 1.5 Pro.

### 4.3 Short Caption Grounding

Grounding models can correlate objects within images to their corresponding mentions in accompanying captions. This capability is particularly significant for enhancing the contextual understanding of visual content across various domains. In Figure 6, we present Grounding DINO 1.5 Pro’s proficiency in short caption grounding, highlighting its versatility and accuracy.

The model exhibits a robust performance in grounding objects across different visual domains. It adeptly handles real-world images while also demonstrating a keen ability to interpret objects within cartoons, sketches, and animations. By aligning textual descriptions with visual features, the model can accurately identify and localize objects, even when they appear in stylized or abstract forms.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x6.png)

Figure 6: Phrase grounding on short captions with Grounding DINO 1.5 Pro.

### 4.4 Long Caption Grounding

Our model, Grounding DINO 1.5 Pro, extends its capability beyond the realm of standard image-caption pairs to adeptly handle long captions, as depicted in Figure 7, Figure 8 and Figure 9. Long captions offer a richer tapestry of details that can more comprehensively describe the visual content of an image. The ability to map each noun phrase in a long caption to corresponding objects within an image is a significant step toward deeper image understanding.

An intriguing observation is the model’s ability to generalize from pre-training on captions with shorter context windows to effectively processing longer contexts. This adaptability suggests that larger models may inherently possess flexibility that can be leveraged for various lengths of textual input.

Grounding DINO 1.5 Pro exhibits an impressive capacity to correlate textual phrases with visual elements. This capability is not only showcased in the model’s handling of long captions but also in its nuanced analysis of images at multiple granularities. Moreover, we notice that the model can detect some terms that did not show in the training data, like fiat logo in the third image. It presents the strong generalization ability of the model.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x7.png)

Figure 7: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 1).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x8.png)

Figure 8: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 2).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x9.png)

Figure 9: Phrase grounding on long captions with Grounding DINO 1.5 Pro (part 3).

### 4.5 Dense Object Detection

Grounding DINO 1.5 Pro showcases an exceptional capability to discern objects within dense scenarios, where multiple objects are closely positioned or overlapping, making detection a challenging task. This ability is vividly demonstrated through the visualizations presented in Figure 10 and Figure 11.

The model’s performance is noteworthy across a wide spectrum of object nomenclature. It adeptly identifies objects labeled with common names such as coin, tree, flower, and land. Moreover, the model also excels at recognizing objects denoted by specialized terminology, including kohlrabi, atlantic puffin, and oxalis purpurea.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x10.png)

Figure 10: Model predictions on dense object scenarios with Grounding DINO 1.5 Pro (part 1).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x11.png)

Figure 11: Model predictions on dense object scenarios with Grounding DINO 1.5 Pro (part 2).

### 4.6 Video Object Detection

We present video detection results of Grounding DINO 1.5 Pro in Figure 12. We notice the model can produce consistent object bounding boxes in most cases. The videos are processed offline.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x12.png)

Figure 12: Model predictions on video object detection with Grounding DINO 1.5 Pro.

### 4.7 Side-by-side Comparison

We present the side-by-side comparison between the results of Grounding DINO 1.5 Pro and Grounding DINO in Figure 13 and Figure 14. The Grounding DINO 1.5 Pro model demonstrates superior performance over the Grounding DINO model in terms of dense scene detection, long-tailed object detection, and the accuracy of semantic understanding.

Moreover, we compare the object hallucinations of Grounding DINO 1.5 Pro and Grounding DINO, as shown in Figure 15. The results show that Grounding DINO 1.5 Pro has better accuracy and fewer object hallucinations. The last line in Figure 15 demonstrates the better context understanding ability of our model.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x13.png)

Figure 13: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO 18 (part 1).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x14.png)

Figure 14: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO 18 (part 2).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x15.png)

Figure 15: Side-by-side comparison between Grounding DINO 1.5 Pro and Grounding DINO 18 regarding object hallucinations.

### 4.8 Advanced Object Detection on Edge Devices

We present the practical, real-time application potential of Grounding DINO 1.5 Edge through a series of demonstrations in Figure 16. The model’s adept performance in office environments is particularly highlighted, offering significant utility for the field of robotics research.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2405.10300/assets/x16.png)

Figure 16: The visualization of Grounding DINO 1.5 Edge on NVIDIA Orin NX. The top left of the screen displays the FPS and prompts, while the top right shows a camera view of the recorded scene.

## 5 Conclusion

This paper has presented Grounding DINO 1.5, a series of models to advance the field of open-set object detection. The flagship model, Grounding DINO 1.5 Pro, has established new records on the COCO and LVIS zero-shot benchmarks, signifying a major stride in detection accuracy and reliability. Moreover, the Grounding DINO 1.5 Edge model enables real-time object detection across various applications, further expanding the practical utility of the Grounding DINO 1.5 series.

## 6 Contributions and Acknowledgments

We would like to express our gratitude to everyone involved in the Grounding DINO 1.5 project. The contributions are as follows (in no particular order):

- Grounding DINO 1.5 Pro Model Design, Training Infra Development, Data Collection, Model Training and Model Evaluation: Tianhe Ren, Qing Jiang, Shilong Liu, and Zhaoyang Zeng.
- Grounding DINO 1.5 Edge Model Design, Training, Evaluation, and Runtime Optimization: Wenlong Liu, Han Gao, Qing Jiang, Hongjie Huang, Zhengyu Ma, Xiaoke Jiang, and Yihao Chen.
- Grounding-20M Data Collection and Annotation Pipeline Construction: Yihao Chen, Hao Zhang, Yuda Xiong, Tianhe Ren, Zhaoyang Zeng, Qing Jiang, Shilong Liu, and Peijun Tang.
- Provide Great Insight and Technical Support: Hao Zhang and Feng Li.
- Grounding DINO 1.5 Edge Model Lead: Kent Yu.
- Overall Project Lead of Grounding DINO 1.5: Lei Zhang.

We would also like to thank everyone involved in the Grounding DINO 1.5 demo support, including application lead Wei Liu, product manager Qin Liu and Xiaohui Wang, front-end developers Yuanhao Zhu, Ce Feng, and Jiongrong Fan, back-end developers Weiqiang Hu and Zhiqiang Li, UX designer Xinyi Ruan, tester Yinuo Chen, and Zijun Deng for helping with demo videos.

## 7 Appendix

### 7.1 Detailed results on the ODinW benchmark

We report the detailed results of Grounding DINO 1.5 Pro on the ODinW35 benchmarks in Table 6

| Datasets | ODinW13 | ODinW35 | Grounding DINO 1.5 Pro (zero-shot) | Grounding DINO 1.5 Pro (fine-tuning) |
| --- | --- | --- | --- | --- |
| AerialMaritimeDrone\_large | ✓ | ✓ | 19.0 | 37.0 |
| AerialMaritimeDrone\_tiled |  | ✓ | 18.2 | 45.0 |
| AmericanSignLanguageLetters |  | ✓ | 13.7 | 83.3 |
| Aquarium | ✓ | ✓ | 38.5 | 60.2 |
| BCCD |  | ✓ | 22.8 | 64.0 |
| ChessPieces |  | ✓ | 6.8 | 80.5 |
| CottontailRabbits | ✓ | ✓ | 65.7 | 75.1 |
| DroneControl |  | ✓ | 8.3 | 79.8 |
| EgoHands\_generic | ✓ | ✓ | 61.8 | 78.6 |
| EgoHands\_specific |  | ✓ | 16.7 | 78.7 |
| HardHatWorkers |  | ✓ | 20.5 | 46.4 |
| MaskWearing |  | ✓ | 16.7 | 63.2 |
| MountainDewCommercial |  | ✓ | 24.9 | 33.1 |
| NorthAmericaMushrooms | ✓ | ✓ | 82.1 | 89.2 |
| OxfordPets\_by\_breed |  | ✓ | 0.9 | 89.7 |
| OxfordPets\_by\_species |  | ✓ | 61.6 | 91.5 |
| PKLot |  | ✓ | 4.4 | 96.5 |
| Packages | ✓ | ✓ | 58.1 | 72.1 |
| PascalVOC | ✓ | ✓ | 67.1 | 77.6 |
| Raccoon | ✓ | ✓ | 72.5 | 81.8 |
| ShellfishOpenImages | ✓ | ✓ | 62.0 | 70.8 |
| ThermalCheetah |  | ✓ | 20.7 | 58.3 |
| UnoCards |  | ✓ | 1.7 | 89.3 |
| VehiclesOpenImages | ✓ | ✓ | 64.3 | 74.6 |
| WildfireSmoke |  | ✓ | 28.9 | 57.5 |
| boggleBoards |  | ✓ | 0.8 | 77.0 |
| brackishUnderwater |  | ✓ | 10.1 | 76.8 |
| dice |  | ✓ | 0.6 | 79.5 |
| openPoetryVision |  | ✓ | 0.9 | 81.2 |
| pistols | ✓ | ✓ | 71.9 | 77.6 |
| plantdoc |  | ✓ | 3.3 | 62.6 |
| pothole | ✓ | ✓ | 29.0 | 62.4 |
| selfdrivingCar |  | ✓ | 7.4 | 53.1 |
| thermalDogsAndPeople | ✓ | ✓ | 71.4 | 84.0 |
| websiteScreenshots |  | ✓ | 2.3 | 41.6 |
| ODinW13 Average AP |  |  | 58.7 | 72.4 |
| ODinW35 Average AP |  |  | 30.2 | 70.6 |

Table 6: Detailed Zero-shot Results of Grounding DINO 1.5 Pro on the ODinW35 benchmark.

[^1]: Han Cai, Junyan Li, Muyan Hu, Chuang Gan, and Song Han. Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 17302–17313, October 2023.

[^2]: Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts. CVPR, 2021.

[^3]: Tianheng Cheng, Lin Song, Yixiao Ge, Wenyu Liu, Xinggang Wang, and Ying Shan. YOLO-World: Real-Time Open-Vocabulary Object Detection. CVPR, 2024.

[^4]: Achal Dave, Piotr Dollár, Deva Ramanan, Alexander Kirillov, and Ross Girshick. Evaluating Large-Vocabulary Object Detectors: The Devil is in the Details. arXiv preprint arXiv:2102.01066, 2022.

[^5]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2020.

[^6]: Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. EVA-02: A Visual Representation for Neon Genesis. arXiv preprint arXiv:2303.11331, 2023.

[^7]: Agrim Gupta, Piotr Dollár, and Ross Girshick. LVIS: A Dataset for Large Vocabulary Instance Segmentation, 2019.

[^8]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2015.

[^9]: Qing Jiang, Feng Li, Zhaoyang Zeng, Tianhe Ren, Shilong Liu, and Lei Zhang. T-Rex2: Towards Generic Object Detection via Text-Visual Prompt Synergy. arXiv preprint arXiv:2403.14610, 2024.

[^10]: Glenn Jocher, Ayush Chaurasia, and Jing Qiu. Ultralytics YOLOv8, January 2023.

[^11]: Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion. MDETR - Modulated Detection for End-to-End Multi-Modal Understanding. ICCV, 2021.

[^12]: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment Anything. ICCV, 2023.

[^13]: Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale. IJCV, 2020.

[^14]: Feng Li, Qing Jiang, Hao Zhang, Tianhe Ren, Shilong Liu, Xueyan Zou, Huaizhe Xu, Hongyang Li, Chunyuan Li, Jianwei Yang, et al. Visual In-Context Prompting. CVPR, 2024.

[^15]: Feng Li, Ailing Zeng, Shilong Liu, Hao Zhang, Hongyang Li, Lei Zhang, and Lionel M. Ni. Lite DETR: An Interleaved Multi-Scale Encoder for Efficient DETR. CVPR, 2023.

[^16]: Liunian Harold Li\*, Pengchuan Zhang\*, Haotian Zhang\*, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded Language-Image Pre-training. CVPR, 2022.

[^17]: Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft COCO: Common Objects in Context. ECCV, 2014.

[^18]: Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499, 2023.

[^19]: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021.

[^20]: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A ConvNet for the 2020s. CVPR, 2022.

[^21]: Matthias Minderer, Alexey Gritsenko, and Neil Houlsby. Scaling Open-Vocabulary Object Detection. NeurIPS, 2023.

[^22]: Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple Open-Vocabulary Object Detection with Vision Transformers. ECCV, 2022.

[^23]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.

[^24]: Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun, Yunsheng Wu, Shaohui Lin, and Rongrong Ji. Aligning and Prompting Everything All at Once for Universal Visual Perception. CVPR, 2024.

[^25]: Jiaqi Wang, Pan Zhang, Tao Chu, Yuhang Cao, Yujie Zhou, Tong Wu, Bin Wang, Conghui He, and Dahua Lin. V3Det: Vast Vocabulary Visual Detection Dataset. ICCV, 2023.

[^26]: Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, and Shengjin Wang. Detecting Everything in the Open World: Towards Universal Object Detection. CVPR, 2023.

[^27]: Junfeng Wu, Yi Jiang, Qihao Liu, Zehuan Yuan, Xiang Bai, and Song Bai. General Object Foundation Model for Images and Videos at Scale. CVPR, 2024.

[^28]: Yifan Xu, Mengdan Zhang, Chaoyou Fu, Peixian Chen, Xiaoshan Yang, Ke Li, and Changsheng Xu. Multi-modal Queried Object Detection in the Wild. NeurIPS, 2023.

[^29]: Lewei Yao, Jianhua Han, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, and Hang Xu. DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment. CVPR, 2023.

[^30]: Lewei Yao, Jianhua Han, Youpeng Wen, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Chunjing Xu, and Hang Xu. DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection. NeurIPS, 2022.

[^31]: Lewei Yao, Renjie Pi, Jianhua Han, Xiaodan Liang, Hang Xu, Wei Zhang, Zhenguo Li, and Dan Xu. DetCLIPv3: Towards Versatile Generative Open-vocabulary Object Detection. CVPR, 2024.

[^32]: Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, Ce Liu, Mengchen Liu, Zicheng Liu, Yumao Lu, Yu Shi, Lijuan Wang, Jianfeng Wang, Bin Xiao, Zhen Xiao, Jianwei Yang, Michael Zeng, Luowei Zhou, and Pengchuan Zhang. Florence: A New Foundation Model for Computer Vision. arXiv preprint arXiv:2111.11432, 2021.

[^33]: Hao Zhang, Feng Li, Shilong Liu, Lei Zhang, Hang Su, Jun Zhu, Lionel M. Ni, and Heung-Yeung Shum. DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection. ICLR, 2023.

[^34]: Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianwei Yang, and Lei Zhang. A Simple Framework for Open-Vocabulary Segmentation and Detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1020–1031, 2023.

[^35]: Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao. GLIPv2: Unifying Localization and Vision-Language Understanding. NeurIPS, 2022.

[^36]: Tiancheng Zhao, Peng Liu, Xuan He, Lu Zhang, and Kyusong Lee. Real-time Transformer-based Open-Vocabulary Detection with Efficient Fusion Head. arXiv preprint arXiv:2403.06892, 2024.

[^37]: Yian Zhao, Wenyu Lv, Shangliang Xu, Jinman Wei, Guanzhong Wang, Qingqing Dang, Yi Liu, and Jie Chen. DETRs Beat YOLOs on Real-time Object Detection. CVPR, 2023.