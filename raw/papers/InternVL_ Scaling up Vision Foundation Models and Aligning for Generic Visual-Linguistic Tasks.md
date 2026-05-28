---
title: "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks"
source: "https://ar5iv.labs.arxiv.org/html/2312.14238"
author:
published:
created: 2026-05-28
description: "The exponential growth of large language models (LLMs) has opened up numerous possibilities for multi-modal AGI systems.However, the progress in vision and vision-language foundation models, which are also critical el…"
tags:
  - "clippings"
---
Zhe Chen <sup>2,1†</sup>, Jiannan Wu <sup>3,1†</sup>, Wenhai Wang <sup>1,4</sup>, Weijie Su <sup>6,1†</sup>, Guo Chen <sup>2,1†</sup>, Sen Xing <sup>5</sup>, Muyan Zhong <sup>5</sup>,  
Qinglong Zhang <sup>1</sup>, Xizhou Zhu <sup>5,7,1</sup>, Lewei Lu <sup>7,1</sup>, Bin Li <sup>6</sup>, Ping Luo <sup>3</sup>, Tong Lu <sup>2</sup>, Yu Qiao <sup>1</sup>, Jifeng Dai <sup>5,1</sup> <sup>🖂</sup>  
<sup>1</sup> OpenGVLab, Shanghai AI Laboratory <sup>2</sup> Nanjing University  
<sup>3</sup> The University of Hong Kong <sup>4</sup> The Chinese University of Hong Kong <sup>5</sup> Tsinghua University  
<sup>6</sup> University of Science and Technology of China <sup>7</sup> SenseTime Research  
[https://github.com/OpenGVLab/InternVL](https://github.com/OpenGVLab/InternVL)

###### Abstract

<sup>†</sup>

The exponential growth of large language models (LLMs) has opened up numerous possibilities for multi-modal AGI systems. However, the progress in vision and vision-language foundation models, which are also critical elements of multi-modal AGI, has not kept pace with LLMs. In this work, we design a large-scale vision-language foundation model (InternVL), which scales up the vision foundation model to 6 billion parameters and progressively aligns it with the LLM, using web-scale image-text data from various sources. This model can be broadly applied to and achieve state-of-the-art performance on 32 generic visual-linguistic benchmarks including visual perception tasks such as image-level or pixel-level recognition, vision-language tasks such as zero-shot image/video classification, zero-shot image/video-text retrieval, and link with LLMs to create multi-modal dialogue systems. It has powerful visual capabilities and can be a good alternative to the ViT-22B. We hope that our research could contribute to the development of multi-modal large models.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x1.png)

Figure 1: Comparisons of different vision and vision-language foundation models. (a) indicates the traditional vision foundation model, e.g. ResNet 57 pre-trained on classification tasks. (b) represents the vision-language foundation models, CLIP 117 pre-trained on image-text pairs. (c) is our InternVL, which presents a workable way to align the large-scale vision foundation model ( i.e, InternViT-6B) with the large language model and is versatile for both contrastive and generative tasks.

## 1 Introduction

Large language models (LLMs) largely promote the development of artificial general intelligence (AGI) systems with their impressive capabilities in open-world language tasks, and their model scale and performance are still increasing at a fast pace. Vision large language models (VLLMs) [^147] [^5] [^92] [^187] [^21] [^115] [^34] [^3] [^23], which leverage LLMs, have also achieved significant breakthroughs, enabling sophisticated vision-language dialogues and interactions. However, the progress of vision and vision-language foundation models, which are also crucial for VLLMs, has lagged behind the rapid growth of LLMs.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x2.png)

Figure 2: Comparison results on various generic visual-linguistic tasks, including image classification, video classification, image-text retrieval, image captioning, and multi-modal dialogue. The proposed InternVL achieves the best performance on all these tasks. Note that only the models trained on public data are included. “IN” is an abbreviation for ImageNet 38.

To bridge vision models with LLMs, existing VLLMs [^81] [^187] [^5] [^177] [^131] commonly employ lightweight “glue” layers, such as QFormer [^81] or linear projection [^92], to align features of vision and language models. Such alignment contains several limitations: (1) *Disparity in parameter scales.* The large LLMs [^48] now boosts up to 1000 billion parameters, while the widely-used vision encoders of VLLMs are still around one billion. This gap may lead to the under-use of LLM’s capacity. (2) *Inconsistent representation.* Vision models, trained on pure-vision data or aligned with the BERT series [^39] [^93] [^70], often exhibit representation inconsistencies with LLMs. (3) *Inefficient connection.* The “glue” layers are usually lightweight and randomly initialized, which may not capture the rich cross-modal interactions and dependencies that are crucial for multi-modal understanding and generation.

These limitations reveal a large gap in both parameter scale and feature representation ability between the vision encoder and the LLM. To bridge this gap, *our inspiration lies in elevating the vision encoder to align with the parameter scale of the LLM and subsequently harmonizing their representations.* However, the training of such large-scale models necessitates a vast amount of image-text data obtained from the Internet. The significant heterogeneity and quality variations within this data pose considerable challenges to the training process. To enhance the efficacy of the training, generative supervision is considered as a complementary approach to contrastive learning, as depicted in Figure 1. This strategy aims to provide additional guidance to the model during training. Yet, the suitability of low-quality data for generative training remains a concern. Besides, how to effectively represent the users’ commands and align the representations between the vision encoder and LLM is another open question.

To address these issues, we formulate the *InternVL, a large-scale vision-language foundation model, which aligns the representation of the scaled-up vision encoder with the LLM and achieves state-of-the-art performance on various visual and vision-language tasks.* As shown in Figure 1 (c), InternVL has three key designs: (1) *Parameter-balanced vision and language components*: It includes a vision encoder scaled up to 6 billion parameters and an LLM middleware with 8 billion parameters, where the middleware functions as a substantial “glue” layer to reorganize visual features based on user commands. Unlike prior vision-only (Figure 1 (a)) or dual-tower (Figure 1 (b)) structures, our vision encoder and middleware offer flexible combinations for both contrastive and generative tasks. (2) *Consistent representations*: To maintain the consistency of representations between the vision encoder and LLM, we employ a pre-trained multilingual LLaMA [^32], to initialize the middleware and align the vision encoder with it. (3) *Progressive image-text alignment*: We leverage image-text data from diverse sources, ensuring training stability through a progressive alignment strategy. This strategy initiates contrastive learning on large-scale noisy image-text data and subsequently transitions to generative learning on fine-grained data. This approach ensures a consistent enhancement of model performance and task scope.

These designs endow our model with several advantages: (1) *Versatile.* It functions as a standalone vision encoder for perception tasks, or collaborates with the language middleware for vision-language tasks and multi-modal dialogue systems. The language middleware bridges the gap between the vision encoder and the LLM decoder. (2) *Strong.* By leveraging the training strategy, large-scale parameters, and web-scale data, our model has a powerful representation that helps to achieve state-of-the-art results on various vision and vision-language tasks, as shown in Figure 2. (3) *LLM-friendly.* Due to the aligned feature space with LLMs, our model can smoothly integrate with existing LLMs, such as LLaMA series [^138] [^139], Vicuna [^184], and InternLM [^135]. These features distinguish our model from the previous approaches and establish a leading vision-language foundation model for various applications.

In summary, our contribution has three folds:

(1) We present a large-scale vision-language foundation model—InternVL, which aligns the large-scale vision encoder with LLMs for the first time. The model demonstrates strong performance on a wide range of generic visual-linguistic tasks, including visual perception tasks, vision-language tasks, and multi-modal dialogue.

(2) We introduce a progressive image-text alignment strategy for the efficient training of large-scale vision-language foundation models. This strategy maximizes the utilization of web-scale noisy image-text data for contrastive learning and fine-grained, high-quality data for generative learning.

(3) We extensively compare the proposed model with the current state-of-the-art vision foundation models and VLLMs. The results indicate that InternVL achieves leading performance on a broad range of generic visual-linguistic tasks, including image classification (ImageNet), semantic segmentation (ADE20K), video classification (Kinetics), image-text retrieval (Flickr30K & COCO), video-text retrieval (MSR-VTT), and image captioning (COCO & Flickr30K & NoCaps). Meanwhile, it is also effective for multi-modal dialogue (MME & POPE & Tiny LVLM).

## 2 Related Work

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x3.png)

Figure 3: The training strategy of the proposed InternVL model. It consists of three progressive stages, including vision-language contrastive training, vision-language generative training, and supervised fine-tuning. These stages effectively leverage public data from diverse sources, ranging from noisy image-text pairs on the web to high-quality caption, VQA, and multi-modal dialogue datasets.

### 2.1 Vision Foundation Models

The past decade has witnessed significant development in foundation models within the field of computer vision. Starting with the pioneering AlexNet [^73], a variety of convolutional neural networks (CNNs) have emerged, continuously refreshing the ImageNet benchmark [^57] [^95] [^148] [^160] [^65] [^40] [^33] [^62]. In particular, the introduction of residual connections [^57] effectively addressed the problem of vanishing gradients. This breakthrough led to an era of “big & deep” neural networks, signifying that, with adequate training and data, larger and deeper models can achieve better performance. In other words, scaling up matters.

In recent years, ViT [^42] has opened up new possibilities for network architectures in the computer vision field. ViT and its variants [^144] [^145] [^178] [^179] [^94] [^37] [^46] [^117] [^25] [^15] have significantly increased their capacity and excelled in various important visual tasks. In the LLM era, these vision foundation models often connect with LLMs through some lightweight “glue” layers [^92] [^80] [^187]. However, a gap exists as these models primarily derive from visual-only datasets like ImageNet [^38] or JFT [^173], or are aligned with the BERT series [^39] [^93] [^70] using image-text pairs, lacking direct alignment with LLMs. Additionally, the prevalent vision models employed to connect with LLMs are still limited to around 1 billion parameters [^46] [^67], which also constrains the performance of VLLMs.

### 2.2 Large Language Models

Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought exclusive to humans [^153] [^110] [^138]. The emergence of GPT-3 [^153] brought a significant leap in capabilities, particularly in few-shot and zero-shot learning, highlighting the immense potential of LLMs. This promise was further realized with the advancements of ChatGPT and GPT-4 [^110]. The progress in the field has been further accelerated by the emergence of open-source LLMs, including the LLaMA series [^138] [^139], Vicuna [^184], InternLM [^135], MOSS [^132], ChatGLM [^44], Qwen [^4], Baichuan [^6], and Falcon [^114], among others [^134] [^154] [^32]. However, in real scenarios, interactions are not limited to natural language. The vision modality can bring additional information, which means more possibilities. Therefore, exploring how to utilize the excellent capabilities of LLMs for multi-modal interactions is poised to become the next research trend.

### 2.3 Vision Large Language Models

Recent advancements have seen the creation of vision large language models (VLLMs) [^180] [^177] [^181] [^156] [^131] [^3] [^188] [^82] [^75] [^165] [^23] [^79] [^175] [^88] [^168], which aim to enhance language models with the capability to process and interpret visual information. Flamingo [^3] uses the visual and language inputs as prompts and shows remarkable few-shot performance for visual question answering. Subsequently, GPT-4 [^110], LLaVA series [^92] [^100] [^91] and MiniGPT-4 [^187] have brought in visual instruction tuning, to improve the instruction-following ability of VLLMs. Concurrently, models such as VisionLLM [^147], KOSMOS-2 [^115], and Qwen-VL *et al*. [^5] [^149] [^21] have improved VLLMs with visual grounding capabilities, facilitating tasks such as region description and localization. Many API-based methods [^96] [^155] [^125] [^166] [^133] [^163] [^97] have also attempted to integrate vision APIs with LLMs for solving vision-centric tasks. Additionally, PaLM-E [^43] and EmbodiedGPT [^108] represent advanced efforts in adapting VLLMs for embodied applications, significantly expanding their potential applications. These works showcase that VLLMs have achieved significant breakthroughs. However, the progress of vision and vision-language foundation models, equally essential for VLLMs, has not kept pace.

## 3 Proposed Method

### 3.1 Overall Architecture

As depicted in Figure 3, unlike traditional vision-only backbones [^57] [^94] [^148] and dual-encoder models [^117] [^67] [^130], the proposed InternVL is designed with a vision encoder InternViT-6B and a language middleware QLLaMA. Specifically, InternViT-6B is a vision transformer with 6 billion parameters, customized to achieve a favorable trade-off between performance and efficiency. QLLaMA is a language middleware with 8 billion parameters, initialized with a multilingual-enhanced LLaMA [^32]. It could provide robust multilingual representation for image-text contrastive learning, or serve as a bridge to connect the vision encoder and the off-the-shelf LLM decoder.

To align the two large-scale components with substantial gaps in modalities and structures, we introduce a progressive alignment training strategy. The training strategy is conducted progressively, beginning with contrastive learning on large-scale noisy data, and gradually moving towards generative learning on exquisite and high-quality data. In this way, we ensure the effective organization and full utilization of web-scale image-text data from a variety of sources. Then, equipped with the aligned vision encoder and language middleware, our model functions like a Swiss Army knife. It boasts a flexible composition that can be adapted for a wide array of generic visual-linguistic tasks. These tasks range from visual perception and image/video-text retrieval to image captioning, visual question answering, and multi-modal dialogue, among others.

| name | width | depth | MLP | #heads | #param (M) |
| --- | --- | --- | --- | --- | --- |
| ViT-G [^173] | 1664 | 48 | 8192 | 16 | 1843 |
| ViT-e [^23] | 1792 | 56 | 15360 | 16 | 3926 |
| EVA-02-ViT-E [^130] | 1792 | 64 | 15360 | 16 | 4400 |
| ViT-6.5B [^128] | 4096 | 32 | 16384 | 32 | 6440 |
| ViT-22B [^37] | 6144 | 48 | 24576 | 48 | 21743 |
| InternViT-6B (ours) | 3200 | 48 | 12800 | 25 | 5903 |

Table 1: Architecture details of the InternViT-6B model.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x4.png)

Figure 4: Different ways to use InternVL. By flexibly combining the vision encoder and the language middleware, InternVL can support various vision-language tasks, including contrastive tasks, generative tasks, and multi-modal dialogue.

### 3.2 Model Design

Large-Scale Vision Encoder: InternViT-6B. We implement the vision encoder of InternVL with vanilla vision transformer (ViT) [^42]. To match the scale of LLMs, we scale up the vision encoder to 6 billion parameters, resulting in the InternViT-6B model. To obtain a good trade-off between accuracy, speed, and stability, we conduct a hyperparameter search for InternViT-6B. We vary the model depth within {32, 48, 64, 80}, the head dimension within {64, 128}, and the MLP ratio within {4, 8}. The model width and the head number are calculated based on the given model scale and other hyperparameters.

We employ contrastive learning on a 100M subset of the LAION-en dataset [^120] to measure the accuracy, speed, and stability of InternViT-6B variants with different configurations. We report the following findings: (1) *Speed.* For different model settings, when computation is not saturated, the models with smaller depths exhibit faster speed per image. However, as the GPU computation is fully utilized, the speed difference becomes negligible; (2) *Accuracy.* With the same number of parameters, the depth, head dimension, and MLP ratio have little impact on the performance. Based on these findings, we identified the most stable configuration for our final model, as shown in Table 1.

Language Middleware: QLLaMA. The language middleware QLLaMA is proposed to align visual and linguistic features. As shown in Figure 3, QLLaMA is developed based on the pre-trained multilingual LLaMA [^32], and newly added 96 learnable queries and cross-attention layers (1 billion parameters) that are randomly initialized. This manner allows QLLaMA to smoothly integrate visual elements into the language model, thereby enhancing the coherence and effectiveness of the combined features.

Compared to recently popular approaches [^81] [^92] that use lightweight “glue” layers, such as QFormer [^81] and linear layers [^92] to connect vision encoder and LLMs, our method has three advantages: (1) By initializing with the pre-trained weights of [^32], QLLaMA can transform image tokens generated by InternViT-6B into the representation that is aligned with the LLMs; (2) QLLaMA has 8 billion parameters for vision-language alignment, which are 42 times larger than the QFormer. Therefore, even with a frozen LLM decoder, InternVL can achieve promising performance on multi-modal dialogue tasks. (3) It can also be applied to contrastive learning, providing a powerful text representation for image-text alignment tasks, such as zero-shot image classification and image-text retrieval.

“Swiss Army Knife” Model: InternVL. By flexibly combining the vision encoder and the language middleware, InternVL can support various vision or vision-language tasks.

(1) *For visual perception tasks*, the vision encoder of InternVL, *i.e*. InternViT-6B, can be used as the backbone for vision tasks. Given an input image $I\in\mathbb{R}^{H\times W\times 3}$, our model can generate a feature map $F\in\mathbb{R}^{H/14\times W/14\times D}$ for dense prediction tasks, or work with global average pooling and linear projection to make image classification.

<table><tbody><tr><td></td><td colspan="2">characteristics</td><td colspan="2">stage 1</td><td colspan="2">stage 2</td></tr><tr><td>dataset</td><td>language</td><td>original</td><td>cleaned</td><td>remain</td><td>cleaned</td><td>remain</td></tr><tr><td>LAION-en <sup><a href="#fn:120">120</a></sup></td><td rowspan="6">English</td><td>2.3B</td><td>1.94B</td><td>84.3%</td><td>91M</td><td>4.0%</td></tr><tr><td>LAION-COCO <sup><a href="#fn:121">121</a></sup></td><td>663M</td><td>550M</td><td>83.0%</td><td>550M</td><td>83.0%</td></tr><tr><td>COYO <sup><a href="#fn:14">14</a></sup></td><td>747M</td><td>535M</td><td>71.6%</td><td>200M</td><td>26.8%</td></tr><tr><td>CC12M <sup><a href="#fn:20">20</a></sup></td><td>12.4M</td><td>11.1M</td><td>89.5%</td><td>11.1M</td><td>89.5%</td></tr><tr><td>CC3M <sup><a href="#fn:124">124</a></sup></td><td>3.0M</td><td>2.6M</td><td>86.7%</td><td>2.6M</td><td>86.7%</td></tr><tr><td>SBU <sup><a href="#fn:112">112</a></sup></td><td>1.0M</td><td>1.0M</td><td>100%</td><td>1.0M</td><td>100%</td></tr><tr><td>Wukong <sup><a href="#fn:55">55</a></sup></td><td>Chinese</td><td>100M</td><td>69.4M</td><td>69.4%</td><td>69.4M</td><td>69.4%</td></tr><tr><td>LAION-multi <sup><a href="#fn:120">120</a></sup></td><td>Multi</td><td>2.2B</td><td>1.87B</td><td>85.0%</td><td>100M</td><td>4.5%</td></tr><tr><td>Total</td><td>Multi</td><td>6.03B</td><td>4.98B</td><td>82.6%</td><td>1.03B</td><td>17.0%</td></tr></tbody></table>

Table 2: Details of the training data for InternVL in stage 1 and stage 2. Among them, LAION-en [^120], LAION-multi [^120], COYO [^14], and Wukong [^55] are web-scale image-text pairs data. LAION-COCO [^121] is a synthetic dataset with high-quality captions from LAION-en. CC12M [^20], CC3M [^124], SBU [^112] are academic caption datasets. “Multi” means multilingual.

(2) *For contrastive tasks*, as shown in Figure 4 (a) (b), we introduce two inference modes: InternVL-C and InternVL-G, using the vision encoder or the combination of InternViT and QLLaMA to encode visual features. Specifically, we apply attention pooling to the visual features of InternViT or the query features of QLLaMA, to calculate the global visual feature $I_{f}$. Besides, we encode text as $T_{f}$ by extracting the feature from the \[EOS\] token of QLLaMA. By computing similarity scores between $I_{f}$ and $T_{f}$, we support various contrastive tasks such as image-text retrieval.

(3) *For generative tasks*, unlike QFormer [^80], QLLaMA inherently has promising image captioning abilities thanks to its scaled-up parameters. The queries of QLLaMA reorganize the visual representations from InternViT-6B and play as the prefix texts for QLLaMA. The subsequent text tokens are generated one by one sequentially.

(4) *For multi-modal dialogue*, we introduce InternVL-Chat, leveraging InternVL as the visual component to connect with LLMs. For this purpose, we have two distinct configurations. One option is to employ the InternViT-6B independently, as shown in Figure 4 (c). The alternative is to employ the complete InternVL model concurrently, as illustrated in Figure 4 (d).

| task | #samples | dataset |
| --- | --- | --- |
| Captioning | 588K | COCO Caption [^22], TextCaps [^126] |
|  |  | VQAv2 [^54], OKVQA [^104], A-OKVQA [^122], |
| VQA | 1.1M | IconQA [^99], AI2D [^71], GQA [^64] |
|  |  | OCR-VQA [^107], ChartQA [^105], DocVQA [^29], |
|  |  | ST-VQA [^12], EST-VQA [^150], InfoVQA [^106], |
| OCR | 294K | LLaVAR [^182] |
| Grounding | 323K | RefCOCO/+/g [^170] [^103], Toloka [^140] |
| Grounded Cap. | 284K | RefCOCO/+/g [^170] [^103] |
|  |  | LLaVA-150K [^92], SVIT [^183], VisDial [^36], |
| Conversation | 1.4M | LRV-Instruction [^90], LLaVA-Mix-665K [^91] |

Table 3: Details of the training data for InternVL in stage 3. We collect a wide range of high-quality instruction data, totaling approximately 4 million samples. For a fair comparison, we only use the training split of these datasets.

### 3.3 Alignment Strategy

As shown in Figure 3, the training of InternVL consists of three progressive stages, including vision-language contrastive training, vision-language generative training, and supervised fine-tuning. These stages effectively leverage public data from diverse sources, ranging from noisy image-text pairs on the web to high-quality caption, VQA, and multi-modal dialogue datasets.

Vision-Language Contrastive Training. In the first stage, we conduct contrastive learning to align InternViT-6B with a multilingual LLaMA-7B [^32] on web-scale, noisy image-text pairs. The data are all publicly available and comprise multilingual content, including LAION-en [^120], LAION-multi [^120], LAION-COCO [^121], COYO [^14], Wukong [^55], etc. We use the combination of these datasets and filter out some extremely low-quality data to train our model. As summarized in Table 2, the original dataset contains 6.03 billion image-text pairs, and 4.98 billion remains after cleaning. More details about data preparation will be provided in the supplementary materials.

During training, we adopt the LLaMA-7B to encode the text as $T_{f}$, and use InternViT-6B to extract the visual feature $I_{f}$. Following the objective function of CLIP [^117], we minimize a symmetric cross-entropy loss on the similarity scores of image-text pairs in a batch. This stage allows InternVL to excel on contrastive tasks like zero-shot image classification and image-text retrieval, and the vision encoder of this stage can also perform well on visual perception tasks like semantic segmentation.

Vision-Language Generative Training. In the second stage of training, we connect InternViT-6B with QLLaMA and adopt a generative training strategy. Specifically, QLLaMA inherits the weights of LLaMA-7B in the first stage. We keep both InternViT-6B and QLLaMA frozen and only train the newly added learnable queries and cross-attention layers with filtered, high-quality data. Table 2 summarizes the datasets for the second stage. It can be seen that we further filtered out data with low-quality captions, reducing it from 4.98 billion in the first stage to 1.03 billion.

Following the loss function of BLIP-2 [^81], the loss in this stage is computed as the sum of three components: image-text contrastive (ITC) loss, image-text matching (ITM) loss, and image-grounded text generation (ITG) loss. This enables the queries to extract powerful visual representations, and further align feature space with LLMs, attributable to the effective training objectives and the utilization of our large-scale, LLM-initialized QLLaMA.

| method | #param | IN-1K | IN-ReaL | IN-V2 | IN-A | IN-R | IN-Ske | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenCLIP-H [^67] | 0.6B | 84.4 | 88.4 | 75.5 | $-$ | $-$ | $-$ | $-$ |
| OpenCLIP-G [^67] | 1.8B | 86.2 | 89.4 | 77.2 | 63.8 | 87.8 | 66.4 | 78.5 |
| DINOv2-g [^111] | 1.1B | 86.5 | 89.6 | 78.4 | 75.9 | 78.8 | 62.5 | 78.6 |
| EVA-01-CLIP-g [^46] | 1.1B | 86.5 | 89.3 | 77.4 | 70.5 | 87.7 | 63.1 | 79.1 |
| MAWS-ViT-6.5B [^128] | 6.5B | 87.8 | – | – | – | – | – | – |
| ViT-22B <sup>∗</sup> [^37] | 21.7B | 89.5 | 90.9 | 83.2 | 83.8 | 87.4 | $-$ | $-$ |
| InternViT-6B (ours) | 5.9B | 88.2 | 90.4 | 79.9 | 77.5 | 89.8 | 69.1 | 82.5 |

Table 4: Linear evaluation on image classification. We report the top-1 accuracy on ImageNet-1K [^38] and its variants [^10] [^119] [^61] [^60] [^141]. <sup>∗</sup> ViT-22B [^37] uses the private JFT-3B dataset [^173].

| method | #param | crop size | $1/16$ | $1/8$ | $1/4$ | $1/2$ | $1$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-L [^137] | 0.3B | 504 <sup>2</sup> | 36.1 | 41.3 | 45.6 | 48.4 | 51.9 |
| ViT-G [^173] | 1.8B | 504 <sup>2</sup> | 42.4 | 47.0 | 50.2 | 52.4 | 55.6 |
| ViT-22B [^37] | 21.7B | 504 <sup>2</sup> | 44.7 | 47.2 | 50.6 | 52.5 | 54.9 |
| InternViT-6B (ours) | 5.9B | 504 <sup>2</sup> | 46.5 | 50.0 | 53.3 | 55.8 | 57.2 |

(a) Few-shot semantic segmentation with limited training data. Following ViT-22B [^37], we fine-tune the InternViT-6B with a linear classifier.

| method | decoder | #param (train/total) | crop size | mIoU |
| --- | --- | --- | --- | --- |
| OpenCLIP-G <sub>frozen</sub> [^67] | Linear | 0.3M / 1.8B | 512 <sup>2</sup> | 39.3 |
| ViT-22B <sub>frozen</sub> [^37] | Linear | 0.9M / 21.7B | 504 <sup>2</sup> | 34.6 |
| InternViT-6B <sub>frozen</sub> (ours) | Linear | 0.5M / 5.9B | 504 <sup>2</sup> | 47.2 |
| ViT-22B <sub>frozen</sub> [^37] | UperNet | 0.8B / 22.5B | 504 <sup>2</sup> | 52.7 |
| InternViT-6B <sub>frozen</sub> (ours) | UperNet | 0.4B / 6.3B | 504 <sup>2</sup> | 54.9 |
| ViT-22B [^37] | UperNet | 22.5B / 22.5B | 504 <sup>2</sup> | 55.3 |
| InternViT-6B (ours) | UperNet | 6.3B / 6.3B | 504 <sup>2</sup> | 58.9 |

(b) Semantic segmentation performance in three different settings, from top to bottom: linear probing, head tuning, and full-parameter tuning.

Table 5: Semantic segmentation on ADE20K. Results show that InternViT-6B has better pixel-level perceptual capacity.

Supervised Fine-tuning. To demonstrate the benefits of InternVL in creating multi-modal dialogue systems, we connect it with an off-the-shelf LLM decoder (*e.g*., Vicuna [^184] or InternLM [^135]) through an MLP layer, and conduct supervised fine-tuning (SFT). As detailed in Table 3, we collect a wide range of high-quality instruction data, totaling approximately 4 million samples. For non-dialogue datasets, we follow the method described in [^91] for conversion. Owing to the similar feature space of QLLaMA and LLMs, we can achieve robust performance even when freezing the LLM decoder, choosing to train just the MLP layer or both the MLP layer and QLLaMA. This approach not only expedites the SFT process but also maintains the original language capabilities of the LLMs.

| method | IN-1K | IN-A | IN-R | IN-V2 | IN-Sketch | ObjectNet | $\Delta$ $\downarrow$ | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenCLIP-H [^67] | 78.0 | 59.3 | 89.3 | 70.9 | 66.6 | 69.7 | 5.7 | 72.3 |
| OpenCLIP-g [^67] | 78.5 | 60.8 | 90.2 | 71.7 | 67.5 | 69.2 | 5.5 | 73.0 |
| OpenAI CLIP-L+ [^117] | 76.6 | 77.5 | 89.0 | 70.9 | 61.0 | 72.0 | 2.1 | 74.5 |
| EVA-01-CLIP-g [^130] | 78.5 | 73.6 | 92.5 | 71.5 | 67.3 | 72.3 | 2.5 | 76.0 |
| OpenCLIP-G [^67] | 80.1 | 69.3 | 92.1 | 73.6 | 68.9 | 73.0 | 3.9 | 76.2 |
| EVA-01-CLIP-g+ [^130] | 79.3 | 74.1 | 92.5 | 72.1 | 68.1 | 75.3 | 2.4 | 76.9 |
| MAWS-ViT-2B [^128] | 81.9 | – | – | – | – | – | – | – |
| EVA-02-CLIP-E+ [^130] | 82.0 | 82.1 | 94.5 | 75.7 | 71.6 | 79.6 | 1.1 | 80.9 |
| CoCa <sup>∗</sup> [^169] | 86.3 | 90.2 | 96.5 | 80.7 | 77.6 | 82.7 | 0.6 | 85.7 |
| LiT-22B <sup>∗</sup> [^37] [^174] | 85.9 | 90.1 | 96.0 | 80.9 | $-$ | 87.6 | $-$ | $-$ |
| InternVL-C (ours) | 83.2 | 83.8 | 95.5 | 77.3 | 73.9 | 80.6 | 0.8 | 82.4 |

(a) ImageNet variants [^38] [^61] [^60] [^119] [^141] and ObjectNet [^8].

| method | EN | ZH | JP | AR | IT | avg. |
| --- | --- | --- | --- | --- | --- | --- |
| M-CLIP [^16] | $-$ | $-$ | $-$ | $-$ | 20.2 | $-$ |
| CLIP-Italian [^11] | $-$ | $-$ | $-$ | $-$ | 22.1 | $-$ |
| Japanese-CLIP-ViT-B [^102] | $-$ | $-$ | 54.6 | $-$ | $-$ | $-$ |
| Taiyi-CLIP-ViT-H [^176] | $-$ | 54.4 | $-$ | $-$ | $-$ | $-$ |
| WuKong-ViT-L-G [^55] | $-$ | 57.5 | $-$ | $-$ | $-$ | $-$ |
| CN-CLIP-ViT-H [^162] | $-$ | 59.6 | $-$ | $-$ | $-$ | $-$ |
| AltCLIP-ViT-L [^26] | 74.5 | 59.6 | $-$ | $-$ | $-$ | $-$ |
| EVA-02-CLIP-E+ [^130] | 82.0 | 3.6 | 5.0 | 0.2 | 41.2 | $-$ |
| OpenCLIP-XLM-R-B [^67] | 62.3 | 42.7 | 37.9 | 26.5 | 43.7 | 42.6 |
| OpenCLIP-XLM-R-H [^67] | 77.0 | 55.7 | 53.1 | 37.0 | 56.8 | 55.9 |
| InternVL-C (ours) | 83.2 | 64.5 | 61.5 | 44.9 | 65.7 | 64.0 |

(b) Multilingual ImageNet-1K [^38] [^76].

Table 6: Comparison of zero-shot image classification performance. “ $\Delta$ $\downarrow$ ”: The gap between the averaged top-1 accuracy and the IN-1K top-1 accuracy. <sup>∗</sup> CoCa [^169] and LiT-22B [^37] use the private JFT-3B dataset [^173] during training. Multilingual evaluation involves 5 languages, including English (EN), Chinese (ZH), Japanese (JP), Arabic (AR), and Italian (IT).

<table><tbody><tr><td></td><td></td><td colspan="6">Flickr30K (English, 1K test set) <sup><a href="#fn:116">116</a></sup></td><td colspan="6">COCO (English, 5K test set) <sup><a href="#fn:22">22</a></sup></td><td></td></tr><tr><td></td><td>multi-</td><td colspan="3">Image <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Text</td><td colspan="3">Text <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Image</td><td colspan="3">Image <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Text</td><td colspan="3">Text <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Image</td><td></td></tr><tr><td>method</td><td>lingual</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>avg.</td></tr><tr><td>Florence <sup><a href="#fn:171">171</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>90.9</td><td>99.1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>76.7</td><td>93.6</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>64.7</td><td>85.9</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>47.2</td><td>71.4</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td></tr><tr><td>ONE-PEACE <sup><a href="#fn:143">143</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>90.9</td><td>98.8</td><td>99.8</td><td>77.2</td><td>93.5</td><td>96.2</td><td>64.7</td><td>86.0</td><td>91.9</td><td>48.0</td><td>71.5</td><td>79.6</td><td>83.2</td></tr><tr><td>OpenCLIP-H <sup><a href="#fn:67">67</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>90.8</td><td>99.3</td><td>99.7</td><td>77.8</td><td>94.1</td><td>96.6</td><td>66.0</td><td>86.1</td><td>91.9</td><td>49.5</td><td>73.4</td><td>81.5</td><td>83.9</td></tr><tr><td>OpenCLIP-g <sup><a href="#fn:67">67</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>91.4</td><td>99.2</td><td>99.6</td><td>77.7</td><td>94.1</td><td>96.9</td><td>66.4</td><td>86.0</td><td>91.8</td><td>48.8</td><td>73.3</td><td>81.5</td><td>83.9</td></tr><tr><td>OpenCLIP-XLM-R-H <sup><a href="#fn:67">67</a></sup></td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>91.8</td><td>99.4</td><td>99.8</td><td>77.8</td><td>94.1</td><td>96.5</td><td>65.9</td><td>86.2</td><td>92.2</td><td>49.3</td><td>73.2</td><td>81.5</td><td>84.0</td></tr><tr><td>EVA-01-CLIP-g+ <sup><a href="#fn:130">130</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>91.6</td><td>99.3</td><td>99.8</td><td>78.9</td><td>94.5</td><td>96.9</td><td>68.2</td><td>87.5</td><td>92.5</td><td>50.3</td><td>74.0</td><td>82.1</td><td>84.6</td></tr><tr><td>CoCa <sup><a href="#fn:169">169</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>92.5</td><td>99.5</td><td>99.9</td><td>80.4</td><td>95.7</td><td>97.7</td><td>66.3</td><td>86.2</td><td>91.8</td><td>51.2</td><td>74.2</td><td>82.0</td><td>84.8</td></tr><tr><td>OpenCLIP-G <sup><a href="#fn:67">67</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>92.9</td><td>99.3</td><td>99.8</td><td>79.5</td><td>95.0</td><td>97.1</td><td>67.3</td><td>86.9</td><td>92.6</td><td>51.4</td><td>74.9</td><td>83.0</td><td>85.0</td></tr><tr><td>EVA-02-CLIP-E+ <sup><a href="#fn:130">130</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>93.9</td><td>99.4</td><td>99.8</td><td>78.8</td><td>94.2</td><td>96.8</td><td>68.8</td><td>87.8</td><td>92.8</td><td>51.1</td><td>75.0</td><td>82.7</td><td>85.1</td></tr><tr><td>BLIP-2 <sup>†</sup> <sup><a href="#fn:81">81</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>97.6</td><td>100.0</td><td>100.0</td><td>89.7</td><td>98.1</td><td>98.9</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td></tr><tr><td>InternVL-C (ours)</td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>94.7</td><td>99.6</td><td>99.9</td><td>81.7</td><td>96.0</td><td>98.2</td><td>70.6</td><td>89.0</td><td>93.5</td><td>54.1</td><td>77.3</td><td>84.6</td><td>86.6</td></tr><tr><td>InternVL-G (ours)</td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>95.7</td><td>99.7</td><td>99.9</td><td>85.0</td><td>97.0</td><td>98.6</td><td>74.9</td><td>91.3</td><td>95.2</td><td>58.6</td><td>81.3</td><td>88.0</td><td>88.8</td></tr><tr><td>method</td><td></td><td colspan="6">Flickr30K-CN (Chinese, 1K test set) <sup><a href="#fn:77">77</a></sup></td><td colspan="6">COCO-CN (Chinese, 1K test set) <sup><a href="#fn:84">84</a></sup></td><td>avg.</td></tr><tr><td>WuKong-ViT-L <sup><a href="#fn:55">55</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>76.1</td><td>94.8</td><td>97.5</td><td>51.7</td><td>78.9</td><td>86.3</td><td>55.2</td><td>81.0</td><td>90.6</td><td>53.4</td><td>80.2</td><td>90.1</td><td>78.0</td></tr><tr><td>R2D2-ViT-L <sup><a href="#fn:159">159</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>77.6</td><td>96.7</td><td>98.9</td><td>60.9</td><td>86.8</td><td>92.7</td><td>63.3</td><td>89.3</td><td>95.7</td><td>56.4</td><td>85.0</td><td>93.1</td><td>83.0</td></tr><tr><td>Taiyi-CLIP-ViT-H <sup><a href="#fn:176">176</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>60.0</td><td>84.0</td><td>93.3</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td></tr><tr><td>AltCLIP-ViT-H <sup><a href="#fn:26">26</a></sup></td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>88.9</td><td>98.5</td><td>99.5</td><td>74.5</td><td>92.0</td><td>95.5</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td></tr><tr><td>CN-CLIP-ViT-H <sup><a href="#fn:162">162</a></sup></td><td><math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math></td><td>81.6</td><td>97.5</td><td>98.8</td><td>71.2</td><td>91.4</td><td>95.5</td><td>63.0</td><td>86.6</td><td>92.9</td><td>69.2</td><td>89.9</td><td>96.1</td><td>86.1</td></tr><tr><td>OpenCLIP-XLM-R-H <sup><a href="#fn:67">67</a></sup></td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>86.1</td><td>97.5</td><td>99.2</td><td>71.0</td><td>90.5</td><td>94.9</td><td>70.0</td><td>91.5</td><td>97.0</td><td>66.1</td><td>90.8</td><td>96.0</td><td>87.6</td></tr><tr><td>InternVL-C (ours)</td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>90.3</td><td>98.8</td><td>99.7</td><td>75.1</td><td>92.9</td><td>96.4</td><td>68.8</td><td>92.0</td><td>96.7</td><td>68.9</td><td>91.9</td><td>96.5</td><td>89.0</td></tr><tr><td>InternVL-G (ours)</td><td><math><semantics><mi>✓</mi> <ci>✓</ci> <annotation>\checkmark</annotation></semantics></math></td><td>92.9</td><td>99.4</td><td>99.8</td><td>77.7</td><td>94.8</td><td>97.3</td><td>71.4</td><td>93.9</td><td>97.7</td><td>73.8</td><td>94.4</td><td>98.1</td><td>90.9</td></tr></tbody></table>

Table 7: Comparison of zero-shot image-text retrieval performance. We evaluate the retrieval capability in English using the Flickr30K [^116] and COCO [^22], as well as in Chinese using Flickr30K-CN [^77] and COCO-CN [^84]. <sup>†</sup> BLIP-2 [^81] is finetuned on COCO and zero-shot transferred to Flickr30K, contributing to the enhanced zero-shot performance on Flickr30K.

## 4 Experiments

### 4.1 Implementation Details

Stage 1. In this stage, the image encoder InternViT-6B is randomly initialized [^7], and the text encoder LLaMA-7B is initialized with the pre-trained weights from [^32]. All parameters are fully trainable.

Stage 2. In this stage, InternViT-6B and QLLaMA inherit their weights from the first stage, while the new learnable queries and cross-attention layers in QLLaMA are randomly initialized. Benefiting from the powerful representations learned in the first stage, we keep both InternViT-6B and QLLaMA frozen and only train the new parameters.

Stage 3. At this stage, we have two different configurations. One is to use InternViT-6B separately, as shown in Figure 4 (c). The other is to use the entire InternVL model simultaneously, as shown in Figure 4 (d). More details will be provided in the supplementary materials.

### 4.2 Visual Perception Benchmarks

First of all, we validate the visual perception capabilities of InternViT-6B, the most core component of InternVL.

Transfer to Image Classification. We evaluate the quality of visual representation produced by InternViT-6B using the ImageNet-1K [^38] dataset. Following common practices [^58] [^111] [^37], we adopt the linear probing evaluation, *i.e*. training a linear classifier while keeping the backbone frozen. In addition to the ImageNet-1K validation set, we also report performance metrics on several ImageNet variants [^10] [^119] [^61] [^60] [^141], to benchmark the domain generalization capability. As shown in Table 4, InternViT-6B achieves a very significant improvement over previous state-of-the-art methods [^46] [^111] [^67] on linear probing. To our knowledge, this represents the currently best linear evaluation results without the JFT dataset [^173].

Transfer to Semantic Segmentation. To investigate the pixel-level perceptual capacity of InternViT-6B, we conduct extensive experiments of semantic segmentation on the ADE20K [^185] dataset. Following ViT-22B [^37], we begin with few-shot learning experiments, *i.e*. fine-tuning the backbone with a linear head on a limited dataset. As indicated in Table 5(a), InternViT-6B consistently outperforms ViT-22B across five experiments with varying proportions of training data. Additionally, Table 5(b) presents our further verification in three distinct settings, including linear probing, head tuning [^158], and full-parameter tuning. Notably, in the case of linear probing, InternViT-6B attains 47.2 mIoU, a substantial +12.6 mIoU improvement over ViT-22B. These results underscore the strong out-of-the-box pixel-level perceptual capacity of our InternViT-6B.

<table><tbody><tr><td></td><td></td><td colspan="2">K400 <sup><a href="#fn:17">17</a></sup></td><td colspan="2">K600 <sup><a href="#fn:18">18</a></sup></td><td colspan="2">K700 <sup><a href="#fn:19">19</a></sup></td></tr><tr><td>method</td><td>#F</td><td>top-1</td><td>avg.</td><td>top-1</td><td>avg.</td><td>top-1</td><td>avg.</td></tr><tr><td>OpenCLIP-g <sup><a href="#fn:67">67</a></sup></td><td>1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>63.9</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>64.1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>56.9</td></tr><tr><td>OpenCLIP-G <sup><a href="#fn:67">67</a></sup></td><td>1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>65.9</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>66.1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>59.2</td></tr><tr><td>EVA-01-CLIP-g+ <sup><a href="#fn:130">130</a></sup></td><td>1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>66.7</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>67.0</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>60.9</td></tr><tr><td>EVA-02-CLIP-E+ <sup><a href="#fn:130">130</a></sup></td><td>1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>69.8</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>69.3</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>63.4</td></tr><tr><td>InternVL-C (ours)</td><td>1</td><td>65.9</td><td>76.1</td><td>65.5</td><td>75.5</td><td>56.8</td><td>67.5</td></tr><tr><td>ViCLIP <sup><a href="#fn:152">152</a></sup></td><td>8</td><td>64.8</td><td>75.7</td><td>62.2</td><td>73.5</td><td>54.3</td><td>66.4</td></tr><tr><td>InternVL-C (ours)</td><td>8</td><td>69.1</td><td>79.4</td><td>68.9</td><td>78.8</td><td>60.6</td><td>71.5</td></tr></tbody></table>

Table 8: Comparison of zero-shot video classification results on Kinetics 400/600/700. We report the top-1 accuracy and the mean of top-1 and top-5 accuracy. “#F” denotes the number of frames.

<table><tbody><tr><td></td><td>visual</td><td>glue</td><td></td><td></td><td></td><td></td><td>train.</td><td colspan="3">image captioning</td><td colspan="4">visual question answering</td><td colspan="2">dialogue</td></tr><tr><td>method</td><td>encoder</td><td>layer</td><td>LLM</td><td>Res.</td><td>PT</td><td>SFT</td><td>param</td><td>COCO</td><td>Flickr</td><td>NoCaps</td><td>VQA <math><semantics><msup><mtext>v2</mtext></msup> <apply><ci><mtext>v2</mtext></ci></apply> <annotation>{}^{\text{v2}}</annotation></semantics></math></td><td>GQA</td><td>VizWiz</td><td>VQA <math><semantics><msup><mtext>T</mtext></msup> <apply><ci><mtext>T</mtext></ci></apply> <annotation>{}^{\text{T}}</annotation></semantics></math></td><td>MME</td><td>POPE</td></tr><tr><td>InstructBLIP <sup><a href="#fn:34">34</a></sup></td><td>EVA-g</td><td>QFormer</td><td>Vicuna-7B</td><td>224</td><td>129M</td><td>1.2M</td><td>188M</td><td>–</td><td>82.4</td><td>123.1</td><td>–</td><td>49.2</td><td>34.5</td><td>50.1</td><td>–</td><td>–</td></tr><tr><td>BLIP-2 <sup><a href="#fn:81">81</a></sup></td><td>EVA-g</td><td>QFormer</td><td>Vicuna-13B</td><td>224</td><td>129M</td><td>–</td><td>188M</td><td>–</td><td>71.6</td><td>103.9</td><td>41.0</td><td>41.0</td><td>19.6</td><td>42.5</td><td>1293.8</td><td>85.3</td></tr><tr><td>InstructBLIP <sup><a href="#fn:34">34</a></sup></td><td>EVA-g</td><td>QFormer</td><td>Vicuna-13B</td><td>224</td><td>129M</td><td>1.2M</td><td>188M</td><td>–</td><td>82.8</td><td>121.9</td><td>–</td><td>49.5</td><td>33.4</td><td>50.7</td><td>1212.8</td><td>78.9</td></tr><tr><td>InternVL-Chat (ours)</td><td>IViT-6B</td><td>QLLaMA</td><td>Vicuna-7B</td><td>224</td><td>1.0B</td><td>4.0M</td><td>64M</td><td>141.4 <sup>∗</sup></td><td>89.7</td><td>120.5</td><td>72.3 <sup>∗</sup></td><td>57.7 <sup>∗</sup></td><td>44.5</td><td>42.1</td><td>1298.5</td><td>85.2</td></tr><tr><td>InternVL-Chat (ours)</td><td>IViT-6B</td><td>QLLaMA</td><td>Vicuna-13B</td><td>224</td><td>1.0B</td><td>4.0M</td><td>90M</td><td>142.4 <sup>∗</sup></td><td>89.9</td><td>123.1</td><td>71.7 <sup>∗</sup></td><td>59.5 <sup>∗</sup></td><td>54.0</td><td>49.1</td><td>1317.2</td><td>85.4</td></tr><tr><td>Shikra <sup><a href="#fn:21">21</a></sup></td><td>CLIP-L</td><td>Linear</td><td>Vicuna-13B</td><td>224</td><td>600K</td><td>5.5M</td><td>7B</td><td>117.5 <sup>∗</sup></td><td>73.9</td><td>–</td><td>77.4 <sup>∗</sup></td><td>–</td><td>–</td><td>–</td><td>–</td><td>–</td></tr><tr><td>IDEFICS-80B <sup><a href="#fn:66">66</a></sup></td><td>CLIP-H</td><td>Cross-Attn</td><td>LLaMA-65B</td><td>224</td><td>1.6B</td><td>–</td><td>15B</td><td>91.8 <sup>∗</sup></td><td>53.7</td><td>65.0</td><td>60.0</td><td>45.2</td><td>36.0</td><td>30.9</td><td>–</td><td>–</td></tr><tr><td>IDEFICS-80B-I <sup><a href="#fn:66">66</a></sup></td><td>CLIP-H</td><td>Cross-Attn</td><td>LLaMA-65B</td><td>224</td><td>353M</td><td>6.7M</td><td>15B</td><td>117.2 <sup>∗</sup></td><td>65.3</td><td>104.5</td><td>37.4</td><td>–</td><td>26.0</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Qwen-VL <sup><a href="#fn:5">5</a></sup></td><td>CLIP-G</td><td>VL-Adapter</td><td>Qwen-7B</td><td>448</td><td>1.4B <sup>†</sup></td><td>50M <sup>†</sup></td><td>9.6B</td><td>–</td><td>85.8</td><td>121.4</td><td>78.8 <sup>∗</sup></td><td>59.3 <sup>∗</sup></td><td>35.2</td><td>63.8</td><td>–</td><td>–</td></tr><tr><td>Qwen-VL-Chat <sup><a href="#fn:5">5</a></sup></td><td>CLIP-G</td><td>VL-Adapter</td><td>Qwen-7B</td><td>448</td><td>1.4B <sup>†</sup></td><td>50M <sup>†</sup></td><td>9.6B</td><td>–</td><td>81.0</td><td>120.2</td><td>78.2 <sup>∗</sup></td><td>57.5 <sup>∗</sup></td><td>38.9</td><td>61.5</td><td>1487.5</td><td>–</td></tr><tr><td>LLaVA-1.5 <sup><a href="#fn:91">91</a></sup></td><td>CLIP-L <sub>336</sub></td><td>MLP</td><td>Vicuna-7B</td><td>336</td><td>558K</td><td>665K</td><td>7B</td><td>–</td><td>–</td><td>–</td><td>78.5 <sup>∗</sup></td><td>62.0 <sup>∗</sup></td><td>50.0</td><td>58.2</td><td>1510.7</td><td>85.9</td></tr><tr><td>LLaVA-1.5 <sup><a href="#fn:91">91</a></sup></td><td>CLIP-L <sub>336</sub></td><td>MLP</td><td>Vicuna-13B</td><td>336</td><td>558K</td><td>665K</td><td>13B</td><td>–</td><td>–</td><td>–</td><td>80.0 <sup>∗</sup></td><td>63.3 <sup>∗</sup></td><td>53.6</td><td>61.3</td><td>1531.3</td><td>85.9</td></tr><tr><td>InternVL-Chat (ours)</td><td>IViT-6B</td><td>MLP</td><td>Vicuna-7B</td><td>336</td><td>558K</td><td>665K</td><td>7B</td><td>–</td><td>–</td><td>–</td><td>79.3 <sup>∗</sup></td><td>62.9 <sup>∗</sup></td><td>52.5</td><td>57.0</td><td>1525.1</td><td>86.4</td></tr><tr><td>InternVL-Chat (ours)</td><td>IViT-6B</td><td>MLP</td><td>Vicuna-13B</td><td>336</td><td>558K</td><td>665K</td><td>13B</td><td>–</td><td>–</td><td>–</td><td>80.2 <sup>∗</sup></td><td>63.9 <sup>∗</sup></td><td>54.6</td><td>58.7</td><td>1546.9</td><td>87.1</td></tr><tr><td>InternVL-Chat (ours)</td><td>IViT-6B</td><td>QLLaMA</td><td>Vicuna-13B</td><td>336</td><td>1.0B</td><td>4.0M</td><td>13B</td><td>146.2 <sup>∗</sup></td><td>92.2</td><td>126.2</td><td>81.2 <sup>∗</sup></td><td>66.6 <sup>∗</sup></td><td>58.5</td><td>61.5</td><td>1586.4</td><td>87.6</td></tr></tbody></table>

Table 9: Comparison with SoTA methods on 9 benchmarks. Image captioning datasets include: COCO Karpathy test [^22], Flickr30K Karpathy test [^116], NoCaps val [^2]. VQA datasets include: VQAv2 test-dev [^54], GQA test-balanced [^64], VizWiz test-dev [^56], and TextVQA val [^127]. <sup>∗</sup> The training annotations of the datasets are observed during training. “IViT-6B” represents our InternViT-6B.

| method | glue layer | LLM decoder | COCO | Flickr30K | NoCaps |
| --- | --- | --- | --- | --- | --- |
| Flamingo-9B [^3] | Cross-Attn | Chinchilla-7B | 79.4 | 61.5 | – |
| Flamingo-80B [^3] | Cross-Attn | Chinchilla-70B | 84.3 | 67.2 | – |
| KOSMOS-2 [^115] | Linear | KOSMOS-1 | – | 66.7 | – |
| PaLI-X-55B [^24] | Linear | UL2-32B | – | – | 126.3 |
| BLIP-2 [^81] | QFormer | Vicuna-13B | – | 71.6 | 103.9 |
| InstructBLIP [^34] | QFormer | Vicuna-13B | – | 82.8 | 121.9 |
| Shikra-13B [^21] | Linear | Vicuna-13B | – | 73.9 | – |
| ASM [^149] | QFormer | Husky-7B | – | 87.7 | 117.2 |
| Qwen-VL [^5] | VL-Adapter | Qwen-7B | – | 85.8 | 121.4 |
| Qwen-VL-Chat [^5] | VL-Adapter | Qwen-7B | – | 81.0 | 120.2 |
| Emu [^131] | QFormer | LLaMA-13B | 112.4 | – | – |
| Emu-I [^131] | QFormer | LLaMA-13B | 117.7 | – | – |
| DreamLLM [^41] | Linear | Vicuna-7B | 115.4 | – | – |
| InternVL-G (ours) | Cross-Attn | QLLaMA | 128.2 | 79.2 | 113.7 |

Table 10: Comparison of zero-shot image captioning. QLLaMA inherently possesses promising zero-shot captioning capabilities thanks to its scaled-up parameters and datasets.

### 4.3 Vision-Language Benchmarks

In this section, we evaluate the inherent capabilities of InternVL on various vision-language tasks.

Zero-Shot Image Classification. We conduct thorough validation of the zero-shot image classification capability of InternVL-C. As depicted in Table 6(a), InternVL-C attains leading performance on various ImageNet variants [^38] [^61] [^60] [^119] [^141] and ObjectNet [^8]. Compared to EVA-02-CLIP-E+ [^130], it exhibits stronger robustness to distribution shift, manifesting in a more consistent accuracy across ImageNet variants. Additionally, as shown in Table 6(b), our model showcases robust multilingual capabilities, outperforming competing models [^26] [^67] [^162] [^16] on the multilingual ImageNet-1K benchmark.

Zero-Shot Video Classification. Following previous methods [^117] [^130] [^152], we report the top-1 accuracy and the mean of top-1 and top-5 accuracy on Kinetics-400/600/700 [^17] [^18] [^19]. As shown in Table 8, when sampling only a single center frame in each video, our method achieves an average accuracy of 76.1%, 75.5%, and 67.5% on the three datasets, surpassing EVA-02-CLIP-E+ [^130] by +6.3, +6.2, and +4.1 points, respectively. Additionally, when uniformly sampling 8 frames in each video, we obtain at least 3.3 points of improvement compared to the single-frame setting, outperforming ViCLIP [^152] trained using web-scale video data. In summary, InternVL-C exhibits remarkable generalization capabilities in video classification.

Zero-Shot Image-Text Retrieval. InternVL exhibits a powerful multilingual image-text retrieval capability. In Table 7, we evaluate these capabilities in English using the Flickr30K [^116] and COCO [^22] datasets, as well as in Chinese using the Flickr30K-CN [^77] and COCO-CN [^84]. Additionally, we leverage the XTD dataset [^1] to evaluate the multilingual image-text retrieval capability across 8 languages (see supplementary materials). In summary, InternVL-C achieves state-of-the-art performance across most retrieval metrics, and with the second stage of pre-training, InternVL-G further enhances zero-shot image-text retrieval performance. These improvements in retrieval tasks suggest a more effective alignment between visual and linguistic features, through additional image encoding using the language middleware–QLLaMA.

Zero-Shot Image Captioning. Benefiting from vision-language generative training on a vast collection of high-quality image-text pairs, our QLLaMA possesses promising capability in zero-shot image captioning. As shown in Table 10, QLLaMA surpasses other models in zero-shot performance on the COCO Karpathy test set [^22]. It also achieves comparable results to current state-of-the-art models on both the Flickr30K Karpathy test [^116] and the NoCaps val set [^2]. When InternVL is linked with an LLM (*e.g*., Vicuna-7B/13B [^184]) and subjected to SFT, a notable enhancement in zero-shot performance is observed for both Flickr30K and NoCaps, as shown in Table 9.

### 4.4 Multi-Modal Dialogue Benchmarks

Beyond the traditional multi-modal tasks, the emergence of ChatGPT [^110] has led to a growing focus on evaluating the performance of multi-modal models in real usage scenarios, specifically within the realm of multi-modal dialogue. We conducted testing of InternVL-Chat models on two prominent multi-modal dialogue benchmarks, including MME [^50] and POPE [^86]. MME is a comprehensive benchmark that includes 14 sub-tasks focusing on the model’s perception and cognition capabilities. POPE is a popular dataset used to evaluate object hallucination. As shown in Table 9, it clearly demonstrates that our models exhibit superior performance compared with previous methods, under the condition of fair trainable parameter counts.

### 4.5 Ablation Study

Hyperparameters of InternViT-6B. As discussed in Section 3.2, we explored variations in model depth {32, 48, 64, 80}, head dimension {64, 128}, and MLP ratio {4, 8}, resulting in 16 distinct models. In selecting the optimal model, we initially narrowed down our focus to 6 models, chosen based on their throughput, as listed in Table 11. These models underwent further evaluation using contrastive learning on a 100M subset of LAION-en [^120] over 10K iterations. For the experimental setup, the primary difference was the use of a randomly initialized text encoder from CLIP-L [^117], in order to speed up the training. For the sake of accuracy, inference speed, and training stability, we ultimately chose variant 3 as the final InternViT-6B.

| name | width | depth | MLP | #heads | #param | FLOPs | throughput | zs IN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| variant 1 | 3968 | 32 | 15872 | 62 | 6051M | 1571G | 35.5 / 66.0 | 65.8 |
| variant 2 | 3200 | 48 | 12800 | 50 | 5903M | 1536G | 28.1 / 64.9 | 66.1 |
| variant 3 | 3200 | 48 | 12800 | 25 | 5903M | 1536G | 28.0 / 64.6 | 66.2 |
| variant 4 | 2496 | 48 | 19968 | 39 | 5985M | 1553G | 28.3 / 65.3 | 65.9 |
| variant 5 | 2816 | 64 | 11264 | 44 | 6095M | 1589G | 21.6 / 61.4 | 66.2 |
| variant 6 | 2496 | 80 | 9984 | 39 | 5985M | 1564G | 16.9 / 60.1 | 66.2 |

Table 11: Comparison of hyperparameters in InternViT-6B. The throughput (img/s) and GFLOPs are measured at 224 $\times$ 224 input resolution, with a batch size of 1 or 128 on a single A100 GPU. Flash Attention [^35] and bf16 precision are used during testing. “zs IN” denotes the zero-shot top-1 accuracy on the ImageNet-1K validation set [^38]. The final selected model is marked in gray.

<table><tbody><tr><td>visual</td><td>glue</td><td rowspan="2">LLM</td><td rowspan="2">dataset</td><td>dialogue</td><td>caption</td><td colspan="3">visual question answering</td></tr><tr><td>encoder</td><td>layer</td><td>MME</td><td>NoCaps</td><td>OKVQA</td><td>VizWiz <sub>val</sub></td><td>GQA</td></tr><tr><td>EVA-E</td><td>MLP</td><td>V-7B</td><td>665K <sup><a href="#fn:91">91</a></sup></td><td>970.5</td><td>75.1</td><td>40.1</td><td>25.5</td><td>41.3</td></tr><tr><td>IViT-6B</td><td>MLP</td><td>V-7B</td><td>665K <sup><a href="#fn:91">91</a></sup></td><td>1022.3</td><td>80.8</td><td>42.9</td><td>28.3</td><td>45.8</td></tr><tr><td>IViT-6B</td><td>QLLaMA</td><td>V-7B</td><td>665K <sup><a href="#fn:91">91</a></sup></td><td>1227.5</td><td>94.5</td><td>51.0</td><td>38.4</td><td>57.4</td></tr><tr><td>IViT-6B</td><td>QLLaMA</td><td>V-7B</td><td>Ours</td><td>1298.5</td><td>120.5</td><td>51.8</td><td>44.9</td><td>57.7</td></tr><tr><td>IViT-6B</td><td>QLLaMA</td><td>V-13B</td><td>Ours</td><td>1317.2</td><td>123.1</td><td>55.5</td><td>55.7</td><td>59.5</td></tr></tbody></table>

Table 12: Ablation studies of using InternVL to build multi-modal dialogue system. V-7B and V-13B denote Vicuna-7B/13B [^184], respectively. “IViT-6B” represents our InternViT-6B.

Consistency of Feature Representation. In this study, we validate the consistency of the feature representation of InternVL with off-the-shelf LLMs. We adopt a minimalist setting, *i.e*. conducting a single-stage SFT using only the LLaVA-Mix-665K [^85] dataset. Moreover, only the MLP layers are trainable, thereby confirming the inherent alignment level among features from various vision foundation models and LLMs. The results are shown in Table 12. We observed that compared to EVA-E [^130], our InternViT-6B achieves better performance under this simple setup. Additionally, it is noteworthy that performance across all three tasks saw significant improvement when using QLLaMA as the “glue layer”. These significant improvements clearly delineate that *the feature representation of InternVL is more consistent with the off-the-shelf LLM.*

## 5 Conclusion

In this paper, we present InternVL, a large-scale vision-language foundation model that scales up the vision foundation model to 6 billion parameters and is aligned for generic visual-linguistic tasks. Specifically, we design a large-scale vision foundation model InternViT-6B, progressively align it with an LLM-initialized language middleware QLLaMA, and leverage web-scale image-text data from various sources for efficient training. It bridges the gap between vision foundation models and LLMs, and demonstrates proficiency in a wide range of generic visual-linguistic tasks, such as image/video classification, image/video-text retrieval, image captioning, visual question answering, and multi-modal dialogue. We hope this work could contribute to the development of the VLLM community.

## Acknowledgement

We thank Shenglong Zhang, Beitong Zhou, Xinyue Zhang, Dongxing Shi, Weigao Sun, Xingcheng Zhang, and Zhifeng Yue for their contributions to the optimization of the training framework. We thank Zhenhang Huang for his assistance in data preparation.

## Appendix A Supplementary Materials

### A.1 More Experiments

Zero-Shot Image Classification on 20 Datasets. In this section, we expand our examination to showcase the effectiveness and robustness of InternVL in 20 different zero-shot image classification benchmarks. As indicated in Table 16, InternVL registers an average performance of 78.1% across all 20 benchmarks. This performance notably exceeds that of the previously leading method, EVA-02-CLIP-E+ [^47], by a margin of 1.0 points. This underscores that, beyond ImageNet [^38] and its variants, InternVL possesses robust generalization capabilities across a variety of different domains in zero-shot image classification.

Zero-Shot Image-Text Retrieval on XTD. Table 13 reports the results of InternVL on the multilingual image-text retrieval dataset XTD [^1], spanning eight languages. As can be seen, InternVL-C achieves an average recall@10 score of 95.1% across these languages. The second stage model, InternVL-G, further improves retrieval performance. It attains the highest scores in each individual language and establishes a new record for average performance at 96.6%.

Zero-Shot Video Retrieval. In Table 14, we present our results of zero-shot video-text retrieval on the MSR-VTT dataset [^161] using our InternVL models, *i.e*. InternVL-C and InternVL-G. In the 1-frame setting, we select a single central frame from each video. In the 8-frame setting, we uniformly extract 8 frames from each video, treat them as independent images for encoding, and then average the embeddings. The results showcase consistent improvement across various metrics such as R@1, R@5, R@10, and the average score. Importantly, both models exhibit promising outcomes in single-frame and multi-frame configurations, with InternVL-G achieving slightly higher performance than InternVL-C, especially in the multi-frame setting. These results underscore the effectiveness of QLLaMA in harmonizing visual and linguistic features.

Fine-tuned Image-Text Retrieval. In Table 15, we report the fine-tuned image-text retrieval results of InternVL, on both the English and Chinese versions of the Flickr30K dataset [^116] [^77]. The specific hyperparameters for fine-tuning are shown in Table 21. As can be seen, our models obtain competitive performance, with InternVL-G-FT marginally surpassing InternVL-C-FT in both datasets. Notably, in the highly challenging Flickr30K-CN, both models show a promising ability to handle cross-lingual retrieval tasks. These results demonstrate the effectiveness of our language middleware, especially in the retrieval tasks.

Tiny LVLM. Tiny LVLM [^123] is an ability-level benchmark for evaluating the performance of multimodal dialogue models. It provides a systematic assessment of five categories of multimodal capabilities, including visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, and object hallucination. We report our results on Tiny LVLM in Table 17.

### A.2 More Ablation Studies

| method | EN | ES | FR | ZH | IT | KO | RU | JP | avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mUSE m3 [^164] | 85.3 | 78.9 | 78.9 | 76.7 | 73.6 | 67.8 | 76.1 | 70.7 | 76.0 |
| M-CLIP [^16] | 92.4 | 91.0 | 90.0 | 89.7 | 91.1 | 85.2 | 85.8 | 81.9 | 88.4 |
| MURAL [^69] | $-$ | 92.9 | $-$ | 89.7 | 91.8 | 88.1 | 87.2 | $-$ | $-$ |
| AltCLIP [^26] | 95.4 | 94.1 | 92.9 | 95.1 | 94.2 | 94.4 | 91.8 | 91.7 | 93.7 |
| OpenCLIP-XLM-R-B [^67] | 95.8 | 94.4 | 92.5 | 91.8 | 94.4 | 86.3 | 89.9 | 90.7 | 92.0 |
| OpenCLIP-XLM-R-H [^67] | 97.3 | 96.1 | 94.5 | 94.7 | 96.0 | 90.2 | 93.9 | 94.0 | 94.6 |
| InternVL-C (ours) | 97.3 | 95.7 | 95.1 | 95.6 | 96.0 | 92.2 | 93.3 | 95.5 | 95.1 |
| InternVL-G (ours) | 98.6 | 97.7 | 96.5 | 96.7 | 96.9 | 95.1 | 94.8 | 96.1 | 96.6 |

Table 13: Comparison of zero-shot multilingual image-text retrieval performance on the XTD dataset. Multiple languages include English (EN), Spanish (ES), French (FR), Chinese (ZH), Italian (IT), Korean (KO), Russian (RU), and Japanese (JP). We follow M-CLIP [^16] to report the recall@10 on Image-to-Text.

<table><tbody><tr><td></td><td></td><td colspan="6">MSR-VTT (1K test set) <sup><a href="#fn:161">161</a></sup></td><td></td></tr><tr><td></td><td></td><td colspan="3">Video <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Text</td><td colspan="3">Text <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Video</td><td></td></tr><tr><td>method</td><td>#F</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>avg.</td></tr><tr><td>OpenAI CLIP-L <sup><a href="#fn:117">117</a></sup></td><td>1</td><td>27.8</td><td>49.4</td><td>58.0</td><td>29.0</td><td>50.5</td><td>59.2</td><td>45.7</td></tr><tr><td>InternVL-C (ours)</td><td>1</td><td>35.3</td><td>56.6</td><td>66.6</td><td>37.5</td><td>60.9</td><td>70.9</td><td>54.6</td></tr><tr><td>InternVL-G (ours)</td><td>1</td><td>36.6</td><td>58.3</td><td>67.7</td><td>39.1</td><td>61.7</td><td>70.7</td><td>55.7</td></tr><tr><td>OpenAI CLIP-L <sup><a href="#fn:117">117</a></sup></td><td>8</td><td>26.6</td><td>50.8</td><td>61.8</td><td>30.7</td><td>54.4</td><td>64.0</td><td>48.1</td></tr><tr><td>Florence <sup><a href="#fn:171">171</a></sup></td><td>8</td><td>–</td><td>–</td><td>–</td><td>37.6</td><td>63.8</td><td>72.6</td><td>–</td></tr><tr><td>InternVideo <sup>†</sup> <sup><a href="#fn:151">151</a></sup></td><td>8</td><td>39.6</td><td>–</td><td>–</td><td>40.7</td><td>–</td><td>–</td><td>–</td></tr><tr><td>UMT-L <sup>†</sup> <sup><a href="#fn:83">83</a></sup></td><td>8</td><td>38.6</td><td>59.8</td><td>69.6</td><td>42.6</td><td>64.4</td><td>73.1</td><td>58.0</td></tr><tr><td>LanguageBind <sup>†</sup> <sup><a href="#fn:186">186</a></sup></td><td>8</td><td>40.9</td><td>66.4</td><td>75.7</td><td>44.8</td><td>70.0</td><td>78.7</td><td>62.8</td></tr><tr><td>InternVL-C (ours)</td><td>8</td><td>40.2</td><td>63.1</td><td>74.1</td><td>44.7</td><td>68.2</td><td>78.4</td><td>61.5</td></tr><tr><td>InternVL-G (ours)</td><td>8</td><td>42.4</td><td>65.9</td><td>75.4</td><td>46.3</td><td>70.5</td><td>79.6</td><td>63.4</td></tr></tbody></table>

Table 14: Comparison of zero-shot video-text retrieval performance on MSR-VTT. “#F” denotes the number of frames. <sup>†</sup> These models are trained with temporal attention layers.

<table><tbody><tr><td></td><td colspan="6">Flickr30K (English, 1K test set) <sup><a href="#fn:116">116</a></sup></td><td></td></tr><tr><td></td><td colspan="3">Image <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Text</td><td colspan="3">Text <math><semantics><mo>→</mo> <ci>→</ci> <annotation>\rightarrow</annotation></semantics></math> Image</td><td></td></tr><tr><td>method</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>avg.</td></tr><tr><td>ALIGN <sup><a href="#fn:70">70</a></sup></td><td>95.3</td><td>99.8</td><td>100.0</td><td>84.9</td><td>97.4</td><td>98.6</td><td>96.0</td></tr><tr><td>FILIP <sup><a href="#fn:167">167</a></sup></td><td>96.6</td><td>100.0</td><td>100.0</td><td>87.1</td><td>97.7</td><td>99.1</td><td>96.8</td></tr><tr><td>Florence <sup><a href="#fn:171">171</a></sup></td><td>97.2</td><td>99.9</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td>87.9</td><td>98.1</td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td><td><math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math></td></tr><tr><td>BLIP <sup><a href="#fn:80">80</a></sup></td><td>97.4</td><td>99.8</td><td>99.9</td><td>87.6</td><td>97.7</td><td>99.0</td><td>96.9</td></tr><tr><td>OmniVL <sup><a href="#fn:142">142</a></sup></td><td>97.3</td><td>99.9</td><td>100.0</td><td>87.9</td><td>97.8</td><td>99.1</td><td>97.0</td></tr><tr><td>BEiT-3 <sup><a href="#fn:146">146</a></sup></td><td>97.5</td><td>99.9</td><td>100.0</td><td>89.1</td><td>98.6</td><td>99.3</td><td>97.4</td></tr><tr><td>ONE-PEACE <sup><a href="#fn:143">143</a></sup></td><td>97.6</td><td>100.0</td><td>100.0</td><td>89.6</td><td>98.0</td><td>99.1</td><td>97.4</td></tr><tr><td>InternVL-C-FT (ours)</td><td>97.2</td><td>100.0</td><td>100.0</td><td>88.5</td><td>98.4</td><td>99.2</td><td>97.2</td></tr><tr><td>InternVL-G-FT (ours)</td><td>97.9</td><td>100.0</td><td>100.0</td><td>89.6</td><td>98.6</td><td>99.2</td><td>97.6</td></tr><tr><td>method</td><td colspan="6">Flickr30K-CN (Chinese, 1K test set) <sup><a href="#fn:77">77</a></sup></td><td>avg.</td></tr><tr><td>Wukong-ViT-L <sup><a href="#fn:55">55</a></sup></td><td>92.7</td><td>99.1</td><td>99.6</td><td>77.4</td><td>94.5</td><td>97.0</td><td>93.4</td></tr><tr><td>CN-CLIP-ViT-H <sup><a href="#fn:162">162</a></sup></td><td>95.3</td><td>99.7</td><td>100.0</td><td>83.8</td><td>96.9</td><td>98.6</td><td>95.7</td></tr><tr><td>R2D2-ViT-L <sup><a href="#fn:159">159</a></sup></td><td>95.6</td><td>99.8</td><td>100.0</td><td>84.4</td><td>96.7</td><td>98.4</td><td>95.8</td></tr><tr><td>InternVL-C-FT (ours)</td><td>96.5</td><td>99.9</td><td>100.0</td><td>85.2</td><td>97.0</td><td>98.5</td><td>96.2</td></tr><tr><td>InternVL-G-FT (ours)</td><td>96.9</td><td>99.9</td><td>100.0</td><td>85.9</td><td>97.1</td><td>98.7</td><td>96.4</td></tr></tbody></table>

Table 15: Comparison of fine-tuned image-text retrieval performance. We evaluate English and Chinese image-text retrieval using Flickr30K [^116] and Flickr30K-CN [^77], with separate fine-tuning for each to prevent data leakage.

| method | CIFAR-10 [^74] | CIFAR-100 [^74] | MNIST [^78] | Caltech-101 [^49] | SUN397 [^157] | FGVC Aircraft [^101] | Country-211 [^117] | Stanford Cars [^72] | Birdsnap [^9] | DTD [^28] | Eurosat [^59] | FER2013 [^52] | Flowers-102 [^109] | Food-101 [^13] | GTSRB [^129] | Pets [^113] | Rendered SST2 [^117] | Resisc45 [^27] | STL10 [^30] | VOC2007 [^45] | avg. top-1 acc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OpenAI CLIP-L+ [^117] | 94.9 | 74.4 | 79.0 | 87.2 | 68.7 | 33.4 | 34.5 | 79.3 | 41.0 | 56.0 | 61.5 | 49.1 | 78.6 | 93.9 | 52.4 | 93.8 | 70.7 | 65.4 | 99.4 | 78.1 | 69.6 |
| EVA-01-CLIP-g [^130] | 98.3 | 88.7 | 62.3 | 87.7 | 74.2 | 32.4 | 28.6 | 91.7 | 50.0 | 61.3 | 73.6 | 52.2 | 74.5 | 93.5 | 49.1 | 94.2 | 58.4 | 70.3 | 98.9 | 83.2 | 71.2 |
| OpenCLIP-g [^67] | 98.2 | 84.7 | 71.9 | 88.1 | 74.1 | 44.6 | 30.9 | 94.0 | 51.0 | 68.7 | 64.7 | 55.8 | 81.0 | 92.4 | 49.7 | 93.9 | 56.7 | 69.6 | 98.9 | 81.6 | 72.5 |
| OpenCLIP-H [^67] | 97.4 | 84.7 | 72.9 | 85.0 | 75.2 | 42.8 | 30.0 | 93.5 | 52.9 | 67.8 | 72.7 | 52.0 | 80.1 | 92.7 | 58.4 | 94.5 | 64.3 | 70.5 | 98.5 | 77.7 | 73.2 |
| EVA-02-CLIP-L+ [^130] | 98.9 | 89.8 | 64.3 | 89.5 | 74.8 | 37.5 | 33.6 | 91.6 | 45.8 | 64.5 | 71.4 | 51.0 | 77.2 | 94.2 | 57.6 | 94.2 | 64.6 | 69.8 | 99.7 | 82.7 | 72.6 |
| EVA-01-CLIP-g+ [^130] | 99.1 | 90.1 | 71.8 | 88.1 | 74.3 | 39.4 | 30.8 | 90.7 | 52.6 | 67.3 | 73.2 | 56.0 | 79.7 | 93.7 | 66.5 | 94.8 | 58.6 | 71.4 | 99.5 | 82.9 | 74.0 |
| OpenCLIP-G [^67] | 98.2 | 87.5 | 71.6 | 86.4 | 74.5 | 49.7 | 33.8 | 94.5 | 54.5 | 69.0 | 70.0 | 59.5 | 81.5 | 93.1 | 62.5 | 95.2 | 65.2 | 72.6 | 98.5 | 80.7 | 74.9 |
| EVA-02-CLIP-E [^130] | 99.3 | 92.5 | 76.7 | 89.0 | 76.5 | 47.9 | 34.7 | 94.4 | 56.3 | 68.2 | 77.6 | 55.1 | 82.5 | 95.2 | 67.1 | 95.6 | 61.1 | 73.5 | 99.2 | 83.0 | 76.3 |
| EVA-02-CLIP-E+ [^130] | 99.3 | 93.1 | 74.7 | 90.5 | 75.1 | 54.1 | 35.7 | 94.6 | 58.1 | 68.2 | 75.8 | 58.6 | 84.5 | 94.9 | 67.7 | 95.8 | 61.4 | 75.6 | 99.2 | 85.6 | 77.1 |
| InternVL-C (ours) | 99.4 | 93.2 | 80.6 | 89.5 | 76.0 | 52.7 | 34.1 | 94.2 | 72.0 | 70.7 | 79.4 | 56.2 | 86.1 | 95.3 | 65.5 | 96.0 | 67.9 | 74.2 | 99.5 | 80.0 | 78.1 |

Table 16: Comparison of zero-shot image classification performance on 20 other datasets. These results indicate that, in addition to ImageNet [^38], InternVL also possesses good generalization capabilities in zero-shot image classification across various domains.

| method | LLM | VR | VP | VKA | VC | OH | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniGPT-4 [^187] | Vicuna-7B | 37.6 | 37.8 | 17.6 | 49.0 | 50.7 | 192.6 |
| LLaVA [^92] | Vicuna-7B | 41.6 | 38.3 | 18.7 | 49.4 | 49.0 | 197.0 |
| VisualGLM [^44] | ChatGLM-6B | 37.3 | 36.3 | 46.9 | 37.6 | 54.0 | 211.9 |
| Otter [^79] | Otter-9B | 41.6 | 37.0 | 15.1 | 52.4 | 74.0 | 216.4 |
| LLaMA-Adapter-V2 [^51] | LLaMA-7B | 43.5 | 46.8 | 22.3 | 56.0 | 60.7 | 229.2 |
| Lynx [^172] | Vicuna-7B | 52.2 | 65.8 | 17.6 | 57.4 | 86.3 | 279.2 |
| BLIP-2 [^81] | FlanT5xl | 44.9 | 49.0 | 64.1 | 44.0 | 82.7 | 284.7 |
| InstructBLIP [^34] | Vicuna-7B | 46.7 | 48.0 | 61.7 | 59.2 | 85.0 | 300.6 |
| LLaVA-1.5 [^91] | Vicuna-7B | 55.6 | 49.0 | 57.0 | 57.2 | 88.3 | 307.2 |
| Qwen-VL-Chat [^5] | Qwen-7B | 62.4 | 54.5 | 55.1 | 54.8 | 90.0 | 316.8 |
| Bard [^53] | Bard | 64.2 | 57.0 | 68.1 | 59.6 | 70.7 | 319.6 |
| InternLM-XComposer [^177] | InternLM-7B | 55.8 | 53.8 | 64.1 | 61.8 | 87.0 | 322.5 |
| InternVL-Chat (ours) | Vicuna-13B | 56.4 | 52.3 | 68.0 | 62.0 | 89.0 | 327.6 |

Table 17: Evaluation of Tiny LVLM test set. Here we report five categories of multimodal capabilities, including visual reasoning (VR), visual perception (VP), visual knowledge acquisition (VKA), visual commonsense (VC), and object hallucination (OH).

Compatibility with Other LLM. In this experiment, we test the compatibility of InternVL with LLMs other than Vicuna [^184]. The experimental setup used here is the same as in Table 9 of the main paper. As shown in Table 18, InternLM-7B [^135] achieves slightly better performance than Vicuna-7B [^184]. This indicates that our InternVL exhibits promising compatibility with various LLMs.

Efficiency Analysis. In this study, we analyze the computational efficiency of InternVL in encoding image-text pairs. The entire encoding process consists of two parts: image encoding and text encoding. The analysis covered two models (InternVL-C and InternVL-G) and their performance across three different image sizes (224, 336, and 448). The results are shown in Table 19.

From these results, we find that: (1) As the image size increases, the encoding time also significantly increases, leading directly to a decrease in frame rate; (2) InternVL-G slightly increased the encoding time due to the introduction of QLLaMA for secondary image encoding, but it still maintains a reasonable frame rate across all image sizes; (3) Even though we scale up the text encoder, the additional cost of text encoding is not significant, as the main time expenditure lies in image encoding. In summary, when choosing between InternVL-C and InternVL-G, one should weigh the trade-off between computational efficiency and potential performance improvements based on specific requirements. Additionally, these results were measured using PyTorch with Flash Attention [^35] and bf16 precision, and there is still considerable room for optimization, such as using model quantization and TensorRT.

<table><tbody><tr><td>visual</td><td>glue</td><td></td><td colspan="4">visual question answering</td><td colspan="2">dialogue</td></tr><tr><td>encoder</td><td>layer</td><td>LLM</td><td>VQA <math><semantics><msup><mtext>v2</mtext></msup> <apply><ci><mtext>v2</mtext></ci></apply> <annotation>{}^{\text{v2}}</annotation></semantics></math></td><td>GQA</td><td>VizWiz</td><td>VQA <math><semantics><msup><mtext>T</mtext></msup> <apply><ci><mtext>T</mtext></ci></apply> <annotation>{}^{\text{T}}</annotation></semantics></math></td><td>MME</td><td>POPE</td></tr><tr><td>IViT-6B</td><td>MLP</td><td>Vicuna-7B</td><td>79.3</td><td>62.9</td><td>52.5</td><td>57.0</td><td>1525.1</td><td>86.4</td></tr><tr><td>IViT-6B</td><td>MLP</td><td>InternLM-7B</td><td>79.7</td><td>63.2</td><td>53.1</td><td>58.0</td><td>1532.8</td><td>86.4</td></tr></tbody></table>

Table 18: Compatibility with other LLM. Here we use InternLM [^135] as an example to verify the compatibility of InternVL with LLMs other than Vicuna [^184]. The experimental settings used here are the same as in Table 9 of the main paper.

<table><tbody><tr><td></td><td>image</td><td colspan="2">encode image (ms)</td><td>encode text (ms)</td><td>total</td><td></td></tr><tr><td>method</td><td>size</td><td>InternViT-6B</td><td>QLLaMA</td><td>QLLaMA</td><td>time</td><td>FPS</td></tr><tr><td>InternVL-C</td><td>224</td><td>15.5</td><td>–</td><td>4.9</td><td>20.4</td><td>48.9</td></tr><tr><td>InternVL-C</td><td>336</td><td>35.2</td><td>–</td><td>4.9</td><td>40.1</td><td>24.9</td></tr><tr><td>InternVL-C</td><td>448</td><td>66.9</td><td>–</td><td>4.9</td><td>71.8</td><td>13.9</td></tr><tr><td>InternVL-G</td><td>224</td><td>15.5</td><td>8.2</td><td>4.9</td><td>28.6</td><td>35.0</td></tr><tr><td>InternVL-G</td><td>336</td><td>35.2</td><td>10.3</td><td>4.9</td><td>50.4</td><td>19.8</td></tr><tr><td>InternVL-G</td><td>448</td><td>66.9</td><td>12.8</td><td>4.9</td><td>84.6</td><td>11.8</td></tr></tbody></table>

Table 19: Efficiency analysis of InternVL for encoding image-text pairs. The total time to encode an image-text pair includes both the image encoding part and the text encoding part. We measure the time cost with a batch size of 128 on a single A100 GPU. Flash Attention [^35] and bf16 precision are used during testing.

### A.3 Detailed Training Settings

Settings of Stage 1. As shown in Table 20, in this stage, the image encoder InternViT-6B is randomly initialized using the BEiT’s initialization method [^7], and the text encoder LLaMA-7B is initialized with the pre-trained weights from [^32], a multilingual LLaMA-7B. All parameters are fully trainable. We employ the AdamW optimizer [^98] with $\beta_{1}=0.9$, $\beta_{2}=0.95$, weight decay at 0.1, and a cosine learning rate schedule starting at 1e-3 and 1e-4 for the image and text encoders, respectively. We adopt a uniform drop path rate of 0.2. The training involves a total batch size of 164K across 640 A100 GPUs, extending over 175K iterations to process about 28.7 billion samples. To enhance efficiency, we initially train at a 196 $\times$ 196 resolution, masking 50% of image tokens [^87], and later switch to 224 $\times$ 224 resolution without masking for the final 0.5 billion samples.

Settings of Stage 2. In this stage, InternViT-6B and QLLaMA inherit their weights from the first stage, while the learnable queries and cross-attention layers in QLLaMA are randomly initialized. Benefiting from the powerful encoding capabilities learned in the first stage, we keep both InternViT-6B and QLLaMA frozen and only train the newly added parameters. The input images are processed at a resolution of 224 $\times$ 224. For optimization, the AdamW optimizer [^98] is employed with $\beta_{1}=0.9$, $\beta_{2}=0.98$, weight decay set at 0.05, and a total batch size of 20K. The training extends over 80K steps across 160 A100 GPUs, inclusive of 2K warm-up steps, and is governed by a cosine learning rate schedule with a peak learning rate of 5e-5. More detailed training settings are listed in Table 20.

| config | stage 1 | stage 2 |
| --- | --- | --- |
| image enc. weight init. | random init. [^7] | from stage 1 |
| text enc. weight init. | from [^32] | from stage 1 |
| image enc. peak learning rate | 1e-3 | frozen |
| text enc. peak learning rate | 1e-4 | frozen |
| cross attn peak learning rate | – | 5e-5 |
| learning rate schedule | cosine decay | cosine decay |
| optimizer | AdamW [^98] | AdamW [^98] |
| optimizer hyper-parameters | $\beta_{1}$, $\beta_{2}$ = 0.9, 0.95 | $\beta_{1}$, $\beta_{2}$ = 0.9, 0.98 |
| weight decay | 0.1 | 0.05 |
| input resolution | 196 <sup>2</sup> $\rightarrow$ 224 <sup>2</sup> | 224 <sup>2</sup> |
| patch size | 14 | 14 |
| total batch size | 164K | 20K |
| warm-up iterations | 5K | 2K |
| total iterations | 175K | 80K |
| samples seen | 28.7B | 1.6B |
| drop path rate [^63] | uniform (0.2) | 0.0 |
| data augmentation | random resized crop | random resized crop |
| numerical precision | DeepSpeed bf16 [^118] | DeepSpeed bf16 [^118] |
| trainable / total parameters | 13B / 13B | 1B / 14B |
| GPUs for training | 640 $\times$ A100 (80G) | 160 $\times$ A100 (80G) |

Table 20: Training settings of InternVL’s stage 1 and stage 2. “196 <sup>2</sup> $\rightarrow$ 224 <sup>2</sup> ” means we initially train at a 196 $\times$ 196 resolution, and later switch to 224 $\times$ 224 resolution for the final 0.5 billion samples, for higher training efficiency.

| config | retrieval fine-tuning |
| --- | --- |
| image-text data | Flickr30K [^116] / Flickr30K-CN [^77] |
| peak learning rate | 1e-6 |
| layer-wise lr decay rate | InternViT-6B (0.9), QLLaMA (0.9) |
| learning rate schedule | cosine decay |
| optimizer | AdamW [^98] |
| optimizer hyper-parameters | $\beta_{1}$, $\beta_{2}$ = 0.9, 0.999 |
| weight decay | 0.05 |
| input resolution | 364 <sup>2</sup> |
| patch size | 14 |
| total batch size | 1024 |
| warm-up iterations | 100 |
| training epochs | 10 |
| drop path rate [^63] | 0.3 |
| data augmentation | random resized crop & flip |
| numerical precision | DeepSpeed bf16 [^118] |
| trainable / total parameters | 14B / 14B |
| GPUs for training | 32 $\times$ A100 (80G) |

Table 21: Training settings of retrieval fine-tuning. We fine-tune InternVL on Flickr30K and Flickr30K-CN separately.

Settings of Stage 3. At this stage, we have two different configurations. One is to use InternViT-6B separately, as shown in Figure 4 (c). The other is to use the entire InternVL model simultaneously, as shown in Figure 4 (d).

(1) InternVL-Chat (w/o QLLaMA): For this setup, we follow the training recipes of LLaVA-1.5 [^91]. We use the same hyperparameters and datasets for supervised fine-tuning, *i.e*. we first train the MLP layers with the LGS-558K [^92] dataset, and then train the LLM with the LLaVA-Mix-665K [^91] dataset, both for one epoch.

(2) InternVL-Chat (w/ QLLaMA): For this more advanced setup, we also conducted the training in two steps. We first train the MLP layers with our custom SFT dataset and then fine-tune the LLM with it. Due to the expansion of the dataset, we increased the batch size to 512.

Settings of Retrieval Fine-tuning. In this experiment, all parameters of InternVL are set to be trainable. We conduct separate fine-tuning on the Flickr30K [^116] and Flickr30K-CN [^77]. Following common practice [^81], a 364 $\times$ 364 resolution is adopted for fine-tuning. To avoid over-fitting, we apply a layer-wise learning rate decay of 0.9 to both InternViT-6B and QLLaMA, along with a drop path rate of 0.3 for InternViT-6B. The AdamW optimizer [^98] is utilized, with a total batch size of 1024, for fine-tuning the InternVL model across 10 epochs. For more detailed training settings, please refer to Table 21.

| config | ImageNet linear probing |
| --- | --- |
| peak learning rate | 0.2 |
| learning rate schedule | cosine decay |
| optimizer | SGD |
| optimizer momentum | 0.9 |
| weight decay | 0.0 |
| input resolution | 224 <sup>2</sup> |
| patch size | 14 |
| total batch size | 1024 |
| warm-up epochs | 1 |
| training epochs | 10 |
| data augmentation | random resized crop & flip |
| GPUs for training | 8 $\times$ A100 (80G) |

Table 22: Training settings of ImageNet linear probing.

| config | linear probing / head tuning / full tuning |
| --- | --- |
| peak learning rate | 4e-5 |
| layer-wise lr decay rate | – / – / 0.95 |
| learning rate schedule | polynomial decay |
| optimizer | AdamW [^98] |
| optimizer hyper-parameters | $\beta_{1}$, $\beta_{2}$ = 0.9, 0.999 |
| weight decay | 0.0 / 0.05 / 0.05 |
| input resolution | 504 <sup>2</sup> |
| patch size | 14 |
| total batch size | 16 |
| warm-up iterations | 1.5K |
| total iterations | 80K |
| drop path rate [^63] | 0.0 / 0.0 / 0.4 |
| data augmentation | default augmentation in MMSeg [^31] |
| numerical precision | DeepSpeed bf16 [^118] |
| GPUs for training | 8 $\times$ A100 (80G) |

Table 23: Training settings of ADE20K semantic segmentation. We list the hyperparameters for three different configurations, including linear probing, head tuning, and full-parameter tuning.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2312.14238/assets/x5.png)

Figure 5: Panoramic overview of the datasets used in InternVL’s stage 1 and stage 2. During the training of stage 1 and stage 2, we utilize web-scale image-text data from a variety of sources to train our InternVL model, as shown in (a). To assess InternVL’s capabilities in handling generic visual-linguistic tasks, we conducted extensive validations across a range of tasks and datasets, including (b) image classification, (c) video classification, (d) image-text retrieval, (e) video-text retrieval, (f) image captioning, and (g) semantic segmentation.

Settings of ImageNet Linear Probing. We follow the common practices of linear probing in previous methods [^58] [^111] [^37]. Specifically, we employ an additional BatchNorm [^68] to normalize the pre-trained backbone features during training. Besides, we concatenate the average-pooled patch token features with the class token. The linear head is trained using the SGD optimizer for 10 epochs on ImageNet-1K [^38], with a total batch size of 1024, a peak learning rate of 0.2, 1 epoch warm-up, and no weight decay. Data augmentation involves random-resized-crop and flip. For more training details, please see Table 22.

Settings of ADE20K Semantic Segmentation. In Table 23, we have listed the hyperparameters for three different configurations in ADE20K semantic segmentation, including linear probing, head tuning, and full-parameter tuning.

### A.4 Data Preparation for Pre-training

Training Data for Stage 1 & Stage 2. During the first and second stages, we employed a vast collection of image-text pair data (see Figure 5 (a)), such as LAION-en [^120], LAION-multi [^120], LAION-COCO [^121], COYO [^14], Wukong [^55], among others [^124] [^20] [^112]. A detailed introduction to these datasets is provided in Table 24.

Training Data Cleaning for Stage 1 & Stage 2. To fully utilize web-scale image-text data, we adopted different data filtering strategies in stage 1 and stage 2.

(1) Stage 1: In the first stage, we applied only minor data filtering, thus retaining the vast majority of the data. We considered six factors: CLIP similarity, watermark probability, unsafe probability, aesthetic score, image resolution, and caption length, to remove extreme data points and avoid disrupting training stability. Additionally, we removed data that was duplicated with ImageNet-1K/22K [^38], Flickr30K [^116], and COCO [^89] to ensure the reliability of our zero-shot evaluations. Due to download failures and the use of our data filtering pipeline, the total amount of data retained in the first stage was 4.98 billion.

(2) Stage 2: In the second stage, we implemented a more stringent data filtering strategy. With generative supervision included, we deleted most of the low-quality data based on the captions, mainly considering the length, completeness, readability, and whether they were gibberish or boilerplate (like menus, error messages, or duplicate text), contained offensive language, placeholder text, or source code. We retained only 1.03 billion entries.

Testing Datasets for Image Classification. We conducted extensive validation on image classification tasks (see Figure 5 (b)), including the linear probing performance of InternViT-6B and the zero-shot performance of InternVL-C. These datasets used are listed in Table 24.

Testing Datasets for Video Classification. As shown in Figure 5 (c), to evaluate the capabilities of video classification, we utilize the following Kinetics datasets: Kinetics 400 [^17], Kinetics 600 [^18], and Kinetics 700 [^19].

Testing Datasets for Image-Text Retrieval. We use five datasets (see Figure 5 (d)) to evaluate InternVL’s zero-shot, multilingual image-text retrieval capabilities. A detailed introduction to these datasets is provided in Table 25.

Testing Dataset for Video-Text Retrieval. As shown in Figure 5 (e), we use the MSR-VTT [^161] dataset to evaluate our InternVL in zero-shot video-text retrieval.

Testing Dataset for Image Captioning. As illustrated in Figure 5 (f), we use three image captioning datasets to test our InternVL model. A detailed introduction to these datasets is provided in Table 26.

Testing Dataset for Semantic Segmentation. We use the ADE20K [^185] dataset to study the pixel-level perceptual capacity of InternViT-6B, as shown in Figure 5 (g). A detailed introduction to this dataset is provided in Table 26.

### A.5 Data Preparation for SFT

Training Data for SFT. In this stage, we collect a wide range of high-quality instruction data. For non-dialogue datasets, we follow the method described in [^91] for conversion. A detailed introduction is provided in Table 27.

Testing Datasets for SFT. We validate the effectiveness of our supervised fine-tuned InternVL-Chat models on three tasks, including image captioning, visual question answering, and multi-modal dialogue. There datasets are listed in Table 28. For most of these datasets, we employ the same response formatting prompt as for LLaVA-1.5 [^91].

<table><tbody><tr><td>dataset</td><td>introduction</td></tr><tr><td colspan="2"><em>Training Data for Stage 1 & Stage 2.</em></td></tr><tr><td>LAION-en <sup><a href="#fn:120">120</a></sup></td><td>LAION-en is a part of the LAION-5B dataset, containing 2.32 billion English-only image-text pairs.</td></tr><tr><td>LAION-multi <sup><a href="#fn:120">120</a></sup></td><td>LAION-multi is another segment of LAION-5B, featuring 2.26 billion image-text pairs across more than 100 languages, and is ideal for multilingual studies.</td></tr><tr><td>Laion-COCO <sup><a href="#fn:121">121</a></sup></td><td>Laion-COCO comprises 663 million synthetic captions for web images, generated using a blend of BLIP-L/14 <sup><a href="#fn:80">80</a></sup> and CLIP models <sup><a href="#fn:117">117</a></sup>.</td></tr><tr><td>COYO <sup><a href="#fn:14">14</a></sup></td><td>COYO-700M is a large-scale dataset that contains 747 million image-text pairs as well as many other meta-attributes to increase the usability to train various models. It follows a similar strategy to previous vision-language datasets, collecting many informative pairs of alt-text and its associated image in HTML documents.</td></tr><tr><td>Wukong <sup><a href="#fn:55">55</a></sup></td><td>Wukong is a large-scale Chinese image-text dataset for benchmarking different multi-modal pre-training methods. It contains 100 million Chinese image-text pairs from the web.</td></tr><tr><td>CC3M <sup><a href="#fn:124">124</a></sup></td><td>This dataset consists of approximately 3 million images, each annotated with a caption.</td></tr><tr><td>CC12M <sup><a href="#fn:20">20</a></sup></td><td>CC12M is a dataset with 12 million image-text pairs. It is larger and covers a much more diverse set of visual concepts than the CC3M <sup><a href="#fn:124">124</a></sup>.</td></tr><tr><td>SBU <sup><a href="#fn:112">112</a></sup></td><td>The SBU Captioned Photo Dataset is a collection of over 1 million images with associated text descriptions extracted from Flicker.</td></tr><tr><td colspan="2"><em>Testing Datasets for Image Classification.</em></td></tr><tr><td>ImageNet-1K <sup><a href="#fn:38">38</a></sup></td><td>A large-scale dataset commonly used in image classification, consisting of over 1 million images across 1K different classes.</td></tr><tr><td>ImageNet-ReaL <sup><a href="#fn:10">10</a></sup></td><td>It contains ImageNet val images augmented with a new set of “re-assessed” labels. These labels are collected using an enhanced protocol, resulting in multi-label and more accurate annotations.</td></tr><tr><td>ImageNet-V2 <sup><a href="#fn:119">119</a></sup></td><td>A dataset created to test the robustness of models trained on ImageNet-1K, containing new test images collected following the original methodology.</td></tr><tr><td>ImageNet-A <sup><a href="#fn:61">61</a></sup></td><td>It consists of real-world, unmodified, and naturally occurring examples that are misclassified by ResNet models <sup><a href="#fn:57">57</a></sup>. It’s designed to highlight the challenges of adversarial examples in natural settings.</td></tr><tr><td>ImageNet-R <sup><a href="#fn:60">60</a></sup></td><td>A set of images labeled with ImageNet labels obtained by collecting art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes. It has renditions of 200 ImageNet classes resulting in 30K images.</td></tr><tr><td>ImageNet-Sketch <sup><a href="#fn:141">141</a></sup></td><td>It consists of 51K images, approximately 50 images for each of the ImageNet classes. It is constructed using Google Image queries with the standard class name followed by “sketch of”.</td></tr><tr><td>ObjectNet <sup><a href="#fn:8">8</a></sup></td><td>ObjectNet is a crowd-sourced test set of 50K images featuring objects in unusual poses and cluttered scenes, designed to challenge recognition performance. It includes controls for rotation, background, and viewpoint, and covers 313 object classes, with 113 overlapping with ImageNet <sup><a href="#fn:38">38</a></sup>.</td></tr><tr><td>Multilingual IN-1K <sup><a href="#fn:76">76</a></sup></td><td>An adaptation of ImageNet-1K supporting multilingual annotations, facilitating research in cross-lingual image classification.</td></tr><tr><td>CIFAR-10/100 <sup><a href="#fn:74">74</a></sup></td><td>It comprises 60K 32 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 32 images in 10 classes (CIFAR-10) or 100 classes (CIFAR-100).</td></tr><tr><td>MNIST <sup><a href="#fn:78">78</a></sup></td><td>A classic dataset containing 70K 28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 28 gray-scale images of handwritten digits.</td></tr><tr><td>Caltech-101 <sup><a href="#fn:49">49</a></sup></td><td>The dataset comprises images of objects from 101 classes and a background clutter class, each labeled with a single object. It contains about 40 to 800 images per class, totaling approximately 9K images.</td></tr><tr><td>SUN397 <sup><a href="#fn:157">157</a></sup></td><td>The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of 397 categories with 109K images.</td></tr><tr><td>FGVC Aircraft <sup><a href="#fn:101">101</a></sup></td><td>The dataset contains 10K images of aircraft, with 100 images for each of 102 different aircraft model variants, most of which are airplanes.</td></tr><tr><td>Country-211 <sup><a href="#fn:117">117</a></sup></td><td>It is a dataset released by OpenAI, designed to assess the geolocation capability of visual representations. It filters the YFCC100M <sup><a href="#fn:136">136</a></sup> dataset to find 211 countries that have at least 300 photos with GPS coordinates. OpenAI built a balanced dataset with 211 categories, by sampling 200 photos for training and 100 photos for testing, for each country.</td></tr><tr><td>Stanford Cars <sup><a href="#fn:72">72</a></sup></td><td>This dataset consists of 196 classes of cars with a total of 16K images, taken from the rear. The data is divided into almost a 50-50 train/test split with 8K training images and 8K testing images.</td></tr></tbody></table>

Table 24: Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.

<table><tbody><tr><td>dataset</td><td>introduction</td></tr><tr><td colspan="2"><em>Testing Datasets for Image Classification.</em></td></tr><tr><td>Birdsnap <sup><a href="#fn:9">9</a></sup></td><td>Birdsnap is a large bird dataset consisting of 49,829 images from 500 bird species with 47,386 images used for training and 2,443 images used for testing. Due to broken links, we are only able to download 1,845 out of the 2,443 testing images.</td></tr><tr><td>DTD <sup><a href="#fn:28">28</a></sup></td><td>The Describable Textures Dataset (DTD) contains 5,640 texture images in the wild. They are annotated with human-centric attributes inspired by the perceptual properties of textures.</td></tr><tr><td>Eurosat <sup><a href="#fn:59">59</a></sup></td><td>This dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27K labeled and geo-referenced samples.</td></tr><tr><td>FER2013 <sup><a href="#fn:52">52</a></sup></td><td>This dataset includes around 30K RGB facial images, categorized into seven expressions: angry, disgust, fear, happy, sad, surprise, and neutral.</td></tr><tr><td>Flowers-102 <sup><a href="#fn:109">109</a></sup></td><td>It is consistent with 102 flower categories commonly occurring in the United Kingdom. Each class consists of between 40 and 258 images.</td></tr><tr><td>Food-101 <sup><a href="#fn:13">13</a></sup></td><td>The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101K images.</td></tr><tr><td>GTSRB <sup><a href="#fn:129">129</a></sup></td><td>The German Traffic Sign Recognition Benchmark (GTSRB) contains 43 classes of traffic signs, split into 39,209 training images and 12,630 test images.</td></tr><tr><td>Pets <sup><a href="#fn:113">113</a></sup></td><td>The Oxford-IIIT Pet Dataset is a 37-category pet dataset with roughly 200 images for each class created by the Visual Geometry Group at Oxford.</td></tr><tr><td>Rendered SST2 <sup><a href="#fn:117">117</a></sup></td><td>This dataset is used to evaluate the model’s capability on optical character recognition. It was generated by rendering sentences in the Standford Sentiment Treebank v2 dataset.</td></tr><tr><td>Resisc45 <sup><a href="#fn:30">30</a></sup></td><td>This is a dataset for remote sensing scene classification. It contains 31,500 RGB images divided into 45 scene classes, each class containing 700 images.</td></tr><tr><td>STL10 <sup><a href="#fn:109">109</a></sup></td><td>The STL-10 dataset, inspired by CIFAR-10 <sup><a href="#fn:74">74</a></sup>, includes 10 classes with 500 training and 800 test color images each, sized 96 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 96 pixels.</td></tr><tr><td>VOC2007 <sup><a href="#fn:45">45</a></sup></td><td>The Pascal VOC 2007 dataset focuses on recognizing objects in realistic scenarios and contains 20 object classes across 9,963 images with 24,640 labeled objects. The data has been divided into 50% for training/validation and 50% for testing. Following common practice, we conduct zero-shot image classification by cropping images to isolate objects using bounding boxes.</td></tr><tr><td colspan="2"><em>Testing Datasets for Video Classification.</em></td></tr><tr><td>Kinetics 400 <sup><a href="#fn:17">17</a></sup></td><td>A large-scale dataset containing around 400 human action classes with at least 400 video clips for each class, sourced from YouTube.</td></tr><tr><td>Kinetics 600 <sup><a href="#fn:18">18</a></sup></td><td>An expansion of Kinetics 400, this dataset includes 600 action classes and provides an increased diversity in video representation.</td></tr><tr><td>Kinetics 700 <sup><a href="#fn:19">19</a></sup></td><td>The latest in the series, Kinetics 700 offers an even broader range with 700 action categories, further challenging the robustness of retrieval models.</td></tr><tr><td colspan="2"><em>Testing Datasets for Image-Text Retrieval.</em></td></tr><tr><td>COCO <sup><a href="#fn:22">22</a></sup></td><td>The COCO Caption dataset contains diverse images with detailed captions, widely used for image-text retrieval and image captioning tasks.</td></tr><tr><td>COCO-CN <sup><a href="#fn:84">84</a></sup></td><td>COCO-CN is a bilingual image description dataset enriching COCO with manually written Chinese sentences and tags. The new dataset can be used for multiple tasks including image tagging, captioning, and retrieval, all in a cross-lingual setting.</td></tr><tr><td>Flickr30K <sup><a href="#fn:116">116</a></sup></td><td>This dataset comprises 31,000 images sourced from Flickr, each annotated with five captions, making it suitable for image-text retrieval.</td></tr><tr><td>Flickr30K-CN <sup><a href="#fn:77">77</a></sup></td><td>Flickr30K-CN offers Chinese captions for the images, enabling studies in cross-lingual and multi-modal retrieval tasks.</td></tr><tr><td>XTD <sup><a href="#fn:1">1</a></sup></td><td>A newly developed 1K multilingual test set, featuring COCO images annotated in various languages.</td></tr><tr><td colspan="2"><em>Testing Dataset for Video-Text Retrieval.</em></td></tr><tr><td>MSR-VTT <sup><a href="#fn:161">161</a></sup></td><td>This is a large-scale dataset for open-domain video captioning and video-text retrieval, comprising 10,000 video clips across 20 categories. Each clip is annotated with 20 English sentences, totaling about 29,000 distinct words in all captions. The standard division of the dataset allocates 6,513 clips for training, 497 for validation, and 2,990 for testing purposes.</td></tr></tbody></table>

Table 25: Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.

<table><tbody><tr><td>dataset</td><td>introduction</td></tr><tr><td colspan="2"><em>Testing Datasets for Image Captioning.</em></td></tr><tr><td>COCO <sup><a href="#fn:22">22</a></sup></td><td>We use the Karpathy test set for testing.</td></tr><tr><td>Flickr30K <sup><a href="#fn:116">116</a></sup></td><td>We use the Karpathy test set for testing.</td></tr><tr><td>NoCaps <sup><a href="#fn:2">2</a></sup></td><td>NoCaps stands out for testing models’ capabilities in open-ended caption generation, using images that go beyond the training data’s domain. We report the performance on the NoCaps val set.</td></tr><tr><td colspan="2"><em>Testing Dataset for Semantic Segmentation.</em></td></tr><tr><td>ADE20K <sup><a href="#fn:185">185</a></sup></td><td>ADE20K contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are a total of 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed. We report the performance on the ADE20K val set.</td></tr></tbody></table>

Table 26: Introduction of datasets used in InternVL’s stage 1 and stage 2. In summary, we utilize a vast amount of image-text data for pre-training and conduct comprehensive evaluation across a wide range of generic visual-linguistic tasks.

<table><tbody><tr><td>dataset</td><td>introduction</td></tr><tr><td colspan="2"><em>Training Data for SFT.</em></td></tr><tr><td>COCO Caption <sup><a href="#fn:22">22</a></sup></td><td>It contains over 0.5 million captions describing over 110K images. Following common practice, we use the Karpathy training set for training. We transform it into a dialogue dataset using the response formatting prompt: “Provide a one-sentence caption for the provided image.”</td></tr><tr><td>TextCaps <sup><a href="#fn:126">126</a></sup></td><td>TextCaps contains 145K captions for 28K images. It challenges a model to recognize text, relate it to its visual context, and decide what part of the text to copy or paraphrase. OCR tokens are used during training. We transform it into a dialogue dataset using the response formatting prompt: “Provide a one-sentence caption for the provided image.”</td></tr><tr><td>VQAv2 <sup><a href="#fn:54">54</a></sup></td><td>VQAv2, the second version of the VQA dataset, features open-ended questions related to images. Answering these questions demands a grasp of vision, language, and common sense. We convert it into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>OKVQA <sup><a href="#fn:104">104</a></sup></td><td>A dataset with over 14K questions requiring external knowledge for answers, focusing on knowledge-based visual question answering. We transform it into a dialogue dataset using the response formatting prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>A-OKVQA <sup><a href="#fn:122">122</a></sup></td><td>An augmented successor of OKVQA <sup><a href="#fn:104">104</a></sup> and contains 25K questions requiring a broad base of commonsense and world knowledge to answer. We transform it into a dialogue dataset using the response formatting prompt: “Answer with the option’s letter from the given choices directly.”</td></tr><tr><td>IconQA <sup><a href="#fn:99">99</a></sup></td><td>A dataset with 107K questions across three sub-tasks, focusing on abstract diagram recognition and comprehensive visual reasoning. We convert it into a dialogue dataset using these prompts: “Answer with the option’s letter from the given choices directly.” and “Answer the question using a single word or phrase.”</td></tr><tr><td>AI2D <sup><a href="#fn:71">71</a></sup></td><td>AI2D features over 5K grade school science diagrams with rich annotations and 15K multiple-choice questions for diagram understanding research. We convert it into a dialogue dataset using the prompt: “Please answer the question based on the options mentioned before.”</td></tr><tr><td>GQA <sup><a href="#fn:64">64</a></sup></td><td>GQA is a large-scale dataset with more than 110K images and 22 million questions, combining real images with balanced question-answer pairs for visual reasoning. We transform it into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>OCR-VQA <sup><a href="#fn:107">107</a></sup></td><td>The OCR-VQA dataset contains 207,572 images of book covers and more than 1 million question-answer pairs about these images. We convert it into a dialogue dataset using the response formatting prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>ChartQA <sup><a href="#fn:105">105</a></sup></td><td>ChartQA is a dataset for question answering about charts, focusing on visual and logical reasoning. It comprises 9.6K human-written questions and 23.1K questions generated from human-written chart summaries. We convert it using the prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>DocVQA <sup><a href="#fn:29">29</a></sup></td><td>The DocVQA dataset consists of 50,000 questions defined on over 12,000 document images. We convert it into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>ST-VQA <sup><a href="#fn:12">12</a></sup></td><td>The ST-VQA dataset contains a total of 31,791 questions over 23,038 images. The training set alone consists of 26,308 questions based on 19,027 images. We convert it into a dialogue dataset using the response formatting prompt: “Answer the question using a single word or phrase.”</td></tr></tbody></table>

Table 27: Introduction of datasets used in InternVL’s stage 3. We collect a wide range of high-quality instruction data. For non-dialogue datasets, we follow the response formatting prompts described in [^91] for conversion. Note that only the training set is used for training.

<table><tbody><tr><td>dataset</td><td>introduction</td></tr><tr><td colspan="2"><em>Training Data for SFT.</em></td></tr><tr><td>EST-VQA <sup><a href="#fn:150">150</a></sup></td><td>The EST-VQA dataset provides questions, images, and answers, but also a bounding box for each question that indicates the area of the image that informs the answer. We convert it into a dialogue dataset using the response formatting prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>InfoVQA <sup><a href="#fn:106">106</a></sup></td><td>This dataset includes a diverse collection of infographics with natural language questions and answers. It focuses on reasoning over document layout, textual content, graphical elements, and data visualizations. We convert it into a dialogue dataset using the prompt: “Answer the question using a single word or phrase.”</td></tr><tr><td>LLaVAR <sup><a href="#fn:182">182</a></sup></td><td>The LLaVAR dataset advances visual instruction tuning for Large Language Models by focusing on text-rich images. It incorporates 422K images processed with OCR and 16K GPT-4 generated conversations, enhancing text-based VQA performance and human interaction capabilities in diverse scenarios. Note that, we only use the 20K high-quality data for fine-tuning of LLaVAR.</td></tr><tr><td>RefCOCO <sup><a href="#fn:170">170</a></sup> <sup><a href="#fn:103">103</a></sup></td><td>A mixed dataset of RefCOCO <sup><a href="#fn:170">170</a></sup>, RefCOCO+ <sup><a href="#fn:170">170</a></sup>, and RefCOCO-g <sup><a href="#fn:103">103</a></sup>. We convert it into a dialogue dataset following LLaVA-1.5 <sup><a href="#fn:91">91</a></sup>.</td></tr><tr><td>Toloka <sup><a href="#fn:140">140</a></sup></td><td>The TolokaVQA dataset comprises images with associated textual questions, each marked with a bounding box indicating the visual answer. It’s sourced from a licensed subset of the COCO dataset and labeled on the Toloka platform. We convert it into a dialogue dataset following LLaVA-1.5 <sup><a href="#fn:91">91</a></sup>.</td></tr><tr><td>LLaVA-150K <sup><a href="#fn:92">92</a></sup></td><td>This is a set of GPT-generated multi-modal instruction-following data, constructed for visual instruction tuning and building large multi-modal models towards GPT-4 vision/language capability. It includes 158K unique language-image instruction-following samples.</td></tr><tr><td>SVIT <sup><a href="#fn:183">183</a></sup></td><td>This dataset includes 3.2 million visual instruction tuning data, with 1.6M conversation QA pairs, 1.6M complex reasoning QA pairs, and 106K detailed image descriptions. It is designed to improve multi-modal performance in visual perception, reasoning, and planning. For this dataset, we merge the QA pairs from the same training image into a single conversation.</td></tr><tr><td>VisDial <sup><a href="#fn:36">36</a></sup></td><td>A dataset based on the COCO images, featuring dialogues created by two Amazon Mechanical Turk workers. One plays the ‘questioner’, seeing only an image’s text description, and the other, the ‘answerer’, sees the image. They engage in a 10-round Q&A session about the image.</td></tr><tr><td>LRV-Instruction <sup><a href="#fn:90">90</a></sup></td><td>The LRV-Instruction dataset is designed to combat hallucination in large multi-modal models. It comprises 120K GPT-4-generated visual instructions for 16 vision-and-language tasks, including both positive and negative instructions for robust tuning. Negative instructions focus on Nonexistent and Existent Element Manipulation. This dataset helps improve accuracy and consistency in multi-modal tasks.</td></tr><tr><td>LLaVA-Mix-665K <sup><a href="#fn:91">91</a></sup></td><td>LLaVA-Mix-665K is an instruction-following dataset mixed from 10 academically oriented datasets.</td></tr><tr><td colspan="2"><em>Testing Dataset for SFT (Image Captioning).</em></td></tr><tr><td>COCO <sup><a href="#fn:22">22</a></sup></td><td>Karpathy test set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”</td></tr><tr><td>Flickr30K <sup><a href="#fn:116">116</a></sup></td><td>Karpathy test set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”</td></tr><tr><td>NoCaps <sup><a href="#fn:2">2</a></sup></td><td>NoCaps val set is used for testing. The prompt is: “Provide a one-sentence caption for the provided image.”</td></tr><tr><td colspan="2"><em>Testing Dataset for SFT (Visual Question Answering).</em></td></tr><tr><td>VQAv2 <sup><a href="#fn:54">54</a></sup></td><td>VQAv2 test-dev set is used for testing. The prompt is: “Answer the question using a single word or phrase.”</td></tr><tr><td>GQA <sup><a href="#fn:64">64</a></sup></td><td>GQA test-balanced set is used. The prompt is: “Answer the question using a single word or phrase.”</td></tr><tr><td>VizWiz <sup><a href="#fn:56">56</a></sup></td><td>VizWiz test-dev set is used for testing. The prompt is: “When the provided information is insufficient, respond with ‘Unanswerable’. Answer the question using a single word or phrase.”</td></tr><tr><td>TextVQA <sup><a href="#fn:127">127</a></sup></td><td>TextVQA val set is used for testing. The prompt is: “Answer the question using a single word or phrase.”</td></tr><tr><td colspan="2"><em>Testing Dataset for SFT (Multi-Modal Dialogue).</em></td></tr><tr><td>MME <sup><a href="#fn:50">50</a></sup></td><td>MME is a comprehensive evaluation benchmark for multi-modal large language models. It measures both perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster, celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation, and code reasoning. The prompt for this dataset is: “Answer the question using a single word or phrase.”</td></tr><tr><td>POPE <sup><a href="#fn:86">86</a></sup></td><td>POPE is a popular dataset used to evaluate object hallucination. The response formatting prompt used for this dataset is: “Answer the question using a single word or phrase.”</td></tr></tbody></table>

Table 28: Introduction of datasets used in InternVL’s stage 3. We collect a wide range of high-quality instruction data. For non-dialogue datasets, we follow the response formatting prompts described in [^91] for conversion. Note that only the training set is used for training. We evaluate our InternVL-Chat models on three tasks, including image captioning, VQA, and multi-modal dialogue. For these datasets, we employ the same response formatting prompts as for LLaVA-1.5 [^91].

[^1]: Pranav Aggarwal and Ajinkya Kale. Towards zero-shot cross-lingual image retrieval. *arXiv preprint arXiv:2012.05107*, 2020.

[^2]: Harsh Agrawal, Karan Desai, Yufei Wang, Xinlei Chen, Rishabh Jain, Mark Johnson, Dhruv Batra, Devi Parikh, Stefan Lee, and Peter Anderson. Nocaps: Novel object captioning at scale. In *ICCV*, pages 8948–8957, 2019.

[^3]: Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. *NeurIPS*, 35:23716–23736, 2022.

[^4]: Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023a.

[^5]: Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*, 2023b.

[^6]: Baichuan. Baichuan 2: Open large-scale language models. *arXiv preprint arXiv:2309.10305*, 2023.

[^7]: Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. In *ICLR*, 2022.

[^8]: Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. *NeurIPS*, 32, 2019.

[^9]: Thomas Berg, Jiongxin Liu, Seung Woo Lee, Michelle L Alexander, David W Jacobs, and Peter N Belhumeur. Birdsnap: Large-scale fine-grained visual categorization of birds. In *CVPR*, pages 2011–2018, 2014.

[^10]: Lucas Beyer, Olivier J Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. Are we done with imagenet? *arXiv preprint arXiv:2006.07159*, 2020.

[^11]: Federico Bianchi, Giuseppe Attanasio, Raphael Pisoni, Silvia Terragni, Gabriele Sarti, and Sri Lakshmi. Contrastive language-image pre-training for the italian language. *arXiv preprint arXiv:2108.08688*, 2021.

[^12]: Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas. Scene text visual question answering. In *ICCV*, pages 4291–4301, 2019.

[^13]: Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In *ECCV*, pages 446–461, 2014.

[^14]: Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset, 2022.

[^15]: Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiangwen Kong, Jun Li, and Xiangyu Zhang. Reversible column networks. *arXiv preprint arXiv:2212.11696*, 2022.

[^16]: Fredrik Carlsson, Philipp Eisen, Faton Rekathati, and Magnus Sahlgren. Cross-lingual and multilingual clip. In *Proceedings of the Thirteenth Language Resources and Evaluation Conference*, pages 6848–6854, 2022.

[^17]: Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In *CVPR*, pages 6299–6308, 2017.

[^18]: Joao Carreira, Eric Noland, Andras Banki-Horvath, Chloe Hillier, and Andrew Zisserman. A short note about kinetics-600. *arXiv preprint arXiv:1808.01340*, 2018.

[^19]: Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. *arXiv preprint arXiv:1907.06987*, 2019.

[^20]: Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In *CVPR*, pages 3558–3568, 2021.

[^21]: Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. *arXiv preprint arXiv:2306.15195*, 2023a.

[^22]: Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. *arXiv preprint arXiv:1504.00325*, 2015.

[^23]: Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, et al. Pali: A jointly-scaled multilingual language-image model. In *ICLR*, 2022a.

[^24]: Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. Pali-x: On scaling up a multilingual vision and language model. *arXiv preprint arXiv:2305.18565*, 2023b.

[^25]: Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. In *ICLR*, 2022b.

[^26]: Zhongzhi Chen, Guang Liu, Bo-Wen Zhang, Fulong Ye, Qinghong Yang, and Ledell Wu. Altclip: Altering the language encoder in clip for extended language capabilities. *arXiv preprint arXiv:2211.06679*, 2022c.

[^27]: Gong Cheng, Junwei Han, and Xiaoqiang Lu. Remote sensing image scene classification: Benchmark and state of the art. *Proceedings of the IEEE*, 105(10):1865–1883, 2017.

[^28]: Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In *CVPR*, pages 3606–3613, 2014.

[^29]: Christopher Clark and Matt Gardner. Simple and effective multi-paragraph reading comprehension. *arXiv preprint arXiv:1710.10723*, 2017.

[^30]: Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *AISTAT*, pages 215–223, 2011.

[^31]: MMSegmentation Contributors. Mmsegmentation: Openmmlab semantic segmentation toolbox and benchmark, 2020.

[^32]: Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and effective text encoding for chinese llama and alpaca. *arXiv preprint arXiv:2304.08177*, 2023.

[^33]: Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In *ICCV*, pages 764–773, 2017.

[^34]: Wenliang Dai, Junnan Li, Dongxu Li, AnthonyMeng Huat, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. *arXiv preprint arXiv:2305.06500*, 2023.

[^35]: Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. *NeurIPS*, 35:16344–16359, 2022.

[^36]: Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José MF Moura, Devi Parikh, and Dhruv Batra. Visual dialog. In *CVPR*, pages 326–335, 2017.

[^37]: Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In *ICML*, pages 7480–7512, 2023.

[^38]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *CVPR*, pages 248–255, 2009.

[^39]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

[^40]: Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In *CVPR*, pages 13733–13742, 2021.

[^41]: Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. *arXiv preprint arXiv:2309.11499*, 2023.

[^42]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2020.

[^43]: Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. *arXiv preprint arXiv:2303.03378*, 2023.

[^44]: Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In *ACL*, pages 320–335, 2022.

[^45]: Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. *IJCV*, 111:98–136, 2015.

[^46]: Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. *arXiv preprint arXiv:2211.07636*, 2022.

[^47]: Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. *arXiv preprint arXiv:2303.11331*, 2023.

[^48]: William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR*, 23(1):5232–5270, 2022.

[^49]: Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In *CVPRW*, pages 178–178, 2004.

[^50]: Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv preprint arXiv:2306.13394*, 2023.

[^51]: Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. *arXiv preprint arXiv:2304.15010*, 2023.

[^52]: Ian J Goodfellow, Dumitru Erhan, Pierre Luc Carrier, Aaron Courville, Mehdi Mirza, Ben Hamner, Will Cukierski, Yichuan Tang, David Thaler, Dong-Hyun Lee, et al. Challenges in representation learning: A report on three machine learning contests. In *ICONIP*, pages 117–124, 2013.

[^53]: Google. Google bard. [https://bard.google.com/](https://bard.google.com/), 2023.

[^54]: Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *CVPR*, pages 6904–6913, 2017.

[^55]: Jiaxi Gu, Xiaojun Meng, Guansong Lu, Lu Hou, Niu Minzhe, Xiaodan Liang, Lewei Yao, Runhui Huang, Wei Zhang, Xin Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. *NeurIPS*, 35:26418–26431, 2022.

[^56]: Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In *CVPR*, pages 3608–3617, 2018.

[^57]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, pages 770–778, 2016.

[^58]: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In *CVPR*, pages 16000–16009, 2022.

[^59]: Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7):2217–2226, 2019.

[^60]: Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In *ICCV*, pages 8340–8349, 2021a.

[^61]: Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In *CVPR*, pages 15262–15271, 2021b.

[^62]: Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In *CVPR*, pages 7132–7141, 2018.

[^63]: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In *ECCV*, pages 646–661, 2016.

[^64]: Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In *CVPR*, pages 6700–6709, 2019.

[^65]: Forrest Iandola, Matt Moskewicz, Sergey Karayev, Ross Girshick, Trevor Darrell, and Kurt Keutzer. Densenet: Implementing efficient convnet descriptor pyramids. *arXiv preprint arXiv:1404.1869*, 2014.

[^66]: IDEFICS. Introducing idefics: An open reproduction of state-of-the-art visual language model. [https://huggingface.co/blog/idefics](https://huggingface.co/blog/idefics), 2023.

[^67]: Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip. Zenodo. Version 0.1. [https://doi.org/10.5281/zenodo.5143773](https://doi.org/10.5281/zenodo.5143773), 2021. DOI: 10.5281/zenodo.5143773.

[^68]: Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *ICML*, pages 448–456, 2015.

[^69]: Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and Jason Baldridge. Mural: multimodal, multitask retrieval across languages. *arXiv preprint arXiv:2109.05125*, 2021.

[^70]: Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In *ICML*, pages 4904–4916, 2021.

[^71]: Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *ECCV*, pages 235–251, 2016.

[^72]: Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In *ICCVW*, pages 554–561, 2013.

[^73]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *NeurIPS*, 25, 2012.

[^74]: Alex Krizhevsky et al. Learning multiple layers of features from tiny images. 2009.

[^75]: Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. *arXiv preprint arXiv:2308.00692*, 2023.

[^76]: LAION-AI. Clip benchmark: Clip-like model evaluation. [https://github.com/LAION-AI/CLIP\_benchmark](https://github.com/LAION-AI/CLIP_benchmark), 2023.

[^77]: Weiyu Lan, Xirong Li, and Jianfeng Dong. Fluency-guided cross-lingual image captioning. In *ACM MM*, pages 1549–1557, 2017.

[^78]: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[^79]: Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. *arXiv preprint arXiv:2305.03726*, 2023a.

[^80]: Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In *ICML*, pages 12888–12900, 2022.

[^81]: Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *arXiv preprint arXiv:2301.12597*, 2023b.

[^82]: KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. *arXiv preprint arXiv:2305.06355*, 2023c.

[^83]: Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. *arXiv preprint arXiv:2303.16058*, 2023d.

[^84]: Xirong Li, Chaoxi Xu, Xiaoxu Wang, Weiyu Lan, Zhengxiong Jia, Gang Yang, and Jieping Xu. Coco-cn for cross-lingual image tagging, captioning, and retrieval. *TMM*, 21(9):2347–2360, 2019.

[^85]: Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Improved multiscale vision transformers for classification and detection. *arXiv preprint arXiv:2112.01526*, 2021.

[^86]: Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. *arXiv preprint arXiv:2305.10355*, 2023e.

[^87]: Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, and Kaiming He. Scaling language-image pre-training via masking. In *CVPR*, pages 23390–23400, 2023f.

[^88]: Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. *arXiv preprint arXiv:2311.06607*, 2023g.

[^89]: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *ECCV*, pages 740–755, 2014.

[^90]: Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. *arXiv preprint arXiv:2306.14565*, 2023a.

[^91]: Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. *arXiv preprint arXiv:2310.03744*, 2023b.

[^92]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. *NeurIPS*, 2023c.

[^93]: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

[^94]: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *ICCV*, pages 10012–10022, 2021.

[^95]: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. *arXiv preprint arXiv:2201.03545*, 2022.

[^96]: Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatgpt beyond language. *arXiv preprint arXiv:2305.05662*, 2023d.

[^97]: Zhaoyang Liu, Zeqiang Lai, Zhangwei Gao, Erfei Cui, Xizhou Zhu, Lewei Lu, Qifeng Chen, Yu Qiao, Jifeng Dai, and Wenhai Wang. Controlllm: Augment language models with tools by searching on graphs. *arXiv preprint arXiv:2310.17796*, 2023e.

[^98]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*, 2017.

[^99]: Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. *arXiv preprint arXiv:2110.13214*, 2021.

[^100]: Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, and Yelong Shen. An empirical study of scaling instruct-tuned large multimodal models. *arXiv preprint arXiv:2309.09958*, 2023.

[^101]: Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. *arXiv preprint arXiv:1306.5151*, 2013.

[^102]: Kei Sawada Makoto Shiin, Tianyu Zhao. Construction and public release of language image pretraining models in japanese. In *The 25th Meeting on Image Recognition and Understanding*, 2022.

[^103]: Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In *CVPR*, pages 11–20, 2016.

[^104]: Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In *CVPR*, pages 3195–3204, 2019.

[^105]: Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. *arXiv preprint arXiv:2203.10244*, 2022.

[^106]: Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In *WACV*, pages 1697–1706, 2022.

[^107]: Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In *ICDAR*, pages 947–952. IEEE, 2019.

[^108]: Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pre-training via embodied chain of thought. *arXiv preprint arXiv:2305.15021*, 2023.

[^109]: Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In *ICVGIP*, pages 722–729, 2008.

[^110]: OpenAI. Gpt-4 technical report, 2023.

[^111]: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. *arXiv preprint arXiv:2304.07193*, 2023.

[^112]: Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. In *NeurIPS*, 2011.

[^113]: Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In *CVPR*, pages 3498–3505, 2012.

[^114]: Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. *arXiv preprint arXiv:2306.01116*, 2023.

[^115]: Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. *arXiv preprint arXiv:2306.14824*, 2023.

[^116]: Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. In *ICCV*, pages 2641–2649, 2015.

[^117]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *ICML*, pages 8748–8763, 2021.

[^118]: Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In *SIGKDD*, pages 3505–3506, 2020.

[^119]: Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In *ICML*, pages 5389–5400, 2019.

[^120]: Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. *NeurIPS*, 35:25278–25294, 2022a.

[^121]: Christoph Schuhmann, Andreas Köpf, Richard Vencu, Theo Coombes, and Romain Beaumont. Laion coco: 600m synthetic captions from laion2b-en. *https://laion.ai/blog/laion-coco/*, 2022b.

[^122]: Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In *ECCV*, pages 146–162, 2022.

[^123]: Wenqi Shao, Yutao Hu, Peng Gao, Meng Lei, Kaipeng Zhang, Fanqing Meng, Peng Xu, Siyuan Huang, Hongsheng Li, Yu Qiao, et al. Tiny lvlm-ehub: Early multimodal experiments with bard. *arXiv preprint arXiv:2308.03729*, 2023.

[^124]: Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *ACL*, 2018.

[^125]: Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. *arXiv preprint arXiv:2303.17580*, 2023.

[^126]: Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh. Textcaps: a dataset for image captioning with reading comprehension. In *ECCV*, pages 742–758, 2020.

[^127]: Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In *CVPR*, pages 8317–8326, 2019.

[^128]: Mannat Singh, Quentin Duval, Kalyan Vasudev Alwala, Haoqi Fan, Vaibhav Aggarwal, Aaron Adcock, Armand Joulin, Piotr Dollár, Christoph Feichtenhofer, Ross Girshick, et al. The effectiveness of mae pre-pretraining for billion-scale pretraining. *arXiv preprint arXiv:2303.13496*, 2023.

[^129]: Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. *Neural networks*, 32:323–332, 2012.

[^130]: Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. *arXiv preprint arXiv:2303.15389*, 2023a.

[^131]: Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. *arXiv preprint arXiv:2307.05222*, 2023b.

[^132]: Tianxiang Sun, Xiaotian Zhang, Zhengfu He, Peng Li, Qinyuan Cheng, Hang Yan, Xiangyang Liu, Yunfan Shao, Qiong Tang, Xingjian Zhao, Ke Chen, Yining Zheng, Zhejian Zhou, Ruixiao Li, Jun Zhan, Yunhua Zhou, Linyang Li, Xiaogui Yang, Lingling Wu, Zhangyue Yin, Xuanjing Huang, and Xipeng Qiu. Moss: Training conversational language models from synthetic data. 2023c.

[^133]: Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. *arXiv preprint arXiv:2303.08128*, 2023.

[^134]: Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B Hashimoto. Alpaca: A strong, replicable instruction-following model. *Stanford Center for Research on Foundation Models. https://crfm. stanford. edu/2023/03/13/alpaca. html*, 3(6):7, 2023.

[^135]: InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. [https://github.com/InternLM/InternLM](https://github.com/InternLM/InternLM), 2023.

[^136]: Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. *Communications of the ACM*, 59(2):64–73, 2016.

[^137]: Hugo Touvron, Matthieu Cord, and Hervé Jégou. Deit iii: Revenge of the vit. In *ECCV*, pages 516–533, 2022.

[^138]: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.

[^139]: Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

[^140]: Dmitry Ustalov, Nikita Pavlichenko, Sergey Koshelev, Daniil Likhobaba, and Alisa Smirnova. Toloka visual question answering benchmark. *arXiv preprint arXiv:2309.16511*, 2023.

[^141]: Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. *NeurIPS*, 32, 2019.

[^142]: Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, and Lu Yuan. Omnivl: One foundation model for image-language and video-language tasks. *NeurIPS*, 35:5696–5710, 2022a.

[^143]: Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. *arXiv preprint arXiv:2305.11172*, 2023a.

[^144]: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *ICCV*, pages 568–578, 2021.

[^145]: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvtv2: Improved baselines with pyramid vision transformer. *CVMJ*, pages 1–10, 2022b.

[^146]: Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for vision and vision-language tasks. In *CVPR*, pages 19175–19186, 2023b.

[^147]: Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. *NeurIPS*, 2023c.

[^148]: Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. In *CVPR*, pages 14408–14419, 2023d.

[^149]: Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. *arXiv preprint arXiv:2308.01907*, 2023e.

[^150]: Xinyu Wang, Yuliang Liu, Chunhua Shen, Chun Chet Ng, Canjie Luo, Lianwen Jin, Chee Seng Chan, Anton van den Hengel, and Liangwei Wang. On the general value of evidence, and bilingual scene-text visual question answering. In *CVPR*, pages 10126–10135, 2020.

[^151]: Yi Wang, Kunchang Li, Yizhuo Li, Yinan He, Bingkun Huang, Zhiyu Zhao, Hongjie Zhang, Jilan Xu, Yi Liu, Zun Wang, et al. Internvideo: General video foundation models via generative and discriminative learning. *arXiv preprint arXiv:2212.03191*, 2022c.

[^152]: Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinyuan Chen, Yaohui Wang, Ping Luo, Ziwei Liu, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. *arXiv preprint arXiv:2307.06942*, 2023f.

[^153]: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. *NeurIPS*, 35:24824–24837, 2022.

[^154]: Tianwen Wei, Liang Zhao, Lichang Zhang, Bo Zhu, Lijie Wang, Haihua Yang, Biye Li, Cheng Cheng, Weiwei Lü, Rui Hu, et al. Skywork: A more open bilingual foundation model. *arXiv preprint arXiv:2310.19341*, 2023.

[^155]: Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. *arXiv preprint arXiv:2303.04671*, 2023a.

[^156]: Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. *arXiv preprint arXiv:2309.05519*, 2023b.

[^157]: Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In *CVPRW*, pages 3485–3492, 2010.

[^158]: Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In *ECCV*, pages 418–434, 2018.

[^159]: Chunyu Xie, Jincheng Li, Heng Cai, Fanjing Kong, Xiaoyu Wu, Jianfei Song, Henrique Morimitsu, Lin Yao, Dexin Wang, Dawei Leng, et al. Zero and r2d2: A large-scale chinese cross-modal benchmark and a vision-language framework. *arXiv preprint arXiv:2205.03860*, 2022.

[^160]: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In *CVPR*, pages 1492–1500, 2017.

[^161]: Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In *CVPR*, pages 5288–5296, 2016.

[^162]: An Yang, Junshu Pan, Junyang Lin, Rui Men, Yichang Zhang, Jingren Zhou, and Chang Zhou. Chinese clip: Contrastive vision-language pretraining in chinese. *arXiv preprint arXiv:2211.01335*, 2022.

[^163]: Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. Gpt4tools: Teaching large language model to use tools via self-instruction. *arXiv preprint arXiv:2305.18752*, 2023a.

[^164]: Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax Law, Noah Constant, Gustavo Hernandez Abrego, Steve Yuan, Chris Tar, Yun-Hsuan Sung, et al. Multilingual universal sentence encoder for semantic retrieval. In *ACL*, pages 87–94, 2020.

[^165]: Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). *arXiv preprint arXiv:2309.17421*, 9, 2023b.

[^166]: Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. *arXiv preprint arXiv:2303.11381*, 2023c.

[^167]: Lewei Yao, Runhui Huang, Lu Hou, Guansong Lu, Minzhe Niu, Hang Xu, Xiaodan Liang, Zhenguo Li, Xin Jiang, and Chunjing Xu. Filip: Fine-grained interactive language-image pre-training. In *ICLR*, 2021.

[^168]: Jiabo Ye, Anwen Hu, Haiyang Xu, Qinghao Ye, Ming Yan, Yuhao Dan, Chenlin Zhao, Guohai Xu, Chenliang Li, Junfeng Tian, Qian Qi, Ji Zhang, and Fei Huang. mplug-docowl: Modularized multimodal large language model for document understanding, 2023.

[^169]: Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. *arXiv preprint arXiv:2205.01917*, 2022.

[^170]: Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In *ECCV*, pages 69–85, 2016.

[^171]: Lu Yuan, Dongdong Chen, Yi-Ling Chen, Noel Codella, Xiyang Dai, Jianfeng Gao, Houdong Hu, Xuedong Huang, Boxin Li, Chunyuan Li, et al. Florence: A new foundation model for computer vision. *arXiv preprint arXiv:2111.11432*, 2021.

[^172]: Yan Zeng, Hanbo Zhang, Jiani Zheng, Jiangnan Xia, Guoqiang Wei, Yang Wei, Yuchen Zhang, and Tao Kong. What matters in training a gpt4-style language model with multimodal inputs? *arXiv preprint arXiv:2307.02469*, 2023.

[^173]: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In *CVPR*, pages 12104–12113, 2022a.

[^174]: Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In *CVPR*, pages 18123–18133, 2022b.

[^175]: Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. *arXiv preprint arXiv:2306.02858*, 2023a.

[^176]: Jiaxing Zhang, Ruyi Gan, Junjie Wang, Yuxiang Zhang, Lin Zhang, Ping Yang, Xinyu Gao, Ziwei Wu, Xiaoqun Dong, Junqing He, et al. Fengshenbang 1.0: Being the foundation of chinese cognitive intelligence. *arXiv preprint arXiv:2209.02970*, 2022.

[^177]: Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, et al. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. *arXiv preprint arXiv:2309.15112*, 2023b.

[^178]: Qinglong Zhang and Yu-Bin Yang. Rest: An efficient transformer for visual recognition. *NeurIPS*, 34:15475–15485, 2021.

[^179]: Qinglong Zhang and Yu-Bin Yang. Rest v2: simpler, faster and stronger. *NeurIPS*, 35:36440–36452, 2022.

[^180]: Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. *arXiv preprint arXiv:2303.16199*, 2023c.

[^181]: Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. *arXiv preprint arXiv:2307.03601*, 2023d.

[^182]: Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. *arXiv preprint arXiv:2306.17107*, 2023e.

[^183]: Bo Zhao, Boya Wu, and Tiejun Huang. Svit: Scaling up visual instruction tuning. *arXiv preprint arXiv:2307.04087*, 2023.

[^184]: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. *arXiv preprint arXiv:2306.05685*, 2023.

[^185]: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *CVPR*, pages 633–641, 2017.

[^186]: Bin Zhu, Bin Lin, Munan Ning, Yang Yan, Jiaxi Cui, HongFa Wang, Yatian Pang, Wenhao Jiang, Junwu Zhang, Zongwei Li, et al. Languagebind: Extending video-language pretraining to n-modality by language-based semantic alignment. *arXiv preprint arXiv:2310.01852*, 2023a.

[^187]: Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. *arXiv preprint arXiv:2304.10592*, 2023b.

[^188]: Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. *arXiv preprint arXiv:2305.17144*, 2023c.