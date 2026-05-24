---
title: "DINOv3"
source: "https://ar5iv.labs.arxiv.org/html/2508.10104"
author:
published:
created: 2026-05-24
description: "Self-supervised learning holds the promise of eliminating the need for manual data annotation, enabling models to scale effortlessly to massive datasets and larger architectures.By not being tailored to specific tasks…"
tags:
  - "clippings"
---
Oriane Siméoni <sup>∗</sup> Huy V. Vo <sup>∗</sup> Maximilian Seitzer <sup>∗</sup> Federico Baldassarre <sup>∗</sup> Maxime Oquab <sup>∗</sup>  
Cijo Jose  Vasil Khalidov  Marc Szafraniec  Seungeun Yi  Michaël Ramamonjisoa  
Francisco Massa    Daniel Haziza    Luca Wehrstedt    Jianyuan Wang  
Timothée Darcet   Théo Moutakanni   Leonel Sentana   Claire Roberts  
Andrea Vedaldi   Jamie Tolan   John Brandt <sup>1</sup>    Camille Couprie  
Julien Mairal <sup>2</sup>    Hervé Jégou   Patrick Labatut   Piotr Bojanowski  
Meta AI Research          <sup>1</sup> WRI          <sup>2</sup> Inria  
<sup>∗</sup> corresponding authors: {osimeoni,huyvvo,seitzer,baldassarre,qas}@meta.com

###### Abstract

Self-supervised learning holds the promise of eliminating the need for manual data annotation, enabling models to scale effortlessly to massive datasets and larger architectures. By not being tailored to specific tasks or domains, this training paradigm has the potential to learn visual representations from diverse sources, ranging from natural to aerial images—using a single algorithm. This technical report introduces DINOv3, a major milestone toward realizing this vision by leveraging simple yet effective strategies. First, we leverage the benefit of scaling both dataset and model size by careful data preparation, design, and optimization. Second, we introduce a new method called Gram anchoring, which effectively addresses the known yet unsolved issue of dense feature maps degrading during long training schedules. Finally, we apply post-hoc strategies that further enhance our models’ flexibility with respect to resolution, model size, and alignment with text. As a result, we present a versatile vision foundation model that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models. We also share the DINOv3 suite of vision models, designed to advance the state of the art on a wide spectrum of tasks and data by providing scalable solutions for diverse resource constraints and deployment scenarios.

## 1 Introduction

Foundation models have become a central building block in modern computer vision, enabling broad generalization across tasks and domains through a single, reusable model. Self-supervised learning (SSL) is a powerful approach for training such models, by learning directly from raw pixel data and leveraging the natural co-occurrences of patterns in images. Unlike weakly and fully supervised pretraining methods [^144] [^48] [^18] which require images paired with high-quality metadata, SSL unlocks training on massive, raw image collections. This is particularly effective for training large-scale visual encoders thanks to the availability of virtually unlimited training data. DINOv2 [^134] exemplifies these strengths, achieving impressive results in image understanding tasks [^199] and enabling pre-training for complex domains such as histopathology [^31]. Models trained with SSL exhibit additional desirable properties: they are robust to input distribution shifts, provide strong global and local features, and generate rich embeddings that facilitate physical scene understanding. Since SSL models are not trained for any specific downstream task, they produce versatile and robust generalist features. For instance, DINOv2 models deliver strong performance across diverse tasks and domains without requiring task-specific finetuning, allowing a single frozen backbone to serve multiple purposes. Importantly, self-supervised learning is especially suitable to train on the vast amount of available observational data in domains like histopathology [^197], biology [^105], medical imaging [^140], remote sensing [^43] [^180], astronomy [^136], or high-energy particle physics [^51]. These domain often lack metadata and have already been shown to benefit from foundation models like DINOv2. Finally, SSL, requiring no human intervention, is well-suited for lifelong learning amid the growing volume of web data.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x1.png)

Figure 1: (a) Evolution of linear probing results on ImageNet1k (IN1k) over the years, comparing fully- (SL), weakly- (WSL) and self-supervised learning (SSL) methods. Despite coming into the picture later, SSL has quickly progressed and now reached the Imagenet accuracy plateau of recent years. On the other hand, we demonstrate that SSL offers the unique promise of high-quality dense features. With DINOv3, we markedly improve over weakly-supervised models on dense tasks, as shown by the relative performance of the best-in-class WSL models to DINOv3 (b). We also produce PCA maps of features obtained from high resolution images with DINOv3 trained on natural (c) and aerial images (d).

In practice, the promise of SSL, namely producing arbitrarily large and powerful models by leveraging large amounts of unconstrained data, remains challenging at scale. While model instabilities and collapse are mitigated by the heuristics proposed by [^134], more problems emerge from scaling further. First, it is unclear how to collect useful data from unlabeled collections. Second, in usual training practice, employing cosine schedules implies knowing the optimization horizon a priori, which is difficult when training on large image corpora. Third, the performance of the features gradually decreases after early training, confirmed by visual inspection of the patch similarity maps. This phenomenon appears in longer training runs with models above ViT-Large size (300M parameters), reducing the usefulness of scaling DINOv2.

Addressing the problems above leads to this work, *DINOv3*, which advances SSL training at scale. We demonstrate that a *single frozen SSL backbone* can serve as a universal visual encoder that achieves state-of-the-art performance on challenging downstream tasks, outperforming supervised and metadata-reliant pre-training strategies. Our research is guided by the following objectives: (1) training a foundational model versatile across tasks and domains, (2) improving the shortcomings of existing SSL models on dense features, (3) disseminating a family of models that can be used off-the-shelf. We discuss the three aims in the following.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x3.png)

(a) Semantic segmentation (ADE20k)

##### Strong & Versatile Foundational Models

DINOv3 aims to offer a high level of versatility along two axes, which is enabled by the scaling of the model size and training data. First, a key desirable property for SSL models is to achieve excellent performance while being kept frozen, ideally reaching similar state-of-the-art results as specialized models. In that case, a single forward pass can deliver cutting-edge results across multiple tasks, leading to substantial computational savings—an essential advantage for practical applications, particularly on edge devices. We show the wide breadth of tasks that DINOv3 can successfully be applied to in Sec.˜6. Second, a scalable SSL training pipeline that does not depend on metadata unlocks numerous scientific applications. By pre-training on a diverse set of images, whether web images or observational data, SSL models generalize across a large set of domains and tasks. As illustrated in Fig.˜1(d), the PCA of DINOv3 features extracted from a high-resolution aerial image clearly allows to separates roads, houses, and greenery, highlighting the model’s feature quality.

##### Superior Feature Maps Through Gram Anchoring

Another key feature of DINOv3 is a significant improvement of its dense feature maps. The DINOv3 SSL training strategy aims at producing models excelling at high-level semantic tasks while producing excellent feature maps amenable to solving geometric tasks such as depth estimation, or 3D matching. In particular, the models should produce dense features that can be used off-the-shelf or with little post-processing. The compromise between dense and global representation is especially difficult to optimize when training with vast amounts of images, since the objective of high-level understanding can conflict with the quality of the dense feature maps. These contradictory objectives lead to a collapse of dense features with large models and long training schedules. Our new Gram anchoring strategy effectively mitigates this collapse (see Sec.˜4). As a result, DINOv3 obtains significantly better dense feature maps than DINOv2, staying clean even at high resolutions (see Fig.˜3).

##### The DINOv3 Family of Models

Solving the degradation of dense feature map with Gram anchoring unlocks the power of scaling. As a consequence, training a much larger model with SSL leads to significant performance improvements. In this work, we successfully train a DINO model with 7B parameters. Since such a large model requires significant resources to run, we apply distillation to compress its knowledge into smaller variants. As a result, we present the *DINOv3 family of vision models*, a comprehensive suite designed to address a wide spectrum of computer vision challenges. This model family aims to advance the state of the art by offering scalable solutions adaptable to diverse resource constraints and deployment scenarios. The distillation process produces model variants at multiple scales, including Vision Transformer (ViT) Small, Base, and Large, as well as ConvNeXt-based architectures. Notably, the efficient and widely adopted ViT-L model achieves performance close to that of the original 7B teacher across a variety of tasks. Overall, the DINOv3 family demonstrates strong performance on a broad range of benchmarks, matching or exceeding the accuracy of competing models on global tasks, while significantly outperforming them on dense prediction tasks, as visible in Fig.˜2.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/figures/introduction/clutter_scene/viridis_hd/cosmap_0_cross.lr.jpg)

Figure 3: High-resolution dense features. We visualize the cosine similarity maps obtained with DINOv3 output features between the patches marked with a red cross and all other patches. Input image at 4096×4096. Please zoom in, do you agree with DINOv3?

##### Overview of Contributions

In this work, we introduce multiple contributions to address the challenge of scaling SSL towards a large frontier model. We build upon recent advances in automatic data curation [^196] to obtain a large “background” training dataset that we carefully mix with a bit of specialized data (ImageNet-1k). This allows leveraging large amounts of unconstrained data to improve the model performance. This contribution (i) around data scaling will be described in Sec.˜3.1.

We increase our main model size to 7B parameters by defining a custom variant of the ViT architecture. We include modern position embeddings (axial RoPE) and develop a regularization technique to avoid positional artifacts. Departing from the multiple cosine schedules in DINOv2, we train with constant hyperparameter schedules for 1M iterations. This allows producing models with stronger performance. This contribution (ii) on model architecture and training will be described in Sec.˜3.2.

With the above techniques, we are able to train a model following the DINOv2 algorithm at scale. However, as mentioned previously, scale leads to a degradation of dense features. To address this, we propose a core improvement of the pipeline with a Gram anchoring training phase. This cleans the noise in the feature maps, leading to impressive similarity maps, and drastically improving the performance on both parametric and non-parametric dense tasks. This contribution (iii) on Gram training will be described in Sec.˜4.

Following previous practice, the last steps of our pipeline consist of a high-resolution post-training phase and distillation into a series of high-performance models of various sizes. For the latter, we develop a novel and efficient single-teacher multiple-students distillation procedure. This contribution (iv) transfers the power of our 7B frontier model to a family of smaller practical models for common usage, that we describe in Sec.˜5.2.

As measured in our thorough benchmarking, results in Sec.˜6 show that our approach defines a new standard in dense tasks and performs comparably to CLIP derivatives on global tasks. In particular, with a frozen vision backbone, we achieve state-of-the-art performance on longstanding computer vision problems such as object detection (COCO detection, mAP 66.1) and image segmentation (ADE20k, mIoU 63.0), outperforming specialized fine-tuned pipelines. Moreover, we provide evidence of the generality of our approach across domains by applying the DINOv3 algorithm to satellite imagery, in Sec.˜8, surpassing all prior approaches.

## 2 Related Work

##### Self-Supervised Learning

Learning without annotations requires an artificial learning task that provides supervision in lieu for training. The art and challenge of SSL lies in carefully designing these so-called pre-text tasks in order to learn powerful representations for downstream tasks. The language domain, by its discrete nature, offers straightforward ways to set up such tasks, which led to many successful unsupervised pre-training approaches for text data. Examples include word embeddings [^128] [^17], sentence representations [^50] [^121], and plain language models [^127] [^232]. In contrast, computer vision presents greater challenges due to the continuous nature of the signal. Early attempts mimicking language approaches extracted supervisory signals from parts of an image to predict other parts, *e.g*. by predicting relative patch position [^53], patch re-ordering [^132] [^130], or inpainting [^138]. Other tasks involve re-colorizing images [^236] or predicting image transformations [^73].

Among these tasks, *inpainting-based* approaches have gathered significant interest thanks to the flexibility of the patch-based ViT architecture [^85] [^8] [^58]. The objective is to reconstruct corrupted regions of an image, which can be viewed as a form of denoising auto-encoding and is conceptually related to the masked token prediction task in BERT pretraining [^50]. Notably, [^85] demonstrated that pixel-based masked auto-encoders (MAE) can be used as strong initializations for finetuning on downstream tasks. In the following, [^5] [^6] [^3] showed that predicting a *learned latent space* instead of the pixel space leads to more powerful, higher-level features—a learning paradigm called JEPA: “Joint-Embedding Predictive Architecture” [^113]. Recently, JEPAs have also been extended to video training [^12] [^4].

A second line of work, closer to ours, leverages *discriminative signals between images* to learn visual representations. This family of methods traces its origins to early deep learning research [^80], but gained popularity with the introduction of instance classification techniques [^54] [^16] [^211]. Subsequent advancements introduced contrastive objectives and information-theoretic criteria [^87] [^84] [^35] [^32] [^79] [^10], as well as self clustering-based strategies [^25] [^2] [^27] [^28]. More recent approaches, such as iBOT [^240], combine these discriminative losses with masked reconstruction objectives. All of these methods show the ability to learn strong features and achieve high performance on standard benchmarks like ImageNet [^155]. However, most face challenges scaling to larger model sizes [^36].

##### Vision Foundation Models

The deep learning revolution began with the AlexNet breakthrough [^110], a deep convolutional neural network that outperformed all previous methods on the ImageNet challenge [^49] [^155]. Already early on, features learned end-to-end on the large manually-labeled ImageNet dataset were found to be highly effective for a wide range of transfer learning tasks [^133]. Early work on vision *foundation models* then focused on architecture development, including VGG [^167], GoogleNet [^175], and ResNets [^83].

Given the effectiveness of *scaling*, subsequent works explored training larger models on big datasets. [^172] expanded supervised training data with the proprietary JFT dataset containing 300 million labeled images, showing impressive results. JFT also enabled significant performance gains for [^107]. In parallel, scaling was explored using a combination of supervised and unsupervised data. For instance, an ImageNet-supervised model can be used to produce pseudo-labels for unsupervised data, which then serve to train larger networks [^220]. Subsequently, the availability of large supervised datasets such as JFT also facilitated the adaptation of the transformer architecture to computer vision [^55]. In particular, achieving performance comparable to that of the original vision transformer (ViT) without access to JFT requires substantial effort [^182] [^183]. Due to the learning capacity of ViTs, scaling efforts were further extended by [^233], culminating in the very large ViT-22B encoder [^48].

Given the complexity of manually labeling large datasets, *weakly-supervised training* —where annotations are derived from metadata associated with images—provides an effective alternative to supervised training. Early on, [^101] demonstrated that a network can be pre-trained by simply predicting all words in the image caption as targets. This initial approach was further refined by leveraging sentence structures [^114], incorporating other types of metadata and involve curation [^124], and scaling [^168]. However, weakly-supervised algorithms only reached their full potential with the introduction of contrastive losses and the joint-training of caption representations, as exemplified by Align [^97] and CLIP [^144].

This highly successful approach inspired numerous *open-source reproductions and scaling efforts*. OpenCLIP [^40] was the first open-source effort to replicate CLIP by training on the LAION dataset [^160]; following works leverage pre-trained backbones by fine-tuning them in a CLIP-style manner [^173] [^174]. Recognizing that data collection is a critical factor in the success of CLIP training, MetaCLIP [^218] precisely follows the original CLIP procedure to reproduce its results, whereas [^63] use supervised datasets to curate pretraining data. Other works focus on improving the training loss, *e.g*. using a sigmoid loss in SigLIP [^235], or leveraging a pre-trained image encoder [^234]. Ultimately though, the most critical components for obtaining cutting-edge foundation models are abundant high-quality data and substantial compute resources. In this vein, SigLIP 2 [^185] and Perception Encoder (PE) [^18] achieve impressive results after training on more than $40$ B image-text pairs. The largest PE model is trained on $86$ B billion samples with a global batch size of $131$ K. Finally, a range of more complex and natively multimodal approaches have been proposed; these include contrastive captioning [^228], masked modeling in the latent space [^8] [^203] [^64] [^201], and auto-regressive training [^68].

In contrast, relatively little work has focused on *scaling unsupervised image pretraining*. Early efforts include [^26] and [^74] utilizing the YFCC dataset [^178]. Further progress has been achieved by focusing on larger datasets and models [^75] [^76], as well as initial attempts at data curation for SSL [^179]. Careful tuning of the training algorithms, larger architectures, and more extensive training data lead to the impressive results of DINOv2 [^134]; for the first time, an SSL model matched or surpassed open-source CLIP variants on a range of tasks. This direction has recently been further pushed by [^62] by scaling to larger models without data curation, or by [^192] using open datasets and improved training recipes.

##### Dense Transformer Features

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/figures/analysis/highres/dog_pca_vit7b/dog.lr.jpeg)

Input

A broad range of modern vision applications consume *dense features* of pre-trained transformers, including multi-modal models [^120] [^14], generative models [^229] [^223], 3D understanding [^199], video understanding [^116] [^207], and robotics [^56] [^104]. On top of that, traditional vision tasks such as detection, segmentation, or depth estimation require accurate local descriptors. To enhance the quality of SSL-trained local descriptors, a substantial body of work focuses on developing *local SSL losses*. Examples include leveraging spatio-temporal consistency in videos, *e.g*. using point track loops as training signal [^94], exploiting the spatial alignment between different crops of the same image [^141] [^11], or enforcing consistency between neighboring patches [^230]. [^47] show that predicting clustered local patches leads to improved dense representations. DetCon [^88] and ORL [^215] perform contrastive learning on region proposals but assume that such proposals exist *a priori*; this assumption is relaxed by approaches such as ODIN [^89] and SlotCon [^210]. Without changing the training objective, [^46] show that adding register tokens to the input sequence greatly improves dense feature maps, and recent works find this can be done without model training [^98] [^37].

A recent trend are distillation-based, “ *agglomerative* ” methods that combine information from multiple image encoders with varying in global and local feature quality, trained using different levels of supervision [^148] [^18]: AM-RADIO [^148] combines the strengths of the fully-supervised SAM [^106], the weakly-supervised CLIP, and the self-supervised DINOv2 into a unified backbone. The Perception Encoder [^18] similarly distills SAM(v2) into a specialized dense variant called PEspatial. They use an objective enforcing cosine similarity between student and teacher patches to be high, where their teacher is trained with mask annotations. Similar losses were shown to be effective in the context of style transfer, by reducing the inconsistency between the Gram matrices of feature dimensions [^71] [^99] [^226]. In this work, we adopt a Gram objective to regularize cosine similarity between student and teacher patches, favoring them being close. In our case, we use earlier iterations of the SSL model itself as the teacher, demonstrating that early-stage SSL models effectively guides SSL training for both global and dense tasks.

Other works focus on post-hoc improvements to the local features of SSL-trained models. For example, [^242] fine-tune a pre-trained model with a dense clustering objective; similarly, [^158] fine-tune by aligning patch features temporally, in both cases enhance the quality of local features. Closer to us, [^135] propose a patch-sorting based objective to encourage the student and teacher to produce features with consistent neighbor ordering. Without finetuning, STEGO [^81] learns a non-linear projection on top of frozen SSL features to form compact clusters and amplify correlation patterns. Alternatively, [^166] augment self-supervised features by concatenating gradients from different self-supervised objectives to frozen SSL features. Recently, [^212] show that noisy feature maps are significantly improved through a weighted average of patches.

Related, but not specific to SSL, some recent works generate high-resolution feature maps from ViT feature maps [^70], which are often low-resolution due to patchification of images. In contrast with this body of work, our models natively deliver high-quality dense feature maps that remain stable and consistent across resolutions, as shown in Fig.˜4.

## 3 Training at Scale Without Supervision

DINOv3 is a next-generation model designed to produce the most robust and flexible visual representations to date by pushing the boundaries of self-supervised learning. We draw inspiration from the success of large language models (LLMs), for which scaling-up the model capacity leads to outstanding *emerging properties*. By leveraging models and training datasets that are an order of magnitude larger, we seek to unlock the full potential of SSL and drive a similar paradigm shift for computer vision, unencumbered by the limitations inherent to traditional supervised or task-specific approaches. In particular, SSL produces rich, high-quality visual features that are not biased toward any specific supervision or task, thereby providing a versatile foundation for a wide range of downstream applications. While previous attempts at scaling SSL models have been hindered by issues of instability, this section describes how we harness the benefits of scaling with careful data preparation, design, and optimization. We first describe the dataset creation procedure (Sec.˜3.1), then present the self-supervised SSL recipe used for this first training phase of DINOv3 (Sec.˜3.2). This includes the choice of architecture, loss functions, and optimization techniques. The second training phase, focusing on dense features, will be described in Sec.˜4.

### 3.1 Data Preparation

Data scaling is one of the driving factors behind the success of large foundation models [^184] [^144] [^218] [^134]. However, increasing naively the size of the training data does not necessarily translate into higher model quality and better performance on downstream benchmarks [^75] [^134] [^196]: Successful data scaling efforts typically involve careful data curation pipelines. These algorithms may have different objectives: either focusing on improving data diversity and balance, or data usefulness—its relevance to common practical applications. For the development of DINOv3, we combine two complementary approaches to improve both the generalizability and performance of the model, striking a balance between the two objectives.

Table 1: Influence of training data on features quality shown via performance on downstream tasks. We compare datasets curated with *clustering* [^196] and *retrieval* [^134] to *raw* data and to our data mixture. This ablation study is run for a shorter schedule of 200k iterations.

| Dataset |  | IN1k k-NN | IN1k Linear | ObjectNet | iNaturalist 2021 | Paris Retrieval |
| --- | --- | --- | --- | --- | --- | --- |
| Raw |  | 80.1 | 84.8 | 70.3 | 70.1 | 63.3 |
| Clustering |  | 79.4 | 85.4 | 72.3 | 81.3 | 85.2 |
| Retrieval |  | 84.0 | 86.7 | 70.7 | 86.0 | 82.7 |
| LVD-1689M (ours) |  | 84.6 | 87.2 | 72.8 | 87.0 | 85.9 |

##### Data Collection and Curation

We build our large-scale pre-training dataset by leveraging a large data pool of web images collected from public posts on Instagram. These images already went through platform-level content moderation to help prevent harmful contents and we obtain an initial data pool of approximately 17 billions of images. Using this raw data pool, we create three dataset *parts*. We construct the first part by applying the automatic curation method based on hierarchical $k$ -means from [^196]. We employ DINOv2 as image embeddings, and use 5 levels of clustering with the number of clusters from the lowest to highest levels being $200$ M, $8$ M, $800$ k, $100$ k, and $25$ k respectively. After building the hierarchy of clusters, we apply the balanced sampling algorithm proposed in [^196]. This results in a curated subset of 1,689 million images (named LVD-1689M) that guarantees a balanced coverage of all visual concepts appearing on the web. For the second part, we adopt a retrieval-based curation system similar to the procedure proposed by [^134]. We retrieve images from the data pool that are similar to those from selected seed datasets, creating a dataset that covers visual concepts relevant for downstream tasks. For the third part, we use raw publicly available computer vision datasets including ImageNet1k [^49], ImageNet22k [^155], and Mapillary Street-level Sequences [^208]. This final part allows us to optimize our model’s performance, following [^134].

##### Data Sampling

During pre-training, we use a sampler to mix different data parts together. There are several different options for mixing the above data components. One is to train with homogeneous batches of data that come from a single, randomly selected component in each iteration. Alternatively, we can optimize the model on heterogeneous batches that are assembled by data from all components, selected using certain ratios. Inspired by [^30], who observed that it is beneficial to have homogeneous batches consisting of very high quality data from a small dataset, we randomly sample in each iteration either a homogeneous batch from ImageNet1k alone or a heterogeneous batch mixing data from all other components. In our training, homogeneous batches from ImageNet1k account for 10% of training.

##### Data Ablation

To assess the impact of our data curation technique, we perform an ablation study to compare our data mix against datasets curated with clustering or retrieval-based methods alone, and the raw data pool. To this end, we train a model on each dataset and compare their performance on standard downstream tasks. For efficiency, we use a shorter schedule of 200k iterations instead of 1M iterations. In Tab.˜1, it can be seen that no single curation technique works best across all benchmarks, and that our full pipeline allows us to obtain the best of both worlds.

### 3.2 Large-Scale Training with Self-Supervision

While models trained with SSL have demonstrated interesting properties [^33] [^28], most SSL algorithms have not been scaled-up to larger models sizes. This is either due to issues with training stability [^47], or overly simplistic solutions that fail to capture the full complexity of the visual world. When trained at scale [^76], models trained with SSL do not necessarily show impressive performance. One notable exception is DINOv2, a model with 1.1 billion parameters trained on curated data, matching the performance of weakly-supervised models like CLIP [^144]. A recent effort to scale DINOv2 to 7 billion parameters [^62] demonstrates promising results on global tasks, but with disappointing results on dense prediction. Here, we aim to scale up the model and data, and obtain even more powerful visual representations with both improved global and local properties.

Table 2: Comparison of the teacher architectures used in DINOv2 and DINOv3 models. We keep the model $40$ blocks deep, and increase the embedding dimension to $4096$. Importantly, we use a patch size of $16$ pixels, changing the effective sequence length for a given resolution.

| Teacher model | DINOv2 | DINOv3 |
| --- | --- | --- |
| Backbone | ViT-giant | ViT-7B |
| #Params | 1.1B | 6.7B |
| #Blocks | 40 | 40 |
| Patch Size | 14 | 16 |
| Pos. Embeddings | Learnable | RoPE |
| Registers | 4 | 4 |
| Embed. Dim. | 1536 | 4096 |
| FFN Type | SwiGLU | SwiGLU |
| FFN Hidden Dim. | 4096 | 8192 |
| Attn. Heads | 24 | 32 |
| Attn. Heads Dim. | 64 | 128 |
| DINO Head MLP | 4096-4096-256 | 8192-8192-512 |
| DINO Prototypes | 128k | 256k |
| iBOT Head MLP | 4096-4096-256 | 8192-8192-384 |
| iBOT Prototypes | 128k | 96k |

##### Learning Objective

We train the model with a discriminative self-supervised strategy which is a mix of several self-supervised objectives with both global and local loss terms. Following DINOv2 [^134], we use an image-level objective [^28] $\mathcal{L_{\mathrm{DINO}}}$, and balance it with a patch-level latent reconstruction objective [^240] $\mathcal{L_{\mathrm{iBOT}}}$. We also replace the centering from DINO with the Sinkhorn-Knopp from SwAV [^27] in both objectives. Each objective is computed using the output of a dedicated head on top of the backbone network, allowing for some specialization of features before the computation of the losses. Additionally, we use a dedicated layer normalization applied to the backbone outputs of the local and global crops. Empirically, we found this change to stabilize ImageNet kNN-classification late in training (+0.2 accuracy) and improve dense performance (*e.g*. +1 mIoU on ADE20k segmentation, -0.02 RMSE on NYUv2 depth estimation). In addition, a Koleo regularizer $\mathcal{L}_{\mathrm{Koleo}}$ is added to encourage the features within a batch to spread uniformly in the space [^157]. We use a distributed implementation of Koleo in which the loss is applied in small batches of $16$ samples—possibly across GPUs. Our initial training phase is carried by optimizing the following loss:

$$
\mathcal{L_{\mathrm{Pre}}}=\mathcal{L_{\mathrm{DINO}}}+\mathcal{L}_{\mathrm{iBOT}}+0.1*\mathcal{L}_{\mathrm{DKoleo}.}
$$

##### Updated Model Architecture

For the model scaling aspect of this work, we increase the size of the model to 7B parameters, and provide in Tab.˜2 a comparison of the corresponding hyperparameters with the 1.1B parameter model trained in the DINOv2 work. We also employ a custom variant of RoPE: our base implementation assigns coordinates in a normalized $[-1,1]$ box to each patch, then applies a bias in the multi-head attention operation depending on the relative position of two patches. In order to improve the robustness of the model to resolutions, scales and aspect ratios, we employ RoPE-box jittering. The coordinate box $[-1,1]$ is randomly scaled to $[-s,s]$, where $s\in[0.5,2]$. Together, these changes enable DINOv3 to better learn detailed and robust visual features, improving its performance and scalability.

##### Optimization

Training large models on very large datasets represents a complicated experimental workflow. Because the interplay between model capacity and training data complexity is hard to assess *a priori*, it is impossible to guess the right optimization horizon. To overcome this, we get rid of all parameter scheduling, and train with constant learning rate, weight decay, and teacher EMA momentum. This has two main benefits. First, we can continue training as long as downstream performance continues to improve. Second, the number of optimization hyperparameters is reduced, making it easier to choose them properly. For the training to start properly, we still use a linear warmup for learning rate and teacher temperature. Following common practices, we use AdamW [^123], and set the total batch size to $4096$ images split across $256$ GPUs. We train our models using the multi-crop strategy [^27], taking $2$ global crops and $8$ local crops per image. We use square images with a side length of $256$ / $112$ pixels for global/local crops, which, along with the change in patch size, results in the same effective sequence length per image as in DINOv2 and a total sequence length of $3.7$ M tokens per batch. Additional hyperparameters can be found in App.˜C and in the code release.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/evolution_cosine_u1/P_20240709_135144.jpg_croped.lr.jpg)

Refer to caption

## 4 Gram Anchoring: A Regularization for Dense Features

To fully leverage the benefits of large-scale training, we aim to train the 7B model for an extended duration, with the notion that it could potentially train indefinitely. As expected, prolonged training leads to improvements on global benchmarks. However, as training progresses, the performance degrades on dense tasks (Figs.˜5 and 5). This phenomenon, which is due to the emergence of patch-level inconsistencies in feature representations, undermines the interest behind extended training.<sup>1</sup> In this section, we first analyze the loss of patch-level consistency, then propose a new objective to mitigate it, called *Gram anchoring*. We finally discuss the impact of our approach on both training stability and model performance.

### 4.1 Loss of Patch-Level Consistency Over Training

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/evolution_cosine_u1/P_20240709_135144.jpg_croped.lr.jpg)

Refer to caption

During extended training, we observe consistent improvements in global metrics but a notable decline in performance on dense prediction tasks. This behavior was previously observed, to a lesser extent, during the training of DINOv2, and also discussed in the scaling effort of [^62]. However, to the best of our knowledge, it remains unresolved to date. We illustrate the phenomenon in Figs.˜5 and 5, which present the performance of the model across iterations on both image classification and segmentation tasks. For classification, we train a linear classifier on ImageNet-1k using the CLS token and report top-1 accuracy. For segmentation, we train a linear layer on patch features extracted from Pascal VOC and report mean Intersection over Union (mIoU). We observe that both for the ViT-g and the ViT-7B, the classification accuracy monotonically improves throughout training. However, segmentation performance declines in both cases after approximately 200k iterations, falling below its early levels in the case of the ViT-7B.

To better understand this degradation, we analyze the quality of patch features by visualizing cosine similarities between patches. Fig.˜6 shows the cosine similarity maps between the backbone’s output patch features and a reference patch (highlighted in red). At 200k iterations, the similarity maps are smooth and well-localized, indicating consistent patch-level representations. However, by 600k iterations and beyond, the maps degrade substantially, with an increasing number of irrelevant patches with high similarity to the reference patch. This loss of patch-level consistency correlates with the drop in dense task performance.

These patch-level irregularities differ from the high-norm patch outliers described in [^46]. Specifically, with the integration of register tokens, patch norms remain stable throughout training. However, we notice that the cosine similarity between the CLS token and the patch outputs gradually increases during training. This is expected, yet it means that the locality of the patch features diminishes. We visualize this phenomenon in Fig.˜5, which depicts the cosine maps at 200k and 1M iterations. In order to mitigate the drop on dense tasks, we propose a new objective specifically designed to regularize the patch features and ensure a good patch-level consistency, while preserving high global performance.

### 4.2 Gram Anchoring Objective

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x8.png)

(a) iBOT loss

Throughout our experiments, we have identified a relative independence between learning strong discriminative features and maintaining local consistency, as observed in the lack of correlation between global and dense performance. While combining the global DINO loss with the local iBOT loss has begun to address this issue, we observe that the balance is unstable, with global representation dominating as training progresses. Building on this insight, we propose a novel solution that explicitly leverages this independence.

We introduce a new objective which mitigates the degradation of patch-level consistency by enforcing the quality of the patch-level consistency, without impacting the features themselves. This new loss function operates on the Gram matrix: the matrix of all pairwise dot products of patch features in an image. We want to push the Gram matrix of the student towards that of an earlier model, referred to as the *Gram teacher*. We select the Gram teacher by taking an early iteration of the teacher network, which exhibits superior dense properties. By operating on the Gram matrix rather than the feature themselves, the local features are free to move, provided the structure of similarities remains the same. Suppose we have an image composed of $P$ patches, and a network that operates in dimension $d$. Let us denote by $\mathbf{X}_{S}$ (respectively $\mathbf{X}_{G}$) the $P\times d$ matrix of $\mathbf{L}_{2}$ -normalized local features of the student (respectively the Gram teacher). We define the loss $\mathcal{L}_{\text{Gram}}$ as follows:

$$
\mathcal{L}_{\text{Gram}}=\left\|\mathbf{X}_{S}\cdot\mathbf{X}_{S}^{\top}-\mathbf{X}_{G}\cdot\mathbf{X}_{G}^{\top}\right\|_{\text{F}}^{2}.
$$

We only compute this loss on the global crops. Even though it can be applied early on during the training, for efficiency, we start only after $1$ M iterations. Interestingly, we observe that the late application of $\mathcal{L}_{\text{Gram}}$ still manages to “repair” very degraded local features. In order to further improve performance, we update the Gram teacher every 10k iterations at which the Gram teacher becomes identical to the main EMA teacher. We call this second step of training the *refinement step*, which optimizes the objective $\mathcal{L}_{\mathrm{Ref}}$, with

$$
\mathcal{L}_{\mathrm{Ref}}=w_{\mathrm{D}}\mathcal{L_{\mathrm{DINO}}}+\mathcal{L}_{\mathrm{iBOT}}+w_{\mathrm{DK}}\mathcal{L}_{\mathrm{DKoleo}}+w_{\mathrm{Gram}}\mathcal{L}_{\mathrm{Gram}}.
$$

We visualize the evolution of different losses in Fig.˜7 and observe that applying the Gram objective significantly influences the iBOT loss, causing it to decrease more rapidly. This suggests that the stability introduced by the stable Gram teacher positively impacts the iBOT objective. In contrast, the Gram objective does not have a significant effect on the DINO losses. This observation implies that the Gram and iBOT objectives impact the features in a similar way, whereas the DINO losses affect them differently.

Regarding performance, we observe the impact of the new loss is almost immediate. As shown in Fig.˜8, incorporating Gram anchoring leads to significant improvements on dense tasks within the first 10k iterations. We also see notable gains on the ADE20k benchmark following the Gram teacher updates. Additionally, longer training further benefits performance on the ObjectNet benchmark and other global benchmarks show mild impact from the new loss.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x11.png)

(a) VOC

### 4.3 Leveraging Higher-Resolution Features

Recent work shows that a weighted average of patch features can yield stronger local representations by smoothing outlier patches and enhancing patch-level consistency [^212]. On the other hand, feeding higher-resolution images into the backbone produces finer and more detailed feature maps. We leverage the benefits of both observations to compute high-quality features for Gram teacher. Specifically, we first input images at twice the normal resolution into the Gram teacher, then $2\times$ down-sample the resulting feature maps with the bicubic interpolation to achieve the desired smooth feature maps that match the size of the student output. Fig.˜9(a) visualizes the Gram matrices of patch features obtained with images at resolutions 256 and 512, as well as those obtained after $2\times$ down-sampling features from the 512-resolution (denoted as ‘downsamp.’). We observe that the superior patch-level consistency in the higher-resolution features is preserved through down-sampling, resulting in smoother and more coherent patch-level representations. As a side note, our model can seamlessly process images at varying resolutions without requiring adaptation, thanks to the adoption of Rotary Positional Embeddings (RoPE) introduced by [^171].

We compute the Gram matrix of the down-sampled features and use it to replace $\mathbf{X}_{G}$ in the objective $\mathcal{L_{\mathrm{Gram}}}$. We note the new resulting refinement objective as $\mathcal{L}_{\mathrm{HRef}}$. This approach enables the Gram objective to effectively distill the improved patch consistency of smoothed high-resolution features into the student model. As shown in Fig.˜8 and Fig.˜9(b), this distillation translates into better predictions on dense tasks, yielding additional gains on top of the benefit brought by $\mathcal{L}_{\mathrm{Ref}}$ (+2 mIoU on ADE20k). We also ablate the choice of Gram teacher in Fig.˜9(b). Interestingly, choosing the Gram teacher from 100k or 200k does not significantly impact the results, but using a much later Gram teacher (1M iterations) is detrimental because the patch-level consistency of such a teacher is inferior.

Finally, we qualitatively illustrate the effect of Gram anchoring to patch-level consistency in Fig.˜10 which visualizes the Gram matrices patch features obtained with the initial training and high-resolution Gram anchoring refinement. We observe great improvements in feature correlations that our high-resolution refinement procedure brings about.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/gram/comparison_res/IMG_0877.HEIC_croped.lr.jpg)

Refer to caption

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/gram/comparison_w_wo/P_20250105_134132.jpg_croped.lr.jpg)

Refer to caption

## 5 Post-Training

This section presents *post-training* stages. This includes a high-resolution adaptation phase enabling effective inference at different input resolutions (Sec.˜5.1), model distillation producing quality and efficient smaller-sized models (Sec.˜5.2), and text alignment adding zero-shot capabilities to DINOv3 (Sec.˜5.3).

### 5.1 Resolution Scaling

We train our model at a relatively small resolution of $256$, which gives us a good trade-off between speed and effectiveness. For a patch size of $16$, this setup leads to the same input sequence length as DINOv2, which was trained with resolution $224$ and patch size $14$. However, many contemporary computer vision applications require processing images at significantly higher resolutions, often $512\times 512$ pixels or greater, to capture intricate spatial information. The inference image resolution is also not fixed in practice and varies depending on specific use cases. To address this, we extend our training regime with a high-resolution adaptation step [^181]. To ensure high performance across a range of resolutions, we utilize *mixed resolutions*, sampling differently-sized pairs of global and local crops per mini-batch. Specifically, we consider global crop sizes from $\{512,768\}$ and local crop sizes from $\{112,168,224,336\}$ and train the model for 10k additional iterations.

Similar to the main training, a key component of this high-resolution adaptation phase is the addition of Gram anchoring, using the 7B teacher as Gram teacher. We found this component to be essential: without it, the model performance on dense prediction tasks degrades significantly. The Gram anchoring encourages the model to maintain consistent and robust feature correlations across spatial locations, which is crucial when dealing with the increased complexity of high-resolution inputs.

Empirically, we observe that this relatively brief but targeted high-resolution step substantially enhances the overall model’s quality and allows it to generalize across a wide range of input sizes, as shown visually in Fig.˜4. In Fig.˜11, we compare our 7B model before and after adaptation. We find that resolution scaling leads to a small gain on ImageNet classification (a) with relatively stable performance w.r.t. resolution. However, in ObjectNet OOD transfer (b), we observe that the performance tends to degrade slightly for lower resolutions, while improving for higher resolutions. This is largely compensated by the improvement in the quality of local features at high resolution, shown by the positive trend in segmentation on ADE20k (c) and tracking on DAVIS (d). Adaptation leads to local features that *improve with image size*, leveraging the richer spatial information available at larger resolutions and effectively enabling high-resolution inference. Interestingly, the adapted model supports resolutions way beyond the maximum training resolution of 768—we visually observe stable feature maps at resolutions above 4k (*c.f*. Fig.˜4).

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x14.png)

(a) IN1k linear

### 5.2 Model Distillation

##### A Family of Models for Multiple Use-Cases

We perform knowledge distillation of the ViT-7B model into smaller Vision Transformer variants (ViT-S, ViT-B, and ViT-L), which are highly valued by the community for their improved manageability and efficiency. Our distillation approach uses the same training objective as in the first training phase, ensuring consistency in learning signals. However, instead of relying on an exponential moving average (EMA) of model weights, we use the 7B model directly as the teacher to guide the smaller student models. In this case, the teacher model is fixed. We do not observe patch-level consistency issues and therefore do not apply the Gram anchoring technique. This strategy enables the distilled models to inherit the rich representational power of the large teacher while being more practical for deployment and experimentation.

Our ViT-7B model is distilled into a series of ViT models with sizes covering a broad range of compute budgets, and allowing proper comparison with concurrent models. They include the standard ViT-S (21M params), B (86M), L (0.3B), along with a custom ViT-S+ (29M) and a custom ViT-H+ (0.8B) model to close the performance gap with the self-distilled 7B teacher model. Indeed, we observe in DINOv2 that smaller student models can reach a performance on par with their teacher as the distillation. As a result, the distilled models deliver frontier-level performance for a fraction of the inference compute as we see in Tab.˜14. We train the models for 1M iterations then perform 250k iterations of learning-rate cooldown following a cosine schedule before applying the high-resolution phase described in Sec.˜5.1 above without Gram anchoring.

##### Efficient Multi-Student Distillation

As the inference cost for a large teacher can be orders of magnitude higher than for students (see Fig.˜16(a)), we design a parallel distillation pipeline that allows training multiple students at the same time and sharing the teacher inference across all nodes involved in the training (see Fig.˜12 for a diagram). Let $C_{T}$ and $C_{S}$ be respectively the cost of running the teacher inference and the student training on a single sample, in single-teacher/single-student distillation with batch-size $B$ where each of the $N$ GPUs processes a $B/N$ slice of the data, the teacher inference costs $B/N\times C_{T}$ and the student training costs $B/N\times C_{S}$ per GPU. In multi-student distillation, we proceed as follows. Each student $Si$ is assigned a set of $N_{Si}$ GPUs for training, and all $N_{T}=\sum{N_{Si}}$ GPUs are part of the global inference group. At each iteration, we first run the teacher inference on the global group for a $B/N_{T}\times C_{T}$ compute cost per GPU. We then run an all-gather collective operation to share the input data and inference result with all compute nodes. Finally, each student group separately performs student training for a $B/N_{Si}\times C_{Si}$ cost.

The above calculations shows that adding an additional student to the distillation pipeline will (1) reduce the per-GPU compute at each iteration, thus globally improving distillation speed, and (2) increase the overall compute only by the training cost of the new student, since the total teacher inference cost is now fixed. The implementation only requires setting up GPU process groups carefully, adapting data-loaders and teacher inference to ensure inputs and outputs are synchronized across groups using NCCL collectives. As the groups are synchronized at each iteration, in order to maximize speed, we adapt the number of GPUs for each student such that their iteration times are roughly the same. With this procedure, we seamlessly train multiple students, and produce a whole family of distilled models from our flagship 7B model.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x18.png)

Figure 12: Multi-student distillation procedure. In this diagram, we distill 3 students in parallel: we first share teacher inference across all T nodes to save compute, and gather inputs and results on all GPUs. Then, smaller groups perform student training. We adjust the size of these groups such that the training step has the same duration across all students S i Si, minimizing idle time waiting at the synchronization barrier.

### 5.3 Aligning DINOv3 with Text

Open-vocabulary image-text alignment has received significant interest and enthusiasm from the research community, thanks to its potential to enable flexible and scalable multimodal understanding. A large body of work has focused on improving the quality of CLIP [^144], which originally learned only a global alignment between image and text representations. While CLIP has demonstrated impressive zero-shot capabilities, its focus on global features limits its ability to capture fine-grained, localized correspondences. More recent works [^234] have shown that effective image-text alignment can be achieved with pre-trained self-supervised visual backbones. This makes it possible to leverage these powerful models in multi-modal settings, facilitating richer and more precise text-to-image associations that extend beyond global semantics while also reducing computational costs, since the visual encoding is already learned.

We align a text encoder with our DINOv3 model by adopting the training strategy previously proposed in [^100]. This approach follows the LiT training paradigm [^234], training a text representation from scratch to match images to their captions with a contrastive objective, while keeping the vision encoder frozen. To allow for some flexibility on the vision side, two transformer layers are introduced on top of the frozen visual backbone. A key enhancement of this method is the concatenation of the mean-pooled patch embeddings with the output CLS token before matching to the text embeddings. This enables aligning both global and local visual features to text, leading to improved performance on dense prediction tasks without requiring additional heuristics or tricks. Furthermore, we use to the same data curation protocol as established in [^100] to ensure consistency and comparability.

## 6 Results

In this section, we evaluate our flagship DINOv3 7B model on a variety of computer vision tasks. Throughout our experiments, unless otherwise specified, *we keep DINOv3 frozen* and solely use its representations. We demonstrate that with DINOv3, finetuning is not necessary to obtain strong performance. This section is organized as follows. We first probe the quality of DINOv3’s dense (Sec.˜6.1) and global (Sec.˜6.2) image representations using lightweight evaluation protocols and compare it to the strongest available vision encoders. We show that DINOv3 learns exceptional dense features while offering robust and versatile global image representations. Then, we consider DINOv3 as a basis for developing more complex computer vision systems (Sec.˜6.3). We show with little effort on top of DINOv3, we are able to achieve results competitive with or exceeding the state of the art in tasks as diverse as object detection, semantic segmentation, 3D view estimation, or relative monocular depth estimation.

### 6.1 DINOv3 provides Exceptional Dense Features

We first investigate the raw quality of DINOv3’s dense representations using a diverse set of lightweight evaluations. In all cases, we utilize the frozen patch features of the last layer, and evaluate them using (1) qualitative visualizations (Sec.˜6.1.1), (2) dense linear probing (Sec.˜6.1.2: segmentation, depth estimation), (3) non-parametric approaches (Sec.˜6.1.3: 3D correspondence estimation, Sec.˜6.1.4: object discovery, Sec.˜6.1.5: tracking), and (4) lightweight attentive probing (Sec.˜6.1.6: video classification).

##### Baselines

We compare the dense features of DINOv3 with those of the strongest publicly available image encoders, both weakly- and self-supervised ones. We consider the weakly-supervised encoders Perception Encoder (PE) Core [^18] and SigLIP 2 [^185], which use CLIP-style image-text contrastive learning. We also compare to the strongest self-supervised methods: DINOv3’s predecessor DINOv2 [^134] with registers [^46], Web-DINO [^62], a recent scaling effort of DINO, and Franca [^192], the best open-data SSL model. Finally, we compare to the agglomerative models AM-RADIOv2.5 [^86], distilled from DINOv2, CLIP [^144], DFN [^63], and Segment Anything (SAM) [^106], and to PEspatial, distilling SAM 2 [^149] into PEcore. For each baseline, we report the performance of the strongest model available and specify the architecture in the tables.

#### 6.1.1 Qualitative Analysis

We start by analyzing DINOv3’s dense feature maps qualitatively. To this end, we project the dense feature space into 3 dimensions using principal component analysis (PCA), and map the resulting 3D space into RGB. Because of the sign ambiguity in PCA (eight variants) and the arbitrary mapping between principal components and colors (six variants), we explore all combinations and report the visually most compelling one. The resulting visualization is shown in Fig.˜13. Compared to other vision backbones, it can be seen that the features of DINOv3 are sharper, containing much less noise, and showing superior semantical coherence.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/pca_comparison/img_3.lr.jpg)

Refer to caption

#### 6.1.2 Dense Linear Probing

We perform linear probing on top of the dense features for two tasks: semantic segmentation and monocular depth estimation. In both cases, we train a linear transform on top of the frozen patch outputs of DINOv3. For semantic segmentation, we evaluate on the ADE20k [^239], Cityscapes [^44], and PASCAL VOC 2012 [^60] datasets and report the mean intersection-over-union (mIoU) metric. For depth estimation, we use the NYUv2 [^163] and KITTI [^72] datasets and report the root mean squared error (RMSE).

Table 3: Dense linear probing results on semantic segmentation and monocular depth estimation with frozen backbones. We report the mean Intersection-over-Union (mIoU) metric for the segmentation benchmarks ADE20k, Cityscapes, and VOC. We report the Root Mean Squared Error (RMSE) metric for the depth benchmarks NYUv2 and KITTI. For segmentation, all models are evaluated with input resolution adapted to 1024 patch tokens (*i.e*. $448\times 448$ for patch size 14, $512\times 512$ for patch size 16).

<table><tbody><tr><td></td><td></td><td></td><td colspan="3">Segmentation</td><td></td><td colspan="2">Depth</td></tr><tr><td>Method</td><td>ViT</td><td></td><td>ADE20k</td><td>Citysc.</td><td>VOC</td><td></td><td>NYUv2 <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>KITTI <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td colspan="2">Agglomerative backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AM-RADIOv2.5</td><td>g/14</td><td></td><td>53.0</td><td>78.4</td><td>85.4</td><td></td><td>0.340</td><td>2.918</td></tr><tr><td>PEspatial</td><td>G/14</td><td></td><td>49.3</td><td>73.2</td><td>82.7</td><td></td><td>0.362</td><td>3.082</td></tr><tr><td colspan="2">Weakly-supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP 2</td><td>g/16</td><td></td><td>42.7</td><td>64.8</td><td>72.7</td><td></td><td>0.494</td><td>3.273</td></tr><tr><td>PEcore</td><td>G/14</td><td></td><td>38.9</td><td>61.1</td><td>69.2</td><td></td><td>0.590</td><td>4.119</td></tr><tr><td colspan="2">Self-supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Franca</td><td>g/14</td><td></td><td>46.3</td><td>68.7</td><td>82.9</td><td></td><td>0.445</td><td>3.140</td></tr><tr><td>DINOv2</td><td>g/14</td><td></td><td>49.5</td><td>75.6</td><td>83.1</td><td></td><td>0.372</td><td>2.624</td></tr><tr><td>Web-DINO</td><td>7B/14</td><td></td><td>42.7</td><td>68.3</td><td>76.1</td><td></td><td>0.466</td><td>3.158</td></tr><tr><td>DINOv3</td><td>7B/16</td><td></td><td>55.9</td><td>81.1</td><td>86.6</td><td></td><td>0.309</td><td>2.346</td></tr></tbody></table>

##### Results ()

The segmentation results demonstrate the superior quality of our dense features. On the general ADE20k dataset, DINOv3 outperforms the self-supervised baselines by more than 6 mIoU points, and the weakly supervised baselines by more than 13 points. Furthermore, DINOv3 surpasses PEspatial by more than 6 points, and AM-RADIOv2.5 by nearly 3 points. These results are remarkable as both are strong baselines, being distilled from the heavily supervised segmentation model SAM [^106]. Similar results are observed on the self-driving benchmark Cityscapes, with DINOv3 achieving the best mIoU of $81.1$, surpassing AM-RADIOv2.5 by 2.5 points, and all other backbones by at least 5.5 points.

On monocular depth estimation, DINOv3 again outperforms all other models by significant margins: the weakly-supervised models PEcore and SigLIP 2 are still lagging, with DINOv2 and the more advanced models derived from SAM are the closest competitors. Interestingly, while PEspatial and AM-RADIO show strong performance on NYU, their performance is lower than DINOv2’s on KITTI. Even there, DINOv3 outperforms its predecessor DINOv2 by 0.278 RMSE.

Both sets of evaluations show the outstanding representation power of the dense features of DINOv3 and reflect the visual results from Fig.˜13. With only a linear predictor, DINOv3 allows robust prediction of object categories and masks, as well as physical measurements of the scene such as relative depth. These results show that the features are not only visually sharp and properly localized, they also represent many important properties of the underlying observations in a linearly separable way. Finally, the absolute performance obtained with a linear classifier on ADE20k (55.9 mIoU) is itself impressive, as it is not far from the absolute the state-of-the-art (63.0 mIoU) on this dataset.

#### 6.1.3 3D Correspondence Estimation

Understanding the 3D world has always been an important goal of computer vision Image foundation models have recently fueled research in 3D understanding by offering *3D-aware features*. In this section, we evaluate the *multi-view consistency* of DINOv3—that is, whether patch features of the same keypoint in different views of an object are similar—following the protocol defined in Probe3D [^7]. We distinguish between *geometric* and *semantic* correspondence estimation. The former refers to matching keypoints for the *same object instance* while the latter refers to matching keypoints for different instances of the *same object class*. We evaluate geometric correspondence on the NAVI dataset [^95] and semantic correspondence on the SPair dataset [^129], and measure performance with correspondence recall in both cases. Please refer to Sec.˜D.3 for more experimental details.

Table 4: Evaluation of 3D consistency of dense representations. We estimate 3D keypoint correspondences across views following the evaluation protocol of Probe3D [^7]. To measure performance, we report the correspondence recall, *i.e*. the percentage of correspondences falling into a specified distance.

<table><tbody><tr><td></td><td></td><td></td><td>Geometric</td><td></td><td>Semantic</td></tr><tr><td>Method</td><td>ViT</td><td></td><td>NAVI</td><td></td><td>SPair</td></tr><tr><td colspan="2">Agglomerative backbones</td><td></td><td></td><td></td><td></td></tr><tr><td>AM-RADIOv2.5</td><td>g/14</td><td></td><td>59.4</td><td></td><td>56.8</td></tr><tr><td>PEspatial</td><td>G/14</td><td></td><td>53.8</td><td></td><td>49.6</td></tr><tr><td colspan="2">Weakly-supervised backbones</td><td></td><td></td><td></td><td></td></tr><tr><td>SigLIP 2</td><td>g/16</td><td></td><td>49.4</td><td></td><td>42.6</td></tr><tr><td>PEcore</td><td>G/14</td><td></td><td>39.9</td><td></td><td>23.1</td></tr><tr><td colspan="2">Self-supervised backbones</td><td></td><td></td><td></td><td></td></tr><tr><td>Franca</td><td>g/14</td><td></td><td>54.6</td><td></td><td>51.0</td></tr><tr><td>DINOv2</td><td>g/14</td><td></td><td>60.1</td><td></td><td>56.1</td></tr><tr><td>Web-DINO</td><td>7B/14</td><td></td><td>55.0</td><td></td><td>32.2</td></tr><tr><td>DINOv3</td><td>7B/16</td><td></td><td>64.4</td><td></td><td>58.7</td></tr></tbody></table>

##### Results ()

For geometric correspondences, DINOv3 outperforms all other models and improves over the second best model (DINOv2) by 4.3% recall. Other SSL scaling endeavors (Franca and WebSSL) lag behind DINOv2, showing that it is still a strong baseline. Weakly-supervised models (PEcore and SigLIP 2) do not fare well on this task, indicating a lack of 3D awareness. For models with SAM distillation, AM-RADIO nearly reaches the performance of DINOv2, but PEspatial still lags behind it ($-11.6$ % recall), and even falls behind Franca ($-0.8$ % recall). This suggests that self-supervised learning is a key component for strong performance on this task. For semantic correspondences, the same conclusions apply. DINOv3 performs best, outperforming both its predecessor ($+2.6$ % recall) and AM-RADIO ($+1.9$ % recall). Overall, these impressive performance on keypoint matching are very promising signals for downstream use of DINOv3 in other 3D-heavy applications.

#### 6.1.4 Unsupervised Object Discovery

Powerful self-supervised features facilitate discovering object instances in images without requiring *any* annotations [^195] [^164] [^161] [^206] [^165]. We test this capability for different vision encoders via the task of unsupervised object discovery, which requires class-agnostic segmentation of objects in images [^156] [^187] [^41] [^193]. In particular, we use the non-parametric graph-based TokenCut algorithm [^206], which has shown strong performance on a variety of backbones. We run it on three widely used datasets: VOC 2007, VOC 2012 [^61], and COCO-20k [^117] [^194]. We follow the evaluation protocol defined by [^164] and report the CorLoc metric. To properly compare backbones with different feature distributions, we perform a search over the main TokenCut hyperparameter, namely the cosine similarity threshold applied when constructing the patch graph used for partitioning. Originally, the best object discovery results were obtained with DINO [^28] using the keys of the last attention layer. However, this hand-crafted choice does not consistently generalize to other backbones. For simplicity, we always employ the output features for all models.

##### Results ()

The original DINO has set a very high bar for this task. Interestingly, while DINOv2 has shown very strong performance for pixel-wise dense tasks, it fails at object discovery. This can in part be attributed to the artifacts present in the dense features (*c.f*. Fig.˜13). DINOv3, with its clean and precise output feature maps outperforms both its predecessors, with a $5.9$ CorLoc improvement on VOC 2007, and all other backbones, whether self-, weakly-supervised or agglomerative. This evaluation confirms that DINOv3’s dense features are both semantically strong and well localized. We believe that this will pave the way for more class-agnostic object detection approaches, especially in scenarios where annotations are costly or unavailable, and where the set of relevant classes is not confined to a predefined subset.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x19.jpg)

Refer to caption

#### 6.1.5 Video Segmentation Tracking

Beyond static images, an important property of visual representations is their *temporal consistency*, *i.e*. whether the features evolve in a stable manner through time. To test for this property, we evaluate DINOv3 on the task of video segmentation tracking: given ground-truth instance segmentation masks in the first frame of a video, the goal is to propagate these masks to subsequent frames. We use the DAVIS 2017 [^142], YouTube-VOS [^219], and MOSE [^52] datasets. We evaluate performance using the standard $\mathcal{J}\&\mathcal{F}$ -mean metric, which combines region similarity ($\mathcal{J}$) and contour accuracy ($\mathcal{F}$) [^139]. Following [^94], we use a non-parametric label propagation algorithm that considers the similarity between patch features across frames. We evaluate at three input resolutions, using a short side length of 420/480 (S), 840/960 (M), and 1260/1440 (L) pixels for models with patch size 14/16 (matching the number of patch tokens). The $\mathcal{J}\&\mathcal{F}$ score is always computed at the native resolution of the videos. See Sec.˜D.5 for more detailed experimental settings.

Table 5: Video segmentation tracking evaluation. We report the $\mathcal{J}\&\mathcal{F}$ -mean on DAVIS, YouTube-VOS, and MOSE at multiple resolutions. For models with patch size 14/16, the small, medium and large resolutions correspond to a video short side of 420/480, 840/960, 1260/1140 pixels.

<table><tbody><tr><th></th><th></th><td colspan="3">DAVIS</td><td colspan="3">YouTube-VOS</td><td colspan="3">MOSE</td></tr><tr><th>Method</th><th>ViT</th><td>S</td><td>M</td><td>L</td><td>S</td><td>M</td><td>L</td><td>S</td><td>M</td><td>L</td></tr><tr><th colspan="2">Agglomerative backbones</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>AM-RADIOv2.5</th><th>g/14</th><td>66.5</td><td>77.3</td><td>81.4</td><td>70.1</td><td>78.1</td><td>79.2</td><td>44.0</td><td>52.6</td><td>54.3</td></tr><tr><th>PEspatial</th><th>G/14</th><td>68.4</td><td>74.5</td><td>70.5</td><td>68.5</td><td>67.5</td><td>55.6</td><td>39.3</td><td>40.2</td><td>34.0</td></tr><tr><th colspan="2">Weakly-supervised backbones</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>SigLIP 2</th><th>g/16</th><td>56.1</td><td>62.3</td><td>62.9</td><td>52.0</td><td>57.3</td><td>55.1</td><td>28.0</td><td>30.3</td><td>29.2</td></tr><tr><th>PEcore</th><th>G/14</th><td>48.2</td><td>53.1</td><td>49.8</td><td>34.7</td><td>33.0</td><td>25.3</td><td>17.8</td><td>19.0</td><td>15.4</td></tr><tr><th colspan="2">Self-supervised backbones</th><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>Franca</th><th>g/14</th><td>61.8</td><td>66.9</td><td>66.5</td><td>67.3</td><td>70.5</td><td>67.9</td><td>40.3</td><td>42.6</td><td>41.9</td></tr><tr><th>DINOv2</th><th>g/14</th><td>63.9</td><td>73.6</td><td>76.6</td><td>65.6</td><td>73.5</td><td>74.6</td><td>40.4</td><td>47.6</td><td>48.5</td></tr><tr><th>Web-DINO</th><th>7B/14</th><td>57.2</td><td>65.8</td><td>69.5</td><td>43.9</td><td>49.6</td><td>50.9</td><td>24.9</td><td>29.9</td><td>31.1</td></tr><tr><th>DINOv3</th><th>7B/16</th><td>71.1</td><td>79.7</td><td>83.3</td><td>74.1</td><td>80.2</td><td>80.7</td><td>46.0</td><td>53.9</td><td>55.6</td></tr></tbody></table>

##### Results ()

Aligned with all previous results, weakly-supervised backbones do not deliver convincing performance. PEspatial, distilled from the video model SAMv2, provides satisfactory performance, surpassing DINOv2 on smaller resolutions, but falling short on larger ones. Across resolutions, DINOv3 outperforms all competitors, with a staggering $83.3$ $\mathcal{J}\&\mathcal{F}$ on DAVIS-L, $6.7$ points above DINOv2. Furthermore, performance as a function of resolution follows a healthy trend, confirming that our model is able to make use of more input pixels to output precise, high-resolution feature maps (*c.f*. Figs.˜3 and 4). In contrast, performance at higher resolutions stays almost flat for SigLIP 2 and PEcore, and degrades for PEspatial. Interestingly, our image model, without any tuning on video, allows to properly track objects in time (see Fig.˜15). This makes it a great candidate to embed videos, allowing to build strong video models on top.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/images/segmentation_tracking/ducks/rgb_frames/000001.jpg)

Initial frame

#### 6.1.6 Video Classification

The previous results have shown the low-level temporal consistency of DINOv3’s representations, allowing to accurately track objects in time. Going beyond, we evaluate in this section the suitability of its dense features for high-level video classification. Similar to the setup of V-JEPA 2 [^4], we train an *attentive probe* —a shallow 4-layer transformer-based classifier—on top of patch features extracted from each frame. This enables reasoning over temporal and spatial dimensions as the features are extracted independently per frame. During evaluation, we either take a single clip per video, or use test-time augmentation (TTA) by averaging the predictions of 3 spatial and 2 temporal crops per video. See Sec.˜D.6 for experimental details. We run this evaluation on three datasets: UCF101 [^169], Something-Something V2 [^78], and Kinetics-400 [^102], and report top-1 accuracy. As an additional baseline, we report the performance of V-JEPA v2, a state-of-the-art SSL model for video understanding.

Table 6: Video classification evaluation using attentive probes. We report top-1 accuracy on UCF101, Something-Something V2 (SSv2), and Kinetics-400 (K400). For each model, we report performance for evaluating a single clip per video, or applying test-time augmentation (TTA) by averaging the predicted probabilities from multiple clips.

<table><tbody><tr><th></th><th></th><td colspan="2">UCF101</td><td colspan="2">SSv2</td><td colspan="2">K400</td></tr><tr><th>Method</th><th>ViT</th><td>Single</td><td>TTA</td><td>Single</td><td>TTA</td><td>Single</td><td>TTA</td></tr><tr><th colspan="2">Agglomerative backbones</th><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>AM-RADIOv2.5</th><th>g/14</th><td>92.8</td><td>92.5</td><td>69.1</td><td>70.0</td><td>84.8</td><td>85.2</td></tr><tr><th>PEspatial</th><th>G/14</th><td>92.7</td><td>92.8</td><td>66.4</td><td>68.4</td><td>83.5</td><td>84.8</td></tr><tr><th colspan="2">Weakly-supervised backbones</th><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>SigLIP 2</th><th>g/16</th><td>93.6</td><td>94.2</td><td>68.8</td><td>70.2</td><td>86.9</td><td>87.7</td></tr><tr><th>PEcore</th><th>G/14</th><td>93.1</td><td>93.3</td><td>69.0</td><td>70.4</td><td>87.9</td><td>88.8</td></tr><tr><th colspan="2">Self-supervised backbones</th><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><th>DINOv2</th><th>g/14</th><td>93.5</td><td>93.8</td><td>67.4</td><td>68.4</td><td>84.4</td><td>85.6</td></tr><tr><th>V-JEPA 2</th><th>g/16</th><td>94.0</td><td>93.8</td><td>73.8</td><td>75.4</td><td>83.3</td><td>84.3</td></tr><tr><th>Web-DINO</th><th>7B/14</th><td>93.9</td><td>94.1</td><td>67.3</td><td>68.1</td><td>86.8</td><td>87.2</td></tr><tr><th>DINOv3</th><th>7B/16</th><td>93.5</td><td>93.5</td><td>70.1</td><td>70.8</td><td>87.8</td><td>88.2</td></tr></tbody></table>

##### Results ()

In line with the conclusion of the previous experiment, we find that DINOv3 can be successfully used for extracting strong video features. As this evaluation involves training several layers of self-attention, the differences between models are less visible. However, DINOv3 lands in the same range as PEcore and SigLIP 2, and clearly outperforms other models (DINOv2, AM-RADIO) across datasets. UCF101 and K400 are appearance-focused, where strong category-level understanding of objects gives most of the performance. SSv2 on the other hand, requires better understanding of motion—the dedicated video model V-JEPA v2 shines on this dataset. Interestingly, the gap between DINOv3 and the weakly-supervised models is slightly bigger on this dataset. This again confirms the suitability of DINOv3 to video tasks.

### 6.2 DINOv3 has Robust and Versatile Global Image Descriptors

In this section, we evaluate DINOv3’s ability to capture global image statistics. To this end, we consider classic classification benchmarks using linear probes (Sec.˜6.2.1) and instance retrieval benchmarks (Sec.˜6.2.2). Again, we compare to the strongest publicly available image encoders. In addition to the models from the previous section, we evaluate the two weakly supervised models AIMv2 [^68], trained using joint auto-regressive pixel and text prediction, and the massive EVA-CLIP-18B [^174].

#### 6.2.1 Image Classification with Linear Probing

We train a linear classifier on top of DINOv3’s output CLS token to evaluate the model on classification benchmarks. We consider the ImageNet1k [^49] dataset and its variants to evaluate out-of-distribution robustness, and a suite of datasets from different domains to understand DINOv3’s ability to distinguish fine-grained classes. See Sec.˜D.7 for evaluation details.

##### Domain Generalization from ImageNet ()

In this experiment, we train on ImageNet- *train*, use ImageNet- *val* as a *validation set* to select hyperparameters, and transfer the best found classifier to different test datasets: ImageNet-V2 [^150] and ReaL [^13] are alternative sets of images and labels for ImageNet, used to test overfitting on the ImageNet validation set; Rendition [^91] and Sketch [^204] show stylized and artificial versions of the ImageNet classes; Adversarial [^92] and ObjectNet [^9] contain deliberately-chosen difficult examples; Corruptions [^90] measures robustness to common image corruptions. For reference, we also list linear probing results from [^48] for ViTs trained using supervised classification on the massive JFT dataset (3B–4B images). Note that these results follow a slightly different evaluation protocol and are not directly comparable to our results.

DINOv3 significantly surpasses all previous self-supervised backbones, with gains of +10% on ImageNet-R, +6% on -Sketch, +13% on ObjectNet over the previously strongest SSL model DINOv2. We note that the strongest weakly-supervised models, SigLIP 2 and PE, are now better than the strongest supervised ones (ViT-22B) on hard OOD tasks like ImageNet-A and ObjectNet. DINOv3 reaches comparable results on ImageNet-R and -Sketch, and, on the hard tasks ImageNet-A and ObjectNet, is closely behind PE, while exceeding SigLIPv2. On ImageNet, while validation scores are 0.7–0.9 points behind SigLIPv2 and PE, the performance on the “cleaner” test sets -V2 and -ReaL is virtually the same. Notably, DINOv3 achieves the best robustness to corruptions (ImageNet-C). All in all, *this is the first time that a SSL model has reached comparable results to weakly- and supervised models on image classification* —a domain which used to be the strong point of (weakly-)supervised training approaches. This is a remarkable result, given that models like ViT-22B, SigLIP 2, and PE are trained using massive human-annotated datasets. In contrast, DINOv3 learns purely from images, which makes it feasible to further scale/improve the approach in the future.

Table 7: Classification accuracy of linear probes trained on ImageNet1k with frozen backbones. Weakly- and self-supervised models are evaluated with image resolution adapted to 1024 patch tokens (*i.e*. $448\times 448$ for patch size 14, $512\times 512$ for patch size 16). For reference, we also list results from [^48] using a different evaluation protocol (marked with <sup>∗</sup>).

<table><tbody><tr><td></td><td></td><td></td><td colspan="3">ImageNet</td><td></td><td colspan="2">Rendition</td><td></td><td colspan="3">Hard</td></tr><tr><td>Method</td><td>ViT</td><td></td><td>Val</td><td>V2</td><td>ReaL</td><td></td><td>R</td><td>S</td><td></td><td>A</td><td>C <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>Obj.</td></tr><tr><td colspan="2">Supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td><sup><a href="#fn:233">233</a></sup> <sup>∗</sup></td><td>G/14</td><td></td><td>89.0</td><td>81.3</td><td>90.6</td><td></td><td>91.7</td><td>—</td><td></td><td>78.8</td><td>—</td><td>69.6</td></tr><tr><td><sup><a href="#fn:34">34</a></sup> <sup>∗</sup></td><td>e/14</td><td></td><td>89.3</td><td>82.5</td><td>90.7</td><td></td><td>94.3</td><td>—</td><td></td><td>81.6</td><td>—</td><td>71.5</td></tr><tr><td><sup><a href="#fn:48">48</a></sup> <sup>∗</sup></td><td>22B/14</td><td></td><td>89.5</td><td>83.2</td><td>90.9</td><td></td><td>94.3</td><td>—</td><td></td><td>83.8</td><td>—</td><td>74.3</td></tr><tr><td colspan="2">Agglomerative backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AM-RADIOv2.5</td><td>g/14</td><td></td><td>88.0</td><td>80.2</td><td>90.3</td><td></td><td>83.8</td><td>67.1</td><td></td><td>81.3</td><td>27.1</td><td>68.4</td></tr><tr><td colspan="2">Weakly-supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PEcore</td><td>G/14</td><td></td><td>89.3</td><td>81.6</td><td>90.4</td><td></td><td>92.2</td><td>71.9</td><td></td><td>89.0</td><td>22.7</td><td>80.2</td></tr><tr><td>SigLIP 2</td><td>g/16</td><td></td><td>89.1</td><td>81.6</td><td>90.5</td><td></td><td>92.2</td><td>71.8</td><td></td><td>84.6</td><td>30.0</td><td>78.6</td></tr><tr><td>AIMv2</td><td>3B/14</td><td></td><td>87.9</td><td>79.5</td><td>89.7</td><td></td><td>82.3</td><td>67.1</td><td></td><td>74.5</td><td>29.5</td><td>69.0</td></tr><tr><td>EVA-CLIP</td><td>18B/14</td><td></td><td>87.9</td><td>79.3</td><td>89.5</td><td></td><td>85.2</td><td>64.0</td><td></td><td>81.6</td><td>33.0</td><td>71.9</td></tr><tr><td colspan="2">Self-supervised backbones</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Web-DINO</td><td>7B/14</td><td></td><td>85.9</td><td>77.1</td><td>88.6</td><td></td><td>75.6</td><td>64.0</td><td></td><td>71.6</td><td>31.2</td><td>69.7</td></tr><tr><td>Franca</td><td>g/14</td><td></td><td>84.8</td><td>75.3</td><td>89.2</td><td></td><td>67.6</td><td>49.5</td><td></td><td>56.5</td><td>40.0</td><td>54.5</td></tr><tr><td>DINOv2</td><td>g/14</td><td></td><td>87.3</td><td>79.5</td><td>89.9</td><td></td><td>81.1</td><td>65.4</td><td></td><td>81.7</td><td>24.1</td><td>66.4</td></tr><tr><td>DINOv3</td><td>7B/16</td><td></td><td>88.4</td><td>81.4</td><td>90.4</td><td></td><td>91.1</td><td>71.3</td><td></td><td>86.9</td><td>19.6</td><td>79.0</td></tr></tbody></table>

##### Finegrained Classification ()

Table 8: Finegrained classification benchmarks. Fine-S averages over 12 datasets, see Tab.˜22 for full results.

<table><tbody><tr><th>Method</th><th>ViT</th><td>Fine-S</td><td>Places</td><td>iNat18</td><td>iNat21</td></tr><tr><th colspan="2">Agglomerative backbones</th><td></td><td></td><td></td><td></td></tr><tr><th>AM-RADIOv2.5</th><th>g/14</th><td>93.9</td><td>70.2</td><td>79.0</td><td>83.7</td></tr><tr><th colspan="2">Weakly-supervised backbones</th><td></td><td></td><td></td><td></td></tr><tr><th>SigLIP 2</th><th>g/16</th><td>93.7</td><td>70.5</td><td>80.7</td><td>82.7</td></tr><tr><th>PEcore</th><th>G/14</th><td>94.5</td><td>71.3</td><td>86.6</td><td>87.0</td></tr><tr><th>AIMv2</th><th>3B/14</th><td>92.9</td><td>70.7</td><td>80.8</td><td>83.2</td></tr><tr><th>EVA CLIP</th><th>18B/14</th><td>92.9</td><td>71.1</td><td>80.7</td><td>83.5</td></tr><tr><th colspan="2">Self-supervised backbones</th><td></td><td></td><td></td><td></td></tr><tr><th>Franca</th><th>g/14</th><td>87.7</td><td>64.6</td><td>61.4</td><td>70.6</td></tr><tr><th>DINOv2</th><th>g/14</th><td>92.6</td><td>68.2</td><td>80.7</td><td>86.1</td></tr><tr><th>Web-DINO</th><th>7B/14</th><td>90.2</td><td>69.6</td><td>65.3</td><td>74.1</td></tr><tr><th>DINOv3</th><th>7B/16</th><td>93.0</td><td>70.0</td><td>85.6</td><td>89.8</td></tr></tbody></table>

Table 9: Instance recognition benchmarks. See Tab.˜23 for additional metrics.

| Oxford-H | Paris-H | Met (GAP) | AmsterTime |
| --- | --- | --- | --- |
| 47.5 | 85.7 | 30.5 | 23.1 |
| 25.1 | 60.9 | 13.9 | 15.5 |
| 32.7 | 68.9 | 10.6 | 23.1 |
| 28.8 | 71.4 | 29.5 | 14.6 |
| 27.1 | 65.6 | 0.5 | 18.9 |
| 14.3 | 51.6 | 27.2 | 21.1 |
| 58.2 | 84.6 | 44.6 | 48.9 |
| 31.2 | 80.3 | 35.2 | 30.6 |
| 60.7 | 87.1 | 55.4 | 56.5 |

We also measure DINOv3’s performance when training linear probes on several datasets for fine-grained classification. In particular, we report the accuracy on 3 large datasets, namely Places205 [^238] for scene recognition, and iNaturalist 2018 [^188] and iNaturalist 2021 [^189]) for detailed plant and animal-species recognition, as well as the average over 12 smaller datasets covering scenes, objects, and textures (as in [^134], here termed Fine-S). See also Tab.˜22 for individual results on those datasets.

We find that, again, DINOv3 surpasses all previous SSL methods. It also shows competitive results compared to the weakly-supervised methods, indicating its robustness and generalization capability across diverse finegrained classification tasks. Notably, DINOv3 attains the highest accuracy on the difficult iNaturalist21 dataset at 89.8%, outperforming even the best weakly-supervised model PEcore with 87.0%.

#### 6.2.2 Instance Recognition

To evaluate the instance-level recognition capabilities of our model, we adopted a non-parametric retrieval approach. Here, database images are ranked by their cosine similarity to a given query image, using the output CLS token. We benchmark performance across several datasets: the Oxford and Paris datasets for landmark recognition [^143], the Met dataset featuring artworks from the Metropolitan Museum [^227], and AmsterTime, which consists of modern street view images matched to historical archival images of Amsterdam [^224]. Retrieval effectiveness is quantified using mean average precision for Oxford, Paris, and AmsterTime, and global average precision for Met. See Sec.˜D.8 for more evaluation details.

##### Results ( and )

Across all evaluated benchmarks, DINOv3 achieves the strongest performance by large margins, *e.g*. improving over the second best model DINOv2 by +10.8 points on Met and +7.6 points on AmsterTime. On this benchmark, weakly-supervised models are lagging far behind DINOv3, with the exception of AM-RADIO, which is distilled from DINOv2 features. These findings highlight the robustness and versatility of DINOv3 for instance-level retrieval tasks, spanning both traditional landmark datasets and more challenging domains such as art and historical image retrieval.

### 6.3 DINOv3 is a Foundation for Complex Computer Vision Systems

The previous two sections already provided solid signal for the quality of DINOv3 in both dense and global tasks. However, these results were obtained under “model probing” experimental protocols, using lightweight linear adapters or even non-parametric algorithms to assess the quality of features. While such simple evaluations allowed to remove confounding factors from involved experimental protocols, they are not enough to evaluate the full potential of DINOv3 as a foundational component in a larger computer vision system. Thus, in this section, we depart from the lightweight protocols, and instead train more involved downstream decoders and consider stronger, task-specific baselines. In particular, we use DINOv3 as a basis for (1) object detection with Plain-DETR (Sec.˜6.3.1), (2) semantic segmentation with Mask2Former (Sec.˜6.3.2), (3) monocular depth estimation with Depth Anything (Sec.˜6.3.3), and (4) 3D understanding with the Visual Geometry Grounded Transformer (Sec.˜6.3.4). These tasks are only intended as explorations for what is possible with DINOv3. Still, we find that building on DINOv3 unlocks competitive or even state-of-the-art results with little effort.

#### 6.3.1 Object Detection

As a first task, we tackle the long-standing computer vision problem of object detection. Given an image, the goal is to provide bounding boxes for all instances of objects of pre-defined categories. This task requires both precise localization and good recognition, as boxes need to match the object boundaries and correspond to the correct category. While performance on standard benchmarks like COCO [^117] is mostly saturated, we propose to tackle this task with a *frozen* backbone, only training a small decoder on top.

##### Datasets and Metrics

We evaluate DINOv3 on object detection capabilities with the COCO dataset [^117], reporting results on the COCO-VAL2017 split. Additionally, we evaluate out-of-distribution performance on the COCO-O evaluation dataset [^126]. This dataset contains the same classes but provides input images under six distribution shift settings. For both datasets, we report mean Average Precision (mAP) with IoU thresholds in $[0.5:0.05:0.95]$. For COCO-O, we additionally report the effective robustness (ER). Since COCO is a small dataset, comprising only 118k training images, we leverage the larger Objects365 dataset [^162] for pre-training the decoder, as is common practice.

##### Implementation

We build upon the Plain-DETR [^119], but make the following modification: We do not fuse the transformer encoder into the backbone, but keep it as a separate module, similar to the original DETR [^24], which allows us to keep the DINOv3 backbone completely frozen during training and inference. To the best of our knowledge, this makes it *the first competitive detection model to use a frozen backbone*. We train the Plain-DETR detector on Objects365 for 22 epochs at resolution 1536, then one epoch at resolution 2048, followed by 12 epochs on COCO at resolution 2048. At inference time, we run at resolution 2048. Optionally, we also apply test-time augmentation (TTA) by forwarding the image at multiple resolutions (from 1536 to 2880). See Sec.˜D.9 for full experimental details.

##### Results ()

We compare our system with four models: EVA-02 with a Cascade detector [^65], EVA-02 with Co-DETR [^243], InternImage-G with DINO [^202], and PEspatial with DETA [^18]. We find that our lightweight detector (100M parameters) trained on top of a frozen DINOv3 backbone manages to reach state-of-the-art performance. For COCO-O, the gap is pronounced, showing that the detection model can effectively leverage the robustness of the DINOv3. Interestingly, our model outperforms all previous models with much fewer trained parameters, with the smallest comparison point still using more than 300M trainable parameters. We argue that achieving such strong performance without specializing the backbone is an enabler for various practical applications: A single backbone forward can provide features that support multiple tasks, reducing compute requirements.

Table 10: Comparison with state-of-the-art systems on object detection. We train a detection adapter on top of a *frozen* DINOv3 backbone. We show results on the validation set of the COCO and COCO-O datasets, and report the mAP across IoU thresholds, as well as the effective robustness (ER). Our detection system based on DINOv3 sets a new state of the art. As the InternImage-G detection model has not been released, we were unable to reproduce their results or compute COCO-O scores.

<table><thead><tr><th></th><th></th><th></th><th colspan="3">Parameters</th><th colspan="2">COCO</th><th colspan="2">COCO-O</th></tr><tr><th>Model</th><th>Detector</th><th>FT</th><th>Encoder</th><th>Decoder</th><th>Trainable</th><th>Simple</th><th>TTA</th><th>mAP</th><th>ER</th></tr></thead><tbody><tr><th>EVA-02</th><th>Cascade</th><th>🔥</th><th>300M</th><td>—</td><td>300M</td><td>64.1</td><td>—</td><td>63.6</td><td>34.7</td></tr><tr><th>InternImage-G</th><th>DINO</th><th>🔥</th><th>6B</th><td>—</td><td>6B</td><td>65.1</td><td>65.3</td><td>—</td><td>—</td></tr><tr><th>EVA-02</th><th>Co-DETR</th><th>🔥</th><th>300M</th><td>—</td><td>300M</td><td>65.4</td><td>65.9</td><td>63.7</td><td>34.3</td></tr><tr><th>PEspatial</th><th>DETA</th><th>🔥</th><th>1.9B</th><td>50M</td><td>2B</td><td>65.3</td><td>66.0</td><td>64.0</td><td>34.7</td></tr><tr><th>DINOv3</th><th>Plain-DETR</th><th>❄</th><th>7B</th><td>100M</td><td>100M</td><td>65.6</td><td>66.1</td><td>66.4</td><td>36.8</td></tr></tbody></table>

#### 6.3.2 Semantic Segmentation

Following the previous experiment, we now evaluate on semantic segmentation, another long-standing computer vision problem. This task also requires strong, well localized representations, and expects a dense per-pixel prediction. However, opposed to object detection, the model does not need to differentiate instances of the same object. Similar to detection, we train a decoder on top of a *frozen* DINOv3 model.

##### Datasets and Metrics

We focus our evaluation on the ADE20k dataset [^239], which contains 150 semantic categories across 20k training images and 2k validation images. We measure performance using the mean Intersection over Union (mIoU). To train the segmentation model, we additionally use the COCO-Stuff [^22] and Hypersim [^154] datasets. Those contain 164k images with 171 semantic categories, and 77k images with 40 categories respectively.

##### Implementation

To build a decoder that maps DINOv3 features to semantic categories, we combine ViT-Adapter [^38] and Mask2Former [^39], similar to prior work [^203] [^202] [^201]. However, in our case, the DINOv3 backbone remains frozen during training. In order to avoid altering the backbone features, we further modify the original ViT-Adapter architecture by removing the injector component. Compared to baselines, we also increase the embedding dimensions from $1024$ to $2048$, to support processing the $4096$ -dimensional output of the DINOv3 backbone. We start by pre-training the segmentation decoder on COCO-Stuff for $80$ k iterations, followed by $10$ k iterations on Hypersim [^154]. Finally, we train for 20k iterations on the training split of ADE20k and report results on the validation split. All training is done at an input resolution of 896. At inference time we consider two setups: single-scale, *i.e*. we forward images at training resolution, or multi-scale, *i.e*. we average predictions at multiple image ratios between ${\times}0.9$ and $1.1$ the original training resolution. We refer to Sec.˜D.10 for more experimental details.

##### Results ()

We compare our model’s performance with several state-of-the-art baselines, including BEIT-3 [^203], InternImage-H [^202] and ONE-PEACE [^201], and report results on additional datasets in Tab.˜24. Our segmentation model based on the frozen DINOv3 backbone reaches state-of-the-art performance, equaling that of ONE-PEACE ($63.0$ mIoU). It also improves over all prior models on the COCO-Stuff [^22] and VOC 2012 [^60] datasets. As semantic segmentation requires accurate per-pixel predictions, vision transformer backbones pose a fundamental problem. Indeed, the 16 pixel-wide input patches make the granularity of the prediction relatively coarse—encouraging solutions like ViT-Adapter. On the other hand, we have shown that we can obtain high-quality feature maps, even at very high resolutions up to 4096 (*c.f*. Figs.˜3 and 4); this corresponds to dense feature maps 512-tokens wide. We hope that future work will be able to leverage these high-resolution features to reach state-of-the-art performance without having to rely on heavy decoders like ViT-Adapter with Mask2Former.

Table 11: Comparison with state-of-the-art systems for semantic segmentation on ADE20k. We evaluate the model in a single- or multi-scale setup (respectively Simple and TTA). Following common practice, we run this evaluation at resolution 896 and report mIoU scores. BEIT3, ONE-PEACE and DINOv3 use a Mask2Former with ViT-Adapter architecture, and the decoder parameters take into account both. We report results on further datasets in Tab.˜24

<table><thead><tr><th></th><th></th><th colspan="3">Parameters</th><th colspan="2">mIoU</th></tr><tr><th>Model</th><th>FT</th><th>Encoder</th><th>Decoder</th><th>Trainable</th><th>Simple</th><th>TTA</th></tr></thead><tbody><tr><th>BEIT3</th><th>🔥</th><td>1.0B</td><td>550M</td><td>1.6B</td><td>62.0</td><td>62.8</td></tr><tr><th>InternImage-H</th><th>🔥</th><td>1.1B</td><td>230M</td><td>1.3B</td><td>62.5</td><td>62.9</td></tr><tr><th>ONE-PEACE</th><th>🔥</th><td>1.5B</td><td>710M</td><td>2.2B</td><td>62.0</td><td>63.0</td></tr><tr><th>DINOv3</th><th>❄</th><td>7B</td><td>927M</td><td>927M</td><td>62.6</td><td>63.0</td></tr></tbody></table>

#### 6.3.3 Monocular Depth Estimation

We now consider building a system for monocular depth estimation. To do so, we follow the setup of Depth Anything V2 (DAv2) [^222], a recent state-of-the-art method. The key innovation of DAv2 is to use a large collection of synthetically generated images with ground truth depth annotations. Critically, this relies on DINOv2 as a feature extractor that is able to bridge the *sim-to-real* gap, a capability that other vision backbones like SAM [^106] do not show [^222]. Thus, we swap DINOv2 with DINOv3 in the DAv2 pipeline to see if we can achieve similar results.

##### Implementation

Like DAv2, we use a Dense Prediction Transformer (DPT) [^147] to predict a pixelwise depth field, using features from four equally spaced layers of DINOv3 as input. We train the model using the set of losses from DAv2 on DAv2’s synthetic dataset, increasing the training resolution to $1024\times 768$ to make use of DINOv3’s high resolution capabilities. In contrast to DAv2, we *keep the backbone frozen* instead of finetuning it, testing the out-of-the-box capabilities of DINOv3. We also found it beneficial to scale up the DPT head to obtain the full potential DINOv3 7B’s larger features. See Sec.˜D.11 for details.

##### Datasets and Metrics

We evaluate our model on 5 real-world datasets (NYUv2 [^163], KITTI [^72], ETH3D [^159], ScanNet (from [^103]) and DIODE [^190]) in the zero-shot scale-invariant depth setup, similar to [^146] [^103] [^222]. We report the standard metrics absolute relative error (ARel) (lower is better) and $\delta_{1}$ (higher is better). We refer to [^221] for a description of those metrics.

##### Results ()

We compare to the state of the art for relative depth estimation: MiDaS [^146], LeReS [^225], Omnidata [^57], DPT [^147], Marigold in the ensemble version [^103] and DAv2. Our depth estimation model reaches a new state-of-the-art on all datasets, only lacking behind in ARel on DIODE compared to DPT. Remarkably, this is possible using a *frozen backbone*, whereas all other baselines need to finetune the backbone for depth estimation. In addition, this validates that DINOv3 inherits DINOv2’s *strong sim-to-real capabilities*, a desirable property that opens up the possibility for downstream tasks to use synthetically generated training data.

Table 12: Comparison with state-of-the-art systems for relative monocular depth estimation. By combining DINOv3 with Depth Anything V2 [^222], we obtain a SotA model for relative depth estimation.

<table><thead><tr><th></th><th></th><th></th><th colspan="2">NYUv2</th><th></th><th colspan="2">KITTI</th><th></th><th colspan="2">ETH3D</th><th></th><th colspan="2">ScanNet</th><th></th><th colspan="2">DIODE</th></tr><tr><th>Method</th><th>FT</th><th></th><th>ARel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><msub><mi>δ</mi> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\delta_{1}\uparrow</annotation></semantics></math></th><th></th><th>ARel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><msub><mi>δ</mi> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\delta_{1}\uparrow</annotation></semantics></math></th><th></th><th>ARel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><msub><mi>δ</mi> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\delta_{1}\uparrow</annotation></semantics></math></th><th></th><th>ARel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><msub><mi>δ</mi> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\delta_{1}\uparrow</annotation></semantics></math></th><th></th><th>ARel <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></th><th><math><semantics><mrow><msub><mi>δ</mi> <mn>1</mn></msub> <mo>↑</mo></mrow> <annotation>\delta_{1}\uparrow</annotation></semantics></math></th></tr></thead><tbody><tr><th>MiDaS</th><td>🔥</td><td></td><td>11.1</td><td>88.5</td><td></td><td>23.6</td><td>63.0</td><td></td><td>18.4</td><td>75.2</td><td></td><td>12.1</td><td>84.6</td><td></td><td>33.2</td><td>71.5</td></tr><tr><th>LeReS</th><td>🔥</td><td></td><td>9.0</td><td>91.6</td><td></td><td>14.9</td><td>78.4</td><td></td><td>17.1</td><td>77.7</td><td></td><td>9.1</td><td>91.7</td><td></td><td>27.1</td><td>76.6</td></tr><tr><th>Omnidata</th><td>🔥</td><td></td><td>7.4</td><td>94.5</td><td></td><td>14.9</td><td>83.5</td><td></td><td>16.6</td><td>77.8</td><td></td><td>7.5</td><td>93.6</td><td></td><td>33.9</td><td>74.2</td></tr><tr><th>DPT</th><td>🔥</td><td></td><td>9.8</td><td>90.3</td><td></td><td>10.0</td><td>90.1</td><td></td><td>7.8</td><td>94.6</td><td></td><td>8.2</td><td>93.4</td><td></td><td>18.2</td><td>75.8</td></tr><tr><th>Marigold</th><td>🔥</td><td></td><td>5.5</td><td>96.4</td><td></td><td>9.9</td><td>91.6</td><td></td><td>6.5</td><td>96.0</td><td></td><td>6.4</td><td>95.1</td><td></td><td>30.8</td><td>77.3</td></tr><tr><th>DAv2 (ViT-g)</th><td>🔥</td><td></td><td>4.4</td><td>97.9</td><td></td><td>7.5</td><td>94.7</td><td></td><td>13.1</td><td>86.5</td><td></td><td>—</td><td>—</td><td></td><td>—</td><td>—</td></tr><tr><th>DINOv3</th><td>❄</td><td></td><td>4.3</td><td>98.0</td><td></td><td>7.3</td><td>96.7</td><td></td><td>5.4</td><td>97.5</td><td></td><td>4.4</td><td>98.1</td><td></td><td>25.6</td><td>82.2</td></tr></tbody></table>

#### 6.3.4 Visual Geometry Grounded Transformer with DINOv3

Finally, we consider 3D understanding with the recent Visual Geometry Grounded Transformer (VGGT) [^199]. Trained on a large set of 3D-annotated data, VGGT learns to estimate all important 3D attributes of a scene, such as camera intrinsics and extrinsics, point maps, or depth maps, in a single forward pass. Using a simple, unified pipeline, it reaches state-of-the-art results on many 3D tasks while being more efficient than specialized methods—constituting a major advance in 3D understanding.

##### Implementation

VGGT uses a DINOv2-pretrained backbone to obtain representations for different views of a scene, before fusing them with a transformer. Here, we simply swap the DINOv2 backbone with DINOv3, using our ViT-L variant (see Sec.˜7) to match DINOv2 ViT-L/14 in the original work. We run the same training pipeline as VGGT, including finetuning of the image backbone. We switch the image resolution from $518\times 518$ to $592\times 592$ to accommodate DINOv3’s patch size 16 and keep the the results comparable to VGGT. We additionally adopt a small number of hyperparameter changes detailed in Sec.˜D.12.

Table 13: 3D understanding using Visual Geometry Grounded Transformer (VGGT) [^199]. Simply by swapping DINOv2 for DINOv3 ViT-L as the image feature extractor in the VGGT pipeline, we are able to obtain state-of-the-art results on various 3D geometry tasks. We reproduce baseline results from [^199]. We also report methods using ground truth camera information, marked with <sup>∗</sup>. Camera pose estimation results are reported with AUC@30.

(a) Camera pose estimation.

| Method | Re10K | CO3Dv2 |
| --- | --- | --- |
| DUSt3R | 67.7 | 76.7 |
| MASt3R | 76.4 | 81.8 |
| VG GSfM v2 | 78.9 | 83.4 |
| CUT3R | 75.3 | 82.8 |
| FLARE | 78.8 | 83.3 |
| VGGT | 85.3 | 88.2 |
| DINOv3 | 86.3 | 89.6 |

(b) Multi-view estimation on DTU.

| Method | Acc.$\downarrow$ | Comp.$\downarrow$ | Overall $\downarrow$ |
| --- | --- | --- | --- |
| Gipuma <sup>∗</sup> | 0.283 | 0.873 | 0.578 |
| CIDER <sup>∗</sup> | 0.417 | 0.437 | 0.427 |
| MASt3R <sup>∗</sup> | 0.403 | 0.344 | 0.374 |
| GeoMVSNet <sup>∗</sup> | 0.331 | 0.259 | 0.295 |
| DUSt3R | 2.677 | 0.805 | 1.741 |
| VGGT | 0.389 | 0.374 | 0.382 |
| DINOv3 | 0.375 | 0.361 | 0.368 |

(c) View matching on ScanNet-1500.

| Method | AUC@5 | AUC@10 |
| --- | --- | --- |
| SuperGlue | 16.2 | 33.8 |
| LoFTR | 22.1 | 40.8 |
| DKM | 29.4 | 50.7 |
| CasMTR | 27.1 | 47.0 |
| Roma | 31.8 | 53.4 |
| VGGT | 33.9 | 55.2 |
| DINOv3 | 35.2 | 56.1 |

##### Datasets and Metrics

Following [^199], we evaluate on camera pose estimation on the Re10K [^241] and CO3Dv2 [^151] datasets, dense multi-view estimation on DTU [^96], and two-view matching on ScanNet-1500 [^45]. For camera pose estimation and two-view matching, we report the standard area-under-curve (AUC) metric. For multi-view estimation, we report the smallest L2-distance between prediction to ground truth as “Accuracy”, the smallest L2-distance from ground truth to prediction as “Completeness” and their average as ‘Overall”. We refer to [^199] for details about method and evaluation.

##### Results ()

We find that VGGT equipped with DINOv3 *further improves over the previous state-of-the-art* set by VGGT on all three considered tasks—using DINOv3 leads to clear and consistent gains. This is encouraging, given that we only applied minimal tuning for DINOv3. These tasks span different levels of visual understanding: high-level abstraction of scene content (camera pose estimation), dense geometric prediction (multi-view depth estimation), and fine-grained pixel-level correspondence (view matching). Together with the previous results on correspondence estimation (Sec.˜6.1.3) and depth estimation (Sec.˜6.3.3), we take this as further empirical evidence for the strong suitability of DINOv3 as a basis for 3D tasks. Additionally, we anticipate further improvements from using the larger DINOv3 7B model.

## 7 Evaluating the Full Family of DINOv3 Models

In this section, we provide quantitative evaluations on the family of models distilled from our 7B-parameters model (See Sec.˜5.2). This family includes variants based on the Vision Transformer (ViT) and the ConvNeXt (CNX) architectures. We provide the detailed parameter counts and inference FLOPs for all models in Fig.˜16(a). These models cover a wide range of computational budgets to accommodate a broad spectrum of users and deployment scenarios. We conduct a thorough evaluation of all ViT (Sec.˜7.1) and ConvNeXt variants to assess their performance across tasks.

Figure˜2 provides an overview comparison of the DINOv3 family versus other model collections. The DINOv3 family significantly outperforms all others on dense prediction tasks. This includes specialized models distilled from supervised backbones like AM-RADIO and PEspatial. At the same time, our models achieve similar results on classification tasks, making them the optimal choice across compute budgets.

In Sec.˜7.1 detail our ViT models and compare them to other open-source alternatives. Then, in Sec.˜7.2, we discuss the ConvNeXt models. Finally, following Sec.˜5.3, we trained a text encoder aligned with the output of our ViT-L model. We present multi-modal alignment results for this model in Sec.˜7.3.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x21.png)

(a) DINOv3 family of models.

### 7.1 A Vision Transformer for Every Use Case

Our ViT family spans architectures from the compact ViT-S to the larger 840 million parameter ViT-H+ models. The former is designed to run efficiently on resource-constrained devices such as laptops, the latter delivers state-of-the-art performance for more demanding applications. We compare our ViT models to the best open-source image encoders of corresponding size, namely DINOv2 [^134], SigLIP 2 [^185] and Perception Encoder [^18]. For a fair comparison, we ensure that the input sequence length is equivalent across models. Specifically, for model with a patch size of 16 we input images of size $512\times 512$ versus $448\times 448$ when models are using patch size 14.

Table 14: Comparison of our family of models against open-source alternatives of comparable size. We showcase our ViT-{S, S+, B, L, H+} models on a representative set of global and dense benchmarks: classification (IN-ReAL, IN-R, ObjectNet), retrieval (Oxford-H), segmentation (ADE20k), depth (NYU), tracking (DAVIS at 960px), and keypoint matching (NAVI, SPair). We match the number of patch tokens for a fair comparison across models of different patch size.

<table><tbody><tr><td></td><td></td><td></td><td colspan="4">Global Tasks</td><td></td><td colspan="5">Dense Tasks</td></tr><tr><td>Size</td><td>Model</td><td></td><td>IN-ReaL</td><td>IN-R</td><td>Obj.</td><td>Ox.-H</td><td></td><td>ADE20k</td><td>NYU <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>DAVIS</td><td>NAVI</td><td>SPair</td></tr><tr><td>S</td><td>DINOv2</td><td></td><td>87.3</td><td>54.0</td><td>47.8</td><td>39.5</td><td></td><td>45.5</td><td>0.446</td><td>73.6</td><td>53.4</td><td>51.6</td></tr><tr><td>S</td><td>DINOv3</td><td></td><td>87.0</td><td>60.4</td><td>50.9</td><td>49.5</td><td></td><td>47.0</td><td>0.403</td><td>72.7</td><td>56.3</td><td>50.4</td></tr><tr><td>S+</td><td>DINOv3</td><td></td><td>88.0</td><td>68.8</td><td>54.6</td><td>50.0</td><td></td><td>48.8</td><td>0.399</td><td>75.5</td><td>57.1</td><td>55.2</td></tr><tr><td>B</td><td>PEcore</td><td></td><td>87.5</td><td>68.4</td><td>57.9</td><td>20.2</td><td></td><td>37.4</td><td>0.641</td><td>44.5</td><td>41.8</td><td>13.7</td></tr><tr><td>B</td><td>SigLIP 2</td><td></td><td>89.3</td><td>80.6</td><td>66.9</td><td>20.2</td><td></td><td>41.6</td><td>0.512</td><td>63.2</td><td>45.4</td><td>32.8</td></tr><tr><td>B</td><td>DINOv2</td><td></td><td>89.0</td><td>68.4</td><td>57.3</td><td>51.0</td><td></td><td>48.4</td><td>0.416</td><td>72.9</td><td>56.9</td><td>57.1</td></tr><tr><td>B</td><td>DINOv3</td><td></td><td>89.3</td><td>76.7</td><td>64.1</td><td>58.5</td><td></td><td>51.8</td><td>0.373</td><td>77.2</td><td>58.8</td><td>57.2</td></tr><tr><td>L</td><td>PEcore</td><td></td><td>90.1</td><td>87.7</td><td>74.9</td><td>25.6</td><td></td><td>39.7</td><td>0.650</td><td>48.2</td><td>42.1</td><td>19.2</td></tr><tr><td>L</td><td>SigLIP 2</td><td></td><td>90.1</td><td>89.2</td><td>75.0</td><td>21.4</td><td></td><td>43.6</td><td>0.484</td><td>66.3</td><td>47.8</td><td>41.9</td></tr><tr><td>L</td><td>DINOv2</td><td></td><td>89.7</td><td>79.1</td><td>64.7</td><td>55.7</td><td></td><td>48.8</td><td>0.394</td><td>73.4</td><td>59.9</td><td>57.0</td></tr><tr><td>L</td><td>DINOv3</td><td></td><td>90.2</td><td>88.1</td><td>74.8</td><td>63.1</td><td></td><td>54.9</td><td>0.352</td><td>79.9</td><td>62.3</td><td>61.2</td></tr><tr><td>SO400m</td><td>SigLIP 2</td><td></td><td>90.3</td><td>90.4</td><td>76.2</td><td>23.0</td><td></td><td>44.0</td><td>0.402</td><td>64.8</td><td>48.8</td><td>38.7</td></tr><tr><td>H+</td><td>DINOv3</td><td></td><td>90.3</td><td>90.0</td><td>78.6</td><td>64.5</td><td></td><td>54.8</td><td>0.352</td><td>79.3</td><td>63.3</td><td>56.3</td></tr></tbody></table>

Our empirical study clearly demonstrates that DINOv3 models consistently outperform their counterparts on dense prediction tasks. Most notably, on the ADE20k benchmark, the DINOv3 ViT-L model achieves an improvement of over 6 mIoU points compared to the best competitor DINOv2. The ViT-B variant shows a gain of approximately 3 mIoU points against the next best competitor. These substantial improvements highlight the effectiveness of DINOv3’s local features in capturing fine-grained spatial details. Furthermore, evaluations on depth estimation tasks also reveal consistent performance gains over competing approaches. This underscores the versatility of the DINOv3 family across different dense vision problems. Importantly, our models achieve competitive results on global recognition benchmarks such as ObjectNet and ImageNet-1k. This indicates that the enhanced dense task performance does not come at the expense of global task accuracy. This balance confirms that DINOv3 models provide a robust and well-rounded solution, excelling across both dense and global vision tasks without compromise.

On another note, we want to also validate if the largest models that we distill capture all the information from the teacher. To this end, we run a comparison of our largest ViT-H+ with the 7B teacher. As shown in Fig.˜16(b), the largest student achieves performance that is on par with the 8 times larger ViT-7B model. This result not only validates the effectiveness of our distillation process but also demonstrates that, when guided by a high-quality teacher, smaller models can learn to deliver comparable levels of performance. This finding reinforces our belief that *training very large models benefits the broader community*. The strength of larger models can be successfully distilled into more efficient, smaller models with little or no loss of quality.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/figures/analysis/family_resolutions/vits512_cat_marc_pca_rgb.png)

Refer to caption

### 7.2 Efficient ConvNeXts for Resource-Constrained Environments

In this section, we evaluate the quality of our ConvNeXt (CNX) models distilled from the 7B teacher. ConvNeXt models are highly efficient in terms of FLOPs and are well-suited for deployment on devices optimized for convolutional computations. Furthermore, transformer models often do not lend themselves well to quantization [^19], whereas quantization of convolutional nets is a well explored subject. We distill CNX architectures of size T, S, B, and L (see Fig.˜16(a)) and compare them to the original ConvNeXt models [^122]. These baselines achieve high performance on ImageNet-1k as they were trained in a supervised fashion using ImageNet-22k labels, and thus represent a strong competitor. For this experiment, we provide results for global tasks at input resolutions 256 and 512, for ADE20k at resolution 512, and for NYU at resolution 640.

Table 15: Evaluation of our distilled DINOv3 ConvNeXt models. We compare our models to off-the-shelf ConvNeXts trained supervised on ImageNet-22k [^122]. For global tasks, we give results at input resolutions 256 and 512, as we found the supervised models to significantly degrade at resolution 512.

<table><tbody><tr><td></td><td></td><td></td><td colspan="8">Global Tasks</td><td></td><td colspan="2">Dense Tasks</td></tr><tr><td>Size</td><td>Model</td><td></td><td colspan="2">IN-ReAL</td><td></td><td colspan="2">IN-R</td><td></td><td colspan="2">Obj.</td><td></td><td>ADE20k</td><td>NYU <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td></tr><tr><td></td><td></td><td></td><td>256</td><td>512</td><td></td><td>256</td><td>512</td><td></td><td>256</td><td>512</td><td></td><td></td><td></td></tr><tr><td>T</td><td>Sup.</td><td></td><td>87.3</td><td>83.0</td><td></td><td>45.0</td><td>33.0</td><td></td><td>44.5</td><td>27.1</td><td></td><td>24.8</td><td>0.666</td></tr><tr><td>T</td><td>DINOv3</td><td></td><td>86.6</td><td>87.7</td><td></td><td>73.7</td><td>74.1</td><td></td><td>52.6</td><td>58.7</td><td></td><td>42.7</td><td>0.448</td></tr><tr><td>S</td><td>Sup.</td><td></td><td>88.9</td><td>86.8</td><td></td><td>52.8</td><td>39.1</td><td></td><td>50.8</td><td>40.0</td><td></td><td>22.6</td><td>0.630</td></tr><tr><td>S</td><td>DINOv3</td><td></td><td>87.9</td><td>88.7</td><td></td><td>73.7</td><td>74.1</td><td></td><td>52.6</td><td>58.7</td><td></td><td>44.8</td><td>0.432</td></tr><tr><td>B</td><td>Sup.</td><td></td><td>89.3</td><td>87.8</td><td></td><td>57.3</td><td>46.2</td><td></td><td>53.6</td><td>46.5</td><td></td><td>26.5</td><td>0.596</td></tr><tr><td>B</td><td>DINOv3</td><td></td><td>88.5</td><td>89.2</td><td></td><td>77.2</td><td>78.2</td><td></td><td>56.2</td><td>61.3</td><td></td><td>46.3</td><td>0.420</td></tr><tr><td>L</td><td>Sup.</td><td></td><td>89.6</td><td>88.1</td><td></td><td>58.4</td><td>46.6</td><td></td><td>55.0</td><td>47.7</td><td></td><td>33.3</td><td>0.567</td></tr><tr><td>L</td><td>DINOv3</td><td></td><td>88.9</td><td>89.4</td><td></td><td>81.3</td><td>82.4</td><td></td><td>59.3</td><td>65.2</td><td></td><td>47.8</td><td>0.403</td></tr></tbody></table>

##### Results ()

We find that on in-distribution image classification, our models slightly lag behind the supervised ones at resolution 256 (*e.g*. $-0.7$ IN-ReAL for CNX-T). However, the trend is reversed at resolution 512, with the supervised ConvNeXts significantly degrading, whereas our models scale with increased input resolution. For out-of-distribution classification (IN-R, ObjectNet), there are significant gaps between the two model families for all sizes—a testament to the robustness of the DINOv3 CNX models. Furthermore, the DINOv3 models offer very large improvement on dense tasks. Indeed, for CNX-T, our model yields a $+17.9$ mIoU (42.7 versus 24.8) improvement, and for CNX-L, our model gets $+14.5$ mIoU (47.8 versus 33.3). The combination of high performance and computational efficiency makes the distilled ConvNeXt models especially promising for real-world applications where resource constraints are critical. Aside from that, the distillation of the ViT-7B model into smaller ConvNeXt models is particularly exciting, as it bridges two fundamentally different architectures. While ViT-7B is based on transformer blocks with a CLS token, ConvNeXt relies on convolutional operations without a CLS token, making this transfer of knowledge non-trivial. This achievement highlights the versatility and effectiveness of our distillation process.

### 7.3 Zero-shot Inference with DINOv3-based dino.txt

As detailed in Sec.˜5.3, we train a text encoder to align both the CLS token and the output patches of the distilled DINOv3 ViT-L model to text, following the recipe of dino.txt [^100]. We evaluate the quality of the alignment both at the global- and patch-level on standard benchmarks. We report the zero-shot classification accuracy using the CLIP protocol [^144] on the ImageNet-1k, ImageNet-Adversarial, ImageNet-Rendition and ObjectNet benchmarks. For image-text retrieval, we evaluate on the COCO2017 dataset [^186] and report Recall@1 on both image-to-text (I $\rightarrow$ T) and text-to-image (T $\rightarrow$ I) tasks. To probe the quality of patch-level alignment, we evaluate our model on the open-vocabulary segmentation task using the common benchmarks ADE20k and Cityscapes, for which we report the mIoU metric.

##### Results ()

We compare our text-aligned DINOv3 ViT-L with competitors in the same size class. Compared to [^100], which aligns DINOv2 to text, DINOv3 leads to significantly better performance on all benchmarks. On global alignment tasks, we compare favorably to the original CLIP [^144] and strong baselines such as EVA-02-CLIP [^173] but slightly behind SigLIP2 [^185] and Perception Encoder [^18]. On dense alignment tasks, our text-aligned model shows excellent performance on two challenging benchmarks ADE20K and Cityscapes thanks to clean feature maps of DINOv3.

Table 16: Comparing our text-aligned DINOv3 ViT-L to the state-of-the-art. Our model achieves excellent dense alignment performance while staying competitive in global alignment tasks. All compared models are of ViT-L size and operate on the same sequence length of 576.

<table><tbody><tr><td></td><th></th><th colspan="4">Classification</th><th></th><th colspan="2">Retrieval</th><th></th><th colspan="2">Segmentation</th></tr><tr><th>Method</th><th></th><th>IN1k</th><th>A</th><th>R</th><th>Obj.</th><th></th><th>I <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> T</th><th>T <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> I</th><th></th><th>ADE20k</th><th>Cityscapes</th></tr><tr><td>CLIP</td><td></td><td>76.6</td><td>77.5</td><td>89.0</td><td>72.3</td><td></td><td>57.9</td><td>37.1</td><td></td><td>6.0</td><td>11.5</td></tr><tr><td>EVA-02-CLIP</td><td></td><td>80.4</td><td>82.9</td><td>93.2</td><td>78.5</td><td></td><td>64.1</td><td>47.9</td><td></td><td>10.9</td><td>14.1</td></tr><tr><td>dino.txt</td><td></td><td>81.6</td><td>83.2</td><td>88.8</td><td>74.5</td><td></td><td>62.5</td><td>45.0</td><td></td><td>19.2</td><td>27.4</td></tr><tr><td>SigLIP 2</td><td></td><td>83.1</td><td>84.3</td><td>95.7</td><td>84.4</td><td></td><td>71.4</td><td>55.3</td><td></td><td>10.8</td><td>16.3</td></tr><tr><td>PE</td><td></td><td>83.5</td><td>89.0</td><td>95.2</td><td>84.7</td><td></td><td>75.9</td><td>57.1</td><td></td><td>17.6</td><td>21.4</td></tr><tr><td>DINOv3 dino.txt</td><td></td><td>82.3</td><td>85.4</td><td>93.0</td><td>80.5</td><td></td><td>63.7</td><td>45.6</td><td></td><td>24.7</td><td>36.9</td></tr></tbody></table>

## 8 DINOv3 on Geospatial Data

Our self-supervised learning recipe is generic and can be applied to any image domain. In this section, we showcase this universality by building a DINOv3 7B model for satellite images, which have very different characteristics (*e.g*. object texture, sensor noise, and focal views) than the web images on which DINOv3 was initially developed.

### 8.1 Pre-Training Data and Benchmarks

Our satellite DINOv3 7B model is pre-trained on SAT-493M, a dataset of 493 millions of $512\times 512$ images sampled randomly from Maxar RGB ortho-rectified imagery at 0.6 meter resolution. We use the exact same set of hyper-parameters that are used for the web DINOv3 7B model, except for the RGB mean and std normalization that are adapted for satellite images, and the training length. Similar to the web model, our training pipeline for the satellite model consists of 100k iterations of initial pre-training with global crops ($256\times 256$), followed by 10k iterations using Gram regularization, and finalized with 8k steps of high resolution fine-tuning at resolution $512$. Also similar to the web model, we distill our 7B satellite model into a more manageable ViT-Large model to facilitate its use in low-budget regime.

We evaluate DINOv3 satellite and web models on multiple earth observation tasks. For the task of global canopy height mapping, we use the Satlidar dataset described in Sec.˜D.13, which consists of one million $512\times 512$ images with LiDAR ground truths split into train/val/test splits with ratios 8/1/1. The splits include the Neon and São Paulo dataset used by [^180]. For national-scale canopy height mapping, we evaluate on Open-Canopy [^69], which combines SPOT 6-7 satellite imagery and aerial LiDAR data over 87,000 km <sup>2</sup> across France. Since images in this dataset have 4 channels including the additional infra-red (IR) channel, we adapt our backbone by taking the average of the three channels in the weights of the patch embed module and adding it to the weights as the fourth channel. We trained a DPT decoder on $512\times 512$ crops of images resized to 1667 to match the Maxar ground sample resolution.

Table 17: Evaluation of different backbones for high-resolution canopy height prediction. All models are trained with a DPT decoder. Results are presented either for experiments with the decoder trained on SatLidar and evaluated on IID samples (SatLidar Val) and OOD test sets (SatLidar Test, Neon and São Paulo), or for experiments with the decoder trained and evaluated on the Open-Canopy dataset. We list mean absolute error (MAE) and the block $R^{2}$ metric from [^180]. For completeness, we additionally evaluate the original decoder of [^180] that was trained on Neon dataset (denoted by <sup>∗</sup>).

<table><thead><tr><th rowspan="3">Method</th><th rowspan="3">Arch.</th><th></th><th colspan="9">SatLidar</th><th></th><th rowspan="2">Open Canopy</th></tr><tr><th></th><th colspan="2">SatLidar Val</th><th></th><th colspan="2">SatLidar Test</th><th colspan="2">Neon Test</th><th colspan="2">São Paulo</th><th></th></tr><tr><th></th><th>MAE↓</th><th><math><semantics><msup><mi>R</mi> <mn>2</mn></msup> <annotation>R^{2}</annotation></semantics></math> ↑</th><th></th><th>MAE↓</th><th><math><semantics><msup><mi>R</mi> <mn>2</mn></msup> <annotation>R^{2}</annotation></semantics></math> ↑</th><th>MAE↓</th><th><math><semantics><msup><mi>R</mi> <mn>2</mn></msup> <annotation>R^{2}</annotation></semantics></math> ↑</th><th>MAE↓</th><th><math><semantics><msup><mi>R</mi> <mn>2</mn></msup> <annotation>R^{2}</annotation></semantics></math> ↑</th><th></th><th>MAE↓</th></tr></thead><tbody><tr><td><sup><a href="#fn:180">180</a></sup> <sup>∗</sup></td><td>ViT-L</td><td></td><td>2.8</td><td>0.86</td><td></td><td>4.0</td><td>0.61</td><td>2.7</td><td>0.73</td><td>5.4</td><td>0.42</td><td></td><td>—</td></tr><tr><td><sup><a href="#fn:180">180</a></sup></td><td>ViT-L</td><td></td><td>2.4</td><td>0.90</td><td></td><td>3.4</td><td>0.81</td><td>2.9</td><td>0.69</td><td>5.4</td><td>0.48</td><td></td><td>2.42</td></tr><tr><td>DINOv3 Web</td><td>ViT-7B</td><td></td><td>2.4</td><td>0.90</td><td></td><td>3.6</td><td>0.74</td><td>2.7</td><td>0.75</td><td>5.9</td><td>0.34</td><td></td><td>2.17</td></tr><tr><td>DINOv3 Sat</td><td>ViT-L</td><td></td><td>2.2</td><td>0.91</td><td></td><td>3.2</td><td>0.81</td><td>2.4</td><td>0.81</td><td>5.8</td><td>0.42</td><td></td><td>2.07</td></tr><tr><td>DINOv3 Sat</td><td>ViT-7B</td><td></td><td>2.2</td><td>0.92</td><td></td><td>3.2</td><td>0.82</td><td>2.6</td><td>0.74</td><td>5.5</td><td>0.51</td><td></td><td>2.02</td></tr></tbody></table>

Semantic geospatial tasks are assessed with GEO-Bench [^111], which comprises six classification and six segmentation tasks spanning various spatial resolutions and optical bands. The GEO-Bench tasks are diverse, including the detection of rooftop-mounted photovoltaic systems, classifying local climate zones, measuring drivers of deforestation, and detecting tree crowns. For high-resolution semantic tasks, we consider the land cover segmentation dataset LoveDA [^200], the object segmentation dataset iSAID [^231], and the horizontal detection dataset DIOR [^115].

### 8.2 Canopy Height Estimation

Estimating canopy height from satellite imagery is a challenging metric task, requiring accurate recovery of continuous spatial structure despite random variations in slope, viewing geometry, sun angle, atmospheric scattering, and quantization artifacts. This task is critical for global carbon monitoring and for forest and agriculture management [^82]. Following [^180], the first work to leverage a SSL backbone trained on satellite images for this task, we train a DPT head on top of frozen DINOv3 on the SatLidar1M training set, then evaluate it on i.i.d. samples on SatLidar1M validation set as well as out-of-distribution test sets including SatLidar1M test, Neon and Sao Paulo. We additionally train and evaluate on the Open-Canopy dataset.

##### Results ()

We compare different SSL backbones, denoting with “DINOv3 Sat” the model trained the SAT-493M dataset, and with “DINOv3 Web” the model trained on LVD-1689M (see Sec.˜3.1). It can be seen that DINOv3 satellite models yield state-of-the-art performance on most benchmarks. Our 7B satellite model sets the new state of the art on SatLidar1M val, SatLidar1M test and Open-Canopy, reducing MAE from $2.4$ to $2.2$, from $3.4$ to $3.2$ and from $2.42$ to $2.02$ respectively. These results show that DINOv3 training recipe is generic and can be effectively applied out-of-the-box to other domains. Interestingly, our distilled ViT-L satellite model performs comparably to its 7B counterpart, achieving comparable results on SatLidar1M and Open-Canopy while faring surprisingly better on Neon test set, reaching the lowest MAE of $2.4$ compared to $2.6$ of the 7B model and $2.9$ of [^180]. Our DINOv3 7B web model reaches decent performance on the benchmarks, outperforming [^180] on SatLidar1M val, Neon and Open-Canopy but stays behind the satellite model. This highlights the strength of domain-specific pretraining for physically grounded tasks like canopy height estimation, where sensor-specific priors and radiometric consistency are important.

### 8.3 Comparison to the Earth Observation State of the Art

Table 18: Comparison of our DINOv3 models against strong baselines DOFA [^217], Prithvi-v2 [^176], and [^180] in Geo-Bench tasks. While Privthi-v2 and DOFA leverage all available optical bands, our models achieve significantly better performance with only RGB inputs.

(a) Classification tasks.

| Method | Arch. | FT | Bands | m-BEnet | m-brick-kiln | m-eurosat | m-forestnet | m-pv4ger | m-so2sat | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DOFA | ViT-L | 🔥 | all | 68.7 | 98.4 | 96.6 | 55.7 | 98.2 | 61.6 | 79.9 |
| Best of Prithvi-v2 | ViT-L/H | 🔥 | all | 71.2 | 98.8 | 96.4 | 54.1 | 98.1 | 59.1 | 79.6 |
| [^180] | ViT-L | ❄ | RGB | 66.0 | 97.1 | 95.2 | 56.3 | 94.3 | 58.1 | 77.8 |
| DINOv3 Sat | ViT-L | ❄ | RGB | 73.0 | 96.5 | 94.1 | 60.6 | 96.0 | 57.4 | 79.6 |
| DINOv3 Sat | 7B | ❄ | RGB | 74.0 | 97.2 | 94.8 | 62.3 | 96.1 | 62.1 | 81.1 |
| DINOv3 Web | 7B | ❄ | RGB | 74.6 | 97.7 | 97.0 | 57.9 | 98.3 | 63.8 | 81.6 |

(b) Segmentation tasks.

| Method | Arch. | FT | Bands | m-cashew <sup>∗</sup> | m-chesapeake | m-NeonTree | m-nz-cattle | m-pv4ger-seg | m-SA-crop | Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DOFA | ViT-L | 🔥 | all | 81.2 | 61.6 | 58.5 | 77.4 | 95.1 | 35.7 | 68.3 |
| Best of Prithvi-v2 | ViT-L/H | 🔥 | all | 90.2 | 69.4 | 59.1 | 81.0 | 95.3 | 41.9 | 72.8 |
| [^180] | ViT-L | ❄ | RGB | 92.8 | 73.7 | 58.1 | 83.1 | 94.7 | 35.1 | 72.9 |
| DINOv3 Sat | ViT-L | ❄ | RGB | 94.2 | 75.6 | 61.8 | 83.7 | 95.2 | 36.8 | 74.5 |
| DINOv3 Sat | 7B | ❄ | RGB | 94.1 | 76.6 | 62.6 | 83.4 | 95.5 | 37.6 | 75.0 |
| DINOv3 Web | 7B | ❄ | RGB | 96.0 | 76.5 | 66.4 | 83.7 | 95.9 | 36.8 | 75.9 |

<sup>∗</sup> Conversion to 6 classes following [^176].

We compare the performance of different methods for Earth observation tasks in Tab.˜18 and Tab.˜19. The frozen DINOv3 satellite and web models set new state-of-the-art results on 12 out of 15 classification, segmentation, and horizontal object detection tasks. Our Geo-Bench results surpass prior models, including Prithvi-v2 [^176] and DOFA [^217], which use 6+ bands for Sentinel-2 and Landsat tasks, as well as task-specific fine-tuning (Tab.˜18). Despite using a frozen backbone with RGB-only input, the DINOv3 satellite model outperforms previous methods on the three unsaturated classification tasks and on five of six segmentation tasks. Interestingly, the DINOv3 7B web model is very competitve on these benchmarks. It achieves comparable or stronger performance on many GEO-Bench tasks as well as on large-scale, high-resolution remote sensing benchmarks for segmentation and detection. As shown in Tab.˜18 and Tab.˜19, the frozen DINOv3 web model establishes new leading results Geo-Bench tasks as well as for segmentation and detection tasks on the LoveDA and DIOR datasets.

These findings have broader implications for the design of geospatial foundation models. Those have recently emphasized heuristic techniques such as multitemporal aggregation, multisensor fusion, or incorporating satellite-specific metadata [^21] [^67]. Our results show that general-purpose SSL can match or exceed satellite-specific approaches for tasks that depend on precise object boundaries (segmentation or object detection). This supports emerging evidence finding that domain-agnostic pretraining can offer strong generalization even in specialized downstream domains [^112].

Collectively, our results suggest task-dependent benefits of domain-specific pretraining. The DINOv3 satellite model excels in metric tasks like depth estimation, leveraging satellite-specific priors. In contrast, the DINOv3 web model achieves state-of-the-art results on semantic geospatial tasks through diverse, universal representations. The complementary strengths of both models illustrate the broad applicability and effectiveness of the DINOv3 SSL paradigm.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/figures/satellite/img.lr.jpg)

Figure 18: Illustration of versatile applications in remote sensing made possible by a single DINOv3 model. The PCA on DINOv3 features shows finer details than DINOv2. The segmentation map was computed using only GEO-Bench chesapeake labels. The canopy height model decoder was trained on the Open-Canopy dataset using 4 channels (RGB + InfraRed), while inference was performed on RGB channels only.

Table 19: We compare the performance of DINOv3 to state-of-the-art models Privthi-v2 [^176], BillionFM [^29] and SkySense V2 [^237] for high resolution semantic geospatial tasks. We report mIoU for the segmentation datasets LoveDA (1024 $\times$) and iSAID (896 $\times$), and mAP for the detection dataset DIOR ($800\times$).

<table><tbody><tr><th>Method</th><th>Arch.</th><td>FT</td><td></td><td>LoveDA</td><td>iSAID</td><td>DIOR</td></tr><tr><th rowspan="2">Prev. SotA</th><th></th><td rowspan="2">🔥</td><td></td><td>BillionFM, ViT-G</td><td>SkySense V2, Swin-G <sup>∗</sup></td><td>SkySense V2, Swin-G <sup>∗</sup></td></tr><tr><th></th><td></td><td>54.4</td><td>71.9</td><td>79.5</td></tr><tr><th>Decoder Arch.</th><th></th><td></td><td></td><td>UPerNet</td><td>UPerNet</td><td>Faster-RCNN</td></tr><tr><th>Privthi-v2</th><th>ViT-H</th><td>🔥</td><td></td><td>52.2</td><td>62.8</td><td>—</td></tr><tr><th>DINOv3 Sat</th><th>ViT-L</th><td>❄</td><td></td><td>54.4</td><td>62.9</td><td>72.7</td></tr><tr><th>DINOv3 Sat</th><th>ViT-7B</th><td>❄</td><td></td><td>55.3</td><td>64.8</td><td>76.6</td></tr><tr><th>DINOv3 Web</th><th>ViT-7B</th><td>❄</td><td></td><td>56.2</td><td>71.4</td><td>80.5</td></tr></tbody></table>

<sup>∗</sup> Uses modified DINOv2 SSL with supervised pretraining alignment on OpenStreetMap, reporting +0.8 mIoU on iSAID.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/figures/satellite/img12.lr.jpg)

Figure 19: A qualitative comparison of the DINOv3 7B satellite model to 180 on the Open Canopy dataset. For both models, the decoder is trained on 448 × \\times 448 input images. It can be seen that DINOv3 produces more accurate maps, for example the accurate height for the trees on the field.

## 9 Environmental Impact

Table 20: Carbon footprint of model training. We report the potential carbon emission of reproducing a full model pre-training, computed using a PUE of 1.1 and carbon intensity factor of 0.385kg CO <sub>2</sub> eq/KWh.

| Model | Arch. | GPU type | Power | Steps | GPU hours | PUE | Total power | Emission |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  | (W) |  |  |  | (MWh) | (tCO <sub>2</sub> eq) |
| MetaCLIP | ViT-G | A100-40GB | 400W | 390k | 368,640 | 1.1 | 160 | 62 |
| DINOv2 | ViT-g | A100-40GB | 400W | 625k | 22,016 | 1.1 | 9.7 | 3.7 |
| DINOv3 | ViT-7B | H100-SXM5 | 700W | 1,000k | 61,440 | 1.1 | 47 | 18 |

To estimate the carbon emission of our pre-training, we follow the methodology used in previous work in natural language processing [^170] [^184] and SSL [^134]. We fix the value of all exogenous variables, *i.e*. the Power Usage Effectiveness (PUE) and carbon intensity factor of a power grid to the same value as used by [^184], *i.e*. we assume a PUE of 1.1 and a carbon intensity factor of the US average of 0.385 kg CO <sub>2</sub> eq/KWh. For the power consumption of GPUs, we take their thermal design power: 400W for A100 GPUs and 700W for H100 GPUs. We report the details of the computation for the pre-training of our ViT-7B in Tab.˜20. For reference, we provide the analogous data for DINOv2 and MetaCLIP. As another point of comparison, the energy required to train one DINOv3 model (47 MWh) is roughly equivalent to that required for 240,000 km of driving with an average electric vehicle.

##### Carbon Footprint of the Whole Project

In order to compute the carbon footprint of the whole project, we use a rough estimate of a total 9M GPU hours. Using the same grid parameters as presented above, we estimate the total footprint to be roughly 2600 tCO <sub>2</sub> eq. For comparison, a full Boeing 777 return flight between Paris and New York corresponds to approximately 560 tCO <sub>2</sub> eq. Supposing 12 such flights per day, the environmental impact of our project represents half of all flights between these two cities for one day. This estimate only considers the electricity for powering the GPUs and ignores other emissions, such as cooling, manufacturing, and disposal.

## 10 Conclusion

DINOv3 represents a significant advancement in the field of self-supervised learning, demonstrating the potential to revolutionize the way visual representations are learned across various domains. By scaling dataset and model size through meticulous data preparation, design, and optimization, DINOv3 showcases the power of self-supervised learning to eliminate the dependency on manual annotations. The introduction of the Gram anchoring method effectively mitigates the degradation of dense feature maps over extended training periods, ensuring robust and reliable performance.

Together with the implementation of post-hoc polishing strategies, such as high-resolution post-training and distillation, we achieve state-of-the-art performance across a wide range of visual tasks with no fine-tuning of the image encoder. The DINOv3 suite of vision models not only sets new benchmarks but also offers a versatile solution across various resource constraints, deployment scenarios, and application use cases. The progress made with DINOv3 is a testament to the promise of self-supervised learning in advancing the state of the art in computer vision and beyond.

## References

Appendix

## Appendix A Artifacts and Outliers in Large-Scale Training

This section provides a discussion about the emergence of artifacts and outliers that has recently been observed in the training of large models in both the LLM [^1] and the visual domains [^46]. Borrowing the definition from [^1], outliers are typically characterized as network’s activations whose values deviate significantly from the average of their distribution. During the training of DINOv3, we identified such outliers at different levels: some occurring at the patch level and others at the feature dimension level. We discuss bellow the different types of outlier observed, their impact on the training and results. We also discuss our different attempts at fixing them and our first conclusions.

### A.1 High-Norm Patch Outliers

[^46] discovered that patch outliers negatively affect performance in DINOv2. These outliers are primarily characterized as high-norm tokens, often located in low-information background regions of an image. These tokens are observed to play a key role in the internal communication between patches and the CLS token. Additionally, this phenomenon affects other models as well, whether trained with supervision or not, such as CLIP [^144]. When scaling to a 7B model, we observe the emergence of such high-norm patches, predominantly in the background area. In this section, we present results from 7B models trained for 150k iterations, which, although limited, provide us with initial signals to guide our decisions. We plot the output patch norms (before the layer norm) in Sec.˜A.1, in the column ‘ $\varnothing$ ’, with high-norm patches in yellow appearing in the sky and other low-information areas.

##### Token Registers

In order to mitigate the appearance of such token outliers, [^46] proposes a simple yet effective solution: introducing additional tokens, called registers, into the input sequence of the ViT. Their role is to take over the internal communication between patches and the CLS. Following the conclusions, we use 4 registers and do not ablate further due to the high experimental cost. Section˜A.1 illustrates examples of this strategy in action, where we observe the elimination of high-norm outliers, as further confirmed by the corresponding histogram of the norm distribution. Moreover, we quantitatively observe in Fig.˜20(b) the benefit of incorporating additional register tokens on the ImageNet-1k (IN1k) benchmark.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2508.10104/assets/x22.png)

(a) Visualization of patch norms by outlier strategy. The bottom two rows share a colormap per row, from dark blue (low) to yellow (high).

[^1]: Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Systematic outliers in large language models. *ICLR*, 2025.

[^2]: Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In *ICLR*, 2020.

[^3]: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. *arXiv preprint arXiv:2301.08243*, 2023.

[^4]: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. *arXiv preprint arXiv:2506.09985*, 2025.

[^5]: Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. *arXiv preprint arXiv:2202.03555*, 2022.

[^6]: Alexei Baevski, Arun Babu, Wei-Ning Hsu, and Michael Auli. Efficient self-supervised learning with contextualized target representations for vision, speech and language. In *ICML*, 2023.

[^7]: Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In *CVPR*, 2024.

[^8]: Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. *arXiv preprint arXiv:2106.08254*, 2021.

[^9]: Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. ObjectNet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In *NeurIPS*, 2019.

[^10]: Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. *arXiv preprint arXiv:2105.04906*, 2021.

[^11]: Adrien Bardes, Jean Ponce, and Yann LeCun. Vicregl: Self-supervised learning of local visual features. *arXiv preprint arXiv:2210.01571*, 2022.

[^12]: Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. *arXiv preprint arXiv:2404.08471*, 2024.

[^13]: Lucas Beyer, Olivier J. Hénaff, Alexander Kolesnikov, Xiaohua Zhai, and Aäron van den Oord. Are we done with imagenet? *CoRR*, abs/2006.07159, 2020.

[^14]: Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3b vlm for transfer. *arXiv preprint arXiv:2407.07726*, 2024.

[^15]: Navaneeth Bodla, Bharat Singh, Rama Chellappa, and Larry S. Davis. Soft-nms – improving object detection with one line of code. In *ICCV*, Oct 2017.

[^16]: Piotr Bojanowski and Armand Joulin. Unsupervised learning by predicting noise. In *ICML*, 2017.

[^17]: Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. *Transactions of the association for computational linguistics*, 5:135–146, 2017.

[^18]: Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. *arXiv preprint arXiv:2504.13181*, 2025.

[^19]: Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. Understanding and overcoming the challenges of efficient transformer quantization. In *Conference on Empirical Methods in Natural Language Processing*, 2021.

[^20]: Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101 – mining discriminative components with random forests. In *ECCV*, 2014.

[^21]: Christopher F. Brown, Michal R. Kazmierski, Valerie J. Pasquarella, William J. Rucklidge, Masha Samsikova, Chenhui Zhang, Evan Shelhamer, Estefania Lahera, Olivia Wiles, Simon Ilyushchenko, Noel Gorelick, Lihui Lydia Zhang, Sophia Alj, Emily Schechter, Sean Askay, Oliver Guinan, Rebecca Moore, Alexis Boukouvalas, and Pushmeet Kohli. Alphaearth foundations: An embedding field model for accurate and efficient global mapping from sparse label data, 2025.

[^22]: Holger Caesar, Jasper Uijlings, and Vittorio Ferrari. Coco-stuff: Thing and stuff classes in context. In *CVPR*, 2018.

[^23]: Zhi Cai, Songtao Liu, Guodong Wang, Zheng Ge, Xiangyu Zhang, and Di Huang. Align-detr: Enhancing end-to-end object detection with aligned loss, 2024.

[^24]: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *European conference on computer vision*, pages 213–229. Springer, 2020.

[^25]: Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In *ECCV*, 2018.

[^26]: Mathilde Caron, Piotr Bojanowski, Julien Mairal, and Armand Joulin. Unsupervised pre-training of image features on non-curated data. In *ICCV*, 2019.

[^27]: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In *NeurIPS*, 2020.

[^28]: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *ICCV*, 2021.

[^29]: Keumgang Cha, Junghoon Seo, and Taekyung Lee. A billion-scale foundation model for remote sensing images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, page 1–17, 2024. ISSN 2151-1535. doi: 10.1109/jstars.2024.3401772.

[^30]: François Charton and Julia Kempe. Emergent properties with repeated examples, 2024.

[^31]: Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Bowen Chen, Andrew Zhang, Daniel Shao, Andrew H Song, Muhammad Shaban, et al. Towards a general-purpose foundation model for computational pathology. *Nature Medicine*, 2024.

[^32]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. *preprint arXiv:2002.05709*, 2020a.

[^33]: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey E Hinton. Big self-supervised models are strong semi-supervised learners. *Advances in neural information processing systems*, 33:22243–22255, 2020b.

[^34]: Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish V Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme Ruiz, Andreas Peter Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. PaLI: A jointly-scaled multilingual language-image model. In *ICLR*, 2023.

[^35]: Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. *preprint arXiv:2011.10566*, 2020.

[^36]: Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. *arXiv preprint arXiv:2104.02057*, 2021.

[^37]: Yinjie Chen, Zipeng Yan, Chong Zhou, Bo Dai, and Andrew F Luo. Vision transformers with self-distilled registers. *arXiv preprint arXiv:2505.21501*, 2025.

[^38]: Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. *arXiv preprint arXiv:2205.08534*, 2022.

[^39]: Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In *CVPR*, 2022.

[^40]: Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In *CVPR*, 2023.

[^41]: Minsu Cho, Suha Kwak, Cordelia Schmid, and Jean Ponce. Unsupervised object discovery and localization in the wild: Part-based matching with bottom-up region proposals. In *CVPR*, 2015.

[^42]: M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed,, and A. Vedaldi. Describing textures in the wild. In *CVPR*, 2014.

[^43]: Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David Lobell, and Stefano Ermon. Satmae: Pre-training transformers for temporal and multi-spectral satellite imagery. *NeurIPS*, 2022.

[^44]: Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 3213–3223, 2016.

[^45]: Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. ScanNet: Richly-annotated 3d reconstructions of indoor scenes. In *CVPR*, 2017.

[^46]: Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In *ICLR*, 2024.

[^47]: Timothée Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Cluster and predict latent patches for improved masked image modeling. *TMLR*, 2025.

[^48]: Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Steiner, Mathilde Caron, Robert Geirhos, Ibrahim M. Alabdulmohsin, Rodolphe Jenatton, Lucas Beyer, Michael Tschannen, Anurag Arnab, Xiao Wang, Carlos Riquelme, Matthias Minderer, Joan Puigcerver, Utku Evci, Manoj Kumar, Sjoerd van Steenkiste, Gamaleldin F. Elsayed, Aravindh Mahendran, Fisher Yu, Avital Oliver, Fantine Huot, Jasmijn Bastings, Mark Collier, Alexey A. Gritsenko, Vighnesh Birodkar, Cristina Nader Vasconcelos, Yi Tay, Thomas Mensink, Alexander Kolesnikov, Filip Paveti’c, Dustin Tran, Thomas Kipf, Mario Luvci’c, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, and Neil Houlsby. Scaling vision transformers to 22 billion parameters. In *ICML*, 2023.

[^49]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In *CVPR*, 2009.

[^50]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *preprint arXiv:1810.04805*, 2018.

[^51]: Barry M Dillon, Gregor Kasieczka, Hans Olischlager, Tilman Plehn, Peter Sorrenson, and Lorenz Vogel. Symmetries, safety, and self-supervision. *SciPost Physics*, 12(6):188, 2022.

[^52]: Henghui Ding, Chang Liu, Shuting He, Xudong Jiang, Philip HS Torr, and Song Bai. Mose: A new dataset for video object segmentation in complex scenes. In *CVPR*, 2023.

[^53]: Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In *ICCV*, 2015.

[^54]: Alexey Dosovitskiy, Philipp Fischer, Jost Tobias Springenberg, Martin Riedmiller, and Thomas Brox. Discriminative unsupervised feature learning with exemplar convolutional neural networks. *IEEE TPAMI*, 2016.

[^55]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *preprint arXiv:2010.11929*, 2020.

[^56]: Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. PaLM-E: An embodied multimodal language model. In *ICML*, 2023.

[^57]: Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for making multi-task mid-level vision datasets from 3d scans. In *ICCV*, 2021.

[^58]: Alaaeldin El-Nouby, Gautier Izacard, Hugo Touvron, Ivan Laptev, Hervé Jegou, and Edouard Grave. Are large-scale datasets necessary for self-supervised pre-training? *arXiv preprint arXiv:2112.10740*, 2021.

[^59]: Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The PASCAL visual object classes challenge 2007 (VOC2007) results, 2007.

[^60]: Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results, 2012.

[^61]: Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. *IJCV*, 111(1):98–136, 2015.

[^62]: David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, et al. Scaling language-free visual representation learning. *arXiv preprint arXiv:2504.01017*, 2025.

[^63]: Alex Fang, Albin Madappally Jose, Amit Jain, Ludwig Schmidt, Alexander T Toshev, and Vaishaal Shankar. Data filtering networks. In *ICLR*, 2024a.

[^64]: Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In *CVPR*, 2023.

[^65]: Yuxin Fang, Quan Sun, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva-02: A visual representation for neon genesis. *Image and Vision Computing*, 149:105171, 2024b.

[^66]: Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In *2004 conference on computer vision and pattern recognition workshop*, pages 178–178. IEEE, 2004.

[^67]: Zhengpeng Feng, Clement Atzberger, Sadiq Jaffer, Jovana Knezevic, Silja Sormunen, Robin Young, Madeline C Lisaius, Markus Immitzer, David A. Coomes, Anil Madhavapeddy, Andrew Blake, and Srinivasan Keshav. TESSERA: Temporal embeddings of surface spectra for earth representation and analysis, 2025.

[^68]: Enrico Fini, Mustafa Shukor, Xiujun Li, Philipp Dufter, Michal Klein, David Haldimann, Sai Aitharaju, Victor Guilherme Turrisi da Costa, Louis Béthune, Zhe Gan, Alexander T Toshev, Marcin Eichner, Moin Nabi, Yinfei Yang, Joshua M. Susskind, and Alaaeldin El-Nouby. Multimodal autoregressive pre-training of large vision encoders. *arXiv preprint arXiv:2411.14402*, 2024.

[^69]: Fajwel Fogel, Yohann Perron, Nikola Besic, Laurent Saint-André, Agnès Pellissier-Tanon, Martin Schwartz, Thomas Boudras, Ibrahim Fayad, Alexandre d’Aspremont, Loic Landrieu, et al. Open-canopy: Towards very high resolution forest monitoring. In *CVPR*, 2025.

[^70]: Stephanie Fu, Mark Hamilton, Laura E Brandt, Axel Feldmann, Zhoutong Zhang, and William T Freeman. Featup: A model-agnostic framework for features at any resolution. In *ICLR*, 2024.

[^71]: Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2414–2423, 2016.

[^72]: Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. *The International Journal of Robotics Research*, 32(11):1231–1237, 2013.

[^73]: Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations, 2018.

[^74]: Priya Goyal, Dhruv Mahajan, Abhinav Gupta, and Ishan Misra. Scaling and benchmarking self-supervised visual representation learning. In *ICCV*, 2019.

[^75]: Priya Goyal, Mathilde Caron, Benjamin Lefaudeux, Min Xu, Pengchao Wang, Vivek Pai, Mannat Singh, Vitaliy Liptchinsky, Ishan Misra, Armand Joulin, et al. Self-supervised pretraining of visual features in the wild. *preprint arXiv:2103.01988*, 2021.

[^76]: Priya Goyal, Quentin Duval, Isaac Seessel, Mathilde Caron, Mannat Singh, Ishan Misra, Levent Sagun, Armand Joulin, and Piotr Bojanowski. Vision models are more robust and fair when pretrained on uncurated images without supervision. *arXiv preprint arXiv:2202.08360*, 2022a.

[^77]: Priya Goyal, Adriana Romero Soriano, Caner Hazirbas, Levent Sagun, and Nicolas Usunier. Fairness indicators for systematic assessments of visual feature extractors. In *2022 ACM Conference on Fairness, Accountability, and Transparency*, pages 70–88, 2022b.

[^78]: Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In *ICCV*, 2017.

[^79]: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In *NeurIPS*, 2020.

[^80]: Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In *CVPR*, 2006.

[^81]: Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In *ICLR*, 2022.

[^82]: Nancy Harris, David Gibbs, A. Baccini, Richard Birdsey, Sytze de Bruin, Mary Farina, Lola Fatoyinbo, Matthew Hansen, Martin Herold, Richard Houghton, Peter Potapov, Daniela Requena Suarez, Rosa Maria Roman-Cuesta, Sassan Saatchi, Christy Slay, Svetlana Turubanova, and Alexandra Tyukavina. Global maps of twenty-first century forest carbon fluxes. *Nature Climate Change*, 11:1–7, 03 2021. doi: 10.1038/s41558-020-00976-6.

[^83]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *CVPR*, 2016.

[^84]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *CVPR*, 2020.

[^85]: Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. *arXiv preprint arXiv:2111.06377*, 2021.

[^86]: Greg Heinrich, Mike Ranzinger, Hongxu, Yin, Yao Lu, Jan Kautz, Andrew Tao, Bryan Catanzaro, and Pavlo Molchanov. RADIOv2.5: Improved baselines for agglomerative vision foundation models. *arXiv preprint arXiv:2412.07679*, 2025.

[^87]: Olivier J Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, SM Eslami, and Aaron van den Oord. Data-efficient image recognition with contrastive predictive coding. *preprint arXiv:1905.09272*, 2019.

[^88]: Olivier J Hénaff, Skanda Koppula, Jean-Baptiste Alayrac, Aaron van den Oord, Oriol Vinyals, and João Carreira. Efficient visual pretraining with contrastive detection. *arXiv preprint arXiv:2103.10957*, 2021.

[^89]: Olivier J Hénaff, Skanda Koppula, Evan Shelhamer, Daniel Zoran, Andrew Jaegle, Andrew Zisserman, João Carreira, and Relja Arandjelović. Object discovery and representation networks. In *ECCV*, 2022.

[^90]: Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*, 2019.

[^91]: Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In *ICCV*, pages 8340–8349, 2021a.

[^92]: Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In *CVPR*, pages 15262–15271, 2021b.

[^93]: Byeongho Heo, Song Park, Dongyoon Han, and Sangdoo Yun. Rotary position embedding for vision transformer. In *European Conference on Computer Vision*, pages 289–305. Springer, 2024.

[^94]: Allan Jabri, Andrew Owens, and Alexei A Efros. Space-time correspondence as a contrastive random walk. In *NeurIPS*, 2020.

[^95]: Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. Navi: Category-agnostic image collections with high-quality 3d shape and pose annotations. In *NeurIPS*, 2023.

[^96]: Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanaes. Large scale multi-view stereopsis evaluation. In *CVPR*, June 2014.

[^97]: Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In *International Conference on Machine Learning*, pages 4904–4916. PMLR, 2021.

[^98]: Nick Jiang, Amil Dravid, Alexei Efros, and Yossi Gandelsman. Vision transformers don’t need trained registers. *arXiv preprint arXiv:2506.08010*, 2025.

[^99]: Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In *ECCV*, 2016.

[^100]: Cijo Jose, Théo Moutakanni, Dahyun Kang, Federico Baldassarre, Timothée Darcet, Hu Xu, Daniel Li, Marc Szafraniec, Michaël Ramamonjisoa, Maxime Oquab, et al. DINOv2 meets text: A unified framework for image-and pixel-level vision-language alignment. In *CVPR*, 2025.

[^101]: Armand Joulin, Laurens Van Der Maaten, Allan Jabri, and Nicolas Vasilache. Learning visual features from large weakly supervised data. In *ECCV*, 2016.

[^102]: Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. *arXiv preprint arXiv:1705.06950*, 2017.

[^103]: Bingxin Ke, Kevin Qu, Tianfu Wang, Nando Metzger, Shengyu Huang, Bo Li, Anton Obukhov, and Konrad Schindler. Marigold: Affordable adaptation of diffusion-based image generators for image analysis. *arXiv preprint arXiv:2505.09358*, 2025.

[^104]: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. OpenVLA: An open-source vision-language-action model. *arXiv preprint arXiv:2406.09246*, 2024.

[^105]: Vladislav Kim, Nikolaos Adaloglou, Marc Osterland, Flavio M Morelli, Marah Halawa, Tim König, David Gnutt, and Paula A Marin Zapata. Self-supervision advances morphological profiling by unlocking powerful image representations. *Scientific Reports*, 15(1):4876, 2025.

[^106]: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In *CVPR*, 2023.

[^107]: Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, and Neil Houlsby. Big transfer (bit): General visual representation learning. In *ECCV*, pages 491–507. Springer, 2020.

[^108]: Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In *4th International IEEE Workshop on 3D Representation and Recognition (3dRR-13)*, Sydney, Australia, 2013.

[^109]: Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.

[^110]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *NeurIPS*, 2012.

[^111]: Alexandre Lacoste, Nils Lehmann, Pau Rodriguez, Evan Sherwin, Hannah Kerner, Björn Lütjens, Jeremy Irvin, David Dao, Hamed Alemohammad, Alexandre Drouin, et al. Geo-bench: Toward foundation models for earth monitoring. *NeurIPS*, 2023.

[^112]: Saad Lahrichi, Zion Sheng, Shufan Xia, Kyle Bradbury, and Jordan Malof. Is self-supervised pre-training on satellite imagery better than imagenet? a systematic study with sentinel-2, 2025.

[^113]: Yann LeCun. A path towards autonomous machine intelligence. *openreview*, 2022.

[^114]: Ang Li, Allan Jabri, Armand Joulin, and Laurens Van Der Maaten. Learning visual n-grams from web data. In *ICCV*, 2017.

[^115]: Ke Li, Gang Wan, Gong Cheng, Liqiu Meng, and Junwei Han. Object detection in optical remote sensing images: A survey and a new benchmark. *ISPRS journal of photogrammetry and remote sensing*, 159:296–307, 2020.

[^116]: Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, 2023a.

[^117]: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and Lawrence Zitnick. Microsoft COCO: common objects in context. In *ECCV*, 2014.

[^118]: Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection, 2018.

[^119]: Yutong Lin, Yuhui Yuan, Zheng Zhang, Chen Li, Nanning Zheng, and Han Hu. Detr does not need multi-scale or locality design. In *ICCV*, 2023b.

[^120]: Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In *NeurIPS*, 2023.

[^121]: Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.

[^122]: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In *CVPR*, 2022.

[^123]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR*, 2017.

[^124]: Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan, Kaiming He, Manohar Paluri, Yixuan Li, Ashwin Bharambe, and Laurens van der Maaten. Exploring the limits of weakly supervised pretraining. In *ECCV*, 2018.

[^125]: S. Maji, J. Kannala, E. Rahtu, M. Blaschko, and A. Vedaldi. Fine-grained visual classification of aircraft. *arXiv preprint arXiv:1306.5151*, 2013.

[^126]: Xiaofeng Mao, Yuefeng Chen, Yao Zhu, Da Chen, Hang Su, Rong Zhang, and Hui Xue. COCO-O: A benchmark for object detectors under natural distribution shifts. In *ICCV*, 2023.

[^127]: Tomáš Mikolov, Martin Karafiát, Lukáš Burget, Jan Černockỳ, and Sanjeev Khudanpur. Recurrent neural network based language model. In *Interspeech*, 2010.

[^128]: Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In *NeurIPS*, 2013.

[^129]: Juhong Min, Jongmin Lee, Jean Ponce, and Minsu Cho. SPair-71k: A large-scale benchmark for semantic correspondence, 2019.

[^130]: Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In *CVPR*, 2020.

[^131]: Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In *Indian Conference on Computer Vision, Graphics and Image Processing*, Dec 2008.

[^132]: Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In Bastian Leibe, Jiri Matas, Nicu Sebe, and Max Welling, editors, *Computer Vision – ECCV 2016*, pages 69–84, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46466-4.

[^133]: Maxime Oquab, Leon Bottou, Ivan Laptev, and Josef Sivic. Learning and transferring mid-level image representations using convolutional neural networks. In *CVPR*, 2014.

[^134]: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. DINOv2: Learning robust visual features without supervision. *TMLR*, 2024.

[^135]: Valentinos Pariza, Mohammadreza Salehi, Gertjan J Burghouts, Francesco Locatello, and Yuki M Asano. Near, far: Patch-ordering enhances vision foundation models’ scene understanding. In *ICLR*, 2025.

[^136]: Liam Parker, Francois Lanusse, Siavash Golkar, Leopoldo Sarra, Miles Cranmer, Alberto Bietti, Michael Eickenberg, Geraud Krawezik, Michael McCabe, Rudy Morel, et al. Astroclip: a cross-modal foundation model for galaxies. *Monthly Notices of the Royal Astronomical Society*, 531(4):4990–5011, 2024.

[^137]: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2012.

[^138]: Deepak Pathak, Philipp Krähenbühl, Jeff Donahue, Trevor Darrell, and Alexei Efros. Context encoders: Feature learning by inpainting. In *CVPR*, 2016.

[^139]: Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 724–732, 2016.

[^140]: Fernando Pérez-García, Harshita Sharma, Sam Bond-Taylor, Kenza Bouzid, Valentina Salvatelli, Maximilian Ilse, Shruthi Bannur, Daniel C Castro, Anton Schwaighofer, Matthew P Lungren, et al. Exploring scalable medical image encoders beyond text supervision. *Nature Machine Intelligence*, pages 1–12, 2025.

[^141]: Pedro O Pinheiro, Amjad Almahairi, Ryan Y Benmaleck, Florian Golemo, and Aaron Courville. Unsupervised learning of dense visual representations. In *NeurIPS*, 2020.

[^142]: Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 DAVIS challenge on video object segmentation. *preprint arXiv:1704.00675*, 2017.

[^143]: Filip Radenović, Ahmet Iscen, Giorgos Tolias, Yannis Avrithis, and Ondřej Chum. Revisiting oxford and paris: Large-scale image retrieval benchmarking. In *CVPR*, 2018.

[^144]: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International Conference on Machine Learning*, pages 8748–8763. PMLR, 2021.

[^145]: Jathushan Rajasegaran, Ilija Radosavovic, Rahul Ravishankar, Yossi Gandelsman, Christoph Feichtenhofer, and Jitendra Malik. An empirical study of autoregressive pre-training from videos. *arXiv preprint arXiv:2501.05453*, 2025.

[^146]: René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE TPAMI*, 44(3):1623–1637, 2020.

[^147]: René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 12179–12188, 2021.

[^148]: Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In *CVPR*, 2024.

[^149]: Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollar, and Christoph Feichtenhofer. SAM 2: Segment anything in images and videos. In *ICLR*, 2025.

[^150]: Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In *ICML*, pages 5389–5400, 2019.

[^151]: Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In *ICCV*, 2021.

[^152]: Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2015.

[^153]: Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression, 2019.

[^154]: Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In *ICCV*, 2021.

[^155]: Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. *IJCV*, 2015.

[^156]: Bryan Russell, William Freeman, Alexei Efros, Josef Sivic, and Andrew Zisserman. Using multiple segmentations to discover objects and their extent in image collections. In *CVPR*, 2006.

[^157]: Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Hervé Jégou. Spreading vectors for similarity search. *arXiv preprint arXiv:1806.03198*, 2018.

[^158]: Mohammadreza Salehi, Efstratios Gavves, Cees G. M. Snoek, and Yuki M. Asano. Time does tell: Self-supervised time-tuning of dense image representations. In *ICCV*, 2023.

[^159]: Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In *CVPR*, 2017.

[^160]: Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. *arXiv preprint arXiv:2111.02114*, 2021.

[^161]: Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, Dominik Zietlow, Tianjun Xiao, Carl-Johann Simon-Gabriel, Tong He, Zheng Zhang, Bernhard Schölkopf, Thomas Brox, and Francesco Locatello. Bridging the gap to real-world object-centric learning. In *ICLR*, 2023.

[^162]: Shuai Shao, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu Zhang, Jing Li, and Jian Sun. Objects365: A large-scale, high-quality dataset for object detection. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 8430–8439, 2019.

[^163]: Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In *European conference on computer vision*, pages 746–760. Springer, 2012.

[^164]: Oriane Siméoni, Gilles Puy, Huy V Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet, and Jean Ponce. Localizing objects with self-supervised transformers and no labels. *arXiv preprint arXiv:2109.14279*, 2021.

[^165]: Oriane Siméoni, Éloi Zablocki, Spyros Gidaris, Gilles Puy, and Patrick Pérez. Unsupervised object localization in the era of self-supervised vits: A survey. *IJCV*, 133(2):781–808, 2025.

[^166]: Walter Simoncini, Andrei Bursuc, Spyridon Gidaris, and Yuki Asano. No train, all gain: Self-supervised gradients improve deep frozen representations. *NeurIPS*, 2024.

[^167]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In *ICLR*, 2015.

[^168]: Mannat Singh, Laura Gustafson, Aaron Adcock, Vinicius de Freitas Reis, Bugra Gedik, Raj Prateek Kosaraju, Dhruv Mahajan, Ross Girshick, Piotr Dollár, and Laurens van der Maaten. Revisiting Weakly Supervised Pre-Training of Visual Perception Models. In *CVPR*, 2022.

[^169]: Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. *arXiv preprint arXiv:1212.0402*, 2012.

[^170]: Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in nlp. *arXiv preprint arXiv:1906.02243*, 2019.

[^171]: Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

[^172]: Chen Sun, Abhinav Shrivastava, Saurabh Singh, and Abhinav Gupta. Revisiting unreasonable effectiveness of data in deep learning era. In *ICCV*, 2017.

[^173]: Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. *arXiv preprint arXiv:2303.15389*, 2023.

[^174]: Quan Sun, Jinsheng Wang, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, and Xinlong Wang. EVA-CLIP-18B: Scaling clip to 18 billion parameters. *arXiv preprint arXiv:2402.04252*, 2024.

[^175]: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In *CVPR*, 2015.

[^176]: Daniela Szwarcman, Sujit Roy, Paolo Fraccaro, Thorsteinn Elí Gíslason, Benedikt Blumenstiel, Rinki Ghosal, Pedro Henrique de Oliveira, Joao Lucas de Sousa Almeida, Rocco Sedona, Yanghui Kang, et al. Prithvi-eo-2.0: A versatile multi-temporal foundation model for earth observation applications. *arXiv preprint arXiv:2412.02732*, 2024.

[^177]: Mingxing Tan and Quoc V Le. Efficientnet: Rethinking model scaling for convolutional neural networks. *preprint arXiv:1905.11946*, 2019.

[^178]: Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. *Communications of the ACM*, 59(2):64–73, 2016.

[^179]: Yonglong Tian, Olivier J Henaff, and Aäron van den Oord. Divide and contrast: Self-supervised learning from uncurated data. In *ICCV*, 2021.

[^180]: Jamie Tolan, Hung-I Yang, Benjamin Nosarzewski, Guillaume Couairon, Huy V Vo, John Brandt, Justine Spore, Sayantan Majumdar, Daniel Haziza, Janaki Vamaraju, et al. Very high resolution canopy height maps from rgb imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote Sensing of Environment*, 300:113888, 2024.

[^181]: Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Hervé Jégou. Fixing the train-test resolution discrepancy. In *NeurIPS*, 2019.

[^182]: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. *preprint arXiv:2012.12877*, 2020.

[^183]: Hugo Touvron, Matthieu Cord, and Hervé Jégou. Deit iii: Revenge of the vit. *arXiv preprint arXiv:2204.07118*, 2022.

[^184]: Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023.

[^185]: Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, and Xiaohua Zhai. SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. *arXiv preprint arXiv:2502.14786*, 2025.

[^186]: Tsung-Yi, Genevieve Patterson, Matteo R. Ronchi, Yin Cui, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays Georgia, Pietro Perona, Deva Ramanan, Larry Zitnick, and Piotr Dollár. COCO 2017: Common Objects in Context, 2017.

[^187]: Tinne Tuytelaars, Christoph Lampert, Matthew Blaschko, and Wray Buntine. Unsupervised object discovery: A comparison. *IJCV*, 2010.

[^188]: Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset. In *CVPR*, 2018.

[^189]: Grant Van Horn, Elijah Cole, Sara Beery, Kimberly Wilber, Serge Belongie, and Oisin Mac Aodha. Benchmarking representation learning for natural world image collections. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 12884–12893, 2021.

[^190]: Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Outdoor DEpth Dataset. *ArXiv 1908.00463*, 2019.

[^191]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *NeurIPS*, 2017.

[^192]: Shashanka Venkataramanan, Valentinos Pariza, Mohammadreza Salehi, Lukas Knobel, Spyros Gidaris, Elias Ramzi, Andrei Bursuc, and Yuki M. Asano. Franca: Nested matryoshka clustering for scalable visual representation learning. *arXiv preprint arXiv:2507.14137*, 2025.

[^193]: Huy V. Vo, Francis Bach, Minsu Cho, Kai Han, Yann LeCun, Patrick Pérez, and Jean Ponce. Unsupervised image matching and object discovery as optimization. In *CVPR*, 2019.

[^194]: Huy V. Vo, Patrick Pérez, and Jean Ponce. Toward unsupervised, multi-object discovery in large-scale image collections. In *Proceedings of the European Conference on Computer Vision (ECCV)*, 2020.

[^195]: Huy V. Vo, Elena Sizikova, Cordelia Schmid, Patrick Pérez, and Jean Ponce. Large-scale unsupervised object discovery. In *Advances in Neural Information Processing Systems 35 (NeurIPS)*, 2021.

[^196]: Huy V. Vo, Vasil Khalidov, Timothée Darcet, Théo Moutakanni, Nikita Smetanin, Marc Szafraniec, Hugo Touvron, Camille Couprie, Maxime Oquab, Armand Joulin, Hervé Jégou, Patrick Labatut, and Piotr Bojanowski. Automatic data curation for self-supervised learning: A clustering-based approach. *TMLR*, 2024.

[^197]: Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, et al. A foundation model for clinical-grade computational pathology and rare cancers detection. *Nature medicine*, 30(10):2924–2935, 2024.

[^198]: Di Wang, Jing Zhang, Minqiang Xu, Lin Liu, Dongsheng Wang, Erzhong Gao, Chengxi Han, Haonan Guo, Bo Du, Dacheng Tao, and Liangpei Zhang. Mtp: Advancing remote sensing foundation model via multi-task pretraining, 2024a.

[^199]: Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In *CVPR*, 2025.

[^200]: Junjue Wang, Zhuo Zheng, Ailong Ma, Xiaoyan Lu, and Yanfei Zhong. Loveda: A remote sensing land-cover dataset for domain adaptive semantic segmentation, 2022a.

[^201]: Peng Wang, Shijie Wang, Junyang Lin, Shuai Bai, Xiaohuan Zhou, Jingren Zhou, Xinggang Wang, and Chang Zhou. One-peace: Exploring one general representation model toward unlimited modalities. *arXiv preprint arXiv:2305.11172*, 2023a.

[^202]: Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. In *CVPR*, 2023b.

[^203]: Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, et al. Image as a foreign language: Beit pretraining for all vision and vision-language tasks. *arXiv preprint arXiv:2208.10442*, 2022b.

[^204]: Xiaolong Wang, Allan Jabri, and Alexei A Efros. Learning correspondence from the cycle-consistency of time. In *CVPR*, 2019.

[^205]: Yangtao Wang, Xi Shen, Shell Xu Hu, Yuan Yuan, James L Crowley, and Dominique Vaufreydaz. Self-supervised transformers for unsupervised object discovery using normalized cut. In *CVPR*, pages 14543–14553, 2022c.

[^206]: Yangtao Wang, Xi Shen, Yuan Yuan, Yuming Du, Maomao Li, Shell Xu Hu, James L Crowley, and Dominique Vaufreydaz. Tokencut: Segmenting objects in images and videos with self-supervised transformer and normalized cut. *IEEE TPAMI*, 45(12):15790–15801, 2023c.

[^207]: Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Chenting Wang, Guo Chen, Baoqi Pei, Rongkun Zheng, Jilan Xu, Zun Wang, et al. Internvideo2: Scaling video foundation models for multimodal video understanding. In *ECCV*, 2024b.

[^208]: Frederik Warburg, Søren Hauberg, Manuel López-Antequera, Pau Gargallo, Yubin Kuang, and Javier Civera. Mapillary street-level sequences: A dataset for lifelong place recognition. In *Computer Vision and Pattern Recognition (CVPR)*, 2020.

[^209]: P. Welinder, S. Branson, T. Mita, C. Wah, F. Schroff, S. Belongie, and P. Perona. Caltech-UCSD Birds 200. Technical Report CNS-TR-2010-001, California Institute of Technology, 2010.

[^210]: Xin Wen, Bingchen Zhao, Anlin Zheng, Xiangyu Zhang, and Xiaojuan Qi. Self-supervised visual representation learning with semantic grouping. In *NeurIPS*, 2022.

[^211]: Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In *CVPR*, 2018.

[^212]: Monika Wysoczańska, Oriane Siméoni, Michaël Ramamonjisoa, Andrei Bursuc, Tomasz Trzciński, and Patrick Pérez. Clip-dinoiser: Teaching clip a few dino tricks. *ECCV*, 2024.

[^213]: J. Xiao, J. Hays, K. A. Ehinger, A. Oliva, and A. Torralba. SUN database: Large-scale scene recognition from abbey to zoo. In *2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, pages 3485–3492, June 2010. doi: 10.1109/CVPR.2010.5539970.

[^214]: Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding, 2018.

[^215]: Jiahao Xie, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, and Chen Change Loy. Unsupervised object-level representation learning from scene images. In *NeurIPS*, 2021.

[^216]: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In *CVPR*, 2017.

[^217]: Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J Stewart, Joëlle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, and Xiao Xiang Zhu. Neural plasticity-inspired foundation model for observing the Earth crossing modalities. *arXiv preprint arXiv:2403.15356*, 2024.

[^218]: Hu Xu, Saining Xie, Xiaoqing Ellen Tan, Po-Yao Huang, Russell Howes, Vasu Sharma, Shang-Wen Li, Gargi Ghosh, Luke Zettlemoyer, and Christoph Feichtenhofer. Demystifying clip data. In *ICLR*, 2024.

[^219]: Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. Youtube-VOS: Sequence-to-sequence video object segmentation. In *Proceedings of the European conference on computer vision (ECCV)*, pages 585–601, 2018.

[^220]: I Zeki Yalniz, Hervé Jégou, Kan Chen, Manohar Paluri, and Dhruv Mahajan. Billion-scale semi-supervised learning for image classification. *arXiv preprint arXiv:1905.00546*, 2019.

[^221]: Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In *CVPR*, 2024a.

[^222]: Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. *NeurIPS*, 2024b.

[^223]: Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In *CVPR*, 2025.

[^224]: Burak Yildiz, Seyran Khademi, Ronald Maria Siebes, and Jan van Gemert. Amstertime: A visual place recognition benchmark dataset for severe domain shift. *arXiv preprint arXiv:2203.16291*, 2022.

[^225]: Wei Yin, Jianming Zhang, Oliver Wang, Simon Niklaus, Long Mai, Simon Chen, and Chunhua Shen. Learning to recover 3d scene shape from a single image. In *CVPR*, 2021.

[^226]: Yongseon Yoo, Seonggyu Kim, and Jong-Min Lee. Sagagan: Style applied using Gram matrix attribution based on stargan v2. In *BMVC*, 2024.

[^227]: Nikolaos-Antonios Ypsilantis, Noa Garcia, Guangxing Han, Sarah Ibrahimi, Nanne Van Noord, and Giorgos Tolias. The met dataset: Instance-level recognition for artworks. In *Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)*, 2021.

[^228]: Jiahui Yu, Zirui Wang, Vijay Vasudevan, Legg Yeung, Mojtaba Seyedhosseini, and Yonghui Wu. Coca: Contrastive captioners are image-text foundation models. *arXiv preprint arXiv:2205.01917*, 2022.

[^229]: Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In *ICLR*, 2025.

[^230]: Sukmin Yun, Hankook Lee, Jaehyung Kim, and Jinwoo Shin. Patch-level representation learning for self-supervised vision transformers. In *CVPR*, 2022.

[^231]: Syed Waqas Zamir, Aditya Arora, Akshita Gupta, Salman Khan, Guolei Sun, Fahad Shahbaz Khan, Fan Zhu, Ling Shao, Gui-Song Xia, and Xiang Bai. isaid: A large-scale dataset for instance segmentation in aerial images, 2019.

[^232]: Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals. Recurrent neural network regularization. *arXiv preprint arXiv:1409.2329*, 2014.

[^233]: Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In *CVPR*, 2022a.

[^234]: Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, and Lucas Beyer. Lit: Zero-shot transfer with locked-image text tuning. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 18123–18133, 2022b.

[^235]: Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 11975–11986, 2023.

[^236]: Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In *ECCV*, 2016.

[^237]: Yingying Zhang, Lixiang Ru, Kang Wu, Lei Yu, Lei Liang, Yansheng Li, and Jingdong Chen. SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing. *arXiv preprint arXiv:2507.13812*, 2025.

[^238]: Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene recognition using places database. In *NeurIPS*, 2014.

[^239]: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 633–641, 2017.

[^240]: Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. *arXiv preprint arXiv:2111.07832*, 2021.

[^241]: Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. Stereo magnification: Learning view synthesis using multiplane images. In *SIGGRAPH*, 2018.

[^242]: Adrian Ziegler and Yuki M Asano. Self-supervised learning of object parts for semantic segmentation. In *CVPR*, 2022.

[^243]: Zhuofan Zong, Guanglu Song, and Yu Liu. Detrs with collaborative hybrid assignments training. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 6748–6758, 2023.

[^244]: Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin D Cubuk, and Quoc V Le. Rethinking pre-training and self-training. *preprint arXiv:2006.06882*, 2020.