---
title: "iBOT : Image BERT Pre-Training with Online Tokenizer"
source: "https://ar5iv.labs.arxiv.org/html/2111.07832"
author:
published:
created: 2026-05-25
description: "The success of language Transformers is primarily attributed to the pretext task of masked language modeling (MLM) (Devlin et al., 2019), where texts are first tokenized into semantically meaningful pieces.In this wor…"
tags:
  - "clippings"
---
Jinghao Zhou <sup>1</sup> Chen Wei <sup>2</sup> Huiyu Wang <sup>2</sup> Wei Shen <sup>3</sup> Cihang Xie <sup>4</sup> Alan Yuille <sup>2</sup> Tao Kong <sup>1</sup>  
  
<sup>1</sup> ByteDance <sup>2</sup> Johns Hopkins University <sup>3</sup> Shanghai Jiao Tong University <sup>4</sup> UC Santa Cruz  

###### Abstract

The success of language Transformers is primarily attributed to the pretext task of masked language modeling (MLM) [^15], where texts are first tokenized into semantically meaningful pieces. In this work, we study masked image modeling (MIM) and indicate the advantages and challenges of using a semantically meaningful visual tokenizer. We present a self-supervised framework iBOT that can perform masked prediction with an online tokenizer. Specifically, we perform self-distillation on masked patch tokens and take the teacher network as the online tokenizer, along with self-distillation on the class token to acquire visual semantics. The online tokenizer is jointly learnable with the MIM objective and dispenses with a multi-stage training pipeline where the tokenizer needs to be pre-trained beforehand. We show the prominence of iBOT by achieving an 82.3% linear probing accuracy and an 87.8% fine-tuning accuracy evaluated on ImageNet-1K. Beyond the state-of-the-art image classification results, we underline emerging local semantic patterns, which helps the models to obtain strong robustness against common corruptions and achieve leading results on dense downstream tasks, *e.g*., object detection, instance segmentation, and semantic segmentation. The code and models are publicly available at [https://github.com/bytedance/ibot](https://github.com/bytedance/ibot).

<sup>†</sup>

## 1 Introduction

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x2.png)

Figure 1: Linear probing accuracy on ImageNet. We compare iBOT with other unsupervised baselines.

Masked Language Modeling (MLM), which first randomly masks and then reconstructs a set of input tokens, is a popular pre-training paradigm for language models. The MLM pre-trained Transformers [^15] have demonstrated their scalability to large-capacity models and datasets, becoming a de-facto standard for lingual tasks. However, its potential for Vision Transformer (ViT), which recently started to revolutionize computer vision research [^43] [^17], has been largely underexplored. Most popular unsupervised pre-training schemes in vision deal with the global views [^12] [^8], neglecting images’ internal structures, as opposed to MLM modeling local tokens. In this work, we seek to continue the success of MLM and explore Masked Image Modeling (MIM) for training better Vision Transformers such that it can serve as a standard component, as it does for NLP.

One of the most crucial components in MLM is the lingual tokenizer which splits language into semantically meaningful tokens, *e.g*., WordPiece [^48] in BERT. Similarly, the crux of MIM lies in a proper design of visual tokenizer, which transforms the masked patches to supervisory signals for the target model, as shown in Fig. 2. However, unlike lingual semantics arising naturally from the statistical analysis of word frequency [^39], visual semantics cannot be extracted such easily due to the continuous property of images. Empirically, visual semantics emerges progressively by bootstrapping online representation that enforces a similarity of distorted image views [^20] [^18] [^7]. This property intuitively indicates a multi-stage training pipeline, where we need to first train an off-the-shelf semantic-rich tokenizer before training the target model. However, since acquiring visual semantics is a common end for both the tokenizer and target model, a single-stage training pipeline where the tokenizer and target model can be jointly optimized awaits further exploration.

Previous works partially tackle the above challenges. Several works use identity mapping as the visual tokenizer, *i.e*., predicting the raw pixel values [^35] [^3]. Such paradigm struggles in semantic abstraction and wastes the capacity at modeling high-frequency details, yielding less competitive performance in semantic understanding [^29]. Recently, BEiT [^4] proposes to use a pre-trained discrete VAE [^36] as the tokenizer. Though providing some level of abstraction, the discrete VAE is still found only to capture low-level semantics within local details (as observed by Tab. 9). Moreover, the tokenizer needs to be offline pre-trained with fixed model architectures and extra dataset [^36], which potentially limits its adapativity to perform MIM using data from different domains.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x3.png)

Figure 2: Masked image modeling. I 𝐼 denotes an image and Tok. denotes a visual tokenizer.

To this end, we present iBOT, short for image BERT pre-training with Online Tokenizer, a new framework that performs MIM with a tokenizer handling above-mentioned challenges favorably. We motivate iBOT by formulating the MIM as knowledge distillation (KD), which learns to distill knowledge from the tokenizer, and further propose to perform self-distillation for MIM with the help of twin teacher as online tokenizer. The target network is fed with a masked image while the online tokenizer with the original image. The goal is to let the target network recover each masked patch token to its corresponding tokenizer output. Our online tokenizer naturally resolves two major challenges. On the one hand, our tokenizer captures high-level visual semantics progressively learned by enforcing the similarity of cross-view images on class tokens. On the other hand, our tokenizer needs no extra stages of training as pre-processing setup since it is jointly optimized with MIM via momentum update.

The online tokenizer enables iBOT to achieve excellent performance for feature representation. Specifically, iBOT advances ImageNet-1K classification benchmark under $k$ -NN, linear probing and fine-tuning protocols to 77.1%, 79.5%, 84.0% with ViT-Base/16 respectively, which is 1.0%, 1.3%, 0.4% higher than previous best results. When pre-trained with ImageNet-22K, iBOT with ViT-L/16 achieves a linear probing accuracy of 82.3% and a fine-tuning accuracy of 87.8%, which is 1.0% and 1.8% higher than previous best results. Beyond that, the advancement is also valid when transferring to other datasets or under semi-supervised and unsupervised classification settings. Of particular interest, we have identified an emerging part-level semantics that can help the model with image recognition both on global and local scales. We identify that the semantic patterns learned in patch tokens, which sufficiently lack in the off-line tokenizer as in BEiT [^4], helps the model to be advanced in linear classification and robustness against common image corruptions. When it is transferred to downstream tasks, we show that in downstream tasks related to image classification, object detection, instance segmentation, and semantic segmentation, iBOT surpasses previous methods with nontrivial margins. All of the evidence demonstrates that iBOT has largely closed the gap of masked modeling pre-training between language and vision Transformers.

## 2 Preliminaries

### 2.1 Masked Image Modeling as Knowledge Distillation

Masked image modeling (MIM), which takes a similar formulation as MLM in BERT, has been proposed in several recent works [^4] [^41]. Specifically, for an image token sequence ${\bm{x}}=\{{\bm{x}}_{i}\}_{i=1}^{N}$, MIM first samples a random mask ${\bm{m}}\in\{0,1\}^{N}$ according to a prediction ratio $r$, where $N$ is the number of tokens. The patch token ${\bm{x}}_{i}$ where $m_{i}$ being $1$, denoted as $\tilde{\bm{x}}\triangleq\{{\bm{x}}_{i}\ |\ m_{i}=1\}$, are then replaced with a mask token ${\bm{e}}_{\texttt{[MASK]}}$, yielding a corrupted image $\hat{\bm{x}}\triangleq\{\hat{\bm{x}}_{i}\ |\ (1-m_{i}){\bm{x}}_{i}+m_{i}{\bm{e}}_{\texttt{[MASK]}}\}_{i=1}^{N}$. MIM is to recover the masked tokens $\tilde{\bm{x}}$ from the corrupted image $\hat{\bm{x}}$, *i.e*., to maximize: $\mathrm{log}\ q_{\bm{\theta}}(\tilde{\bm{x}}|\hat{\bm{x}})\approx\sum_{i=1}^{N}\ m_{i}\cdot\mathrm{log}\ q_{\bm{\theta}}({\bm{x}}_{i}|\hat{\bm{x}})$, where $\approx$ holds with an independence assumption that each masked token can be reconstructed separately. In BEiT [^4], $q_{\bm{\theta}}$ is modelled as a categorical distribution and the task is to minimize

$$
-\sum_{i=1}^{N}\ m_{i}\cdot P_{\bm{\phi}}({\bm{x}}_{i})^{\mathrm{T}}\ \mathrm{log}\ P_{\bm{\theta}}(\hat{\bm{x}}_{i}),
$$

where $P(\cdot)$ transforms the input to a probability distribution over $K$ dimensions, and ${\bm{\phi}}$ is parameters of a discrete VAE [^36] that clusters image patches into $K$ categories and assigns each patch token a one-hot encoding identifying its category. We note this loss is formulated similarly to knowledge distillation [^24], where knowledge is distilled from a pre-fixed tokenizer parameterized by ${\bm{\phi}}$ to current model parameterized by ${\bm{\theta}}$.

### 2.2 Self-Distillation

Self-distillation, proposed recently in DINO [^8], distills knowledge not from posterior distributions $P_{\bm{\phi}}({\bm{x}})$ but past iterations of model itself $P_{{\bm{\theta^{\prime}}}}({\bm{x}})$ and is cast as a discriminative self-supervised objective. Given the training set $\mathcal{I}$, an image ${\bm{x}}\sim\mathcal{I}$ is sampled uniformly, over which two random augmentations are applied, yielding two distorted views ${\bm{u}}$ and ${\bm{v}}$. The two distorted views are then put through a teacher-student framework to get the predictive categorical distributions from the \[CLS\] token: ${\bm{v}}_{t}^{\texttt{[CLS]}}=P_{{\bm{\theta}}^{\prime}}^{\texttt{[CLS]}}({\bm{v}})$ and ${\bm{u}}_{s}^{\texttt{[CLS]}}=P_{{\bm{\theta}}}^{\texttt{[CLS]}}({\bm{u}})$. The knowledge is distilled from teacher to student by minimizing their cross-entropy, formulated as

$$
\mathcal{L}_{\texttt{[CLS]}}=-P_{\bm{\theta^{\prime}}}^{\texttt{[CLS]}}({\bm{v}})^{\mathrm{T}}\ \mathrm{log}\ P_{\bm{\theta}}^{\texttt{[CLS]}}({\bm{u}}).
$$

The teacher and the student share the same architecture consisting of a backbone $f$ (*e.g*., ViT) and a projection head $h^{\texttt{[CLS]}}$. The parameters of the student network ${\bm{\theta}}$ are Exponentially Moving Averaged (EMA) to the parameters of teacher network ${\bm{\theta^{\prime}}}$. The loss is symmetrized by averaging with another cross-entropy term between ${\bm{v}}_{s}^{\texttt{[CLS]}}$ and ${\bm{u}}_{t}^{\texttt{[CLS]}}$.

## 3 iBOT

We motivate our method by identifying the similar formulation of Eq. (1) and Eq. (2). A visual tokenizer parameterized by online ${\bm{\theta^{\prime}}}$ instead of pre-fixed ${\bm{\phi}}$ thus arises naturally. In this section, we present iBOT, casting self-distillation as a token-generation self-supervised objective and perform MIM via self-distillation. We illustrate the framework of iBOT in Fig. 3 and demonstrate the pseudo-code in Appendix A. In Sec. 3.2, we briefly introduce the architecture and pre-training setup.

### 3.1 Framework

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x5.png)

Figure 3: Overview of iBOT framework, performing masked image modeling with an online tokenizer. Given two views 𝒖 {\\bm{u}} and 𝒗 {\\bm{v}} of an image 𝒙 {\\bm{x}}, each view is passed through a teacher network h t ∘ f subscript ℎ 𝑡 𝑓 h\_{t}\\circ f\_{t} and a student network s 𝑠 h\_{s}\\circ f\_{s}. iBOT minimizes two losses. The first loss ℒ \[CLS\] \\mathcal{L}\_{\\texttt{\[CLS\]}} is self-distillation between cross-view tokens. The second loss MIM \\mathcal{L}\_{\\mathrm{MIM}} is self-distillation between in-view patch tokens, with some tokens masked and replaced by 𝒆 \[MASK\] {\\bm{e}}\_{\\texttt{\[MASK\]}} for the student network. The objective is to reconstruct the masked tokens with the teacher networks’ outputs as supervision.

First, we perform blockwise masking [^4] on the two augmented views ${\bm{u}}$ and ${\bm{v}}$ and obtain their masked views $\hat{\bm{u}}$ and $\hat{\bm{v}}$. Taking $\hat{\bm{u}}$ as an example for simplicity, the student network outputs for the masked view $\hat{\bm{u}}$ projections of its patch tokens $\hat{\bm{u}}_{s}^{\mathrm{patch}}=P_{{\bm{\theta}}}^{\mathrm{patch}}(\hat{\bm{u}})$ and the teacher network outputs for the non-masked view ${\bm{u}}$ projections of its patch tokens ${\bm{u}}_{t}^{\mathrm{patch}}=P_{{\bm{\theta^{\prime}}}}^{\mathrm{patch}}({\bm{u}})$. We here define the training objective of MIM in iBOT as

$$
\mathcal{L}_{\mathrm{MIM}}=-\sum_{i=1}^{N}\ m_{i}\cdot P_{\bm{\theta^{\prime}}}^{\mathrm{patch}}({\bm{u}}_{i})^{\mathrm{T}}\ \mathrm{log}\ P_{\bm{\theta}}^{\mathrm{patch}}(\hat{\bm{u}}_{i}).
$$

We symmetrize the loss by averaging with another CE term between $\hat{\bm{v}}_{s}^{\mathrm{patch}}$ and ${\bm{v}}_{t}^{\mathrm{patch}}$.

The backbone together with the projection head of teacher network $h_{t}^{\mathrm{patch}}\circ f_{t}$ is, therefore, a visual tokenizer that generates online token distributions for each masked patch token. The tokenizer used in iBOT is jointly learnable to MIM objective without a need of being pre-trained in an extra stage, a bonus feature of which is now its domain knowledge can be distilled from the current dataset rather than fixed to the specified dataset.

To ensure that the online tokenizer is semantically-meaningful, we perform self-distillation on \[CLS\] token of cross-view images such that visual semantics can be obtained via bootstrapping, as achieved by the majority of the self-supervised methods [^20] [^18] [^8]. In practice, iBOT works with $\mathcal{L}_{\texttt{[CLS]}}$ in Eq. (2) proposed in DINO [^8], except that now we have $\hat{\bm{u}}_{s}^{\texttt{[CLS]}}$ instead of ${\bm{u}}_{s}^{\texttt{[CLS]}}$ as input for the student network. To further borrow the capability of semantics abstraction acquired from self-distillatin on \[CLS\] token, we share the parameters of projection heads for \[CLS\] token and patch tokens, *i.e*., $h_{s}^{\texttt{[CLS]}}=h_{s}^{\mathrm{patch}}$, $h_{t}^{\texttt{[CLS]}}=h_{t}^{\mathrm{patch}}$. We empirically find that it produces better results than using separate heads.

Unlike tokenized words whose semantics are almost certain, image patch is ambiguous in its semantic meaning. Therefore, tokenization as one-hot discretization can be sub-optimal for images. In iBOT, we use the token distribution after softmax instead of the one-hot token id as a supervisory signal, which plays an important role in iBOT pre-training as shown in Tab. 18.

### 3.2 Implementation

##### Architecture.

We use the Vision Transformers [^17] and Swin Transformers [^30] with different amounts of parameters, ViT-S/16, ViT-B/16, ViT-L/16, and Swin-T/{7,14} as the backbone $f$. For ViTs, /16 denotes the patch size being $16$. For Swins, $/\{7,14\}$ denotes the window size being $7$ or $14$. We pre-train and fine-tune the Transformers with $224$ -size images, so the total number of patch tokens is $196$. The projection head $h$ is a $3$ -layer MLPs with $l_{2}$ -normalized bottleneck following DINO [^8]. Towards a better design to acquire visual semantics, we studied different sharing strategies between projection heads $h^{\texttt{[CLS]}}$ and $h^{\mathrm{patch}}$, considering that semantics obtained in distillation on \[CLS\] token helps the training of MIM on patch tokens. We empirically find that sharing the entire head prompts the best performance. We set the output dimension of the shared head to $8192$.

##### Pre-Training Setup.

We by default pre-train iBOT on ImageNet-1K [^14] training set with AdamW [^31] optimizer and a batch size of $1024$. We pre-train iBOT with ViT-S/16 for $800$ epochs, ViT-B/16 for $400$ epochs, ViT-L/16 for $250$ epochs, and Swin-T/{7,14} for $300$ epochs. We also pre-train on ImageNet-22K training set with ViT-B/16 for $80$ epochs and ViT-L/16 for $50$ epochs. The learning rate is linearly ramped up during the first $10$ epochs to its base value scaled with the total batch size: $\mathrm{lr}=5e^{-4}\times\mathrm{batch\_size}/256$. We use random MIM, with prediction ratio $r$ set as $00$ with a probability of $0.5$ and uniformly sampled from range \[$0.1$, $0.5$\] with a probability of $0.5$. We sum $\mathcal{L}_{\texttt{[CLS]}}$ and $\mathcal{L}_{\mathrm{MIM}}$ up without scaling.

## 4 Experiment

We first transfer iBOT to downstream tasks, following the standard evaluation protocols adopted in prior arts, the details of which are delayed in Appendix C. We then study several interesting properties of Transformers pre-trained with iBOT. Finally, we give a brief ablation study on the crucial composing of iBOT.

### 4.1 Classification on ImageNet-1K

We consider five classification protocols on ImageNet-1K: $k$ -NN, linear probing, fine-tuning, semi-supervised learning, and unsupervised learning.

Table 1: $k$ -NN and linear probing on ImageNet-1K. <sup>†</sup> denotes using selective kernel. <sup>‡</sup> denotes pre-training on ImageNet-22K.

<table><tbody><tr><td>Method</td><td>Arch.</td><td>Par.</td><td>im/s</td><td>Epo.<sup>1</sup></td><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>Lin.</td></tr><tr><td colspan="7">SSL big ResNets</td></tr><tr><td>MoCov3</td><td>RN50</td><td>23</td><td>1237</td><td>1600</td><td>-</td><td>74.6</td></tr><tr><td>SwAV</td><td>RN50</td><td>23</td><td>1237</td><td>2400</td><td>65.7</td><td>75.3</td></tr><tr><td>DINO</td><td>RN50</td><td>23</td><td>1237</td><td>3200</td><td>67.5</td><td>75.3</td></tr><tr><td>BYOL</td><td>RN200w2</td><td>250</td><td>123</td><td>2000</td><td>73.9</td><td>79.6</td></tr><tr><td>SCLRv2</td><td>RN152w3 <sup>†</sup></td><td>794</td><td>46</td><td>2000</td><td>73.1</td><td>79.8</td></tr><tr><td colspan="7">SSL Transformers</td></tr><tr><td>MoCov3</td><td>ViT-S/16</td><td>21</td><td>1007</td><td>1200</td><td>-</td><td>73.4</td></tr><tr><td>MoCov3</td><td>ViT-B/16</td><td>85</td><td>312</td><td>1200</td><td>-</td><td>76.7</td></tr><tr><td>SwAV</td><td>ViT-S/16</td><td>21</td><td>1007</td><td>2400</td><td>66.3</td><td>73.5</td></tr><tr><td>DINO</td><td>ViT-S/16</td><td>21</td><td>1007</td><td>3200</td><td>74.5</td><td>77.0</td></tr><tr><td>DINO</td><td>ViT-B/16</td><td>85</td><td>312</td><td>1600</td><td>76.1</td><td>78.2</td></tr><tr><td>EsViT</td><td>Swin-T/7</td><td>28</td><td>726</td><td>1200</td><td>75.7</td><td>78.1</td></tr><tr><td>EsViT</td><td>Swin-T/14</td><td>28</td><td>593</td><td>1200</td><td>77.0</td><td>78.7</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>21</td><td>1007</td><td>3200</td><td>75.2</td><td>77.9</td></tr><tr><td>iBOT</td><td>Swin-T/7</td><td>28</td><td>726</td><td>1200</td><td>75.3</td><td>78.6</td></tr><tr><td>iBOT</td><td>Swin-T/14</td><td>28</td><td>593</td><td>1200</td><td>76.2</td><td>79.3</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>85</td><td>312</td><td>1600</td><td>77.1</td><td>79.5</td></tr><tr><td>iBOT</td><td>ViT-L/16</td><td>307</td><td>102</td><td>1200</td><td>78.0</td><td>81.0</td></tr><tr><td>iBOT <sup>‡</sup></td><td>ViT-L/16</td><td>307</td><td>102</td><td>200</td><td>72.9</td><td>82.3</td></tr></tbody></table>

Table 2: Fine-tuning on ImageNet-1K.

| Method | Arch. | Epo.<sup>1</sup> | Acc. |
| --- | --- | --- | --- |
| Rand. | ViT-S/16 | \- | 79.9 |
| MoCov3 | ViT-S/16 | 600 | 81.4 |
| DINO | ViT-S/16 | 3200 | 82.0 |
| iBOT | ViT-S/16 | 3200 | 82.3 |
| Rand. | ViT-B/16 | \- | 81.8 |
| MoCov3 | ViT-B/16 | 600 | 83.2 |
| BEiT | ViT-B/16 | 800 | 83.4 |
| DINO | ViT-B/16 | 1600 | 83.6 |
| iBOT | ViT-B/16 | 1600 | 84.0 |
| MoCov3 | ViT-L/16 | 600 | 84.1 |
| iBOT | ViT-L/16 | 1000 | 84.8 |
| BEiT | ViT-L/16 | 800 | 85.2 |

Table 3: Fine-tuning on ImageNet-1K. Pre-training on ImageNet-22K.

| Method | Arch. | Epo.<sup>1</sup> | Acc. |
| --- | --- | --- | --- |
| BEiT | ViT-B/16 | 150 | 83.7 |
| iBOT | ViT-B/16 | 320 | 84.4 |
| BEiT | ViT-L/16 | 150 | 86.0 |
| iBOT | ViT-L/16 | 200 | 86.6 |
| iBOT | ViT <sub>512</sub> -L/16 | 200 | 87.8 |

##### k𝑘k-NN and Linear Probing.

To evaluate the quality of pre-trained features, we either use a $k$ -nearest neighbor ($k$ -NN) classifier or a linear classifier on the frozen representation. We follow the evaluation protocols in DINO [^8]. For $k$ -NN evaluation, we sweep over different numbers of nearest neighbors. For linear evaluation, we sweep over different learning rates. In Tab. 3, our method reaches a linear probing accuracy 77.9% with ViT-S/16, a linear probing accuracy 79.5% with ViT-B/16, and a $k$ -NN accuracy 78.0% and linear probing accuracy 81.0% with ViT-L/16, achieving state-of-the-art performance. With Swin-T/{7,14}, iBOT achieves a linear probing accuracy of 78.6% and 79.3% respectively.With ViT-L/16 and ImageNet-22K as pre-training data, iBOT further achieves a linear probing accuracy 82.3%, surpassing previous state of the art, 81.3% with Swin-B/14 by EsViT [^26]. A linear probing accuracy of 79.5% with ViT-B/16 is comparable to 79.8% by SimCLRv2 with RN152 ($3\times$) <sup>†</sup> but with $10\times$ less parameters. We underline that the performance gain over DINO gets larger (0.9% w/ ViT-S versus 1.3% w/ ViT-B) with more parameters, suggesting iBOT is more scalable to larger models.

##### Fine-Tuning.

We study the fine-tuning on ImageNet-1K and focus on the comparison with self-supervised methods for Transformers and its supervised baseline (Rand.) [^43]. As shown in Tab. 3, iBOT achieves an 82.3%, 84.0%, and 84.8% top-1 accuracy with ViT-S/16, ViT-B/16, and ViT-L/16, respectively. As shown in Tab. 3, iBOT pre-trained with ImageNet-22K achieves 84.4% and 86.6% top-1 accuracy with ViT-B/16 and ViT-L/16, respectively, outperforming ImageNet-22K pre-trained BEiT by 0.7% and 0.6%. When fine-tuned on an image size of 512, we achieve 87.8% accuracy. We note that, with ViT-L/16, iBOT is 0.4% worse than BEiT using 1K data but 0.6% better using 22K data. This implies that iBOT requires more data to train larger model.

Table 4: Semi-supervised learning on ImageNet-1K. 1% and 10% denotes label fraction. SD denotes self-distillation.

| Method | Arch. | 1% | 10% |
| --- | --- | --- | --- |
| SimCLRv2 | RN50 | 57.9 | 68.1 |
| BYOL | RN50 | 53.2 | 68.8 |
| SwAV | RN50 | 53.9 | 70.2 |
| SimCLRv2+SD | RN50 | 60.0 | 70.5 |
| DINO | ViT-S/16 | 60.3 | 74.3 |
| iBOT | ViT-S/16 | 61.9 | 75.1 |

Table 5: Unsupervised learning on ImageNet-1K. <sup>†</sup> denotes $k$ -means clustering on frozen features.

| Method | Arch. | ACC | ARI | NMI | FMI |
| --- | --- | --- | --- | --- | --- |
| Self-label <sup>†</sup> | RN50 | 30.5 | 16.2 | 75.4 | \- |
| InfoMin <sup>†</sup> | RN50 | 33.2 | 14.7 | 68.8 | \- |
| SCAN | RN50 | 39.9 | 27.5 | 72.0 | \- |
| DINO | ViT-S/16 | 41.4 | 29.8 | 76.8 | 32.8 |
| iBOT | ViT-S/16 | 43.4 | 32.8 | 78.6 | 35.6 |

##### Semi-Supervised and Unsupervised Learning.

For semi-supervised learning, we focus our comparison with methods following the unsupervised pre-train, supervised fine-tune paradigm. As shown in Tab. 5, iBOT advances DINO by 1.6% and 0.8% using 1% and 10% data, respectively, suggesting a higher label efficiency. For unsupervised learning, we use standard evaluation metrics, including accuracy (ACC), adjusted random index (ARI), normalized mutual information (NMI), and Fowlkes-Mallows index (FMI). We compare our methods to SimCLRv2 [^10], Self-label [^2], InfoMin [^42], and SCAN [^44]. As shown in Tab. 5, we achieve a 32.8% NMI, outperforming the previous state of the art by 1.8%, suggesting MIM helps the model learn stronger visual semantics on a global scale.

### 4.2 Downstream Tasks

Table 6: Object detection (Det.) & instance segmentation (ISeg.) on COCO and Semantic segmentation (Seg.) on ADE20K. We report the results of ViT-S/16 (left) and ViT-B/16 (right). Seg.<sup>†</sup> denotes using a linear head for semantic segmentation.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Arch.</td><td rowspan="2">Param.</td><td>Det.</td><td>ISeg.</td><td>Seg.</td></tr><tr><td>AP <sup>b</sup></td><td>AP <sup>m</sup></td><td>mIoU</td></tr><tr><td>Sup.</td><td>Swin-T</td><td>29</td><td>48.1</td><td>41.7</td><td>44.5</td></tr><tr><td>MoBY</td><td>Swin-T</td><td>29</td><td>48.1</td><td>41.5</td><td>44.1</td></tr><tr><td>Sup.</td><td>ViT-S/16</td><td>21</td><td>46.2</td><td>40.1</td><td>44.5</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>21</td><td>49.4</td><td>42.6</td><td>45.4</td></tr></tbody></table>

<table><tbody><tr><td rowspan="2">Method</td><td>Det.</td><td>ISeg.</td><td>Seg.<sup>†</sup></td><td>Seg.</td></tr><tr><td>AP <sup>b</sup></td><td>AP <sup>m</sup></td><td>mIoU</td><td>mIoU</td></tr><tr><td>Sup.</td><td>49.8</td><td>43.2</td><td>35.4</td><td>46.6</td></tr><tr><td>BEiT</td><td>50.1</td><td>43.5</td><td>27.4</td><td>45.8</td></tr><tr><td>DINO</td><td>50.1</td><td>43.4</td><td>34.5</td><td>46.8</td></tr><tr><td>iBOT</td><td>51.2</td><td>44.2</td><td>38.3</td><td>50.0</td></tr></tbody></table>

##### Object Detection and Instance Segmentation on COCO.

Object detection and instance segmentation require simultaneous object location and classification.We consider Cascade Mask R-CNN [^5] [^19] that produces bounding boxes and instance masks simultaneously on COCO dataset [^28]. Several recent works [^30] [^45] proposes Vision Transformers that suit dense downstream tasks. To compare, we include the results of supervised Swin-T [^30] which shares approximate parameter numbers with ViT-S/16 and its self-supervised counterpart MoBY [^52] in Tab. 6. iBOT improves ViT-S’s AP <sup>b</sup> from 46.2 to 49.4 and AP <sup>m</sup> from 40.1 to 42.6, surpassing both supervised Swin-T and its self-supervised counterpart by a nontrivial margin. With ViT-B/16, iBOT achieves an AP <sup>b</sup> of 51.2 and an AP <sup>m</sup> of 44.2, surpassing previous best results by a large margin.

##### Semantic Segmentation on ADE20K.

Semantic segmentation can be seen as a pixel-level classification problem. We mainly consider two segmentation settings on ADE20K dataset [^56]. First, similar to linear evaluation protocol in classification, we evaluate on the fixed patch features and only fine-tune a linear layer, which gives us a more explicit comparison of the quality of representations. Second, we use the task layer in UPerNet [^51] and fine-tune the entire network. From Tab. 6, we can see that iBOT advances its supervised baseline with ViT-S/16 with a large margin of 0.9 on mIoU, surpassing Swin-T. With ViT-B/16, iBOT advances previous best methods DINO by 3.2 on mIoU with UperNet. We notice a performance drop of BEiT using linear head, indicating BEiT’s features lack local semantics. As analyzed later, the property of strong local semantics induces a 2.9 mIoU gain compared to the supervised baseline with a linear head.

Table 7: Transfer learning by fine-tuning pre-trained models on different datasets. We report Top-1 accuracy of ViT-S/16 (left) and ViT-B/16 (right).

| Method | Cif <sub>10</sub> | Cif <sub>100</sub> | iNa <sub>18</sub> | iNa <sub>19</sub> | Flwrs | Cars |
| --- | --- | --- | --- | --- | --- | --- |
| Rand. | 99.0 | 89.5 | 70.7 | 76.6 | 98.2 | 92.1 |
| BEiT | 98.6 | 87.4 | 68.5 | 76.5 | 96.4 | 92.1 |
| DINO | 99.0 | 90.5 | 72.0 | 78.2 | 98.5 | 93.0 |
| iBOT | 99.1 | 90.7 | 73.7 | 78.5 | 98.6 | 94.0 |

| Method | Cif <sub>10</sub> | Cif <sub>100</sub> | iNa <sub>18</sub> | iNa <sub>19</sub> | Flwrs | Cars |
| --- | --- | --- | --- | --- | --- | --- |
| Rand. | 99.0 | 90.8 | 73.2 | 77.7 | 98.4 | 92.1 |
| BEiT | 99.0 | 90.1 | 72.3 | 79.2 | 98.0 | 94.2 |
| DINO | 99.1 | 91.7 | 72.6 | 78.6 | 98.8 | 93.0 |
| iBOT | 99.2 | 92.2 | 74.6 | 79.6 | 98.9 | 94.3 |

##### Transfer Learning.

We study transfer learning where we pre-train on ImageNet-1K and fine-tune on several smaller datasets.We follow the training recipe and protocol used in [^17]. The results are demonstrated in Tab. 7. While the results on several datasets (*e.g*., CIFAR10, CIFAR100, Flowers, and Cars) have almost plateaued, iBOT consistently performs favorably against other SSL frameworks, achieving state-of-the-art transfer results. We observe greater performance gain over DINO in larger datasets like iNaturalist18 and iNaturalist19, indicating the results are still far from saturation. We also find that with larger models, we typically get larger performance gain compared with DINO (*e.g*., 1.7% with ViT/S-16 versus 2.0% with ViT-B/16 on iNaturalist18, and 0.3% with ViT/S-16 versus 1.0% with ViT-B/16 on iNaturalist19).

### 4.3 Properties of ViT trained with MIM

In the previous sections, we have shown the priority of iBOT on various tasks and datasets. To reveal the strengths of iBOT pre-trained Vision Transformers, we analyze its property from several aspects.

#### 4.3.1 Discovering the Pattern Layout of Image Patches

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x6.png)

Figure 4: Pattern layout of patch tokens. Two left figures showcase patterns, headlight of the vehicle and ear of the dog, that share part semantics. Two right figures showcase patterns, stripped curly surface, that share part textures.

##### What Patterns Does MIM Learn?

The output from the projection head used for self-distillation depicts for patch token a probabilistic distribution. To help understand what patterns MIM induces to learn, we visualize several pattern layouts. We use $800$ -epoch pre-trained ViT-S/16 and visualize the top- $36$ patches with the highest confidence on ImageNet-1K validation set. We visualize a $5\times$ context for each $16\times 16$ patch (colored orange). We observe the emergence of both high-level semantics and low-level details. As shown in Fig. 4, several patches are grouped with clear semantic meaning, *e.g*., headlight and dog’s ear. Such behavior stands a distinct contrast with the offline tokenizer used in BEiT [^4], which encapsulates mostly low-level details as shown in Fig. 16. Apart from patch patterns that share high-level semantics, we also observe clusters accounting for low-level textures, indicating the diversity of learned part patterns. The comparison with previous work [^8] [^4] and the visualization of more pattern layouts are provided in Appendix G.1.

##### How Does MIM Help Image Recognition?

To illustrate how the property of better part semantics can help image recognition, we use part-wise linear classification to study the relationship between representations of patch tokens and \[CLS\] token. Specifically, we average $k$ patch tokens with the top- $k$ highest self-attention scores. The results are demonstrated in Fig. 6. While the performance gap between DINO and iBOT is only 0.9% in the standard setting (77.9% v.s. 77.0%) with $[\texttt{CLS}]$ token, we observe that iBOT outperforms DINO when using the patch representations directly. We observe that using top- $56$ patch tokens yields an optimal result, and iBOT is 5.9% higher than DINO. The performance gap becomes more prominent when using fewer patch tokens. When using only the patch token with the highest self-attention score, iBOT advances by 17.9%. These results reveal much semantic information in iBOT representations for patch tokens, which helps the model to be more robust to the loss of local details and further boosts its performance on image-level recognition.

#### 4.3.2 Discriminative Parts in Self-Attention Map

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x7.png)

Figure 5: Part-wise linear probing accuracy. Top- k 𝑘 tokens with the highest attention scores are averaged for classification.

To analyze, we visualize the self-attention map with ViT-S/16. We choose \[CLS\] token as the query and visualize attention maps from different heads of the last layer with different colors, as shown in Fig. 6. Of particular interest, we indicate that iBOT shows a solid ability to separate different objects or different parts of one object apart. For example, in the leftmost figure, we observe iBOT fairly distinct the bird from the tree branch. Also, iBOT focuses mainly on the discriminative parts of the object (*e.g*., the wheel of the car, the beak of the bird). These properties are crucial for iBOT to excel at image recognition, especially in complicated scenarios with object occlusion or distracting instances. While these properties are not unique strengths brought by MIM and we observe similar behaviors in DINO, we show in Appendix G.2 that iBOT generally gives better visualized results.

#### 4.3.3 Robustness

Table 8: Robustness evaluation of pre-trained models against background change, occlusion, and out-of-distribution examples.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="7">Background Change</td><td>Clean</td><td colspan="2">Occlusion</td><td colspan="2">Out-of-Dist.</td><td>Clean</td></tr><tr><td>O.F.</td><td>M.S.</td><td>M.R.</td><td>M.N.</td><td>N.F.</td><td>O.BB.</td><td>O.BT.</td><td>IN-9</td><td><math><semantics><msub><mi>S</mi><mn>.5</mn></msub> <apply><csymbol>subscript</csymbol> <ci>𝑆</ci><cn>.5</cn></apply> <annotation>S_{.5}</annotation></semantics></math></td><td><math><semantics><mrow><mi>N</mi> <mo></mo><msub><mi>S</mi><mn>.5</mn></msub></mrow> <apply><ci>𝑁</ci> <apply><csymbol>subscript</csymbol> <ci>𝑆</ci><cn>.5</cn></apply></apply> <annotation>NS_{.5}</annotation></semantics></math></td><td>IN-A</td><td>IN-C <math><semantics><mo>↓</mo> <ci>↓</ci> <annotation>\downarrow</annotation></semantics></math></td><td>IN</td></tr><tr><td>DINO</td><td>89.2</td><td>89.2</td><td>80.4</td><td>78.3</td><td>52.0</td><td>21.9</td><td>18.4</td><td>96.4</td><td>64.7</td><td>42.0</td><td>12.3</td><td>51.7</td><td>77.0</td></tr><tr><td>iBOT</td><td>90.9</td><td>89.7</td><td>81.7</td><td>80.3</td><td>53.5</td><td>22.7</td><td>17.4</td><td>96.8</td><td>65.9</td><td>43.4</td><td>13.8</td><td>48.1</td><td>77.9</td></tr></tbody></table>

The above-mentioned properties brought by MIM objective can improve the model’s robustness to uncommon examples. We quantitatively benchmark robustness in terms of $3$ aspects: background change, occlusion, and out-of-distribution examples, with a ViT-S/16 pre-trained for $800$ epochs and then linearly evaluated for $100$ epochs. Results are shown in Tab. 8. For background change, we study images under $7$ types of change, detailed in Appendix D. iBOT is more robust against background changes except for O.BT.. For occlusion, we study the linear accuracy with salient and non-salient patch dropping following [^33] with an information loss ratio of $0.5$. iBOT has a smaller performance drop under both settings. For out-of-distribution examples, we study natural adversarial examples in ImageNet-A [^23] and image corruptions in ImageNet-C [^22]. iBOT has higher accuracy on the ImageNet-A and a smaller mean corruptions error (mCE) on the ImageNet-C.

### 4.4 Ablation Study on Tokenizer

In this section, we ablate the importance of using a semantically meaningful tokenizer using a $300$ -epoch pre-trained ViT-S/16 with a prediction ratio $r=0.3$ and without multi-crop augmentation. Additional ablations are given in Appendix E. iBOT works with self-distillation on \[CLS\] token with cross-view images ($\mathcal{L}_{\texttt{[CLS]}}$) to acquire visual semantics. To verify, we conduct experiments to perform MIM without $\mathcal{L}_{\texttt{[CLS]}}$ or with alternative models as visual tokenizer. Specifically, $\circ$ denotes a standalone DINO and $\triangle$ denotes a pre-tranined DALL-E encoder [^36].

Table 9: Effect of design choices of semantically meaningful tokenization.

<table><tbody><tr><td>Method</td><td><math><semantics><msub><mi>ℒ</mi> <mi>MIM</mi></msub> <apply><csymbol>subscript</csymbol> <ci>ℒ</ci> <ci>MIM</ci></apply> <annotation>\mathcal{L}_{\mathrm{MIM}}</annotation></semantics></math></td><td><math><semantics><msub><mi>ℒ</mi> <mtext>[CLS]</mtext></msub> <apply><csymbol>subscript</csymbol> <ci>ℒ</ci> <ci><mtext>[CLS]</mtext></ci></apply> <annotation>\mathcal{L}_{\texttt{[CLS]}}</annotation></semantics></math></td><td>SH</td><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>Lin.</td><td>Fin.</td></tr><tr><td>iBOT</td><td>✓</td><td>✓</td><td>✓</td><td>69.1</td><td>74.2</td><td>81.5</td></tr><tr><td></td><td>✓</td><td>✓</td><td>✗</td><td>69.0</td><td>73.8</td><td>81.5</td></tr><tr><td></td><td>✓</td><td>✗</td><td>-</td><td>9.5</td><td>29.8</td><td>79.4</td></tr><tr><td></td><td><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math></td><td>✗</td><td>-</td><td>44.3</td><td>60.0</td><td>81.7</td></tr><tr><td>BEiT</td><td><math><semantics><mi>△</mi> <ci>△</ci> <annotation>\triangle</annotation></semantics></math></td><td>✗</td><td>-</td><td>6.9</td><td>23.5</td><td>81.4</td></tr><tr><td>DINO</td><td>✗</td><td>✓</td><td>-</td><td>67.9</td><td>72.5</td><td>80.6</td></tr><tr><td>BEiT + DINO</td><td><math><semantics><mi>△</mi> <ci>△</ci> <annotation>\triangle</annotation></semantics></math></td><td>✓</td><td>-</td><td>48.0</td><td>62.7</td><td>81.2</td></tr><tr><td colspan="7"><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math>: standalone DINO (w/o mcrop, <math><semantics><mn>300</mn> <cn>300</cn> <annotation>300</annotation></semantics></math> -epoch)</td></tr><tr><td colspan="7"><math><semantics><mi>△</mi> <ci>△</ci> <annotation>\triangle</annotation></semantics></math>: pre-trained DALL-E encoder</td></tr></tbody></table>

We find that performing MIM without $\mathcal{L}_{\texttt{[CLS]}}$ leads to undesirable results of 9.5% $k$ -NN accuracy and 29.8% linear accuracy, indicating that visual semantics can hardly be obtained with only MIM. While semantics emerges with a standalone DINO as a visual tokenizer, it is still far from reaching a decent result (44.3% versus 69.1% in $k$ -NN accuracy). Comparing iBOT with multi-tasking of DINO and BEiT (DINO+BEiT), we see the strengths of merging the semantics acquired by self-distillation with the visual tokenizer with an 11.5% advance in linear probing and 0.3% in fine-tuning. Moreover, we empirically observe a performance improvement using a Shared projection Head (SH) for \[CLS\] token and patch tokens, which shares the semantics acquired in \[CLS\] token to MIM.

## 5 Related Work

##### Visual Representation Learning.

Most self-supervised methods assume an augmentation invariance of images and achieve so by enforcing similarity over distorted views of one image while avoiding model collapse. Avoiding collapse can be achieved by noise-contrastive estimation with negative samples [^49] [^20] [^9], introducing asymmetric network [^18] [^11], or explicitly enforcing the distribution of image distribution over the channel to be uniform as well as one-hot [^7] [^1] [^8]. In fact, the idea of simultaneously enforcing distribution uniform and one-hot is hidden from earlier studies performing representation learning via clustering [^6] [^7] [^54], where the cluster assignment naturally meets these two requirements. Other methods rely on handcrafted pretext tasks and assume the image representation should instead be aware of image augmentation by solving image jigsaw puzzle [^34] [^47], predicting rotation [^25] or relative position [^16].

##### Masked Prediction in Images.

Predicting masked images parts is a popular self-supervised pretext task drawing on the idea of auto-encoding and has been previously achieved by either recovering raw pixels [^35] [^3] [^27] or mask contrastive learning [^21] [^55]. Recently, it is formulated into MIM [^4] [^41] with a discrete VAE [^38] [^36] as visual tokenizer. As a counterpart of MLM in NLP, MIM eases masked prediction into a classification problem supervised by labels output from the tokenizer, mitigating the problem of excessive focus on high-frequency details. Concurrently, masked image prediction has been explored in the field of multi-modality, *i.e*., vision-language representation learning. These methods operate on local regions instead of global images thus reply on pre-trained detection models, *i.e*., Faster-RCNN [^37] to propose regions of interest. [^40] [^32] [^13] perform masked region classification tasking the category distribution output from the detection model as the ground-truth.

## 6 Conclusion

In this work, we study BERT-like pre-training for Vision Transformers and underline the significance of a semantically meaningful visual tokenizer. We present a self-supervised framework iBOT that performs masked image modeling via self-distillation with an online tokenizer, achieving state-of-the-art results on downstream tasks related to classification, object detection, instance segmentation, and semantic segmentation. Of particular interest, we identify an emerging part-level semantics for models trained with MIM that helps for not only recognition accuracy but also robustness against common image corruptions. In the future, we plan to scale up iBOT to a larger dataset (*e.g*., ImageNet-22K) or larger model size (*e.g*., ViT-L/16 and ViT-H/16) and investigate whether MIM can help Vision Transformers more scalable to unlabelled data in the wild.

Acknowledgement Tao Kong is the corresponding author. We would like to acknowledge Feng Wang, Rufeng Zhang, and Zongwei Zhou for helpful discussions. We thank Mathilde Caron, Julien Mairal, and Hugo Touvronfor for sharing details of DINO. We thank Li Dong and Hangbo Bao for sharing details of BEiT.

## References

## Appendix A Pseudocode

Input:

$g_{s}$, $g_{t}$;

// student and teacher network

$C,C^{\prime}$;

// center on \[CLS\] token and patch tokens

$\tau_{s},\tau_{t}$;

// temperature on \[CLS\] token for student and teacher network

$\tau^{\prime}_{s},\tau^{\prime}_{t}$;

// temperature on patch tokens for student and teacher network

$l$;

// momentum rate for network

$m,m^{\prime}$;

// momentum rates for center on \[CLS\] token and patch tokens

$g_{t}$.params = $g_{s}$.params

for *${\bm{x}}\mathrm{\ in\ loader}$* do

${\bm{u}}$, ${\bm{v}}$ = augment(${\bm{x}}$), augment(${\bm{x}}$);

// random views

$\hat{\bm{u}}$, ${\bm{m}}_{u}$ = blockwise\_mask(${\bm{u}}$);

// random block-wise masking

$\hat{\bm{v}}$, ${\bm{m}}_{v}$ = blockwise\_mask(${\bm{v}}$);

// random block-wise masking

$\hat{\bm{u}}_{s}^{\texttt{[CLS]}}$, $\hat{\bm{u}}_{s}^{\mathrm{patch}}$ = $g_{s}$ ($\hat{\bm{u}}$, return\_all\_tok=true);

// $[n,K]$, $[n,S^{2},K]$

$\hat{\bm{v}}_{s}^{\texttt{[CLS]}}$, $\hat{\bm{v}}_{s}^{\mathrm{patch}}$ = $g_{s}$ ($\hat{\bm{v}}$, return\_all\_tok=true);

// $[n,K]$, $[n,S^{2},K]$

${\bm{u}}_{t}^{\texttt{[CLS]}}$, ${\bm{u}}_{t}^{\mathrm{patch}}$ = $g_{t}$ (${\bm{u}}$, return\_all\_tok=true);

// $[n,K]$, $[n,S^{2},K]$

${\bm{v}}_{t}^{\texttt{[CLS]}}$, ${\bm{v}}_{t}^{\mathrm{patch}}$ = $g_{t}$ (${\bm{v}}$, return\_all\_tok=true);

// $[n,K]$, $[n,S^{2},K]$

$\mathcal{L}_{\texttt{[CLS]}}$ = $\mathrm{H}(\hat{\bm{u}}_{s}^{\texttt{[CLS]}},v_{t}^{\texttt{[CLS]}},C,\tau_{s},\tau_{t})$ / 2 $+\ \mathrm{H}(\hat{\bm{v}}_{s}^{\texttt{[CLS]}},{\bm{u}}_{t}^{\texttt{[CLS]}},C,\tau_{s},\tau_{t})$ / 2

$\mathcal{L}_{\mathrm{MIM}}$ = $({\bm{m}}_{u}\cdot\mathrm{H}(\hat{\bm{u}}_{s}^{\mathrm{patch}},{\bm{u}}_{t}^{\mathrm{patch}},C^{\prime},\tau^{\prime}_{s},\tau^{\prime}_{t})$.sum(dim=1) / ${\bm{m}}_{u}$.sum(dim=1) / 2

$+\ ({\bm{m}}_{v}\cdot\mathrm{H}(\hat{\bm{v}}_{s}^{\mathrm{patch}},{\bm{v}}_{t}^{\mathrm{patch}},C^{\prime},\tau^{\prime}_{s},\tau^{\prime}_{t})$.sum(dim=1) / ${\bm{m}}_{v}$.sum(dim=1) / 2

$(\mathcal{L}_{\texttt{[CLS]}}$.mean() $+\mathcal{L}_{\mathrm{MIM}}$.mean()$)$.backward()

update($g_{s}$);

// student, teacher and center update

$g_{t}$.params = $l\cdot$ $g_{t}$.params $+(1-l)\cdot$ $g_{s}$.params

$C$ = $m\cdot C+(1-m)\cdot$ cat(\[${\bm{u}}_{t}^{\texttt{[CLS]}}$, ${\bm{v}}_{t}^{\texttt{[CLS]}}$\]).mean(dim=0)

$C^{\prime}$ = $m^{\prime}\cdot C^{\prime}+(1-m^{\prime})\cdot$ cat(\[${\bm{u}}_{t}^{\mathrm{patch}}$, ${\bm{v}}_{t}^{\mathrm{patch}}$\]).mean(dim=(0, 1))

end for

def *$\mathrm{H}$ (*$s$, $t$, $c$, $\tau_{s}$, $\tau_{t}$*)*:

$t$ = $t$.detach();

// stop gradient

$s$ = softmax($s$ / $\tau_{s}$, dim=1)

$t$ = softmax($(t-c)$ / $\tau_{t}$, dim=1);

// center + sharpen

return $-(t\cdot$ log($s$)).sum(dim=-1);

Algorithm 1 iBOT PyTorch-like Pseudocode w/o multi-crop augmentation

## Appendix B Multi-Crop

The advanced performance of several recent state-of-the-art methods [^8] [^7] relies on multi-crop augmentation, as well as iBOT. In our early experiments, we find the direct usage of multi-crop augmentation leads to instability issues that degrade accuracy. We reveal that these results can be attributed to the distribution mismatch between masked images and non-masked images and can be resolved by minimal changes in iBOT framework.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x9.png)

Figure 7: Computation pipelines for iBOT with or without multi-crop augmentation. (a) iBOT w/o multi-crop augmentation. (b), (c), and (d) are three pipelines w/ multi-crop augmentation. (b) does not perform MIM for local crops, whereas (c) performs MIM for all crops. (d) only performs MIM for one of the two global crops. iBOT uses (b) with random MIM.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x10.png)

Figure 8: Training curves of different multi-crop strategy.

##### Stability of MIM Pre-trained with Multi-Crop.

We first showcase several practices where training instability occurs, shown in Fig. 7. To reveal the instability, we monitor the NMI curves during training for each epoch as shown in Fig. 8. The most intuitive ideas are to compute as (b) or (c). In (b), MIM is only performed on global crops. This pipeline is unstable during training, and we observe a dip in the NMI training curve. We hypothesize that it can be caused by the distribution mismatch of masked global crops and non-masked local crops. To alleviate this, a straightforward solution is to also perform MIM on local crops with an extra computation cost as (c). However, we do not observe this circumvents training instability. We hypothesize that the regions corresponding to patch tokens of the local crops are small in size, in which there exist few meaningful contents to predict. This hypothesis can be supported by the experiments that when we set the local crop scale in (c) from $(0.05,0.4)$ to $(0.2,0.4)$, denoted as (e), the performance drop is mitigated.

##### Stabilizing the Training with Non-Masked Global Crops.

Another solution to alleviate the distribution mismatch between masked global crops and non-masked local crops is to train with non-masked global crops, as shown in (d). In other words, we perform random MIM when training ViT with multi-crop augmentation. This computation pipeline is stable and achieves a substantial performance gain. In practice, to include non-masked global crops in training, we use (b) and randomly choose a prediction ratio between \[$00$, $r\ (r>0)$\] for each image. When the ratio $00$ is chosen, the whole framework excludes MIM and can be seen as DINO. When the ratio $r\ (r>0)$ is chosen, MIM is performed for both of the two global crops. We observe the latter practice performs sightly better since it is more flexible in task composition and data in a batch is mutually independent.

##### Range of Scales in Multi-Crop.

We further study the performance with different local and global scale. Following DINO [^8], we conduct the experiments by tweaking $s$, where $s$ is the scale deviding the local and global crops. The local crops are sampled from (0.05, $s$) and the global crops are sampled from ($s$, 1).

| ViT-S/16, 300 epochs | 0.25 | 0.4 | 0.32 |
| --- | --- | --- | --- |
| $k$ -NN | 74 | 74.3 | 74.6 |

| ViT-B/16, 50 epochs | 0.25 | 0.4 | 0.32 |
| --- | --- | --- | --- |
| $k$ -NN | 70 | 70.1 | 70.4 |

We empirically find that $s=0.32$ yields optimal performance for both small-size and base-size models. Therefore, we use an $s$ of $0.32$ by default.

Table 10: $k$ -NN and linear probing accuracy on ImageNet-1K without multi-crop augmentation (left) and with multi-crop augmentation (right) multi-crop augmentation. We split the table into results without or with multi-crop augmentation.

<table><tbody><tr><td>Method</td><td>Arch</td><td>Param.</td><td>Epo.</td><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>Linear</td></tr><tr><td rowspan="3">MoCov3</td><td>RN50</td><td>23</td><td>800</td><td>-</td><td>74.6</td></tr><tr><td>ViT-S/16</td><td>21</td><td>600</td><td>-</td><td>73.4</td></tr><tr><td>ViT-B/16</td><td>85</td><td>600</td><td>-</td><td>76.7</td></tr><tr><td rowspan="2">DINO</td><td>ViT-S/16</td><td>21</td><td>800</td><td>70.0</td><td>73.7</td></tr><tr><td>ViT-B/16</td><td>85</td><td>400</td><td>68.9</td><td>72.8</td></tr><tr><td></td><td>ViT-S/16</td><td>21</td><td>800</td><td>72.4</td><td>76.2</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>85</td><td>400</td><td>71.2</td><td>76.0</td></tr></tbody></table>

<table><tbody><tr><td>Method</td><td>Arch</td><td>Param.</td><td>Epo.</td><td><math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>Linear</td></tr><tr><td rowspan="2">SwAV</td><td>RN50</td><td>23</td><td>800</td><td>65.7</td><td>75.3</td></tr><tr><td>ViT-S/16</td><td>21</td><td>800</td><td>66.3</td><td>73.5</td></tr><tr><td rowspan="3">DINO</td><td>RN50</td><td>23</td><td>800</td><td>67.5</td><td>75.3</td></tr><tr><td>ViT-S/16</td><td>21</td><td>800</td><td>74.5</td><td>77.0</td></tr><tr><td>ViT-B/16</td><td>85</td><td>400</td><td>76.1</td><td>78.2</td></tr><tr><td></td><td>ViT-S/16</td><td>21</td><td>800</td><td>75.2</td><td>77.9</td></tr><tr><td>iBOT</td><td>ViT-B/16</td><td>85</td><td>400</td><td>76.8</td><td>79.4</td></tr></tbody></table>

##### State-of-the-Art Comparison w/o and w/ Multi-Crop.

Including iBOT, several recent state-of-the-art works [^8] [^7] rely heavily on multi-crop augmentation during pre-training. Except for several specific self-supervised methods [^18], multi-crop works well on most of the self-supervised methods and consistently yields performance gain [^8]. While a more fair comparison with our methods without multi-crop augmentation can be conducted, we believe it is a unique strength of iBOT to work well with multi-crop. In Tab. 10, we categorize the state-of-the-art comparison into two parts where one for methods without multi-crop and the other with multi-crop. For the former, we mainly compare our method without multi-crop with MoCov3 [^12] and DINO without multi-crop. We observe that our method achieves state-of-the-art performance with ViT-S/16 even without multi-crop and comparable performance with ViT-B/16 compared with MoCov3. For the latter, we mainly compare our method with SwAV [^7] and DINO with multi-crop augmentation. We observe that iBOT achieves higher performance with 79.4% of linear probing accuracy when using ViT-S/16.

##### Effective Training Epochs.

Due to extra computation costs brought by multi-crop augmentation, different methods with the same pre-training epochs actually see different total numbers of images. To mitigate, we propose to measure the effective training epochs, defined as actual pre-training epochs multiplied with a scaling factor accounting for extra trained images of different resolutions induced by multi-crop augmentation. DINO and iBOT are by default trained with $2$ global crops of size $224\times 224$ and $10$ local crops of size $96\times 96$. Thus $r=2+(\frac{96}{224})^{2}\times 10=3.84\approx 4$ for DINO and iBOT. $r\approx 3$ for SwAV or DINO with RN50 as the backbone and pre-trained with $2$ global crops and $6$ local crops. $r=2$ for contrastive methods without multi-crop augmentation (*e.g*., MoCo, SimCLR, BYOL, *etc*.) and $r=1$ for non-contrastive methods (*e.g*., BEiT, Jigsaw, *etc*.).

## Appendix C Additional Implementations

Table 11: Different fine-tuning recipes. LD denotes layerwise learning rate decay. DS denotes mixed-precision training with DeepSpeed.

<table><tbody><tr><td></td><td>Epo.</td><td>LD</td><td>DS</td><td>BEiT</td><td>DINO</td><td>iBOT</td></tr><tr><td colspan="7">ViT-S/16</td></tr><tr><td>1</td><td>300</td><td>1.0</td><td>✗</td><td>81.5</td><td>81.1</td><td>81.2</td></tr><tr><td>2</td><td>300</td><td>0.75</td><td>✓</td><td>81.7</td><td>82.0</td><td>82.3</td></tr><tr><td>3</td><td>200</td><td>0.65</td><td>✗</td><td>80.7</td><td>-</td><td>-</td></tr><tr><td>4</td><td>200</td><td>0.75</td><td>✗</td><td>81.4</td><td>81.9</td><td>82.3</td></tr><tr><td>5</td><td>200</td><td>0.75</td><td>✓</td><td>81.4</td><td>82.0</td><td>82.2</td></tr><tr><td>6</td><td>200</td><td>0.85</td><td>✗</td><td>81.2</td><td>-</td><td>-</td></tr><tr><td colspan="7">ViT-B/16</td></tr><tr><td>7</td><td>300</td><td>1.0</td><td>✗</td><td>82.1</td><td>82.8</td><td>82.4</td></tr><tr><td>8</td><td>200</td><td>0.65</td><td>✓</td><td>82.7</td><td>83.1</td><td>83.2</td></tr><tr><td>9</td><td>100</td><td>0.65</td><td>✗</td><td>83.4</td><td>83.5</td><td>84.0</td></tr><tr><td>10</td><td>100</td><td>0.65</td><td>✓</td><td>83.2</td><td>83.6</td><td>83.8</td></tr></tbody></table>

Table 12: Evaluation protocols for semi-supervised learning. Proj. denotes fine-tuning from the middle layer of the projection head. LR denotes logistic regression.

<table><tbody><tr><td></td><td colspan="2">Method</td><td>Proj.</td><td>1%</td><td>10%</td></tr><tr><td colspan="6">frozen features</td></tr><tr><td>1</td><td>DINO</td><td>+ <math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>-</td><td>61.3</td><td>69.1</td></tr><tr><td>2</td><td>iBOT</td><td>+ <math><semantics><mi>k</mi> <ci>𝑘</ci> <annotation>k</annotation></semantics></math> -NN</td><td>-</td><td>62.3</td><td>70.1</td></tr><tr><td>3</td><td>DINO</td><td>+ Lin.</td><td>-</td><td>60.5</td><td>71.0</td></tr><tr><td>4</td><td>iBOT</td><td>+ Lin.</td><td>-</td><td>62.5</td><td>72.2</td></tr><tr><td>5</td><td>DINO</td><td>+ LR</td><td>-</td><td>64.5</td><td>72.2</td></tr><tr><td>6</td><td>iBOT</td><td>+ LR</td><td>-</td><td>65.9</td><td>73.4</td></tr><tr><td colspan="6">end-to-end fine-tuning</td></tr><tr><td>7</td><td colspan="2">DINO</td><td>✗</td><td>50.6</td><td>73.2</td></tr><tr><td>8</td><td colspan="2">iBOT</td><td>✗</td><td>55.0</td><td>74.0</td></tr><tr><td>9</td><td colspan="2">DINO</td><td>✓</td><td>60.3</td><td>74.3</td></tr><tr><td>10</td><td colspan="2">iBOT</td><td>✓</td><td>61.9</td><td>75.1</td></tr></tbody></table>

##### Fine-Tuning Recipes of Classification on ImageNet-1K.

By default, we follow the fine-tuning protocol in BEiT [^4] to use a layer-wise learning rate decay, weight decay and AdamW optimizer and train small-, base-size models with $200$, $100$, and $50$ epochs respectively. We sweep over four learning rates $\{8e^{-4},9e^{-4},1e^{-3},2e^{-3}\}$. Comparatively, traditional fine-tuning recipe is is to fine-tune the network for $300$ epochs with a learning rate $5e^{-4}$, no weight decay, and SGD optimizer [^43] (Row 1 versus 8). For a fair comparison, we compare the impact of different fine-tuning recipes with different methods, shown in Tab. 12. We empirically find that fine-tuning protocol used in BEiT consistently yields better fine-tuning results and greatly reduces the training epochs. By default, we use a layerwise decay of $0.75$ with a training epoch of $200$ for ViT-S/16, a layerwise decay of $0.65$ with a training epoch of $100$ for ViT-B/16, and a layerwise decay of $0.75$ with a training epoch of $50$ for ViT-L/16. We report the higher results between using or not using DS since we find it brings different impacts to different methods.

##### Evaluation Protocols of Semi-Supervised Learning on ImageNet-1K.

We study the impact of different evaluation protocols for semi-supervised learning. Under conventional semi-supervised evaluation protocol, pre-trained models are end-to-end fine-tuned with a linear classification head. SimCLRv2 [^10] found that keeping the first layer of the projection head can improve accuracy, especially under the low-shot setting. We fine-tune the pre-trained model from the first layer of the projection head and verify this conclusion holds true for Vision Transformers. We empirically find that Vision Transformer performs better with a frozen backbone with $1\%$ of training data (62.5% in row 4 versus 61.9 % in row 7). In DINO, a logistic regressor built upon the frozen features is found to perform better compared with the multi-class linear classifier upon the frozen features, especially with $1\%$ data (65.9% in row 6 versus 62.5% in row 4). When using $10\%$ data, we empirically find that end-to-end fine-tuning from the first layer of the projection layer yields the best performance (75.1% in row 10 versus 73.4% in row 6).

##### Fine-Tuning Recipes of Object Detection and Instance Segmentation on COCO.

For both small- and base-size models, we utilize multi-scale training (resizing image with shorter size between $480$ and $800$ while the longer side no larger than $1333$), a learning rate $1e^{-4}$, a weight decay of $0.05$, and fine-tune the entire network for $1\times$ schedule (12 epochs with the learning rate decayed by $10\times$ at epochs $9$ and $11$). We sweep a layer decay rate of { $0.65$, $0.75$, $0.8$, $0.9$ }. Note that a layer decay rate of $1.0$ denotes no layer is decayed. To produce hierarchical feature maps, we use the features output from layer $4$, $6$, $8$, and $12$, with $2$ deconvolutions, $1$ deconvolution, identity mapping, and max-pooling appended after, respectively. We do not use multi-scale testing.

##### Fine-Tuning Recipes of Semantic Segmentation on ADE20K.

For semantic segmentation, we follow the configurations in BEiT [^4], fine-tuning $160$ k iterations with $512\times 512$ images and a layer decay rate of $0.65$. We do not use multi-scale training and testing. We sweep the learning rate $\{3e^{-5},8e^{-5},1e^{-4},3e^{-4},8e^{-4}\}$. Similar to object detection and instance segmentation, to produce hierarchical feature maps, we add additional deconvolution layers after ViT.

| DINO, w/o \[LN\] | DINO, w/ \[LN\] | iBOT, w/o \[LN\] | iBOT, w/ \[LN\] |
| --- | --- | --- | --- |
| 33.7 | 34.5 | 37.8 | 38.3 |

When using linear (Lin.) as the task layer, we find that appending the last LayerNorm (\[LN\]) for \[CLS\] token to each patch tokens before the decoder consistently yields better performance, while we do not spot the substantial gain when with UperNet as the task layer. By default, we report the segmentation result with \[LN\] for both linear head for UperNet head.

##### Part-Wise Linear Probing.

We use the average of the last-layer self-attention map with \[CLS\] as the query from multiple heads to rank all the patch tokens. We remove the extra LayerNorm (LN) after the final block following MoCov3 [^12].

## Appendix D Additional Results

In this section, we provide detailed results for dense downstream tasks, *i.e*., object detection, instance segmentation, and semantic segmentation. We give the complete figures for occlusion robustness analysis. We also provide extra experiments of nearest neighbor retrieval, robustness analysis against occlusion and shuffle.

Table 13: Additional object detection, instance segmentation, and semantic segmentation results with small-size models. We pre-train iBOT with ViT-S/16 for 800 epochs.

<table><tbody><tr><td rowspan="2">Method</td><td rowspan="2">Arch.</td><td rowspan="2">Param.</td><td colspan="6">Det. & Inst. Seg. w/ Cascade Mask R-CNN</td><td colspan="2">Seg. w/ UperNet</td></tr><tr><td>AP <sup>b</sup></td><td>AP <math><semantics><mmultiscripts><mn>50</mn> <mi>b</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>b</ci></apply> <cn>50</cn></apply> <annotation>{}^{\mathrm{b}}_{\mathrm{50}}</annotation></semantics></math></td><td>AP <math><semantics><mmultiscripts><mn>75</mn> <mi>b</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>b</ci></apply> <cn>75</cn></apply> <annotation>{}^{\mathrm{b}}_{\mathrm{75}}</annotation></semantics></math></td><td>AP <sup>m</sup></td><td>AP <math><semantics><mmultiscripts><mn>50</mn> <mi>m</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>m</ci></apply> <cn>50</cn></apply> <annotation>{}^{\mathrm{m}}_{\mathrm{50}}</annotation></semantics></math></td><td>AP <math><semantics><mmultiscripts><mn>75</mn> <mi>m</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>m</ci></apply> <cn>75</cn></apply> <annotation>{}^{\mathrm{m}}_{\mathrm{75}}</annotation></semantics></math></td><td>mIoU</td><td>mAcc</td></tr><tr><td>Sup.</td><td>Swin-T</td><td>29</td><td>48.1</td><td>67.1</td><td>52.5</td><td>41.7</td><td>64.4</td><td>45.0</td><td>44.5</td><td>-</td></tr><tr><td>MoBY</td><td>Swin-T</td><td>29</td><td>48.1</td><td>67.1</td><td>52.1</td><td>41.5</td><td>64.0</td><td>44.7</td><td>44.1</td><td>-</td></tr><tr><td>Sup.</td><td>ViT-S/16</td><td>21</td><td>46.2</td><td>65.9</td><td>49.6</td><td>40.1</td><td>62.9</td><td>42.8</td><td>44.5</td><td>55.5</td></tr><tr><td>iBOT</td><td>ViT-S/16</td><td>21</td><td>49.4</td><td>68.7</td><td>53.3</td><td>42.6</td><td>65.6</td><td>45.8</td><td>45.4</td><td>56.2</td></tr></tbody></table>

Table 14: Additional object detection, instance segmentation, and semantic segmentation results with base-size models. We pre-train iBOT with ViT-B/16 for 400 epochs.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="6">Det. & Inst. Seg. w/ Cascade Mask R-CNN</td><td colspan="2">Seg. w/ Lin.</td><td colspan="2">Seg. w/ UperNet</td></tr><tr><td>AP <sup>b</sup></td><td>AP <math><semantics><mmultiscripts><mn>50</mn> <mi>b</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>b</ci></apply> <cn>50</cn></apply> <annotation>{}^{\mathrm{b}}_{\mathrm{50}}</annotation></semantics></math></td><td>AP <math><semantics><mmultiscripts><mn>75</mn> <mi>b</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>b</ci></apply> <cn>75</cn></apply> <annotation>{}^{\mathrm{b}}_{\mathrm{75}}</annotation></semantics></math></td><td>AP <sup>m</sup></td><td>AP <math><semantics><mmultiscripts><mn>50</mn> <mi>m</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>m</ci></apply> <cn>50</cn></apply> <annotation>{}^{\mathrm{m}}_{\mathrm{50}}</annotation></semantics></math></td><td>AP <math><semantics><mmultiscripts><mn>75</mn> <mi>m</mi></mmultiscripts> <apply><csymbol>subscript</csymbol> <apply><csymbol>superscript</csymbol> <csymbol>absent</csymbol> <ci>m</ci></apply> <cn>75</cn></apply> <annotation>{}^{\mathrm{m}}_{\mathrm{75}}</annotation></semantics></math></td><td>mIoU</td><td>mAcc</td><td>mIoU</td><td>mAcc</td></tr><tr><td>Sup.</td><td>49.8</td><td>69.6</td><td>53.8</td><td>43.2</td><td>66.6</td><td>46.5</td><td>35.4</td><td>44.6</td><td>46.6</td><td>57.0</td></tr><tr><td>BEiT</td><td>50.1</td><td>68.5</td><td>54.6</td><td>43.5</td><td>66.2</td><td>47.1</td><td>27.4</td><td>35.5</td><td>45.8</td><td>55.9</td></tr><tr><td>DINO</td><td>50.1</td><td>69.5</td><td>54.3</td><td>43.4</td><td>66.8</td><td>47.0</td><td>34.5</td><td>43.7</td><td>46.8</td><td>57.1</td></tr><tr><td>iBOT</td><td>51.2</td><td>70.8</td><td>55.5</td><td>44.2</td><td>67.8</td><td>47.7</td><td>38.3</td><td>48.0</td><td>50.0</td><td>60.3</td></tr></tbody></table>

##### Object Detection, Instance Segmentation, and Semantic Segmentation.

We here provide more detailed results on object detection, instance segmentation, and semantic segmentation with small- and base-size models, shown in Tab. 13 and Tab. 14 respectively. Specifically, we include AP ${}^{\mathrm{b}}_{\mathrm{50}}$ and AP ${}^{\mathrm{b}}_{\mathrm{75}}$ for object detection, AP ${}^{\mathrm{m}}_{\mathrm{50}}$ and AP ${}^{\mathrm{m}}_{\mathrm{75}}$ for instance segmentation, mAcc for semantic segmentation. For object detection (Det.) and instance segmentation (Inst. Seg.), we consider Cascade Mask R-CNN as the task layer. For semantic segmentation (Seg.), we consider two evaluation settings where a linear head (Lin.) and UPerNet are taken as the task layer.

Table 15: $k$ -NN and linear probing on ImageNet-1K with different pre-training datasets.

| Arch. | Pre-Train Data | Param. | Epoch | $k$ -NN | Linear |
| --- | --- | --- | --- | --- | --- |
| ViT-S/16 | ImageNet-1K | 21 | 800 | 75.2 | 77.9 |
| ViT-S/16 | ImageNet-22K | 21 | 160 | 69.3 | 76.5 |
| ViT-B/16 | ImageNet-1K | 85 | 400 | 77.1 | 79.5 |
| ViT-B/16 | ImageNet-22K | 85 | 80 | 71.1 | 79.0 |
| ViT-L/16 | ImageNet-1K | 307 | 300 | 78.0 | 81.0 |
| ViT-L/16 | ImageNet-22K | 307 | 50 | 72.9 | 82.3 |

##### k𝑘k-NN and Linear Probing with ImageNet-22K.

We further report $k$ -NN and linear probing accuracy on ImageNet-1K with models pre-trained on ImageNet-22K dataset. We empirically observe that ImageNet-1K pre-training incurs better ImageNet-1K $k$ -NN and linear probing performance, which is opposite to the fine-tuning performance observed in Table 3 and Table 3. We hypothesize that the data distribution plays a more crucial rule under evaluation protocols based on frozen features, such that models pre-trained with smaller ImageNet-1K dataset consistently achieve better results.

Table 16: Effectiveness of pre-trained features on nearest neighbor retrieval. We report the results on different downstream tasks whose evaluation is based on nearest neighbor retrieval.

<table><tbody><tr><td rowspan="3">Method</td><td colspan="4">Image Retrieval</td><td colspan="3" rowspan="2">Vid. Obj. Segment.</td></tr><tr><td colspan="2"><math><semantics><mi>ℛ</mi> <ci>ℛ</ci> <annotation>\mathcal{R}</annotation></semantics></math> Ox</td><td colspan="2"><math><semantics><mi>ℛ</mi> <ci>ℛ</ci> <annotation>\mathcal{R}</annotation></semantics></math> Par</td></tr><tr><td>M</td><td>H</td><td>M</td><td>H</td><td><math><semantics><msub><mrow><mo>(</mo><mrow><mi>𝒥</mi> <mo>&</mo> <mi>ℱ</mi></mrow><mo>)</mo></mrow> <mi>m</mi></msub> <apply><csymbol>subscript</csymbol> <apply><ci>𝒥</ci> <ci>ℱ</ci></apply> <ci>𝑚</ci></apply> <annotation>(\mathcal{J}\&\mathcal{F})_{m}</annotation></semantics></math></td><td><math><semantics><msub><mi>𝒥</mi> <mi>m</mi></msub> <apply><csymbol>subscript</csymbol> <ci>𝒥</ci> <ci>𝑚</ci></apply> <annotation>\mathcal{J}_{m}</annotation></semantics></math></td><td><math><semantics><msub><mi>ℱ</mi> <mi>m</mi></msub> <apply><csymbol>subscript</csymbol> <ci>ℱ</ci> <ci>𝑚</ci></apply> <annotation>\mathcal{F}_{m}</annotation></semantics></math></td></tr><tr><td>DINO</td><td>37.2</td><td>13.9</td><td>63.1</td><td>34.4</td><td>61.8</td><td>60.2</td><td>63.4</td></tr><tr><td>iBOT</td><td>36.6</td><td>13.0</td><td>61.5</td><td>34.1</td><td>61.8</td><td>60.4</td><td>63.2</td></tr></tbody></table>

##### Nearest Neighbor Retrieval.

Nearest neighbor retrieval is considered using the frozen pre-trained features following the evaluation protocol as in DINO [^8]. DINO has demonstrated the strong potential of pre-trained ViT features to be directly used for retrieval. To validate, DINO designed several downstream tasks, including image retrieval and video object segmentation, where video object segmentation can be seen as a dense retrieval task by finding the nearest neighbor between consecutive frames to propagate masks. We compare iBOT with DINO on these benchmarks with the same evaluation settings. As demonstrated in Tab. 16, iBOT has comparable results with DINO. While iBOT has higher $k$ -NN results on Imagenet-1K, the performance is not better for iBOT in image retrieval. We empirically find that the results on image retrieval are sensitive to image resolution, multi-scale features, *etc*., and the performance varies using pre-trained models with minimal differences on hyper-parameter setup. For this reason, we do not further push iBOT for better results.

##### Robustness against Background Change.

Deep models rely on both foreground objects and backgrounds. Robust models should be tolerant to background changes and able to locate discriminative foreground parts. We evaluate this property on ImageNet-9 (IN-9) dataset [^50]. IN-9 includes $9$ coarse-grained classes and $7$ variants by mixing up the foreground and background from different images. Only-FG (O.F.), Mixed-Same (M.S.), Mixed-Rand (M.R.), and Mixed-Next (M.N.) are $4$ variant datasets where the original foreground is present but the background is modified, whereas No-FG (N.F.), Only-BG-B (O.BB.), and Only-BG-T (O.BT.) are $3$ variants where the foreground is masked. As shown in Tab. 8, we observe a performance gain except for O.BT., indicating iBOT’s robustness against background changes. We note in O.BT. neither foreground nor foreground mask is visible, contradicting the pre-training objective of MIM.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x11.png)

Figure 9: Robustness against occlusion. Model’s robustness against occlusion with different information loss ratios is studied. 3 patch dropping settings: Random Patch Dropping (left), Salient Patch Dropping (middle), and Non-Salient Patch Dropping (right) are considered.

##### Robustness against Occlusion.

Masked prediction has a natural strength in cases where parts of the image are masked out since the models are trained to predict their original contents. We here provide the detailed results of occlusion with different information loss ratios in Fig. 9 under three dropping settings: random, salient, and non-salient. We showcase the results of iBOT end-to-end fine-tuned or with a linear head over the pre-trained backbone. We include the results of supervised results with both ViT-S/16 and ResNet-50 for comparison. ViT shows higher robustness compared to its CNN counterpart, *i.e*., ResNet-50, given that Transformers’ dynamic receptive field makes it less dependent on images’ spatial structure. We empirically find iBOT has stronger robustness against occlusion compared to its supervised baseline, consolidating that MIM help to model the interaction between the sequence of image patches using self-attention such that discarding proportion of elements does not degrade the performance significantly.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x14.png)

Figure 10: Robustness against shuffle. Model’s robustness against shuffle with different grid shuffle sizes is studied.

##### Robustness against Shuffle.

We study the model’s sensitivity to the spatial structure by shuffling on input image patches. Specifically, we shuffle the image patches with different grid sizes following [^33]. We showcase the results of iBOT end-to-end fine-tuned or with a linear head over the pre-trained backbone. We include the results of supervised results with both ViT-S/16 and ResNet-50 for comparison. Note that a shuffle grid size of $1$ means no shuffle, and a shuffle grid size of $196$ means all patch tokens are shuffled. Fig. 10 suggests that iBOT retain accuracy better than its supervised baseline and ResNet-50. It also indicates that iBOT relies less on positional embedding to preserve the global image context for right classification decisions.

## Appendix E Additional Ablations

In this section, we study the impact of other parameters that we have conducted experiments on. Without extra illustrations, we use $300$ -epoch pre-trained ViT-S/16, a prediction ratio $r=0.3$ and without multi-crop augmentation for the ablative study.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x15.png)

Table 17: Different head sharing strategy.

##### Architecture of Projection Head.

As mentioned earlier, a shared head can transfer the semantics acquired in \[CLS\] token to patch tokens, slightly improving the performance. We notice that the head for patch tokens in the student network only see the masked tokens throughout the training, the distribution of which mismatches tokens with natural textures. Therefore, we conduct an experiment using a non-shared head for the student network but a shared head for the teacher network denoted as semi-shared head. Their differences are demonstrated in Fig. 18, where S and T denotes student and teacher network respectively. The heads with the same index and color denotes they have shared parameters.

| Arch. | vanilla | shared <sup>†</sup> | sm. shared | sm. shared <sup>†</sup> | shared |
| --- | --- | --- | --- | --- | --- |
| $k$ -NN. | 68.9 | 68.0 | 68.4 | 68.4 | 69.1 |
| Lin. | 73.9 | 73.7 | 73.7 | 73.8 | 74.2 |

<sup>†</sup> denotes only the first $2$ layers out of the $3$ -layer MLP share the parameters. However, we do not observe that semi-shared head is better than shared head. By default, we share the entire projection head for \[CLS\] token and patch tokens.

##### Comparing MIM with Dense Self-Distillation.

To identify the superiority of MIM to model internal structure using over its alternatives, we conduct experiments performing self-distillation on original patch tokens along with the \[CLS\] token. We consider two matching strategies to construct patch token pairs for self-distillation.

| Arch. | DINO | DINO + pos. | DINO + feat. | iBOT |
| --- | --- | --- | --- | --- |
| $k$ -NN | 67.9 | 67.1 ($-$ 0.8) | 68.5 ($+$ 0.6) | 69.1 ($+$ 1.2) |
| Lin. | 72.5 | 72.5 ($+$ 0.0) | 73.4 ($+$ 0.9) | 74.2 ($+$ 1.7) |

Specifically, pos. denotes matching according to the absolute position of two views. Similar to [^53]. $j$ is defined as $\mathop{\arg\min}_{j}dist(p_{i},p^{\prime}_{j})$, where $p$ is the position in the original image space and $dist(u,v)$ is euclidean distance. The losses are only computed for the overlapped regions of two views. We do not observe substantial gain brought by matching via patches’ absolute position. feat. denotes matching according to the similarity of the backbone similarity of two views. Similar to [^46], we match for each patch token $f_{i}$ the most similar patch token from another view $f^{\prime}_{j}$, where $j=\mathop{\arg\max}_{j}sim(f_{i},f^{\prime}_{j})$. $sim(u,v)$ is cosine distance. Such practice brings a $0.6\%$ performance gain in terms of linear probing accuracy, which is also observed by a concurrent work, EsViT [^26]. Comparatively, iBOT prompts an $1.2\%$ gain on linear probing, verifying the necessity and advancement of MIM.

##### Hard Label versus Soft Label

We study the importance of using a continuous token distribution (softmax <sup>†</sup>) instead of a discretized id (hardmax) when performing MIM. Results in Tab. 18 indicate continuous tokenization plays a crucial part. We empirically find the improvement brought by centering, whose roles are less important compared to centering in self-distillation on \[CLS\] token. Only sharpening can produce a $k$ -NN accuracy of 69.4 and a linear probing accuracy of 73.9.

##### Centering and Sharpening.

Different from the \[CLS\] token, patch tokens do not have certain semantic cluster and vary more widely from each others. We study the impact of several critical parameters that decide the distillation process and customize them for distillation over the patch tokens.

| $m^{\prime}$ | $.8$ | $.99$ | $.999$ | $.9$ | $.9$ | $.9$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\tau^{\prime}_{t}$ | $.04\to.07$ | $.04\to.07$ | $.04\to.07$ | $.04\to.06$ | $.05\to.08$ | $.04\to.07$ |
| $k$ -NN | 68.7 | 68.8 | 68.9 | 68.5 | 68.7 | 69.1 |
| Lin. | 74.0 | 73.8 | 73.8 | 73.5 | 73.9 | 74.2 |

Specifically, the smoothing momentum for online centering $m^{\prime}$ and sharpening temperature $\tau^{\prime}_{t}$ are studied. Note we keep the parameters for \[CLS\] token the same as DINO and only study for parameters for the patch tokens.

##### Loss Ratio.

We study the impact of different ratio between $\mathcal{L}_{\texttt{[CLS]}}$ and $\mathcal{L}_{\mathrm{MIM}}$. We keep the base of $\mathcal{L}_{\texttt{[CLS]}}$ to $1$ and scale $\mathcal{L}_{\mathrm{MIM}}$ with different ratios.

| $\mathcal{L}_{\texttt{[CLS]}}$ / $\mathcal{L}_{\mathrm{MIM}}$ | $0.5$ | $2$ | $1$ |
| --- | --- | --- | --- |
| $k$ -NN | 68.7 | 69.4 | 69.1 |
| Lin. | 73.8 | 74.1 | 74.2 |

We observe that directly adding two losses up without scaling yields the best performance in terms of linear probing accuracy.

##### Output Dimension.

We follow the structure of projection head in DINO with l2-normalized bottleneck and without batch normalization. We study the impact of output dimension $K$ of the last layer.

| $K$ | $4096$ | $16384$ | $8192$ |
| --- | --- | --- | --- |
| $k$ -NN | 68.3 | 68.8 | 69.1 |
| Lin. | 74.5 | 74.0 | 74.2 |

While our method excludes large output dimensionality since each patch token has an output distribution, we do not observe substantial performance gain brought by larger output dimensions. Therefore, we choose $K=8192$ by default.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x16.png)

Figure 11: Impact of the prediction ratio. ± plus-or-minus \\pm denotes to randomly sample from a region.

##### Prediction Ratios.

Masked modeling is based on a formulation of partial prediction, the objective of which is to maximize the log-likelihood of the target tokens conditioned on the non-target tokens. We experiment with different prediction ratios for masked image modeling. The results are shown in Fig. 12. We observe that the performance is not sensitive to variant prediction ratios between $0.05$ and $0.4$. Adding a variance upon the fixed value can also consistently bring a performance gain, which can be explained as stronger data augmentation. The teacher output of non-masked images is now pulled together with the student output of masked images with different ratios. By default, we use $0.3\ (\pm 0.2)$ as the prediction ratio. For models with multi-crop augmentation, following the above discussions, we randomly choose a prediction of $00$ or $0.3\ (\pm 0.2)$ for each image.

##### Training Epochs.

We provide the linear probing top-1 accuracy with ViT-S/16 pre-trained for different epochs. For comparison, we also include the accuracy curve of other methods with comparable numbers of parameters, *i.e*., ResNet-50. From Fig. 12, we observe that longer training for $800$ epochs can improve the model’s performance. It’s north worthy that iBOT can achieve a Top-1 accuracy of SwAV [^7] pre-trained with $800$ epochs in less than $100$ epochs. iBOT pre-trained with $800$ epochs brings a 0.9% improvement over previous state-of-the-art method.

Table 19: Time and Memory Requirements. We detail the actual training time (T) and GPU memory (Mem.) of different methods, together with their respective linear probing (Lin.) and fine-tuning (Fin.) accuracy. All methods are trained on two 8-GPU V100 machines with a batch size of 1024.

| Method | Crops Number | T <sub>100</sub> | T <sub>300</sub> | T <sub>800</sub> | Mem. | Lin.<sub>300</sub> | Lin.<sub>800</sub> | Fin.<sub>800</sub> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BEiT | $1\times 224^{2}$ | 11.3h | 33.7h | 90.1h | 5.6G | 20.7 | 24.2 | 81.4 |
| DINO | $2\times 224^{2}$ | 15.1h | 44.7h | 111.6h | 9.3G | 72.5 | 73.7 | 81.6 |
| iBOT | $2\times 224^{2}$ | 15.6h | 47.0h | 126.4h | 13.1G | 74.8 | 76.2 | 82.0 |
| DINO | $2\times 224^{2}+10\times 96^{2}$ | 24.2h | 72.6h | 180.0h | 15.4G | 76.2 | 77.0 | 82.0 |
| iBOT | $2\times 224^{2}+10\times 96^{2}$ | 24.3h | 73.3h | 193.4h | 19.5G | 77.4 | 77.9 | 82.3 |

##### Time and Memory Requirements.

BEiT is trained with a non-contrastive objective and without multi-crop augmentation, thus it consumes only a memory of 5.6G and takes 90.1h for $800$ epochs. Comparing iBOT and DINO with multi-crop augmentation, iBOT with MIM induces $25\%$ more memory requirements and $7.4\%$ more actual training time. Considering pre-training efficiency (accuracy versus time), 800-epochs pre-trained DINO requiring for 180.0h, while 300-epochs iBOT only requires 73.3h with 0.4% higher linear probing accuracy (77.0 versus 77.4).

## Appendix F Alternative Tokenizers

Table 20: Methodology comparison over different approaches to tokenize the patches. We report ImageNet-1K $k$ -NN, linear and fine-tuning validation accuracy. Models are pre-trained with ViT-S/16 and 300 epochs.

| Method | $k$ -NN | Linear | Fine-Tune |
| --- | --- | --- | --- |
| Rand. | \- | \- | 79.9 |
| MPP [^17] | 16.4 | 37.2 | 80.8 |
| Patch Clustering | 19.2 | 40.1 | 81.3 |
| BEiT [^4] | 6.9 | 24.2 | 81.4 |
| Standalone DINO as tokenizer | 44.3 | 60.0 | 81.7 |
| iBOT | 70.3 | 74.8 | 81.5 |

To investigate how different approaches to tokenize the patches affect MIM, we study several alternatives. In BEiT [^4], masked patches are tokenized by a DALL-E encoder. MPP [^17] tokenizes the masked patches using their 3-bit mean color. For Patch Clustering, we first perform $K$ -Means algorithm to the flattened color vector of each $16\times 16$ patch ($d=768$). $10\%$ data of ImageNet-1K training set is sampled and clustered. We set $K$ to $4096$. During pre-training, each patch is tokenized by the index of its closest centroids. Lastly, we use $300$ -epoch pre-trained DINO as a standalone tokenizer. Each patch can be tokenized by the argmax of its output from the pre-trained DINO. We use average pooling to aggregate the patch representations. From Tab. 20, we see that all methods achieve decent fine-tuning results compared to the supervised baseline, while only methods tokenized by semantically meaningful tokenizer have proper results on $k$ -NN and linear classification. MPP [^17] and patch clustering rely purely on offline statistics without the extra stage of online training. We find patch clustering has slightly better performance in all three protocols compared to MPP, suggesting the benefits brought by visual semantics. While BEiT has poor $k$ -NN and linear probing accuracy, a good fine-tuning result also suggests relatively low requirements for fine-tuning protocol on high-level semantics.

## Appendix G Visualization

In this section, we first give more visualized pattern layouts and self-attention maps. Beyond that, we consider an additional task of mining sparse correspondences between two images and illustrating the superiority of ViTs by showcasing several visualized results.

### G.1 Pattern Layout

##### Pattern Layout for Patch Tokens.

To illustrate versatile, interesting behaviors iBOT has learned, we organize the visualization of pattern layout in two figures. In Fig. 13, we mainly showcase additional pattern layouts that share high-level semantics. In Fig. 14, we mainly showcase additional pattern layouts that share low-level details like color, texture, shape, *etc*. Top $100$ patches with the highest confidence over the validation set are visualized with a $5\times 5$ context around each $16\times 16$ patch token (colored orange).

##### Composing Images with Representative Patterns.

In Fig. 15, we visualize $4$ patches with the highest self-attention score (with non-overlapped assigned index) and also show the pattern layout of that assigned index. The visualized results indicate iBOT can only be represented by several representative patches, which helps the model’s robustness and performance in recognition. This is also validated by our part-wise linear probing experiments.

##### Comparison with Other Methods.

We visualize pattern layout for patch tokens using other self-supervised methods [^4] [^8] in Fig. 16. For BEiT, the DALL-E encoder generates a discrete number for each patch token. For DINO, we directly use the projection head for \[CLS\] token and generate a $65536$ -d probability distribution for each patch token. The index with the highest probability is assigned for the token.

##### Pattern Layout for \[CLS\] Token.

We here also provide additional visualization of semantic patterns emerge in \[CLS\] token, which is obtained via self-distillation on cross-view images. We also observe similar behavior in DINO since it’s not a unique property brought by MIM. In fact, semantics are now believed to emerge as long as a similarity between two distorted views of one image is enforced [^18] [^20] [^7] [^6].

### G.2 Self-Attention Visualizations

Similar to the setting of Sec. 4.3.2, we here provided more self-attention map visualization from multiple heads of the last layer in Fig. 18.

### G.3 Sparse Correspondence.

We consider a sparse correspondence task where the overlapped patches from two augmented views of one image, or patches from two images labeled as one class, are required to be matched. The correlation is sparse since at most $14\times 14$ matched pairs can be extracted with a ViT-S/16 model. We visualize $12$ correspondences with the highest self-attention score extracted from iBOT with ViT-S/16 pre-trained for $800$ epochs. The score is averaged between multiple heads of the last layer. Several sampled sets of image pairs are shown in Fig. 19. We observe empirically that iBOT perform well for two views drawn from one image, nearly matched the majority of correspondence correctly. In the second column, iBOT can match different parts of two instances from the same class (*e.g*., tiles and windows of two cars) despite their huge differences in texture or color. We observe the DINO also has comparable visualized effects, illustrating the representation pre-trained with self-distillation also suits well for retrieval in a patch-level scale.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x18.png)

Figure 13: Visualization for pattern layout of patch tokens that share high-level semantics. In the first row, we visualize different human-related semantic parts. We observe clear patterns accounting for human hair, human shoulder & arm, and human elbow respectively in the left, middle, and right figure. In the figures from the second row and the left figure from the the third row, we visualize animal-related semantic parts. dog’s ear dog’s nose bird’s wing dragonfly’s wing can be observed. In the rest of figures from the third row, we visualize semantic parts related to outdoor scenes. front window of the vehicle and window of the architecture can be observed. In the last row, we visualize indoor objects like ceiling glass bottle.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x19.png)

Figure 14: Visualization for pattern layout of patch tokens that share low-level details. In the first two columns, we visualize patches that share similar textures. In the first figure, fur of leopard and the skin of lizard share a similar dotted texture. In the second figure, shell of hedgehog and the skin of elephant share similar striped texture. In the third column, we visualize pattern layouts related to shape. For example, the shape of objects in the left and middle figures share similar curvature. The rightmost patterns clearly depict the shape of a straight line. We visualize pattern layout related to color in the last column, where blue, green and white can be observed.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x20.png)

Figure 15: Top- 4 representative patches with each of their pattern layout. Order index 0, 1, 2, 3 are ranked according to its self-attention score. In the top-left corner for each pattern layout subfigure, its order index and cluster index are annotated. In the top panel, we can observe that pattern 0,2,3 show explicit semantic information of nose, eyes, ears respectively. Interestingly, patch 1 also locates around the eyes of the Samoyed but its corresponding pattern share visual similarity in shape instead of semantics. This illustrate the diverse behaviour for each learned pattern. In the bottom panel, a library is represented by 0 two- or multi-color joints, 1,3 knurlling texture 2 texts. Similarly, we have patterns 0,1,3 focusing more on texture & color and pattern focusing more on semantics. All of these visualized results illustrate versatile behaviour for each index.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x21.png)

Figure 16: Visualization for pattern layout of patch tokens using BEiT (top) and DINO (bottom). In the layout extracted from the DALL-E encoder, we observe minimal semantic patterns. In most cases, patches with similar color ( e.g., black area in left figure) or texture ( line in right figure) are clustered. In the layout extracted from DINO, while more complex textures are visible, most patches share similar local details instead of high-level semantics. In the right figure, the semantic part eyes can be somehow observed, yet it is mixed with plenty of irrelevant semantic parts.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x22.png)

Figure 17: Visualization for pattern layout of \[CLS\] token. We here indicate the high quality of semantic layout brought by self-distillation of cross-view images on token. This property is not brought by MIM and is also prominent in DINO.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x23.png)

Figure 18: Visualization for self-attention map from Multiple Heads. In the first 8 columns, we showcase iBOT’s attention map along with DINO’s. In the last 10 columns, we showcase more attention map from iBOT. We indicate that iBOT shows visually stronger ability to separate different objects or different parts of one object apart by giving more attentive visualized results for each part, compared with DINO. For example, in the fifth column, there is an attention head in iBOT accounting for the ear of the fox solely, while in DINO, it emerges with other parts; In the eighth column, iBOT separates the mushroom into more semantically meaningful parts.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2111.07832/assets/x24.png)

Figure 19: Visualization for sparse correspondence. The top panel are images pairs sampled from two views of one image. The extracted correspondence from iBOT is mostly correct despite augmentations on scale and color. The bottom panel are image pairs sampled from two images of one class. The first row is images with salient objects but different sizes, positions and textures. The second row are images draw from animals, and we can observe more clearly that iBOT matches the semantic parts of animals correctly ( e.g., tails of the fox beak of the bird ). The third row is human-centered images with human bodies or clothing. The fourth row is natural or domestic scenes where salient objects are invisible. Although no explicit semantic parts can be matched visible to human’s understanding, we can still observe the iBOT can extract correspondence based on their texture or color ( wooden texture of signboard and boxes. All these visual results demonstrate strong capability for iBOT in part retrieval or matching in a local scale.

[^1]: Elad Amrani and Alex Bronstein. Self-supervised classification network. *arXiv preprint arXiv:2103.10994*, 2021.

[^2]: Yuki M. Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In *ICLR*, 2020.

[^3]: Sara Atito, Muhammad Awais, and Josef Kittler. SiT: Self-supervised vision transformer. *arXiv preprint arXiv:2104.03602*, 2021.

[^4]: Hangbo Bao, Li Dong, and Furu Wei. BEiT: BERT pre-training of image transformers. *arXiv preprint arXiv:2106.08254*, 2021.

[^5]: Zhaowei Cai and Nuno Vasconcelos. Cascade R-CNN: High quality object detection and instance segmentation. *TPAMI*, 2019.

[^6]: Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In *ECCV*, 2018.

[^7]: Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In *NeurIPS*, 2020.

[^8]: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *ICCV*, 2021.

[^9]: Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In *ICML*, 2020a.

[^10]: Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey Hinton. Big self-supervised models are strong semi-supervised learners. In *NeurIPS*, 2020b.

[^11]: Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In *CVPR*, 2021.

[^12]: Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In *ICCV*, 2021.

[^13]: Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu. Uniter: Universal image-text representation learning. In *ECCV*, 2020c.

[^14]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In *CVPR*, 2009.

[^15]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In *NAACL*, 2019.

[^16]: Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In *ICCV*, 2015.

[^17]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In *ICLR*, 2021.

[^18]: Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, Bilal Piot, koray kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In *NeurIPS*, 2020.

[^19]: Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick. Mask R-CNN. In *ICCV*, 2017.

[^20]: Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In *CVPR*, 2020.

[^21]: Olivier Henaff. Data-efficient image recognition with contrastive predictive coding. In *ICML*, 2020.

[^22]: Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In *ICLR*, 2019.

[^23]: Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In *CVPR*, 2021.

[^24]: Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. In *NeurIPS*, 2015.

[^25]: Nikos Komodakis and Spyros Gidaris. Unsupervised representation learning by predicting image rotations. In *ICLR*, 2018.

[^26]: Chunyuan Li, Jianwei Yang, Pengchuan Zhang, Mei Gao, Bin Xiao, Xiyang Dai, Lu Yuan, and Jianfeng Gao. Efficient self-supervised vision transformers for representation learning. *arXiv preprint arXiv:2106.09785*, 2021a.

[^27]: Zhaowen Li, Zhiyang Chen, Fan Yang, Wei Li, Yousong Zhu, Chaoyang Zhao, Rui Deng, Liwei Wu, Rui Zhao, Ming Tang, et al. MST: Masked self-supervised transformer for visual representation. *arXiv preprint arXiv:2106.05656*, 2021b.

[^28]: Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In *ECCV*, 2014.

[^29]: Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. Self-supervised learning: Generative or contrastive. *TKDE*, 2021a.

[^30]: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. *arXiv preprint arXiv:2103.14030*, 2021b.

[^31]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *ICLR*, 2019.

[^32]: Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In *NeurIPS*, 2019.

[^33]: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang. Intriguing properties of vision transformers. *arXiv preprint arXiv:2105.10497*, 2021.

[^34]: Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In *ECCV*, 2016.

[^35]: Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In *CVPR*, 2016.

[^36]: Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In *ICML*, 2021.

[^37]: Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. *NeurIPS*, 2015.

[^38]: Jason Tyler Rolfe. Discrete variational autoencoders. In *ICLR*, 2017.

[^39]: Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In *ACL*, 2016.

[^40]: Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai. Vl-bert: Pre-training of generic visual-linguistic representations. *ICLR*, 2020.

[^41]: Hao Tan, Jie Lei, Thomas Wolf, and Mohit Bansal. Vimpac: Video pre-training via masked token prediction and contrastive learning. *arXiv preprint arXiv:2106.11250*, 2021.

[^42]: Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for good views for contrastive learning? In *NeurIPS*, 2020.

[^43]: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Herve Jegou. Training data-efficient image transformers & distillation through attention. In *ICML*, 2021.

[^44]: Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. Scan: Learning to classify images without labels. In *ECCV*, 2020.

[^45]: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *ICCV*, 2021a.

[^46]: Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In *CVPR*, 2021b.

[^47]: Chen Wei, Lingxi Xie, Xutong Ren, Yingda Xia, Chi Su, Jiaying Liu, Qi Tian, and Alan L Yuille. Iterative reorganization with weak spatial constraints: Solving arbitrary jigsaw puzzles for unsupervised representation learning. In *CVPR*, 2019.

[^48]: Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. *arXiv preprint arXiv:1609.08144*, 2016.

[^49]: Zhirong Wu, Yuanjun Xiong, X Yu Stella, and Dahua Lin. Unsupervised feature learning via non-parametric instance discrimination. In *CVPR*, 2018.

[^50]: Kai Yuanqing Xiao, Logan Engstrom, Andrew Ilyas, and Aleksander Madry. Noise or signal: The role of image backgrounds in object recognition. In *ICLR*, 2020.

[^51]: Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In *ECCV*, 2018.

[^52]: Zhenda Xie, Yutong Lin, Zhuliang Yao, Zheng Zhang, Qi Dai, Yue Cao, and Han Hu. Self-supervised learning with swin transformers. *arXiv preprint arXiv:2105.04553*, 2021a.

[^53]: Zhenda Xie, Yutong Lin, Zheng Zhang, Yue Cao, Stephen Lin, and Han Hu. Propagate yourself: Exploring pixel-level consistency for unsupervised visual representation learning. In *CVPR*, 2021b.

[^54]: Asano YM., Rupprecht C., and Vedaldi A. Self-labelling via simultaneous clustering and representation learning. In *ICLR*, 2020.

[^55]: Yucheng Zhao, Guangting Wang, Chong Luo, Wenjun Zeng, and Zheng-Jun Zha. Self-supervised visual representations learning by contrastive mask prediction. *arXiv preprint arXiv:2108.07954*, 2021.

[^56]: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *CVPR*, 2017.