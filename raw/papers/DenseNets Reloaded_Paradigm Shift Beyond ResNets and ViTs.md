---
title: "DenseNets Reloaded:Paradigm Shift Beyond ResNets and ViTs"
source: "https://ar5iv.labs.arxiv.org/html/2403.19588"
author:
published:
created: 2026-08-31
description: "This paper revives Densely Connected Convolutional Networks (DenseNets) and reveals the underrated effectiveness over predominant ResNet-style architectures.We believe DenseNets’ potential was overlooked due to untouc…"
tags:
  - "clippings"
---
## DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs

Donghyun Kim Thanks: Equal contribution. Correspondence to Dongyoon Han.    Byeongho Heo Affiliation: NAVER Cloud AI, NAVER AI Lab    Dongyoon Han <sup>⋆</sup> Affiliation: NAVER Cloud AI, NAVER AI Lab

###### Abstract

This paper revives Densely Connected Convolutional Networks (DenseNets) and reveals the underrated effectiveness over predominant ResNet-style architectures. We believe DenseNets’ potential was overlooked due to untouched training methods and traditional design elements not fully revealing their capabilities. Our pilot study shows dense connections through concatenation are strong, demonstrating that DenseNets can be revitalized to compete with modern architectures. We methodically refine suboptimal components - architectural adjustments, block redesign, and improved training recipes towards widening DenseNets and boosting memory efficiency while keeping concatenation shortcuts. Our models, employing simple architectural elements, ultimately surpass Swin Transformer, ConvNeXt, and DeiT-III — key architectures in the residual learning lineage. Furthermore, our models exhibit near state-of-the-art performance on ImageNet-1K, competing with the very recent models and downstream tasks, ADE20k semantic segmentation, and COCO object detection/instance segmentation. Finally, we provide empirical analyses that uncover the merits of the concatenation over additive shortcuts, steering a renewed preference towards DenseNet-style designs. Our code is available at [https://github.com/naver-ai/rdnet](https://github.com/naver-ai/rdnet).

## 1 Introduction

The “ImageNet moment” was sparked by the emergence of Convolutional Neural Networks (ConvNets), starting with the milestone AlexNet [^45]. Subsequently, VGG [^70] and GoogleNet [^72] further highlighted the benefits of stacking multiple convolutional layers in ConvNets. In the same era, a monumental architecture ResNet [^30] and its family [^31] [^95] stands out for introducing a groundbreaking concept - additive skip connections (also known as additive shortcuts or identity mapping [^31]), which allowed for the stacking of up to 1,000 layers. The introduction of residual learning with it was a game-changer, diminishing the gradient vanishing problem by ensuring the input gradient always remained at one from the derivative of the identity mapping. This innovation sparked a series of successors, including the milestone ConvNets - EfficientNet [^74] and ConvNeXt [^55]; it paved the way for the next leap, such as Transformers [^83], Vision Transformers (ViTs) [^21], and Hierarchical ViTs [^54], which accentuates the lasting influence of additive shortcuts.

In the early stage of this period governed by residual learning, Densely Connected Convolutional Networks (DenseNets [^37]) introduced a novel approach: maintaining shortcut connections through feature concatenation instead of using additive shortcuts. This led to the concept of feature reuse [^37], allowing more compact models and reducing overfitting through explicit supervision propagation to the early layers. DenseNets showcased efficiency and superior performance in tasks like semantic segmentation [^41]. The evolution of architectural designs post-DenseNet appeared to challenge the dominance of ResNets but saw a decline in their popularity, shaded by the advantages of additive shortcuts.

Successors of DenseNets [^87] [^47] [^84] revisited DenseNets to advance its design spirit but struggled against more dominant architectural trends again. We argue the potential of DenseNets still remained underexplored due to low accessibility, being gradually hindered by outdated training methods and the limitations of low-capacity components; they struggled to keep pace with the advancements in modern architectures that benefited from years of evolutionary refinements. Furthermore, we presume DenseNets requires an overhaul due to its limited applicability and memory challenges caused by increasing feature dimensionality. While the authors addressed memory concerns [^37] [^61], these issues continue to restrict the expansion of the architecture, particularly for width scaling. Despite the drawbacks, we conjecture the core design concept is still highly potent.

Bearing this in mind, this paper revitalizes DenseNets by highlighting the undervalued efficacy of concatenations. Through a comprehensive pilot study training with over 10k random networks across varied setups, we validate our claim that concatenation can surpass the additive shortcut. Afterward, we modernize DenseNet with a more memory-efficient design to widen it, abandoning ineffective components and enhancing architectural and block designs, while preserving the essence of dense connectivity via concatenation. We employ contemporary strategies that synergize with DenseNets as well. Our methodology eventually exceeds strong modern architectures [^65] [^105] [^23] [^27] [^52] [^49] and some milestones like Swin Transformer [^54], ConvNeXt [^55], and DeiT-III [^78] in performance trade-offs on ImageNet-1K [^67]. Our models demonstrate competitive performance on downstream tasks such as ADE20K semantic segmentation and COCO object detection/instance segmentation. Remarkably, our models do not exhibit slowdown or degradation as the input size increases. Ultimately, our empirical analyses shed light on the unique benefits of concatenation.

## 2 Related Work

#### Densely Connected Neural Networks

(DenseNets) [^37] pioneered dense connections within Convolutional Neural Networks (ConvNets) beyond additive shortcuts, highlighted by parameter efficiency and enhanced precisions. Building on this framework, several variants were proposed. PeleeNet [^87] successfully proposed modifications to achieve real-time inference capabilities upon DenseNet. VovNet [^47] departed from DenseNets’ dense feature reuse in favor of a sparser one-shot aggregation aimed at real-time object detection. CSPNet [^84], by omitting features and later concatenating them at cross-stage layers, reduces computational demands, barely affecting precision. DenseNets were further highlighted with the effectiveness on dense prediction tasks; for example, Jegou *et al*. [^41] showed the effectiveness of DenseNets on semantic segmentation. MDU-Net [^102] exploited the dense connectivity for enhanced biomedical image segmentation. DCCT [^60] integrated dense connections into a Transformer architecture [^83] to facilitate image dehazing. For video snapshot compressive imaging, EfficientSCI [^86] also leveraged the benefits of dense connectivity. Wang *et al*. [^90] utilized dense connections to improve the detection of small objects.

We believe these references demonstrate DenseNet-based designs’ potential, to our knowledge, but none recently challenged the ImageNet benchmarks using the principle of dense connections.

#### Modern architectures.

DeiT [^76] and AugReg [^71] exhibited modernized training recipes [^74] [^6] [^92] could replace massive training data for ViT [^21] training. Descendant hierarchical ViTs [^54] [^20] [^97] [^96] [^88] [^89], which got closer to ConvNets, showed locality offers efficacy along with computational efficiency. Hybrid architectures [^50] [^16] [^82] [^105] [^27] then explicitly equip convolutions for the locality. Ironically, incomers have become closer to ConvNets, aiming not to forsake the proven effectiveness of simple convolution, albeit using Transformers.

ConvNets [^30] [^74] [^64] [^7] initially predominated due to strong capability along with efficiency. Interestingly, advancements from the ViT side have also contributed to modernizing ConvNets; many recent architectures [^25] [^77] [^81] [^98] were inspired by ViT’s designs but armed with locality, demonstrating the continued high competitiveness of convolutions. Successors like RepLKNet [^19] and SLaK [^53] employed large-scale kernel convolutions built upon the predecessor’s legacy to emulate the globality of attention [^21], offering to learn enhanced global representations. RevCol [^8] introduced a new concept to mix multi-level features repeatedly through multiple columns. InceptionNeXt [^99] adopted the inception module [^72] inside ConvNeXt to show improved performance. HorNet [^65] and MogaNet [^49] both have presented remarkable performance by employing multiple gated convolution and multi-level features, respectively, which also took advantage of multi-scale features for globality.

Those architectures surpassed ViTs on ImageNet and dense prediction tasks as well, but similarities like using additive shortcuts and architectural complexity continue to restrict architectural diversity and innovation. Furthermore, network modernization methods [^87] [^55] [^6] [^92] have successfully revisited existing architectures but did not handle beyond baselines using additive shortcuts. This work follows a general direction but ensures our starts from a distinct baseline, acknowledging uncertainties about the effectiveness of existing roadmaps.

## 3 Methodology: Revitalizing DenseNets

This section starts with our conjecture that DenseNet may not fall behind modern architectures and proves it by substantiating revised DenseNet architectures. Based on our conjecture with our pilot experiments, we propose our methodology, which encompasses some modernized materials to revive DenseNet.

### 3.1 Preliminary

#### Motivation.

ResNets [^30] have renowned due to the simple formulation at $l$ -th layer: $\mathbf{X_{l+1}}=\mathbf{X_{l}}+f(\mathbf{X_{l}}\mathbf{W})$, where the input $\mathbf{X}_{l}$ and the weight $\mathbf{W}$. A pivotal element is the residual connection (*i.e*., additive shortcut, +), which facilitates modularized architectural designs as evidenced in Swin [^54], ConvNeXt [^55], and ViT [^21]. While DenseNets [^37] follow the formulation: $\mathbf{X_{l+1}}=[\mathbf{X_{l}},\ f(\mathbf{X_{l}}\mathbf{W})]$ based on the dense connection through concatenation having explicit parameter efficiency. However, this formulation should constrain feature dimensionality due to memory concerns, making it challenging to scale in width.

DenseNets [^37] initially outperformed ResNets [^30] but failed to realize a complete paradigm shift, losing initial momentum due to their applicability. In particular, despite the efforts for memory [^61] [^47], width scaling remains problematic for DenseNets, with wider models like DenseNet-161/-233 consuming more memory [^37]. Nonetheless, inspired by the prior works [^92] [^6] [^55] and motivated by successes in dense prediction tasks such as semantic segmentation [^102], we believe DenseNets would outperform popular architectures and warrant further exploration of their potential: 1) feature concatenation merits strong capability; 2) the above concerns in DenseNets can be mitigated through a strategic design.

#### Our conjecture.

Concatenation shortcut is an effective way of increasing rank. Consider the layer output $f(\mathbf{X}\mathbf{W})$ with the weight $\mathbf{W}{\in}\mathbb{R}^{d_{in}{\times}d_{out}}$ and the input $\mathbf{X}{\in}\mathbb{R}^{N{\times}d_{in}}$ with a nonlinearity $f$, where we assume the number of instances $N{\gg}d_{in}$. As focusing on the matrix rank of $f$, $\mathrm{rank}(f(\mathbf{X}\mathbf{W}))$ generally gets closer to $d_{out}$ due to the nonlinearity when $d_{in}$ is not that small [^26]. Literature [^24] [^11] [^26] manifested the layer $\mathbf{W}$ with $d_{out}>d_{in}$ offers increased representational capacity. Intriguingly, DenseNets enjoy a similar aspect because we can decompose $\mathbf{W}=[\mathbf{W_{P}},\mathbf{I}]$, where $\mathbf{W_{P}}$ and $\mathbf{I}$ denote the weights in the building block and concatenation. We further argue that increasing rank like this frequently would be more beneficial. The output dimension of $\mathbf{W_{P}}$ is called growth rate.

A strategic design mitigates memory concerns. Consider the output of stacked layers $\mathbf{X}\mathbf{W_{1}}f(\mathbf{W_{2}})$, where the weights $\mathbf{W_{1}}{\in}\mathbb{R}^{d_{in}{\times}d_{r}}$, $\mathbf{W_{2}}{\in}\mathbb{R}^{d_{r}{\times}d_{out}}$, $d_{r}<d_{in},d_{out}$, and a nonlinearity $f$ after $\mathbf{W_{2}}$. Likewise, the rank is likely preserved as $d_{r}$ is not that small. This suggests that using intermediate dimension reducers like $\mathbf{W}_{1}$ (*i.e*., transition layer) may not impact the rank significantly. We argue a frequent application would effectively address memory concerns.

#### Pilot study.

We conduct a pilot study to verify our conjecture by sampling over 15k networks on Tiny-ImageNet [^46], where their shortcuts are either additive like ResNets [^30] or concatenation in DenseNets [^37]. We carefully control experiments regarding computational costs and involve diverse training setups to ensure a balanced and comprehensive comparison. Intriguingly, concatenation shortcuts all outperform additive ones with the averaged Tiny-ImageNet accuracy - 56.8±3.9 (concat) vs. 55.9±4.1 (add), thereby empirically supporting our claim. Detailed results and setups are provided in §5.1.

### 3.2 Revitalizing DenseNets

We revisit DenseNets while maintaining its core principle via concatenation. Our strategy explores ways to widen DenseNets and identify effective elements. Elements that contribute to the performance improvements are detailed in Table 1.

#### Baseline.

As the series of revisiting ResNets [^6] [^92] showed, refined training recipes bring significant improvements. Likewise, we train DenseNet-201 with a modern training setup, establishing it as our baseline. Following the well-explored setups [^76] [^55] [^54] [^75] [^92], we include Label Smoothing [^73], RandAugment [^15], Random Erasing [^103] [^18], Mixup [^101], CutMix [^100], and Stochastic Depth [^38]; we use AdamW [^57] with the cosine learning rate schedule [^56] and linear warmup [^22] with a popular large epochs training setup (*i.e*., 300).

Table 1: ImageNet-1K performance progressions. Beginning from the baseline - DenseNet-201 [^37], we report every performance change throughout progressions. We uphold DenseNet’s principle of feature reuse through concatenation as the core of the model progression. $+\alpha$ denotes a new element $\alpha$ was added to each prior model. Both enhancements in efficiency or accuracy are colored in red, while degradations are marked in blue.; GR denotes the growth rate, the amount of feature concatenation [^37].

<table><tbody><tr><td></td><td rowspan="2">Elements</td><td>Top-1</td><td>Param</td><td>  FLOPs</td><td>Lat (ms)</td><td>Lat (ms)</td><td>Lat (ms)</td><td>Mem (GB)</td></tr><tr><td></td><td>Acc (%)</td><td>(M)</td><td>  (G)</td><td>(b1, Infer)</td><td>(b128, Infer)</td><td>(Train)</td><td>(Train)</td></tr><tr><td>(a)</td><td>DenseNet-201 <sup><a href="#fn:37">37</a></sup></td><td>79.7</td><td>20.0</td><td>  4.3</td><td>38.4</td><td>190</td><td>131</td><td>3.9</td></tr><tr><td>(b)</td><td>(a) + Wider & shallower</td><td>79.5 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.2)</td><td>21.8 (+1.8)</td><td>11.1 (+6.8)</td><td>  8.5 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 29.9)</td><td>170 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 20)</td><td>  85 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 46)</td><td>3.2 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.7)</td></tr><tr><td>(c)</td><td>(b) + Modernized blocks</td><td>80.4 (+0.9)</td><td>12.9 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 8.9)</td><td>  4.8 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 6.3)</td><td>10.4 (+  2.9)</td><td>230 (+60)</td><td>112 (+27)</td><td>3.4 (+0.2)</td></tr><tr><td>(d)</td><td>(c) + Channel dim↑(GR↓)</td><td>80.8 (+0.4)</td><td>19.9 (+7.0)</td><td>  4.7 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.1)</td><td>11.8 (+  1.4)</td><td>184 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 46)</td><td>  88 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 24)</td><td>3.1 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.3)</td></tr><tr><td>(e)</td><td>(d) + Trans. layers↑(GR↑)</td><td>82.3 (+1.5)</td><td>21.2 (+1.3)</td><td>  5.0 (+0.3)</td><td>11.0 (+  0.8)</td><td>183 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math>   1)</td><td>    90 (+  2)</td><td>3.4 (+0.3)</td></tr><tr><td>(f)</td><td>(e) + Patchification stem</td><td>82.4 (+0.1)</td><td>21.2 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.0)</td><td>  4.9 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.1)</td><td>11.0 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math>   0.0)</td><td>179 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math>   6)</td><td>    88 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math>   2)</td><td>3.2 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.2)</td></tr><tr><td>(g)</td><td>(f) + Refined Trans. layers</td><td>82.6 (+0.2)</td><td>22.4 (+1.2)</td><td>  4.9 (+0.0)</td><td>13.6 (+  2.6)</td><td>170 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math>   9)</td><td>    97 (+  9)</td><td>3.1 (<math><semantics><mo>−</mo> <annotation>-</annotation></semantics></math> 0.1)</td></tr><tr><td>(h)</td><td>(g) + Channel re-scaling</td><td>82.8 (+0.2)</td><td>23.9 (+1.5)</td><td>  5.0 (+0.1)</td><td>14.0 (+  0.4)</td><td>175 (+  5)</td><td>    99 (+  2)</td><td>3.1 (+0.0)</td></tr></tbody></table>

#### Going wider and shallower.

DenseNets originally proposed exceedingly deep architectures (*e.g*., DenseNet-265 [^37]), which effectively showed the scalability. We argue that enhancing feature dimension through a high growth rate (GR) and increasing depth is hardly achieved simultaneously under resource constraints. Prior works [^87] [^47] [^64] [^26] [^55] designed shallower networks to achieve efficiency, particularly latency. Inspired by this, we modify DenseNet to a favorable baseline accordingly; widening the network by augmenting GR while diminishing its depth. Specifically, we vastly increase GR - from 32 to 120 here - to achieve it; we adjust the number of blocks per stage, being reduced from (6, 12, 48, 32) to a much smaller (3, 3, 12, 3) for a depth adjustment. We do not shrink the depth as much to maintain minimal nonlinearity. Table 1(b) shows this strategic modification has led to notable latencies and memory efficiency - around 35% and 18% decreases in training speed and memory, respectively. The marked increase in GFLOPs to 11.1 will be adjusted through the later elements. Further study supports our decision - prioritizing width while balancing depth (see Table 8a).

#### Improved feature mixers.

We employ the base block [^55] for our feature mixer block, which has been extensively studied to reveal its effectiveness. Before using it, we should reevaluate the studies for our case because 1) DenseNets did not use additive shortcuts, and 2) the building block was originally designed to reduce dimensions successively. We find using the following setups still holds: using 1) Layer Normalization (LN) [^3] instead of Batch Normalization (BN) [^40]; 2) post-activation; 3) depthwise convolution [^36] 4) fewer normalizations and activations; 5) a kernel size of 7. A unique aspect of our block is that the output channel (GR) is smaller than the input channel (C); mixed features are eventually more compressed features. As can be seen in Table 1(c), our design improves accuracy by a large margin (+0.9%p) while slightly increasing computational costs. We supplement factor analyses for our study here (see Table 8b).

Figure 1: Schematic illustration of RDNet. RDNet features a unique design distinguishing it from ResNet-style architectures, primarily due to the use of feature concatenation. We design four stages in RDNet across all scales, where each stage-N comprises $L_{N}$ mixing blocks consisting of three feature mixers and one transition layer (the last mixing block does not employ the transition layer). Feature mixer $f$ denotes our building block combines previously concatenated features to compress them into GR-dimensional features for concatenation. The growth rate (GR) adjusts the amount of concatenated features and is predetermined for each stage. Transition layers for downsampling are positioned after each stage as before. S and C denote stride and channel size. This figure illustratively sets GR to two.

#### Larger intermediate channel dimensions.

A large input dimension for the depthwise convolution is crucial [^68]. By adeptly modulating expansion ratio (ER) for inverted bottlenecks in the previous works [^68] [^74] [^26] [^75] [^55] successfully achieved significant performance, by enlarging intermediate tensor size within the block beyond input dimensions (*e.g*., ER was tuned to 6).

DenseNets similarly employed the ER concept; however, they distinctively applied it to the growth rate (GR) (*e.g*., ER=4 $\times$ GR) rather than to the input dimension to reduce both input and output dimensions. We argue that this harms the capability of encoded features through the nonlinearity [^26]. Thus, we reengineer the approach by directing ER proportional to the input dimension (*i.e*., decoupling ER from GR). This change results in increased computational demands from a larger intermediate dimension; thus, halving GR (*e.g*., from 120 to 60) manages these demands without compromising accuracy. Namely, we enrich the features before applying nonlinearity and further compress the channels to control computational costs. Thereafter, we achieve both a faster training speed of 21% and 0.4%p improvement in accuracy shown in Table 1. Additionally, we conduct a factor analysis to ascertain whether reducing ER and increasing GR is preferable, or conversely, elevating ER and decreasing GR; Table 8c displays employing GR of 4 ultimately yields the optimal results.

#### More transition layers.

The transition layers [^37] between stages are intended to reduce the number of channels. Due to the dense connections in every block, the intensified accumulation of features does not allow a high growth rate (GR). This gets worse as multiple blocks are stacked within a single stage, such as in the third stage, where numerous blocks accumulate in a single stage with low GRs. We introduce a novel aspect using more transition layers to address it. To be specific, we propose to use a transition layer in a stage, not solely after each stage, but after every three blocks with a stride of 1. These transition layers focus on dimension reduction rather than downsampling. This modification evidently reduces the computational costs substantially; therefore, we successfully increase overall GRs thanks to it <sup>1</sup>. This is further supported by the results in Table 8e, which reveals using transition layers frequently often improves accuracy.

Additionally, we note that the models exhibit low parameter counts compared to their FLOPs. We remedy this by introducing variable GR at different stages (*e.g*., 64, 104, 128, 192) instead of a uniform GR. Our further study in Table 8d suggests that a uniform growth rate (GR) compromises both accuracy and efficiency. Finally, Table 1(e) shows our design achieves significant accuracy improvements without greatly affecting computational costs.

#### Patchification stem

Recent advancements revealed the effectiveness of using image patches as inputs within a stem [^77] [^55] [^65]. We use the identical setup of a patch size 4 with a stride 4 as the patchification (LN [^3] follows). Our empirical findings suggest that employing the patchification yields a notable acceleration in computational speed without loss of precision (see Table 1(f)).

#### Refined transition layers

Another role of the transition layers was downsampling, and extra average poolings to downsample were adopted. We refine the transition layers, removing the average pooling and replacing the convolution by adjusting the kernel size and stride with the stride (LN replaces BN as well). Therefore, our transition layers play two additional roles: 1) dimension reduction, as aforementioned; 2) downsampling. Placing the transition layer after each stage exhibits +0.2%p gain, barely hurting efficiency (see Table 1(g)). For the dimension reduction ratio, we reexamine the impact, previously explored in [^37]; Table 8f reconfirms 0.5 is optimal; higher transition ratios degrade precision.

#### Channel re-scaling.

We investigate if channel re-scaling is required due to the diverse variance of concatenated features. We examine our proposed re-scaling approach, which has a similar formulation by merging the channel layer-scale [^76] and an effective squeeze-excitation network [^48]. Table 1(h) indicates it achieves a slight +0.2%p improvement, albeit with very minor inefficiency.

### 3.3 Revitialized DenseNet (RDNet)

We finally introduce Revitalized DenseNet (dubbed RDNet), illustrated in Fig. 1. Our final model achieves both enhanced precision and efficiency, particularly enjoying significantly faster speed (see Table 1(h) vs. Table 1(a)). RDNet model family aligns with the widely-adopted scales [^30] [^54] [^55]. Our models distinctively include the Growth Rate, GR= $(GR_{1},GR_{2},GR_{3},GR_{4})$, and the number of the feature mixers in each stage, B= $(B_{1},B_{2},B_{3},B_{4})$, where we assign the number of the feature mixers per each stage being a multiple of 3 (*i.e*., $B_{N}{=}3L_{N}$), where $L_{N}$ is the number of the mixing blocks. We summarize the configurations below:

- RDNet-T: $\text{GR}=(64,104,128,224),\text{B}=(3,3,12,3)$
- RDNet-S: $\text{GR}=(64,128,128,240),\text{B}=(3,3,21,6)$
- RDNet-B: $\text{GR}=(96,128,168,336),\text{B}=(3,3,21,6)$
- RDNet-L: $\text{GR}=(128,192,256,360),\text{B}=(3,3,24,6)$

## 4 Experiment

### 4.1 Image Classification

We evaluate our model family on ImageNet-1K [^67]. Our models are trained following the training setups in Swin Transformer [^54] and ConvNeXt [^55] to ensure a fair comparison and not aimed to finetune the setups. The models are trained using AdamW [^57] with a batch size of 512 and an initial learning rate of 1e-4 for 300 epochs. As aforementioned in our baseline in §3.2, we employed identical data augmentations/regularization techniques to ConvNeXt’s; EMA is not used for our training. The comprehensive details of the training recipe are detailed in Appendix. We follow the standard evaluation protocols [^30] [^54] [^55].

Our superiority is first underscored as compared with those of the current top-performing architectures [^65] [^27] [^23] [^52] [^105]. We visualize the trade-off plots in Fig. 2 and detail the accuracies with diverse computational costs in Table 2. Ours show very competitive results compared with state-of-the-art models. Table 2 exhibits that while our models slightly fall behind in accuracy, they significantly make up with speed metrics. For example, RDNet-S can match with other lighter models such as SMT-S or MogaNet-S. Notably, ours do not require large memory usage as we aimed but achieve further efficiency.

We further exhibit a comparison with the popular models in Table 4. Ours surpass competitors by high precision, with decent memory usage and faster speeds. We further visualize trade-offs in Fig. 3, where RDNet demonstrates competitive performance even when juxtaposed with the milestone architectures.

Figure 2: ImageNet-1K performance trade-off among state-of-the-arts. We provide comparative visualizations among state-of-the-art models, which were known for top-performing models. It turns out that RDNet is highly competitive in practice in terms of model speed and memory consumption.

Table 2: ImageNet-1K comparison with the latest models. Fig. 2 visualized this table. We thoroughly compare our models against the latest architectures in practical latency and memory usage to demonstrate superiority. b $n$ denotes latency (ms), measured with a batch size of $n$. Mem denotes the memory occupation (GB) measured with a batch size of 16. Interestingly, while our models slightly lag in accuracy, they significantly compensate with superior speed metrics.

| Model | Date | Param | FLOPs | Top-1 | b1 | b8 | b16 | b32 | b64 | b128 | Mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RDNet-T | Ours | 24 | 5.0 | 82.8 | 7.4 | 13.4 | 24.6 | 45.7 | 88.9 | 175.2 | 4.1 |
| HorNet-T <sub>7×7</sub> [^65] | NeurIPS’2022 | 22 | 4.0 | 82.8 | 21.2 | 23.2 | 27.0 | 50.7 | 96.1 | 183.7 | 4.1 |
| VAN-B2 [^23] | CVMJ’2023 | 27 | 5.0 | 82.8 | 24.2 | 28.0 | 39.0 | 75.4 | 144.4 | 274.8 | 5.2 |
| BiFormer-S [^105] | CVPR’2023 | 26 | 4.5 | 83.8 | 51.6 | 50.5 | 50.9 | 86.8 | 167.5 | 197.2 | 8.5 |
| NAT-T [^27] | CVPR’2023 | 28 | 4.3 | 83.2 | 26.5 | 28.0 | 33.2 | 53.0 | 102.2 | 335.3 | 3.8 |
| SMT-S [^52] | ICCV’2023 | 21 | 4.7 | 83.7 | 46.9 | 48.2 | 55.8 | 96.0 | 176.5 | 335.3 | 5.3 |
| MogaNet-S [^49] | ICLR’2024 | 25 | 5.0 | 83.4 | 20.0 | 22.4 | 40.9 | 77.2 | 147.4 | 288.1 | 6.4 |
| RDNet-S | Ours | 50 | 8.7 | 83.7 | 11.9 | 21.5 | 39.8 | 74.0 | 144.2 | 289.0 | 5.4 |
| HorNet-S <sub>7×7</sub> [^65] | NeurIPS’2022 | 50 | 8.8 | 84.0 | 23.2 | 25.7 | 46.0 | 88.3 | 171.4 | 328.9 | 5.7 |
| VAN-B3 [^23] | CVMJ’2023 | 45 | 9.0 | 83.9 | 45.1 | 49.4 | 64.1 | 123.1 | 237.2 | 446.9 | 7.4 |
| BiFormer-B [^105] | CVPR’2023 | 57 | 9.8 | 84.3 | 60.4 | 67.8 | 85.8 | 161.2 | 311.9 | 584.2 | 12.2 |
| NAT-S [^27] | CVPR’2023 | 51 | 7.8 | 83.7 | 28.1 | 28.2 | 43.4 | 82.7 | 159.9 | 310.4 | 5.2 |
| SMT-B [^52] | ICCV’2023 | 32 | 7.7 | 84.3 | 69.3 | 70.9 | 87.1 | 149.6 | 272.5 | 518.7 | 7.8 |
| MogaNet-B [^49] | ICLR’2024 | 44 | 9.9 | 84.3 | 37.9 | 43.8 | 80.9 | 152.9 | 294.0 | 576.6 | 11.1 |
| RDNet-B | Ours | 87 | 15.4 | 84.4 | 11.7 | 32.2 | 61.4 | 116.6 | 233.7 | 471.6 | 6.9 |
| HorNet-B <sub>7×7</sub> [^65] | NeurIPS’2022 | 87 | 15.6 | 84.3 | 22.9 | 37.9 | 71.5 | 134.5 | 259.6 | 500.0 | 7.7 |
| VAN-B4 [^23] | CVMJ’2023 | 60 | 12.2 | 84.2 | 60.4 | 67.8 | 85.8 | 161.2 | 311.9 | 584.2 | 9.0 |
| NAT-B [^27] | CVPR’2023 | 90 | 13.7 | 84.3 | 28.3 | 33.5 | 43.4 | 82.7 | 159.9 | 310.4 | 5.2 |
| MogaNet-L [^49] | ICLR’2024 | 83 | 15.9 | 84.7 | 60.8 | 64.9 | 118.5 | 224.3 | 429.2 | 838.6 | 14.9 |
| RDNet-L | Ours | 186 | 34.7 | 84.8 | 15.7 | 63.2 | 121.0 | 233.3 | 460.7 | 933.7 | 10.9 |
| MogaNet-XL [^49] | ICLR’2024 | 181 | 34.5 | 85.1 | 66.3 | 112.3 | 207.5 | 394.0 | 771.9 | 1512.5 | 24.1 |

### 4.2 Zero-shot Image Classification

Table 3: ImageNet-1K zero-shot classification results. Ours outperforms ConvNeXt further in efficiency.

| Models | Param | Top-1 | Top-5 |
| --- | --- | --- | --- |
| ConvNeXt-B | 152 | 51.2 | 79.3 |
| RDNet-B | 150 | 54.1 | 82.1 |

We evaluate RDNet on ImageNet-1k zero-shot performance by training CLIP [^63] to verify the applicability under a different training scheme. We follow the training protocol in ConvNeXt-OpenCLIP [^12] using 1.28B seen images from the aggregated set of CC3M [^69], CC12M [^10], and RedCaps [^17]. We use the OpenCLIP codebase <sup>2</sup>.

Figure 3: ImageNet-1K performance trade-off among previous milestones. We provide comparative visualizations between previous architectures and our models. Notice that we also include speed comparisons to highlight actual differences in practice. Our models outperform the competing modern architectures revealing the potential of feature concatenation in designing networks.

Table 4: ImageNet-1K performance comparison with milestones. We report top-1 accuracy (%), parameter count (M), FLOPs (G), inference time (ms) for 128 images, and memory usage (GB) with a batch size of 16. All models are (pre-)trained on ImageNet-1k from scratch.

| Model | Res | Param | FLOPs | Lat | Mem | Top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| RSB-ResNet50 [^92] | $224^{2}$ | 26 | 4.1 | 115 | 2.1 | 80.4 |
| RegNetY-4GF [^64] | $224^{2}$ | 21 | 4.0 | 128 | 2.7 | 81.5 |
| Deit-S [^76] | $224^{2}$ | 22 | 4.6 | 128 | 1.9 | 79.8 |
| CoaT-Lite-S [^50] | $224^{2}$ | 22 | 4.0 | 211 | 3.3 | 81.9 |
| Swin-T [^54] | $224^{2}$ | 28 | 4.5 | 173 | 2.6 | 81.3 |
| PVTv2-B2-Li [^89] | $224^{2}$ | 23 | 3.9 | 173 | 4.4 | 82.1 |
| FocalNet-T [^96] | $224^{2}$ | 29 | 4.5 | 181 | 4.0 | 82.3 |
| ConvNeXt-T [^55] | $224^{2}$ | 29 | 4.5 | 150 | 2.7 | 82.1 |
| CSWin-T [^20] | $224^{2}$ | 23 | 4.3 | 194 | 2.7 | 82.8 |
| Deit III-S [^78] | $224^{2}$ | 22 | 4.6 | 128 | 2.0 | 81.4 |
| RevCol-T [^8] | $224^{2}$ | 30 | 4.5 | 189 | 2.0 | 82.2 |
| SLaK-T [^53] | $224^{2}$ | 30 | 5.0 | 238 | 3.3 | 82.5 |
| InceptionNeXt-T [^99] | $224^{2}$ | 28 | 4.2 | 132 | 3.3 | 82.3 |
| RDNet-T | $224^{2}$ | 24 | 5.0 | 175 | 4.1 | 82.8 |
| RSB-ResNet101 [^92] | $224^{2}$ | 45 | 7.9 | 190 | 3.9 | 81.5 |
| RegNetY-8GF [^64] | $224^{2}$ | 39 | 8.0 | 238 | 4.0 | 82.2 |
| NFNet-F0 [^7] | $224^{2}$ | 71 | 12.4 | 235 | 3.5 | 83.6 |
| CoaT-Lite-M [^50] | $224^{2}$ | 45 | 9.8 | 396 | 5.5 | 83.6 |
| Swin-S [^54] | $224^{2}$ | 50 | 8.7 | 293 | 3.9 | 83.0 |
| PVTv2-B4 [^89] | $224^{2}$ | 63 | 10.1 | 370 | 7.2 | 83.6 |
| ConvNeXt-S [^55] | $224^{2}$ | 50 | 8.7 | 266 | 4.0 | 83.1 |
| CSWin-S [^20] | $224^{2}$ | 35 | 6.9 | 313 | 4.0 | 83.6 |
| FocalNet-S [^96] | $224^{2}$ | 50 | 8.7 | 313 | 4.6 | 83.5 |
| RevCol-S [^8] | $224^{2}$ | 60 | 9.0 | 377 | 2.4 | 83.5 |
| SLaK-S [^53] | $224^{2}$ | 55 | 9.8 | 372 | 5.0 | 83.8 |
| InceptionNeXt-S [^99] | $224^{2}$ | 49 | 8.4 | 245 | 3.2 | 83.5 |
| RDNet-S | $224^{2}$ | 50 | 8.7 | 289 | 5.4 | 83.7 |

| Model | Res | Param | FLOPs | Lat | Mem | Top-1 |
| --- | --- | --- | --- | --- | --- | --- |
| RSB-ResNet152 [^92] | $224^{2}$ | 60 | 11.6 | 270 | 4.7 | 82.0 |
| RegNetY-16GF [^64] | $224^{2}$ | 84 | 15.9 | 389 | 5.4 | 82.2 |
| DeiT-B [^76] | $224^{2}$ | 87 | 17.5 | 418 | 3.8 | 81.8 |
| Swin-B [^54] | $224^{2}$ | 89 | 15.4 | 445 | 5.4 | 83.5 |
| PVTv2-B5 [^89] | $224^{2}$ | 82 | 11.8 | 414 | 7.0 | 83.8 |
| ConvNeXt-B [^55] | $224^{2}$ | 89 | 15.4 | 417 | 5.4 | 83.8 |
| CSWin-B [^20] | $224^{2}$ | 78 | 15.0 | 543 | 6.5 | 84.2 |
| RepLKNet-31B [^19] | $224^{2}$ | 79 | 15.3 | 461 | 2.7 | 83.5 |
| DeiT III-B [^78] | $224^{2}$ | 87 | 17.5 | 422 | 4.0 | 83.8 |
| FocalNet-B [^96] | $224^{2}$ | 89 | 15.4 | 476 | 6.1 | 83.9 |
| RevCol-B [^8] | $224^{2}$ | 138 | 16.6 | 653 | 3.5 | 84.1 |
| SLaK-B [^53] | $224^{2}$ | 95 | 17.1 | 558 | 6.9 | 84.0 |
| InceptionNeXt-B [^99] | $224^{2}$ | 87 | 14.9 | 405 | 6.1 | 84.0 |
| RDNet-B | $224^{2}$ | 87 | 15.4 | 472 | 6.9 | 84.4 |
| RegNetY-32GF [^64] | $224^{2}$ | 145 | 32.3 | 638 | 7.3 | 82.5 |
| NFNet-F1 [^7] | $320^{2}$ | 133 | 35.5 | 421 | 5.9 | 84.7 |
| DeiT III-L [^78] | $224^{2}$ | 304 | 61.6 | 1375 | 10.5 | 84.9 |
| DeiT III-L [^78] | $384^{2}$ | 304 | 191.2 | 4586 | 28.1 | 85.8 |
| ConvNeXt-L [^55] | $224^{2}$ | 198 | 34.4 | 857 | 8.6 | 84.3 |
| ConvNeXt-L [^55] | $384^{2}$ | 198 | 101.1 | 2550 | 19.0 | 85.5 |
| RDNet-L | $224^{2}$ | 186 | 34.7 | 934 | 10.9 | 84.8 |
| RDNet-L | $384^{2}$ | 186 | 101.9 | 2714 | 24.3 | 85.8 |

### 4.3 Semantic Segmentation

We employ ImageNet-1K pre-trained weights to perform semantic segmentation on the ADE20K [^104] dataset using UperNet [^94]. We use a learning rate of 8e-5 with a weight decay of $0.03$, and utilize stochastic depth rate $0.1$, $0.2$, and $0.3$ for the RDNet-T, -S, and -B, respectively. The remainder of the training settings follows ConvNeXt [^55]. As demonstrated in Table 5, RDNet exhibits strong performance, which reveals the effectiveness on dense prediction tasks.

Table 5: ADE20K semantic segmentation results. All trained with the unified head UperNet (160K) on ADE20K. FLOPs (G) are measured at $512\times 2048$ resolutions.

| Architecture | Crop | Param | FLOPs | mIoU <sup>ss</sup> | mIoU <sup>ms</sup> |
| --- | --- | --- | --- | --- | --- |
| Swin-T [^54] | 512 <sup>2</sup> | 60 | 945 | 44.5 | 46.1 |
| ConvNeXt-T [^55] | 512 <sup>2</sup> | 60 | 939 | 46.0 | 46.7 |
| RevCol-T [^8] | 512 <sup>2</sup> | 60 | 937 | 47.4 | 47.8 |
| NAT-T [^27] | 512 <sup>2</sup> | 58 | 934 | 47.1 | 48.4 |
| RDNet-T | 512 <sup>2</sup> | 58 | 961 | 47.6 | 48.6 |
| Swin-S [^54] | 512 <sup>2</sup> | 81 | 1038 | 47.6 | 49.5 |
| ConvNeXt-S [^55] | 512 <sup>2</sup> | 82 | 1027 | 48.7 | 49.6 |
| RevCol-S [^8] | 512 <sup>2</sup> | 90 | 1031 | 47.9 | 49.0 |
| NAT-S [^27] | 512 <sup>2</sup> | 82 | 1010 | 48.0 | 49.5 |
| RDNet-S | 512 <sup>2</sup> | 86 | 1040 | 48.7 | 49.8 |
| Swin-B [^54] | 512 <sup>2</sup> | 121 | 1188 | 48.1 | 49.7 |
| ConvNeXt-B [^55] | 512 <sup>2</sup> | 122 | 1170 | 49.1 | 49.9 |
| DeiT III-B [^78] | 512 <sup>2</sup> | 128 | 1283 | 49.3 | 50.2 |
| RevCol-B [^8] | 512 <sup>2</sup> | 122 | 1169 | 49.0 | 50.1 |
| NAT-B [^27] | 512 <sup>2</sup> | 123 | 1137 | 48.5 | 49.7 |
| RDNet-B | 512 <sup>2</sup> | 127 | 1187 | 49.6 | 50.5 |

### 4.4 Object Detection

We evaluate object detection performance on COCO [^51] using Mask-RCNN [^29]. We use a learning rate of 3e-5 with a stochastic depth rate of 0.2. The remainder of the training settings follows ConvNeXt [^55] again. As demonstrated in Table 6, RDNet exhibits competitive performance.

Table 6: COCO object detection and segmentation results. We utilize Mask-RCNN with 3x schedule. FLOPs (G) are calculated with image size (1280, 800). The result of Swin-T is from the official repository [^1].

| Backbone | Param | FLOPs | $\text{AP}^{\text{box}}$ | $\text{AP}^{\text{box}}_{50}$ | $\text{AP}^{\text{box}}_{75}$ | $\text{AP}^{\text{mask}}$ | $\text{AP}^{\text{mask}}_{\text{50}}$ | $\text{AP}^{\text{mask}}_{75}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PVT-S [^88] | 44M | 304G | 43.0 | 65.3 | 46.9 | 39.9 | 62.5 | 42.8 |
| Swin-T [^54] | 48M | 267G | 46.0 | 68.1 | 50.3 | 41.6 | 65.1 | 44.9 |
| ConvNeXt-T [^55] | 48M | 262G | 46.2 | 67.9 | 50.8 | 41.7 | 65.0 | 44.9 |
| RDNet-T | 43M | 278G | 47.5 | 68.5 | 52.1 | 42.4 | 65.6 | 45.7 |

## 5 Discussions

### 5.1 Pilot Study - Random Network Experiments

This study aims to reveal the effectiveness of dense connections over residual connections. We train tons of random networks across various scenarios, which include 1) multiple network scales; 2) multiple types of building blocks; 3) a range of network architectural elements; and 4) different training setups.

#### Parameter spaces and cost constraints.

Table 7 (left) shows our parameter spaces for three individual scales, where RandNet <sub>A,B,C</sub> are trained. We diversify the search space with respect to the budgets, such as parameter count, FLOPs, and memory consumption. We expand space from $\mathcal{C}$ to $\mathcal{D}$ by incorporating data augmentation and further to $\mathcal{E}$ with both data augmentation and a different optimizer [^57]. Only randomly generated networks that meet the predefined budget are trained. We use the 90-epochs training setup [^30] trained on Tiny-ImageNet [^93]. For $\mathcal{C,E}$ spaces using data augmentation [^73] [^101] [^100] [^103] [^38] [^15], training is done for 180 epochs. Overall, the cumulative number of trained networks reach over 15k.

Table 7: Random network experiments. We present our experimental setups (left) and results (right). Five parameter spaces guide random network generations for two distinct shortcuts. We sample random networks within each parameter space, ensuring similar computational costs. Each parameter space varies in 1) architectural elements - channel sizes, activations, normalizations, and convolution kernel sizes in $\mathcal{A,B,C,D,E}$; 2) data augmentations in $\mathcal{D,E}$; 3) optimizers in $\mathcal{E}$. $\mathcal{D,E}$ is based on the architectural space $\mathcal{C}$. \[$a_{1},\dots,a_{n}$\] and ($a$, $b$, $c$) denote a closed interval: a list of $n$ elements and a range of elements from $a$ to $b$ with a step of $c$, respectively. All results are averaged.

<table><tbody><tr><td>Parameter space</td><td><math><semantics><mi>𝒜</mi> <annotation>{\mathcal{A}}</annotation></semantics></math></td><td colspan="2">  <math><semantics><mi>ℬ</mi> <annotation>{\mathcal{B}}</annotation></semantics></math></td><td><math><semantics><mi>𝒞</mi> <annotation>{\mathcal{C}}</annotation></semantics></math></td></tr><tr><td>Param (<math><semantics><mi>x</mi> <annotation>x</annotation></semantics></math> M)</td><td><math><semantics><mrow><mn>2</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>2.5</mn></mrow> <annotation>2{<}x{<}2.5</annotation></semantics></math></td><td colspan="2">  <math><semantics><mrow><mn>4</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>5</mn></mrow> <annotation>4{<}x{<}5</annotation></semantics></math></td><td><math><semantics><mrow><mn>9</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>10</mn></mrow> <annotation>9{<}x{<}10</annotation></semantics></math></td></tr><tr><td>FLOPs (<math><semantics><mi>x</mi> <annotation>x</annotation></semantics></math> G)</td><td><math><semantics><mrow><mn>2</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>2.5</mn></mrow> <annotation>2{<}x{<}{2.5}</annotation></semantics></math></td><td colspan="2">  <math><semantics><mrow><mn>4</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>5</mn></mrow> <annotation>4{<}x{<}5</annotation></semantics></math></td><td><math><semantics><mrow><mn>9</mn> <mo><</mo> <mi>x</mi> <mo><</mo> <mn>10</mn></mrow> <annotation>9{<}x{<}10</annotation></semantics></math></td></tr><tr><td>Depth</td><td>(3, 6, 1)</td><td colspan="2">  (3, 8, 1)</td><td>(3, 12, 1)</td></tr><tr><td>Inter. channel dim</td><td>(32, 96, 8)</td><td colspan="2">  (64, 128, 8)</td><td>(64, 192, 8)</td></tr><tr><td>Output channel dim</td><td>(32, 96, 8)</td><td colspan="2">  (64, 128, 8)</td><td>(64, 192, 8)</td></tr><tr><td>Activations</td><td colspan="4">[ReLU, SiLU, Mish, GELU, LeakyReLU]</td></tr><tr><td>Normalization layers</td><td colspan="4">[BatchNorm, LayerNorm]</td></tr><tr><td>Kernel sizes</td><td colspan="4">[3, 5, 7, 9]</td></tr><tr><td>Parameter space</td><td colspan="2">      <math><semantics><mi>𝒟</mi> <annotation>{\mathcal{D}}</annotation></semantics></math></td><td colspan="2"><math><semantics><mi>ℰ</mi> <annotation>{\mathcal{E}}</annotation></semantics></math></td></tr><tr><td>Base space</td><td colspan="2">      <math><semantics><mi>𝒞</mi> <annotation>\mathcal{C}</annotation></semantics></math></td><td colspan="2"><math><semantics><mi>𝒞</mi> <annotation>\mathcal{C}</annotation></semantics></math></td></tr><tr><td>Optimizer</td><td colspan="2">      -</td><td colspan="2">AdamW</td></tr><tr><td>Data augmentation</td><td colspan="2">      ✓</td><td colspan="2">✓</td></tr></tbody></table>

| Model | Skip | FLOPs | Param | Mem | Top-1 (%) |
| --- | --- | --- | --- | --- | --- |
| RandNet <sub>A</sub> | add | 2.25±0.13 | 2.21±0.13 | 0.65±0.03 | 45.8±2.0 |
| RandNet <sub>A</sub> | concat | 2.24±0.14 | 2.24±0.13 | 0.75±0.08 | 47.5±2.1 |
| RandNet <sub>B</sub> | add | 4.53±0.27 | 4.44±0.26 | 0.78±0.05 | 50.1±2.1 |
| RandNet <sub>B</sub> | concat | 4.53±0.29 | 4.51±0.28 | 0.90±0.11 | 51.2±2.1 |
| RandNet <sub>C</sub> | add | 9.61±0.23 | 9.41±0.23 | 1.01±0.09 | 53.2±2.2 |
| RandNet <sub>C</sub> | concat | 9.53±0.26 | 9.44±0.26 | 1.24±0.17 | 54.3±2.0 |
| RandNet <sub>D</sub> | add | 9.60±0.23 | 9.40±0.23 | 1.02±0.09 | 57.4±1.5 |
| RandNet <sub>D</sub> | concat | 9.54±0.26 | 9.44±0.26 | 1.24±0.16 | 58.1±1.4 |
| RandNet <sub>E</sub> | add | 9.59±0.23 | 9.38±0.23 | 1.02±0.09 | 58.2±1.5 |
| RandNet <sub>E</sub> | concat | 9.54±0.26 | 9.44±0.26 | 1.25±0.17 | 58.9±1.6 |

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.19588/assets/New_RandNet_cumulate_tiny.png)

(a) RandNet A

#### RandNet architecture.

Based on [^30], we stack random building blocks within the first stage. We generate random networks in the parameter space containing diverse depths, widths, activations, normalizations, and kernel sizes to provide flexibility under constrained costs (see Table 7). Additionally, we diversify building blocks across all search spaces to conduct more extensive experiments. Three distinct architectural blocks - dubbed PreNorm, PostNorm, and PostNorm (w/o act) - are differentiated by the use of pre-activation and shortcut positions. PreNorm block adopts the pre-normalization [^31] [^37] precedes a skip connection. In contrast, two PostNorms enjoy post-normalization [^30] [^55]. PostNorm varies from PostNorm (w/o act) based on the activation function post-skip connection.

#### Result interpretation.

Table 7 (right) exhibits that concatenation consistently outperforms additive shortcuts across all configurations. Furthermore, Fig. 4 demonstrates the superior capability of concatenation-based architectures.

### 5.2 Impact of Input Size on Performance

Figure 5: Accuracy/latency/memory vs. resolution. RDNet enjoys resolution-robustness against various input image sizes to maintain accuracy. Furthermore, RDNet exhibits a similar latency/memory trend to ConvNeXt and Swin Transformer, maintaining minimal increase with larger images compared to DeiT-S and DenseNet161.

We provide compelling findings regarding versus input size. First, Fig. 5 (left) shows RDNet enjoys strong adaptability to input size variations. Intriguingly, DenseNet161, even trained without strong data augmentations, still enjoys adaptability, surpassing DeiT-S trained with strong data augmentations. We attribute this to the effectiveness of dense connections.

Our finding further shows that, unlike width-oriented networks that slow with larger input sizes (due to the large intermediate tensors), our model’s optimized width avoids latency/memory loss. Fig. 5 (middle, right) illustrates that RDNet compete with ConvNeXt and Swin Transformer, diverging from DenseNet161 [^37] that gets slower and consumes more memory as image size grows. We note that larger scales (*e.g*., -S, -B, and -L) all follow the same trend.

### 5.3 CKA analysis

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.19588/assets/cka_beforeskip_rdnet_t.png)

(a) RDNet -T

We analyze the layer-specific features of RDNet compared to ConvNeXt using Centered Kernel Alignment (CKA) [^42]. Fig. 6 displays RDNet learns distinct features at every layer, showcasing different patterns compared to ConvNeXt. In the third column, ConvNeXt and RDNet astonishingly learn different features when compared, highlighting the unique learning dynamics of each model.

Table 8: Ablation study results are reported here with each ImageNet-1K accuracy (%) with parameter count (M) and FLOPS (G). The best options are marked in gray.

| Depth | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 3, 3,   9, 3 | 23.3 | 5.0 | 82.5 |
| 3, 3, 12, 3 | 23.9 | 5.0 | 82.8 |
| 3, 3, 15, 3 | 23.5 | 5.0 | 82.8 |
| 3, 3, 18, 6 | 50.3 | 8.7 | 83.5 |
| 3, 3, 21, 6 | 50.4 | 8.7 | 83.7 |
| 3, 3, 24, 6 | 49.9 | 8.7 | 83.6 |
| 3, 3, 18, 6 | 89.2 | 15.4 | 84.2 |
| 3, 3, 21, 6 | 86.2 | 15.4 | 84.4 |
| 3, 3, 24, 6 | 87.4 | 15.4 | 84.2 |

(a) Depth/width scaling

| Block conf | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| (a) RDNet-T | 23.9 | 5.0 | 82.8 |
| (a) + more act | 23.9 | 5.0 | 81.9 |
| (a) $\leftrightarrow$ 3 ${\times}$ 3 dwconv | 23.5 | 4.9 | 82.4 |
| (a) $\leftrightarrow$ 5 ${\times}$ 5 dwconv | 23.7 | 5.0 | 82.5 |
| (a) $\leftrightarrow$ 9 ${\times}$ 9 dwconv | 23.9 | 5.1 | 82.8 |
| (a) $\leftrightarrow$ 11 ${\times}$ 11 dwconv | 24.4 | 5.2 | 82.7 |
| (a) $\leftrightarrow$ 13 ${\times}$ 13 dwconv | 24.8 | 5.3 | 82.7 |
| (b) (a) $\leftrightarrow$ dwconv at last | 23.6 | 5.0 | 82.3 |
| (b) + more act/norm | 23.6 | 5.0 | 81.1 |
| (b) LN $\leftrightarrow$ BN | 23.6 | 5.0 | 82.2 |

(b) Block configuration

| ER | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 1.0 | 24.4 | 5.0 | 82.1 |
| 2.0 | 23.8 | 5.0 | 82.6 |
| 3.0 | 24.2 | 5.0 | 82.7 |
| 4.0 | 23.9 | 5.0 | 82.8 |
| 6.0 | 24.3 | 5.0 | 82.6 |

(c) Expansion ratio (ER)

| GR | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 90,   90,   90,   90 | 13.2 | 5.0 | 81.6 |
| 120, 120, 120, 120 | 23.9 | 8.9 | 83.0 |
| 64, 104, 128, 224 | 23.9 | 5.0 | 82.8 |

(d) Growth rate (GR)

| Interval | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 2 | 24.3 | 5.0 | 82.7 |
| 3 | 23.9 | 5.0 | 82.8 |
| 4 | 24.1 | 5.0 | 82.6 |
| 6 | 23.7 | 5.0 | 82.1 |

(e) Transition layer intervals

| Ratio | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 0.3 | 24.4 | 5.0 | 82.6 |
| 0.4 | 23.9 | 5.0 | 82.6 |
| 0.5 | 23.9 | 5.0 | 82.8 |
| 0.6 | 23.8 | 5.0 | 82.5 |
| 0.7 | 23.6 | 5.0 | 82.3 |

(f) Transition ratio

### 5.4 Revisiting Stochastic Depth

Figure 7: Stochastic depth proves effective with dense connections. It still acts as a regularizer.

Notably, DenseNets primitively did not employ Stochastic Depth [^38] for model training due to sharing the similarity in connectivity patterns of networks.

Table 9: Stochastic depth is compatible with dense connections.

| Ratio | Param | FLOPs | Top-1 |
| --- | --- | --- | --- |
| 0 | 23.9 | 5.0 | 81.6 |
| 0.05 | 23.9 | 5.0 | 82.5 |
| 0.10 | 23.9 | 5.0 | 82.6 |
| 0.15 | 23.9 | 5.0 | 82.8 |
| 0.20 | 23.9 | 5.0 | 82.6 |

We posit that Stochastic Depth should not be overlooked; our results demonstrate a noticeable improvement when it is incorporated into our model, as illustrated in Fig. 7. We also observe that a small stochastic depth ratio affects profoundly (see Table 9).

### 5.5 Ablation Studies

We gather all ablation studies in Table 8. Each table contains several models that are meticulously adjusted for almost equivalent computational costs with others to ensure a fair comparison of our specific focuses. Our methodology, in §3.2, methodically referenced each study.

## 6 Conclusion

In this paper, we have revisited the past success of DenseNet, which once outperformed ResNet in this era dominated by models using addition-based shortcuts, such as ResNet, ConvNeXt, and ViT. We have first rediscovered the potential of DenseNet, focusing on the underappreciated fact that DenseNet’s concatenation shortcuts surpass the expressivity of the convention of ResNet-style addition-based shortcuts through our pilot study. We then highlight the outdated training setups and classical macro-block designs that diminish DenseNet’s effectiveness against modernized architectures. By achieving our goal to widen DenseNet with modernized elements, we have proven that DenseNet’s foundational principles are competitive in achieving robust modeling performance on their own. Our models exhibit strong performance competitive to the latest modern architectures; the employment of diverse concatenated features has significantly enhanced performance in dense prediction tasks, showcasing an advantage overlooked in models utilizing addition shortcuts. We hope that our work sheds light on the advantages of using concatenations in network design, advocating for the consideration of DenseNet-style architectures alongside ResNet-style ones.

Limitations. Our models have been scaled to a ‘-large’ level, but resource limitation prevents more extensions to upper scales such as ViT-G.

## References

Appendix

In this Appendix, we provide additional experiments and details to complement the main paper. The contents are as follows:

- §0.A presents our experimental setups for ImageNet and downstream tasks training and evaluation setups;
- §0.B presents detailed layer configuration of RDNet-T;
- §0.C evaluates further ImageNet top-accuracy versus latency trade-offs on various testbeds, encompassing both PyTorch and TensorRT A100 inference, as well as CPU inference outcomes;
- §0.D conducts an ablation study by aligning the depth, parameters, and FLOPs with ResNet and ConvNeXt to reaffirm the strengths of dense connections;
- §0.E reports further COCO [^51] object detection and instance segmentation results with Mask-RCNN [^29] and Cascade Mask-RCNN [^9];
- §0.F benchmarks the transferability of ImageNet-1K pre-trained models. We utilize fine-grained classification datasets and long-tailed classification datasets, including CIFAR-10 [^44], CIFAR-100 [^44], Flowers-102 [^59], Stanford-Cars [^43], iNaturalist-2018 [^34], and iNaturalist-2019 [^35];
- §0.G evaluates the effectiveness of dense connections on generative models;
- §0.H benchmarks robustness of the ImageNet-1K [^67] pre-trained models on the out-of-distribution datasets, including ImageNet-V2 [^66], ObjectNet [^5], ImageNet-A [^33], ImageNet-Sketch [^85], and ImageNet-R [^32];
- §0.I gives more details of our pilot study described in §5.1 with overall results, and scaled-up experiments on ImageNet-1K.

Table A: ImageNet-1K training settings. Most training setups are consistently used except for the multiple stochastic depth rates (e.g., 0.15/0.35/0.4/0.45) that regularize the corresponding models (e.g., RDNet-T/S/B/L), respectively.

|  | RDNet-T/S/B/L | RDNet-L |
| --- | --- | --- |
|  | (Pre-)Training | Fine-Tuning |
| image size | 224 | 384 |
| weight init | kaiming normal | pre-trained |
| optimizer | AdamW | AdamW |
| base learning rate | 1e-3 | 2e-5 |
| weight decay | 0.05 | 1e-8 |
| optimizer momentum ($\beta_{1},\beta_{2}$) | 0.9, 0.999 | 0.9, 0.999 |
| batch size | 512 | 512 |
| training epochs | 300 | 30 |
| learning rate schedule | cosine decay | cosine decay |
| warmup epochs | 20 | 5 |
| warmup schedule | linear | linear |
| layer-wise lr decay [^14] [^4] | None | 0.7 |
| randaugment [^15] | (9, 0.5) | (9, 0.5) |
| mixup [^101] | 0.8 | 0.0 |
| cutmix [^100] | 1.0 | 0.0 |
| random erasing [^103] | 0.25 | 0.25 |
| label smoothing [^73] | 0.1 | 0.1 |
| stochastic depth [^39] | 0.15/0.35/0.4/0.5 | 0.6 |
| layer scale [^79] | 1e-6 | pre-trained |
| head init scale [^79] | None | 1e-3 |
| gradient clip | None | None |
| center crop percent | 0.9 | 1.0 |
| exp. mov. avg. (EMA) [^62] | None | None |

## Appendix 0.A Experimental Settings

### 0.A.1 ImageNet Training

Table A presents the training settings for RDNet on ImageNet-1K. Each variant of RDNet adheres to these settings, except for the stochastic depth rate [^39], which is tailored to each model variant. For fine-tuning, we group three consecutive feature mixer blocks for layer-wise learning rate decay [^4] [^14] akin to the approach taken in ConvNeXt. We use the timm package [^91] for model training.

### 0.A.2 Downstream Tasks

We adhere to the hyper-parameter sweep protocol outlined in [^55] but sweep much lightly. For UperNet [^94] training on ADE20K, we explore the following hyperparameters: learning rate {8e-5, 1e-3}, weight decay {0.01, 0.03, 0.05}. For Mask-RCNN [^29] training on COCO, we explore hyperparameters such as learning rate {1e-3, 2e-3, 3e-3}, weight decay {0.05, 0.1}, stochastic depth {0.1, 0.2}. For Cascade Mask-RCNN [^9] training on COCO, we explore hyperparameters such as learning rate {8e-5, 1e-4}, weight decay {0.05, 0.1}, stochastic depth {0.4, 0.5, 0.6}, and layer-wise learning rate decay {0.7, 0.8}. We note that our search space is similar to or less extensive than the known search space in ConvNeXt [^55] (*e.g*., 6 (ours) vs. 12 (ConvNeXt) for ADE20K (UperNet) and 8/24 (ours) vs. 48 (ConvNeXt) for COCO (Cascade Mask-RCNN) experiments).

### 0.A.3 Benchmark Setup

We measure latency and memory on the V100 GPU utilizing PyTorch 1.13.1 and CUDA 11.6. In all measurements, we employ the channels-last memory format [^58]. Memory is measured in the training phase with a batch size of 16.

## Appendix 0.B Model Configurations of RDNet

From a practitioner’s perspective, it can be challenging to ascertain the number of channels of features of RDNet. Therefore, we provide a detailed model configuration in Table B. The values of resolution and channel are based on the output feature. We ensured that the output dimensions of the transition layers are multiples of 8 for efficiency.

Table B: Model configurations of RDNet. The table on the left presents the layer-wise configuration of RDNet-T. The table on the right details the configuration of the RDNet model family. As described in Fig. 1, each stage of the RDNet comprises $L_{N}$ mixing blocks.

<table><tbody><tr><td>Stage</td><td>GR</td><td>Layer</td><td>Resolution</td><td>#Channels</td></tr><tr><td></td><td></td><td>Patchification</td><td>56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 56</td><td>64</td></tr><tr><td rowspan="3">S1</td><td rowspan="3">64</td><td>Feature mixer</td><td>56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 56</td><td>128</td></tr><tr><td>Feature mixer</td><td>56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 56</td><td>192</td></tr><tr><td>Feature mixer</td><td>56 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 56</td><td>256</td></tr><tr><td></td><td></td><td>Transition S2</td><td>28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 28</td><td>128</td></tr><tr><td rowspan="3">S2</td><td rowspan="3">104</td><td>Feature mixer</td><td>28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 28</td><td>232</td></tr><tr><td>Feature mixer</td><td>28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 28</td><td>336</td></tr><tr><td>Feature mixer</td><td>28 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 28</td><td>440</td></tr><tr><td></td><td></td><td>Transition S2</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>216</td></tr><tr><td rowspan="15">S3</td><td rowspan="15">128</td><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>344</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>472</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>600</td></tr><tr><td>Transition S1</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>296</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>424</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>552</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>680</td></tr><tr><td>Transition S1</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>336</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>464</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>592</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>720</td></tr><tr><td>Transition S1</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>360</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>488</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>616</td></tr><tr><td>Feature mixer</td><td>14 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 14</td><td>744</td></tr><tr><td></td><td></td><td>Transition S2</td><td>7 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 7</td><td>368</td></tr><tr><td rowspan="3">S4</td><td rowspan="3">224</td><td>Feature mixer</td><td>7 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 7</td><td>592</td></tr><tr><td>Feature mixer</td><td>7 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 7</td><td>816</td></tr><tr><td>Feature mixer</td><td>7 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 7</td><td>1040</td></tr><tr><td></td><td></td><td>Classifier</td><td>1 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 1</td><td>1000</td></tr></tbody></table>

<table><tbody><tr><td>Stage</td><td>Layer</td><td colspan="4">RDNet</td></tr><tr><td></td><td>Settings</td><td>Tiny</td><td>Small</td><td>Base</td><td>Large</td></tr><tr><td rowspan="2">S1</td><td>Growth Rate</td><td>64</td><td>64</td><td>96</td><td>128</td></tr><tr><td># Mixing Block</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td colspan="6">Transition S/2</td></tr><tr><td rowspan="2">S2</td><td>Growth Rate</td><td>104</td><td>128</td><td>128</td><td>192</td></tr><tr><td># Mixing Block</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td colspan="6">Transition S/2</td></tr><tr><td rowspan="2">S3</td><td>Growth Rate</td><td>128</td><td>128</td><td>168</td><td>256</td></tr><tr><td># Mixing Block</td><td>4</td><td>7</td><td>7</td><td>8</td></tr><tr><td colspan="6">Transition S/2</td></tr><tr><td rowspan="2">S4</td><td>Growth Rate</td><td>192</td><td>240</td><td>336</td><td>360</td></tr><tr><td># Mixing Block</td><td>1</td><td>2</td><td>2</td><td>2</td></tr><tr><td colspan="2">Classifier</td><td colspan="4">GAP, Linear</td></tr><tr><td colspan="2">Parameters (M)</td><td>24</td><td>50</td><td>87</td><td>186</td></tr><tr><td colspan="2">FLOPs (G)</td><td>5.0</td><td>8.7</td><td>15.4</td><td>34.7</td></tr></tbody></table>

Figure B: Further trade-offs in ImageNet-1K performance. We provide comparative visualizations with diverse environments (including A100, TesorRT, and CPU) between state-of-the-art models, which were known for top-performing models. It turns out that RDNet is highly competitive in practice in terms of model speed.

(a) PyTorch (A100 GPU)

(b) TensorRT (A100 GPU)

(c) CPU (Xeon Gold 5120)

## Appendix 0.C More ImageNet Accuracy vs. Latency Trade-offs

To assess the practicality of our models, we measure speeds across diverse testbeds. We precisely measure inference speeds using the PyTorch framework on NVIDIA A100 GPU, Intel Xeon Gold 5120 CPU, and TensorRT Inference Engine on NVIDIA A100 GPU. Our testing environment incorporates PyTorch version 1.13.1, CUDA version 11.6, and TensorRT version 8.5.3 for these experiments. Fig. B shows that our models consistently show superior accuracy vs. latency trade-offs across all evaluation setups. Note that (b) in Fig. B includes fewer models because those that are challenging to convert to inference engines due to factors like the use of CUDA custom kernels are not evaluated. We report all the numbers in Table C.

Table C: ImageNet-1K comparison with the latest models. Fig. B visualized this table. We thoroughly compare our models against the latest architectures in practical latencies. b $n$ denotes latency, measured with a batch size of $n$. Certain models were excluded from evaluation because their CUDA custom kernels were not compiled.

<table><tbody><tr><td>Model</td><td>Date</td><td>Param</td><td>FLOPs</td><td>Top-1</td><td colspan="4">PyTorch (A100, ms)</td><td colspan="4">TensorRT (A100, ms)</td><td colspan="4">PyTorch (Xeon 5120, s)</td></tr><tr><td></td><td></td><td>(M)</td><td>(G)</td><td>(%)</td><td>b1</td><td>b8</td><td>b32</td><td>b128</td><td>b1</td><td>b8</td><td>b32</td><td>b128</td><td>b1</td><td>b8</td><td>b32</td><td>b128</td></tr><tr><td>RDNet-T</td><td>Ours</td><td>24</td><td>5.0</td><td>82.8</td><td>9.2</td><td>9.2</td><td>17.8</td><td>60.5</td><td>3.9</td><td>7.5</td><td>19.7</td><td>69.3</td><td>0.07</td><td>0.25</td><td>1.09</td><td>4.85</td></tr><tr><td>HorNet-T <sub>7×7</sub> <sup><a href="#fn:65">65</a></sup></td><td>NeurIPS’2022</td><td>22</td><td>4.0</td><td>82.8</td><td>21.9</td><td>21.9</td><td>27.8</td><td>100.7</td><td>7.4</td><td>10.6</td><td>22.8</td><td>68.4</td><td>0.09</td><td>0.21</td><td>0.89</td><td>5.79</td></tr><tr><td>VAN-B2 <sup><a href="#fn:23">23</a></sup></td><td>CVMJ’2023</td><td>27</td><td>5.0</td><td>82.8</td><td>15.8</td><td>16.4</td><td>32.9</td><td>122.5</td><td>4.1</td><td>8.6</td><td>21.2</td><td>71.8</td><td>0.08</td><td>0.38</td><td>1.61</td><td>7.33</td></tr><tr><td>BiFormer-S <sup><a href="#fn:105">105</a></sup></td><td>CVPR’2023</td><td>26</td><td>4.5</td><td>83.8</td><td>31.0</td><td>35.3</td><td>54.2</td><td>200.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.13</td><td>0.39</td><td>2.17</td><td>13.60</td></tr><tr><td>NAT-T <sup><a href="#fn:27">27</a></sup></td><td>CVPR’2023</td><td>28</td><td>4.3</td><td>83.2</td><td>14.7</td><td>14.7</td><td>32.9</td><td>122.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.22</td><td>1.43</td><td>5.73</td><td>24.42</td></tr><tr><td>SMT-S <sup><a href="#fn:52">52</a></sup></td><td>ICCV’2023</td><td>21</td><td>4.7</td><td>83.7</td><td>26.1</td><td>26.4</td><td>53.0</td><td>191.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.11</td><td>0.32</td><td>1.16</td><td>7.91</td></tr><tr><td>MogaNet-S <sup><a href="#fn:49">49</a></sup></td><td>ICLR’2024</td><td>25</td><td>5.0</td><td>83.4</td><td>22.7</td><td>22.7</td><td>34.9</td><td>127.2</td><td>5.6</td><td>10.3</td><td>27.6</td><td>91.0</td><td>0.11</td><td>0.42</td><td>1.97</td><td>9.46</td></tr><tr><td>RDNet-S</td><td>Ours</td><td>50</td><td>8.7</td><td>83.7</td><td>14.3</td><td>14.4</td><td>26.4</td><td>88.3</td><td>5.7</td><td>10.8</td><td>29.3</td><td>99.4</td><td>0.11</td><td>0.38</td><td>1.73</td><td>7.34</td></tr><tr><td>HorNet-S <sub>7×7</sub> <sup><a href="#fn:65">65</a></sup></td><td>NeurIPS’2022</td><td>50</td><td>8.8</td><td>84.0</td><td>21.8</td><td>22.2</td><td>46.9</td><td>173.9</td><td>8.0</td><td>14.1</td><td>33.6</td><td>104.5</td><td>0.11</td><td>0.38</td><td>2.43</td><td>11.51</td></tr><tr><td>VAN-B3 <sup><a href="#fn:23">23</a></sup></td><td>CVMJ’2023</td><td>45</td><td>9.0</td><td>83.9</td><td>27.5</td><td>32.8</td><td>50.4</td><td>194.8</td><td>6.8</td><td>13.7</td><td>34.5</td><td>119.9</td><td>0.15</td><td>0.59</td><td>2.50</td><td>11.27</td></tr><tr><td>BiFormer-B <sup><a href="#fn:105">105</a></sup></td><td>CVPR’2023</td><td>57</td><td>9.8</td><td>84.3</td><td>31.2</td><td>31.1</td><td>89.3</td><td>336.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.15</td><td>0.78</td><td>4.90</td><td>23.03</td></tr><tr><td>NAT-S <sup><a href="#fn:27">27</a></sup></td><td>CVPR’2023</td><td>51</td><td>7.8</td><td>83.7</td><td>15.8</td><td>20.6</td><td>51.9</td><td>194.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.31</td><td>2.10</td><td>8.78</td><td>38.27</td></tr><tr><td>SMT-B <sup><a href="#fn:52">52</a></sup></td><td>ICCV’2023</td><td>32</td><td>7.7</td><td>84.3</td><td>37.4</td><td>39.1</td><td>81.2</td><td>295.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.19</td><td>0.60</td><td>2.24</td><td>13.41</td></tr><tr><td>MogaNet-B <sup><a href="#fn:49">49</a></sup></td><td>ICLR’2024</td><td>44</td><td>9.9</td><td>84.3</td><td>44.0</td><td>47.4</td><td>70.6</td><td>258.0</td><td>9.4</td><td>19.0</td><td>53.4</td><td>183.4</td><td>0.20</td><td>0.77</td><td>4.08</td><td>19.10</td></tr><tr><td>RDNet-B</td><td>Ours</td><td>87</td><td>15.4</td><td>84.4</td><td>14.9</td><td>15.1</td><td>36.0</td><td>124.2</td><td>6.2</td><td>13.6</td><td>39.5</td><td>139.8</td><td>0.15</td><td>0.58</td><td>2.52</td><td>10.84</td></tr><tr><td>HorNet-B <sub>7×7</sub> <sup><a href="#fn:65">65</a></sup></td><td>NeurIPS’2022</td><td>87</td><td>15.6</td><td>84.3</td><td>22.1</td><td>22.3</td><td>68.9</td><td>266.1</td><td>9.0</td><td>16.5</td><td>41.8</td><td>139.1</td><td>0.13</td><td>0.57</td><td>3.43</td><td>15.86</td></tr><tr><td>VAN-B4 <sup><a href="#fn:23">23</a></sup></td><td>CVMJ’2023</td><td>60</td><td>12.2</td><td>84.2</td><td>53.3</td><td>53.6</td><td>80.7</td><td>263.6</td><td>8.4</td><td>17.0</td><td>45.1</td><td>151.6</td><td>0.19</td><td>0.70</td><td>3.26</td><td>14.65</td></tr><tr><td>NAT-B <sup><a href="#fn:27">27</a></sup></td><td>CVPR’2023</td><td>90</td><td>13.7</td><td>84.3</td><td>15.9</td><td>22.5</td><td>88.6</td><td>296.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.42</td><td>3.03</td><td>12.33</td><td>52.47</td></tr><tr><td>MogaNet-L <sup><a href="#fn:49">49</a></sup></td><td>ICLR’2024</td><td>83</td><td>15.9</td><td>84.7</td><td>81.6</td><td>90.3</td><td>98.8</td><td>357.7</td><td>14.9</td><td>28.7</td><td>77.4</td><td>263.2</td><td>0.31</td><td>1.08</td><td>51.56</td><td>26.93</td></tr><tr><td>RDNet-L</td><td>Ours</td><td>186</td><td>34.7</td><td>84.8</td><td>17.0</td><td>17.3</td><td>59.8</td><td>216.1</td><td>8.1</td><td>20.4</td><td>62.9</td><td>231.8</td><td>0.26</td><td>1.15</td><td>4.72</td><td>19.13</td></tr><tr><td>MogaNet-XL <sup><a href="#fn:49">49</a></sup></td><td>ICLR’2024</td><td>181</td><td>34.5</td><td>85.1</td><td>82.3</td><td>93.3</td><td>146.3</td><td>549.5</td><td>17.4</td><td>38.6</td><td>115.4</td><td>421.9</td><td>0.40</td><td>2.57</td><td>10.27</td><td>49.49</td></tr></tbody></table>

## Appendix 0.D More Comparison with ResNets and ConvNeXts

We presented the performance of RDNets matched with ResNets and ConvNeXts in terms of the number of parameters and FLOPs in Table 8a. Here, we further report some RDNets’ performances, which are more closely aligned with ResNets and ConvNeXts than our previous models by 1) having identical depth; 2) having the same number of building blocks in each stage; 3) aligning the parameters and FLOPs as closely as possible with those of ResNet and ConvNext. Therefore, the comparison is more like an apples-to-apples comparison. As shown in Table D, despite RDNet not having the optimal width and depth for dense connections, RDNet demonstrates superior performance compared to the competing models.

Table D: Apples-to-apples comparison with ResNet and ConvNeXt. We align the parameters, FLOPs, and depth of ResNet and ConvNext as closely as possible to compare their top-1 accuracy (%).

| Model | Params | FLOPs | Depth | Top-1 |
| --- | --- | --- | --- | --- |
| ResNet-50 | 25.6M | 4.1G | \[3,4,6,3\] | 78.8 |
| RDNet | 25.5M | 4.1G | \[3,4,6,3\] | 82.1 |
| ResNet-152 | 60.2M | 11.5G | \[3,8,36,3\] | 80.8 |
| RDNet | 59.7M | 11.5G | \[3,8,36,3\] | 83.7 |
| ConvNeXt-T | 28.6M | 4.5G | \[3,3,9,3\] | 82.1 |
| RDNet | 27.1M | 4.5G | \[3,3,9,3\] | 82.4 |
| ConvNeXt-B | 88.6M | 15.4G | \[3,3,27,3\] | 83.8 |
| RDNet | 88.3M | 15.3G | \[3,3,27,3\] | 84.1 |

Table E: COCO object detection and segmentation results - Mask-RCNN 1x schedule. FLOPs (G) are calculated with image size (1280, 800).

| Backbone | Param | FLOPs | $\text{AP}^{\text{box}}$ | $\text{AP}^{\text{box}}_{50}$ | $\text{AP}^{\text{box}}_{75}$ | $\text{AP}^{\text{mask}}$ | $\text{AP}^{\text{mask}}_{\text{50}}$ | $\text{AP}^{\text{mask}}_{75}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PVT-S [^88] | 44M | 245G | 40.4 | 62.9 | 43.8 | 37.8 | 60.1 | 40.3 |
| Swin-T [^54] | 48M | 264G | 43.6 | 66.2 | 47.7 | 39.6 | 62.9 | 42.2 |
| PVTv2-B2 [^89] | 45M | 309G | 45.3 | 67.1 | 49.6 | 41.2 | 64.2 | 44.4 |
| ConvNeXt-T [^55] | 48M | 262G | 43.5 | 65.6 | 48.0 | 39.7 | 62.5 | 42.7 |
| CSwin-T [^20] | 42M | 279G | 46.7 | 68.6 | 51.3 | 42.2 | 65.6 | 45.4 |
| FocalNet-S (SRF) [^96] | 49M | 267G | 45.9 | 68.3 | 50.1 | 41.3 | 65.0 | 44.3 |
| RDNet-T | 43M | 278G | 46.1 | 68.0 | 50.8 | 41.4 | 65.1 | 44.2 |
| PVT-M [^88] | 64M | 302G | 42.0 | 64.4 | 45.6 | 39.0 | 61.6 | 42.1 |
| Swin-S [^54] | 69M | 354G | 46.5 | 68.7 | 51.3 | 42.1 | 65.8 | 45.2 |
| PVTv2-B3 [^89] | 65M | 397G | 47.0 | 68.1 | 51.7 | 42.5 | 65.7 | 45.7 |
| ConvNeXt-S [^55] | 70M | 348G | 46.8 | 69.0 | 51.5 | 42.1 | 65.8 | 45.2 |
| CSwin-S [^20] | 54M | 342G | 47.9 | 70.1 | 52.6 | 43.2 | 67.1 | 46.2 |
| FocalNet-S (SRF) [^96] | 71M | 356G | 48.0 | 69.9 | 52.7 | 42.7 | 66.7 | 45.7 |
| RDNet-S | 70M | 354G | 48.2 | 69.9 | 53.0 | 43.0 | 66.9 | 46.3 |
| Swin-B [^54] | 107M | 496G | 46.9 | 69.2 | 51.6 | 42.3 | 66.0 | 45.5 |
| PVTv2-B5 [^89] | 102M | 557G | 47.4 | 68.6 | 51.9 | 42.5 | 65.7 | 46.0 |
| ConvNeXt-B [^55] | 108M | 486G | 47.5 | 69.9 | 51.9 | 42.5 | 66.8 | 45.7 |
| CSwin-B [^20] | 97M | 526G | 48.7 | 70.4 | 53.9 | 43.9 | 67.8 | 47.3 |
| FocalNet-B (SRF) [^96] | 109M | 496G | 48.8 | 70.7 | 53.5 | 43.3 | 67.5 | 46.5 |
| RDNet-B | 107M | 493G | 48.8 | 70.4 | 53.5 | 43.4 | 67.5 | 46.6 |

Table F: COCO object detection and segmentation results - Cascade Mask-RCNN 3x schedule. FLOPs (G) are calculated with image size (1280, 800). The result of Swin-T is from the official repository [^1].

| Backbone | Param | FLOPs | $\text{AP}^{\text{box}}$ | $\text{AP}^{\text{box}}_{50}$ | $\text{AP}^{\text{box}}_{75}$ | $\text{AP}^{\text{mask}}$ | $\text{AP}^{\text{mask}}_{\text{50}}$ | $\text{AP}^{\text{mask}}_{75}$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-T [^54] | 86M | 745G | 50.4 | 69.2 | 54.7 | 43.7 | 66.6 | 47.3 |
| PVTv2-B2 [^89] | 83M | 788G | 51.1 | 69.8 | 55.3 | \- | \- | \- |
| FocalNet-T [^96] | 86M | 746G | 51.5 | 70.1 | 55.8 | \- | \- | \- |
| ConvNeXt-T [^55] | 86M | 741G | 50.4 | 69.1 | 54.8 | 43.7 | 66.5 | 47.3 |
| NAT-T [^27] | 85M | 737G | 51.4 | 70.0 | 55.9 | 44.5 | 67.6 | 47.9 |
| RDNet-T | 81M | 757G | 51.6 | 70.5 | 56.0 | 44.6 | 67.9 | 48.3 |
| Swin-S [^54] | 107M | 838G | 51.9 | 70.7 | 56.3 | 45.0 | 68.2 | 48.8 |
| ConvNeXt-S [^55] | 108M | 827G | 51.9 | 70.8 | 56.5 | 45.0 | 68.4 | 49.1 |
| NAT-S [^27] | 108M | 809G | 52.0 | 70.4 | 56.3 | 44.9 | 68.1 | 48.6 |
| RDNet-S | 108M | 832G | 52.3 | 70.8 | 56.6 | 45.4 | 68.5 | 49.3 |
| Swin-B [^54] | 145M | 982G | 51.9 | 70.5 | 56.4 | 45.0 | 68.1 | 48.9 |
| ConvNeXt-B [^55] | 146M | 964G | 52.7 | 71.3 | 57.2 | 45.6 | 68.9 | 49.5 |
| NAT-B [^27] | 147M | 931G | 52.5 | 71.1 | 57.1 | 45.2 | 68.6 | 49.0 |
| RDNet-B | 144M | 971G | 52.9 | 71.5 | 57.2 | 46.0 | 69.1 | 50.0 |

## Appendix 0.E More Object Detection/Instance Segmentation results

An extension to the Mask-RCNN [^29] with 3x schedule [^28] results in Table 6 in the main paper, we additionally train Mask-RCNN using a 1x schedule [^28] to perform a more comprehensive and fair comparison with a broader range of models. As shown in Table E, pre-trained RDNet models demonstrate competitive performance. Furthermore, we employ the Cascade Mask-RCNN head [^9] to evaluate our pre-trained models. As demonstrated in Table F, RDNet exhibits competitive performance. Note that our models do not experience exhaustive fine-tuning of training hyperparameters for maximum precisions compared with ConvNeXt’s, which indicates additional potential for achieving higher accuracy.

## Appendix 0.F Transferability Evaluation

We benchmark the transferability of ImageNet-1K pre-trained models. We utilize fine-grained classification datasets - CIFAR-10 [^44], CIFAR-100 [^44], Flowers-102 [^59], and Stanford-Cars [^43]. Furthermore, to verify the ability of long-tailed classification, we utilize iNaturalist-2018 [^34] and iNaturalist-2019 [^35]. We follow the training recipe of DeiT [^76]. As demonstrated in Table G, RDNet exhibits strong transferability. Notably, RDNet delivers impressive performance on long-tailed classification tasks. RDNet-T surpasses DeiT-III-L on iNaturalist-2018 and iNaturalist-2019.

Table G: Top-1 accuracy (%), parameter count (M), and FLOPs (G) on various classification tasks with ImageNet-1k pre-trained models.

| Model | Param | FLOPs | CIFAR10 | CIFAR100 | Flowers | Cars | iNat18 | iNat19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grafit ResNet-50 [^80] | 26 | 4.1 | \- | \- | 98.2 | 92.5 | 69.8 | 75.9 |
| DeiT-III-S [^78] | 22 | 4.6 | 98.9 | 90.6 | 96.4 | 89.9 | 67.1 | 72.7 |
| RDNet-T | 24 | 5.0 | 98.9 | 90.4 | 98.6 | 93.9 | 77.0 | 81.2 |
| ResNet-152 [^13] | 60 | 11.6 | \- | \- | \- | \- | 69.1 | \- |
| RDNet-S | 50 | 8.7 | 99.0 | 91.1 | 98.4 | 94.2 | 79.1 | 82.4 |
| ViT-B/16 [^21] | 86 | 35.1 | 98.1 | 87.1 | 89.5 | \- | \- | \- |
| ViT-B/16 [^71] | 86 | 35.1 | \- | 87.8 | 96.0 | \- | \- | \- |
| DeiT-B [^76] | 87 | 17.5 | 99.1 | 90.8 | 98.4 | 92.1 | 73.2 | 77.7 |
| DeiT-III-B [^78] | 87 | 17.5 | 99.3 | 92.5 | 98.6 | 93.4 | 73.6 | 78.0 |
| RDNet-B | 87 | 15.4 | 99.3 | 91.4 | 98.6 | 94.1 | 80.5 | 83.5 |
| RDNet-L | 186 | 34.7 | 99.3 | 91.5 | 98.6 | 94.2 | 81.5 | 83.7 |
| ViT-L/16 [^21] | 303 | 122.9 | 97.9 | 86.4 | 89.7 | \- | \- | \- |
| ViT-L/16 [^71] | 303 | 122.9 | \- | 86.2 | 91.4 | \- | \- | \- |
| DeiT-III-L [^78] | 304 | 61.6 | 99.3 | 93.4 | 98.9 | 94.5 | 75.6 | 79.3 |

Table H: WGAN experiments. We demonstrate the capability of dense connections in RDNet for generation tasks. We utilize the WGAN architecture, redesigning only the generator to showcase whether the generation ability could improve.

| Model | Param | IS↑ | FID↓ |
| --- | --- | --- | --- |
| ResBlock | 5.4M | 7.27±0.26 | 27.79±1.06 |
| Ours | 5.6M | 7.52±0.10 | 25.37±0.21 |

## Appendix 0.G Dense Connections for Generative Models

We test a generative model, WGAN [^2], by replacing ResNet-GAN in the generator with our concatenation-based model. We maintained identical discriminator architecture, training setups, and similar model sizes. We focus on a proof of concept on CIFAR-10 to showcase whether our models could outperform with faster convergence. Table H results confirmed that our model works across multiple runs. We plan to extend our research to larger models in future work.

Table I: Robustness evaluation. We compare the models evaluating the out-of-distribution (OOD) metrics ImageNet-V2/A/Sketch/ObjNet/R. We further average the OOD scores to show the averaged distribution shifts denoted by Avg Shift. Interestingly, even when RDNet demonstrates lower accuracy on ImageNet-1K compared to other networks, it consistently attains high OOD scores compared with other datasets.

| Model | Param | FLOPs | IN | Avg Shift | V2 | Obj | A | Sketch | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Swin-T [^54] | 28 | 4.5 | 81.3 | 38.9 | 69.7 | 33.1 | 21.1 | 29.3 | 41.5 |
| ConvNeXt-T [^55] | 29 | 4.5 | 82.1 | 42.7 | 72.5 | 35.6 | 24.2 | 33.8 | 47.2 |
| HorNet-T [^65] | 22 | 4.0 | 82.8 | 43.4 | 72.3 | 37.5 | 26.6 | 34.1 | 46.6 |
| SLaK-T [^53] | 30 | 5.0 | 82.5 | 43.3 | 72.0 | 36.6 | 30.0 | 32.4 | 45.3 |
| NAT-T [^27] | 28 | 4.3 | 83.2 | 44.0 | 72.2 | 37.8 | 33.0 | 31.9 | 44.9 |
| RDNet-T | 24 | 5.0 | 82.8 | 44.7 | 72.9 | 36.9 | 27.7 | 37.0 | 49.0 |
| Swin-S [^54] | 50 | 8.7 | 83.0 | 43.8 | 72.0 | 36.8 | 32.5 | 32.3 | 45.2 |
| ConvNeXt-S [^55] | 50 | 8.7 | 83.1 | 45.7 | 72.5 | 38.0 | 31.3 | 37.1 | 49.6 |
| HorNet-S [^65] | 50 | 8.8 | 84.0 | 47.3 | 73.6 | 39.9 | 36.2 | 36.9 | 49.7 |
| SLaK-S [^53] | 55 | 9.8 | 83.8 | 48.2 | 73.6 | 39.6 | 39.3 | 37.5 | 50.9 |
| NAT-S [^27] | 51 | 7.8 | 83.7 | 46.4 | 73.2 | 39.9 | 37.4 | 34.3 | 47.3 |
| RDNet-S | 50 | 8.7 | 83.7 | 47.8 | 73.8 | 39.3 | 33.5 | 39.8 | 52.8 |
| Swin-B [^54] | 88 | 15.4 | 83.5 | 44.9 | 72.4 | 37.6 | 35.4 | 32.7 | 46.5 |
| ConvNeXt-B [^55] | 89 | 15.4 | 83.8 | 47.9 | 73.7 | 39.9 | 36.7 | 38.2 | 51.2 |
| HorNet-B [^65] | 87 | 15.6 | 84.3 | 48.8 | 73.9 | 41.0 | 39.9 | 38.1 | 51.2 |
| SLaK-B [^53] | 95 | 17.1 | 84.0 | 48.9 | 74.0 | 39.7 | 41.6 | 38.5 | 50.8 |
| NAT-B [^27] | 90 | 13.7 | 84.3 | 48.5 | 74.1 | 40.7 | 41.4 | 36.6 | 49.7 |
| RDNet-B | 87 | 15.4 | 84.4 | 49.0 | 74.2 | 39.7 | 38.1 | 40.1 | 52.7 |
| ConvNeXt-L [^55] | 198 | 34.4 | 84.3 | 49.9 | 74.2 | 40.6 | 41.3 | 40.1 | 53.5 |
| RDNet-L | 186 | 34.7 | 84.8 | 52.2 | 75.0 | 42.1 | 42.9 | 44.5 | 56.5 |

## Appendix 0.H Robustness Evaluation

We further evaluate the robustness of our models using the ImageNet out-of-distribution (OOD) benchmarks - ImageNet-V2 [^66], ImageNet-A [^33], ImageNet-Sketch [^85], ImageNet-R [^32], and ObjectNet [^5]. Table I shows our RDNet demonstrates superior robustness in comparison to the other models. We specifically select models (HorNet, SLaK, and NAT) having comparable ImageNet-1K accuracies to demonstrate the superior out-of-distribution (OOD) performance of our models compared with those. Nobaly, even when RDNet demonstrates lower accuracy on ImageNet-1K than competing models, RDNet achieves high OOD scores across various benchmarks.

## Appendix 0.I More Details and Extended Pilot Study

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.19588/assets/new_randnet_cum_basic_merged.png)

Figure C: Cumulative prabability vs. error of trained models in Table J is visualized here following Radosavovic et al. 64. Each row demonstrates three block types, and each column exhibits parameter spaces 𝒜 \\mathcal{A}, ℬ \\mathcal{B}, and 𝒞 \\mathcal{C}

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2403.19588/assets/new_randnet_cum_aug_merged.png)

Figure D: Cumulative prabability vs. error of trained models in Table K is visualized here following Radosavovic et al. 64. The figure corresponds to the manuscript’s parameter spaces 𝒟 \\mathcal{D} and ℰ \\mathcal{E}. Each row demonstrates optimizers, and each column exhibits augmentations.

### 0.I.1 More Details of Pilot Study

We present individual RandNet experimental results performed under controlled setups. We conduct 200 experiments for each configuration of budget, block type, and skip connection type. Fig. C and Table J, concatenation outperforms addition across parameter spaces $\mathcal{A}$, $\mathcal{B}$, and $\mathcal{C}$. For the parameter spaces $\mathcal{D}$ and $\mathcal{E}$ using data augmentations, we use the following hyper-parameters for experiments. For RandAugment [^15], we limit the magnitude values of {3, 5, 7}; for MixUp [^101] and CutMix [^100], we employ alpha values of {0.1, 0.3, 0.5} respectively; Stochastic Depth [^39] ratio are set to {0.05, 0.1}; Random Erasing [^103] is fixed to 0.25.

Due to the diverse setups in each data argumentation, we conduct 600 experiments for each element. Additionally, even when switching the optimizer to AdamW, we train 600 random networks for every augmentation as well. Fig. D and Table K suggest that that data augmentation reduces the performance gap between additive and concatenative shortcuts, particularly noting that the element (*i.e*., trained using stochastic depth with AdamW) benefit more from an additive shortcut. Nevertheless, we continue to observe that concatenation-based models dominate over advantage over additive shortcuts, even in the AdamW training configurations.

Table J: Concatenation vs. addition. The table corresponds to the manuscript’s parameter spaces $\mathcal{A}$, $\mathcal{B}$, and $\mathcal{C}$. We sample 200 random networks within each parameter space, ensuring similar computational costs of FLOPs, the number of parameters (Param), and memory consumption (Mem), and individually train them on Tiny-ImageNet. All results are averaged along with the standard deviation (± std). Experimental results with higher accuracy are shaded in gray.

| Model | Skip type | Block type | FLOPs (G) | Param (M) | Mem (GB) | Top-1 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| RandNet <sub>A</sub> | Add | PostNorm | 2.24±0.14 | 2.20±0.13 | 0.66±0.03 | 45.6±2.3 |
| RandNet <sub>A</sub> | Concat | PostNorm | 2.24±0.14 | 2.25±0.14 | 0.71±0.05 | 46.9±2.2 |
| RandNet <sub>A</sub> | Add | PostNorm (w/o act) | 2.24±0.13 | 2.20±0.13 | 0.66±0.02 | 45.1±2.1 |
| RandNet <sub>A</sub> | Concat | PostNorm (w/o act) | 2.23±0.14 | 2.24±0.14 | 0.76±0.06 | 46.8±2.2 |
| RandNet <sub>A</sub> | Add | PreNorm | 2.24±0.13 | 2.2±0.13 | 0.66±0.02 | 46.8±1.67 |
| RandNet <sub>A</sub> | Concat | PreNorm | 2.25±0.14 | 2.25±0.14 | 0.82±0.08 | 48.7±1.6 |
| RandNet <sub>B</sub> | Add | PostNorm | 4.51±0.27 | 4.42±0.26 | 0.78±0.05 | 49.9±2.2 |
| RandNet <sub>B</sub> | Concat | PostNorm | 4.53±0.28 | 4.52±0.28 | 0.88±0.09 | 50.7±2.2 |
| RandNet <sub>B</sub> | Add | PostNorm (w/o act) | 4.52±0.27 | 4.43±0.26 | 0.78±0.05 | 49.4±2.3 |
| RandNet <sub>B</sub> | Concat | PostNorm (w/o act) | 4.51±0.28 | 4.50±0.28 | 0.84±0.08 | 50.7±2.1 |
| RandNet <sub>B</sub> | Add | PreNorm | 4.51±0.26 | 4.42±0.26 | 0.78±0.05 | 51.1±1.6 |
| RandNet <sub>B</sub> | Concat | PreNorm | 4.50±0.30 | 4.48±0.29 | 0.97±0.12 | 52.2±1.5 |
| RandNet <sub>C</sub> | Add | PostNorm | 9.58±0.23 | 9.37±0.23 | 1.00±0.09 | 52.8±2.2 |
| RandNet <sub>C</sub> | Concat | PostNorm | 9.55±0.27 | 9.46±0.27 | 1.05±0.11 | 53.8±2.1 |
| RandNet <sub>C</sub> | Add | PostNorm (w/o act) | 9.60±0.24 | 9.39±0.23 | 1.02±0.09 | 52.5±2.2 |
| RandNet <sub>C</sub> | Concat | PostNorm (w/o act) | 9.56±0.27 | 9.46±0.26 | 1.11±0.13 | 53.9±2.1 |
| RandNet <sub>C</sub> | Add | PreNorm | 9.58±0.22 | 9.37±0.21 | 1.02±0.10 | 54.4±1.6 |
| RandNet <sub>C</sub> | Concat | PreNorm | 9.56±0.26 | 9.46±0.25 | 1.24±0.17 | 55.1±1.3 |

Table K: Concatenation vs. addition with data augmentations. The table corresponds to the manuscript’s parameter spaces $\mathcal{D}$ and $\mathcal{E}$. We utilize the PreNorm block for augmentation experiments. We sample 600 random networks due to diverse degrees of data augmentations and report identically in Table J. Experimental results with higher accuracy are shaded in gray.

| Model | Skip type | Augmentation | AdamW | FLOPs (G) | Param (M) | Mem (GB) | Top-1 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RandNet <sub>D</sub> | Add | RandAug |  | 9.60±0.23 | 9.40±0.23 | 1.02±0.09 | 57.8±1.3 |
| RandNet <sub>D</sub> | Concat | RandAug |  | 9.54±0.26 | 9.44±0.26 | 1.25±0.17 | 58.7±1.2 |
| RandNet <sub>D</sub> | Add | MixUp |  | 9.59±0.23 | 9.39±0.22 | 1.02±0.09 | 57.3±1.4 |
| RandNet <sub>D</sub> | Concat | MixUp |  | 9.55±0.26 | 9.45±0.26 | 1.24±0.16 | 57.9±1.2 |
| RandNet <sub>D</sub> | Add | CutMix |  | 9.60±0.23 | 9.39±0.23 | 1.03±0.09 | 57.0±1.7 |
| RandNet <sub>D</sub> | Concat | CutMix |  | 9.55±0.27 | 9.45±0.26 | 1.23±0.16 | 57.5±1.5 |
| RandNet <sub>D</sub> | Add | Sto. Depth |  | 9.59±0.23 | 9.39±0.23 | 1.02±0.09 | 57.3±1.4 |
| RandNet <sub>D</sub> | Concat | Sto. Depth |  | 9.55±0.27 | 9.45±0.26 | 1.24±0.17 | 57.8±1.2 |
| RandNet <sub>D</sub> | Add | RandErase |  | 9.59±0.23 | 9.38±0.23 | 1.02±0.09 | 57.8±1.3 |
| RandNet <sub>D</sub> | Concat | RandErase |  | 9.56±0.26 | 9.45±0.26 | 1.24±0.16 | 58.2±1.2 |
| RandNet <sub>E</sub> | Add | RandAug | ✓ | 9.60±0.23 | 9.40±0.23 | 1.02±0.09 | 58.5±1.4 |
| RandNet <sub>E</sub> | Concat | RandAug | ✓ | 9.54±0.26 | 9.44±0.26 | 1.25±0.17 | 59.2±1.3 |
| RandNet <sub>E</sub> | Add | MixUp | ✓ | 9.60±0.23 | 9.39±0.22 | 1.02±0.09 | 58.2±1.3 |
| RandNet <sub>E</sub> | Concat | MixUp | ✓ | 9.55±0.25 | 9.44±0.25 | 1.24±0.16 | 58.9±1.3 |
| RandNet <sub>E</sub> | Add | CutMix | ✓ | 9.59±0.23 | 9.38±0.23 | 1.02±0.09 | 58.9±1.6 |
| RandNet <sub>E</sub> | Concat | CutMix | ✓ | 9.54±0.26 | 9.44±0.26 | 1.25±0.17 | 60.2±1.4 |
| RandNet <sub>E</sub> | Add | Sto. Depth | ✓ | 9.60±0.23 | 9.39±0.23 | 1.02±0.09 | 57.5±1.3 |
| RandNet <sub>E</sub> | Concat | Sto. Depth | ✓ | 9.55±0.26 | 9.44±0.26 | 1.25±0.16 | 57.5±1.1 |
| RandNet <sub>E</sub> | Add | RandErase | ✓ | 9.58±0.23 | 9.37±0.23 | 1.02±0.08 | 58.1±1.1 |
| RandNet <sub>E</sub> | Concat | RandErase | ✓ | 9.56±0.26 | 9.46±0.26 | 1.24±0.18 | 58.1±1.0 |

### 0.I.2 Scaled-up Pilot Study

To further extend our pilot study, we conduct ImageNet-1K experiments across the parameter spaces $\mathcal{C}$, $\mathcal{D}$, and $\mathcal{E}$, as described in §5.1. For the parameter space $\mathcal{C}$, we train the model for 60 epochs. For $\mathcal{D}$ and $\mathcal{E}$, we extend the training epochs to 90 to assess the impact of augmentation. In the parameter space $\mathcal{C}$, we carry out 150 experiments for each type of skip connection. In $\mathcal{D}$ and $\mathcal{E}$, we conduct 60 experiments for each combination of augmentation and skip connection type. As shown in Table L, even on the ImageNet-1K dataset, the trained models using concatenation statistically surpass those using addition.

Table L: Scaled-up pilot study on ImageNet-1K - concatenation vs. addition experiments. The table corresponds to our extended pilot study on the same parameter spaces $\mathcal{C}$, $\mathcal{D}$ and $\mathcal{E}$. Each row in parameter space $\mathcal{C}$ contains statistics derived from 150 experiments, while each row in parameter spaces $\mathcal{D}$ and $\mathcal{E}$ encompasses statistics from 300 experiments. We utilize the PreNorm block only for scaled-up experiments. Experimental results with higher accuracy are shaded in gray.

| Model | Skip type | FLOPs (G) | Param (M) | Mem (GB) | Top-1 (%) |
| --- | --- | --- | --- | --- | --- |
| RandNet <sub>C</sub> | Add | 9.51±0.28 | 9.41±0.28 | 1.02±0.09 | 54.9±2.2 |
| RandNet <sub>C</sub> | Concat | 9.45±0.22 | 9.47±0.20 | 1.24±0.17 | 55.1±2.0 |
| RandNet <sub>D</sub> | Add | 9.60±0.24 | 9.46±0.20 | 1.02±0.09 | 54.7±2.6 |
| RandNet <sub>D</sub> | Concat | 9.55±0.26 | 9.45±0.26 | 1.25±0.16 | 55.1±2.7 |
| RandNet <sub>E</sub> | Add | 9.60±0.27 | 9.38±0.26 | 1.02±0.09 | 55.8±2.8 |
| RandNet <sub>E</sub> | Concat | 9.55±0.27 | 9.44±0.26 | 1.25±0.16 | 56.7±2.7 |

[^1]: Github repository: Swin transformer for object detection, [https://github.com/SwinTransformer/Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection)

[^2]: Arjovsky, M., Chintala, S., Bottou, L.: Wasserstein generative adversarial networks. In: International Conference on Machine Learning (ICML) (2017)

[^3]: Ba, J., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016)

[^4]: Bao, H., Dong, L., Wei, F.: Beit: Bert pre-training of image transformers. In: International Conference on Learning Representations (ICLR) (2022)

[^5]: Barbu, A., Mayo, D., Alverio, J., Luo, W., Wang, C., Gutfreund, D., Tenenbaum, J., Katz, B.: Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. In: Wallach, H., Larochelle, H., Beygelzimer, A., d'Alché-Buc, F., Fox, E., Garnett, R. (eds.) Conference on Neural Information Processing Systems (NeurIPS). vol. 32 (2019)

[^6]: Bello, I., Fedus, W., Du, X., Cubuk, E.D., Srinivas, A., Lin, T.Y., Shlens, J., Zoph, B.: Revisiting resnets: Improved training and scaling strategies. Conference on Neural Information Processing Systems (NeurIPS) 34, 22614–22627 (2021)

[^7]: Brock, A., De, S., Smith, S.L., Simonyan, K.: High-performance large-scale image recognition without normalization. In: International Conference on Machine Learning (ICML). pp. 1059–1071 (2021)

[^8]: Cai, Y., Zhou, Y., Han, Q., Sun, J., Kong, X., Li, J., Zhang, X.: Reversible column networks. In: International Conference on Learning Representations (ICLR) (2023)

[^9]: Cai, Z., Vasconcelos, N.: Cascade r-cnn: High-quality object detection and instance segmentation. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) (2019)

[^10]: Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 3558–3568 (2021)

[^11]: Chen, Y., Li, J., Xiao, H., Jin, X., Yan, S., Feng, J.: Dual path networks. Conference on Neural Information Processing Systems (NIPS) 30 (2017)

[^12]: Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., Jitsev, J.: Reproducible scaling laws for contrastive language-image learning. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 2818–2829 (2023)

[^13]: Chu, P., Bian, X., Liu, S., Ling, H.: Feature space augmentation for long-tailed data. arXiv preprint arXiv:2008.03673 (2020)

[^14]: Clark, K., Luong, M.T., Le, Q.V., Manning, C.D.: Electra: Pre-training text encoders as discriminators rather than generators. In: International Conference on Learning Representations (ICLR) (2020)

[^15]: Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: Randaugment: Practical automated data augmentation with a reduced search space. In: IEEE Transactions on Computer Vision and Pattern Recognition Workshop (CVPRW). pp. 702–703 (2020)

[^16]: Dai, Z., Liu, H., Le, Q.V., Tan, M.: Coatnet: Marrying convolution and attention for all data sizes. In: Conference on Neural Information Processing Systems (NeurIPS). pp. 3965–3977 (2021)

[^17]: Desai, K., Kaul, G., Aysola, Z., Johnson, J.: Redcaps: Web-curated image-text data created by the people, for the people. arXiv preprint arXiv:2111.11431 (2021)

[^18]: DeVries, T., Taylor, G.W.: Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552 (2017)

[^19]: Ding, X., Zhang, X., Zhou, Y., Han, J., Ding, G., Sun, J.: Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) (2022)

[^20]: Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., Guo, B.: Cswin transformer: A general vision transformer backbone with cross-shaped windows. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) (2022)

[^21]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (ICLR) (2020)

[^22]: Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., He, K.: Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677 (2017)

[^23]: Guo, M.H., Lu, C.Z., Liu, Z.N., Cheng, M.M., Hu, S.M.: Visual attention network. Computational Visual Media (CVMJ) (2023)

[^24]: Han, D., Kim, J., Kim, J.: Deep pyramidal residual networks. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 5927–5935 (2017)

[^25]: Han, D., Yoo, Y., Kim, B., Heo, B.: Learning features with parameter-free layers. arXiv preprint arXiv:2202.02777 (2022)

[^26]: Han, D., Yun, S., Heo, B., Yoo, Y.: Rethinking channel dimensions for efficient model design. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 732–741 (2021)

[^27]: Hassani, A., Walton, S., Li, J., Li, S., Shi, H.: Neighborhood attention transformer. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 6185–6194 (2023)

[^28]: He, K., Girshick, R., Dollár, P.: Rethinking imagenet pre-training. In: International Conference on Computer Vision (ICCV). pp. 4918–4927 (2019)

[^29]: He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: International Conference on Computer Vision (ICCV) (2017)

[^30]: He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016)

[^31]: He, K., Zhang, X., Ren, S., Sun, J.: Identity mappings in deep residual networks. In: ECCV. pp. 630–645. Springer (2016)

[^32]: Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., Song, D., Steinhardt, J., Gilmer, J.: The many faces of robustness: A critical analysis of out-of-distribution generalization. International Conference on Computer Vision (ICCV) (2021)

[^33]: Hendrycks, D., Zhao, K., Basart, S., Steinhardt, J., Song, D.: Natural adversarial examples. IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) (2021)

[^34]: Horn, G.V., Mac Aodha, O., Song, Y., Shepard, A., Adam, H., Perona, P., Belongie, S.J.: The inaturalist challenge 2018 dataset. arXiv preprint arXiv:1707.06642 (2018)

[^35]: Horn, G.V., Mac Aodha, O., Song, Y., Shepard, A., Adam, H., Perona, P., Belongie, S.J.: The iNaturalist species classification and detection dataset. arXiv preprint arXiv:1707.06642 (2019)

[^36]: Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)

[^37]: Huang, G., Liu, Z., Pleiss, G., Maaten, L.v.d., Weinberger, K.Q.: Convolutional networks with dense connectivity. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) 44(12), 8704–8716 (2022)

[^38]: Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: ECCV. pp. 646–661. Springer (2016)

[^39]: Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K.Q.: Deep networks with stochastic depth. In: European Conference on Computer Vision (ECCV) (2016)

[^40]: Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International Conference on Machine Learning (ICML). pp. 448–456. PMLR (2015)

[^41]: Jégou, S., Drozdzal, M., Vazquez, D., Romero, A., Bengio, Y.: The one hundred layers tiramisu: Fully convolutional densenets for semantic segmentation. In: IEEE Transactions on Computer Vision and Pattern Recognition Workshop (CVPRW). pp. 11–19 (2017)

[^42]: Kornblith, S., Norouzi, M., Lee, H., Hinton, G.: Similarity of neural network representations revisited. In: International Conference on Machine Learning (ICML). pp. 3519–3529 (2019)

[^43]: Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for fine-grained categorization. In: IEEE Workshop on 3D Representation and Recognition (2013)

[^44]: Krizhevsky, A.: Learning multiple layers of features from tiny images. Tech. rep., CIFAR (2009)

[^45]: Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Communications of the ACM 60, 84 – 90 (2012)

[^46]: Le, Y., Yang, X.S.: Tiny imagenet visual recognition challenge (2015), [https://api.semanticscholar.org/CorpusID:16664790](https://api.semanticscholar.org/CorpusID:16664790)

[^47]: Lee, Y., Hwang, J.w., Lee, S., Bae, Y., Park, J.: An energy and gpu-computation efficient backbone network for real-time object detection. In: IEEE Transactions on Computer Vision and Pattern Recognition Workshop (CVPRW). pp. 752–760 (2019)

[^48]: Lee, Y., Park, J.: Centermask: Real-time anchor-free instance segmentation. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) (2020)

[^49]: Li, S., Wang, Z., Liu, Z., Tan, C., Lin, H., Wu, D., Chen, Z., Zheng, J., Li, S.Z.: Moganet: Multi-order gated aggregation network. In: International Conference on Learning Representations (ICLR) (2024), [https://openreview.net/forum?id=XhYWgjqCrV](https://openreview.net/forum?id=XhYWgjqCrV)

[^50]: Li, Y., Chen, Y., Dai, X., Chen, D., Liu, M., Yuan, L., Liu, Z., Zhang, L., Vasconcelos, N.: Micronet: Improving image recognition with extremely low flops. In: International Conference on Computer Vision (ICCV). pp. 468–477 (2021)

[^51]: Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European Conference on Computer Vision (ECCV). pp. 740–755. Springer (2014)

[^52]: Lin, W., Wu, Z., Chen, J., Huang, J., Jin, L.: Scale-aware modulation meet transformer. In: International Conference on Computer Vision (ICCV). pp. 5992–6003 (10 2023)

[^53]: Liu, S., Chen, T., Chen, X., Chen, X., Xiao, Q., Wu, B., Pechenizkiy, M., Mocanu, D.C., Wang, Z.: More convnets in the 2020s: Scaling up kernels beyond 51x51 using sparsity. In: International Conference on Learning Representations (ICLR) (2023)

[^54]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: International Conference on Computer Vision (ICCV) (2021)

[^55]: Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 11976–11986 (2022)

[^56]: Loshchilov, I., Hutter, F.: Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983 (2016)

[^57]: Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR) (2019)

[^58]: Mingfei Ma, Vitaly Fedyunin, W.W.: Accelerating pytorch vision models with channels last on cpu (2022), [https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/](https://pytorch.org/blog/accelerating-pytorch-vision-models-with-channels-last-on-cpu/)

[^59]: Nilsback, M.E., Zisserman, A.: Automated flower classification over a large number of classes. In: Proceedings of the Indian Conference on Computer Vision, Graphics and Image Processing (2008)

[^60]: Parihar, A.S., Java, A.: Densely connected convolutional transformer for single image dehazing. Journal of Visual Communication and Image Representation p. 103722 (2023)

[^61]: Pleiss, G., Chen, D., Huang, G., Li, T., van der Maaten, L., Weinberger, K.Q.: Memory-efficient implementation of densenets. arXiv preprint arXiv:1707.06990 (2017)

[^62]: Polyak, B., Juditsky, A.B.: Acceleration of stochastic approximation by averaging. Siam Journal on Control and Optimization 30, 838–855 (1992)

[^63]: Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning (ICML). pp. 8748–8763. PMLR (2021)

[^64]: Radosavovic, I., Kosaraju, R.P., Girshick, R.B., He, K., Dollár, P.: Designing network design spaces. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 10425–10433 (2020)

[^65]: Rao, Y., Zhao, W., Tang, Y., Zhou, J., Lim, S.N., Lu, J.: Hornet: Efficient high-order spatial interactions with recursive gated convolutions. In: Conference on Neural Information Processing Systems (NeurIPS) (2022)

[^66]: Recht, B., Roelofs, R., Schmidt, L., Shankar, V.: Do imagenet classifiers generalize to imagenet? In: International Conference on Machine Learning (ICML) (2019)

[^67]: Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: Imagenet large scale visual recognition challenge. International Journal of Computer Vision (IJCV) 115(3), 211–252 (2015)

[^68]: Sandler, M., Howard, A.G., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 4510–4520 (2018)

[^69]: Sharma, P., Ding, N., Goodman, S., Soricut, R.: Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2556–2565 (2018)

[^70]: Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556 (2014)

[^71]: Steiner, A., Kolesnikov, A., Zhai, X., Wightman, R., Uszkoreit, J., Beyer, L.: How to train your vit? data, augmentation, and regularization in vision transformers. arXiv preprint arXiv:2106.10270 (2021)

[^72]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 1–9 (2015)

[^73]: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) pp. 2818–2826 (2016)

[^74]: Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: International Conference on Machine Learning (ICML). pp. 6105–6114. PMLR (2019)

[^75]: Tan, M., Le, Q.V.: Efficientnetv2: Smaller models and faster training. In: International Conference on Machine Learning (ICML) (2021)

[^76]: Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jegou, H.: Training data-efficient image transformers & distillation through attention. In: International Conference on Machine Learning (ICML). pp. 10347–10357 (2021)

[^77]: Touvron, H., Cord, M., El-Nouby, A., Bojanowski, P., Joulin, A., Synnaeve, G., Verbeek, J., J’egou, H.: Augmenting convolutional networks with attention-based aggregation. arXiv preprint arXiv:2112.13692 (2021)

[^78]: Touvron, H., Cord, M., J’egou, H.: Deit iii: Revenge of the vit. In: European Conference on Computer Vision (ECCV) (2022)

[^79]: Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., J’egou, H.: Going deeper with image transformers. International Conference on Computer Vision (ICCV) pp. 32–42 (2021)

[^80]: Touvron, H., Sablayrolles, A., Douze, M., Cord, M., Jégou, H.: Grafit: Learning fine-grained image representations with coarse labels. International Conference on Computer Vision (ICCV) (2021)

[^81]: Trockman, A., Kolter, J.Z.: Patches are all you need? arXiv preprint arXiv:2201.09792 (2022)

[^82]: Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A.C., Li, Y.: Maxvit: Multi-axis vision transformer. In: European Conference on Computer Vision (ECCV) (2022)

[^83]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: Conference on Neural Information Processing Systems (NIPS) (2017)

[^84]: Wang, C.Y., Liao, H.y., Wu, Y.H., Chen, P.Y., Hsieh, J.W., Yeh, I.H.: Cspnet: A new backbone that can enhance learning capability of cnn. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 1571–1580 (2020)

[^85]: Wang, H., Ge, S., Lipton, Z., Xing, E.P.: Learning robust global representations by penalizing local predictive power. In: Conference on Neural Information Processing Systems (NeurIPS). pp. 10506–10518 (2019)

[^86]: Wang, L., Cao, M., Yuan, X.: Efficientsci: Densely connected network with space-time factorization for large-scale video snapshot compressive imaging. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 18477–18486 (2023)

[^87]: Wang, R.J., Li, X., Ling, C.X.: Pelee: a real-time object detection system on mobile devices. In: Conference on Neural Information Processing Systems (NeurIPS). p. 1967–1976 (2018)

[^88]: Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: International Conference on Computer Vision (ICCV). pp. 548–558 (2021)

[^89]: Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pvtv2: Improved baselines with pyramid vision transformer. Computational Visual Media (CVMJ) (2022)

[^90]: Wang, Z., Xie, K., Zhang, X.Y., Chen, H.Q., Wen, C., He, J.: Small-object detection based on yolo and dense block via image super-resolution. IEEE Access 9, 56416–56429 (2021)

[^91]: Wightman, R.: Github repository: Pytorch image models, [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)

[^92]: Wightman, R., Touvron, H., Jégou, H.: Resnet strikes back: An improved training procedure in timm. [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) (2021)

[^93]: Wu, K., Zhang, J., Peng, H., Liu, M., Xiao, B., Fu, J., Yuan, L.: Tinyvit: Fast pretraining distillation for small vision transformers. In: European Conference on Computer Vision (ECCV) (2022)

[^94]: Xiao, T., Liu, Y., Zhou, B., Jiang, Y., Sun, J.: Unified perceptual parsing for scene understanding. In: European Conference on Computer Vision (ECCV). Springer (2018)

[^95]: Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 1492–1500 (2017)

[^96]: Yang, J., Li, C., Dai, X., Gao, J.: Focal modulation networks. In: Conference on Neural Information Processing Systems (NeurIPS) (2022)

[^97]: Yang, J., Li, C., Zhang, P., Dai, X., Xiao, B., Yuan, L., Gao, J.: Focal self-attention for local-global interactions in vision transformers. In: Conference on Neural Information Processing Systems (NeurIPS) (2021)

[^98]: Yu, W., Luo, M., Zhou, P., Si, C., Zhou, Y., Wang, X., Feng, J., Yan, S.: Metaformer is actually what you need for vision. In: IEEE Transactions on Computer Vision and Pattern Recognition (CVPR). pp. 10819–10829 (2022)

[^99]: Yu, W., Zhou, P., Yan, S., Wang, X.: Inceptionnext: When inception meets convnext. arXiv preprint arXiv:2303.16900 (2023)

[^100]: Yun, S., Han, D., Oh, S.J., Chun, S., Choe, J., Yoo, Y.: Cutmix: Regularization strategy to train strong classifiers with localizable features. In: International Conference on Computer Vision (ICCV). pp. 6023–6032 (2019)

[^101]: Zhang, H., Cisse, M., Dauphin, Y.N., Lopez-Paz, D.: mixup: Beyond empirical risk minimization. In: International Conference on Learning Representations (ICLR) (2018)

[^102]: Zhang, J., Jin, Y., Xu, J., Xu, X., Zhang, Y.: Mdu-net: Multi-scale densely connected u-net forbiomedical image segmentation. Health Information Science and Systems (HISS) (2023)

[^103]: Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y.: Random erasing data augmentation. In: AAAI Conference on Artificial Intelligence (AAAI). pp. 13001–13008 (2020)

[^104]: Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Semantic understanding of scenes through the ade20k dataset. International Journal of Computer Vision (IJCV) 127, 302–321 (2018)

[^105]: Zhu, L., Wang, X., Ke, Z., Zhang, W., Lau, R.: Biformer: Vision transformer with bi-level routing attention. IEEE Transactions on Computer Vision and Pattern Recognition (CVPR) (2023)