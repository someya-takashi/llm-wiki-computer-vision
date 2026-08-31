---
title: "InceptionNeXt: When Inception Meets ConvNeXt"
source: "https://ar5iv.labs.arxiv.org/html/2303.16900"
author:
published:
created: 2026-08-31
description: "Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt whi…"
tags:
  - "clippings"
---
Weihao Yu    Pan Zhou    Shuicheng Yan Affiliation: National University of Singapore Singapore Management University Sea AI Lab Skywork AIweihaoyu@u.nus.edu  panzhou@smu.edu.sg  shuicheng.yan@kunlun-inc.com  xinchao@nus.edu.sgCode: [https://github.com/sail-sg/inceptionnext](https://github.com/sail-sg/inceptionnext)    Xinchao Wang Thanks: Corresponding Author.

###### Abstract

Inspired by the long-range modeling ability of ViTs, large-kernel convolutions are widely studied and adopted recently to enlarge the receptive field and improve model performance, like the remarkable work ConvNeXt which employs $7\times 7$ depthwise convolution. Although such depthwise operator only consumes a few FLOPs, it largely harms the model efficiency on powerful computing devices due to the high memory access costs. For example, ConvNeXt-T has similar FLOPs with ResNet-50 but only achieves $\sim 60\%$ throughputs when trained on A100 GPUs with full precision. Although reducing the kernel size of ConvNeXt can improve speed, it results in significant performance degradation, which poses a challenging problem: How to speed up large-kernel-based CNN models while preserving their performance. To tackle this issue, inspired by Inceptions, we propose to decompose large-kernel depthwise convolution into four parallel branches along channel dimension, i.e., small square kernel, two orthogonal band kernels, and an identity mapping. With this new Inception depthwise convolution, we build a series of networks, namely IncepitonNeXt, which not only enjoy high throughputs but also maintain competitive performance. For instance, InceptionNeXt-T achieves $1.6\times$ higher training throughputs than ConvNeX-T, as well as attains 0.2% top-1 accuracy improvement on ImageNet-1K. We anticipate InceptionNeXt can serve as an economical baseline for future architecture design to reduce carbon footprint.

## 1 Introduction

Figure 1: Trade-off between accuracy and training throughput. All models are trained under the DeiT training hyperparameters [^65] [^39] [^40] [^73]. The training throughput is measured on an A100 GPU with batch size of 128. ConvNeXt-T/k $n$ means variants with depthwise convolution kernel size of $n\times n$. InceptionNeXt-T enjoys both ResNet-50’s speed and ConvNeXt-T’s accuracy.

Figure 2: Block illustration of MetaFormer, MetaNext, ConvNeXt and InceptionNeXt. Similar to MetaFormer block [^78], MetaNeXt is a general block abstracted from ConvNeXt [^40]. MetaNeXt can be regarded as a simpler version obtained from MetaFormer by merging two residual sub-blocks into one. It is worth noting that the token mixer used in MetaNeXt cannot be too complex (*e.g.*, self-attention [^67]) or it may fail to train to converge. By specifying the token mixer as depthwise convolution or Inception depthwise convolution, the model is instantiated as ConvNeXt or InceptionNeXt block. Compared with ConvNeXt, InceptionNeXt is more efficient because it decomposes expensive large-kernel depthwise convolution into four efficient parallel branches.

Reviewing the history of deep learning [^35], Convolutional Neural Networks (CNNs) [^33] [^34] are definitely the most popular models in computer vision. The watershed moment arrived in 2012 when AlexNet [^32] claimed victory in the ImageNet contest, ushering in a new era for CNNs in computer vision [^32] [^11] [^54]. Since then, a myriad of influential CNNs has emerged like Network In Network [^37], VGG [^57], Inception Nets [^59], ResNe(X)t [^22] [^75], DenseNet [^27] and other efficient models [^25] [^55] [^85] [^62] [^63].

Motivated by the great achievement of Transformer in NLP, researchers attempt to integrate its modules or blocks into vision CNN models [^71] [^4] [^28] [^2], *e.g.*, the representative works like Non-local Neural Networks [^71] and DETR [^4], or even make self-attention as stand-alone primitive [^50] [^86]. Moreover, inspired by the language generative pre-training [^46], Image GPT (iGPT) [^6] treats pixels as tokens and adopts pure Transformer for visual self-supervised learning. However, iGPT faces limitations in handling high-resolution images due to computational costs [^6]. The breakthrough came with Vision Transformer (ViT) [^16], which treats image patches as tokens, leverages a pure Transformer as the backbone, and has demonstrated remarkable performance in image classification after large-scale supervised image pre-training.

Apparently, the success of ViT [^16] further ignites the enthusiasm for Transformer’s application in computer vision. Many ViT variants [^65] [^80] [^68] [^39] [^15] [^76] [^36], like DeiT [^65] and Swin [^39], are proposed and have achieved remarkable performance across a wide range of vision tasks. The superior performance of ViT-like models over traditional CNNs (*e.g.*, Swin-T’s 81.2% v.s. ResNet-50’s 76.1% on ImageNet [^39] [^22] [^11] [^54]) leads many researchers to believe that Transformers will eventually replace CNNs and dominate the field of computer vision.

It is time for CNN to fight back. With advanced training techniques in DeiT [^65] and Swin [^39], the work of “ResNet strikes back” [^73] shows that the performance of ResNet-50 can rise by 2.3%, up to 78.4%. Further, ConvNeXt [^40] demonstrates that with modern modules like GELU [^23] activation and large kernel size similar to attention window size [^39], CNN models can consistently outperform Swin Transformer [^39] in various settings and tasks. ConvNeXt is not alone: More and more works have shown similar observations [^14] [^18] [^77] [^51] [^38] [^79] [^69] [^24], like RepLKNet [^14] and SLaK [^38]. Among these modern CNN models, the common key feature is the large receptive field that is usually achieved by depthwise convolution [^43] [^7] with large kernel size (*e.g.*, $7\times 7$).

However, despite its small FLOPs, depthwise convolution is actually an “expensive” operator because it brings high memory access costs and can be a bottleneck on powerful computing devices, like GPUs [^42]. Moreover, as observed in [^14], larger kernel sizes lead to significantly lower speeds. As shown in Figure 1, the ConvNeXt-T with a default $7\times 7$ kernel size is $1.4\times$ slower than that with small kernel size of $3\times 3$, and is $1.8\times$ slower than ResNet-50, although they have similar FLOPs. However, using a smaller kernel size limits the receptive field, which can result in performance degradation. For example, ConvNeXt-T/k3 suffers a performance drop of $0.6\%$ top-1 accuracy on the ImageNet-1K dataset when compared to ConvNeXt-T/k7, where k $n$ denotes a kernel size of $n\times n$.

This poses a challenging problem: How to speed up large-kernel CNNs while preserving their performance? In this paper, we aim to address this issue by building upon ConvNeXt as our baseline and improving the depthwise convolution module. Through our preliminary experiments based on ConvNeXt (see Table 1), we find that not all input channels need to undergo the computationally expensive depthwise convolution operation [^42]. Accordingly, we propose to leave some channels unaltered and process only a portion of the channels with the depthwise convolution operation. Next, we propose to decompose large kernel of depthwise convolution into several groups of small kernels in Inception style [^59] [^60] [^61]. Specifically, for the processing channels, $1/3$ of channels are conducted with kernel of $3\times 3$, another $1/3$ are with $1\times k$, and the remaining $1/3$ are with $k\times 1$. With this new simple and cheap operator, termed as “Inception depthwise convolution”, our built model InceptionNeXt achieves a much better trade-off between accuracy and speed. For example, as shown in Figure 1, InceptionNeXt-T achieves higher accuracy than ConvNeXt-T while enjoying $1.6\times$ speedup of training throughput similar to ResNet-50.

The contributions of this paper are two-fold. Firstly, we identify the speed bottleneck of ConvNeXt as shown in Figure 1. To solve this speed bottleneck while keeping accuracy, we propose Inception depthwise convolution which decomposes the expensive depthwise convolution into three convolution branches with small kernel sizes as well as a branch of identity mapping. Secondly, extensive experiments on image classification and semantic segmentation show a better speed-accuracy trade-off of our model InceptionNeXt than ConvNeXt. We hope that InceptionNeXt can serve as a new CNN baseline to speed up the research of neural architecture design.

## 2 Related work

### 2.1 Transformer v.s.CNN

Transformer [^67] was introduced in 2017 for NLP tasks because of its parallel training and also better performance than LSTM. Then many famous NLP models are built on Transformer, including GPT series [^46] [^47] [^3] [^44], BERT [^12], T5 [^49], and OPT [^84]. For the application of the Transformer in vision tasks, Vision Transformer (ViT) is definitely the seminal work, showing that Transformer can achieve impressive performance after large-scale supervised training. Follow-up works [^65] [^80] [^68] [^20] [^70] [^52] [^53] like Swin [^39] continually improve model performance, achieving new state-of-the-art on various vision tasks. These results seem to tell us “Attention is all you need” [^67].

But it is not that simple. ViT variants like DeiT usually adopt modern training procedures including various advanced techniques of data augmentation [^10] [^9] [^83] [^81] [^87], regularization [^60] [^26] and optimizers [^30] [^41]. Wightman et al. find that with similar training procedures, the performance of ResNet can be largely improved. Besides, Yu et al. [^78] argue that the general architecture instead of attention plays a key role in model performance. Han et al. [^21] find by replacing attention in Swin with regular or dynamic depthwise convolution, the model can also obtain comparable performance. ConvNeXt [^40], a remarkable work, modernizes ResNet into an advanced version with some designs from ViTs, and the resulting models consistently outperform Swin [^39]. Other works like RepLKNet [^14], VAN [^18], FocalNets [^77], HorNet [^51], SLKNet [^38], ConvFormer [^79], Conv2Former [^24], and InternImage [^69] constantly improve performance of CNNs. Despite the high performance obtained, these models neglect efficiency, exhibiting lower speed than ConvNeXt. Actually, ConvNeXt is also not an efficient model compared with ResNet. We argue that CNN models should keep the original advantage of efficiency. Thus, in this paper, we aim to improve the model efficiency of CNNs while maintaining high performance.

### 2.2 Convolution with large kernels.

Well-known works, like AlexNet [^32] and Inception v1 [^59] already utilize large kernels up to $11\times 11$ and $7\times 7$, respectively. To improve the efficiency of large kernels, VGG [^57] proposes to heavily stack $3\times 3$ convolutions while Inception v3 [^60] factorizes $k\times k$ convolution into $1\times k$ and $k\times 1$ staking sequentially. For depthwise convolution, MixConv [^64] splits kernels into several groups from $3\times 3$ to $k\times k$. Besides, Peng et al. find that large kernels are important for semantic segmentation and they decompose large kernels similar to Inception v3 [^60]. Witnessing the success of Transformer in vision tasks [^16] [^68] [^39], large-kernel convolution is more emphasized since it can offer a large receptive field to imitate attention [^21] [^40]. For example, ConvNeXt adopts kernel size of $7\times 7$ for depthwise convolution by default. To employ larger kernels, RepLKNet [^14] proposes to utilize structural re-parameterization techniques [^82] [^13] to scale up kernel size to $31\times 31$; VAN [^18] sequentially stacks large-kernel depth-wise convolution (DW-Conv) and depth-wise dilation convolution to obtain $21\times 21$ receptive filed; FocalNets [^77] employ a gating mechanism to fuse multi-level features from stacking depthwise convolutions; SegNeXt [^17] learns multi-scale features by multiple branches of staking $1\times k$ and $k\times 1$. Recently, SLaK [^38] factorizes large kernel $k\times k$ into two small non-square kernels ($k\times s$ and $s\times k$ with $s<k$). Unlike these works, we do not aim to scale up larger kernels. Instead, we target efficiency and decompose large kernels in a simple and speed-friendly way while keeping comparable performance.

## 3 Formulation and Method

Algorithm 1 Inception Depthwise Convolution (PyTorch-like Code)

⬇

import torch.nn as nn

class InceptionDWConv2d(nn.Module):

def \_\_init\_\_(self, in\_channels, square\_kernel\_size=3, band\_kernel\_size=11, branch\_ratio=1/8):

super().\_\_init\_\_()

gc = int(in\_channels \* branch\_ratio) # channel number of a convolution branch

self.dwconv\_hw = nn.Conv2d(gc, gc, square\_kernel\_size, padding=square\_kernel\_size//2, groups=gc)

self.dwconv\_w = nn.Conv2d(gc, gc, kernel\_size=(1, band\_kernel\_size), padding=(0, band\_kernel\_size//2), groups=gc)

self.dwconv\_h = nn.Conv2d(gc, gc, kernel\_size=(band\_kernel\_size, 1), padding=(band\_kernel\_size//2, 0), groups=gc)

self.split\_indexes = (gc, gc, gc, in\_channels - 3 \* gc)

def forward(self, x):

\# B, C, H, W = x.shape

x\_hw, x\_w, x\_h, x\_id = torch.split(x, self.split\_indexes, dim=1)

return torch.cat(

(self.dwconv\_hw(x\_hw),

self.dwconv\_w(x\_w),

self.dwconv\_h(x\_h),

x\_id),

dim=1)

### 3.1 MetaNeXt

Formulation of MetaNeXt Block. In ConvNeXt [^40], for its each ConvNeXt block, the input $X$ is first processed by a depthwise convolutioin to propagate information along spatial dimensions. We follow MetaFormer [^78] to abstract the depthwise convolution as a token mixer which is responsible for spatial information interaction. Accordingly, as shown in the second subfigure in Figure 2, the ConvNeXt is abstracted as MetaNeXt block. Formally, in a MetaNeXt block, its input $X$ is firstly processed as

$$
X^{\prime}=\mathrm{TokenMixer}(X),
$$

where $X,X^{\prime}\in\mathbb{R}^{B\times C\times H\times W}$ with $B$, $C$, $H$ and $W$ respectively denoting batch size, channel number, height and width. Then the output from the token mixer is normalized

$$
Y=\mathrm{Norm}(X^{\prime}).
$$

After normalization [^29] [^1], the features are then fed into an MLP module which consists of two fully-connected layers with an activation function between them, the same as feed-forward network in Transformer [^67]. The two fully-connected layers can also be implemented by $1\times 1$ convolutions. Also, shortcut connection [^22] [^58] is adopted. This process can be expressed by

$$
Y=\mathrm{Conv}_{1\times 1}^{rC\rightarrow C}\{\sigma[\mathrm{Conv}_{1\times 1}^{C\rightarrow rC}(Y)]\}+X,
$$

where $\mathrm{Conv}_{k\times k}^{C_{i}\rightarrow C_{o}}$ means convolution with kernel size of $k\times k$, input channels of $C_{i}$ and output channels of $C_{o}$; $r$ is the expansion ratio and $\sigma$ denotes activation function.

Comparison to MetaFormer block. As shown in Figure 2, it can be found that MetaNeXt block shares similar modules with MetaFormer block [^78], *e.g.*, token mixer and MLP. Nevertheless, a critical difference between the two models lies in the number of shortcut connections [^22] [^58]. MetaNeXt block implements a single shortcut connection, whereas the MetaFormer block incorporates two, one for the token mixer and the other for the MLP. From this aspect, MetaNeXt block can be regarded as a result of merging two residual sub-blocks from MetaFormer, thereby simplifying the overall architecture. As a result, the MetaNeXt architecture exhibits a higher speed compared to MetaFormer. However, this simpler design comes with a limitation: the token mixer component in MetaNeXt cannot be complicated (*e.g.*, Attention) as shown in our experiments (Table 5).

Instantiation to ConvNeXt. As shown in Figure 2, in ConvNeXt, the token mixer is simply implemented by a depthwise convolution

$$
X^{\prime}=\mathrm{TokenMixer}(X)=\mathrm{DWConv}_{k\times k}^{C\rightarrow C}(X),
$$

where $\mathrm{DWConv}_{k\times k}^{C\rightarrow C}$ denotes depthwise convolution with kernel size of $k\times k$. In ConvNeXt, $k$ is set as 7 by default.

### 3.2 Inception depthwise convolution

<table><tbody><tr><td rowspan="2">Kernel size of DWConv</td><td rowspan="2">Convolution ratio</td><td rowspan="2">Params (M)</td><td rowspan="2">MACs (G)</td><td colspan="2">Throughput</td><td rowspan="2">Top-1 (%)</td></tr><tr><td>Train</td><td>Inference</td></tr><tr><td><math><semantics><mrow><mn>7</mn> <mo>×</mo> <mn>7</mn></mrow> <annotation>7\times 7</annotation></semantics></math></td><td>1.0</td><td>28.6</td><td>4.5</td><td>575</td><td>2413</td><td>82.1*</td></tr><tr><td><math><semantics><mrow><mn>5</mn> <mo>×</mo> <mn>5</mn></mrow> <annotation>5\times 5</annotation></semantics></math></td><td>1.0</td><td>28.4</td><td>4.4</td><td>675</td><td>2704</td><td>82.0</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td>1.0</td><td>28.3</td><td>4.4</td><td>798</td><td>2802</td><td>81.5</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>/</mo> <mn>2</mn></mrow> <annotation>1/2</annotation></semantics></math></td><td>28.3</td><td>4.4</td><td>818</td><td>2740</td><td>81.4</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td><math><semantics><mrow><mn>3</mn> <mo>/</mo> <mn>8</mn></mrow> <annotation>3/8</annotation></semantics></math></td><td>28.3</td><td>4.4</td><td>847</td><td>2762</td><td>81.4</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>/</mo> <mn>4</mn></mrow> <annotation>1/4</annotation></semantics></math></td><td>28.3</td><td>4.4</td><td>871</td><td>2808</td><td>81.3</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>/</mo> <mn>8</mn></mrow> <annotation>1/8</annotation></semantics></math></td><td>28.3</td><td>4.4</td><td>901</td><td>2833</td><td>80.8</td></tr><tr><td><math><semantics><mrow><mn>3</mn> <mo>×</mo> <mn>3</mn></mrow> <annotation>3\times 3</annotation></semantics></math></td><td><math><semantics><mrow><mn>1</mn> <mo>/</mo> <mn>16</mn></mrow> <annotation>1/16</annotation></semantics></math></td><td>28.3</td><td>4.4</td><td>916</td><td>2846</td><td>80.1</td></tr></tbody></table>

Table 1: Preliminary experiments based on ConvNeXt-T. Convolution ratio means the ratio of channels to be processed by depthwise convolution while the other channels keep unchanged. Throughputs are measured on an A100 GPU with batch size of 128 and TF32. \* The result is reported in ConvNeXt paper [^40].

Preliminary experiments on ConvNeXt-T. We first conducted preliminary experiments based on ConvNeXt-T and report the results in Table 1. Firstly, the kernel size of depthwise convolution is reduced from $7\times 7$ to $3\times 3$. Compared to the model with kernel size of $7\times 7$, the one with kernel size of $3\times 3$ enjoys $1.4\times$ higher training throughput, but suffers a significant performance drop from 82.1% to 81.5%. Next, inspired by ShuffleNet V2 [^42], we only feed partial input channels into depthwise convolution while the remaining ones keep unchanged. The number of processed input channels is controlled by a ratio. It is found that when the ratio is reduced from 1 to $1/4$, the training throughput can be further improved while the performance almost maintains. In summary, these preliminary experiments convey two findings on ConvNeXt. Finding 1: Large-kernel depthwise convolution is the speed bottleneck. Finding 2: Processing partial channels is good enough in single depthwise convolution layer [^42].

| Conv. type | Params | FLOPs |
| --- | --- | --- |
| Conventional conv. | $k^{2}C^{2}$ | $2k^{2}C^{2}HW$ |
| Depthwise conv. | $k^{2}C$ | $2k^{2}CHW$ |
| Inception dep. conv. | $(2k+9)C/8$ | $(2k+9)CHW/4$ |

Table 2: Complexity of different types of convolution. For simplicity, assume input and output channels are the same, and the bias term is omitted. $k$, $C$, $H$ and $W$ denote kernel size, channel number, height and width, respectively. The parameters and FLOPs of vanilla convolution and depthwise convolution are quadratic to kernel size $k$. In contrast, Inception depthwise convolution is linear to $k$.

Figure 3: Comparison of FLOPs between depthwise convolution and Inception depthwise convolution. Inception depthwise convolution is much more efficient than depthwise convolution as kernel size increases.

<table><tbody><tr><td rowspan="2">Stage</td><td rowspan="2">#Tokens</td><td colspan="2" rowspan="2">Layer Specification</td><td colspan="4">InceptionNeXt</td></tr><tr><td>A</td><td>T</td><td>S</td><td>B</td></tr><tr><td rowspan="5">1</td><td rowspan="5"><math><semantics><mrow><mfrac><mi>H</mi> <mn>4</mn></mfrac> <mo>×</mo> <mfrac><mi>W</mi> <mn>4</mn></mfrac></mrow> <annotation>\frac{H}{4}\times\frac{W}{4}</annotation></semantics></math></td><td rowspan="2">Down- sampling</td><td>Kernel Size</td><td colspan="4"><math><semantics><mrow><mn>4</mn> <mo>×</mo> <mn>4</mn></mrow> <annotation>4\times 4</annotation></semantics></math>, stride <math><semantics><mn>4</mn> <annotation>4</annotation></semantics></math></td></tr><tr><td>Embed. Dim.</td><td><math><semantics><mn>40</mn> <annotation>40</annotation></semantics></math></td><td colspan="2"><math><semantics><mn>96</mn> <annotation>96</annotation></semantics></math></td><td><math><semantics><mn>128</mn> <annotation>128</annotation></semantics></math></td></tr><tr><td rowspan="4">InceptionNeXt Block</td><td>Band kernel size</td><td>9</td><td colspan="3">11</td></tr><tr><td>Conv. group ratio</td><td>1/4</td><td colspan="3">1/8</td></tr><tr><td>MLP Ratio</td><td colspan="4">4</td></tr><tr><td></td><td></td><td># Block</td><td>2</td><td colspan="3">3</td></tr><tr><td rowspan="5">2</td><td rowspan="5"><math><semantics><mrow><mfrac><mi>H</mi> <mn>8</mn></mfrac> <mo>×</mo> <mfrac><mi>W</mi> <mn>8</mn></mfrac></mrow> <annotation>\frac{H}{8}\times\frac{W}{8}</annotation></semantics></math></td><td rowspan="2">Down- sampling</td><td>Kernel Size</td><td colspan="4"><math><semantics><mrow><mn>2</mn> <mo>×</mo> <mn>2</mn></mrow> <annotation>2\times 2</annotation></semantics></math>, stride <math><semantics><mn>2</mn> <annotation>2</annotation></semantics></math></td></tr><tr><td>Embed. Dim.</td><td><math><semantics><mn>90</mn> <annotation>90</annotation></semantics></math></td><td colspan="2"><math><semantics><mn>192</mn> <annotation>192</annotation></semantics></math></td><td><math><semantics><mn>256</mn> <annotation>256</annotation></semantics></math></td></tr><tr><td rowspan="4">InceptionNeXt Block</td><td>Band kernel size</td><td>9</td><td colspan="3">11</td></tr><tr><td>Conv. group ratio</td><td>1/4</td><td colspan="3">1/8</td></tr><tr><td>MLP Ratio</td><td colspan="4">4</td></tr><tr><td></td><td></td><td># Block</td><td>2</td><td colspan="3">3</td></tr><tr><td rowspan="5">3</td><td rowspan="5"><math><semantics><mrow><mfrac><mi>H</mi> <mn>16</mn></mfrac> <mo>×</mo> <mfrac><mi>W</mi> <mn>16</mn></mfrac></mrow> <annotation>\frac{H}{16}\times\frac{W}{16}</annotation></semantics></math></td><td rowspan="2">Down- sampling</td><td>Kernel Size</td><td colspan="4"><math><semantics><mrow><mn>2</mn> <mo>×</mo> <mn>2</mn></mrow> <annotation>2\times 2</annotation></semantics></math>, stride <math><semantics><mn>2</mn> <annotation>2</annotation></semantics></math></td></tr><tr><td>Embed. Dim.</td><td><math><semantics><mn>180</mn> <annotation>180</annotation></semantics></math></td><td colspan="2"><math><semantics><mn>384</mn> <annotation>384</annotation></semantics></math></td><td><math><semantics><mn>512</mn> <annotation>512</annotation></semantics></math></td></tr><tr><td rowspan="4">InceptionNeXt Block</td><td>Band kernel size</td><td>9</td><td colspan="3">11</td></tr><tr><td>Conv. group ratio</td><td>1/4</td><td colspan="3">1/8</td></tr><tr><td>MLP Ratio</td><td colspan="4">4</td></tr><tr><td></td><td></td><td># Block</td><td>6</td><td>9</td><td colspan="2">27</td></tr><tr><td rowspan="5">4</td><td rowspan="5"><math><semantics><mrow><mfrac><mi>H</mi> <mn>32</mn></mfrac> <mo>×</mo> <mfrac><mi>W</mi> <mn>32</mn></mfrac></mrow> <annotation>\frac{H}{32}\times\frac{W}{32}</annotation></semantics></math></td><td rowspan="2">Down- sampling</td><td>Kernel Size</td><td colspan="4"><math><semantics><mrow><mn>2</mn> <mo>×</mo> <mn>2</mn></mrow> <annotation>2\times 2</annotation></semantics></math>, stride <math><semantics><mn>2</mn> <annotation>2</annotation></semantics></math></td></tr><tr><td>Embed. Dim.</td><td><math><semantics><mn>320</mn> <annotation>320</annotation></semantics></math></td><td colspan="2"><math><semantics><mn>768</mn> <annotation>768</annotation></semantics></math></td><td><math><semantics><mn>1024</mn> <annotation>1024</annotation></semantics></math></td></tr><tr><td rowspan="4">InceptionNeXt Block</td><td>Band kernel size</td><td>9</td><td colspan="3">11</td></tr><tr><td>Conv. group ratio</td><td>1/4</td><td colspan="3">1/8</td></tr><tr><td>MLP Ratio</td><td colspan="4">3</td></tr><tr><td></td><td></td><td># Block</td><td>2</td><td colspan="3">3</td></tr><tr><td colspan="8">Global average pooling, MLP</td></tr><tr><td colspan="4">Parameters (M)</td><td>4.2</td><td>28.1</td><td>49.4</td><td>86.7</td></tr><tr><td colspan="4">MACs (G)</td><td>0.5</td><td>4.2</td><td>8.4</td><td>14.9</td></tr></tbody></table>

Table 3: Configurations of InceptionNeXt models which have similar model configurations to ConvNeXt [^40]. “A”, “T”, “S” and “B” represent “Atto”, “Tiny”, “Small” and “Base”, respectively.

Formulation. Based on the above findings, we propose a new type of convolution to keep both accuracy and efficiency. According to Fingding 2, we leave partial channels unchanged and denote them as a branch of identity mapping. Motivated by Fingding 1, for the processing channels, we propose to decompose the depthwise operations in Inception style [^59] [^60] [^61]. Inception [^59] utilizes several branches of small kernels (*e.g.*, $3\times 3$) and large kernels (*e.g.*, $5\times 5$). Similarly, we adopt $3\times 3$ as one of our branches but get rid of the usage of the large square kernels because of their slow practical speed. Instead, large kernel $k_{h}\times k_{w}$ is decomposed as $1\times k_{w}$ and $k_{h}\times 1$ inspired by Inception v3 [^60].

Specifically, for input $X$, we split it into four groups along the channel dimension,

$$
\begin{split}X_{\mathrm{hw}},X_{\mathrm{w}},X_{\mathrm{h}},X_{\mathrm{id}}&=\mathrm{Split}(X)\\
&=X_{:,:g},X_{:,g:2g},X_{:,2g:3g},X_{:,3g:},\end{split}
$$

where $g$ is the channel numbers of convolution branches. We can set a ratio $r_{g}$ to determine the branch channel numbers by $g=r_{g}C$. Next, the splitting inputs are fed into different parallel branches,

$$
\begin{split}X^{\prime}_{\mathrm{hw}}&=\mathrm{DWConv}_{k_{s}\times k_{s}}^{g\rightarrow g}(X_{\mathrm{hw}}),\\
X^{\prime}_{\mathrm{w}}&=\mathrm{DWConv}_{1\times k_{b}}^{g\rightarrow g}(X_{\mathrm{w}}),\\
X^{\prime}_{\mathrm{h}}&=\mathrm{DWConv}_{k_{b}\times 1}^{g\rightarrow g}(X_{\mathrm{h}}),\\
X^{\prime}_{\mathrm{id}}&=X_{\mathrm{id}},\\
\end{split}
$$

where $k_{s}$ denotes the small square kernel size set as 3 by default; $k_{b}$ represents the band kernel size set as 11 by default. Finally, the outputs from each branch are concatenated,

$$
X^{\prime}=\mathrm{Concat}(X^{\prime}_{\mathrm{hw}},X^{\prime}_{\mathrm{w}},X^{\prime}_{\mathrm{h}},X^{\prime}_{\mathrm{id}}).
$$

The illustration of InceptionNeXt block is shown in Figure 2. Moreover, its PyTorch [^45] code is summarized in Algorithm 1.

Complexity. The complexity of three types of convolution, *i.e.*, conventional, depthwise, and Inception depthwise convolution is shown in Table 2. As can be seen, Inception depthwise convolution is much more efficient than the other two types of convolution in terms of parameter numbers of FLOPs. Inception depthwise convolution consumes parameters and FLOPs linear to both channel and kernel size. The comparison of depthwise and Inception depthwise convolutions regarding FLOPs is also clearly shown in Figure 3.

### 3.3 InceptionNeXt

Based on InceptionNeXt block, we can build a series of models named InceptionNeXt. Since ConvNeXt [^40] is the our main comparing baseline, we mainly follow it to build models with several sizes [^72]. Specifically, similar to ResNet [^22] and ConvNeXt, InceptionNeXt also adopts 4-stage framework. The same as ConvNeXt, the numbers of 4 stages are \[2, 2, 6, 2\] for atto size, \[3, 3, 9, 3\] for small size and \[3, 3, 27, 3\] for base size. We adopt Batch Normalization since this paper emphasizes speed. Another difference with ConvNeXt is that InceptionNeXt uses an MLP ratio of 3 in stage 4 and moves the saved parameters to the classifier, which can help reduce a few FLOPs (*e.g.*, 3% for base size). The detailed model configurations are reported in Table 3.

## 4 Experiment

### 4.1 Image classification

<table><tbody><tr><td rowspan="3">Model</td><td rowspan="3">Mixing Type</td><td rowspan="3">Image size</td><td rowspan="3">Params (M)</td><td rowspan="3">MACs (G)</td><td colspan="4">Throughput (img/s)</td><td rowspan="3">Top-1 (%)</td></tr><tr><td colspan="2">A100</td><td colspan="2">2080Ti</td></tr><tr><td>Train</td><td>Infer</td><td>Train</td><td>Infer</td></tr><tr><td>MobileNetV2 (1.4) <sup><a href="#fn:55">55</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>6.1</td><td>0.60</td><td>1001</td><td>5190</td><td>471</td><td>1859</td><td>74.7</td></tr><tr><td>EfficientNet-B0 <sup><a href="#fn:62">62</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>5.3</td><td>0.40</td><td>954</td><td>5502</td><td>464</td><td>1944</td><td>77.1</td></tr><tr><td>GhostNet 1.3 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> <sup><a href="#fn:19">19</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>7.3</td><td>0.24</td><td>946</td><td>7451</td><td>589</td><td>2757</td><td>75.7</td></tr><tr><td>ConvNeXt-A <sup><a href="#fn:40">40</a></sup> <sup><a href="#fn:72">72</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>3.7</td><td>0.55</td><td>835</td><td>4539</td><td>345</td><td>1568</td><td>75.7</td></tr><tr><td>InceptionNeXt-A (Ours)</td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>4.2</td><td>0.51</td><td>2661 <sub>+219%</sub></td><td>9876 <sub>+118%</sub></td><td>992 <sub>+188%</sub></td><td>3595 <sub>+129%</sub></td><td>75.3 <sub>-0.4</sub></td></tr><tr><td>DeiT-S <sup><a href="#fn:65">65</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>22</td><td>4.6</td><td>1227</td><td>3781</td><td>276</td><td>784</td><td>79.8</td></tr><tr><td>T2T-ViT-14 <sup><a href="#fn:80">80</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>22</td><td>4.8</td><td>–</td><td>–</td><td>–</td><td>–</td><td>81.5</td></tr><tr><td>TNT-S <sup><a href="#fn:20">20</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>24</td><td>5.2</td><td>–</td><td>–</td><td>–</td><td>–</td><td>81.5</td></tr><tr><td>Swin-T <sup><a href="#fn:39">39</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>29</td><td>4.5</td><td>564</td><td>1768</td><td>184</td><td>554</td><td>81.3</td></tr><tr><td>Focal-T <sup><a href="#fn:76">76</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>29</td><td>4.9</td><td>–</td><td>–</td><td>–</td><td>–</td><td>82.2</td></tr><tr><td>ResNet-50 <sup><a href="#fn:22">22</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>26</td><td>4.1</td><td>969</td><td>3149</td><td>278</td><td>977</td><td>78.4</td></tr><tr><td>RSB-ResNet-50 <sup><a href="#fn:22">22</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>26</td><td>4.1</td><td>969</td><td>3149</td><td>278</td><td>977</td><td>79.8</td></tr><tr><td>RegNetY-4G <sup><a href="#fn:48">48</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>21</td><td>4.0</td><td>670</td><td>2694</td><td>222</td><td>859</td><td>81.3</td></tr><tr><td>FocalNet-T <sup><a href="#fn:77">77</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>29</td><td>4.5</td><td>–</td><td>–</td><td>–</td><td>–</td><td>82.3</td></tr><tr><td>ConvNeXt-T <sup><a href="#fn:40">40</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>29</td><td>4.5</td><td>575</td><td>2413</td><td>177</td><td>590</td><td>82.1</td></tr><tr><td>InceptionNeXt-T (Ours)</td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>28</td><td>4.2</td><td>901 <sub>+57%</sub></td><td>2900 <sub>+20%</sub></td><td>254 <sub>+44%</sub></td><td>822 <sub>+39%</sub></td><td>82.3 <sub>+0.2</sub></td></tr><tr><td>T2T-ViT-19 <sup><a href="#fn:80">80</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>39</td><td>8.5</td><td>–</td><td>–</td><td>–</td><td>–</td><td>81.9</td></tr><tr><td>PVT-Medium <sup><a href="#fn:68">68</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>44</td><td>6.7</td><td>–</td><td>–</td><td>–</td><td>–</td><td>81.2</td></tr><tr><td>Swin-S <sup><a href="#fn:39">39</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>50</td><td>8.7</td><td>359</td><td>1131</td><td>109</td><td>328</td><td>83.0</td></tr><tr><td>Focal-S <sup><a href="#fn:76">76</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>51</td><td>9.1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>83.5</td></tr><tr><td>RSB-ResNet-101 <sup><a href="#fn:22">22</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>45</td><td>7.9</td><td>620</td><td>2057</td><td>168</td><td>592</td><td>81.3</td></tr><tr><td>RegNetY-8G <sup><a href="#fn:48">48</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>39</td><td>8.0</td><td>689</td><td>1326</td><td>124</td><td>480</td><td>82.1</td></tr><tr><td>FocalNet-S <sup><a href="#fn:77">77</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>50</td><td>8.7</td><td>–</td><td>–</td><td>–</td><td>–</td><td>83.5</td></tr><tr><td>ConvNeXt-S <sup><a href="#fn:40">40</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>50</td><td>8.7</td><td>361</td><td>1535</td><td>105</td><td>353</td><td>83.1</td></tr><tr><td>InceptionNeXt-S (Ours)</td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>49</td><td>8.4</td><td>521 <sub>+44%</sub></td><td>1750 <sub>+14%</sub></td><td>130 <sub>+24%</sub></td><td>447 <sub>+27%</sub></td><td>83.5 <sub>+0.4</sub></td></tr><tr><td>DeiT-B <sup><a href="#fn:65">65</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>86</td><td>17.5</td><td>541</td><td>1608</td><td>86</td><td>259</td><td>81.8</td></tr><tr><td>T2T-ViT-24 <sup><a href="#fn:80">80</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>64</td><td>13.8</td><td>–</td><td>–</td><td>–</td><td>–</td><td>82.3</td></tr><tr><td>TNT-B <sup><a href="#fn:20">20</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>66</td><td>14.1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>82.9</td></tr><tr><td>PVT-Large <sup><a href="#fn:68">68</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>62</td><td>9.8</td><td>–</td><td>–</td><td>–</td><td>–</td><td>81.7</td></tr><tr><td>Swin-B <sup><a href="#fn:39">39</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>88</td><td>15.4</td><td>271</td><td>843</td><td>72</td><td>223</td><td>83.5</td></tr><tr><td>Focal-B <sup><a href="#fn:76">76</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>90</td><td>16.0</td><td>–</td><td>–</td><td>–</td><td>–</td><td>83.8</td></tr><tr><td>RSB-ResNet-152 <sup><a href="#fn:22">22</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>60</td><td>11.6</td><td>437</td><td>1457</td><td>115</td><td>415</td><td>81.8</td></tr><tr><td>RegNetY-16G <sup><a href="#fn:48">48</a></sup> <sup><a href="#fn:73">73</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>84</td><td>15.9</td><td>322</td><td>1100</td><td>76</td><td>295</td><td>82.2</td></tr><tr><td>RepLKNet-31B <sup><a href="#fn:14">14</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>79</td><td>15.3</td><td>–</td><td>–</td><td>–</td><td>–</td><td>83.5</td></tr><tr><td>FocalNet-B <sup><a href="#fn:77">77</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>89</td><td>15.4</td><td>–</td><td>–</td><td>–</td><td>–</td><td>83.9</td></tr><tr><td>ConvNeXt-B <sup><a href="#fn:40">40</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>89</td><td>15.4</td><td>267</td><td>1122</td><td>68</td><td>236</td><td>83.8</td></tr><tr><td>InceptionNeXt-B (Ours)</td><td>Conv</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td>87</td><td>14.9</td><td>375 <sub>+40%</sub></td><td>1244 <sub>+11%</sub></td><td>80 <sub>+18%</sub></td><td>287 <sub>+22%</sub></td><td>84.0 <sub>+0.2</sub></td></tr><tr><td>DeiT-B <sup><a href="#fn:65">65</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td><td>86</td><td>55.4</td><td>131</td><td>361</td><td>25</td><td>73</td><td>83.1</td></tr><tr><td>Swin-B <sup><a href="#fn:39">39</a></sup></td><td>Attn</td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td><td>88</td><td>47.1</td><td>104</td><td>296</td><td>21</td><td>65</td><td>84.5</td></tr><tr><td>RepLKNet-31B <sup><a href="#fn:14">14</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td><td>79</td><td>45.1</td><td>–</td><td>–</td><td>–</td><td>–</td><td>84.8</td></tr><tr><td>ConvNeXt-B <sup><a href="#fn:40">40</a></sup></td><td>Conv</td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td><td>89</td><td>45.0</td><td>95</td><td>393</td><td>23</td><td>79</td><td>85.1</td></tr><tr><td>InceptionNeXt-B (Ours)</td><td>Conv</td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td><td>87</td><td>43.6</td><td>139 <sub>+46%</sub></td><td>428 <sub>+9%</sub></td><td>27 <sub>+17%</sub></td><td>97 <sub>+23%</sub></td><td>85.2 <sub>+0.1</sub></td></tr></tbody></table>

Table 4: Performance of models trained on ImageNet-1K. The throughputs are measured on an A100 GPU (PyTorch 1.13.0 and CUDA 11.7.1) with TF32 (TensorFloat-32), and on a 2080Ti (PyTorch 1.8.1 and CUDA 10.2) with FP32. The batch size for throughput benchmarking is initially set as 128 and is reduced until the GPU can host. The better results of “Channel First” and “Channel Last” memory layouts are reported.

Setup. For the image classification task, ImageNet-1K [^11] [^54] is one of the most commonly-used benchmarks, which contains around 1.3 million images in the training set and 50 thousand images in the validation set. To fairly compare with the widely-used baselines, *e.g.*, Swin [^39] and ConvNeXt [^40], we mainly follow the training hyper-parameters from DeiT [^65] without distillation. Specifically, the models are trained by AdamW [^41] optimizer with a learning rate $lr=0.001\times\mathrm{batchsize}/1024$ ($lr=4e-3$ and $\mathrm{batchsize}=4096$ are used in this paper the same as ConvNeXt). Following DeiT, data augmentation includes standard random resized crop, horizontal flip, RandAugment [^10], Mixup [^83], CutMix [^81], Random Erasing [^87] and color jitter. For regularization, label smoothing [^60], stochastic depth [^26], and weight decay are adopted. Like ConvNeXt, we also use LayerScale [^66], a technique to help train deep models. Our code is based on PyTorch [^45] and timm [^72].

<table><tbody><tr><td rowspan="2">Model</td><td rowspan="2">Params (M)</td><td rowspan="2">MACs (G)</td><td colspan="2">Throughput (img/s)</td><td rowspan="2">Top-1 (%)</td></tr><tr><td>Train</td><td>Infer</td></tr><tr><td>DeiT-S <sup><a href="#fn:65">65</a></sup></td><td>22</td><td>4.6</td><td>276</td><td>784</td><td>79.8</td></tr><tr><td>MetaNeXt-Attn</td><td>22</td><td>4.6</td><td>288</td><td>816</td><td>3.9</td></tr><tr><td>ConvNeXt-S (iso.) <sup><a href="#fn:40">40</a></sup></td><td>22</td><td>4.3</td><td>270</td><td>879</td><td>79.7</td></tr><tr><td>InceptionNeXt-S (iso.)</td><td>22</td><td>4.2</td><td>310</td><td>998</td><td>79.7</td></tr></tbody></table>

Table 5: Comparison among ViT, isotropic ConvNeXt and InceptionNeXt. MetaNeXt-Attn is instantiated from MetaNeXt with token mixer of self-attention [^67]. The throughputs are measured on 2080Ti (PyTorch 1.8.1 and CUDA 10.2) with FP32. The batch size for throughput benchmarking is initially set as 128 and is reduced until the GPU can host. The better results of “Channel First” and “Channel Last” memory layouts are reported.

Results. We compare InceptionNeXt with various state-of-the-art models, including attention-based and convolution-based models. As can be seen in Table 4, InceptionNeXt achieves highly competitive performance as well as enjoys higher speed. InceptionNeXt consistently enjoys better accuracy-speed trade-off than ConvNeXt [^40]. For example, InceptionNeXt-T not only surpasses ConvNeXt-T by 0.2%, but also enjoys $1.6\times$ / $1.2\times$ training/inference throughputs on A100 than ConvNeXts, similar to those of ResNet-50. That is to say, InceptionNeXt-T enjoys both ResNet-50’s speed and ConvNeXt-T’s accuracy. Moreover, following Swin and ConvNeXt, we also finetuned the InceptionNeXt-B trained at the resolution of $224\times 224$ to $384\times 384$ for 30 epochs. We can see that InceptionNeXt-B obtains higher train and inference throughputs than ConvNeXt-B while keeping competitive accuracy.

It is observed that the speed improvement is much more significant for the lightweight model size, and the improvement gradually becomes smaller when the model size scales up. The reason is that computation complexity of depthwise and Inception depthwise convolutions are linear to channel number, *i.e.*, $\mathcal{O}(C)$ where $C$ is channel number. For MLPs, their computation complexity is $\mathcal{O}(C^{2})$. For larger models (larger $C$), its computation is further dominated by MLPs. By only improving depthwise convolution, the speed improvement becomes smaller when the model is larger.

Besides the 4-stage framework [^57] [^22] [^39], another notable one is ViT-style [^16] isotropic architecture which has only one stage. To match the parameters and MACs of DeiT-S, we construct InceptionNeXt-S (iso.) following ConvNeXt-S (iso.) [^40]. Specifically, we set the embedding dimension as 384 and the block number as 18. Besides, we build a model called MetaNeXt-Attn which is instantiated from MetaNeXt block by specifying self-attention as token mixer. The aim of this model is to investigate whether it is possible to merge two residual sub-blocks of the Transformer block into a single one. The experiment results are shown in Table 5. It can be seen that InceptionNeXt can also perform well with the isotropic architecture, demonstrating InceptionNeXt exhibits good generalization across different frameworks. It is worth noting that MetaNeXt-Attn could not be trained to converge and only achieved an accuracy of 3.9%. This result suggests that, unlike the token mixer in MetaFormer, the token mixer in MetaNeXt cannot be too complex. If it is, the model may not be trainable.

<table><tbody><tr><td rowspan="2">Backbone</td><td colspan="3">UperNet</td><td></td></tr><tr><td>Params (M)</td><td>MACs (G)</td><td>FPS</td><td>mIoU (%)</td></tr><tr><td>Swin-T <sup><a href="#fn:39">39</a></sup></td><td>60</td><td>945</td><td>20.6</td><td>45.8</td></tr><tr><td>ConvNeXt-T <sup><a href="#fn:40">40</a></sup></td><td>60</td><td>939</td><td>20.6</td><td>46.7</td></tr><tr><td>InceptionNeXt-T</td><td>56</td><td>933</td><td>22.7</td><td>47.9</td></tr><tr><td>Swin-S <sup><a href="#fn:39">39</a></sup></td><td>81</td><td>1038</td><td>16.2</td><td>49.5</td></tr><tr><td>ConvNeXt-S <sup><a href="#fn:40">40</a></sup></td><td>82</td><td>1027</td><td>16.8</td><td>49.6</td></tr><tr><td>InceptionNeXt-S</td><td>78</td><td>1020</td><td>17.6</td><td>50.0</td></tr><tr><td>Swin-B <sup><a href="#fn:39">39</a></sup></td><td>121</td><td>1188</td><td>16.2</td><td>49.7</td></tr><tr><td>ConvNeXt-B <sup><a href="#fn:40">40</a></sup></td><td>122</td><td>1170</td><td>15.7</td><td>49.9</td></tr><tr><td>InceptionNeXt-B</td><td>115</td><td>1159</td><td>17.5</td><td>50.6</td></tr></tbody></table>

Table 6: Performance of semantic segmentation with UperNet [^74] on ADE20K [^88] validation set. Images are cropped to $512\times 512$ for training. The MACs are measured with input size of $512\times 2048$. The FPS are benchamrked on 2080Ti.

<table><tbody><tr><td rowspan="2">Backbone</td><td colspan="3">Semantic FPN</td><td></td></tr><tr><td>Params (M)</td><td>MACs (G)</td><td>FPS</td><td>mIoU (%)</td></tr><tr><td>ResNet-50 <sup><a href="#fn:22">22</a></sup></td><td>29</td><td>46</td><td>30.2</td><td>36.7</td></tr><tr><td>PVT-Small <sup><a href="#fn:68">68</a></sup></td><td>28</td><td>45</td><td>27.2</td><td>39.8</td></tr><tr><td>PoolFormer-S24 <sup><a href="#fn:78">78</a></sup></td><td>23</td><td>39</td><td>28.8</td><td>40.3</td></tr><tr><td>InceptionNeXt-T</td><td>28</td><td>44</td><td>31.4</td><td>43.1</td></tr><tr><td>ResNet-101 <sup><a href="#fn:22">22</a></sup></td><td>48</td><td>65</td><td>22.2</td><td>38.8</td></tr><tr><td>ResNeXt-101-32x4d <sup><a href="#fn:75">75</a></sup></td><td>47</td><td>65</td><td>–</td><td>39.7</td></tr><tr><td>PVT-Medium <sup><a href="#fn:68">68</a></sup></td><td>48</td><td>61</td><td>20.0</td><td>41.6</td></tr><tr><td>PoolFormer-S36 <sup><a href="#fn:78">78</a></sup></td><td>35</td><td>48</td><td>21.6</td><td>42.0</td></tr><tr><td>PoolFormer-M36 <sup><a href="#fn:78">78</a></sup></td><td>60</td><td>68</td><td>15.4</td><td>42.4</td></tr><tr><td>InceptionNeXt-S</td><td>50</td><td>65</td><td>20.7</td><td>45.6</td></tr><tr><td>PVT-Large <sup><a href="#fn:68">68</a></sup></td><td>65</td><td>80</td><td>16.0</td><td>42.1</td></tr><tr><td>ResNeXt-101-64x4d <sup><a href="#fn:75">75</a></sup></td><td>86</td><td>104</td><td>–</td><td>40.2</td></tr><tr><td>PoolFormer-M48 <sup><a href="#fn:78">78</a></sup></td><td>77</td><td>82</td><td>12.1</td><td>42.7</td></tr><tr><td>InceptionNeXt-B</td><td>85</td><td>100</td><td>20.2</td><td>46.4</td></tr></tbody></table>

Table 7: Performance of semantic segmentation with Semantic FPN [^31] on ADE20K [^88] validation set. Images are cropped to $512\times 512$ for training. The MACs are measured with input size of $512\times 512$. The FPS are benchamrked on 2080Ti.

<table><tbody><tr><td rowspan="2">Ablation</td><td rowspan="2">Variant</td><td rowspan="2">Params (M)</td><td rowspan="2">MACs (G)</td><td colspan="2">Throughput</td><td rowspan="2">Top-1 (%)</td></tr><tr><td>Train</td><td>Inference</td></tr><tr><td>Baseline</td><td>None (InceptionNeXt-T)</td><td>28.1</td><td>4.2</td><td>901</td><td>2900</td><td>82.3</td></tr><tr><td rowspan="4">Branch</td><td>Remove horizontal band kernel</td><td>28.0</td><td>4.2</td><td>947</td><td>3093</td><td>81.9</td></tr><tr><td>Remove vertical band kernel</td><td>28.0</td><td>4.2</td><td>954</td><td>3173</td><td>81.9</td></tr><tr><td>Remove small band kernel</td><td>28.0</td><td>4.2</td><td>940</td><td>3004</td><td>82.0</td></tr><tr><td>horizontal and vertical band kernel in parallel <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> in sequence</td><td>28.1</td><td>4.2</td><td>903</td><td>2971</td><td>82.1</td></tr><tr><td rowspan="3">Band kernel size</td><td>Band kernel size 11 <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 7</td><td>28.0</td><td>4.2</td><td>905</td><td>2946</td><td>82.1</td></tr><tr><td>Band kernel size 11 <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 9</td><td>28.1</td><td>4.2</td><td>904</td><td>2916</td><td>82.1</td></tr><tr><td>Band kernel size 11 <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 13</td><td>28.1</td><td>4.2</td><td>896</td><td>2895</td><td>82.0</td></tr><tr><td rowspan="2">Convolution branch ratio</td><td>Conv. branch ratio <math><semantics><mrow><mrow><mn>1</mn> <mo>/</mo> <mn>8</mn></mrow> <mo>→</mo> <mrow><mn>1</mn> <mo>/</mo> <mn>4</mn></mrow></mrow> <annotation>1/8\rightarrow 1/4</annotation></semantics></math></td><td>28.1</td><td>4.2</td><td>834</td><td>2499</td><td>82.2</td></tr><tr><td>Conv. branch ratio <math><semantics><mrow><mrow><mn>1</mn> <mo>/</mo> <mn>8</mn></mrow> <mo>→</mo> <mrow><mn>1</mn> <mo>/</mo> <mn>16</mn></mrow></mrow> <annotation>1/8\rightarrow 1/16</annotation></semantics></math></td><td>28.0</td><td>4.2</td><td>936</td><td>3097</td><td>81.8</td></tr><tr><td>Normalization</td><td>Batch Norm <sup><a href="#fn:29">29</a></sup> <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> Layer Norm <sup><a href="#fn:1">1</a></sup></td><td>28.1</td><td>4.2</td><td>721</td><td>2646</td><td>82.4</td></tr></tbody></table>

Table 8: Ablation for InceptionNeXt on ImageNet-1K classification benchmark. InceptionNeXt-T is utilized as the baseline for the ablation study. Top-1 accuracy on the validation set is reported. The throughputs are measured on an A100 GPU (PyTorch 1.13.0 and CUDA 11.7.1) with TF32 and batch size of 128.

### 4.2 Semantic segmentation

Setup. ADE20K [^88], a commonly used scene parsing benchmark, is used to evaluate our models on semantic segmentation task. ADE20K includes 150 fine-grained semantic categories, containing twenty thousand and two thousand images in the training set and validation set, respectively. The checkpoints trained on ImageNet-1K [^11] at the resolution of $224^{2}$ are utilized to initialize the backbones. Following Swin [^39] and ConvNeXt [^40], we firstly evaluate InceptionNeXt with UperNet [^74]. The models are trained with AdamW [^41] optimizer with learning rate of 6e-5 and batch size of 16 for 160K iterations. Following PVT [^68] and PoolFormer [^78], InceptionNeXt is also evaluated with Semantic FPN [^31]. In common practices [^31] [^5], the batch size is 16 for the setting of 80K iterations. Following PoolFormer [^78], we increase the batch size to 32 and decrease the iterations to 40K to speed up training. AdamW [^30] [^41] is adopted with a learning rate of 2e-4 and a polynomial decay schedule of 0.9 power. Our code is based on PyTorch [^45] and mmsegmentation [^8].

Results. For segmentation with UpNet [^74], the results are shown in Table 6. As can be seen, InceptionNeXt consistently outperforms Swin [^39] and ConvNeXt [^40] for different model sizes. In the method of Semantic FPN [^31] as shown in Table 7, InceptionNeXt significantly surpasses other backbones, like PVT [^68] and PoolFormer [^78]. These results show that InceptionNeXt also has a high potential for dense prediction tasks.

### 4.3 Ablation studies

We conduct ablation studies on ImageNet-1K [^11] [^54] using InceptionNeXt-T as baseline from the following aspects.

Branch. Inception depthwise convolution includes four branches, three convolutional ones, and identity mapping. When removing any branch of horizontal or vertical band kernel, performance significantly drops from 82.3% to 81.9%, demonstrating the importance of these two branches. This is because these two branches with band kernels can enlarge the receptive field of the model. For the branch of small square kernel size of $3\times 3$, removing it can still achieve up to 82.0% top-1 accuracy and bring higher throughput. This inspires us that if we attach more importance to the model speed, the simple version of InceptionNeXt without the square kernel of $3\times 3$ can be adopted. For the band kernel, Inception v3 mostly equips them in a sequential way. We find that this assembling method can also obtain similar performance and even a little speed up the model. A possible reason is that PyTorch/CUDA may have optimized sequential convolutions well, and we only implement the parallel branches at a high level (see Algorithm 1). We believe the parallel method will be faster when it is optimized better. Thus, parallel method for the band kernels is adopted by default.

Band kernel size. It is found the performance can be improved from kernel size 7 to 11, but it drops when the band kernel size increases to 13. This phenomenon may result from the optimization and can be solved by methods like structural re-parameterization [^13] [^14]. For simplicity, we set the kernel size as 11 by default except for atto size.

Convolution branch ratio. When the ratio increases from $1/8$ to $1/4$, performance improvement can not be observed. Ma et al. [^42] also point out that it is not necessary for all channels to conduct convolution. But when the ratio decreases to $1/16$, it brings a serious performance drop. This is because a smaller ratio would limit the degree of token mixing, resulting in performance drop. We thus set the convolution branch ratio as $1/8$ by default except for atto size.

Normalization. When replacing the Batch Normalization [^29] with Layer Normalization [^1], the performance improvement improve by 0.1% but suffer throughput drop in both training and inference. Since this paper focuses on efficiency, we adopt Batch Normalization for InceptionNeXt.

## 5 Conclusion

In this work, we propose an effective and efficient CNN architecture, InceptionNeXt, which enjoys a better trade-off between the practical speed and the performance than previous network architectures. InceptionNeXt decomposes large-kernel depthwise convolution along channel dimension into four parallel branches, including identity mapping, a small square kernel, and two orthogonal band kernels. All these four branches are much more computationally efficient than a large-kernel depthwise convolution in practice, and can also work together to have a large spatial receptive field for good performance. Extensive experimental results demonstrate the superior performance and the high practical efficiency of InceptionNeXt.

Acknowledgement. This project is supported by the National Research Foundation Singapore under its Medium Sized Center for Advanced Robotics Technology Innovation, and the Advanced Research and Technology Innovation Centre (ARTIC), the National University of Singapore under Grant (project number: A0005947-21-00, project reference: ECT-RP2). Weihao Yu was partly supported by Snap Research Fellowship, Google’s TPU Research Cloud (TRC), and Google Cloud Research Credits program. Pan Zhou was supported by the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant. We would like to thank Ross Wightman for integrating the model and code into Hugging Face’s pytorch-image-models repository.

Supplementary Material

## Appendix A Hyper-parameters

### A.1 ImageNet-1K image classification

On ImageNet-1K [^11] [^54] classification benchmark, following ConvNeXt [^40] and ConvNeXt-A trained by timm [^72], we adopt the hyper-parameters shown in Table 9 to train InceptionNeXt at the input resolution of $224^{2}$ and fine-tune it at $384^{2}$. Our code is implemented by PyTorch [^45] based on timm library [^72].

### A.2 Semantic segmentation

For ADE20K [^88] semantic segmentation, we utilize ConvNeXt as the backbone with UpNet [^74] following the configs of Swin [^39], and FPN [^31] following the configs of PVT [^68] and PoolFormer [^78]. The backbone is initialized by checkpoints pre-trained on ImageNet-1K at the resolution of $224^{2}$. The peak stochastic depth rates of the InceptionNeXt backbone are shown in Table 10. Our implementation is based on PyTorch [^45] and mmsegmentation library [^8].

## Appendix B Qualitative results

Grad-CAM [^56] is employed to visualize the activation maps of different models trained on ImageNet-1K, including RSB-ResNet-50 [^22] [^73], Swin-T [^39], ConvNeXt-T [^40] and our InceptionNeXt-T. The results are shown in Figure 4. Compared with other models, InceptionNeXt-T locates key parts more accurately with smaller activation areas.

<table><tbody><tr><td></td><td colspan="5">InceptionNeXt</td></tr><tr><td></td><td colspan="4">Train</td><td>Finetune</td></tr><tr><td></td><td>A</td><td>T</td><td>S</td><td>B</td><td>B</td></tr><tr><td>Input resolution</td><td><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td colspan="3"><math><semantics><msup><mn>224</mn> <mn>2</mn></msup> <annotation>224^{2}</annotation></semantics></math></td><td><math><semantics><msup><mn>384</mn> <mn>2</mn></msup> <annotation>384^{2}</annotation></semantics></math></td></tr><tr><td>Epochs</td><td>450</td><td colspan="3">300</td><td>30</td></tr><tr><td>Batch size</td><td>1280</td><td colspan="3">4096</td><td>1024</td></tr><tr><td>Optimizer</td><td>AdamW</td><td colspan="3">AdamW</td><td>AdamW</td></tr><tr><td>Adam <math><semantics><mi>ϵ</mi> <annotation>\epsilon</annotation></semantics></math></td><td>1e-8</td><td colspan="3">1e-8</td><td>1e-8</td></tr><tr><td>Adam <math><semantics><mrow><mo>(</mo><msub><mi>β</mi> <mn>1</mn></msub><mo>,</mo><msub><mi>β</mi> <mn>2</mn></msub><mo>)</mo></mrow> <annotation>(\beta_{1},\beta_{2})</annotation></semantics></math></td><td>(0.9, 0.999)</td><td colspan="3">(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr><tr><td>Learning rate</td><td>1e-3</td><td colspan="3">4e-3</td><td>5e-5</td></tr><tr><td>Learning rate decay</td><td>Cosine</td><td colspan="3">Cosine</td><td>Cosine</td></tr><tr><td>Gradient clipping</td><td>None</td><td colspan="3">None</td><td>None</td></tr><tr><td>Warmup epochs</td><td>5</td><td colspan="3">20</td><td>None</td></tr><tr><td>Weight decay</td><td>0.05</td><td colspan="3">0.05</td><td>0.05</td></tr><tr><td>Rand Augment</td><td>5/uniform</td><td colspan="3">9/0.5</td><td>9/0.5</td></tr><tr><td>Repeated Augmentation</td><td>off</td><td colspan="3">off</td><td>off</td></tr><tr><td>Cutmix</td><td>1.0</td><td colspan="3">1.0</td><td>1.0</td></tr><tr><td>Mixup</td><td>0.2</td><td colspan="3">0.8</td><td>0.8</td></tr><tr><td>Cutmix-Mixup switch prob</td><td>0.5</td><td colspan="3">0.5</td><td>0.5</td></tr><tr><td>Random erasing prob</td><td>0.1</td><td colspan="3">0.25</td><td>0.25</td></tr><tr><td>Label smoothing</td><td>0.1</td><td colspan="3">0.1</td><td>0.1</td></tr><tr><td>Peak stochastic depth rate</td><td>0.1</td><td>0.1</td><td>0.3</td><td>0.4</td><td>0.7</td></tr><tr><td>Dropout in classifier</td><td>0.0</td><td colspan="3">0.0</td><td>0.5</td></tr><tr><td>LayerScale initialization</td><td>1e-6</td><td colspan="3">1e-6</td><td>Pre-trained</td></tr><tr><td>Random erasing prob</td><td>0.1</td><td colspan="3">0.25</td><td>0.25</td></tr><tr><td>EMA decay rate</td><td>None</td><td colspan="3">None</td><td>0.9999</td></tr></tbody></table>

Table 9: Hyper-parameters of InceptionNeXt on ImageNet-1K image classification.

<table><tbody><tr><td rowspan="2">Method</td><td colspan="3">InceptionNeXt stochastic depth rate</td></tr><tr><td>T</td><td>S</td><td>B</td></tr><tr><td>UperNet <sup><a href="#fn:74">74</a></sup></td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td>FPN <sup><a href="#fn:31">31</a></sup></td><td>0.1</td><td>0.2</td><td>0.2</td></tr></tbody></table>

Table 10: Stochasic depth rate of InceptionNeXt backbone with UperNet and FPN for ADE20K semantic segmentation.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2303.16900/assets/figures/qualitative_results/n07873807/ILSVRC2012_val_00016614_224.JPEG)

Figure 4: Grad-CAM 56 activation maps of different models trained on ImageNet-1K. The visualized images are from the validation set of ImageNet-1K.

[^1]: Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. *arXiv preprint arXiv:1607.06450*, 2016.

[^2]: Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens, and Quoc V Le. Attention augmented convolutional networks. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 3286–3295, 2019.

[^3]: Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

[^4]: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16*, pages 213–229. Springer, 2020.

[^5]: Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille. Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. *IEEE transactions on pattern analysis and machine intelligence*, 40(4):834–848, 2017.

[^6]: Mark Chen, Alec Radford, Rewon Child, Jeffrey Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In *International conference on machine learning*, pages 1691–1703. PMLR, 2020.

[^7]: François Chollet. Xception: Deep learning with depthwise separable convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1251–1258, 2017.

[^8]: MMSegmentation Contributors. MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. [https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation), 2020.

[^9]: Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le. Autoaugment: Learning augmentation policies from data. *arXiv preprint arXiv:1805.09501*, 2018.

[^10]: Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le. Randaugment: Practical automated data augmentation with a reduced search space. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops*, pages 702–703, 2020.

[^11]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pages 248–255. Ieee, 2009.

[^12]: Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

[^13]: Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 13733–13742, 2021.

[^14]: Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11963–11975, 2022.

[^15]: Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, and Baining Guo. Cswin transformer: A general vision transformer backbone with cross-shaped windows. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 12124–12134, 2022.

[^16]: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *arXiv preprint arXiv:2010.11929*, 2020.

[^17]: Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, and Shi-Min Hu. Segnext: Rethinking convolutional attention design for semantic segmentation. *Advances in Neural Information Processing Systems*, 35:1140–1156, 2022a.

[^18]: Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, and Shi-Min Hu. Visual attention network. *arXiv preprint arXiv:2202.09741*, 2022b.

[^19]: Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, and Chang Xu. Ghostnet: More features from cheap operations. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 1580–1589, 2020.

[^20]: Kai Han, An Xiao, Enhua Wu, Jianyuan Guo, Chunjing Xu, and Yunhe Wang. Transformer in transformer. *Advances in Neural Information Processing Systems*, 34:15908–15919, 2021a.

[^21]: Qi Han, Zejia Fan, Qi Dai, Lei Sun, Ming-Ming Cheng, Jiaying Liu, and Jingdong Wang. On the connection between local attention and dynamic depth-wise convolution. *arXiv preprint arXiv:2106.04263*, 2021b.

[^22]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.

[^23]: Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). *arXiv preprint arXiv:1606.08415*, 2016.

[^24]: Qibin Hou, Cheng-Ze Lu, Ming-Ming Cheng, and Jiashi Feng. Conv2former: A simple transformer-style convnet for visual recognition. *arXiv preprint arXiv:2211.11943*, 2022.

[^25]: Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications. *arXiv preprint arXiv:1704.04861*, 2017.

[^26]: Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In *Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14*, pages 646–661. Springer, 2016.

[^27]: Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q Weinberger. Densely connected convolutional networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4700–4708, 2017.

[^28]: Zilong Huang, Xinggang Wang, Lichao Huang, Chang Huang, Yunchao Wei, and Wenyu Liu. Ccnet: Criss-cross attention for semantic segmentation. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 603–612, 2019.

[^29]: Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pages 448–456. pmlr, 2015.

[^30]: Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*, 2015.

[^31]: Alexander Kirillov, Ross Girshick, Kaiming He, and Piotr Dollár. Panoptic feature pyramid networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6399–6408, 2019.

[^32]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Communications of the ACM*, 60(6):84–90, 2017.

[^33]: Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. *Neural computation*, 1(4):541–551, 1989.

[^34]: Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

[^35]: Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. *nature*, 521(7553):436–444, 2015.

[^36]: Yanghao Li, Chao-Yuan Wu, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Mvitv2: Improved multiscale vision transformers for classification and detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 4804–4814, 2022.

[^37]: Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. In *ICLR*, 2014.

[^38]: Shiwei Liu, Tianlong Chen, Xiaohan Chen, Xuxi Chen, Qiao Xiao, Boqian Wu, Mykola Pechenizkiy, Decebal Mocanu, and Zhangyang Wang. More convnets in the 2020s: Scaling up kernels beyond 51x51 using sparsity. *arXiv preprint arXiv:2207.03620*, 2022a.

[^39]: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 10012–10022, 2021.

[^40]: Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 11976–11986, 2022b.

[^41]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.

[^42]: Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for efficient cnn architecture design. In *Proceedings of the European conference on computer vision (ECCV)*, pages 116–131, 2018.

[^43]: Franck Mamalet and Christophe Garcia. Simplifying convnets for fast learning. In *Artificial Neural Networks and Machine Learning–ICANN 2012: 22nd International Conference on Artificial Neural Networks, Lausanne, Switzerland, September 11-14, 2012, Proceedings, Part II 22*, pages 58–65. Springer, 2012.

[^44]: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*, 2022.

[^45]: Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. *Advances in neural information processing systems*, 32, 2019.

[^46]: Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. a.

[^47]: Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. b.

[^48]: Ilija Radosavovic, Raj Prateek Kosaraju, Ross Girshick, Kaiming He, and Piotr Dollár. Designing network design spaces. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10428–10436, 2020.

[^49]: Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *The Journal of Machine Learning Research*, 21(1):5485–5551, 2020.

[^50]: Prajit Ramachandran, Niki Parmar, Ashish Vaswani, Irwan Bello, Anselm Levskaya, and Jon Shlens. Stand-alone self-attention in vision models. *Advances in neural information processing systems*, 32, 2019.

[^51]: Yongming Rao, Wenliang Zhao, Yansong Tang, Jie Zhou, Ser-Nam Lim, and Jiwen Lu. Hornet: Efficient high-order spatial interactions with recursive gated convolutions. *arXiv preprint arXiv:2207.14284*, 2022.

[^52]: Sucheng Ren, Daquan Zhou, Shengfeng He, Jiashi Feng, and Xinchao Wang. Shunted self-attention via multi-scale token aggregation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10853–10862, 2022.

[^53]: Sucheng Ren, Xingyi Yang, Songhua Liu, and Xinchao Wang. Sg-former: Self-guided transformer with evolving token reallocation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 6003–6014, 2023.

[^54]: Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115:211–252, 2015.

[^55]: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 4510–4520, 2018.

[^56]: Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In *Proceedings of the IEEE international conference on computer vision*, pages 618–626, 2017.

[^57]: Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In *3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings*, 2015.

[^58]: Rupesh Kumar Srivastava, Klaus Greff, and Jürgen Schmidhuber. Highway networks. *arXiv preprint arXiv:1505.00387*, 2015.

[^59]: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1–9, 2015.

[^60]: Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 2818–2826, 2016.

[^61]: Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander Alemi. Inception-v4, inception-resnet and the impact of residual connections on learning. In *Proceedings of the AAAI conference on artificial intelligence*, 2017.

[^62]: Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks. In *International conference on machine learning*, pages 6105–6114. PMLR, 2019a.

[^63]: Mingxing Tan and Quoc Le. Efficientnetv2: Smaller models and faster training. In *International conference on machine learning*, pages 10096–10106. PMLR, 2021.

[^64]: Mingxing Tan and Quoc V Le. Mixconv: Mixed depthwise convolutional kernels. *arXiv preprint arXiv:1907.09595*, 2019b.

[^65]: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *International conference on machine learning*, pages 10347–10357. PMLR, 2021a.

[^66]: Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going deeper with image transformers. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 32–42, 2021b.

[^67]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. *Advances in neural information processing systems*, 30, 2017.

[^68]: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 568–578, 2021.

[^69]: Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. *arXiv preprint arXiv:2211.05778*, 2022a.

[^70]: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan, Kaitao Song, Ding Liang, Tong Lu, Ping Luo, and Ling Shao. Pvt v2: Improved baselines with pyramid vision transformer. *Computational Visual Media*, 8(3):415–424, 2022b.

[^71]: Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He. Non-local neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 7794–7803, 2018.

[^72]: Ross Wightman. Pytorch image models. [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models), 2019.

[^73]: Ross Wightman, Hugo Touvron, and Hervé Jégou. Resnet strikes back: An improved training procedure in timm. *arXiv preprint arXiv:2110.00476*, 2021.

[^74]: Tete Xiao, Yingcheng Liu, Bolei Zhou, Yuning Jiang, and Jian Sun. Unified perceptual parsing for scene understanding. In *Proceedings of the European conference on computer vision (ECCV)*, pages 418–434, 2018.

[^75]: Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 1492–1500, 2017.

[^76]: Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, and Jianfeng Gao. Focal self-attention for local-global interactions in vision transformers. *arXiv preprint arXiv:2107.00641*, 2021.

[^77]: Jianwei Yang, Chunyuan Li, and Jianfeng Gao. Focal modulation networks. *arXiv preprint arXiv:2203.11926*, 2022.

[^78]: Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, and Shuicheng Yan. Metaformer is actually what you need for vision. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10819–10829, 2022.

[^79]: Weihao Yu, Chenyang Si, Pan Zhou, Mi Luo, Yichen Zhou, Jiashi Feng, Shuicheng Yan, and Xinchao Wang. Metaformer baselines for vision. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024.

[^80]: Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 558–567, 2021.

[^81]: Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 6023–6032, 2019.

[^82]: Sergey Zagoruyko and Nikos Komodakis. Diracnets: Training very deep neural networks without skip-connections. *arXiv preprint arXiv:1706.00388*, 2017.

[^83]: Hongyi Zhang, Moustapha Cissé, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In *6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings*. OpenReview.net, 2018a.

[^84]: Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.

[^85]: Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, and Jian Sun. Shufflenet: An extremely efficient convolutional neural network for mobile devices. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6848–6856, 2018b.

[^86]: Hengshuang Zhao, Jiaya Jia, and Vladlen Koltun. Exploring self-attention for image recognition. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10076–10085, 2020.

[^87]: Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In *Proceedings of the AAAI conference on artificial intelligence*, pages 13001–13008, 2020.

[^88]: Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 633–641, 2017.