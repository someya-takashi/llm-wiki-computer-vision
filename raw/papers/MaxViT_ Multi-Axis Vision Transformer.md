---
title: "MaxViT: Multi-Axis Vision Transformer"
source: "https://ar5iv.labs.arxiv.org/html/2204.01697"
author:
published:
created: 2026-08-17
description: "Transformers have recently gained significant attention in the computer vision community.However, the lack of scalability of self-attention mechanisms with respect to image size has limited their wide adoption in stat…"
tags:
  - "clippings"
---
Zhengzhong Tu Affiliation: Google Research Affiliation: University of Texas at Austin    Hossein Talebi Affiliation: Google Research    Han Zhang Affiliation: Google Research    Feng Yang Affiliation: Google Research       Peyman Milanfar Affiliation: Google Research    Alan Bovik Affiliation: University of Texas at Austin    Yinxiao Li Affiliation: Google Research

###### Abstract

Transformers have recently gained significant attention in the computer vision community. However, the lack of scalability of self-attention mechanisms with respect to image size has limited their wide adoption in state-of-the-art vision backbones. In this paper we introduce an efficient and scalable attention model we call multi-axis attention, which consists of two aspects: blocked local and dilated global attention. These design choices allow global-local spatial interactions on arbitrary input resolutions with only linear complexity. We also present a new architectural element by effectively blending our proposed attention model with convolutions, and accordingly propose a simple hierarchical vision backbone, dubbed MaxViT, by simply repeating the basic building block over multiple stages. Notably, MaxViT is able to “see” globally throughout the entire network, even in earlier, high-resolution stages. We demonstrate the effectiveness of our model on a broad spectrum of vision tasks. On image classification, MaxViT achieves state-of-the-art performance under various settings: without extra data, MaxViT attains 86.5% ImageNet-1K top-1 accuracy; with ImageNet-21K pre-training, our model achieves 88.7% top-1 accuracy. For downstream tasks, MaxViT as a backbone delivers favorable performance on object detection as well as visual aesthetic assessment. We also show that our proposed model expresses strong generative modeling capability on ImageNet, demonstrating the superior potential of MaxViT blocks as a universal vision module. The source code and trained models will be available at [https://github.com/google-research/maxvit](https://github.com/google-research/maxvit).

###### Keywords:

Transformer, Image classification, Multi-axis attention.

## 1 Introduction

Convolutional Neural Networks (ConvNets) have been the dominant architectural design choice for computer vision [^48] [^29] [^75] [^76] since AlexNet [^48]. ConvNets continue to excel on numerous vision problems by going deeper [^75], wider [^76] [^74], adding dense connections [^37], efficient separable convolutions [^35] [^70], atrous convolutions [^9], using encoder-decoder frameworks [^67], and even introducing modern micro-design components [^57]. Meanwhile, as inspired by the evolution of self-attention models like Transformers [^85] in natural language processing [^20] [^100] [^49] [^63], numerous researchers have started to introduce attention mechanisms into vision [^88] [^6]. The Vision Transformer (ViT) [^22] is perhaps the first fully Transformer-based architecture for vision, whereby image patches are simply regarded as sequences of words and a transformer encoder is applied on these visual tokens. When pre-trained on large-scale datasets [^73], ViT can achieve compelling results on image recognition.

However, it has been observed that without extensive pre-training [^22] [^81] ViT underperforms on image recognition. This is due to the strong model capacity of Transformers, that is imbued with less inductive bias, which leads to overfitting. To properly regularize the model capacity and improve its scalability, numerous subsequent efforts have studied sparse Transformer models tailored for vision tasks such as local attention [^56] [^99] [^50] [^16]. These methods typically re-introduce hierarchical architectures to compensate for the loss of non-locality. The Swin Transformer [^56] is one such successful attempt to modify Transformers by applying self-attention on shifted non-overlapping windows. For the first time, this approach outperformed ConvNets on the ImageNet benchmark with a pure vision Transformer. Despite having more flexibility and generalizability than the full attention used in ViT, window-based attention has been observed to have limited model capacity due to the loss of non-locality, and henceforth scales unfavorably on larger data regimes such as ImageNet-21K and JFT [^19]. However, acquiring global interactions via full-attention at early or high-resolution stages in a hierarchical network is computationally heavy, as the attention operator requires quadratic complexity. How to efficiently incorporate global and local interactions to balance the model capacity and generalizability under a computation budget still remains challenging.

(a) Accuracy *vs.* FLOPs performance scaling curve under ImageNet-1K training setting at input resolution 224 $\times$ 224.

(b) Accuracy *vs.* Parameters scaling curve under ImageNet-1K fine-tuning setting allowing for higher sizes (384/512).

Figure 1: Performance comparison of MaxViT with state-of-the-art vision Transformers on ImageNet-1K. Our model shows superior performance in terms of both accuracy *vs.* computation and accuracy *vs.* parameters tradeoff.

In this paper, we present a new type of Transformer module, called multi-axis self-attention (Max-SA), that capably serves as a basic architecture component which can perform both local and global spatial interactions in a single block. Compared to full self-attention, Max-SA enjoys greater flexibility and efficiency, *i.e.*, naturally adaptive to different input lengths with linear complexity; in contrast to (shifted) window/local attention, Max-SA allows for stronger model capacity by proposing a global receptive field. Moreover, with merely linear complexity, Max-SA can be used as a general stand-alone attention module in any layer of a network, even in earlier, high-resolution stages.

To demonstrate its effectiveness and universality, we further design a simple but effective vision backbone called Multi-axis Vision Transformer (MaxViT) by hierarchically stacking repeated blocks composed of Max-SA and convolutions. While our proposed model belongs to the category of hybrid vision Transformers, MaxViT distinguishes from previous approaches [^94] [^19] in that we strive for simplicity, by designing a basic block unifying convolution, local, and global attention, then simply repeating it. Our experiments shows that the MaxViT significantly improves upon state-of-the-art (SOTA) performance under all data regimes for a broad range of visual tasks including classification, object detection and segmentation, image aesthetics assessment, and image generation. Specifically, as Figure 1 shows, MaxViT outperforms all recent Transformer-based models in regards to both accuracy *vs.* FLOPs and accuracy *vs.* parameter curves. Our contributions are:

- A generic strong Transformer backbone, MaxViT, that can capture both local and global spatial interactions throughout every stage of the network.
- A novel stand-alone multi-axis attention module composed of blocked local and dilated global attention, enjoying global perception in linear complexity.
- We demonstrate large amounts of design choices including number of layers, layouts, the use of MBConv, *etc.* with extensive ablation studies, that eventually converge towards our final modular design, the MaxViT-Block.
- Our extensive experiments show that MaxViT achieves SOTA results under various data regimes for a broad range of tasks including image classification, object detection, image aesthetic assessment, and image generation.

## 2 Related work

Convolutional networks. Since AlexNet [^48], convolutional neural networks (ConvNets) have been used as de facto solutions to almost all vision tasks [^29] [^8] [^37] [^104] [^13] [^51] [^90] [^78] [^89] before the “Roaring 20s” [^57]. Phenomenal architectural improvements have been made in the past decade: residual [^29] and dense connections [^37], fully-convolutional networks [^58], encoder-decoder schemes [^67], feature pyramids [^52], increased depths and widths [^75], spatial- and channel-wise attention models [^36] [^91], non-local interactions [^88], to name a few. A remarkable recent work ConvNeXt [^57] has re-introduced core designs of vision Transformers and shown that a ‘modernized’ pure ConvNet can achieve performance comparable to Transformers on broad vision tasks.

Transformers in vision. Transformers were originally proposed for natural language processing [^85]. The debut of the Vision Transformer (ViT) [^22] in 2020 showed that pure Transformer-based architectures are also effective solutions for vision problems. The elegantly novel view of ViT that treats image patches as visual words has stimulated explosive research interest in visual Transformers. To account for locality and 2D nature of images, the Swin Transformer aggregates attention in shifted windows in a hierarchical architecture [^56]. More recent works have been focused on improving model and data efficiency, including sparse attention [^21] [^99] [^64] [^86] [^96] [^1], improved locality [^101] [^27], pyramidal designs [^87] [^24] [^97], improved training strategies [^81] [^82] [^105] [^3], *etc.* We refer readers to dedicated surveys [^44] [^44] of vision Transformers for a comprehensive review.

Hybrid models. Pure Transformer-based vision models have been observed to generalize poorly due to relatively less inductive bias [^22] [^19] [^81]. Vision Transformers also exhibit substandard optimizability [^94]. An intriguingly simple improvement is to adopt a hybrid design of Transformer and convolution layers such as using a few convolutions to replace the coarse patchify stem [^94] [^19]. A broad range of works fall into this category, either explicitly hybridized [^93] [^23] [^19] [^94] [^24] [^98] [^4] or in an implicit fashion [^56] [^16].

Transformer for GANs. Transformers have also proven effective in generative adversarial networks (GANs) [^26]. TransGAN [^40] built a pure Transformer GAN with a careful design of local attention and upsampling layers, demonstrating effectiveness on small scale datasets [^47] [^18]. GANformer [^38] explored efficient global attention mechanisms to improve on StyleGAN [^42] generator. HiT [^103] presents an efficient Transformer generator based on local-global attention that can scale up to 1K high-resolution image generation.

## 3 Method

Inspired by the sparse approaches presented in [^103] [^83], we introduce a new type of attention module, dubbed blocked multi-axis self-attention (Max-SA), by decomposing the fully dense attention mechanisms into two sparse forms – window attention and grid attention – which reduces the quadratic complexity of vanilla attention to linear, without any loss of non-locality. Our sequential design offers greater simplicity and flexibility, while performing even better than previous methods – each individual module can be used either standalone or combined in any order (Tables 10-10), whereas parallel designs [^103] [^83] offer no such benefits. Because of the flexibility and scalability of Max-SA, we are able to build a novel vision backbone, which we call MaxViT, by simply stacking alternative layers of Max-SA with MBConv [^35] in a hierarchical architecture, as shown in Figure 2. MaxViT benefits from global and local receptive fields throughout the entire network, from shallow to deep stages, demonstrating superior performance in regards to both model capacity and generalization abilities.

Figure 2: MaxViT architecture. We follow a typical hierarchical design of ConvNet practices (e.g., ResNet) but instead build a new type of basic building block that unifies MBConv, block, and grid attention layers. Normalization and activation layers are omitted for simplicity.

### 3.1 Attention

Self-attention allows for spatial mixing of entire spatial (or sequence) locations while also benefiting from content-dependent weights based on normalized pairwise similarity. The standard self-attention defined in [^85] [^22] is location-unaware, *i.e.*, non-translation equivariant, an important inductive bias imbued in ConvNets. Relative self-attention [^56] [^19] [^71] [^40] has been proposed to improve on vanilla attention by introducing a relative learned bias added to the attention weights, which has been shown to consistently outperform original attention on many vision tasks [^56] [^19] [^40]. In this work, we mainly adopt the pre-normalized relative self-attention defined in [^19] as the key operator in MaxViT.

### 3.2 Multi-axis Attention

Global interaction is one of the key advantages of self-attention as compared to local convolution. However, directly applying attention along the entire space is computationally infeasible as the attention operator requires quadratic complexity. To tackle this problem, we present a multi-axis approach to decompose the full-size attention into two sparse forms – local and global – by simply decomposing the spatial axes. Let $X\in\mathbb{R}^{H\times W\times C}$ be an input feature map. Instead of applying attention on the flattened spatial dimension $HW$, we block the feature into a tensor of shape $(\frac{H}{P}\times\frac{W}{P},P\times P,C)$, representing partitioning into non-overlapping windows, each of size $P\times P$. Applying self-attention on the local spatial dimension *i.e.*, $P\times P$, is equivalent to attending within a small window [^56]. We will use this block attention to conduct local interactions.

Figure 3: Multi-axis self-attention (Max-SA) (best viewed in color). An illustration of the multi-axis approach for computing self-attention (window/grid size is 4 $\times$ 4). The block-attention module performs self-attention within windows, while the grid-attention module attends globally to pixels in a sparse, uniform grid overlaid on the entire 2D space, with both having linear complexity against input size, as we use fixed attention footage. The same colors are spatially mixed by the self-attention operation.

Despite bypassing the notoriously heavy computation of full self-attention, local-attention models have been observed to underfit on huge-scale datasets [^19] [^22]. Inspired by block attention, we present a surprisingly simple but effective way to gain sparse global attention, which we call grid attention. Instead of partitioning feature maps using fixed window size, we grid the tensor into the shape $(G\times G,\frac{H}{G}\times\frac{W}{G},C)$ using a fixed $G\times G$ uniform grid, resulting in windows having adaptive size $\frac{H}{G}\times\frac{W}{G}$. Employing self-attention on the decomposed grid axis *i.e.*, $G\times G$, corresponds to dilated, global spatial mixing of tokens. By using the same fixed window and grid sizes (we use $P=G=7$ following Swin [^56]), we can fully balance the computation between local and global operations, both having only linear complexity with respect to spatial size or sequence length. Note that our proposed Max-SA module can be a drop-in replacement of the Swin attention module [^56] with exactly the same number of parameters and FLOPs. Yet it enjoys global interaction capability without requiring masking, padding, or cyclic-shifting, making it more implementation friendly, preferable to the shifted window scheme [^56]. For instance, the multi-axis attention can be easily implemented with einops [^66] without modifying the original attention operation (see Appendix). It is worth mentioning that our proposed multi-axis attention (Max-SA) is fundamentally different from the axial-attention models [^86] [^33]. Please see Appendix for a detailed comparison.

MaxViT block. We sequentially stack the two types of attentions to gain both local and global interactions in a single block, as shown in Figure 3. Note that we also adopt typical designs in Transformers [^22] [^56], including LayerNorm [^2], Feedforward networks (FFNs) [^22] [^56], and skip-connections. We also add a MBConv block [^35] with squeeze-and-excitation (SE) module [^36] prior to the multi-axis attention, as we have observed that using MBConv together with attention further increases the generalization as well as the trainability of the network [^94]. Using MBConv layers prior to attention offers another advantage, in that depthwise convolutions can be regarded as conditional position encoding (CPE) [^17], making our model free of explicit positional encoding layers. Note that our proposed stand-alone multi-axis attention may be used together or in isolation for different purposes – block attention for local interaction, and grid attention for global mixing. These elements can be easily plugged into many vision architectures, especially on high-resolution tasks that can benefit by global interactions with affordable computation.

### 3.3 Architecture Variants

We designed a series of extremely simple architectural variants to explore the effectiveness of our proposed MaxViT block, as shown in Figure 2. We use a hierarchical backbone similar to common ConvNet practices [^29] [^57] [^19] [^80] where the input is first downsampled using Conv3x3 layers in stem stage (S0). The body of the network contains four stages (S1-S4), with each stage having half the resolution of the previous one with a doubled number of channels (hidden dimension). In our network, we employ identical MaxViT blocks throughout the entire backbone. We apply downsampling in the Depthwise Conv3x3 layer of the first MBConv block in each stage. The expansion and shrink rates for inverted bottleneck [^35] and squeeze-excitation (SE) [^36] are 4 and 0.25 by default. We set the attention head size to be 32 for all attention blocks. We scale up the model by increasing block numbers per stage $B$ and the channel dimension $C$. We summarize the architectural configurations of the MaxViT variants in Table 1.

Table 1: MaxViT architecture variants. B and C denotes number of blocks and number of channels for each stage. We set each attention head to 32 for all attention layers. For MBConv, we always use expansion rate 4 and shrinkage rate 0.25 in SE [^36], following [^79] [^80] [^19]. We use two Conv layers in the stem.

| Stage | Size | MaxViT-T | MaxViT-S | MaxViT-B | MaxViT-L | MaxViT-XL |
| --- | --- | --- | --- | --- | --- | --- |
| S0: Conv-stem | $\nicefrac{{1}}{{2}}$ | B=2 C=64 | B=2 C=64 | B=2 C=64 | B=2 C=128 | B=2 C=192 |
| S1: MaxViT-Block | $\nicefrac{{1}}{{4}}$ | B=2 C=64 | B=2 C=96 | B=2 C=96 | B=2 C=128 | B=2 C=192 |
| S2: MaxViT-Block | $\nicefrac{{1}}{{8}}$ | B=2 C=128 | B=2 C=192 | B=6 C=192 | B=6 C=256 | B=6 C=384 |
| S3: MaxViT-Block | $\nicefrac{{1}}{{16}}$ | B=5 C=256 | B=5 C=384 | B=14 C=384 | B=14 C=512 | B=14 C=768 |
| S4: MaxViT-Block | $\nicefrac{{1}}{{32}}$ | B=2 C=512 | B=2 C=768 | B=2 C=768 | B=2 C=1024 | B=2 C=1536 |

## 4 Experiments

We validated the efficacy of our proposed model on various vision tasks: ImageNet classification [^48], image object detection and instance segmentation [^53], image aesthetics/quality assessment [^61], and unconditional image generation [^26]. More experimental details can be found in the Appendix.

Table 2: Performance comparison under ImageNet-1K setting. Throughput is measured on a single V100 GPU with batch size 16, following [^56] [^80] [^57].

<table><thead><tr><th></th><th>Model</th><th>Eval size</th><th>Params</th><th>FLOPs</th><th>Throughput (image/s)</th><th>IN-1K top-1 acc.</th></tr></thead><tbody><tr><th rowspan="10">ConvNets</th><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B6 <sup><a href="#fn:79">79</a></sup></th><th>528</th><td>43M</td><td>19.0G</td><td>96.9</td><td>84.0</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B7 <sup><a href="#fn:79">79</a></sup></th><th>600</th><td>66M</td><td>37.0G</td><td>55.1</td><td>84.3</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> RegNetY-16 <sup><a href="#fn:62">62</a></sup></th><th>224</th><td>84M</td><td>16.0G</td><td>334.7</td><td>82.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F0 <sup><a href="#fn:5">5</a></sup></th><th>256</th><td>72M</td><td>12.4G</td><td>533.3</td><td>83.6</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F1 <sup><a href="#fn:5">5</a></sup></th><th>320</th><td>132M</td><td>35.5G</td><td>228.5</td><td>84.7</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-S <sup><a href="#fn:80">80</a></sup></th><th>384</th><td>24M</td><td>8.8G</td><td>666.6</td><td>83.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-M <sup><a href="#fn:80">80</a></sup></th><th>480</th><td>55M</td><td>24.0G</td><td>280.7</td><td>85.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-S <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>50M</td><td>8.7G</td><td>447.1</td><td>83.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-B <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>89M</td><td>15.4G</td><td>292.1</td><td>83.8</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-L <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>198M</td><td>34.4G</td><td>146.8</td><td>84.3</td></tr><tr><th rowspan="13">ViTs</th><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/32 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>86M</td><td>55.4G</td><td>85.9</td><td>77.9</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>307M</td><td>190.7G</td><td>27.3</td><td>76.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeiT-B <sup><a href="#fn:81">81</a></sup></th><th>384</th><td>86M</td><td>55.4G</td><td>85.9</td><td>83.1</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CaiT-M24 <sup><a href="#fn:82">82</a></sup></th><th>224</th><td>186M</td><td>36.0G</td><td>-</td><td>83.4</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CaiT-M24 <sup><a href="#fn:82">82</a></sup></th><th>384</th><td>186M</td><td>116.1G</td><td>-</td><td>84.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeepViT-L <sup><a href="#fn:105">105</a></sup></th><th>224</th><td>55M</td><td>12.5G</td><td>-</td><td>83.1</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> T2T-ViT-24 <sup><a href="#fn:101">101</a></sup></th><th>224</th><td>64M</td><td>15.0G</td><td>-</td><td>82.6</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-S <sup><a href="#fn:56">56</a></sup></th><th>224</th><td>50M</td><td>8.7G</td><td>436.9</td><td>83.0</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-B <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>88M</td><td>47.0G</td><td>84.7</td><td>84.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-B <sup><a href="#fn:21">21</a></sup></th><th>224</th><td>78M</td><td>15.0G</td><td>250</td><td>84.2</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-B <sup><a href="#fn:21">21</a></sup></th><th>384</th><td>78M</td><td>47.0G</td><td>-</td><td>85.4</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Focal-S <sup><a href="#fn:99">99</a></sup></th><th>224</th><td>51M</td><td>9.1G</td><td>-</td><td>83.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Focal-B <sup><a href="#fn:99">99</a></sup></th><th>224</th><td>90M</td><td>16.0G</td><td>-</td><td>83.8</td></tr><tr><th rowspan="17">Hybrid</th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-21 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>32M</td><td>24.9G</td><td>-</td><td>83.3</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-2 <sup><a href="#fn:19">19</a></sup></th><th>224</th><td>75M</td><td>15.7G</td><td>247.7</td><td>84.1</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>224</th><td>168M</td><td>34.7G</td><td>163.3</td><td>84.5</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>168M</td><td>107.4G</td><td>48.5</td><td>85.8</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>168M</td><td>203.1G</td><td>22.4</td><td>86.0</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>224</th><td>31M</td><td>5.6G</td><td>349.6</td><td>83.62</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>224</th><td>69M</td><td>11.7G</td><td>242.5</td><td>84.45</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>224</th><td>120M</td><td>23.4G</td><td>133.6</td><td>84.95</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>224</th><td>212M</td><td>43.9G</td><td>99.4</td><td>85.17</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>384</th><td>31M</td><td>17.7G</td><td>121.9</td><td>85.24</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>384</th><td>69M</td><td>36.1G</td><td>82.7</td><td>85.74</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>384</th><td>120M</td><td>74.2G</td><td>45.8</td><td>86.34</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>384</th><td>212M</td><td>133.1G</td><td>34.3</td><td>86.40</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>512</th><td>31M</td><td>33.7G</td><td>63.8</td><td>85.72</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>512</th><td>69M</td><td>67.6G</td><td>43.3</td><td>86.19</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>512</th><td>120M</td><td>138.5G</td><td>24.0</td><td>86.66</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>512</th><td>212M</td><td>245.4G</td><td>17.8</td><td>86.70</td></tr></tbody></table>

Table 3: Performance comparison for large-scale data regimes: ImageNet-21K and JFT pretrained models.

<table><thead><tr><th></th><th rowspan="2">Model</th><th rowspan="2">Eval size</th><th rowspan="2">Params</th><th rowspan="2">FLOPs</th><th colspan="2">IN-1K top-1 acc.</th></tr><tr><th></th><th>21K <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 1K</th><th>JFT <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 1K</th></tr></thead><tbody><tr><th rowspan="7">ConvNets</th><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> BiT-R-101x3 <sup><a href="#fn:46">46</a></sup></th><th>384</th><td>388M</td><td>204.6G</td><td>84.4</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> BiT-R-152x4 <sup><a href="#fn:46">46</a></sup></th><th>480</th><td>937M</td><td>840.5G</td><td>85.4</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-L <sup><a href="#fn:80">80</a></sup></th><th>480</th><td>121M</td><td>53.0G</td><td>86.8</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-XL <sup><a href="#fn:80">80</a></sup></th><th>512</th><td>208M</td><td>94.0G</td><td>87.3</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-L <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>198M</td><td>101.0G</td><td>87.5</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-XL <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>350M</td><td>179.0G</td><td>87.8</td><td>-</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F4+ <sup><a href="#fn:5">5</a></sup></th><th>512</th><td>527M</td><td>367G</td><td>-</td><td>89.20</td></tr><tr><th rowspan="7">ViTs</th><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>87M</td><td>55.5G</td><td>84.0</td><td>-</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>305M</td><td>191.1G</td><td>85.2</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>512</th><td>305M</td><td>364G</td><td>-</td><td>87.76</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-H/14 <sup><a href="#fn:22">22</a></sup></th><th>518</th><td>632M</td><td>1021G</td><td>-</td><td>88.55</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> HaloNet-H4 <sup><a href="#fn:84">84</a></sup></th><th>512</th><td>85M</td><td>-</td><td>85.8</td><td>-</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> SwinV2-B <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>88M</td><td>-</td><td>87.1</td><td>-</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> SwinV2-L <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>197M</td><td>-</td><td>87.7</td><td>-</td></tr><tr><th rowspan="11">Hybrid</th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-W24 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>277M</td><td>193.2G</td><td>87.7</td><td>-</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> R+ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>330M</td><td>-</td><td>-</td><td>87.12</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>168M</td><td>107.4G</td><td>87.6</td><td>88.52</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>168M</td><td>214G</td><td>87.9</td><td>88.81</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-4 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>275M</td><td>360.9G</td><td>88.1</td><td>89.11</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-5 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>688M</td><td>812G</td><td>-</td><td>89.77</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>384</th><td>119M</td><td>74.2G</td><td>88.24</td><td>88.69</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>384</th><td>212M</td><td>128.7G</td><td>88.32</td><td>89.12</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-XL</th><th>384</th><td>475M</td><td>293.7G</td><td>88.51</td><td>89.36</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>512</th><td>119M</td><td>138.3G</td><td>88.38</td><td>88.82</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>512</th><td>212M</td><td>245.2G</td><td>88.46</td><td>89.41</td></tr><tr><th></th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-XL</th><th>512</th><td>475M</td><td>535.2G</td><td>88.70</td><td>89.53</td></tr></tbody></table>

### 4.1 Image Classification on ImageNet-1K

(a) Accuracy *vs.* Params performances for ImageNet-21K pre-trained models.

(b) Accuracy *vs.* Params scaling curve for JFT-300M pre-trained models.

Figure 4: Performance comparison on large-scale pre-trained models. MaxViT shows superior scaling performance under both ImageNet-21K and JFT-300M pre-trained settings.

ImageNet-1K. We show in Table 2 the performance comparisons on ImageNet-1K classification. Under the basic 224 $\times$ 224 setting, MaxViT outperformed the most recent strong hybrid model CoAtNet by a large margin across the entire FLOPs spectrum, as shown in Figure 1(a). The MaxViT-L model sets a new performance record of 85.17% at $224\times 224$ training without extra training strategies, outperforming CoAtNet-3 by 0.67%. In regards to throughput-accuracy trade-offs at $224^{2}$, MaxViT-S obtains 84.45% top-1 accuracy, 0.25% higher than CSWin-B and 0.35% higher than CoAtNet-2 with comparable throughput.

When fine-tuned at higher resolutions (384/512), MaxViT continues to deliver high performance compared to strong ConvNet and Transformer competitors: (1) at $384^{2}$, MaxViT-B attains 86.34% top-1 accuracy, outperforming EfficientNetV2-L by 0.64%; (2) when fine-tuned at $512^{2}$, our MaxViT-L (212M) achieves top-1 accuracy 86.7%, setting new SOTA performance on ImageNet-1K under the normal training setting. As Figure 1 shows, MaxViT scales much better than SOTA vision Transformers on the ImageNet-1K trained model scale.

ImageNet-21K. Table 3 shows the results of models pre-trained on ImageNet-21K. Remarkably, the MaxViT-B model achieves 88.38% accuracy, outperforming the previous best model CoAtNet-4 by 0.28% using only 43% of parameter count and 38% of FLOPs, demonstrating greater parameter and computing efficiency. Figure 4(a) visualizes the model size comparison – MaxViT scales significantly better than previous attention-based models of similar complexities, across the board. Additionally, the MaxViT-XL model achieves new SOTA performance, an accuracy of 88.70% when fine-tuned at resolution $512\times 512$.

JFT-300M. We also trained our model on a larger-scale proprietary dataset JFT-300M which contains $\sim$ 300 million weakly labeled images. As shown in Table 3 and Figure 4(b), our model is also scalable to massive scale training data – MaxViT-XL achieves a high accuracy of 89.53% with 475 million parameters, outperforming previous models under comparable model sizes. Due to resource limitations, we leave experiments on billion-parameter-scale models on planet-scale datasets (*e.g.*, JFT-3B [^102]) as future work.

### 4.2 Object Detection and Instance Segmentation

Setting. We evaluated the MaxViT architectures on the COCO2017 [^53] object bounding box detection and instance segmentation tasks with a two-stage framework [^65]. On the object detection task, a feature-pyramid architecture [^52] was employed to boost different levels of objectiveness. In the instance segmentation task, a well-known Cascade Mask-RCNN framework [^28] was employed. The dataset contains 118K training and 5K validation samples. For all the compared models, the backbones are first pretrained using ImageNet-1K. The pretrained models are then used to finetune on the detection and segmentation tasks.

Results on COCO. As shown in Table 4, $AP$, $AP_{50}$, and $AP_{75}$ are reported for comparison. The parameters and FLOPs are also reported as a reference for model complexity. The MaxViT backbone models, used in object detection and segmentation tasks, outperform all other backbones by large margins, including Swin, ConvNeXt, and UViT at various model sizes with respect to both accuracy and efficiency. Note that MaxViT-S outperforms other base-level models (*e.g.*, Swin-B, UViT-B), with about 40% less computational cost.

Table 4: Comparison of two-stage object detection and instance segmentation on COCO2017. All models are pretrained on ImageNet-1K.

| Backbone | Resolution | AP | AP <sub>50</sub> | AP <sub>75</sub> | AP <sup>m</sup> | AP ${}^{m}_{50}$ | AP ${}^{m}_{75}$ | FLOPs | Pars. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| $\bullet$ ResNet-50 [^29] | 1280 $\times$ 800 | 46.3 | 64.3 | 50.5 | 40.1 | 61.7 | 43.4 | 739G | 82M |
| $\bullet$ X101-32 [^95] | 1280 $\times$ 800 | 48.1 | 66.5 | 52.4 | 41.6 | 63.9 | 45.2 | 819G | 101M |
| $\bullet$ X101-64 [^95] | 1280 $\times$ 800 | 48.3 | 66.4 | 52.3 | 41.7 | 64.0 | 45.1 | 972G | 140M |
| $\bullet$ ConvNeXt-T [^57] | 1280 $\times$ 800 | 50.4 | 69.1 | 54.8 | 43.7 | 66.5 | 47.3 | 741G | \- |
| $\bullet$ ConvNeXt-S [^57] | 1280 $\times$ 800 | 51.9 | 70.8 | 56.5 | 45.0 | 68.4 | 49.1 | 827G | \- |
| $\bullet$ ConvNeXt-B [^57] | 1280 $\times$ 800 | 52.7 | 71.3 | 57.2 | 45.6 | 68.9 | 49.5 | 964G | \- |
| $\circ$ Swin-T [^56] | 1280 $\times$ 800 | 50.4 | 69.2 | 54.7 | 43.7 | 66.6 | 47.3 | 745G | 86M |
| $\circ$ Swin-S [^56] | 1280 $\times$ 800 | 51.9 | 70.7 | 56.3 | 45.0 | 68.2 | 48.8 | 838G | 107M |
| $\circ$ Swin-B [^56] | 1280 $\times$ 800 | 51.9 | 70.5 | 56.4 | 45.0 | 68.1 | 48.9 | 982G | 145M |
| $\circ$ UViT-T [^14] | 896 $\times$ 896 | 51.1 | 70.4 | 56.2 | 43.6 | 67.7 | 47.2 | 613G | 47M |
| $\circ$ UViT-S [^14] | 896 $\times$ 896 | 51.4 | 70.8 | 56.2 | 44.1 | 68.2 | 48.0 | 744G | 54M |
| $\circ$ UViT-B [^14] | 896 $\times$ 896 | 52.5 | 72.0 | 57.6 | 44.3 | 68.7 | 48.3 | 975G | 74M |
| $\circ$ As-ViT-L [^15] | 1024 $\times$ 1024 | 52.7 | 72.3 | 57.9 | 45.2 | 69.7 | 49.8 | 1094G | 139M |
| $\diamond$ MaxViT-T | 896 $\times$ 896 | 52.1 | 71.9 | 56.8 | 44.6 | 69.1 | 48.4 | 475G | 69M |
| $\diamond$ MaxViT-S | 896 $\times$ 896 | 53.1 | 72.5 | 58.1 | 45.4 | 69.8 | 49.5 | 595G | 107M |
| $\diamond$ MaxViT-B | 896 $\times$ 896 | 53.4 | 72.9 | 58.1 | 45.7 | 70.3 | 50.0 | 856G | 157M |

### 4.3 Image Aesthetic Assessment.

Setting. We train and evaluate the MaxViT model on the AVA benchmark [^61] which contains 255K images with aesthetics scores rated by amateur photographers. Similar to [^77], we split the dataset into 80%/20% training and test sets. We followed [^77] and used the normalized Earth Mover’s Distance as our training loss. We trained MaxViT at three different input resolutions: $224^{2}$, $384^{2}$ and $512^{2}$, initialized with ImageNet-1K pre-trained weights.

Results on AVA. To evaluate and compare our model against existing methods, we present a summary of our results in Table 6. For similar input resolutions, the proposed MaxViT-T model outperforms existing image aesthetic assessment methods. As the input resolution increases, the performance improves, benefiting from its strong non-local capacity. Also, MaxViT shows better linear correlation compared to the SOTA method [^43] which uses multi-resolution inputs.

### 4.4 Image Generation

Setting. We evaluate the generative ability of MaxViT blocks to generate images of 128x128 resolution on ImageNet-1K. We choose the unconditional image generation to focus on the performance of different generators in GANs. We use the Inception Score (IS) [^69] and the Fréchet Inception Distance (FID) [^32] as quantitative evaluation metrics. 50,000 samples were randomly generated to calculate the FID and IS scores. We compared MaxViT against HiT [^103], a SOTA generative Transformer model, which uses attention at low resolutions (e.g., 32, 64), and using implicit neural functions at high resolutions (e.g., 128). By contrast, MaxViT uses the proposed MaxViT block at every resolution. Note that we use an inverse block order (GA-BA-Conv) as we found it to perform better (see Table 10). Since Batch Normalization [^39] [^103] achieves better results on image generation, we replaced all Layer Norm with Batch Norm under this setting.

Results on ImageNet-1K. The results are shown in Table 6. Our MaxViT achieved better FID and IS with significantly lower number of parameters. These results demonstrate the effectiveness of MaxViT blocks for generation tasks. More details of the generative experiment can be found in Appendix.

Table 5: Image aesthetic assessment results on the AVA benchmark [^61]. PLCC and SRCC represent the Pearson’s linear and Spearman’s rank correlation coefficients.

| Model | Res. | Pars. | PLCC $\uparrow$ | SRCC $\uparrow$ |
| --- | --- | --- | --- | --- |
| $\bullet$ NIMA [^77] | $224$ | 56M | 0.636 | 0.612 |
| $\bullet$ EffNet-B0 [^79] | $224$ | 5.3M | 0.642 | 0.620 |
| $\bullet$ AFDC [^10] | $224$ | 44.5M | 0.671 | 0.649 |
| $\circ$ ViT-S/32 [^43] | $384$ | 22M | 0.665 | 0.656 |
| $\circ$ ViT-B/32 [^43] | $384$ | 88M | 0.664 | 0.664 |
| $\circ$ MUSIQ [^43] | $224\!\sim\!512$ | 27M | 0.720 | 0.706 |
| $\diamond$ MaxViT-T | $224$ | 31M | 0.707 | 0.685 |
| $\diamond$ MaxViT-T | $384$ | 31M | 0.736 | 0.699 |
| $\diamond$ MaxViT-T | $512$ | 31M | 0.745 | 0.708 |

Table 6: Comparison of image generation on ImageNet. $\ddagger$ used a pre-trained ImageNet classifier.

| Model | FID $\downarrow$ | IS $\uparrow$ |
| --- | --- | --- |
| $\bullet$ GAN [^26] | 54.17 | 14.01 |
| $\bullet$ PacGAN2 [^54] | 57.51 | 13.50 |
| $\bullet$ MGAN [^34] | 50.90 | 14.44 |
| $\bullet$ LogoGAN [^68] $\ddagger$ | 38.41 | 18.86 |
| $\bullet$ SS-GAN [^12] | 43.87 | \- |
| $\bullet$ SC GAN [^55] | 40.30 | 15.82 |
| $\bullet$ ConvNet- $R_{1}$ [^103] | 37.18 | 19.55 |
| $\circ$ HiT [^103] (32.9M) | 30.83 | 21.64 |
| $\diamond$ MaxViT (18.6M) | 30.77 | 22.58 |

### 4.5 Ablation Studies.

In this section, we ablate important design choices in MaxViT on ImageNet-1K image classification. We use the MaxViT-T model trained for 300 epochs by default and report top-1 accuracy on ImageNet-1K. Except for the ablated design choice, we used the same training configurations, unless stated otherwise.

Table 7: Effects of global grid-attention. Ablate-S1 means we remove grid-attention in stage 1 while Replace-S1 means replacing grid-attention with block-attention.

| Model | Pars. | FLOPs | Top-1 Acc. |
| --- | --- | --- | --- |
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| Ablate-S1 | 30.8M | 5.3G | 83.36(-0.26) |
| Ablate-S2 | 30.5M | 5.3G | 83.38(-0.24) |
| Ablate-S3 | 26.9M | 4.9G | 83.00(-0.62) |
| Replace-S1 | 30.9M | 5.6G | 83.49(-0.13) |
| Replace-S2 | 30.9M | 5.6G | 83.41(-0.22) |
| Replace-S3 | 30.9M | 5.6G | 83.40(-0.23) |

Table 8: Block order study. C, BA, GA represent MBConv, block-, and grid-attention respectively.

<table><tbody><tr><td>Model</td><td>Pars.</td><td>FLOPs</td><td>Top-1 acc.</td></tr><tr><td>C-BA-GA</td><td>30.9M</td><td>5.6G</td><td>83.62</td></tr><tr><td>C-GA-BA</td><td>30.9M</td><td>5.6G</td><td>83.54(-0.08)</td></tr><tr><td>BA-C-GA</td><td>31.1M</td><td>5.3G</td><td>83.07(-0.55)</td></tr><tr><td>BA-GA-C</td><td>31.1M</td><td>5.3G</td><td>83.02(-0.60)</td></tr><tr><td>GA-C-BA</td><td>31.1M</td><td>5.3G</td><td>83.08(-0.54)</td></tr><tr><td>GA-BA-C</td><td>31.1M</td><td>5.3G</td><td>83.03(-0.59)</td></tr><tr><td colspan="4">GAN experiments</td></tr><tr><td>Model</td><td>Pars.</td><td>FID <math><semantics><mo>↓</mo> <annotation>\downarrow</annotation></semantics></math></td><td>IS <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>GA-BA-C</td><td>18.6M</td><td>30.77</td><td>22.68</td></tr><tr><td>C-BA-GA</td><td>18.6M</td><td>31.40</td><td>21.49(-1.19)</td></tr></tbody></table>

Table 9: Ablation of MBConv. Ablate-S1 means we delete MBConv layers in stage 1. Note that the network will also be smaller if we ablate MBConv layers in some stage.

| Model | Pars. | FLOPs | Top-1 acc. |
| --- | --- | --- | --- |
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| Ablate-S1 | 30.8M | 5.2G | 83.24(-0.38) |
| Ablate-S2 | 30.5M | 5.4G | 83.02(-0.60) |
| Ablate-S3 | 27.6M | 5.1G | 82.65(-0.97) |
| Ablate-S4 | 25.7M | 5.4G | 83.09(-0.53) |

Table 10: Sequential *vs.* parallel. We compared our model with modified parallel multi-axis scheme $Paral$ - $\star$.

| Model | Pars. | FLOPs | Top-1 acc. |
| --- | --- | --- | --- |
| MaxViT-T | 30.9M | 5.6G | 83.62 |
| $Paral$ -T | 34.5M | 6.2G | 82.64(-0.98) |
| MaxViT-S | 68.9M | 11.7G | 84.45 |
| $Paral$ -S | 76.9M | 13.0G | 83.45(-1.00) |
| MaxViT-B | 119.4M | 24.2G | 84.95 |
| $Paral$ -B | 133.4M | 26.9G | 83.70(-1.25) |
| MaxViT-L | 211.8M | 43.9G | 85.17 |
| $Paral$ -L | 236.6M | 48.8G | 83.54(-1.63) |

Global grid-attention. One of our main contributions is the grid-attention module, which allows for sparse global interactions at linear time, enabling our model to capture global information at all stages. We conducted two ablations to understand its gain: 1) completely removed global attention at each stage; 2) replaced grid attention with block attention to retain the same parameter count and FLOPs. As Table 10 shows, enabling global attention at earlier stages can further boost performance over using only local attention or convolutions.

MBConv layer. We also ablated the usage of MBConv layers in MaxViT by removing all MBConv in each stage. Note that we should also consider the reduction of parameter count and FLOPs when removing the MBConv layers. Plus, Stage 3 has 5 blocks whereas other stages have only 2. As Table 10 shows, the usage of MBConv layers in MaxViT significantly boosts performance.

Block order study. We present three different modules to build the MaxViT block – MBConv, block-, and grid-attention – which captures spatial interactions from local to global. To investigate the most effective way to combine them, we evaluated the MaxViT-T model using all 6 permutations. We always apply downsampling in the first layer, which might cause a minor model size difference. We can observe from Table 10 that placing MBConv before attention layers is almost always better than other combinations. The reason might be that it is more suitable to get local features/patterns in early layers, then aggregate them globally, which is aligned with existing hybrid models [^19] [^94], which puts Conv layers in front of attention. In generative experiments (Section 4.4), however, we found the best order to be from global to local: GA-BA-C. We hypothesize that it may be advantageous for generation tasks to first obtain the overall structures correct with global processing blocks (*i.e.*, grid-attention layers), then fill in finer details using local processing blocks (*i.e.*, MBConv).

Figure 5: Vertical layout ablation. Our model scales better than Swin layeout [^56].

Sequential *vs.* parallel. In our approach, we sequentially stack the multi-axis attention modules following [^56] [^86], while there also exist other models that adopt a parallel design [^103] [^83]. In this ablation, we compare our sequential Max-SA against parallel branches containing block- and grid-attention respectively. Note that we use an input projection to double the channels, then split the heads to feed the two branches in order to remain similar complexity to MaxViT, and an output projection that reduces the concatenated branches. We did rough parameter tuning and found that an initial learning rate of $10^{-3}$ performs significantly better than $3\times 10^{-3}$ for parallel models. We use all the same parameters except the learning rate. As Table 10 shows, our sequential approach remarkably outperforms parallel counterparts with fewer parameters and computation. The reason may be that the parallel designs learn complementary cues with less interactions between them, whereas our sequential stack is able to learn more powerful fusions between local and global layers.

Vertical layout. We further examine our vertical layout design, *i.e.*, the number of blocks each stage. We compared our design against the choice of Swin/ConvNeXt [^56] [^57]. We change MaxViT-T and -S to blocks $B=(2,2,6,2)$, and MaxViT-B, -L to have blocks $B=(2,2,18,2)$ strictly following the stage ratio of Swin [^56]. It may be seen from Figure 5 that our layout performed comparably to Swin for small models, but scales significantly better for larger models.

## 5 Discussion and Conclusion

While recent works in the 2020s have arguably shown that ConvNets and vision Transformers can achieve similar performance on image recognition, our work presents a unified design that takes advantages of the best of both worlds – efficient convolution and sparse attention – and demonstrates that a model built on top, namely MaxViT, can achieve state-of-the-art performance on a variety of vision tasks, and more importantly, scale extremely well to massive scale data sizes. Even though we present our model in the context of vision tasks, the proposed multi-axis approach can easily extend to language modeling to capture both local and global dependencies in linear time. We also look forward to studying other forms of sparse attention in higher-dimensional or multi-modal signals such as videos, point clouds, and vision-languages.

Societal impact. Investigating the performance and scalability of large model designs would consume considerable computing resources. These efforts can contribute to increased carbon emissions, which could hence raise environmental concerns. However, the proposed model offers strong modular candidates that expand the network’s design space for future efforts on automated architectural design. If trained improperly, the proposed model may express bias and fairness issues. The proposed generative model can be abused to generate misleading media and fake news. These issues demand caution in future related research.

Acknowledgment. We thank Xianzhi Du and Wuyang Chen for extensive help on experiments. We also thank Hanxiao Liu, Zihang Dai, Anurag Arnab, Huiwen Chang, Junjie Ke, Mauricio Delbracio, Sungjoon Choi, and Irene Zhu for valuable discussions and help.

## Appendix

In this Appendix we provide the following material:

- Sec. 0.A describes the detailed architectures of MaxViT for image classification (Sec. 0.A.1), object detection and segmentation (Sec. 0.A.2), image aesthetics assessment (Sec. 0.A.3), and image generation (Sec. 0.A.4).
- Sec. 0.B presents complete training settings and hyperparameters for image classification (Sec. 0.B.1), object detection and segmentation (Sec. 0.B.2), image aesthetics assessment (Sec. 0.B.3), and image generation (Sec. 0.B.4).
- Sec. 0.C demonstrates comprehensive experimental results, including image classification on ImageNet-1K (Table 13), ImageNet-21K and JFT (Table 14), as well as more image generation visualizations on ImageNet-1K (Figure 8).

## Appendix 0.A Model Details

### 0.A.1 Backbone Details

#### MBConv

MaxViT leverages the MBConv block [^70] [^79] as the main convolution operator. We also adopt a pre-activation structure [^30] [^19] to promote homogeneity between MBConv and Transformer blocks. Specifically, assume $\mathbf{x}$ to be the input feature, the MBConv block without downsampling is formulated as:

$$
\mathbf{x}\leftarrow\mathbf{x}+\mathsf{Proj}(\mathsf{SE}(\mathsf{DWConv}(\mathsf{Conv}(\mathsf{Norm}(\mathbf{x}))))),
$$

where $\mathsf{Norm}$ is $\mathsf{BatchNorm}$ [^39], $\mathsf{Conv}$ is the expansion Conv1x1 followed by $\mathsf{BatchNorm}$ and $\mathsf{GELU}$ [^31] activation, a typical choice for Transformer-based models. $\mathsf{DWConv}$ is the Depthwise Conv3x3 followed by $\mathsf{BatchNorm}$ and $\mathsf{GELU}$. $\mathsf{SE}$ is the Squeeze-Excitation layer [^36], while $\mathsf{Proj}$ is the shrink Conv1x1 to down-project the number of channels. Note that for the first MBConv block in every stage, the downsampling is done by applying stride-2 Depthwise Conv3x3 while the shortcut branch should also apply pooling and channel projection:

$$
\mathbf{x}\leftarrow\mathsf{Proj}(\mathsf{Pool2D}(\mathbf{x}))+\mathsf{Proj}(\mathsf{SE}(\mathsf{DWConv}\!\downarrow\!(\mathsf{Conv}(\mathsf{Norm}(\mathbf{x}))))).
$$

#### Relative Attention

Relative attention has been explored in several previous studies for both NLP [^71] [^92] and vision [^56] [^84] [^19] [^40]. Here to simplify the presentation, we present our model using only a single head of the multi-head self-attention. In the actual implementation, we always use multi-head attention with the same head dimension. The relative attention can be defined as:

$$
\mathsf{RelAttention}(Q,K,V)=\mathsf{softmax}(QK^{T}/\sqrt{d}+B)V,
$$

where $Q,K,V\in\mathbb{R}^{(H\times W)\times C}$ are the query, key, and value matrices and $d$ is the hidden dimension. The attention weights are co-decided by a learned static location-aware matrix $B$ and the scaled input-adaptive attention $QK^{T}/\sqrt{d}$. Considering the differences in 2D coordinates, the relative position bias $B$ is parameterized by a matrix $\hat{B}\in\mathbb{R}^{(2H-1)(2W-1)}$. Following typical practices [^56] [^19], when fine-tuned at a higher resolution *e.g.*, $H^{\prime}\times W^{\prime}$, we use bilinear interpolation to map the relative positional bias from $\mathbb{R}^{(2H-1)(2W-1)}$ to $\mathbb{R}^{(2H^{\prime}-1)(2W^{\prime}-1)}$. This relative attention benefits from input-adaptivity, translation equivariance, and global interactions, which is a preferred choice over the vanilla self-attention on 2D vision tasks. In our model, all the attention operators use this relative attention defined in Eq. 3 by default.

#### Multi-Axis Attention

We assume the relative attention operator in Eq. 3 follows the convention for 1D input sequences *i.e.*, always regards the second last dimension of an input $(...,L,C)$ as the spatial axis where ${L},C$ represent sequence length and channels. The proposed Multi-Axis Attention can be implemented without modification to the self-attention operation. To start with, we first define the $\mathsf{Block}(\cdot)$ operator with parameter $P$ as partitioning the input image/feature $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$ into non-overlapping blocks with each block having size $P\times P$. Note that after window partition, the block dimensions are gathered onto the spatial dimension (*i.e.*, -2 axis):

$$
\mathsf{Block}:(H,W,C)\rightarrow(\frac{H}{P}\times P,\frac{W}{P}\times P,C)\rightarrow(\frac{HW}{P^{2}},P^{2},C).
$$

We denote the $\mathsf{Unblock}(\cdot)$ operation as the reverse of the above block partition procedure. Similarly, we define the $\mathsf{Grid}(\cdot)$ operation with parameter $G$ as dividing the input feature into a uniform $G\times G$ grid, with each lattice having adaptive size $\frac{H}{G}\times\frac{W}{G}$. Unlike the $\mathsf{block}$ operator, we need to apply an extra $\mathsf{Transpose}$ to place the grid dimension in the assumed spatial axis (*i.e.*, -2 axis):

$$
\mathsf{Grid}:(H,W,C)\rightarrow(G\times\frac{H}{G},G\times\frac{W}{G},C)\rightarrow\underbrace{(G^{2},\frac{HW}{G^{2}},C)\rightarrow(\frac{HW}{G^{2}},G^{2},C)}_{\text{swapaxes(axis1=-2,axis2=-3)}}
$$

with its inverse operation $\mathsf{Ungrid}(\cdot)$ that reverses the gridded input back to the normal 2D feature space.

To this end, we are ready to explain the multi-axis attention module. Given an input tensor $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$, the local Block Attention can be expressed as:

$$
\displaystyle\begin{split}\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{Unblock}(\mathsf{RelAttention}(\mathsf{Block}(\mathsf{LN}(\mathbf{x}))))\\
\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{MLP}(\mathsf{LN}(\mathbf{x}))\\
\end{split}
$$

while the global, dilated Grid Attention module is formulated as:

$$
\displaystyle\begin{split}\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{Ungrid}(\mathsf{RelAttention}(\mathsf{Grid}(\mathsf{LN}(\mathbf{x}))))\\
\mathbf{x}&\leftarrow\mathbf{x}+\mathsf{MLP}(\mathsf{LN}(\mathbf{x}))\\
\end{split}
$$

where we omit the $QKV$ input format in the $\mathsf{RelAttention}$ operation for simplicity. $\mathsf{LN}$ denotes the Layer Normalization [^2], where $\mathsf{MLP}$ is a standard MLP network [^22] [^56] consisting of two linear layers: $\mathbf{x}\leftarrow W_{2}\mathsf{GELU}(W_{1}\mathbf{x})$.

#### Comparison to Axial attention

Figure 6: Comparison of Axial attention and our proposed Multi-Axis attention.

It should be noted that our proposed multi-axis attention (Max-SA) module is completely different from the axial attention proposed in [^33] [^86]. As shown in Figure 6(a), Axial attention proposes to first apply column-wise attention then row-wise, which achieves a global receptive field with $\mathcal{O}(N\sqrt{N})$ complexity (assuming $N$ equals to the number of pixels). On the contrary, our proposed Max-SA shown in Figure 6(b) first employs local attention, then sparse global attention, enjoying global receptive fields with only $\mathcal{O}(N)$ linear complexity. Moreover, we deem the proposed Max-SA a more natural approach for vision since the design of attended regions account for the 2D structure of images, *e.g.*, mixing tokens in a spatially-local small window.

#### MaxViT Block

We demonstrate in Algo. 1 an einops-style pseudocode of the MaxViT block which contains MBConv, block attention, and grid attention.

Algo. 1 Pseudocode of MaxViT Block

⬇

\# input: features (b, h, w, c). Assume h==w; x/output: features (b, h, w, c).

\# p/g: block/grid size. Use 7 by default.

def RelSelfAttn(x): return x # A self-attn function applied on the -2 axis

\# Window/grid partition function

from einops import rearrange

def block(x,p):

return rearrange(x,"b(hy)(wx)c->b(hw)(yx)c",h=x.shape\[1\]//p,w=x.shape\[2\]//p,y=p,x=p)

def unblock(x,g,p):

return rearrange(x,"b(hw)(yx)c->b(hy)(wx)c",h=g,w=g,y=p,x=p)

x = MBConv(input) # MBConv layer

x = block(x,p) # window partition

x = RelSelfAttn(x) # Apply window-attention

x = unblock(x,x.shape\[1\]//p,p) # reverse

x = block(x,x.shape\[1\]//g) # grid partition

x = swapaxes(x,-2,-3) # move grid-axis to -2

x = RelSelfAttn(x) # Apply grid-attention

x = swapaxes(x,-2,-3) # reverse swapaxes

output = unblock(x,g,x.shape\[1\]//g) # reverse

#### Classification Head

Instead of using the \[cls\] token [^22], we simply apply global average pooling to the output of the last stage (S4) to obtain the feature representation, followed by the final classification head.

#### Architectural Specifications

Finally, we present detailed architectural specifications for the MaxViT model family (T/S/B/L) in Table 11.

Table 11: Detailed architectural specifications for MaxViT families.

|  | dsp. rate (out size) | MaxViT-T | MaxViT-S |
| --- | --- | --- | --- |
| stem | $2\times$ ($112\!\times\!112$) | 3 $\times$ 3, 64, stride 23 $\times$ 3, 64, stride 1 | 3 $\times$ 3, 64, stride 23 $\times$ 3, 64, stride 1 |
| S1 | $4\times$ ($56\times 56$) | $\left[\begin{array}[]{c}\text{MBConv, 64, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 2}\\ \text{Rel-MSA, G 7$\times$7, H 2}\end{array}\right]\times 2$ | $\left[\begin{array}[]{c}\text{MBConv, 96, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 3}\\ \text{Rel-MSA, G 7$\times$7, H 3}\end{array}\right]\times 2$ |
| S2 | $8\times$ ($28\times 28$) | $\left[\begin{array}[]{c}\text{MBConv, 128, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 4}\\ \text{Rel-MSA, G 7$\times$7, H 4}\end{array}\right]\times 2$ | $\left[\begin{array}[]{c}\text{MBConv, 192, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 6}\\ \text{Rel-MSA, G 7$\times$7, H 6}\end{array}\right]\times 2$ |
| S3 | $16\times$ ($14\times 14$) | $\left[\begin{array}[]{c}\text{MBConv, 256, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 8}\\ \text{Rel-MSA, G 7$\times$7, H 8}\end{array}\right]\times 5$ | $\left[\begin{array}[]{c}\text{MBConv, 384, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 12}\\ \text{Rel-MSA, G 7$\times$7, H 12}\end{array}\right]\times 5$ |
| S4 | $32\times$ ($7\times 7$) | $\left[\begin{array}[]{c}\text{MBConv, 512, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 16}\\ \text{Rel-MSA, G 7$\times$7, H 16}\end{array}\right]\times 2$ | $\left[\begin{array}[]{c}\text{MBConv, 768, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 24}\\ \text{Rel-MSA, G 7$\times$7, H 24}\end{array}\right]\times 2$ |
|  | dsp. rate (out size) | MaxViT-B | MaxViT-L |
| stem | $2\times$ ($112\!\times\!112$) | 3 $\times$ 3, 64, stride 23 $\times$ 3, 64, stride 1 | 3 $\times$ 3, 128, stride 23 $\times$ 3, 128, stride 1 |
| S1 | $4\times$ ($56\times 56$) | $\left[\begin{array}[]{c}\text{MBConv, 96, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 3}\\ \text{Rel-MSA, G 7$\times$7, H 3}\end{array}\right]\times 2$ | $\left[\begin{array}[]{c}\text{MBConv, 128, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 4}\\ \text{Rel-MSA, G 7$\times$7, H 4}\end{array}\right]\times 2$ |
| S2 | $8\times$ ($28\times 28$) | $\left[\begin{array}[]{c}\text{MBConv, 192, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 6}\\ \text{Rel-MSA, G 7$\times$7, H 6}\end{array}\right]\times 6$ | $\left[\begin{array}[]{c}\text{MBConv, 256, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 8}\\ \text{Rel-MSA, G 7$\times$7, H 8}\end{array}\right]\times 6$ |
| S3 | $16\times$ ($14\times 14$) | $\left[\begin{array}[]{c}\text{MBConv, 384, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 12}\\ \text{Rel-MSA, G 7$\times$7, H 12}\end{array}\right]\!\times\!14$ | $\left[\begin{array}[]{c}\text{MBConv, 512, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 16}\\ \text{Rel-MSA, G 7$\times$7, H 16}\end{array}\right]\times 14$ |
| S4 | $32\times$ ($7\times 7$) | $\left[\begin{array}[]{c}\text{MBConv, 768, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 24}\\ \text{Rel-MSA, G 7$\times$7, H 24}\end{array}\right]\times 2$ | $\left[\begin{array}[]{c}\text{MBConv, 1024, E 4, R 4}\\ \text{Rel-MSA, P 7$\times$7, H 32}\\ \text{Rel-MSA, G 7$\times$7, H 32}\end{array}\right]\times 2$ |

### 0.A.2 Detection and Segmentation Models

We follow the settings of the cascaded Faster-RCNN [^65] and Mask-RCNN [^28], but replace the feature extraction backbone with our MaxViT backbone. We also applied FPN [^52] in the feature map generation, where the S2, S3, S4 (multi-scale features of targeted resolution $1/8$, $1/16$, $1/32$ in MaxViT, respectively) are used. Then the generated feature maps are fed into the detection head. For fair comparison, we follow the original implementation without adopting any system-level strategies to further boost the final performance, such as the HTC framework [^7], instaboost [^25], *etc.* used in Swin [^56]. We show the results of MaxViT-T/S/B on these two tasks to compare it against recent strong models at similar model complexity.

### 0.A.3 Image Aesthetics Model

This task requires incorporating both local and global information of an image to accurately predict human perceptual preference. To this end, the model needs to have the capacity to learn pixel-level quality aspects such as sharpness, noisiness and contrast as well as semantic-level aspects such as composition and depth-of-field. We follow [^77] and use the normalized Earth Mover’s Distance as our training loss. Given the ground truth and predicted probability mass functions p and $\widehat{\textbf{p}}$ representing the histogram of scores, the normalized Earth Mover’s Distance can be expressed as:

$$
\mbox{EMD}(\textbf{p},\widehat{\textbf{p}})=\left(\frac{1}{N}\sum_{k=1}^{N}|\mbox{CDF}_{\textbf{p}}(k)-\mbox{CDF}_{\widehat{\textbf{p}}}(k)|^{r}\right)^{1/r}
$$

where $\mbox{CDF}_{\textbf{p}}(k)$ is the cumulative distribution function as $\sum_{i=1}^{k}\textbf{p}_{i}$, and $N=10$ represents the number score bins. In our experiments we set $r=2$. We remove the classification head used in MaxViT, and instead append a fully-connected layer with 10 neurons followed by $\mathsf{softmax}$.

Figure 7: Generator architecture using the MaxViT block for the GAN experiment. In every stage, we first use the cross-attention module to let the features attend to the latent embedding projected from the input code, which are then fed into the proposed MaxViT block consisting of grid attention, block attention, and MBConv layer. Note that unlike the main model in Sec. 0.A.1, the order of applying the three layers are reversed: from global to local.

### 0.A.4 GAN Model

The above image recognition tasks can validate the power of our proposed MaxViT block used in downsampling (contracting) models. For this GAN experiment, we would like to demonstrate its effectiveness in upsampling (expanding) architectures. The MaxViT-GAN model for image generation is illustrated in Figure 7. For unconditional image generation, MaxViT-GAN first takes a latent code $z\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ as input, then progressively generates an image of target resolution through a hierarchically upsampling structure. We start by linearly projecting the input to a feature with spatial dimension $8\times 8$. During the generation, the feature will go through five stages consisting of identical GAN blocks with gradually increased spatial resolution, similar to the design of our main model. Similar to [^103], we apply a cross-attention layer before the MaxViT block as a memory-efficient form of self-modulation in every stage, which has been shown to stabilize GAN training and also improve mode coverage [^11] [^103]. We use pixel shuffle [^72] for upsampling in the end of each stage.

## Appendix 0.B Experimental Settings

### 0.B.1 ImageNet Classification

We provide ImageNet-1K experimental settings of MaxViT models for both pre-training and fine-tuning in Table 12. All the MaxViT variants used similar hyperparameters except that we mainly customize the stochastic depth rate to regularize each model separately.

Table 12: Detailed hyperparameters used in ImageNet-1K experiments. Multiple values separated by ‘ $/$ ’ are for each model size respectively.

<table><tbody><tr><th rowspan="3">Hyperparameter</th><td colspan="2">ImageNet-1K</td><td colspan="2">ImageNet-21K</td><td colspan="2">JFT-300M</td></tr><tr><td>Pre-training</td><td>Fine-tuning</td><td>Pre-training</td><td>Fine-tuning</td><td>Pre-training</td><td>Fine-tuning</td></tr><tr><td colspan="2">(MaxViT-T/S/B/L)</td><td colspan="2">(MaxViT-B/L/XL)</td><td colspan="2">(MaxViT-B/L/XL)</td></tr><tr><th>Stochastic depth</th><td colspan="2"><math><semantics><mrow><mn>0.2</mn> <mo>/</mo> <mn>0.3</mn> <mo>/</mo> <mn>0.4</mn> <mo>/</mo> <mn>0.6</mn></mrow> <annotation>0.2/0.3/0.4/0.6</annotation></semantics></math></td><td><math><semantics><mrow><mn>0.3</mn> <mo>/</mo> <mn>0.4</mn> <mo>/</mo> <mn>0.6</mn></mrow> <annotation>0.3/0.4/0.6</annotation></semantics></math></td><td><math><semantics><mrow><mn>0.4</mn> <mo>/</mo> <mn>0.5</mn> <mo>/</mo> <mn>0.9</mn></mrow> <annotation>0.4/0.5/0.9</annotation></semantics></math></td><td><math><semantics><mrow><mn>0.0</mn> <mo>/</mo> <mn>0.0</mn> <mo>/</mo> <mn>0.0</mn></mrow> <annotation>0.0/0.0/0.0</annotation></semantics></math></td><td><math><semantics><mrow><mn>0.1</mn> <mo>/</mo> <mn>0.2</mn> <mo>/</mo> <mn>0.2</mn></mrow> <annotation>0.1/0.2/0.2</annotation></semantics></math></td></tr><tr><th>Center crop</th><td>True</td><td>False</td><td>True</td><td>False</td><td>True</td><td>False</td></tr><tr><th>RandAugment</th><td>2, 15</td><td>2, 15</td><td>2, 5</td><td>2, 15</td><td>2, 5</td><td>2, 15</td></tr><tr><th>Mixup alpha</th><td>0.8</td><td>0.8</td><td>None</td><td>None</td><td>None</td><td>None</td></tr><tr><th>Loss type</th><td>Softmax</td><td>Softmax</td><td>Sigmoid</td><td>Softmax</td><td>Sigmoid</td><td>Softmax</td></tr><tr><th>Label smoothing</th><td>0.1</td><td>0.1</td><td>0.0001</td><td>0.1</td><td>0</td><td>0.1</td></tr><tr><th>Train epochs</th><td>300</td><td>30</td><td>90</td><td>30</td><td>14</td><td>30</td></tr><tr><th>Train batch size</th><td>4096</td><td>512</td><td>4096</td><td>512</td><td>4096</td><td>512</td></tr><tr><th>Optimizer type</th><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><th>Peak learning rate</th><td>3e-3</td><td>5e-5</td><td>1e-3</td><td>5e-5</td><td>1e-3</td><td>5e-5</td></tr><tr><th>Min learning rate</th><td>1e-5</td><td>5e-5</td><td>1e-5</td><td>5e-5</td><td>1e-5</td><td>5e-5</td></tr><tr><th>Warm-up</th><td>10K steps</td><td>None</td><td>5 epochs</td><td>None</td><td>20K steps</td><td>None</td></tr><tr><th>LR decay schedule</th><td>Cosine</td><td>None</td><td>Linear</td><td>None</td><td>Linear</td><td>None</td></tr><tr><th>Weight decay rate</th><td>0.05</td><td>1e-8</td><td>0.01</td><td>1e-8</td><td>0.01</td><td>1e-8</td></tr><tr><th>Gradient clip</th><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><th>EMA decay rate</th><td>None</td><td>0.9999</td><td>None</td><td>0.9999</td><td>None</td><td>0.9999</td></tr></tbody></table>

### 0.B.2 Coco Detection and Segmentation

We evaluated MaxViT on the COCO2017 [^53] object bounding box detection and instance segmentation tasks. The dataset contains 118K training and 5K validation samples. All the MaxViT backbones used are pretrained on ImageNet-1k at resolution $224\times 224$. These pretrained checkpoints are then used as the warm-up weights for fine-tuning the detection and segmentation tasks. For both tasks, the input images are resized to $896\times 896$. The training is conducted with a batch size of 256, using the AdamW [^59] optimizer with learning rate of 1e-3, 3e-3, 3e-3, and stochastic depth of $0.8,0.3,0.3$ for MaxViT-T/S/B, respectively.

### 0.B.3 Image Aesthetics Assessment

We trained and evaluated the MaxViT model on the AVA benchmark [^61]. This dataset consists of 255K images rated by armature photographers through photography contests. Each image is rated by an average of 200 human raters, assigning a score from 1 to 10 to images. The higher the score, the better the visual aesthetic quality of the image. Each image in the dataset has a histogram of scores associated with it, which we use as the ground truth label. Similar to [^77] [^43], we split the dataset into train and test sets, such that 20% of the data is used for testing. We train MaxViT for three different input resolutions: $224\times 224$, $384\times 384$ and $512\times 512$. We initialized the model with ImageNet-1K 224 $\times$ 224 pre-trained weights. The weight and bias momentums are set to 0.9, and a dropout rate of 0.75 is applied on the last layer of the baseline network. We use an initial learning rate of 1e-3, exponentially decayed with decay factor 0.9 every 10 epochs. We set the stochastic depth rate to 0.5.

### 0.B.4 Image Generation

We use a ResNet-based discriminator following [^42]. To train the model, we also used the standard non-saturating logistic GAN loss with $R1$ gradient penalty [^60] applied to the discriminator with the gradient penalty weight set to 10. We employ the Adam [^45] optimizer with a learning rate of 1e-4 for both generator and discriminator. The model is trained on TPU for one million steps with batch size 256. Notably, we do not employ extra GAN training tricks such as pixel norm, noise injection, progressive growing, *etc.* on which recent state-of-the-art models are heavily relied to attain good results [^41] [^42]. The overall objectives of the GAN training are defined as:

$$
\displaystyle\mathcal{L}_{G}
$$
 
$$
\displaystyle=-\mathbb{E}_{z\sim P_{z}}[\log(D(G(z))],
$$
$$
\displaystyle\mathcal{L}_{D}
$$
 
$$
\displaystyle=-\mathbb{E}_{x\sim P_{x}}[\log(D(x))]-\mathbb{E}_{z\sim P_{z}}[\log(1-D(G(z)))]+\gamma\mathbb{E}_{x\sim P_{x}}[\|\nabla_{x}D(x)\|_{2}^{2}],
$$

where $\gamma$ denotes the $R_{1}$ gradient penalty weight.

## Appendix 0.C Complete Experimental Results

We provide complete experiment comparisons for ImageNet-1K, Image-21K, and JFT datasets in Table 13 and Table 14, respectively. We also provide more visual results for unconditional image generation on ImageNet-1K in Figure 8.

Table 13: Complete performance comparison under ImageNet-1K only setting.

<table><thead><tr><th></th><th>Model</th><th>Eval size</th><th>Params</th><th>FLOPs</th><th>throughput (img/s)</th><th>ImageNet top-1 acc.</th></tr></thead><tbody><tr><th rowspan="20">ConvNets</th><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B3 <sup><a href="#fn:79">79</a></sup></th><th>300</th><td>12M</td><td>1.8G</td><td>732.1</td><td>81.6</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B4 <sup><a href="#fn:79">79</a></sup></th><th>380</th><td>19M</td><td>4.2G</td><td>349.4</td><td>82.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B5 <sup><a href="#fn:79">79</a></sup></th><th>456</th><td>30M</td><td>9.9G</td><td>169.1</td><td>83.6</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B6 <sup><a href="#fn:79">79</a></sup></th><th>528</th><td>43M</td><td>19.0G</td><td>96.9</td><td>84.0</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNet-B7 <sup><a href="#fn:79">79</a></sup></th><th>600</th><td>66M</td><td>37.0G</td><td>55.1</td><td>84.3</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> RegNetY-8GF <sup><a href="#fn:62">62</a></sup></th><th>224</th><td>39M</td><td>8.0G</td><td>591.6</td><td>81.7</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> RegNetY-16GF <sup><a href="#fn:62">62</a></sup></th><th>224</th><td>84M</td><td>16.0G</td><td>334.7</td><td>82.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F0 <sup><a href="#fn:5">5</a></sup></th><th>256</th><td>72M</td><td>12.4G</td><td>533,.3</td><td>83.6</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F1 <sup><a href="#fn:5">5</a></sup></th><th>320</th><td>132M</td><td>35.5G</td><td>228.5</td><td>84.7</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F2 <sup><a href="#fn:5">5</a></sup></th><th>352</th><td>194M</td><td>62.6G</td><td>129.0</td><td>85.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F3 <sup><a href="#fn:5">5</a></sup></th><th>416</th><td>255M</td><td>114.7G</td><td>78.8</td><td>85.7</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F4 <sup><a href="#fn:5">5</a></sup></th><th>512</th><td>316M</td><td>215.2G</td><td>51.7</td><td>85.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F5 <sup><a href="#fn:5">5</a></sup></th><th>544</th><td>377M</td><td>289.8G</td><td>-</td><td>86.0</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-S <sup><a href="#fn:80">80</a></sup></th><th>384</th><td>24M</td><td>8.8G</td><td>666.6</td><td>83.9</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-M <sup><a href="#fn:80">80</a></sup></th><th>380</th><td>55M</td><td>24.0G</td><td>280.7</td><td>85.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-L <sup><a href="#fn:80">80</a></sup></th><th>480</th><td>121M</td><td>53.0G</td><td>163.2</td><td>85.7</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-T <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>29M</td><td>4.5G</td><td>774.7</td><td>82.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-S <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>50M</td><td>8.7G</td><td>447.1</td><td>83.1</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-B <sup><a href="#fn:57">57</a></sup></th><th>224</th><td>89M</td><td>15.4G</td><td>292.1</td><td>83.8</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-L <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>198M</td><td>101.0G</td><td>50.4</td><td>85.5</td></tr><tr><th rowspan="20">ViTs</th><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/32 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>86M</td><td>55.4G</td><td>85.9</td><td>77.9</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>307M</td><td>190.7G</td><td>27.3</td><td>76.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeiT-S <sup><a href="#fn:81">81</a></sup></th><th>224</th><td>22M</td><td>4.6G</td><td>940.4</td><td>79.8</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeiT-B <sup><a href="#fn:81">81</a></sup></th><th>224</th><td>86M</td><td>17.5G</td><td>292.3</td><td>81.8</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeiT-B <sup><a href="#fn:81">81</a></sup></th><th>384</th><td>86M</td><td>55.4G</td><td>85.9</td><td>83.1</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CaiT-S36 <sup><a href="#fn:82">82</a></sup></th><th>224</th><td>68M</td><td>13.9G</td><td>-</td><td>83.3</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CaiT-M24 <sup><a href="#fn:82">82</a></sup></th><th>224</th><td>186M</td><td>36.0G</td><td>-</td><td>83.4</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CaiT-M24 <sup><a href="#fn:82">82</a></sup></th><th>384</th><td>186M</td><td>116.1G</td><td>-</td><td>84.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeepViT-S <sup><a href="#fn:105">105</a></sup></th><th>224</th><td>27M</td><td>6.2G</td><td>-</td><td>82.3</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> DeepViT-L <sup><a href="#fn:105">105</a></sup></th><th>224</th><td>55M</td><td>12.5G</td><td>-</td><td>83.1</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> T2T-ViT-14 <sup><a href="#fn:101">101</a></sup></th><th>224</th><td>22M</td><td>6.1G</td><td>-</td><td>81.7</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> T2T-ViT-19 <sup><a href="#fn:101">101</a></sup></th><th>224</th><td>39M</td><td>9.8G</td><td>-</td><td>82.2</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> T2T-ViT-24 <sup><a href="#fn:101">101</a></sup></th><th>224</th><td>64M</td><td>15.0G</td><td>-</td><td>82.6</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-T <sup><a href="#fn:56">56</a></sup></th><th>224</th><td>29M</td><td>4.5G</td><td>755.2</td><td>81.3</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-S <sup><a href="#fn:56">56</a></sup></th><th>224</th><td>50M</td><td>8.7G</td><td>436.9</td><td>83.0</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-B <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>88M</td><td>47.0G</td><td>84.7</td><td>84.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-B <sup><a href="#fn:21">21</a></sup></th><th>224</th><td>78M</td><td>15.0G</td><td>250</td><td>84.2</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-B <sup><a href="#fn:21">21</a></sup></th><th>384</th><td>78M</td><td>47.0G</td><td>-</td><td>85.4</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Focal-S <sup><a href="#fn:99">99</a></sup></th><th>224</th><td>51M</td><td>9.1G</td><td>-</td><td>83.5</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Focal-B <sup><a href="#fn:99">99</a></sup></th><th>224</th><td>90M</td><td>16.0G</td><td>-</td><td>83.8</td></tr><tr><th rowspan="17">Hybrid</th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-13 <sup><a href="#fn:93">93</a></sup></th><th>224</th><td>20M</td><td>4.5G</td><td>-</td><td>81.6</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-21 <sup><a href="#fn:93">93</a></sup></th><th>224</th><td>32M</td><td>7.1G</td><td>-</td><td>82.5</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-21 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>32M</td><td>24.9G</td><td>-</td><td>83.3</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-0 <sup><a href="#fn:19">19</a></sup></th><th>224</th><td>25M</td><td>4.2G</td><td>534.5</td><td>81.6</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-1 <sup><a href="#fn:19">19</a></sup></th><th>224</th><td>42M</td><td>8.4G</td><td>336.5</td><td>83.3</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-2 <sup><a href="#fn:19">19</a></sup></th><th>224</th><td>75M</td><td>15.7G</td><td>247.6</td><td>84.1</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>168M</td><td>107.4G</td><td>48.5</td><td>85.8</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>168M</td><td>203.1G</td><td>22.4</td><td>86.0</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>224</th><td>31M</td><td>5.6G</td><td>349.6</td><td>83.62</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>224</th><td>69M</td><td>11.7G</td><td>242.5</td><td>84.45</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>224</th><td>120M</td><td>23.4G</td><td>133.6</td><td>84.95</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>224</th><td>212M</td><td>43.9G</td><td>99.4</td><td>85.17</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>384</th><td>31M</td><td>17.7G</td><td>121.9</td><td>85.24</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>384</th><td>69M</td><td>36.1G</td><td>82.7</td><td>85.74</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>384</th><td>120M</td><td>74.2G</td><td>45.8</td><td>86.34</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>384</th><td>212M</td><td>133.1G</td><td>34.3</td><td>84.40</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-T</th><th>512</th><td>31M</td><td>33.7G</td><td>63.8</td><td>85.72</td></tr><tr><th></th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-S</th><th>512</th><td>69M</td><td>67.6G</td><td>43.3</td><td>86.19</td></tr><tr><th></th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>512</th><td>120M</td><td>138.5G</td><td>24.0</td><td>86.66</td></tr><tr><th></th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>512</th><td>212M</td><td>245.4G</td><td>17.8</td><td>86.70</td></tr></tbody></table>

Table 14: Complete performance comparison for ImageNet-21K and JFT pre-trained models.

<table><thead><tr><th></th><th rowspan="2">Model</th><th rowspan="2">Eval size</th><th rowspan="2">Params</th><th rowspan="2">FLOPs</th><th colspan="2">IN-1K top-1 acc.</th></tr><tr><th></th><th>21K <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 1K</th><th>JFT <math><semantics><mo>→</mo> <annotation>\rightarrow</annotation></semantics></math> 1K</th></tr></thead><tbody><tr><th rowspan="10">ConvNets</th><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> BiT-R-101x3 <sup><a href="#fn:46">46</a></sup></th><th>384</th><td>388M</td><td>204.6G</td><td>84.4</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> BiT-R-152x4 <sup><a href="#fn:46">46</a></sup></th><th>480</th><td>937M</td><td>840.5G</td><td>85.4</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-S <sup><a href="#fn:80">80</a></sup></th><th>384</th><td>24M</td><td>8.8G</td><td>85.0</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-M <sup><a href="#fn:80">80</a></sup></th><th>480</th><td>55M</td><td>24.0G</td><td>86.1</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-L <sup><a href="#fn:80">80</a></sup></th><th>480</th><td>121M</td><td>53.0G</td><td>86.8</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> EffNetV2-XL <sup><a href="#fn:80">80</a></sup></th><th>512</th><td>208M</td><td>94.0G</td><td>87.3</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> NFNet-F4+ <sup><a href="#fn:5">5</a></sup></th><th>512</th><td>527M</td><td>367G</td><td>-</td><td>89.20</td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-B <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>89M</td><td>45.1G</td><td>86.8</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-L <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>198M</td><td>101.0G</td><td>87.5</td><td></td></tr><tr><th><math><semantics><mo>∙</mo> <annotation>\bullet</annotation></semantics></math> ConvNeXt-XL <sup><a href="#fn:57">57</a></sup></th><th>384</th><td>350M</td><td>179.0G</td><td>87.8</td><td></td></tr><tr><th rowspan="12">ViTs</th><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-B/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>87M</td><td>55.5G</td><td>84.0</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>305M</td><td>191.1G</td><td>85.2</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>512</th><td>305M</td><td>364G</td><td>-</td><td>87.76</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> ViT-H/14 <sup><a href="#fn:22">22</a></sup></th><th>518</th><td>632M</td><td>1021G</td><td>-</td><td>88.55</td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> HaloNet-H4 <sup><a href="#fn:84">84</a></sup></th><th>384</th><td>85M</td><td>-</td><td>85.6</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> HaloNet-H4 <sup><a href="#fn:84">84</a></sup></th><th>512</th><td>85M</td><td>-</td><td>85.8</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-B <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>88M</td><td>47.0G</td><td>86.4</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> Swin-L <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>197M</td><td>103.9G</td><td>87.3</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> SwinV2-B <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>88M</td><td>-</td><td>87.1</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> SwinV2-L <sup><a href="#fn:56">56</a></sup></th><th>384</th><td>197M</td><td>-</td><td>87.7</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-B <sup><a href="#fn:21">21</a></sup></th><th>384</th><td>78M</td><td>47.0G</td><td>87.0</td><td></td></tr><tr><th><math><semantics><mo>∘</mo> <annotation>\circ</annotation></semantics></math> CSwin-L <sup><a href="#fn:21">21</a></sup></th><th>384</th><td>173M</td><td>96.8G</td><td>87.5</td><td></td></tr><tr><th rowspan="17">Hybrid</th><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-13 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>20M</td><td>16.0G</td><td>83.3</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-21 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>32M</td><td>25.0G</td><td>84.9</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CvT-W24 <sup><a href="#fn:93">93</a></sup></th><th>384</th><td>277M</td><td>193.2G</td><td>87.7</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> ResNet+ViT-L/16 <sup><a href="#fn:22">22</a></sup></th><th>384</th><td>330M</td><td>-</td><td>-</td><td>87.12</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-2 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>75M</td><td>49.8G</td><td>87.1</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>168M</td><td>107.4G</td><td>87.6</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-4 <sup><a href="#fn:19">19</a></sup></th><th>384</th><td>275M</td><td>189.5G</td><td>87.9</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-2 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>75M</td><td>96.7G</td><td>87.3</td><td></td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-3 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>168M</td><td>203.1G</td><td>87.9</td><td>88.81</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-4 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>275M</td><td>360.9G</td><td>88.1</td><td>89.11</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> CoAtNet-5 <sup><a href="#fn:19">19</a></sup></th><th>512</th><td>688M</td><td>812G</td><td>-</td><td>89.77</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>384</th><td>119M</td><td>74.2G</td><td>88.24</td><td>88.69</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>384</th><td>212M</td><td>128.7G</td><td>88.32</td><td>89.12</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-XL</th><th>384</th><td>475M</td><td>293.7G</td><td>88.51</td><td>89.36</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-B</th><th>512</th><td>119M</td><td>138.3G</td><td>88.38</td><td>88.82</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-L</th><th>512</th><td>212M</td><td>245.2G</td><td>88.46</td><td>89.41</td></tr><tr><th><math><semantics><mo>⋄</mo> <annotation>\diamond</annotation></semantics></math> MaxViT-XL</th><th>512</th><td>475M</td><td>535.2G</td><td>88.70</td><td>89.53</td></tr></tbody></table>

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2204.01697/assets/figures/1.png)

Figure 8: Unconditional generation results on ImageNet-1k 128 × 128\\times 128.

[^1]: Arnab, A., Dehghani, M., Heigold, G., Sun, C., Lučić, M., Schmid, C.: Vivit: A video vision transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6836–6846 (2021)

[^2]: Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016)

[^3]: Bello, I., Fedus, W., Du, X., Cubuk, E.D., Srinivas, A., Lin, T.Y., Shlens, J., Zoph, B.: Revisiting resnets: Improved training and scaling strategies. Advances in Neural Information Processing Systems 34, 22614–22627 (2021)

[^4]: Bello, I., Zoph, B., Vaswani, A., Shlens, J., Le, Q.V.: Attention augmented convolutional networks. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 3286–3295 (2019)

[^5]: Brock, A., De, S., Smith, S.L., Simonyan, K.: High-performance large-scale image recognition without normalization. In: International Conference on Machine Learning. pp. 1059–1071. PMLR (2021)

[^6]: Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: End-to-end object detection with transformers. In: European conference on computer vision. pp. 213–229. Springer (2020)

[^7]: Chen, K., Pang, J., Wang, J., Xiong, Y., Li, X., Sun, S., Feng, W., Liu, Z., Shi, J., Ouyang, W., Loy, C.C., Lin, D.: Hybrid task cascade for instance segmentation. In: IEEE Conference on Computer Vision and Pattern Recognition (2019)

[^8]: Chen, L.H., Bampis, C.G., Li, Z., Norkin, A., Bovik, A.C.: Proxiqa: A proxy approach to perceptual optimization of learned image compression. IEEE Transactions on Image Processing 30, 360–373 (2020)

[^9]: Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L.: Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE transactions on pattern analysis and machine intelligence 40(4), 834–848 (2017)

[^10]: Chen, Q., Zhang, W., Zhou, N., Lei, P., Xu, Y., Zheng, Y., Fan, J.: Adaptive fractional dilated convolution network for image aesthetics assessment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 14114–14123 (2020)

[^11]: Chen, T., Lucic, M., Houlsby, N., Gelly, S.: On self modulation for generative adversarial networks. arXiv preprint arXiv:1810.01365 (2018)

[^12]: Chen, T., Zhai, X., Ritter, M., Lucic, M., Houlsby, N.: Self-supervised gans via auxiliary rotation loss. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12154–12163 (2019)

[^13]: Chen, W.T., Huang, Z.K., Tsai, C.C., Yang, H.H., Ding, J.J., Kuo, S.Y.: Learning multiple adverse weather removal via two-stage knowledge learning and multi-contrastive regularization: Toward a unified model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17653–17662 (2022)

[^14]: Chen, W., Du, X., Yang, F., Beyer, L., Zhai, X., Lin, T., Chen, H., Li, J., Song, X., Wang, Z., Zhou, D.: A simple single-scale vision transformer for object localization and instance segmentation. CoRR abs/2112.09747 (2021), [https://arxiv.org/abs/2112.09747](https://arxiv.org/abs/2112.09747)

[^15]: Chen, W., Huang, W., Du, X., Song, X., Wang, Z., Zhou, D.: Auto-scaling vision transformers without training. arXiv preprint arXiv:2202.11921 (2022)

[^16]: Chu, X., Tian, Z., Wang, Y., Zhang, B., Ren, H., Wei, X., Xia, H., Shen, C.: Twins: Revisiting the design of spatial attention in vision transformers. Advances in Neural Information Processing Systems 34 (2021)

[^17]: Chu, X., Tian, Z., Zhang, B., Wang, X., Wei, X., Xia, H., Shen, C.: Conditional positional encodings for vision transformers. arXiv preprint arXiv:2102.10882 (2021)

[^18]: Coates, A., Ng, A., Lee, H.: An analysis of single-layer networks in unsupervised feature learning. In: Gordon, G., Dunson, D., Dudík, M. (eds.) Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. Proceedings of Machine Learning Research, vol. 15, pp. 215–223. PMLR, Fort Lauderdale, FL, USA (11–13 Apr 2011), [https://proceedings.mlr.press/v15/coates11a.html](https://proceedings.mlr.press/v15/coates11a.html)

[^19]: Dai, Z., Liu, H., Le, Q., Tan, M.: Coatnet: Marrying convolution and attention for all data sizes. Advances in Neural Information Processing Systems 34 (2021)

[^20]: Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018)

[^21]: Dong, X., Bao, J., Chen, D., Zhang, W., Yu, N., Yuan, L., Chen, D., Guo, B.: Cswin transformer: A general vision transformer backbone with cross-shaped windows. arXiv preprint arXiv:2107.00652 (2021)

[^22]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

[^23]: d’Ascoli, S., Touvron, H., Leavitt, M.L., Morcos, A.S., Biroli, G., Sagun, L.: Convit: Improving vision transformers with soft convolutional inductive biases. In: International Conference on Machine Learning. pp. 2286–2296. PMLR (2021)

[^24]: Fan, H., Xiong, B., Mangalam, K., Li, Y., Yan, Z., Malik, J., Feichtenhofer, C.: Multiscale vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 6824–6835 (2021)

[^25]: Fang, H.S., Sun, J., Wang, R., Gou, M., Li, Y.L., Lu, C.: Instaboost: Boosting instance segmentation via probability map guided copy-pasting. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 682–691 (2019)

[^26]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

[^27]: Han, K., Xiao, A., Wu, E., Guo, J., Xu, C., Wang, Y.: Transformer in transformer. Advances in Neural Information Processing Systems 34 (2021)

[^28]: He, K., Gkioxari, G., Dollár, P., Girshick, R.: Mask r-cnn. In: 2017 IEEE International Conference on Computer Vision (ICCV). pp. 2980–2988 (2017). https://doi.org/10.1109/ICCV.2017.322

[^29]: He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

[^30]: He, K., Zhang, X., Ren, S., Sun, J.: Identity mappings in deep residual networks. In: European conference on computer vision. pp. 630–645. Springer (2016)

[^31]: Hendrycks, D., Gimpel, K.: Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415 (2016)

[^32]: Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: GANs trained by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS. pp. 6629–6640 (2017)

[^33]: Ho, J., Kalchbrenner, N., Weissenborn, D., Salimans, T.: Axial attention in multidimensional transformers. arXiv preprint arXiv:1912.12180 (2019)

[^34]: Hoang, Q., Nguyen, T.D., Le, T., Phung, D.: Mgan: Training generative adversarial nets with multiple generators. In: International conference on learning representations (2018)

[^35]: Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H.: Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)

[^36]: Hu, J., Shen, L., Sun, G.: Squeeze-and-excitation networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7132–7141 (2018)

[^37]: Huang, G., Liu, Z., Van Der Maaten, L., Weinberger, K.Q.: Densely connected convolutional networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4700–4708 (2017)

[^38]: Hudson, D.A., Zitnick, L.: Generative adversarial transformers. In: International Conference on Machine Learning. pp. 4487–4499. PMLR (2021)

[^39]: Ioffe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by reducing internal covariate shift. In: International conference on machine learning. pp. 448–456. PMLR (2015)

[^40]: Jiang, Y., Chang, S., Wang, Z.: Transgan: Two pure transformers can make one strong gan, and that can scale up. Advances in Neural Information Processing Systems 34 (2021)

[^41]: Karras, T., Aila, T., Laine, S., Lehtinen, J.: Progressive growing of gans for improved quality, stability, and variation. arXiv preprint arXiv:1710.10196 (2017)

[^42]: Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T.: Analyzing and improving the image quality of stylegan. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8110–8119 (2020)

[^43]: Ke, J., Wang, Q., Wang, Y., Milanfar, P., Yang, F.: Musiq: Multi-scale image quality transformer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 5148–5157 (2021)

[^44]: Khan, S., Naseer, M., Hayat, M., Zamir, S.W., Khan, F.S., Shah, M.: Transformers in vision: A survey. ACM Computing Surveys (CSUR) (2021)

[^45]: Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014)

[^46]: Kolesnikov, A., Beyer, L., Zhai, X., Puigcerver, J., Yung, J., Gelly, S., Houlsby, N.: Big transfer (bit): General visual representation learning. In: European conference on computer vision. pp. 491–507. Springer (2020)

[^47]: Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images (2009)

[^48]: Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 25 (2012)

[^49]: Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., Soricut, R.: Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942 (2019)

[^50]: Li, Y., Zhang, K., Cao, J., Timofte, R., Van Gool, L.: Localvit: Bringing locality to vision transformers. arXiv preprint arXiv:2104.05707 (2021)

[^51]: Li, Y., Jin, P., Yang, F., Liu, C., Yang, M.H., Milanfar, P.: Comisr: Compression-informed video super-resolution. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2543–2552 (2021)

[^52]: Lin, T.Y., Dollár, P., Girshick, R.B., He, K., Hariharan, B., Belongie, S.J.: Feature pyramid networks for object detection. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) pp. 936–944 (2017)

[^53]: Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. In: European conference on computer vision. pp. 740–755. Springer (2014)

[^54]: Lin, Z., Khetan, A., Fanti, G., Oh, S.: Pacgan: The power of two samples in generative adversarial networks. Advances in neural information processing systems 31 (2018)

[^55]: Liu, S., Wang, T., Bau, D., Zhu, J.Y., Torralba, A.: Diverse image generation via self-conditioned gans. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14286–14295 (2020)

[^56]: Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin transformer: Hierarchical vision transformer using shifted windows. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 10012–10022 (2021)

[^57]: Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. arXiv preprint arXiv:2201.03545 (2022)

[^58]: Long, J., Shelhamer, E., Darrell, T.: Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3431–3440 (2015)

[^59]: Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017)

[^60]: Mescheder, L., Geiger, A., Nowozin, S.: Which training methods for gans do actually converge? In: International conference on machine learning. pp. 3481–3490. PMLR (2018)

[^61]: Murray, N., Marchesotti, L., Perronnin, F.: Ava: A large-scale database for aesthetic visual analysis. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 2408–2415. IEEE (2012)

[^62]: Radosavovic, I., Kosaraju, R.P., Girshick, R., He, K., Dollár, P.: Designing network design spaces. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10428–10436 (2020)

[^63]: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., Liu, P.J.: Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683 (2019)

[^64]: Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems 34 (2021)

[^65]: Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. In: Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 28. Curran Associates, Inc. (2015), [https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf](https://proceedings.neurips.cc/paper/2015/file/14bfa6bb14875e45bba028a21ed38046-Paper.pdf)

[^66]: Rogozhnikov, A.: Einops: Clear and reliable tensor manipulations with einstein-like notation. In: International Conference on Learning Representations (2022), [https://openreview.net/forum?id=oapKSVM2bcj](https://openreview.net/forum?id=oapKSVM2bcj)

[^67]: Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. pp. 234–241. Springer (2015)

[^68]: Sage, A., Agustsson, E., Timofte, R., Van Gool, L.: Logo synthesis and manipulation with clustered generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 5879–5888 (2018)

[^69]: Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X., Chen, X.: Improved techniques for training GANs. In: NeurIPS (2016)

[^70]: Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: Inverted residuals and linear bottlenecks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 4510–4520 (2018)

[^71]: Shaw, P., Uszkoreit, J., Vaswani, A.: Self-attention with relative position representations. arXiv preprint arXiv:1803.02155 (2018)

[^72]: Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P., Bishop, R., Rueckert, D., Wang, Z.: Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1874–1883 (2016)

[^73]: Sun, C., Shrivastava, A., Singh, S., Gupta, A.: Revisiting unreasonable effectiveness of data in deep learning era. In: Proceedings of the IEEE international conference on computer vision. pp. 843–852 (2017)

[^74]: Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A.A.: Inception-v4, inception-resnet and the impact of residual connections on learning. In: Thirty-first AAAI conference on artificial intelligence (2017)

[^75]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1–9 (2015)

[^76]: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z.: Rethinking the inception architecture for computer vision. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2818–2826 (2016)

[^77]: Talebi, H., Milanfar, P.: Nima: Neural image assessment. IEEE transactions on image processing 27(8), 3998–4011 (2018)

[^78]: Talebi, H., Milanfar, P.: Learning to resize images for computer vision tasks. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 497–506 (2021)

[^79]: Tan, M., Le, Q.: Efficientnet: Rethinking model scaling for convolutional neural networks. In: International conference on machine learning. pp. 6105–6114. PMLR (2019)

[^80]: Tan, M., Le, Q.: Efficientnetv2: Smaller models and faster training. In: International Conference on Machine Learning. pp. 10096–10106. PMLR (2021)

[^81]: Touvron, H., Cord, M., Douze, M., Massa, F., Sablayrolles, A., Jégou, H.: Training data-efficient image transformers & distillation through attention. In: International Conference on Machine Learning. pp. 10347–10357. PMLR (2021)

[^82]: Touvron, H., Cord, M., Sablayrolles, A., Synnaeve, G., Jégou, H.: Going deeper with image transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 32–42 (2021)

[^83]: Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., Li, Y.: Maxim: Multi-axis mlp for image processing. arXiv preprint arXiv:2201.02973 (2022)

[^84]: Vaswani, A., Ramachandran, P., Srinivas, A., Parmar, N., Hechtman, B., Shlens, J.: Scaling local self-attention for parameter efficient visual backbones. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 12894–12904 (2021)

[^85]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

[^86]: Wang, H., Zhu, Y., Green, B., Adam, H., Yuille, A., Chen, L.C.: Axial-deeplab: Stand-alone axial-attention for panoptic segmentation. In: European Conference on Computer Vision. pp. 108–126. Springer (2020)

[^87]: Wang, W., Xie, E., Li, X., Fan, D.P., Song, K., Liang, D., Lu, T., Luo, P., Shao, L.: Pyramid vision transformer: A versatile backbone for dense prediction without convolutions. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 568–578 (2021)

[^88]: Wang, X., Girshick, R., Gupta, A., He, K.: Non-local neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 7794–7803 (2018)

[^89]: Wang, Y., Ke, J., Talebi, H., Yim, J.G., Birkbeck, N., Adsumilli, B., Milanfar, P., Yang, F.: Rich features for perceptual quality assessment of ugc videos. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13435–13444 (2021)

[^90]: Whang, J., Delbracio, M., Talebi, H., Saharia, C., Dimakis, A.G., Milanfar, P.: Deblurring via stochastic refinement. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 16293–16303 (2022)

[^91]: Woo, S., Park, J., Lee, J.Y., Kweon, I.S.: Cbam: Convolutional block attention module. In: Proceedings of the European conference on computer vision (ECCV). pp. 3–19 (2018)

[^92]: Wu, F., Fan, A., Baevski, A., Dauphin, Y.N., Auli, M.: Pay less attention with lightweight and dynamic convolutions. arXiv preprint arXiv:1901.10430 (2019)

[^93]: Wu, H., Xiao, B., Codella, N., Liu, M., Dai, X., Yuan, L., Zhang, L.: Cvt: Introducing convolutions to vision transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22–31 (2021)

[^94]: Xiao, T., Dollar, P., Singh, M., Mintun, E., Darrell, T., Girshick, R.: Early convolutions help transformers see better. Advances in Neural Information Processing Systems 34 (2021)

[^95]: Xie, S., Girshick, R., Dollár, P., Tu, Z., He, K.: Aggregated residual transformations for deep neural networks. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 1492–1500 (2017)

[^96]: Xu, R., Tu, Z., Xiang, H., Shao, W., Zhou, B., Ma, J.: Cobevt: Cooperative bird’s eye view semantic segmentation with sparse transformers. arXiv preprint arXiv:2207.02202 (2022)

[^97]: Xu, R., Xiang, H., Tu, Z., Xia, X., Yang, M.H., Ma, J.: V2x-vit: Vehicle-to-everything cooperative perception with vision transformer. arXiv preprint arXiv:2203.10638 (2022)

[^98]: Xu, W., Xu, Y., Chang, T., Tu, Z.: Co-scale conv-attentional image transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9981–9990 (2021)

[^99]: Yang, J., Li, C., Zhang, P., Dai, X., Xiao, B., Yuan, L., Gao, J.: Focal self-attention for local-global interactions in vision transformers. arXiv preprint arXiv:2107.00641 (2021)

[^100]: Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R.R., Le, Q.V.: Xlnet: Generalized autoregressive pretraining for language understanding. Advances in neural information processing systems 32 (2019)

[^101]: Yuan, L., Chen, Y., Wang, T., Yu, W., Shi, Y., Jiang, Z.H., Tay, F.E., Feng, J., Yan, S.: Tokens-to-token vit: Training vision transformers from scratch on imagenet. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 558–567 (2021)

[^102]: Zhai, X., Kolesnikov, A., Neil, H., Beyer, L.: Scaling vision transformers. arXiv preprint arXiv:2106.04560 (2021)

[^103]: Zhao, L., Zhang, Z., Chen, T., Metaxas, D., Zhang, H.: Improved transformer for high-resolution gans. Advances in Neural Information Processing Systems 34 (2021)

[^104]: Zhao, Z., Wu, Z., Zhuang, Y., Li, B., Jia, J.: Tracking objects as pixel-wise distributions. arXiv preprint arXiv:2207.05518 (2022)

[^105]: Zhou, D., Kang, B., Jin, X., Yang, L., Lian, X., Jiang, Z., Hou, Q., Feng, J.: Deepvit: Towards deeper vision transformer. arXiv preprint arXiv:2103.11886 (2021)