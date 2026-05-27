---
title: "A Simple Framework for Contrastive Learning of Visual Representations"
source: "https://ar5iv.labs.arxiv.org/html/2002.05709"
author:
published:
created: 2026-05-27
description: "This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectur…"
tags:
  - "clippings"
---
Ting Chen    Simon Kornblith    Mohammad Norouzi    Geoffrey Hinton

###### Abstract

This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100 $\times$ fewer labels. <sup>1</sup>

Self-supervised Learning, Contrastive Learning, Deep Learning

## 1 Introduction

Learning effective visual representations without human supervision is a long-standing problem. Most mainstream approaches fall into one of two classes: generative or discriminative. Generative approaches learn to generate or otherwise model pixels in the input space [^26] [^31] [^20]. However, pixel-level generation is computationally expensive and may not be necessary for representation learning. Discriminative approaches learn representations using objective functions similar to those used for supervised learning, but train networks to perform pretext tasks where both the inputs and labels are derived from an unlabeled dataset. Many such approaches have relied on heuristics to design pretext tasks [^13] [^60] [^43] [^19], which could limit the generality of the learned representations. Discriminative approaches based on contrastive learning in the latent space have recently shown great promise, achieving state-of-the-art results [^22] [^16] [^44] [^2].

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x1.png)

Figure 1: ImageNet Top-1 accuracy of linear classifiers trained on representations learned with different self-supervised methods (pretrained on ImageNet). Gray cross indicates supervised ResNet-50. Our method, SimCLR, is shown in bold.

In this work, we introduce a simple framework for contrastive learning of visual representations, which we call SimCLR. Not only does SimCLR outperform previous work (Figure 1), but it is also simpler, requiring neither specialized architectures [^2] [^25] nor a memory bank [^54] [^52] [^24] [^41].

In order to understand what enables good contrastive representation learning, we systematically study the major components of our framework and show that:

- Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations. In addition, unsupervised contrastive learning benefits from stronger data augmentation than supervised learning.
- Introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations.
- Representation learning with contrastive cross entropy loss benefits from normalized embeddings and an appropriately adjusted temperature parameter.
- Contrastive learning benefits from larger batch sizes and longer training compared to its supervised counterpart. Like supervised learning, contrastive learning benefits from deeper and wider networks.

We combine these findings to achieve a new state-of-the-art in self-supervised and semi-supervised learning on ImageNet ILSVRC-2012 [^46]. Under the linear evaluation protocol, SimCLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of-the-art [^25]. When fine-tuned with only 1% of the ImageNet labels, SimCLR achieves 85.8% top-5 accuracy, a relative improvement of 10% [^25]. When fine-tuned on other natural image classification datasets, SimCLR performs on par with or better than a strong supervised baseline [^33] on 10 out of 12 datasets.

## 2 Method

### 2.1 The Contrastive Learning Framework

Inspired by recent contrastive learning algorithms (see Section 7 for an overview), SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. As illustrated in Figure 2, this framework comprises the following four major components.

- A stochastic data augmentation module that transforms any given data example randomly resulting in two correlated views of the same example, denoted $\tilde{\bm{x}}_{i}$ and $\tilde{\bm{x}}_{j}$, which we consider as a positive pair. In this work, we sequentially apply three simple augmentations: random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur. As shown in Section 3, the combination of random crop and color distortion is crucial to achieve a good performance.
- A neural network base encoder $f(\cdot)$ that extracts representation vectors from augmented data examples. Our framework allows various choices of the network architecture without any constraints. We opt for simplicity and adopt the commonly used ResNet [^23] to obtain $\bm{h}_{i}=f(\tilde{\bm{x}}_{i})=\mathrm{ResNet}(\tilde{\bm{x}}_{i})$ where $\bm{h}_{i}\in\mathbb{R}^{d}$ is the output after the average pooling layer.
- A small neural network projection head $g(\cdot)$ that maps representations to the space where contrastive loss is applied. We use a MLP with one hidden layer to obtain $\bm{z}_{i}=g(\bm{h}_{i})=W^{(2)}\sigma(W^{(1)}\bm{h}_{i})$ where $\sigma$ is a ReLU non-linearity. As shown in section 4, we find it beneficial to define the contrastive loss on $\bm{z}_{i}$ ’s rather than $\bm{h}_{i}$ ’s.
- A contrastive loss function defined for a contrastive prediction task. Given a set $\{\tilde{\bm{x}}_{k}\}$ including a positive pair of examples $\tilde{\bm{x}}_{i}$ and $\tilde{\bm{x}}_{j}$, the contrastive prediction task aims to identify $\tilde{\bm{x}}_{j}$ in $\{\tilde{\bm{x}}_{k}\}_{k\neq i}$ for a given $\tilde{\bm{x}}_{i}$.

<svg id="S2.F2.pic1" height="192.77" overflow="visible" version="1.1" width="255.17"><g transform="translate(0,192.77) matrix(1 0 0 -1 0 0) translate(127.59,0) translate(0,55.15)" fill="#000000" stroke="#000000" stroke-width="0.4pt"><g transform="matrix(1.0 0.0 0.0 1.0 -55.66 67.75)" fill="#000000" stroke="#000000"><foreignObject width="24.04" height="11.21" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\longleftarrow\,"><semantics id="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mo mathsize="90%" stretchy="false" id="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">⟵</mo> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">⟵</ci></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1c">\longleftarrow\,</annotation></semantics></math><span id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1" style="font-size:90%;">Representation <math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\,\longrightarrow"><semantics id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1a"><mo stretchy="false" id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1.1" xref="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1.1.cmml">⟶</mo> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1b"><ci id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1.1.cmml" xref="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1.1">⟶</ci></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1.m1.1c">\,\longrightarrow</annotation></semantics></math></span></foreignObject></g> <path d="M 15.51 -39.37 C 15.51 -30.81 8.56 -23.86 0 -23.86 C -8.56 -23.86 -15.51 -30.81 -15.51 -39.37 C -15.51 -47.93 -8.56 -54.88 0 -54.88 C 8.56 -54.88 15.51 -47.93 15.51 -39.37 Z M 0 -39.37" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 -9.78 -42.05)" fill="#000000" stroke="#000000"><foreignObject width="19.56" height="5.36" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\,~{}\bm{x}~{}\,"><semantics id="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1a"><mi mathsize="90%" id="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">𝒙</mi> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1.1">𝒙</ci></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.3.3.3.3.3.3.3.3.3.3.3.3.1.1.1.1.1.1.1.1.1.m1.1c">\,~{}\bm{x}~{}\,</annotation></semantics></math></foreignObject></g><path d="M -84.62 0 C -84.62 7.62 -90.8 13.8 -98.43 13.8 C -106.05 13.8 -112.23 7.62 -112.23 0 C -112.23 -7.62 -106.05 -13.8 -98.43 -13.8 C -90.8 -13.8 -84.62 -7.62 -84.62 0 Z M -98.43 0" style="fill:none"></path> <g transform="matrix(1.0 0.0 0.0 1.0 -103.47 -3.85)" fill="#000000" stroke="#000000"><foreignObject width="10.09" height="12.29" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\tilde{\bm{x}}_{i}"><semantics id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mover accent="true" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.2" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml">𝒙</mi> <mo mathsize="90%" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.1" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.1.cmml">~</mo></mover> <mi mathsize="90%" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">i</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <apply id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2"><ci id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.1.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.1">~</ci> <ci id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.2.2">𝒙</ci></apply> <ci id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑖</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.4.4.4.4.4.4.4.4.4.4.4.4.1.1.1.1.1.1.1.1.1.m1.1c">\tilde{\bm{x}}_{i}</annotation></semantics></math></foreignObject></g> <path d="M 112.98 0 C 112.98 8.04 106.47 14.56 98.43 14.56 C 90.39 14.56 83.87 8.04 83.87 0 C 83.87 -8.04 90.39 -14.56 98.43 -14.56 C 106.47 -14.56 112.98 -8.04 112.98 0 Z M 98.43 0" style="fill:none"></path><g transform="matrix(1.0 0.0 0.0 1.0 92.95 -3.17)" fill="#000000" stroke="#000000"><foreignObject width="10.96" height="13.65" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\tilde{\bm{x}}_{j}"><semantics id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mover accent="true" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.2" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml">𝒙</mi> <mo mathsize="90%" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.1" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.1.cmml">~</mo></mover> <mi mathsize="90%" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">j</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <apply id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2"><ci id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.1.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.1">~</ci> <ci id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.2.2">𝒙</ci></apply> <ci id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑗</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.5.5.5.5.5.5.5.5.5.5.5.5.1.1.1.1.1.1.1.1.1.m1.1c">\tilde{\bm{x}}_{j}</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 -103.21 67.69)" fill="#000000" stroke="#000000"><foreignObject width="9.58" height="10.95" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{h}_{i}"><semantics id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mi mathsize="90%" id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml">𝒉</mi> <mi mathsize="90%" id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">i</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <ci id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.2">𝒉</ci> <ci id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑖</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.6.6.6.6.6.6.6.6.6.6.6.6.1.1.1.1.1.1.1.1.1.m1.1c">\bm{h}_{i}</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 93.2 68.37)" fill="#000000" stroke="#000000"><foreignObject width="10.45" height="12.3" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{h}_{j}"><semantics id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mi mathsize="90%" id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml">𝒉</mi> <mi mathsize="90%" id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">j</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <ci id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.2">𝒉</ci> <ci id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑗</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.7.7.7.7.7.7.7.7.7.7.7.7.1.1.1.1.1.1.1.1.1.m1.1c">\bm{h}_{j}</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 -102.8 116.58)" fill="#000000" stroke="#000000"><foreignObject width="8.74" height="7.66" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{z}_{i}"><semantics id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mi mathsize="90%" id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml">𝒛</mi> <mi mathsize="90%" id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">i</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <ci id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.2">𝒛</ci> <ci id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑖</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.8.8.8.8.8.8.8.8.8.8.8.8.1.1.1.1.1.1.1.1.1.m1.1c">\bm{z}_{i}</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 93.62 117.26)" fill="#000000" stroke="#000000"><foreignObject width="9.61" height="9.02" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\bm{z}_{j}"><semantics id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1a"><msub id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mi mathsize="90%" id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml">𝒛</mi> <mi mathsize="90%" id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">j</mi></msub> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="ambiguous" id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1">subscript</csymbol> <ci id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.2">𝒛</ci> <ci id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝑗</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.9.9.9.9.9.9.9.9.9.9.9.9.1.1.1.1.1.1.1.1.1.m1.1c">\bm{z}_{j}</annotation></semantics></math></foreignObject></g> <path d="M -14.65 -33.51 L -80.73 -7.08" style="fill:none"></path><g transform="matrix(-0.92848 0.37138 -0.37138 -0.92848 -80.73 -7.08)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(0.90631 -0.42262 0.42262 0.90631 -67.54 -25.46)" fill="#000000" stroke="#000000"><foreignObject width="26.64" height="8.51" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="t\sim\mathcal{T}"><semantics id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><mi mathsize="90%" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml">t</mi> <mo mathsize="90%" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml">∼</mo> <mi mathsize="90%" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">𝒯</mi></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="latexml" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1">similar-to</csymbol> <ci id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2">𝑡</ci> <ci id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝒯</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.10.10.10.10.10.10.10.10.10.10.10.10.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1c">t\sim\mathcal{T}</annotation></semantics></math></foreignObject></g> <path d="M 14.65 -33.51 L 80.03 -7.36" style="fill:none"></path><g transform="matrix(0.92848 0.37138 -0.37138 0.92848 80.03 -7.36)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(0.90631 0.42262 -0.42262 0.90631 44.21 -37.52)" fill="#000000" stroke="#000000"><foreignObject width="25.09" height="9.6" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="t^{\prime}\sim\mathcal{T}"><semantics id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml"><msup id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.2" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml">t</mi> <mo mathsize="90%" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.3" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.3.cmml">′</mo></msup> <mo mathsize="90%" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml">∼</mo> <mi mathsize="90%" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml">𝒯</mi></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1"><csymbol cd="latexml" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.1">similar-to</csymbol> <apply id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2"><csymbol cd="ambiguous" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.1.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2">superscript</csymbol> <ci id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.2.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.2">𝑡</ci> <ci id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.3.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.2.3">′</ci></apply> <ci id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3.cmml" xref="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.3">𝒯</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.11.11.11.11.11.11.11.11.11.11.11.11.2.2.2.2.2.1.1.1.1.1.1.1.1.1.1.1.m1.1c">t^{\prime}\sim\mathcal{T}</annotation></semantics></math></foreignObject></g> <path d="M -98.43 14.08 L -98.43 55.98" style="fill:none"></path><g transform="matrix(0.0 1.0 -1.0 0.0 -98.43 55.98)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -123.44 34.41)" fill="#000000" stroke="#000000"><foreignObject width="20.58" height="12.45" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="f(\cdot)"><semantics id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml">f</mi> <mo lspace="0em" rspace="0em" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1.cmml"></mo><mrow id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.1" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">(</mo><mo lspace="0em" mathsize="90%" rspace="0em" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">⋅</mo><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.2" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2"><ci id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2">𝑓</ci> <ci id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">⋅</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.12.12.12.12.12.12.12.12.12.12.12.12.3.3.3.3.1.1.1.1.1.1.1.1.1.1.1.m1.1c">f(\cdot)</annotation></semantics></math></foreignObject></g> <path d="M 98.43 14.83 L 98.43 55.31" style="fill:none"></path><g transform="matrix(0.0 1.0 -1.0 0.0 98.43 55.31)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 102.85 34.45)" fill="#000000" stroke="#000000"><foreignObject width="20.58" height="12.45" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="f(\cdot)"><semantics id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml">f</mi> <mo lspace="0em" rspace="0em" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1.cmml"></mo><mrow id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.1" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">(</mo><mo lspace="0em" mathsize="90%" rspace="0em" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">⋅</mo><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.2" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2"><ci id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2">𝑓</ci> <ci id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">⋅</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.13.13.13.13.13.13.13.13.13.13.13.13.4.4.4.1.1.1.1.1.1.1.1.1.1.1.m1.1c">f(\cdot)</annotation></semantics></math></foreignObject></g> <path d="M -98.43 80.77 L -98.43 104.87" style="fill:none"></path><g transform="matrix(0.0 1.0 -1.0 0.0 -98.43 104.87)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -122.38 92.2)" fill="#000000" stroke="#000000"><foreignObject width="19.53" height="12.45" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="g(\cdot)"><semantics id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml">g</mi> <mo lspace="0em" rspace="0em" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1.cmml"></mo><mrow id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.1" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">(</mo><mo lspace="0em" mathsize="90%" rspace="0em" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">⋅</mo><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.2" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2"><ci id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2">𝑔</ci> <ci id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">⋅</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.14.14.14.14.14.14.14.14.14.14.14.14.5.5.1.1.1.1.1.1.1.1.1.1.1.m1.1c">g(\cdot)</annotation></semantics></math></foreignObject></g> <path d="M 98.43 81.45 L 98.43 104.19" style="fill:none"></path><g transform="matrix(0.0 1.0 -1.0 0.0 98.43 104.19)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 102.85 92.2)" fill="#000000" stroke="#000000"><foreignObject width="19.53" height="12.45" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="g(\cdot)"><semantics id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mrow id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mi mathsize="90%" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml">g</mi> <mo lspace="0em" rspace="0em" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.1.cmml"></mo><mrow id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml"><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.1" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">(</mo><mo lspace="0em" mathsize="90%" rspace="0em" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">⋅</mo><mo maxsize="90%" minsize="90%" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.3.2.2" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml">)</mo></mrow></mrow> <annotation-xml encoding="MathML-Content" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><apply id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.cmml" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2"><ci id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2.cmml" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.2.2">𝑔</ci> <ci id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">⋅</ci></apply></annotation-xml> <annotation encoding="application/x-tex" id="S2.F2.pic1.15.15.15.15.15.15.15.15.15.15.15.15.6.1.1.1.1.1.1.1.1.1.1.1.m1.1c">g(\cdot)</annotation></semantics></math></foreignObject></g> <path d="M -84.65 118.11 L 84.21 118.11" style="fill:none"></path><g transform="matrix(-1.0 0.0 0.0 -1.0 -84.65 118.11)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 84.21 118.11)"><path d="M 4.98 0 C 3.51 0.28 1.11 1.11 -0.55 2.08 L -0.55 -2.08 C 1.11 -1.11 3.51 -0.28 4.98 0" style="stroke:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 -56.45 124.96)" fill="#000000" stroke="#000000"><foreignObject width="112.46" height="10.93" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><span id="S2.F2.pic1.16.16.16.1.1" style="font-size:90%;">Maximize agreement</span></foreignObject></g></g></svg>

Figure 2: A simple framework for contrastive learning of visual representations. Two separate data augmentation operators are sampled from the same family of augmentations ($t\sim\mathcal{T}$ and $t^{\prime}\sim\mathcal{T}$) and applied to each data example to obtain two correlated views. A base encoder network $f(\cdot)$ and a projection head $g(\cdot)$ are trained to maximize agreement using a contrastive loss. After training is completed, we throw away the projection head $g(\cdot)$ and use encoder $f(\cdot)$ and representation $\bm{h}$ for downstream tasks.

We randomly sample a minibatch of $N$ examples and define the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in $2N$ data points. We do not sample negative examples explicitly. Instead, given a positive pair, similar to [^7], we treat the other $2(N-1)$ augmented examples within a minibatch as negative examples. Let $\mathrm{sim}(\bm{u},\bm{v})=\bm{u}^{\top}\bm{v}/\lVert\bm{u}\rVert\lVert\bm{v}\rVert$ denote the dot product between $\ell_{2}$ normalized $\bm{u}$ and $\bm{v}$ (i.e. cosine similarity). Then the loss function for a positive pair of examples $(i,j)$ is defined as

$$
\ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(\bm{z}_{i},\bm{z}_{j})/\tau)}{\sum_{k=1}^{2N}\mathbbm{1}_{[k\neq i]}\exp(\mathrm{sim}(\bm{z}_{i},\bm{z}_{k})/\tau)}~{},
$$

where $\mathbbm{1}_{[k\neq i]}\in\{0,1\}$ is an indicator function evaluating to $1$ iff $k\neq i$ and $\tau$ denotes a temperature parameter. The final loss is computed across all positive pairs, both $(i,j)$ and $(j,i)$, in a mini-batch. This loss has been used in previous work [^49] [^54] [^44]; for convenience, we term it NT-Xent (the normalized temperature-scaled cross entropy loss).

Algorithm 1 summarizes the proposed method.

Algorithm 1 SimCLR’s main learning algorithm.

input: batch size $N$, constant $\tau$, structure of $f$, $g$, $\mathcal{T}$.

for sampled minibatch $\{\bm{x}_{k}\}_{k=1}^{N}$ do

for all $k\in\{1,\ldots,N\}$ do

    draw two augmentation functions $t\!\sim\!\mathcal{T}$, $t^{\prime}\!\sim\!\mathcal{T}$

    # the first augmentation

     $\tilde{\bm{x}}_{2k-1}=t(\bm{x}_{k})$

     $\bm{h}_{2k-1}=f(\tilde{\bm{x}}_{2k-1})$ # representation

     $\bm{z}_{2k-1}=g({\bm{h}}_{2k-1})$ # projection

    # the second augmentation

     $\tilde{\bm{x}}_{2k}=t^{\prime}(\bm{x}_{k})$

     $\bm{h}_{2k}=f(\tilde{\bm{x}}_{2k})$ # representation

     $\bm{z}_{2k}=g({\bm{h}}_{2k})$ # projection

end for

for all $i\in\{1,\ldots,2N\}$ and $j\in\{1,\dots,2N\}$ do

     $s_{i,j}=\bm{z}_{i}^{\top}\bm{z}_{j}/(\lVert\bm{z}_{i}\rVert\lVert\bm{z}_{j}\rVert)$ # pairwise similarity

end for

define $\ell(i,j)$ as $\ell(i,j)\!=\!-\log\frac{\exp(s_{i,j}/\tau)}{\sum_{k=1}^{2N}\mathbbm{1}_{[k\neq i]}\exp(s_{i,k}/\tau)}$

 $\mathcal{L}=\frac{1}{2N}\sum_{k=1}^{N}\left[\ell(2k\!-\!1,2k)+\ell(2k,2k\!-\!1)\right]$

update networks $f$ and $g$ to minimize $\mathcal{L}$

end for

return encoder network $f(\cdot)$, and throw away $g(\cdot)$

### 2.2 Training with Large Batch Size

To keep it simple, we do not train the model with a memory bank [^54] [^24]. Instead, we vary the training batch size $N$ from 256 to 8192. A batch size of 8192 gives us 16382 negative examples per positive pair from both augmentation views. Training with large batch size may be unstable when using standard SGD/Momentum with linear learning rate scaling [^21]. To stabilize the training, we use the LARS optimizer [^58] for all batch sizes. We train our model with Cloud TPUs, using 32 to 128 cores depending on the batch size.<sup>2</sup>

Global BN. Standard ResNets use batch normalization [^29]. In distributed training with data parallelism, the BN mean and variance are typically aggregated locally per device. In our contrastive learning, as positive pairs are computed in the same device, the model can exploit the local information leakage to improve prediction accuracy without improving representations. We address this issue by aggregating BN mean and variance over all devices during the training. Other approaches include shuffling data examples across devices [^24], or replacing BN with layer norm [^25].

### 2.3 Evaluation Protocol

Here we lay out the protocol for our empirical studies, which aim to understand different design choices in our framework.

Dataset and Metrics. Most of our study for unsupervised pretraining (learning encoder network $f$ without labels) is done using the ImageNet ILSVRC-2012 dataset [^46]. Some additional pretraining experiments on CIFAR-10 [^35] can be found in Appendix B.9. We also test the pretrained results on a wide range of datasets for transfer learning. To evaluate the learned representations, we follow the widely used linear evaluation protocol [^60] [^44] [^2] [^32], where a linear classifier is trained on top of the frozen base network, and test accuracy is used as a proxy for representation quality. Beyond linear evaluation, we also compare against state-of-the-art on semi-supervised and transfer learning.

Default setting. Unless otherwise specified, for data augmentation we use random crop and resize (with random flip), color distortions, and Gaussian blur (for details, see Appendix A). We use ResNet-50 as the base encoder network, and a 2-layer MLP projection head to project the representation to a 128-dimensional latent space. As the loss, we use NT-Xent, optimized using LARS with learning rate of 4.8 ($=0.3\times\mathrm{BatchSize}/256$) and weight decay of $10^{-6}$. We train at batch size 4096 for 100 epochs.<sup>3</sup> Furthermore, we use linear warmup for the first 10 epochs, and decay the learning rate with the cosine decay schedule without restarts [^37].

## 3 Data Augmentation for Contrastive Representation Learning

<svg id="S3.F3.sf1.pic1" height="111.34" overflow="visible" version="1.1" width="111.34"><g transform="translate(0,111.34) matrix(1 0 0 -1 0 0) translate(0.55,0) translate(0,0.55)" fill="#000000" stroke="#000000"><g stroke-width="0.8pt"><path d="M 0 0 L 110.24 0 L 110.24 110.24 L 0 110.24 L 0 0" style="fill:none"></path></g><g stroke-width="0.4pt"><g stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 30.31 30.31 L 68.9 30.31 L 68.9 68.9 L 30.31 68.9 L 30.31 30.31" style="fill:none"></path></g><g stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 22.05 22.05 L 96.46 22.05 L 96.46 96.46 L 22.05 96.46 L 22.05 22.05" style="fill:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 55.44 33.85)" fill="#000000" stroke="#000000"><foreignObject width="10.38" height="9.46" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="A"><semantics id="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mi id="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">A</mi> <annotation-xml encoding="MathML-Content" id="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">𝐴</ci></annotation-xml> <annotation encoding="application/x-tex" id="S3.F3.sf1.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1c">A</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 82.59 33.85)" fill="#000000" stroke="#000000"><foreignObject width="11.19" height="9.46" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="B"><semantics id="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1a"><mi id="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">B</mi> <annotation-xml encoding="MathML-Content" id="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1">𝐵</ci></annotation-xml> <annotation encoding="application/x-tex" id="S3.F3.sf1.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1c">B</annotation></semantics></math></foreignObject></g></g></g></svg>

(a) Global and local views.

<svg id="S3.F3.sf2.pic1" height="111.34" overflow="visible" version="1.1" width="111.34"><g transform="translate(0,111.34) matrix(1 0 0 -1 0 0) translate(0.55,0) translate(0,0.55)" fill="#000000" stroke="#000000"><g stroke-width="0.8pt"><path d="M 0 0 L 110.24 0 L 110.24 110.24 L 0 110.24 L 0 0" style="fill:none"></path></g><g stroke-width="0.4pt"><g stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 13.78 13.78 L 49.61 13.78 L 49.61 49.61 L 13.78 49.61 L 13.78 13.78" style="fill:none"></path></g><g stroke-dasharray="3.0pt,3.0pt" stroke-dashoffset="0.0pt"><path d="M 55.12 55.12 L 104.72 55.12 L 104.72 104.72 L 55.12 104.72 L 55.12 55.12" style="fill:none"></path></g><g transform="matrix(1.0 0.0 0.0 1.0 35.9 20.08)" fill="#000000" stroke="#000000"><foreignObject width="10.88" height="9.46" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="C"><semantics id="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1a"><mi id="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">C</mi> <annotation-xml encoding="MathML-Content" id="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1.1">𝐶</ci></annotation-xml> <annotation encoding="application/x-tex" id="S3.F3.sf2.pic1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.1.m1.1c">C</annotation></semantics></math></foreignObject></g> <g transform="matrix(1.0 0.0 0.0 1.0 90.54 58.66)" fill="#000000" stroke="#000000"><foreignObject width="11.84" height="9.46" transform="matrix(1 0 0 -1 0 16.6)" overflow="visible"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="D"><semantics id="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1a"><mi id="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1" xref="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1.cmml">D</mi> <annotation-xml encoding="MathML-Content" id="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1b"><ci id="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1.cmml" xref="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1.1">𝐷</ci></annotation-xml> <annotation encoding="application/x-tex" id="S3.F3.sf2.pic1.2.2.2.2.2.2.2.2.2.2.2.2.1.1.1.1.1.1.1.1.1.m1.1c">D</annotation></semantics></math></foreignObject></g></g></g></svg>

(b) Adjacent views.

Figure 3: Solid rectangles are images, dashed rectangles are random crops. By randomly cropping images, we sample contrastive prediction tasks that include global to local view ($B\rightarrow A$) or adjacent view ($D\rightarrow C$) prediction.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x2.png)

(a) Original

Data augmentation defines predictive tasks. While data augmentation has been widely used in both supervised and unsupervised representation learning [^36] [^25] [^2], it has not been considered as a systematic way to define the contrastive prediction task. Many existing approaches define contrastive prediction tasks by changing the architecture. For example, [^27] [^2] achieve global-to-local view prediction via constraining the receptive field in the network architecture, whereas [^44] [^25] achieve neighboring view prediction via a fixed image splitting procedure and a context aggregation network. We show that this complexity can be avoided by performing simple random cropping (with resizing) of target images, which creates a family of predictive tasks subsuming the above mentioned two, as shown in Figure 3. This simple design choice conveniently decouples the predictive task from other components such as the neural network architecture. Broader contrastive prediction tasks can be defined by extending the family of augmentations and composing them stochastically.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x12.png)

Figure 5: Linear evaluation (ImageNet top-1 accuracy) under individual or composition of data augmentations, applied only to one branch. For all columns but the last, diagonal entries correspond to single transformation, and off-diagonals correspond to composition of two transformations (applied sequentially). The last column reflects the average over the row.

### 3.1 Composition of data augmentation operations is crucial for learning good representations

To systematically study the impact of data augmentation, we consider several common augmentations here. One type of augmentation involves spatial/geometric transformation of data, such as cropping and resizing (with horizontal flipping), rotation [^19] and cutout [^11]. The other type of augmentation involves appearance transformation, such as color distortion (including color dropping, brightness, contrast, saturation, hue) [^28] [^51], Gaussian blur, and Sobel filtering. Figure 4 visualizes the augmentations that we study in this work.

To understand the effects of individual data augmentations and the importance of augmentation composition, we investigate the performance of our framework when applying augmentations individually or in pairs. Since ImageNet images are of different sizes, we always apply crop and resize images [^36] [^51], which makes it difficult to study other augmentations in the absence of cropping. To eliminate this confound, we consider an asymmetric data transformation setting for this ablation. Specifically, we always first randomly crop images and resize them to the same resolution, and we then apply the targeted transformation(s) only to one branch of the framework in Figure 2, while leaving the other branch as the identity (i.e. $t(\bm{x}_{i})=\bm{x}_{i}$). Note that this asymmetric data augmentation hurts the performance. Nonetheless, this setup should not substantively change the impact of individual data augmentations or their compositions.

Figure 5 shows linear evaluation results under individual and composition of transformations. We observe that no single transformation suffices to learn good representations, even though the model can almost perfectly identify the positive pairs in the contrastive task. When composing augmentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically. Appendix B.2 provides a further study on composing broader set of augmentations.

One composition of augmentations stands out: random cropping and random color distortion. We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color histograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortion in order to learn generalizable features.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x13.png)

(a) Without color distortion.

### 3.2 Contrastive learning needs stronger data augmentation than supervised learning

<table><tbody><tr><td></td><td colspan="5">Color distortion strength</td><td></td></tr><tr><td>Methods</td><td>1/8</td><td>1/4</td><td>1/2</td><td>1</td><td>1 (+Blur)</td><td>AutoAug</td></tr><tr><td>SimCLR</td><td>59.6</td><td>61.0</td><td>62.6</td><td>63.2</td><td>64.5</td><td>61.1</td></tr><tr><td>Supervised</td><td>77.0</td><td>76.7</td><td>76.5</td><td>75.7</td><td>75.4</td><td>77.1</td></tr></tbody></table>

Table 1: Top-1 accuracy of unsupervised ResNet-50 using linear evaluation and supervised ResNet-50 <sup>5</sup>, under varied color distortion strength (see Appendix A) and other data transformations. Strength 1 (+Blur) is our default data augmentation policy.

To further demonstrate the importance of the color augmentation, we adjust the strength of color augmentation as shown in Table 1. Stronger color augmentation substantially improves the linear evaluation of the learned unsupervised models. In this context, AutoAugment [^10], a sophisticated augmentation policy found using supervised learning, does not work better than simple cropping + (stronger) color distortion. When training supervised models with the same set of augmentations, we observe that stronger color augmentation does not improve or even hurts their performance. Thus, our experiments show that unsupervised contrastive learning benefits from stronger (color) data augmentation than supervised learning. Although previous work has reported that data augmentation is useful for self-supervised learning [^13] [^2] [^25] [^1], we show that data augmentation that does not yield accuracy benefits for supervised learning can still help considerably with contrastive learning.

## 4 Architectures for Encoder and Head

### 4.1 Unsupervised contrastive learning benefits (more) from bigger models

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x17.png)

Figure 7: Linear evaluation of models with varied depth and width. Models in blue dots are ours trained for 100 epochs, models in red stars are ours trained for 1000 epochs, and models in green crosses are supervised ResNets trained for 90 epochs 7 Training longer does not improve supervised ResNets (see Appendix B.3 ). 23.

Figure 7 shows, perhaps unsurprisingly, that increasing depth and width both improve performance. While similar findings hold for supervised learning [^23], we find the gap between supervised models and linear classifiers trained on unsupervised models shrinks as the model size increases, suggesting that unsupervised learning benefits more from bigger models than its supervised counterpart.

| Name | Negative loss function | Gradient w.r.t. $\bm{u}$ |
| --- | --- | --- |
| NT-Xent | $\bm{u}^{T}\bm{v}^{+}/\tau-\log\sum_{\bm{v}\in\{\bm{v}^{+},\bm{v}^{-}\}}\exp(\bm{u}^{T}\bm{v}/\tau)$ | $(1-\frac{\exp(\bm{u}^{T}\bm{v}^{+}/\tau)}{Z(\bm{u})})/\tau\bm{v}^{+}-\sum_{\bm{v}^{-}}\frac{\exp(\bm{u}^{T}\bm{v}^{-}/\tau)}{Z(\bm{u})}/\tau\bm{v}^{-}$ |
| NT-Logistic | $\log\sigma(\bm{u}^{T}\bm{v}^{+}/\tau)+\log\sigma(-\bm{u}^{T}\bm{v}^{-}/\tau)$ | $(\sigma(-\bm{u}^{T}\bm{v}^{+}/\tau))/\tau\bm{v}^{+}-\sigma(\bm{u}^{T}\bm{v}^{-}/\tau)/\tau\bm{v}^{-}$ |
| Margin Triplet | $-\max(\bm{u}^{T}\bm{v}^{-}-\bm{u}^{T}\bm{v}^{+}+m,0)$ | $\bm{v}^{+}-\bm{v}^{-}\text{ if }\bm{u}^{T}\bm{v}^{+}-\bm{u}^{T}\bm{v}^{-}<m\text{ else }\bm{0}$ |

Table 2: Negative loss functions and their gradients. All input vectors, i.e. $\bm{u},\bm{v}^{+},\bm{v}^{-}$, are $\ell_{2}$ normalized. NT-Xent is an abbreviation for “Normalized Temperature-scaled Cross Entropy”. Different loss functions impose different weightings of positive and negative examples.

### 4.2 A nonlinear projection head improves the representation quality of the layer before it

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x18.png)

Figure 8: Linear evaluation of representations with different projection heads g ( ⋅ ) 𝑔 g(\\cdot) and various dimensions of 𝒛 = 𝒉 \\bm{z}=g(\\bm{h}). The representation \\bm{h} (before projection) is 2048-dimensional here.

We then study the importance of including a projection head, i.e. $g(\bm{h})$. Figure 8 shows linear evaluation results using three different architecture for the head: (1) identity mapping; (2) linear projection, as used by several previous approaches [^54]; and (3) the default nonlinear projection with one additional hidden layer (and ReLU activation), similar to [^2]. We observe that a nonlinear projection is better than a linear projection (+3%), and much better than no projection (>10%). When a projection head is used, similar results are observed regardless of output dimension. Furthermore, even when nonlinear projection is used, the layer before the projection head, $\bm{h}$, is still much better (>10%) than the layer after, $\bm{z}=g(\bm{h})$, which shows that the hidden layer before the projection head is a better representation than the layer after.

We conjecture that the importance of using the representation before the nonlinear projection is due to loss of information induced by the contrastive loss. In particular, $\bm{z}=g(\bm{h})$ is trained to be invariant to data transformation. Thus, $g$ can remove information that may be useful for the downstream task, such as the color or orientation of objects. By leveraging the nonlinear transformation $g(\cdot)$, more information can be formed and maintained in $\bm{h}$. To verify this hypothesis, we conduct experiments that use either $\bm{h}$ or $g(\bm{h})$ to learn to predict the transformation applied during the pretraining. Here we set $g(h)=W^{(2)}\sigma(W^{(1)}h)$, with the same input and output dimensionality (i.e. 2048). Table 3 shows $\bm{h}$ contains much more information about the transformation applied, while $g(\bm{h})$ loses information. Further analysis can be found in Appendix B.4.

<table><tbody><tr><td rowspan="2">What to predict?</td><td rowspan="2">Random guess</td><td colspan="2">Representation</td></tr><tr><td><math><semantics><mi>𝒉</mi> <ci>𝒉</ci> <annotation>\bm{h}</annotation></semantics></math></td><td><math><semantics><mrow><mi>g</mi> <mo></mo><mrow><mo>(</mo><mi>𝒉</mi><mo>)</mo></mrow></mrow> <apply><ci>𝑔</ci> <ci>𝒉</ci></apply> <annotation>g(\bm{h})</annotation></semantics></math></td></tr><tr><td>Color vs grayscale</td><td>80</td><td>99.3</td><td>97.4</td></tr><tr><td>Rotation</td><td>25</td><td>67.6</td><td>25.6</td></tr><tr><td>Orig. vs corrupted</td><td>50</td><td>99.5</td><td>59.6</td></tr><tr><td>Orig. vs Sobel filtered</td><td>50</td><td>96.6</td><td>56.3</td></tr></tbody></table>

Table 3: Accuracy of training additional MLPs on different representations to predict the transformation applied. Other than crop and color augmentation, we additionally and independently add rotation (one of $\{0\degree,90\degree,180\degree,270\degree\}$), Gaussian noise, and Sobel filtering transformation during the pretraining for the last three rows. Both $\bm{h}$ and $g(\bm{h})$ are of the same dimensionality, i.e. 2048.

## 5 Loss Functions and Batch Size

### 5.1 Normalized cross entropy loss with adjustable temperature works better than alternatives

We compare the NT-Xent loss against other commonly used contrastive loss functions, such as logistic loss [^40], and margin loss [^47]. Table 2 shows the objective function as well as the gradient to the input of the loss function. Looking at the gradient, we observe 1) $\ell_{2}$ normalization (i.e. cosine similarity) along with temperature effectively weights different examples, and an appropriate temperature can help the model learn from hard negatives; and 2) unlike cross-entropy, other objective functions do not weigh the negatives by their relative hardness. As a result, one must apply semi-hard negative mining [^47] for these loss functions: instead of computing the gradient over all loss terms, one can compute the gradient using semi-hard negative terms (i.e., those that are within the loss margin and closest in distance, but farther than positive examples).

To make the comparisons fair, we use the same $\ell_{2}$ normalization for all loss functions, and we tune the hyperparameters, and report their best results.<sup>8</sup> Table 4 shows that, while (semi-hard) negative mining helps, the best result is still much worse than our default NT-Xent loss.

We next test the importance of the $\ell_{2}$ normalization (i.e. cosine similarity vs dot product) and temperature $\tau$ in our default NT-Xent loss. Table 5 shows that without normalization and proper temperature scaling, performance is significantly worse. Without $\ell_{2}$ normalization, the contrastive task accuracy is higher, but the resulting representation is worse under linear evaluation.

| Margin | NT-Logi. | Margin (sh) | NT-Logi.(sh) | NT-Xent |
| --- | --- | --- | --- | --- |
| 50.9 | 51.6 | 57.5 | 57.9 | 63.9 |

Table 4: Linear evaluation (top-1) for models trained with different loss functions. “sh” means using semi-hard negative mining.

<table><tbody><tr><td><math><semantics><msub><mi>ℓ</mi> <mn>2</mn></msub> <apply><csymbol>subscript</csymbol> <ci>ℓ</ci> <cn>2</cn></apply> <annotation>\ell_{2}</annotation></semantics></math> norm?</td><td><math><semantics><mi>τ</mi> <ci>𝜏</ci> <annotation>\tau</annotation></semantics></math></td><td>Entropy</td><td>Contrastive acc.</td><td>Top 1</td></tr><tr><td rowspan="4">Yes</td><td>0.05</td><td>1.0</td><td>90.5</td><td>59.7</td></tr><tr><td>0.1</td><td>4.5</td><td>87.8</td><td>64.4</td></tr><tr><td>0.5</td><td>8.2</td><td>68.2</td><td>60.7</td></tr><tr><td>1</td><td>8.3</td><td>59.1</td><td>58.0</td></tr><tr><td rowspan="2">No</td><td>10</td><td>0.5</td><td>91.7</td><td>57.2</td></tr><tr><td>100</td><td>0.5</td><td>92.1</td><td>57.0</td></tr></tbody></table>

Table 5: Linear evaluation for models trained with different choices of $\ell_{2}$ norm and temperature $\tau$ for NT-Xent loss. The contrastive distribution is over 4096 examples.

### 5.2 Contrastive learning benefits (more) from larger batch sizes and longer training

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x19.png)

Figure 9: Linear evaluation models (ResNet-50) trained with different batch size and epochs. Each bar is a single run from scratch. 10 A linear learning rate scaling is used here. Figure B.1 shows using a square root learning rate scaling can improve performance of ones with small batch sizes.

Figure 9 shows the impact of batch size when models are trained for different numbers of epochs. We find that, when the number of training epochs is small (e.g. 100 epochs), larger batch sizes have a significant advantage over the smaller ones. With more training steps/epochs, the gaps between different batch sizes decrease or disappear, provided the batches are randomly resampled. In contrast to supervised learning [^21], in contrastive learning, larger batch sizes provide more negative examples, facilitating convergence (i.e. taking fewer epochs and steps for a given accuracy). Training longer also provides more negative examples, improving the results. In Appendix B.1, results with even longer training steps are provided.

<table><tbody><tr><td>Method</td><td>Architecture</td><td>Param (M)</td><td>Top 1</td><td>Top 5</td></tr><tr><td colspan="5">Methods using ResNet-50:</td></tr><tr><td>Local Agg.</td><td>ResNet-50</td><td>24</td><td>60.2</td><td>-</td></tr><tr><td>MoCo</td><td>ResNet-50</td><td>24</td><td>60.6</td><td>-</td></tr><tr><td>PIRL</td><td>ResNet-50</td><td>24</td><td>63.6</td><td>-</td></tr><tr><td>CPC v2</td><td>ResNet-50</td><td>24</td><td>63.8</td><td>85.3</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50</td><td>24</td><td>69.3</td><td>89.0</td></tr><tr><td colspan="5">Methods using other architectures:</td></tr><tr><td>Rotation</td><td>RevNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>86</td><td>55.4</td><td>-</td></tr><tr><td>BigBiGAN</td><td>RevNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>86</td><td>61.3</td><td>81.9</td></tr><tr><td>AMDIM</td><td>Custom-ResNet</td><td>626</td><td>68.1</td><td>-</td></tr><tr><td>CMC</td><td>ResNet-50 (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</td><td>188</td><td>68.4</td><td>88.2</td></tr><tr><td>MoCo</td><td>ResNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>375</td><td>68.6</td><td>-</td></tr><tr><td>CPC v2</td><td>ResNet-161 (<math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math>)</td><td>305</td><td>71.5</td><td>90.1</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</td><td>94</td><td>74.2</td><td>92.0</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>375</td><td>76.5</td><td>93.2</td></tr></tbody></table>

Table 6: ImageNet accuracies of linear classifiers trained on representations learned with different self-supervised methods.

<table><tbody><tr><td rowspan="3">Method</td><td rowspan="3">Architecture</td><td colspan="2">Label fraction</td></tr><tr><td>1%</td><td>10%</td></tr><tr><td colspan="2">Top 5</td></tr><tr><td>Supervised baseline</td><td>ResNet-50</td><td>48.4</td><td>80.4</td></tr><tr><td colspan="4">Methods using other label-propagation:</td></tr><tr><td>Pseudo-label</td><td>ResNet-50</td><td>51.6</td><td>82.4</td></tr><tr><td>VAT+Entropy Min.</td><td>ResNet-50</td><td>47.0</td><td>83.4</td></tr><tr><td>UDA (w. RandAug)</td><td>ResNet-50</td><td>-</td><td>88.5</td></tr><tr><td>FixMatch (w. RandAug)</td><td>ResNet-50</td><td>-</td><td>89.1</td></tr><tr><td>S4L (Rot+VAT+En. M.)</td><td>ResNet-50 (4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math>)</td><td>-</td><td>91.2</td></tr><tr><td colspan="4">Methods using representation learning only:</td></tr><tr><td>InstDisc</td><td>ResNet-50</td><td>39.2</td><td>77.4</td></tr><tr><td>BigBiGAN</td><td>RevNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>55.2</td><td>78.8</td></tr><tr><td>PIRL</td><td>ResNet-50</td><td>57.2</td><td>83.8</td></tr><tr><td>CPC v2</td><td>ResNet-161(<math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math>)</td><td>77.9</td><td>91.2</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50</td><td>75.5</td><td>87.8</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</td><td>83.0</td><td>91.2</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>85.8</td><td>92.6</td></tr></tbody></table>

Table 7: ImageNet accuracy of models trained with few labels.

<table><tbody><tr><td></td><td>Food</td><td>CIFAR10</td><td>CIFAR100</td><td>Birdsnap</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>VOC2007</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td></tr><tr><td colspan="5">Linear evaluation:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR (ours)</td><td>76.9</td><td>95.3</td><td>80.2</td><td>48.4</td><td>65.9</td><td>60.0</td><td>61.2</td><td>84.2</td><td>78.9</td><td>89.2</td><td>93.9</td><td>95.0</td></tr><tr><td>Supervised</td><td>75.2</td><td>95.7</td><td>81.2</td><td>56.4</td><td>64.9</td><td>68.8</td><td>63.8</td><td>83.8</td><td>78.7</td><td>92.3</td><td>94.1</td><td>94.2</td></tr><tr><td colspan="5">Fine-tuned:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR (ours)</td><td>89.4</td><td>98.6</td><td>89.0</td><td>78.2</td><td>68.1</td><td>92.1</td><td>87.0</td><td>86.6</td><td>77.8</td><td>92.1</td><td>94.1</td><td>97.6</td></tr><tr><td>Supervised</td><td>88.7</td><td>98.3</td><td>88.7</td><td>77.8</td><td>67.0</td><td>91.4</td><td>88.0</td><td>86.5</td><td>78.8</td><td>93.2</td><td>94.2</td><td>98.0</td></tr><tr><td>Random init</td><td>88.3</td><td>96.0</td><td>81.9</td><td>77.0</td><td>53.7</td><td>91.3</td><td>84.8</td><td>69.4</td><td>64.1</td><td>82.7</td><td>72.5</td><td>92.5</td></tr></tbody></table>

Table 8: Comparison of transfer learning performance of our self-supervised approach with supervised baselines across 12 natural image classification datasets, for ResNet-50 $(4\times)$ models pretrained on ImageNet. Results not significantly worse than the best ($p>0.05$, permutation test) are shown in bold. See Appendix B.8 for experimental details and results with standard ResNet-50.

## 6 Comparison with State-of-the-art

In this subsection, similar to [^32] [^24], we use ResNet-50 in 3 different hidden layer widths (width multipliers of $1\times$, $2\times$, and $4\times$). For better convergence, our models here are trained for 1000 epochs.

Linear evaluation. Table 6 compares our results with previous approaches [^61] [^24] [^41] [^25] [^32] [^14] [^2] [^52] in the linear evaluation setting (see Appendix B.6). Table 1 shows more numerical comparisons among different methods. We are able to use standard networks to obtain substantially better results compared to previous methods that require specifically designed architectures. The best result obtained with our ResNet-50 ($4\times$) can match the supervised pretrained ResNet-50.

Semi-supervised learning. We follow [^59] and sample 1% or 10% of the labeled ILSVRC-12 training datasets in a class-balanced way ($\sim$ 12.8 and $\sim$ 128 images per class respectively). <sup>11</sup> We simply fine-tune the whole base network on the labeled data without regularization (see Appendix B.5). Table 7 shows the comparisons of our results against recent methods [^59] [^56] [^50] [^54] [^14] [^41] [^25]. The supervised baseline from [^59] is strong due to intensive search of hyper-parameters (including augmentation). Again, our approach significantly improves over state-of-the-art with both 1% and 10% of the labels. Interestingly, fine-tuning our pretrained ResNet-50 (2 $\times,4\times$) on full ImageNet are also significantly better then training from scratch (up to 2%, see Appendix B.2).

Transfer learning. We evaluate transfer learning performance across 12 natural image datasets in both linear evaluation (fixed feature extractor) and fine-tuning settings. Following [^33], we perform hyperparameter tuning for each model-dataset combination and select the best hyperparameters on a validation set. Table 8 shows results with the ResNet-50 ($4\times$) model. When fine-tuned, our self-supervised model significantly outperforms the supervised baseline on 5 datasets, whereas the supervised baseline is superior on only 2 (i.e. Pets and Flowers). On the remaining 5 datasets, the models are statistically tied. Full experimental details as well as results with the standard ResNet-50 architecture are provided in Appendix B.8.

## 7 Related Work

The idea of making representations of an image agree with each other under small transformations dates back to [^3]. We extend it by leveraging recent advances in data augmentation, network architecture and contrastive loss. A similar consistency idea, but for class label prediction, has been explored in other contexts such as semi-supervised learning [^56] [^5].

Handcrafted pretext tasks. The recent renaissance of self-supervised learning began with artificially designed pretext tasks, such as relative patch prediction [^13], solving jigsaw puzzles [^43], colorization [^60] and rotation prediction [^19] [^8]. Although good results can be obtained with bigger networks and longer training [^32], these pretext tasks rely on somewhat ad-hoc heuristics, which limits the generality of learned representations.

Contrastive visual representation learning. Dating back to [^22], these approaches learn representations by contrasting positive pairs against negative pairs. Along these lines, [^16] proposes to treat each instance as a class represented by a feature vector (in a parametric form). [^54] proposes to use a memory bank to store the instance class representation vector, an approach adopted and extended in several recent papers [^61] [^52] [^24] [^41]. Other work explores the use of in-batch samples for negative sampling instead of a memory bank [^12] [^57] [^30].

Recent literature has attempted to relate the success of their methods to maximization of mutual information between latent representations [^44] [^25] [^27] [^2]. However, it is not clear if the success of contrastive approaches is determined by the mutual information, or by the specific form of the contrastive loss [^53].

We note that almost all individual components of our framework have appeared in previous work, although the specific instantiations may be different. The superiority of our framework relative to previous work is not explained by any single design choice, but by their composition. We provide a comprehensive comparison of our design choices with those of previous work in Appendix C.

## 8 Conclusion

In this work, we present a simple framework and its instantiation for contrastive visual representation learning. We carefully study its components, and show the effects of different design choices. By combining our findings, we improve considerably over previous methods for self-supervised, semi-supervised, and transfer learning.

Our approach differs from standard supervised learning on ImageNet only in the choice of data augmentation, the use of a nonlinear head at the end of the network, and the loss function. The strength of this simple framework suggests that, despite a recent surge in interest, self-supervised learning remains undervalued.

## Acknowledgements

We would like to thank Xiaohua Zhai, Rafael Müller and Yani Ioannou for their feedback on the draft. We are also grateful for general support from Google Research teams in Toronto and elsewhere.

## References

## Appendix A Data Augmentation Details

In our default pretraining setting (which is used to train our best models), we utilize random crop (with resize and random flip), random color distortion, and random Gaussian blur as the data augmentations. The details of these three augmentations are provided below.

##### Random crop and resize to 224x224

We use standard Inception-style random cropping [^51]. The crop of random size (uniform from 0.08 to 1.0 in area) of the original size and a random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop is finally resized to the original size. This has been implemented in Tensorflow as “ $\mathrm{slim.preprocessing.inception\_preprocessing.distorted\_bounding\_box\_crop}$ ”, or in Pytorch as “ $\mathrm{torchvision.transforms.RandomResizedCrop}$ ”. Additionally, the random crop (with resize) is always followed by a random horizontal/left-to-right flip with $50\%$ probability. This is helpful but not essential. By removing this from our default augmentation policy, the top-1 linear evaluation drops from 64.5% to 63.4% for our ResNet-50 model trained in 100 epochs.

##### Color distortion

Color distortion is composed by color jittering and color dropping. We find stronger color jittering usually helps, so we set a strength parameter.

A pseudo-code for color distortion using TensorFlow is as follows.

```
import tensorflow as tf
def color_distortion(image, s=1.0):
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.

    def color_jitter(x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(x):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])

    # randomly apply transformation with probability p.
    image = random_apply(color_jitter, image, p=0.8)
    image = random_apply(color_drop, image, p=0.2)
    return image
```

A pseudo-code for color distortion using Pytorch is as follows <sup>12</sup>.

```
from torchvision import transforms
def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort
```

##### Gaussian blur

This augmentation is in our default policy. We find it helpful, as it improves our ResNet-50 trained for 100 epochs from 63.2% to 64.5%. We blur the image 50% of the time using a Gaussian kernel. We randomly sample $\sigma\in[0.1,2.0]$, and the kernel size is set to be 10% of the image height/width.

## Appendix B Additional Experimental Results

### B.1 Batch Size and Training Steps

Figure B.2 shows the top-5 accuracy on linear evaluation when trained with different batch sizes and training epochs. The conclusion is very similar to top-1 accuracy shown before, except that the differences between different batch sizes and training steps seems slightly smaller here.

In both Figure 9 and Figure B.2, we use a linear scaling of learning rate similar to [^21] when training with different batch sizes. Although linear learning rate scaling is popular with SGD/Momentum optimizer, we find a square root learning rate scaling is more desirable with LARS optimizer. With square root learning rate scaling, we have $\mathrm{LearningRate}=0.075\times\sqrt{\mathrm{BatchSize}}$, instead of $\mathrm{LearningRate}=0.3\times\mathrm{BatchSize}/256$ in the linear scaling case, but the learning rate is the same under both scaling methods when batch size of 4096 (our default batch size). A comparison is presented in Table B.1, where we observe that square root learning rate scaling improves the performance for models trained with small batch sizes and in smaller number of epochs.

| Batch size \\ Epochs | 100 | 200 | 400 | 800 |
| --- | --- | --- | --- | --- |
| 256 | 57.5 / 62.8 | 61.9 / 64.3 | 64.7 / 65.7 | 66.6 / 66.5 |
| 512 | 60.7 / 63.8 | 64.0 / 65.6 | 66.2 / 66.7 | 67.8 / 67.4 |
| 1024 | 62.8 / 64.3 | 65.3 / 66.1 | 67.2 / 67.2 | 68.5 / 68.3 |
| 2048 | 64.0 / 64.7 | 66.1 / 66.8 | 68.1 / 67.9 | 68.9 / 68.8 |
| 4096 | 64.6 / 64.5 | 66.5 / 66.8 | 68.2 / 68.0 | 68.9 / 69.1 |
| 8192 | 64.8 / 64.8 | 66.6 / 67.0 | 67.8 / 68.3 | 69.0 / 69.1 |

Table B.1: Linear evaluation (top-1) under different batch sizes and training epochs. On the left side of slash sign are models trained with linear LR scaling, and on the right are models trained with square root LR scaling. The result is bolded if it is more than 0.5% better. Square root LR scaling works better for smaller batch size trained in fewer epochs (with LARS optimizer).

We also train with larger batch size (up to 32K) and longer (up to 3200 epochs), with the square root learning rate scaling. A shown in Figure B.2, the performance seems to saturate with a batch size of 8192, while training longer can still significantly improve the performance.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x20.png)

Figure B.1: Linear evaluation (top-5) of ResNet-50 trained with different batch sizes and epochs. Each bar is a single run from scratch. See Figure 9 for top-1 accuracy.

### B.2 Broader composition of data augmentations further improves performance

Our best results in the main text (Table 6 and 7) can be further improved when expanding the default augmentation policy to include the following: (1) Sobel filtering, (2) additional color distortion (equalize, solarize), and (3) motion blur. For linear evaluation protocol, the ResNet-50 models ($1\times,2\times,4\times$) trained with broader data augmentations achieve 70.0 (+0.7), 74.4 (+0.2), 76.8 (+0.3), respectively.

Table B.2 shows ImageNet accuracy obtained by fine-tuning the SimCLR model (see Appendix B.5 for the details of fine-tuning procedure). Interestingly, when fine-tuned on full (100%) ImageNet training set, our ResNet (4 $\times$) model achieves 80.4% top-1 / 95.4% top-5 <sup>13</sup>, which is significantly better than that (78.4% top-1 / 94.2% top-5) of training from scratch using the same set of augmentations (i.e. random crop and horizontal flip). For ResNet-50 (2 $\times$), fine-tuning our pre-trained ResNet-50 (2 $\times$) is also better than training from scratch (77.8% top-1 / 93.9% top-5). There is no improvement from fine-tuning for ResNet-50.

<table><tbody><tr><td rowspan="3">Architecture</td><td colspan="6">Label fraction</td></tr><tr><td colspan="2">1%</td><td colspan="2">10%</td><td colspan="2">100%</td></tr><tr><td>Top 1</td><td>Top 5</td><td>Top 1</td><td>Top 5</td><td>Top 1</td><td>Top 5</td></tr><tr><td>ResNet-50</td><td>49.4</td><td>76.6</td><td>66.1</td><td>88.1</td><td>76.0</td><td>93.1</td></tr><tr><td>ResNet-50 (2 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math>)</td><td>59.4</td><td>83.7</td><td>71.8</td><td>91.2</td><td>79.1</td><td>94.8</td></tr><tr><td>ResNet-50 (4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math>)</td><td>64.1</td><td>86.6</td><td>74.8</td><td>92.8</td><td>80.4</td><td>95.4</td></tr></tbody></table>

Table B.2: Classification accuracy obtained by fine-tuning the SimCLR (which is pretrained with broader data augmentations) on 1%, 10% and full of ImageNet. As a reference, our ResNet-50 (4 $\times$) trained from scratch on 100% labels achieves 78.4% top-1 / 94.2% top-5.

### B.3 Effects of Longer Training for Supervised Models

Here we perform experiments to see how training steps and stronger data augmentation affect supervised training. We test ResNet-50 and ResNet-50 (4 $\times$) under the same set of data augmentations (random crops, color distortion, 50% Gaussian blur) as used in our unsupervised models. Figure B.3 shows the top-1 accuracy. We observe that there is no significant benefit from training supervised models longer on ImageNet. Stronger data augmentation slightly improves the accuracy of ResNet-50 (4 $\times$) but does not help on ResNet-50. When stronger data augmentation is applied, ResNet-50 generally requires longer training (e.g. 500 epochs <sup>14</sup>) to obtain the optimal result, while ResNet-50 (4 $\times$) does not benefit from longer training.

<table><tbody><tr><td rowspan="2">Model</td><td rowspan="2">Training epochs</td><td colspan="3">Top 1</td></tr><tr><td>Crop</td><td>+Color</td><td>+Color+Blur</td></tr><tr><td rowspan="3">ResNet-50</td><td>90</td><td>76.5</td><td>75.6</td><td>75.3</td></tr><tr><td>500</td><td>76.2</td><td>76.5</td><td>76.7</td></tr><tr><td>1000</td><td>75.8</td><td>75.2</td><td>76.4</td></tr><tr><td rowspan="3">ResNet-50 (4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math>)</td><td>90</td><td>78.4</td><td>78.9</td><td>78.7</td></tr><tr><td>500</td><td>78.3</td><td>78.4</td><td>78.5</td></tr><tr><td>1000</td><td>77.9</td><td>78.2</td><td>78.3</td></tr></tbody></table>

Table B.3: Top-1 accuracy of supervised models trained longer under various data augmentation procedures (from the same set of data augmentations for contrastive learning).

### B.4 Understanding The Non-Linear Projection Head

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x22.png)

(a) Y-axis in uniform scale.

Figure B.4 shows the eigenvalue distribution of linear projection matrix $W\in R^{2048\times 2048}$ used to compute $\bm{z}=W\bm{h}$. This matrix has relatively few large eigenvalues, indicating that it is approximately low-rank.

Figure B.4 shows t-SNE [^38] visualizations of $\bm{h}$ and $\bm{z}=g(\bm{h})$ for randomly selected 10 classes by our best ResNet-50 (top-1 linear evaluation 69.3%). Classes represented by $\bm{h}$ are better separated compared to $\bm{z}$.

### B.5 Semi-supervised Learning via Fine-Tuning

##### Fine-tuning Procedure

We fine-tune using the Nesterov momentum optimizer with a batch size of 4096, momentum of 0.9, and a learning rate of 0.8 (following $\mathrm{LearningRate}=0.05\times\mathrm{BatchSize}/256$) without warmup. Only random cropping (with random left-to-right flipping and resizing to 224x224) is used for preprocessing. We do not use any regularization (including weight decay). For 1% labeled data we fine-tune for 60 epochs, and for 10% labeled data we fine-tune for 30 epochs. For the inference, we resize the given image to 256x256, and take a single center crop of 224x224.

Table B.4 shows the comparisons of top-1 accuracy for different methods for semi-supervised learning. Our models significantly improve state-of-the-art.

<table><tbody><tr><td rowspan="3">Method</td><td rowspan="3">Architecture</td><td colspan="2">Label fraction</td></tr><tr><td>1%</td><td>10%</td></tr><tr><td colspan="2">Top 1</td></tr><tr><td>Supervised baseline</td><td>ResNet-50</td><td>25.4</td><td>56.4</td></tr><tr><td colspan="4">Methods using label-propagation:</td></tr><tr><td>UDA (w. RandAug)</td><td>ResNet-50</td><td>-</td><td>68.8</td></tr><tr><td>FixMatch (w. RandAug)</td><td>ResNet-50</td><td>-</td><td>71.5</td></tr><tr><td>S4L (Rot+VAT+Ent. Min.)</td><td>ResNet-50 (4 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math>)</td><td>-</td><td>73.2</td></tr><tr><td colspan="4">Methods using self-supervised representation learning only:</td></tr><tr><td>CPC v2</td><td>ResNet-161(<math><semantics><mo>∗</mo> <annotation>*</annotation></semantics></math>)</td><td>52.7</td><td>73.1</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50</td><td>48.3</td><td>65.6</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>2</mn> <mo>×</mo></mrow> <annotation>2\times</annotation></semantics></math>)</td><td>58.5</td><td>71.7</td></tr><tr><td>SimCLR (ours)</td><td>ResNet-50 (<math><semantics><mrow><mn>4</mn> <mo>×</mo></mrow> <annotation>4\times</annotation></semantics></math>)</td><td>63.0</td><td>74.4</td></tr></tbody></table>

Table B.4: ImageNet top-1 accuracy of models trained with few labels. See Table 7 for top-5 accuracy.

### B.6 Linear Evaluation

For linear evaluation, we follow similar procedure as fine-tuning (described in Appendix B.5), except that a larger learning rate of 1.6 (following $\mathrm{LearningRate}=0.1\times\mathrm{BatchSize}/256$) and longer training of 90 epochs. Alternatively, using LARS optimizer with the pretraining hyper-parameters also yield similar results. Furthermore, we find that attaching the linear classifier on top of the base encoder (with a $\mathrm{stop\_gradient}$ on the input to linear classifier to prevent the label information from influencing the encoder) and train them simultaneously during the pretraining achieves similar performance.

### B.7 Correlation Between Linear Evaluation and Fine-Tuning

Here we study the correlation between linear evaluation and fine-tuning under different settings of training step and network architecture.

Figure B.5 shows linear evaluation versus fine-tuning when training epochs of a ResNet-50 (using batch size of 4096) are varied from 50 to 3200 as in Figure B.2. While they are almost linearly correlated, it seems fine-tuning on a small fraction of labels benefits more from training longer.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x26.png)

Figure B.5: Top-1 accuracy of models trained in different epochs (from Figure B.2 ), under linear evaluation and fine-tuning.

Figure B.6 shows shows linear evaluation versus fine-tuning for different architectures of choice.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x27.png)

Figure B.6: Top-1 accuracy of different architectures under linear evaluation and fine-tuning.

### B.8 Transfer Learning

We evaluated the performance of our self-supervised representation for transfer learning in two settings: linear evaluation, where a logistic regression classifier is trained to classify a new dataset based on the self-supervised representation learned on ImageNet, and fine-tuning, where we allow all weights to vary during training. In both cases, we follow the approach described by [^33], although our preprocessing differs slightly.

#### B.8.1 Methods

##### Datasets

We investigated transfer learning performance on the Food-101 dataset [^6], CIFAR-10 and CIFAR-100 [^35], Birdsnap [^4], the SUN397 scene dataset [^55], Stanford Cars [^34], FGVC Aircraft [^39], the PASCAL VOC 2007 classification task [^17], the Describable Textures Dataset (DTD) [^9], Oxford-IIIT Pets [^45], Caltech-101 [^18], and Oxford 102 Flowers [^42]. We follow the evaluation protocols in the papers introducing these datasets, i.e., we report top-1 accuracy for Food-101, CIFAR-10, CIFAR-100, Birdsnap, SUN397, Stanford Cars, and DTD; mean per-class accuracy for FGVC Aircraft, Oxford-IIIT Pets, Caltech-101, and Oxford 102 Flowers; and the 11-point mAP metric as defined in [^17] for PASCAL VOC 2007. For DTD and SUN397, the dataset creators defined multiple train/test splits; we report results only for the first split. Caltech-101 defines no train/test split, so we randomly chose 30 images per class and test on the remainder, for fair comparison with previous work [^15] [^48].

We used the validation sets specified by the dataset creators to select hyperparameters for FGVC Aircraft, PASCAL VOC 2007, DTD, and Oxford 102 Flowers. For other datasets, we held out a subset of the training set for validation while performing hyperparameter tuning. After selecting the optimal hyperparameters on the validation set, we retrained the model using the selected parameters using all training and validation images. We report accuracy on the test set.

##### Transfer Learning via a Linear Classifier

We trained an $\ell_{2}$ -regularized multinomial logistic regression classifier on features extracted from the frozen pretrained network. We used L-BFGS to optimize the softmax cross-entropy objective and we did not apply data augmentation. As preprocessing, all images were resized to 224 pixels along the shorter side using bicubic resampling, after which we took a $224\times 224$ center crop. We selected the $\ell_{2}$ regularization parameter from a range of 45 logarithmically spaced values between $10^{-6}$ and $10^{5}$.

##### Transfer Learning via Fine-Tuning

We fine-tuned the entire network using the weights of the pretrained network as initialization. We trained for 20,000 steps at a batch size of 256 using SGD with Nesterov momentum with a momentum parameter of 0.9. We set the momentum parameter for the batch normalization statistics to $\max(1-10/s,0.9)$ where $s$ is the number of steps per epoch. As data augmentation during fine-tuning, we performed only random crops with resize and flips; in contrast to pretraining, we did not perform color augmentation or blurring. At test time, we resized images to 256 pixels along the shorter side and took a $224\times 224$ center crop. (Additional accuracy improvements may be possible with further optimization of data augmentation, particularly on the CIFAR-10 and CIFAR-100 datasets.) We selected the learning rate and weight decay, with a grid of 7 logarithmically spaced learning rates between 0.0001 and 0.1 and 7 logarithmically spaced values of weight decay between $10^{-6}$ and $10^{-3}$, as well as no weight decay. We divide these values of weight decay by the learning rate.

##### Training from Random Initialization

We trained the network from random initialization using the same procedure as for fine-tuning, but for longer, and with an altered hyperparameter grid. We chose hyperparameters from a grid of 7 logarithmically spaced learning rates between 0.001 and 1.0 and 8 logarithmically spaced values of weight decay between $10^{-5}$ and $10^{-1.5}$. Importantly, our random initialization baselines are trained for 40,000 steps, which is sufficiently long to achieve near-maximal accuracy, as demonstrated in Figure 8 of [^33].

On Birdsnap, there are no statistically significant differences among methods, and on Food-101, Stanford Cars, and FGVC Aircraft datasets, fine-tuning provides only a small advantage over training from random initialization. However, on the remaining 8 datasets, pretraining has clear advantages.

##### Supervised Baselines

We compare against architecturally identical ResNet models trained on ImageNet with standard cross-entropy loss. These models are trained with the same data augmentation as our self-supervised models (crops, strong color augmentation, and blur) and are also trained for 1000 epochs. We found that, although stronger data augmentation and longer training time do not benefit accuracy on ImageNet, these models performed significantly better than a supervised baseline trained for 90 epochs and ordinary data augmentation for linear evaluation on a subset of transfer datasets. The supervised ResNet-50 baseline achieves 76.3% top-1 accuracy on ImageNet, vs. 69.3% for the self-supervised counterpart, while the ResNet-50 ($4\times$) baseline achieves 78.3%, vs. 76.5% for the self-supervised model.

##### Statistical Significance Testing

We test for the significance of differences between model with a permutation test. Given predictions of two models, we generate 100,000 samples from the null distribution by randomly exchanging predictions for each example and computing the difference in accuracy after performing this randomization. We then compute the percentage of samples from the null distribution that are more extreme than the observed difference in predictions. For top-1 accuracy, this procedure yields the same result as the exact McNemar test. The assumption of exchangeability under the null hypothesis is also valid for mean per-class accuracy, but not when computing average precision curves. Thus, we perform significance testing for a difference in accuracy on VOC 2007 rather than a difference in mAP. A caveat of this procedure is that it does not consider run-to-run variability when training the models, only variability arising from using a finite sample of images for evaluation.

#### B.8.2 Results with Standard ResNet

<table><tbody><tr><td></td><td>Food</td><td>CIFAR10</td><td>CIFAR100</td><td>Birdsnap</td><td>SUN397</td><td>Cars</td><td>Aircraft</td><td>VOC2007</td><td>DTD</td><td>Pets</td><td>Caltech-101</td><td>Flowers</td></tr><tr><td colspan="5">Linear evaluation:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR (ours)</td><td>68.4</td><td>90.6</td><td>71.6</td><td>37.4</td><td>58.8</td><td>50.3</td><td>50.3</td><td>80.5</td><td>74.5</td><td>83.6</td><td>90.3</td><td>91.2</td></tr><tr><td>Supervised</td><td>72.3</td><td>93.6</td><td>78.3</td><td>53.7</td><td>61.9</td><td>66.7</td><td>61.0</td><td>82.8</td><td>74.9</td><td>91.5</td><td>94.5</td><td>94.7</td></tr><tr><td colspan="5">Fine-tuned:</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SimCLR (ours)</td><td>88.2</td><td>97.7</td><td>85.9</td><td>75.9</td><td>63.5</td><td>91.3</td><td>88.1</td><td>84.1</td><td>73.2</td><td>89.2</td><td>92.1</td><td>97.0</td></tr><tr><td>Supervised</td><td>88.3</td><td>97.5</td><td>86.4</td><td>75.8</td><td>64.3</td><td>92.1</td><td>86.0</td><td>85.0</td><td>74.6</td><td>92.1</td><td>93.3</td><td>97.6</td></tr><tr><td>Random init</td><td>86.9</td><td>95.9</td><td>80.2</td><td>76.1</td><td>53.6</td><td>91.4</td><td>85.9</td><td>67.3</td><td>64.8</td><td>81.5</td><td>72.6</td><td>92.0</td></tr></tbody></table>

Table B.5: Comparison of transfer learning performance of our self-supervised approach with supervised baselines across 12 natural image datasets, using ImageNet-pretrained ResNet models. See also Figure 8 for results with the ResNet ($4\times$) architecture.

The ResNet-50 ($4\times$) results shown in Table 8 of the text show no clear advantage to the supervised or self-supervised models. With the narrower ResNet-50 architecture, however, supervised learning maintains a clear advantage over self-supervised learning. The supervised ResNet-50 model outperforms the self-supervised model on all datasets with linear evaluation, and most (10 of 12) datasets with fine-tuning. The weaker performance of the ResNet model compared to the ResNet ($4\times$) model may relate to the accuracy gap between the supervised and self-supervised models on ImageNet. The self-supervised ResNet gets 69.3% top-1 accuracy, 6.8% worse than the supervised model in absolute terms, whereas the self-supervised ResNet $(4\times)$ model gets 76.5%, which is only 1.8% worse than the supervised model.

### B.9 CIFAR-10

While we focus on using ImageNet as the main dataset for pretraining our unsupervised model, our method also works with other datasets. We demonstrate it by testing on CIFAR-10 as follows.

##### Setup

As our goal is not to optimize CIFAR-10 performance, but rather to provide further confirmation of our observations on ImageNet, we use the same architecture (ResNet-50) for CIFAR-10 experiments. Because CIFAR-10 images are much smaller than ImageNet images, we replace the first 7x7 Conv of stride 2 with 3x3 Conv of stride 1, and also remove the first max pooling operation. For data augmentation, we use the same Inception crop (flip and resize to 32x32) as ImageNet,<sup>15</sup> and color distortion (strength=0.5), leaving out Gaussian blur. We pretrain with learning rate in $\{0.5,1.0,1.5\}$, temperature in $\{0.1,0.5,1.0\}$, and batch size in $\{256,512,1024,2048,4096\}$. The rest of the settings (including optimizer, weight decay, etc.) are the same as our ImageNet training.

Our best model trained with batch size 1024 can achieve a linear evaluation accuracy of 94.0%, compared to 95.1% from the supervised baseline using the same architecture and batch size. The best self-supervised model that reports linear evaluation result on CIFAR-10 is AMDIM [^2], which achieves 91.2% with a model $25\times$ larger than ours. We note that our model can be improved by incorporating extra data augmentations as well as using a more suitable base network.

##### Performance under different batch sizes and training steps

Figure B.7 shows the linear evaluation performance under different batch sizes and training steps. The results are consistent with our observations on ImageNet, although the largest batch size of 4096 seems to cause a small degradation in performance on CIFAR-10.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x29.png)

Figure B.7: Linear evaluation of ResNet-50 (with adjusted stem) trained with different batch size and epochs on CIFAR-10 dataset. Each bar is averaged over 3 runs with different learning rates (0.5, 1.0, 1.5) and temperature τ = 0.5 𝜏 \\tau=0.5. Error bar denotes standard deviation.

##### Optimal temperature under different batch sizes

Figure B.8 shows the linear evaluation of model trained with three different temperatures under various batch sizes. We find that when training to convergence (e.g. training epochs > 300), the optimal temperature in $\{0.1,0.5,1.0\}$ is 0.5 and seems consistent regardless of the batch sizes. However, the performance with $\tau=0.1$ improves as batch size increases, which may suggest a small shift of optimal temperature towards 0.1.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2002.05709/assets/x30.png)

(a) Training epochs ≤ \\leq 300

### B.10 Tuning For Other Loss Functions

The learning rate that works best for NT-Xent loss may not be a good learning rate for other loss functions. To ensure a fair comparison, we also tune hyperparameters for both margin loss and logistic loss. Specifically, we tune learning rate in $\{0.01,0.1,0.3,0.5,1.0\}$ for both loss functions. We further tune the margin in $\{0,0.4,0.8,1.6\}$ for margin loss, the temperature in $\{0.1,0.2,0.5,1.0\}$ for logistic loss. For simplicity, we only consider the negatives from one augmentation view (instead of both sides), which slightly impairs performance but ensures fair comparison.

## Appendix C Further Comparison to Related Methods

As we have noted in the main text, most individual components of SimCLR have appeared in previous work, and the improved performance is a result of a combination of these design choices. Table C.1 provides a high-level comparison of the design choices of our method with those of previous methods. Compared with previous work, our design choices are generally simpler.

| Model | Data Augmentation | Base Encoder | Projection Head | Loss | Batch Size | Train Epochs |
| --- | --- | --- | --- | --- | --- | --- |
| CPC v2 | Custom | ResNet-161 (modified) | PixelCNN | Xent | 512 <sup>#</sup> | $\sim$ 200 |
| AMDIM | Fast AutoAug. | Custom ResNet | Non-linear MLP | Xent w/ clip,reg | 1008 <sup>#</sup> | 150 |
| CMC | Fast AutoAug. | ResNet-50 ($2\times$, L+ab) | Linear layer | Xent w/ $\ell_{2},\tau$ | 156 <sup>∗</sup> | 280 |
| MoCo | Crop+color | ResNet-50 (4 $\times$) | Linear layer | Xent w/ $\ell_{2},\tau$ | 256 <sup>∗</sup> | 200 |
| PIRL | Crop+color | ResNet-50 (2 $\times$) | Linear layer | Xent w/ $\ell_{2},\tau$ | 1024 <sup>∗</sup> | 800 |
| SimCLR | Crop+color+blur | ResNet-50 ($4\times$) | Non-linear MLP | Xent w/ $\ell_{2},\tau$ | 4096 | 1000 |

Table C.1: A high-level comparison of design choices and training setup (for best result on ImageNet) for each method. Note that descriptions provided here are general; even when they match for two methods, formulations and implementations may differ (e.g. for color augmentation). Refer to the original papers for more details. <sup>#</sup> Examples are split into multiple patches, which enlarges the effective batch size. <sup>∗</sup> A memory bank is employed.

In below, we provide an in-depth comparison of our method to the recently proposed contrastive representation learning methods:

- DIM/AMDIM [^27] [^2] achieve global-to-local/local-to-neighbor prediction by predicting the middle layer of ConvNet. The ConvNet is a ResNet that has bewen modified to place significant constraints on the receptive fields of the network (e.g. replacing many 3x3 Convs with 1x1 Convs). In our framework, we decouple the prediction task and encoder architecture, by random cropping (with resizing) and using the final representations of two augmented views for prediction, so we can use standard and more powerful ResNets. Our NT-Xent loss function leverages normalization and temperature to restrict the range of similarity scores, whereas they use a tanh function with regularization. We use a simpler data augmentation policy, while they use FastAutoAugment for their best result.
- CPC v1 and v2 [^44] [^25] define the context prediction task using a deterministic strategy to split examples into patches, and a context aggregation network (a PixelCNN) to aggregate these patches. The base encoder network sees only patches, which are considerably smaller than the original image. We decouple the prediction task and the encoder architecture, so we do not require a context aggregation network, and our encoder can look at the images of wider spectrum of resolutions. In addition, we use the NT-Xent loss function, which leverages normalization and temperature, whereas they use an unnormalized cross-entropy-based objective. We use simpler data augmentation.
- InstDisc, MoCo, PIRL [^54] [^24] [^41] generalize the Exemplar approach originally proposed by [^16] and leverage an explicit memory bank. We do not use a memory bank; we find that, with a larger batch size, in-batch negative example sampling suffices. We also utilize a nonlinear projection head, and use the representation before the projection head. Although we use similar types of augmentations (e.g., random crop and color distortion), we expect specific parameters may be different.
- CMC [^52] uses a separated network for each view, while we simply use a single network shared for all randomly augmented views. The data augmentation, projection head and loss function are also different. We use larger batch size instead of a memory bank.
- Whereas [^57] maximize similarity between augmented and unaugmented copies of the same image, we apply data augmentation symmetrically to both branches of our framework (Figure 2). We also apply a nonlinear projection on the output of base feature network, and use the representation before projection network, whereas [^57] use the linearly projected final hidden vector as the representation. When training with large batch sizes using multiple accelerators, we use global BN to avoid shortcuts that can greatly decrease representation quality.

[^1]: Asano, Y. M., Rupprecht, C., and Vedaldi, A. A critical analysis of self-supervision, or what we can learn from a single image. *arXiv preprint arXiv:1904.13132*, 2019.

[^2]: Bachman, P., Hjelm, R. D., and Buchwalter, W. Learning representations by maximizing mutual information across views. In *Advances in Neural Information Processing Systems*, pp. 15509–15519, 2019.

[^3]: Becker, S. and Hinton, G. E. Self-organizing neural network that discovers surfaces in random-dot stereograms. *Nature*, 355(6356):161–163, 1992.

[^4]: Berg, T., Liu, J., Lee, S. W., Alexander, M. L., Jacobs, D. W., and Belhumeur, P. N. Birdsnap: Large-scale fine-grained visual categorization of birds. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 2019–2026. IEEE, 2014.

[^5]: Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A., and Raffel, C. A. Mixmatch: A holistic approach to semi-supervised learning. In *Advances in Neural Information Processing Systems*, pp. 5050–5060, 2019.

[^6]: Bossard, L., Guillaumin, M., and Van Gool, L. Food-101–mining discriminative components with random forests. In *European conference on computer vision*, pp. 446–461. Springer, 2014.

[^7]: Chen, T., Sun, Y., Shi, Y., and Hong, L. On sampling strategies for neural network-based collaborative filtering. In *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 767–776, 2017.

[^8]: Chen, T., Zhai, X., Ritter, M., Lucic, M., and Houlsby, N. Self-supervised gans via auxiliary rotation loss. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 12154–12163, 2019.

[^9]: Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3606–3613. IEEE, 2014.

[^10]: Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., and Le, Q. V. Autoaugment: Learning augmentation strategies from data. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 113–123, 2019.

[^11]: DeVries, T. and Taylor, G. W. Improved regularization of convolutional neural networks with cutout. *arXiv preprint arXiv:1708.04552*, 2017.

[^12]: Doersch, C. and Zisserman, A. Multi-task self-supervised visual learning. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 2051–2060, 2017.

[^13]: Doersch, C., Gupta, A., and Efros, A. A. Unsupervised visual representation learning by context prediction. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 1422–1430, 2015.

[^14]: Donahue, J. and Simonyan, K. Large scale adversarial representation learning. In *Advances in Neural Information Processing Systems*, pp. 10541–10551, 2019.

[^15]: Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., and Darrell, T. Decaf: A deep convolutional activation feature for generic visual recognition. In *International Conference on Machine Learning*, pp. 647–655, 2014.

[^16]: Dosovitskiy, A., Springenberg, J. T., Riedmiller, M., and Brox, T. Discriminative unsupervised feature learning with convolutional neural networks. In *Advances in neural information processing systems*, pp. 766–774, 2014.

[^17]: Everingham, M., Van Gool, L., Williams, C. K., Winn, J., and Zisserman, A. The pascal visual object classes (voc) challenge. *International Journal of Computer Vision*, 88(2):303–338, 2010.

[^18]: Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshop on Generative-Model Based Vision*, 2004.

[^19]: Gidaris, S., Singh, P., and Komodakis, N. Unsupervised representation learning by predicting image rotations. *arXiv preprint arXiv:1803.07728*, 2018.

[^20]: Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. Generative adversarial nets. In *Advances in neural information processing systems*, pp. 2672–2680, 2014.

[^21]: Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., and He, K. Accurate, large minibatch sgd: Training imagenet in 1 hour. *arXiv preprint arXiv:1706.02677*, 2017.

[^22]: Hadsell, R., Chopra, S., and LeCun, Y. Dimensionality reduction by learning an invariant mapping. In *2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06)*, volume 2, pp. 1735–1742. IEEE, 2006.

[^23]: He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

[^24]: He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. *arXiv preprint arXiv:1911.05722*, 2019.

[^25]: Hénaff, O. J., Razavi, A., Doersch, C., Eslami, S., and Oord, A. v. d. Data-efficient image recognition with contrastive predictive coding. *arXiv preprint arXiv:1905.09272*, 2019.

[^26]: Hinton, G. E., Osindero, S., and Teh, Y.-W. A fast learning algorithm for deep belief nets. *Neural computation*, 18(7):1527–1554, 2006.

[^27]: Hjelm, R. D., Fedorov, A., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., and Bengio, Y. Learning deep representations by mutual information estimation and maximization. *arXiv preprint arXiv:1808.06670*, 2018.

[^28]: Howard, A. G. Some improvements on deep convolutional neural network based image classification. *arXiv preprint arXiv:1312.5402*, 2013.

[^29]: Ioffe, S. and Szegedy, C. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*, 2015.

[^30]: Ji, X., Henriques, J. F., and Vedaldi, A. Invariant information clustering for unsupervised image classification and segmentation. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 9865–9874, 2019.

[^31]: Kingma, D. P. and Welling, M. Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*, 2013.

[^32]: Kolesnikov, A., Zhai, X., and Beyer, L. Revisiting self-supervised visual representation learning. In *Proceedings of the IEEE conference on Computer Vision and Pattern Recognition*, pp. 1920–1929, 2019.

[^33]: Kornblith, S., Shlens, J., and Le, Q. V. Do better ImageNet models transfer better? In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 2661–2671, 2019.

[^34]: Krause, J., Deng, J., Stark, M., and Fei-Fei, L. Collecting a large-scale dataset of fine-grained cars. In *Second Workshop on Fine-Grained Visual Categorization*, 2013.

[^35]: Krizhevsky, A. and Hinton, G. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009. URL [https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).

[^36]: Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems*, pp. 1097–1105, 2012.

[^37]: Loshchilov, I. and Hutter, F. Sgdr: Stochastic gradient descent with warm restarts. *arXiv preprint arXiv:1608.03983*, 2016.

[^38]: Maaten, L. v. d. and Hinton, G. Visualizing data using t-sne. *Journal of machine learning research*, 9(Nov):2579–2605, 2008.

[^39]: Maji, S., Kannala, J., Rahtu, E., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. Technical report, 2013.

[^40]: Mikolov, T., Chen, K., Corrado, G., and Dean, J. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*, 2013.

[^41]: Misra, I. and van der Maaten, L. Self-supervised learning of pretext-invariant representations. *arXiv preprint arXiv:1912.01991*, 2019.

[^42]: Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In *Computer Vision, Graphics & Image Processing, 2008. ICVGIP’08. Sixth Indian Conference on*, pp. 722–729. IEEE, 2008.

[^43]: Noroozi, M. and Favaro, P. Unsupervised learning of visual representations by solving jigsaw puzzles. In *European Conference on Computer Vision*, pp. 69–84. Springer, 2016.

[^44]: Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. *arXiv preprint arXiv:1807.03748*, 2018.

[^45]: Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. Cats and dogs. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3498–3505. IEEE, 2012.

[^46]: Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., et al. Imagenet large scale visual recognition challenge. *International journal of computer vision*, 115(3):211–252, 2015.

[^47]: Schroff, F., Kalenichenko, D., and Philbin, J. Facenet: A unified embedding for face recognition and clustering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 815–823, 2015.

[^48]: Simonyan, K. and Zisserman, A. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

[^49]: Sohn, K. Improved deep metric learning with multi-class n-pair loss objective. In *Advances in neural information processing systems*, pp. 1857–1865, 2016.

[^50]: Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H., and Raffel, C. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. *arXiv preprint arXiv:2001.07685*, 2020.

[^51]: Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 1–9, 2015.

[^52]: Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. *arXiv preprint arXiv:1906.05849*, 2019.

[^53]: Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., and Lucic, M. On mutual information maximization for representation learning. *arXiv preprint arXiv:1907.13625*, 2019.

[^54]: Wu, Z., Xiong, Y., Yu, S. X., and Lin, D. Unsupervised feature learning via non-parametric instance discrimination. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 3733–3742, 2018.

[^55]: Xiao, J., Hays, J., Ehinger, K. A., Oliva, A., and Torralba, A. Sun database: Large-scale scene recognition from abbey to zoo. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 3485–3492. IEEE, 2010.

[^56]: Xie, Q., Dai, Z., Hovy, E., Luong, M.-T., and Le, Q. V. Unsupervised data augmentation. *arXiv preprint arXiv:1904.12848*, 2019.

[^57]: Ye, M., Zhang, X., Yuen, P. C., and Chang, S.-F. Unsupervised embedding learning via invariant and spreading instance feature. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, pp. 6210–6219, 2019.

[^58]: You, Y., Gitman, I., and Ginsburg, B. Large batch training of convolutional networks. *arXiv preprint arXiv:1708.03888*, 2017.

[^59]: Zhai, X., Oliver, A., Kolesnikov, A., and Beyer, L. S4l: Self-supervised semi-supervised learning. In *The IEEE International Conference on Computer Vision (ICCV)*, October 2019.

[^60]: Zhang, R., Isola, P., and Efros, A. A. Colorful image colorization. In *European conference on computer vision*, pp. 649–666. Springer, 2016.

[^61]: Zhuang, C., Zhai, A. L., and Yamins, D. Local aggregation for unsupervised learning of visual embeddings. In *Proceedings of the IEEE International Conference on Computer Vision*, pp. 6002–6012, 2019.