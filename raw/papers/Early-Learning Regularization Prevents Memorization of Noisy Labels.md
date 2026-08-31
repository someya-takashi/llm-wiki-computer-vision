---
title: "Early-Learning Regularization Prevents Memorization of Noisy Labels"
source: "https://ar5iv.labs.arxiv.org/html/2007.00151"
author:
published:
created: 2026-08-30
description: "We propose a novel framework to perform classification via deep learning in the presence of noisy annotations. When trained on noisy labels, deep neural networks have been observed to first fit the training data with c…"
tags:
  - "clippings"
---
Jonathan Niles-Weed Affiliation: Center for Data Science, and Affiliation: Courant Inst. of Mathematical Sciences Affiliation: New York University    Narges Razavian Affiliation: Department of Population Health, and Affiliation: Department of Radiology Affiliation: NYU School of Medicine    Carlos Fernandez-Granda Affiliation: Center for Data Science, and Affiliation: Courant Inst. of Mathematical Sciences Affiliation: New York University

###### Abstract

We propose a novel framework to perform classification via deep learning in the presence of noisy annotations. When trained on noisy labels, deep neural networks have been observed to first fit the training data with clean labels during an “early learning” phase, before eventually memorizing the examples with false labels. We prove that early learning and memorization are fundamental phenomena in high-dimensional classification tasks, even in simple linear models, and give a theoretical explanation in this setting. Motivated by these findings, we develop a new technique for noisy classification tasks, which exploits the progress of the early learning phase. In contrast with existing approaches, which use the model output during early learning to detect the examples with clean labels, and either ignore or attempt to correct the false labels, we take a different route and instead capitalize on early learning via regularization. There are two key elements to our approach. First, we leverage semi-supervised learning techniques to produce target probabilities based on the model outputs. Second, we design a regularization term that steers the model towards these targets, implicitly preventing memorization of the false labels. The resulting framework is shown to provide robustness to noisy annotations on several standard benchmarks and real-world datasets, where it achieves results comparable to the state of the art.

## 1 Introduction

Deep neural networks have become an essential tool for classification tasks [^19] [^15] [^11]. These models tend to be trained on large curated datasets such as CIFAR-10 [^18] or ImageNet [^9], where the vast majority of labels have been manually verified. Unfortunately, in many applications such datasets are not available, due to the cost or difficulty of manual labeling (e.g. [^13] [^32] [^25] [^1]). However, datasets with lower quality annotations, obtained for instance from online queries [^5] or crowdsourcing [^49] [^53], may be available. Such annotations inevitably contain numerous mistakes or *label noise*. It is therefore of great importance to develop methodology that is robust to the presence of noisy annotations.

When trained on noisy labels, deep neural networks have been observed to first fit the training data with clean labels during an *early learning* phase, before eventually *memorizing* the examples with false labels [^3] [^54]. In this work we study this phenomenon and introduce a novel framework that exploits it to achieve robustness to noisy labels. Our main contributions are the following:

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/True_label_behavior_04.png)

Refer to caption

## 2 Related Work

In this section we describe existing techniques to train deep-learning classification models using data with noisy annotations. We focus our discussion on methods that do not assume the availability of small subsets of training data with clean labels (as opposed, for example, to [^16] [^34] [^41]). We also assume that the correct classes are known (as opposed to [^44]).

*Robust-loss* methods propose cost functions specifically designed to be robust in the presence of noisy labels. These include Mean Absolute Error (MAE) [^10], Improved MAE [^43], which is a reweighted MAE, Generalized Cross Entropy [^56], which can be interpreted as a generalization of MAE, Symmetric Cross Entropy [^45], which adds a reverse cross-entropy term to the usual cross-entropy loss, and $\mathcal{L}_{\text{DIM}}$ [^48], which is based on information-theoretic considerations. *Loss-correction* methods explicitly correct the loss function to take into account the noise distribution, represented by a transition matrix of mislabeling probabilities [^31] [^12] [^46] [^39].

Robust-loss and loss-correction techniques do not exploit the early-learning phenomenon mentioned in the introduction. This phenomenon was described in [^3] (see also [^54]), and analyzed theoretically in [^23]. Our theoretical approach differs from theirs in two respects. First, Ref. [^23] focus on a least squares regression task, whereas we focus on the noisy label problem in classification. Second, and more importantly, we prove that early learning and memorization occur even in a *linear* model.

Early learning can be exploited through *sample selection*, where the model output during the early-learning stage is used to predict which examples are mislabeled and which have been labeled correctly. The prediction is based on the observation that mislabeled examples tend to have higher loss values. Co-teaching [^14] [^52] performs sample selection by using two networks, each trained on a subset of examples that have a small training loss for the other network (see [^17] [^28] for related approaches). A limitation of this approach is that the examples that are selected tend to be *easier*, in the sense that the model output during early learning approaches the true label. As a result, the gradient of the cross-entropy with respect to these examples is small, which slows down learning [^6]. In addition, the subset of selected examples may not be rich enough to generalize effectively to held-out data [^35].

An alternative to sample selection is *label correction*. During the early-learning stage the model predictions are accurate on a subset of the mislabeled examples (see the top row of Figure 1). This suggests correcting the corresponding labels. This can be achieved by computing new labels equal to the probabilities estimated by the model (known as *soft labels*) or to one-hot vectors representing the model predictions (*hard labels*) [^38] [^51]. Another option is to set the new labels to equal a convex combination of the noisy labels and the soft or hard labels [^33]. Label correction is usually combined with some form of iterative sample selection [^2] [^27] [^35] [^22] or with additional regularization terms [^38]. SELFIE [^35] uses label replacement to correct a subset of the labels selected by considering past model outputs. Ref. [^27] computes a different convex combination with hard labels for each example based on a measure of model dimensionality. Ref. [^2] fits a two-component mixture model to carry out sample selection, and then corrects labels via convex combination as in [^33]. They also apply mixup data augmentation [^55] to enhance performance. In a similar spirit, DivideMix [^22] uses two networks to perform sample selection via a two-component mixture model, and applies the semi-supervised learning technique MixMatch [^4].

Our proposed approach is somewhat related in spirit to label correction. We compute a probability estimate that is analogous to the soft labels mentioned above, and then exploit it to avoid memorization. However it is also fundamentally different: instead of modifying the labels, we propose a novel regularization term explicitly designed to correct the gradient of the cross-entropy cost function. This yields strong empirical performance, without needing to incorporate sample selection.

## 3 Early learning as a general phenomenon of high-dimensional classification

As the top row of Figure 1 makes clear, deep neural networks trained with noisy labels make progress during the early learning stage before memorization occurs. In this section, we show that far from being a peculiar feature of deep neural networks, this phenomenon is intrinsic to high-dimensional classification tasks, even in the simplest setting. Our theoretical analysis is also the inspiration for the early-learning regularization procedure we propose in Section 4.

We exhibit a simple *linear* model with noisy labels which evinces the same behavior as described above: the *early learning* stage, when the classifier learns to correctly predict the true labels, even on noisy examples, and the *memorization* stage, when the classifier begins to make incorrect predictions because it memorizes the wrong labels. This is illustrated in Figure A.1, which demonstrates that empirically the linear model has the same qualitative behavior as the deep-learning model in Figure 1. We show that this behavior arises because, early in training, the gradients corresponding to the correctly labeled examples dominate the dynamics—leading to early progress towards the true optimum—but that the gradients corresponding to wrong labels soon become dominant—at which point the classifier simply learns to fit the noisy labels.

We consider data drawn from a mixture of two Gaussians in $\mathbb{R}^{p}$. The (clean) dataset consists of $n$ i.i.d. copies of $(\mathbf{x},\mathbf{y}^{*})$. The label $\mathbf{y}^{*}\in\{0,1\}^{2}$ is a one-hot vector representing the cluster assignment, and

$$
\displaystyle\mathbf{x}
$$
 
$$
\displaystyle\sim\mathcal{N}(+\mathbf{v},\sigma^{2}I_{p\times p})\hskip 10.00002pt\text{ if $\mathbf{y}^{*}=(1,0)$}
$$
 
$$
\displaystyle\mathbf{x}
$$
 
$$
\displaystyle\sim\mathcal{N}(-\mathbf{v},\sigma^{2}I_{p\times p})\hskip 10.00002pt\text{ if $\mathbf{y}^{*}=(0,1)$}\,,
$$

where $\mathbf{v}$ is an arbitrary unit vector in $\mathbb{R}^{p}$ and $\sigma^{2}$ is a small constant. The optimal separator between the two classes is a hyperplane through the origin perpendicular to $\mathbf{v}$. We focus on the setting where $\sigma^{2}$ is fixed while $n,p\to\infty$. In this regime, the classification task is nontrivial, since the clusters are, approximately, two spheres whose centers are separated by 2 units with radii $\sigma\sqrt{p}\gg 2$.

We only observe a dataset with noisy labels $(\mathbf{y}^{[1]},\dots,\mathbf{y}^{[n]})$,

$$
\displaystyle\mathbf{y}^{[i]}=\left\{\begin{array}[]{ll}(\mathbf{y}^{*})^{[i]}&\text{with probability $1-\Delta$}\\
\tilde{\mathbf{y}}^{[i]}&\text{with probability $\Delta$,}\end{array}\right.
$$

where $\{\tilde{\mathbf{y}}^{[i]}\}_{i=1}^{n}$ are i.i.d. random one-hot vectors which take values $(1,0)$ and $(0,1)$ with equal probability.

We train a linear classifier by gradient descent on the cross entropy:

$$
\min_{\Theta\in\mathbb{R}^{2\times p}}\mathcal{L}_{\text{CE}}(\Theta):=-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{2}\mathbf{y}^{[i]}_{c}\log(\mathcal{S}(\Theta\mathbf{x}^{[i]})_{c})\,,
$$

where $\mathcal{S}:\mathbb{R}^{2}\to[0,1]^{2}$ is a softmax function. In order to separate the true classes well (and not overfit to the noisy labels), the rows of $\Theta$ should be correlated with the vector $\mathbf{v}$.

The gradient of this loss with respect to the model parameters $\Theta$ corresponding to class $c$ reads

$$
\displaystyle\nabla\mathcal{L}_{\text{CE}}(\Theta)_{c}
$$
 
$$
\displaystyle=\frac{1}{n}\sum_{i=1}^{n}\mathbf{x}^{[i]}\left(\mathcal{S}(\Theta\mathbf{x}^{[i]})_{c}-\mathbf{y}^{[i]}_{c}\right),
$$

Each term in the gradient therefore corresponds to a weighted sum of the examples $\mathbf{x}^{[i]}$, where the weighting depends on the agreement between $\mathcal{S}(\Theta\mathbf{x}^{[i]})_{c}$ and $\mathbf{y}^{[i]}_{c}$.

Our main theoretical result shows that this linear model possesses the properties described above. During the early-learning stage, the algorithm makes progress and the accuracy on wrongly labeled examples increases. However, during this initial stage, the relative importance of the wrongly labeled examples continues to grow; once the effect of the wrongly labeled examples begins to dominate, memorization occurs.

###### Theorem 1 (Informal).

Denote by $\{\Theta_{t}\}$ the iterates of gradient descent with step size $\eta$. For any $\Delta\in(0,1)$, there exists a constant $\sigma_{\Delta}$ such that, if $\sigma\leq\sigma_{\Delta}$ and $p/n\in(1-\Delta/2,1)$, then with probability $1-o(1)$ as $n,p\to\infty$ there exists a $T=\Omega(1/\eta)$ such that:

- Early learning succeeds: For $t<T$, $-\nabla\mathcal{L}(\Theta_{t})$ is well correlated with the correct separator $\mathbf{v}$, and at $t=T$ the classifier has higher accuracy on the wrongly labeled examples than at initialization.
- Gradients from correct examples vanish: Between $t=0$ and $t=T$, the magnitudes of the coefficients $\left(\mathcal{S}(\Theta_{t}\mathbf{x}^{[i]})_{c}-\mathbf{y}^{[i]}_{c}\right)$ corresponding to examples with clean labels decreases while the magnitudes of the coefficients for examples with wrong labels increases.
- Memorization occurs: As $t\to\infty$, the classifier $\Theta_{t}$ memorizes all noisy labels.

Due to space constraints, we defer the formal statement of Theorem 1 and its proof to the supplementary material.

The proof of Theorem 1 is based on two observations. First, while $\Theta$ is still not well correlated with $\mathbf{v}$, the coefficients $\mathcal{S}(\Theta\mathbf{x}^{[i]})_{c}-\mathbf{y}^{[i]}_{c}$ are similar for all $i$, so that $\nabla\mathcal{L}_{\text{CE}}$ points approximately in the average direction of the examples. Since the majority of data points are correctly labeled, this means the gradient is still well correlated with the correct direction during the early learning stage. Second, once $\Theta$ becomes correlated with $\mathbf{v}$, the gradient begins to point in directions orthogonal to the correct direction $\mathbf{v}$; when the dimension is sufficiently large, there are enough of these orthogonal directions to allow the classifier to completely memorize the noisy labels.

This analysis suggests that in order to learn on the correct labels and avoid memorization it is necessary to (1) ensure that the contribution to the gradient from examples with clean labels remains large, and (2) neutralize the influence of the examples with wrong labels on the gradient. In Section 4 we propose a method designed to achieve this via regularization.

## 4 Methodology

### 4.1 Gradient analysis of softmax classification from noisy labels

In this section we explain the connection between the linear model from Section 3 and deep neural networks. Recall the gradient of the cross-entropy loss with respect to $\Theta$ given in (3). Performing gradient descent modifies the parameters iteratively to push $\mathcal{S}(\Theta\mathbf{x}^{[i]})$ closer to $\mathbf{y}^{[i]}$. If $c$ is the true class so that $\mathbf{y}^{[i]}_{c}=1$, the contribution of the $i$ th example to $\nabla\mathcal{L}_{\text{CE}}(\Theta)_{c}$ is aligned with $-\mathbf{x}^{[i]}$, and gradient descent moves in the direction of $\mathbf{x}^{[i]}$. However, if the label is noisy and $\mathbf{y}^{[i]}_{c}=0$, then gradient descent moves in the opposite direction, which eventually leads to memorization as established by Theorem 1.

We now show that for nonlinear models based on neural networks, the effect of label noise is analogous. We consider a classification problem with $C$ classes, where the training set consists of $n$ examples $\{\mathbf{x}^{[i]},\mathbf{y}^{[i]}\}_{i=1}^{n}$, $\mathbf{x}^{[i]}\in\mathbb{R}^{d}$ is the $i$ th input and $\mathbf{y}^{[i]}\in\{0,1\}^{C}$ is a one-hot label vector indicating the corresponding class. The classification model maps each input $\mathbf{x}^{[i]}$ to a $C$ -dimensional encoding using a deep neural network $\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)$ and then feeds the encoding into a softmax function $\mathcal{S}$ to produce an estimate $\mathbf{p}^{[i]}$ of the conditional probability of each class given $\mathbf{x}^{[i]}$,

$$
\displaystyle\mathbf{p}^{[i]}:=\mathcal{S}\left(\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\right).
$$

$\Theta$ denotes the parameters of the neural network. The gradient of the cross-entropy loss,

$$
\displaystyle\mathcal{L}_{\text{CE}}(\Theta)
$$
 
$$
\displaystyle:=-\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}\mathbf{y}^{[i]}_{c}\log\mathbf{p}^{[i]}_{c},
$$

with respect to $\Theta$ equals

$$
\displaystyle\nabla\mathcal{L}_{\text{CE}}(\Theta)
$$
 
$$
\displaystyle=\frac{1}{n}\sum_{i=1}^{n}\nabla\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\left(\mathbf{p}^{[i]}-\mathbf{y}^{[i]}\right),
$$

where $\nabla\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)$ is the Jacobian matrix of the neural-network encoding for the $i$ th input with respect to $\Theta$. Here we see that label noise has the same effect as in the simple linear model. If $c$ is the true class, but $\mathbf{y}^{[i]}_{c}=0$ due to the noise, then the contribution of the $i$ th example to $\nabla\mathcal{L}_{\text{CE}}(\Theta)_{c}$ is reversed. The entry corresponding to the *impostor* class $c^{\prime}$, is also reversed because $\mathbf{y}^{[i]}_{c^{\prime}}=1$. As a result, performing stochastic gradient descent eventually results in memorization, as in the linear model (see Figures 1 and A.1). Crucially, the influence of the label noise on the gradient of the cross-entropy loss is restricted to the term $\mathbf{p}^{[i]}-\mathbf{y}^{[i]}$ (see Figure B.1). In Section 4.2 we describe how to counteract this influence by exploiting the early-learning phenomenon.

### 4.2 Early-learning regularization

In this section we present a novel framework for learning from noisy labels called early-learning regularization (ELR). We assume that we have available a *target* <sup>1</sup> vector of probabilities $\mathbf{t}^{[i]}$ for each example $i$, which is computed using past outputs of the model. Section 4.3 describes several techniques to compute the targets. Here we explain how to use them to avoid memorization.

Due to the early-learning phenomenon, we assume that at the beginning of the optimization process the targets do not overfit the noisy labels. ELR exploits this using a regularization term that seeks to maximize the inner product between the model output and the targets,

$$
\displaystyle\mathcal{L}_{\text{ELR}}(\Theta)
$$
 
$$
\displaystyle:=\mathcal{L}_{\text{CE}}(\Theta)+\frac{\lambda}{n}\sum_{i=1}^{n}\log\left(1-\langle\mathbf{p}^{[i]},\mathbf{t}^{[i]}\rangle\right).
$$

The logarithm in the regularization term counteracts the exponential function implicit in the softmax function in $\mathbf{p}^{[i]}$. A possible alternative to this approach would be to penalize the Kullback-Leibler divergence between the model outputs and the targets. However, this does not exploit the early-learning phenomenon effectively, because it leads to overfitting the targets as demonstrated in Section C.

The key to understanding why ELR is effective lies in its gradient, derived in the following lemma, which is proved in Section E.

###### Lemma 2 (Gradient of the ELR loss).

The gradient of the loss defined in Eq. (7) is equal to

$$
\displaystyle\nabla\mathcal{L}_{\text{ELR}}(\Theta)
$$
 
$$
\displaystyle=\frac{1}{n}\sum_{i=1}^{n}\nabla\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\left(\mathbf{p}^{[i]}-\mathbf{y}^{[i]}+\lambda\mathbf{g}^{[i]}\right)
$$

where the entries of $\mathbf{g}^{[i]}\in\mathbb{R}^{C}$ are given by

$$
\displaystyle\mathbf{g}^{[i]}_{c}:=\frac{\mathbf{p}^{[i]}_{c}}{1-\langle\mathbf{p}^{[i]},\mathbf{t}^{[i]}\rangle}\sum_{k=1}^{C}(\mathbf{t}_{k}^{[i]}-\mathbf{t}_{c}^{[i]})\mathbf{p}_{k}^{[i]},\hskip 20.00003pt1\leq c\leq C.
$$
![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/NN_grad_true_logit_ours_sum.png)

Refer to caption

In words, the sign of $\mathbf{g}^{[i]}_{c}$ is determined by a weighted combination of the difference between $\mathbf{t}_{c}^{[i]}$ and the rest of the entries in the target.

If $c^{\ast}$ is the true class, then the $c^{\ast}$ th entry of $\mathbf{t}^{[i]}$ tends to be dominant during early-learning. In that case, the $c^{\ast}$ th entry of $\mathbf{g}^{[i]}$ is negative. This is useful both for examples with clean labels and for those with wrong labels. For examples with clean labels, the cross-entropy term $\mathbf{p}^{[i]}-\mathbf{y}^{[i]}$ tends to vanish after the early-learning stage because $\mathbf{p}^{[i]}$ is very close to $\mathbf{y}^{[i]}$, allowing examples with wrong labels to dominate the gradient. Adding $\mathbf{g}^{[i]}$ counteracts this effect by ensuring that the magnitudes of the coefficients on examples with clean labels remains large. The center image of Figure 2 shows this effect. For examples with wrong labels, the cross entropy term $\mathbf{p}^{[i]}_{c^{\ast}}-\mathbf{y}_{c^{\ast}}^{[i]}$ is positive because $\mathbf{y}_{c^{\ast}}^{[i]}=0$. Adding the negative term $\mathbf{g}^{[i]}_{c^{\ast}}$ therefore dampens the coefficients on these mislabeled examples, thereby diminishing their effect on the gradient (see right image in Figure 2). Thus, ELR fulfils the two desired properties outlined at the end of Section 3: boosting the gradient of examples with clean labels, and neutralizing the gradient of the examples with false labels.

### 4.3 Target estimation

ELR requires a target probability for each example in the training set. The target can be set equal to the model output, but using a running average is more effective. In semi-supervised learning, this technique is known as temporal ensembling [^20]. Let $\mathbf{t}^{[i]}(k)$ and $\mathbf{p}^{[i]}(k)$ denote the target and model output respectively for example $i$ at iteration $k$ of training. We set

$$
\displaystyle\mathbf{t}^{[i]}(k):=\beta\mathbf{t}^{[i]}(k-1)+(1-\beta)\mathbf{p}^{[i]}(k),
$$

where $0\leq\beta<1$ is the momentum. The basic version of our proposed method alternates between computing the targets and minimizing the cost function (7) via stochastic gradient descent.

Target estimation can be further improved in two ways. First, by using the output of a model obtained through a running average of the model weights during training. In semi-supervised learning, this *weight averaging* approach has been proposed to mitigate confirmation bias [^40]. Second, by using two separate neural networks, where the target of each network is computed from the output of the other network. The approach is inspired by Co-teaching and related methods [^14] [^52] [^22]. The ablation results in Section 6 show that weight averaging, two networks, and mixup data augmentation [^55] all separately improve performance. We call the combination of all these elements ELR+. A detailed description of ELR and ELR+ is provided in Section F of the supplementary material.

<table><thead><tr><th rowspan="2">Datasets (Architecture)</th><th rowspan="2">Methods</th><th colspan="4">Symmetric label noise</th><th colspan="4">Asymmetric label noise</th></tr><tr><th>20%</th><th>40%</th><th>60%</th><th>80%</th><th>10%</th><th>20%</th><th>30%</th><th>40%</th></tr></thead><tbody><tr><th rowspan="7">CIFAR10 (ResNet34)</th><th>Cross entropy</th><td>86.98 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.12</td><td>81.88 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.29</td><td>74.14 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.56</td><td>53.82 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.04</td><td>90.69 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.17</td><td>88.59 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.34</td><td>86.14 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.40</td><td>80.11 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.44</td></tr><tr><th>Bootstrap <sup><a href="#fn:33">33</a></sup></th><td>86.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.23</td><td>82.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.37</td><td>75.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.56</td><td>54.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.32</td><td>90.32 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.21</td><td>88.26 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td><td>86.57 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.35</td><td>81.21 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.47</td></tr><tr><th>Forward <sup><a href="#fn:31">31</a></sup></th><td>87.99 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.36</td><td>83.25 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.38</td><td>74.96 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.65</td><td>54.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.44</td><td>90.52 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td><td>89.09 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.47</td><td>86.79 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.36</td><td>83.55 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.58</td></tr><tr><th>GSE <sup><a href="#fn:56">56</a></sup></th><td>89.83 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.20</td><td>87.13 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>82.54 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.23</td><td>64.07 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.38</td><td>90.91 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>89.33 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.17</td><td>85.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.74</td><td>76.74 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.61</td></tr><tr><th>SL <sup><a href="#fn:45">45</a></sup></th><td>89.83 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.32</td><td>87.13 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td><td>82.81 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.61</td><td>68.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.81</td><td>91.72 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.31</td><td>90.44 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.27</td><td>88.48 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.46</td><td>82.51 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.45</td></tr><tr><th>ELR</th><td>91.16 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.08</td><td>89.15 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.17</td><td>86.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.49</td><td>73.86 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.61</td><td>93.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.11</td><td>93.52 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.23</td><td>91.89 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>90.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.47</td></tr><tr><th>ELR <sup><math><semantics><mo>⋆</mo> <annotation>\star</annotation></semantics></math></sup></th><td>92.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.35</td><td>91.43 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.21</td><td>88.87 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td><td>80.69 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.57</td><td>94.57 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.23</td><td>93.28 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.19</td><td>92.70 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.41</td><td>90.35 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.38</td></tr><tr><th rowspan="7">CIFAR100 (ResNet34)</th><th>Cross entropy</th><td>58.72 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td><td>48.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.65</td><td>37.41 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.94</td><td>18.10 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.82</td><td>66.54 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.42</td><td>59.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.18</td><td>51.40 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.16</td><td>42.74 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.61</td></tr><tr><th>Bootstrap <sup><a href="#fn:33">33</a></sup></th><td>58.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.21</td><td>47.66 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.55</td><td>34.68 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.1</td><td>21.64 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.97</td><td>67.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.78</td><td>62.14 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.32</td><td>52.87 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.19</td><td>45.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.57</td></tr><tr><th>Forward <sup><a href="#fn:31">31</a></sup></th><td>39.19 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.61</td><td>31.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.44</td><td>19.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.95</td><td>8.99 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.58</td><td>45.96 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.21</td><td>42.46 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.16</td><td>38.13 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 2.97</td><td>34.44 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.93</td></tr><tr><th>GSE <sup><a href="#fn:56">56</a></sup></th><td>66.81 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.42</td><td>61.77 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td><td>53.16 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.78</td><td>29.16 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.74</td><td>68.36 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.42</td><td>66.59 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>61.45 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td><td>47.22 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 1.15</td></tr><tr><th>SL <sup><a href="#fn:45">45</a></sup></th><td>70.38 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.13</td><td>62.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>54.82 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.57</td><td>25.91 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.44</td><td>73.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>72.56 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>72.12 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.24</td><td>69.32 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.87</td></tr><tr><th>ELR</th><td>74.21 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>68.28 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.31</td><td>59.28 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.67</td><td>29.78 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.56</td><td>74.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.31</td><td>74.03 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.31</td><td>73.71 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.22</td><td>73.26 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.64</td></tr><tr><th>ELR <sup><math><semantics><mo>⋆</mo> <annotation>\star</annotation></semantics></math></sup></th><td>74.68 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.31</td><td>68.43 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.42</td><td>60.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.78</td><td>30.27 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.86</td><td>74.52 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.32</td><td>74.20 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.25</td><td>74.02 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.33</td><td>73.73 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.34</td></tr></tbody></table>

- Results with cosine annealing learning rate.

Table 1: Comparison with state-of-the-art methods on CIFAR-10 and CIFAR-100 with symmetric and asymmetric label noise. The bootstrap and SL methods were reimplemented using publicly available code, the rest of results are taken from [^56]. The mean accuracy and its standard deviation are computed over five noise realizations.

## 5 Experiments

We evaluate the proposed methodology on two standard benchmarks with simulated label noise, CIFAR-10 and CIFAR-100 [^18], and two real-world datasets, Clothing1M [^47] and WebVision [^24]. For CIFAR-10 and CIFAR-100 we simulate label noise by randomly flipping a certain fraction of the labels in the training set following a symmetric uniform distribution (as in Eq. (3)), as well as a more realistic asymmetric class-dependent distribution, following the scheme proposed in [^31]. Clothing1M consists of 1 million training images collected from online shopping websites with labels generated using surrounding text. Its noise level is estimated at $38.5\%$ [^36]. For ease of comparison to previous works [^17] [^7], we consider the mini WebVision dataset which contains the top 50 classes from the Google image subset of WebVision, which results in approximately 66 thousand images. The noise level of WebVision is estimated at $20\%$ [^24]. Table G.1 in the supplementary material reports additional details about the datasets, and our training, validation and test splits.

In our experiments, we prioritize making our results comparable to the existing literature. When possible we use the same preprocessing, and architectures as previous methods. The details are described in Section G of the supplementary material. We focus on two variants of the proposed approach: ELR with temporal ensembling, which we call ELR, and ELR with temporal ensembling, weight averaging, two networks, and mixup data augmentation, which we call ELR+ (see Section F). The choice of hyperparameters is performed on separate validation sets. Section H shows that the sensitivity to different hyperparameters is quite low. Finally, we also perform an ablation study on CIFAR-10 for two levels of symmetric noise (40% and 80%) in order to evaluate the contribution of the different elements in ELR+. Code to reproduce the experiments is publicly available online at [https://github.com/shengliu66/ELR](https://github.com/shengliu66/ELR).

## 6 Results

\[t\] Cross entropy Co-teaching+ [^52] Mixup [^55] PENCIL [^51] MD-DYR-SH [^2] DivideMix [^22] ELR+ ELR+ <sup><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="\ast"><semantics><mo>∗</mo> <annotation>\ast</annotation></semantics></math></sup> CIFAR-10 Sym. label noise 20% 86.8 89.5 95.6 92.4 94.0 96.1 94.6 95.8 50% 79.4 85.7 87.1 89.1 92.0 94.6 93.8 94.8 80% 62.9 67.4 71.6 77.5 86.8 93.2 91.1 93.3 90% 42.7 47.9 52.2 58.9 69.1 76.0 75.2 78.7 Asym. 40% 83.2 - - 88.5 87.4 93.4 92.7 93.0 CIFAR-100 Sym. label noise 20% 62.0 65.6 67.8 69.4 73.9 77.3 77.5 77.6 50% 46.7 51.8 57.3 57.5 66.1 74.6 72.4 73.6 80% 19.9 27.9 30.8 31.1 48.2 60.2 58.2 60.8 90% 10.1 13.7 14.6 15.3 24.3 31.5 30.8 33.4 Asym. 40% - - - - - 72.1 76.5 77.5

Table 2: Comparison with state-of-the-art methods on CIFAR-10 and CIFAR-100 with symmetric and asymmetric noise. For ELR+, we use 10% of the training set for validation, and treat the validation set as a held-out test set. The result for DivideMix on CIFAR-100 with 40% asymmetric noise was obtained using publicly available code. The rest of the results are taken from [^22], which reports the highest accuracy observed on the validation set during training. We also report the performance of ELR+ under this metric on the rightmost column (ELR+ <sup>∗</sup>).

Table 1 evaluates the performance of ELR on CIFAR-10 and CIFAR-100 with different levels of symmetric and asymmetric label noise. We compare to the best performing methods that only modify the training loss. All techniques use the same architecture (ResNet34), batch size, and training procedure. ELR consistently outperforms the rest by a significant margin. To illustrate the influence of the training procedure, we include results with a different learning-rate scheduler (cosine annealing [^26]), which further improves the results.

In Table 2, we compare ELR+ to state-of-the-art methods, which also apply sample selection and data augmentation, on CIFAR-10 and CIFAR-100. All methods use the same architecture (PreAct ResNet-18). The results from other methods may not be completely comparable to ours because they correspond to the best test performance during training, whereas we use a separate validation set. Nevertheless, ELR+ outperforms all other methods except DivideMix.

| CE | Forward [^31] | GCE [^56] | SL [^45] | Joint-Optim [^38] | DivideMix [^22] | ELR | ELR+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 69.10 | 69.84 | 69.75 | 71.02 | 72.16 | 74.76 | 72.87 | 74.81 |

Table 3: Comparison with state-of-the-art methods in test accuracy (%) on Clothing1M. All methods use a ResNet-50 architecture pretrained on ImageNet. Results of other methods are taken from the original papers (except for GCE, which is taken from [^45]).

Table 3 compares ELR and ELR+ to state-of-the-art methods on the Clothing1M dataset. ELR+ achieves state-of-the-art performance, slightly superior to DivideMix.

<table><thead><tr><th></th><th></th><th>D2L <sup><a href="#fn:27">27</a></sup></th><th>MentorNet <sup><a href="#fn:17">17</a></sup></th><th>Co-teaching <sup><a href="#fn:14">14</a></sup></th><th>Iterative-CV <sup><a href="#fn:44">44</a></sup></th><th>DivideMix <sup><a href="#fn:22">22</a></sup></th><th>ELR</th><th>ELR+</th></tr></thead><tbody><tr><td rowspan="2">WebVision</td><td>top1</td><td>62.68</td><td>63.00</td><td>63.58</td><td>65.24</td><td>77.32</td><td>76.26</td><td>77.78</td></tr><tr><td>top5</td><td>84.00</td><td>81.40</td><td>85.20</td><td>85.34</td><td>91.64</td><td>91.26</td><td>91.68</td></tr><tr><td rowspan="2">ILSVRC12</td><td>top1</td><td>57.80</td><td>57.80</td><td>61.48</td><td>61.60</td><td>75.20</td><td>68.71</td><td>70.29</td></tr><tr><td>top5</td><td>81.36</td><td>79.92</td><td>84.70</td><td>84.98</td><td>90.84</td><td>87.84</td><td>89.76</td></tr></tbody></table>

Table 4: Comparison with state-of-the-art methods trained on the mini WebVision dataset. Results of other methods are taken from [^22]. All methods use an InceptionResNetV2 architecture.

Table 4 compares ELR and ELR+ to state-of-the-art methods trained on the mini WebVision dataset and evaluated on both the WebVision and ImageNet ILSVRC12 validation sets. ELR+ achieves state-of-the-art performance, slightly superior to DivideMix, on WebVision. ELR also performs strongly, despite its simplicity. On ILSVRC12 DivideMix produces superior results (particularly in terms of top1 accuracy).

<table><tbody><tr><td></td><td></td><td></td><td colspan="2">40%</td><td colspan="2">80%</td></tr><tr><td></td><td></td><td></td><td colspan="2">Weight Averaging</td><td colspan="2">Weight Averaging</td></tr><tr><td></td><td></td><td></td><td>✓</td><td>✗</td><td>✓</td><td>✗</td></tr><tr><td rowspan="2">1 Network</td><td rowspan="2">mixup</td><td>✓</td><td>93.04 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.12</td><td>91.05 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.13</td><td>87.23 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.30</td><td>81.43 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.52</td></tr><tr><td>✗</td><td>92.09 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.08</td><td>90.83 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.07</td><td>76.50 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.65</td><td>72.54 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.35</td></tr><tr><td rowspan="2">2 Networks</td><td rowspan="2">mixup</td><td>✓</td><td>93.68 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.51</td><td>93.51 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.47</td><td>88.62 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td><td>84.75 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.26</td></tr><tr><td>✗</td><td>92.95 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.05</td><td>91.86 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.14</td><td>80.13 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.51</td><td>73.49 <math><semantics><mo>±</mo> <annotation>\pm</annotation></semantics></math> 0.47</td></tr></tbody></table>

Table 5: Ablation study evaluating the influence of weight averaging, the use of two networks, and mixup data augmentation for the CIFAR-10 dataset with medium (40%) and high (80%) levels of symmetric noise. The mean accuracy and its standard deviation are computed over five noise realizations.

Table 5 shows the results of an ablation study evaluating the influence of the different elements of ELR+ for the CIFAR-10 dataset with medium (40%) and high (80%) levels of symmetric noise. Each element seems to provide an independent performance boost. At the medium noise level the improvement is modest, but at the high noise level it is very significant. This is in line with recent works showing the effectiveness of semi-supervised learning techniques in such settings [^2] [^22].

## 7 Discussion and Future Work

In this work we provide a theoretical characterization of the early-learning and memorization phenomena for a linear generative model, and build upon the resulting insights to propose a novel framework for learning from data with noisy annotations. Our proposed methodology yields strong results on standard benchmarks and real-world datasets for several different network architectures. However, there remain multiple open problems for future research. On the theoretical front, it would be interesting to bridge the gap between linear and nonlinear models (see [^23] for some work in this direction), and also to investigate the dynamics of the proposed regularization scheme. On the methodological front, we hope that our work will trigger interest in the design of new forms of regularization that provide robustness to label noise.

## 8 Broader Impact

This work has the potential to advance the development of machine-learning methods that can be deployed in contexts where it is costly to gather accurate annotations. This is an important issue in applications such as medicine, where machine learning has great potential societal impact.

#### Acknowledgments

This research was supported by NSF NRT-HDR Award 1922658. SL was partially supported by NSF grant DMS 2009752. CFG was partially supported by NSF Award HDR-1940097. JNW gratefully acknowledges the support of the Institute for Advanced Study, where a portion of this research was conducted.

## References

## Appendix A Theoretical analysis of early learning and memorization in a linear model

In this section, we formalize and substantiate the claims of Theorem 1.

Theorem 1 has three parts, which we address in the following sections. First, in Section A.2, we show that the classifier makes progress during the early-learning phase: over the first $T$ iterations, the gradient is well correlated with $\mathbf{v}$ and the accuracy on mislabeled examples increases. However, as noted in the main text, this early progress halts because the gradient terms corresponding to correctly labeled examples begin to disappear. We prove this rigorously in Section A.3, which shows that the overall magnitude of the gradient terms corresponding to correctly labeled examples shrinks over the first $T$ iterations. Finally, in Section A.4, we prove the claimed asymptotic behavior: as $t\to\infty$, gradient descent perfectly memorizes the noisy labels.

### A.1 Notation and setup

We consider a softmax regression model parameterized by two weight vectors $\Theta_{1}$ and $\Theta_{2}$, which are the rows of the parameter matrix $\Theta\in\mathbb{R}^{2\times p}$. In the linear case this is equivalent to a logistic regression model, because the cross-entropy loss on two classes depends only on the vector $\Theta_{1}-\Theta_{2}$. If we reparametrize the labels as

$$
\mathbf{\varepsilon}^{[i]}=\left\{\begin{array}[]{ll}1&\text{ if $\mathbf{y}^{[i]}_{1}=1$}\\
-1&\text{ if $\mathbf{y}^{[i]}_{2}=1$}\,,\end{array}\right.
$$

and set $\theta:=\Theta_{1}-\Theta_{2}$, we can then write the loss as

$$
\mathcal{L}_{\text{CE}}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\log(1+e^{-\mathbf{\varepsilon}^{[i]}\theta^{\top}\mathbf{x}^{[i]}})\,.
$$

We write $\mathbf{\varepsilon}^{\ast}$ for the true cluster assignments: $(\mathbf{\varepsilon}^{\ast})^{[i]}=1$ if $\mathbf{x}^{[i]}$ comes from the cluster with mean $+\mathbf{v}$, and $(\mathbf{\varepsilon}^{\ast})^{[i]}=-1$ otherwise. Note that, with this convention, we can always write $\mathbf{x}^{[i]}=(\mathbf{\varepsilon}^{\ast})^{[i]}(\mathbf{v}-\sigma\mathbf{z}^{[i]})$, where $\mathbf{z}^{[i]}$ is a standard Gaussian random vector independent of all other random variables.

In terms of $\theta$ and $\mathbf{\varepsilon}$, the gradient (3) reads

$$
\displaystyle\nabla\mathcal{L}_{\text{CE}}(\theta)
$$
 
$$
\displaystyle=\frac{1}{2n}\sum_{i=1}^{n}\mathbf{x}^{[i]}\left(\tanh(\theta^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]}\right),
$$

As noted in the main text, the coefficient $\tanh(\theta^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]}$ is the key quantity governing the properties of the gradient.

Let us write $C$ for the set of indices for which the labels are correct, and $W$ for the set of indices for which labels are wrong.

We assume that $\theta_{0}$ is initialized randomly on the sphere with radius $2$, and then optimized to minimize $\mathcal{L}$ via gradient descent with fixed step size $\eta<1$. We denote the iterates by $\theta_{t}$.

We consider the asymptotic regime where $\sigma\ll 1$ and $\Delta$ are constants and $p,n\to\infty$, with $p/n\in(1-\Delta/2,1)$. We will let $\sigma_{\Delta}$ denote a constant, whose precise value may change from proposition to proposition; however, in all cases the requirements on $\sigma$ will be *independent* of $p$ and $n$. For convenience, we assume that $\Delta\leq 1/2$, though it is straightforward to extend the analysis below to any $\Delta$ bounded away from $1$. Note that when $\Delta=1$, each observed label is independent of the data, so no learning is possible. We will use the phrase “with high probability” to denote an event which happens with probability $1-o(1)$ as $n,p\to\infty$, and we use $o_{P}(1)$ to denote a random quantity which converges to $0$ in probability. We use the symbol $c$ to refer to an unspecified positive constant whose value may change from line to line. We use subscripts to indicate when this constant depends on other parameters of the problem.

We let $T$ be the smallest positive integer such that $\theta_{T}^{\top}\mathbf{v}\geq 1/10$. By Lemmas 7 and 8 in Section A.5, $T=\Omega(1/\eta)$ with high probability.

### A.2 Early-learning succeeds

We first show that, for the first $T$ iterations, the negative gradient $-\nabla\mathcal{L}_{\text{CE}}(\theta_{t})^{\top}$ has constant correlation with $\mathbf{v}$. (Note that, by contrast, a *random* vector in $\mathbb{R}^{p}$ typically has *negligible* correlation with $\mathbf{v}$.)

###### Proposition 3.

There exists a constant $\sigma_{\Delta}$, depending only on $\Delta$, such that if $\sigma\leq\sigma_{\Delta}$ then with high probability, for all $t<T$, we have $\|\theta_{t}-\theta_{0}\|\leq 1$ and

$$
-\nabla\mathcal{L}_{\text{CE}}(\theta_{t})^{\top}\mathbf{v}/\|\nabla\mathcal{L}_{\text{CE}}(\theta_{t})\|\geq 1/6\,.
$$

###### Proof.

We will prove the claim by induction. We write

$$
-\nabla\mathcal{L}_{\text{CE}}(\theta_{t})=\frac{1}{2n}\sum_{i=1}^{n}\mathbf{\varepsilon}^{[i]}\mathbf{x}^{[i]}-\frac{1}{2n}\sum_{i=1}^{n}\mathbf{x}^{[i]}\tanh(\theta_{t}^{\top}\mathbf{x}^{[i]})\,.
$$

Since $\mathbb{E}\mathbf{v}^{\top}(\mathbf{\varepsilon}^{[i]}\mathbf{x}^{[i]})=(1-\Delta)$, the law of large numbers implies

$$
\mathbf{v}^{\top}\Big(\frac{1}{2n}\sum_{i=1}^{n}\mathbf{\varepsilon}^{[i]}\mathbf{x}^{[i]}\Big)=\frac{1}{2}(1-\Delta)+o_{P}(1)\,.
$$

Moreover, by Lemma 9, there exists a positive constant $c$ such that with high probability

$$
\displaystyle\Big|\mathbf{v}^{\top}\Big(\frac{1}{2n}\sum_{i=1}^{n}\mathbf{x}^{[i]}\tanh(\theta_{t}^{\top}\mathbf{x}^{[i]})\Big)\Big|
$$
 
$$
\displaystyle\leq\frac{1}{2}\Big(\frac{1}{n}\sum_{i=1}^{n}(\mathbf{v}^{\top}\mathbf{x}^{[i]})^{2}\Big)^{1/2}\left(\frac{1}{n}\sum_{i=1}^{n}\tanh(\theta_{t}^{\top}\mathbf{x}^{[i]})^{2}\right)^{1/2}
$$
 
$$
\displaystyle\leq\frac{1}{2}|\tanh(\theta_{t}^{\top}\mathbf{v})|+c\sigma(1+\|\theta_{t}-\theta_{0}\|)\,.
$$

Thus, applying Lemma 8 yields that with high probability

$$
-\nabla\mathcal{L}_{\text{CE}}(\theta_{t})^{\top}v/\|\nabla\mathcal{L}_{\text{CE}}(\theta_{t})\|\geq\frac{1}{2}((1-\Delta)-|\tanh(\theta_{t}^{\top}\mathbf{v})|)-c\sigma(1+\|\theta_{t}-\theta_{0}\|)\,.
$$

When $t=0$, the first term is $\frac{1}{2}(1-\Delta)+o_{P}(1)$ by Lemma 7, and the second term is $c\sigma$. Since we have assumed that $\Delta\leq 1/2$, as long as $\sigma\leq\sigma_{\Delta}<c^{-1}\left(\frac{2}{3}-\frac{\Delta}{2}\right)$ we will have that $-\nabla\mathcal{L}_{\text{CE}}(\theta_{0})^{\top}v/\|\nabla\mathcal{L}_{\text{CE}}(\theta_{0})\|\geq 1/6$ with high probability, as desired.

We proceed with the induction. We will show that $\|\theta_{t}-\theta_{0}\|\leq 1$ with high probability for $t<T$, and use (12) to show that this implies the desired bound on the correlation of the gradient. If we assume the claim holds up to time $t$, then the definition of gradient descent implies

$$
\theta_{t}-\theta_{0}=\eta\sum_{s=0}^{t-1}\mathbf{g}_{s}\,,
$$

where $\mathbf{g}_{s}$ satisfies $\mathbf{g}_{s}^{\top}\mathbf{v}/\|\mathbf{g}_{s}\|\geq 1/6$. Since the set of vectors satisfying this requirement forms a convex cone, we obtain that

$$
(\theta_{t}-\theta_{0})^{\top}\mathbf{v}/\|\theta_{t}-\theta_{0}\|\geq 1/6\,
$$

From this observation, we obtain two facts about $\theta_{t}$. First, since $t<T$, the definition of $T$ implies that $\theta_{t}^{\top}\mathbf{v}<.1$. Since $|\theta_{0}^{\top}\mathbf{v}|=o_{P}(1)$ by Lemma 7, we obtain that $\|\theta_{t}-\theta_{0}\|\leq 1$ with high probability. Second, $\theta_{t}^{\top}\mathbf{v}\geq\theta_{0}^{\top}\mathbf{v}$, and since $|\theta_{0}^{\top}\mathbf{v}|=o_{P}(1)$ we have in particular that $\theta_{t}^{\top}\mathbf{v}>-.1$. Therefore, with high probability, we also have $|\theta_{t}\top\mathbf{v}|<.1$

Examining (12), we therefore see that the quantity on the right side is at least

$$
\frac{1}{2}((1-\Delta)-.1)-2c\sigma\,.
$$

Again since we have assumed that $\Delta\leq 1/2$, as long as $\sigma\leq\sigma_{\Delta}<(2c)^{-1}\left(\frac{2}{3}-\frac{\Delta}{2}-.1\right)$, we obtain by (12) that $-\nabla\mathcal{L}_{\text{CE}}(\theta_{t})^{\top}v/\|\nabla\mathcal{L}_{\text{CE}}(\theta_{t})\|\geq 1/6$.

∎

Given $\theta_{t}$, we denote by

$$
\hat{\mathcal{A}}(\theta_{t})=\frac{1}{|W|}\sum_{i\in W}\mathds{1}\{\mathrm{sign}(\theta_{t}^{\top}\mathbf{x}^{[i]})=(\mathbf{\varepsilon}^{\ast})^{[i]}\}
$$

the accuracy of $\theta_{t}$ on mislabeled examples. We now show that the classifier’s accuracy on the mislabeled examples improves over the first $T$ rounds. In fact, we show that with high probability, $\hat{\mathcal{A}}(\theta_{0})\approx 1/2$ whereas $\hat{\mathcal{A}}(\theta_{T})\approx 1$.

###### Theorem 4.

There exists a $\sigma_{\Delta}$ such that if $\sigma\leq\sigma_{\Delta}$, then

$$
\hat{\mathcal{A}}(\theta_{0})\leq.5001
$$

and

$$
\hat{\mathcal{A}}(\theta_{T})>.9999
$$

with high probability.

###### Proof.

Let us write $\mathbf{x}^{[i]}=(\mathbf{\varepsilon}^{\ast})^{[i]}(\mathbf{v}-\sigma\mathbf{z}^{[i]})$, where $\mathbf{z}^{[i]}$ is a standard Gaussian vector. If we fix $\theta_{0}$, then $\mathrm{sign}(\theta_{0}^{\top}\mathbf{x}^{[i]})=(\mathbf{\varepsilon}^{\ast})^{[i]}$ if and only if $\sigma\theta_{0}^{\top}\mathbf{z}^{[i]}<\theta_{0}^{\top}\mathbf{v}$. In particular this yields

$$
\mathbb{E}[\mathds{1}\{\mathrm{sign}(\theta_{0}^{\top}\mathbf{x}^{[i]})=(\mathbf{\varepsilon}^{\ast})^{[i]}\}|\theta_{0}]=\mathbb{P}[\sigma\theta_{0}^{\top}\mathbf{z}^{[i]}<\theta_{0}^{\top}\mathbf{v}|\theta_{0}]\leq 1/2+O(|\theta_{0}^{\top}\mathbf{v}|/\sigma)\,.
$$

By the law of large numbers, we have that, conditioned on $\theta_{0}$,

$$
\hat{\mathcal{A}}(\theta_{0})\leq 1/2+O(|\theta_{0}^{\top}\mathbf{v}|/\sigma)+o_{P}(1)\,,
$$

and applying Lemma 7 yields $\hat{\mathcal{A}}(\theta_{0})\leq 1/2+o_{P}(1)$.

In the other direction, we employ a method based on [^30]. The proof of Proposition 3 establishes that $\|\theta_{t}-\theta_{0}\|\leq 1$ for all $t<T-1$ with high probability. Since $\eta<1$ and $\|\theta_{0}\|=2$, Lemma 8 implies that as long as $\sigma<1/2$, $\|\theta_{T}\|\leq 5$ with high probability. Since $\theta_{T}^{\top}\mathbf{v}\geq.1$ by assumption, we obtain that $\theta_{T}^{\top}\mathbf{v}/\|\theta_{T}\|\geq 1/50$ with high probability.

Note that $W$ is a random subset of $[n]$. For now, let us condition on this random variable. If we write $\Phi$ for the Gaussian CDF, then by the same reasoning as above, for any fixed $\theta\in\mathbb{R}^{p}$,

$$
\mathbb{E}[\mathds{1}\{\mathrm{sign}(\theta^{\top}\mathbf{x}^{[i]})=(\mathbf{\varepsilon}^{\ast})^{[i]}\}]=\mathbb{P}[\sigma\theta^{\top}\mathbf{z}^{[i]}<\theta^{\top}\mathbf{v}]=\Phi(\sigma^{-1}\theta^{\top}\mathbf{v}/\|\theta\|)
$$

Therefore, if $\theta^{\top}\mathbf{v}/\|\theta\|\geq\tau$, then for any $\delta>0$, we have

$$
\hat{\mathcal{A}}(\theta)\geq\Phi(\sigma^{-1}\tau-\delta)-\frac{1}{|W|}\sum_{i\in W}\Phi(\sigma^{-1}\tau-\delta)-\mathds{1}\{\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|<\sigma^{-1}\tau\}
$$

Set

$$
\phi(x):=\left\{\begin{array}[]{ll}1&\text{if $x<\sigma^{-1}\tau-\delta$}\\
\frac{1}{\delta}(\sigma^{-1}\tau-x)&\text{if $x\in[\sigma^{-1}\tau-\delta,\sigma^{-1}\tau]$}\\
0&\text{if $x>\sigma^{-1}\tau$.}\end{array}\right.
$$

By construction, $\phi$ is $\frac{1}{\delta}$ -Lipschitz and satisfies

$$
\mathds{1}\{x<\sigma^{-1}\tau-\delta\}\leq\phi(x)\leq\mathds{1}\{x<\sigma^{-1}\tau\}
$$

for all $x\in\mathbb{R}$. In particular, we have

$$
\Phi(\sigma^{-1}\tau-\delta)-\mathds{1}\{\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|<\sigma^{-1}\tau\}\leq\mathbb{E}[\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)]-\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)\,.
$$

Denote the set of $\theta\in\mathbb{R}^{p}$ satisfying $\theta^{\top}\mathbf{v}/\|\theta\|\geq\tau$ by $\mathcal{C}_{\tau}$. Combining the last display with (15) yields

$$
\mathbb{E}\inf_{\theta\in\mathcal{C}_{\tau}}\hat{\mathcal{A}}(\theta)\geq\Phi(\sigma^{-1}\tau-\delta)-\mathbb{E}\sup_{\theta\in\mathcal{C}_{\tau}}\frac{1}{|W|}\sum_{i\in W}\mathbb{E}[\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)]-\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)\,.
$$

To control the last term, we employ symmetrization and contraction (see [^21]) to obtain

$$
\displaystyle\mathbb{E}\sup_{\theta\in\mathcal{C}_{\tau}}\frac{1}{|W|}\sum_{i\in W}\mathbb{E}[\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)]-\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)
$$
 
$$
\displaystyle\leq\mathbb{E}\sup_{\theta\in\mathcal{C}_{\tau}}\frac{1}{|W|}\sum_{i\in W}\epsilon_{i}\phi(\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|)
$$
 
$$
\displaystyle\leq\frac{1}{\delta}\mathbb{E}\sup_{\theta\in\mathcal{C}_{\tau}}\frac{1}{|W|}\sum_{i\in W}\epsilon_{i}\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|
$$
 
$$
\displaystyle\leq\frac{1}{\delta}\mathbb{E}\sup_{\theta\in\mathbb{R}^{p}}\frac{1}{|W|}\sum_{i\in W}\epsilon_{i}\theta^{\top}\mathbf{z}^{[i]}/\|\theta\|
$$
 
$$
\displaystyle=\frac{1}{\delta}\mathbb{E}\left\|\frac{1}{|W|}\sum_{i\in W}\epsilon_{i}\mathbf{z}^{[i]}\right\|\,.
$$

where $\epsilon_{i}$ are independent Rademacher random variables. The final quantity is easily seen to be at most $\frac{1}{\delta}\sqrt{p/|W|}$. Therefore we have

$$
\mathbb{E}\inf_{\theta\in\mathcal{C}_{\tau}}\hat{\mathcal{A}}(\theta)\geq\Phi(\sigma^{-1}\tau-\delta)-\frac{1}{\delta}\sqrt{p/|W|}\,,
$$

and a standard application of Azuma’s inequality implies that this bound also holds with high probability. Since $\theta_{T}^{\top}\mathbf{v}/\|\theta_{T}\|\geq 1/50$ and $|W|\geq\Delta n/2$ with high probability, there exists a positive constant $c_{\Delta}$ such that

$$
\hat{\mathcal{A}}(\theta_{T})\geq\Phi((50\sigma)^{-1}-\delta)-c_{\Delta}/\delta\,.
$$

If we choose $\delta=10^{-4}/2c_{\Delta}$, then there exists a $\sigma_{\Delta}$ for which $\Phi((50\sigma_{\Delta})^{-1}-\delta)>1-10^{-4}/2$. We obtain that for any $\sigma\leq\sigma_{\Delta}$, $\hat{\mathcal{A}}(\theta_{T})\leq 1-10^{-4}$, as claimed. ∎

### A.3 Vanishing gradients

We now show that, over the first $T$ iterations, the coefficients $\tanh(\theta^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]}$ associated with the correctly labeled examples decrease, while the coefficients on mislabeled examples increase. For simplicity, we write $\kappa^{[i]}:=\tanh(\theta^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]}$.

###### Proposition 5.

There exists a constant $\sigma_{\Delta}$ such that, for any $\sigma\leq\sigma_{\Delta}$, with high probability,

$$
\displaystyle\frac{1}{|C|}\sum_{i\in C}(\kappa^{[i]}(\theta_{T}))^{2}
$$
 
$$
\displaystyle<\frac{1}{|C|}\sum_{i\in C}(\kappa^{[i]}(\theta_{0}))^{2}-.05
$$
 
$$
\displaystyle\frac{1}{|W|}\sum_{i\in W}(\kappa^{[i]}(\theta_{T}))^{2}
$$
 
$$
\displaystyle>\frac{1}{|W|}\sum_{i\in W}(\kappa^{[i]}(\theta_{0}))^{2}+.05\,.
$$

That is, during the first stage, the coefficients on correct examples decrease while the coefficients on wrongly labeled examples increase.

###### Proof.

Let us first consider

$$
\frac{1}{|C|}\sum_{i\in C}(\tanh(\theta_{0}^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]})^{2}
$$

For fixed initialization $\theta_{0}$, the law of large numbers implies that this quantity is near

$$
\mathbb{E}_{\mathbf{x},\mathbf{\varepsilon}}(\mathbf{\varepsilon}\tanh(\theta_{0}^{\top}\mathbf{x})-1)^{2}\geq\left(\mathbb{E}_{\mathbf{x},\mathbf{\varepsilon}}\mathbf{\varepsilon}\tanh(\theta_{0}^{\top}\mathbf{x})-1\right)^{2}\,.
$$

Let us write $\mathbf{x}=\mathbf{\varepsilon}^{\ast}(\mathbf{v}-\sigma\mathbf{z})$, where $\mathbf{z}$ is a standard Gaussian vector. Then the fact that $\tanh$ is Lipschitz implies

$$
\mathbb{E}_{\mathbf{x},\mathbf{\varepsilon}}\mathbf{\varepsilon}\tanh(\theta_{0}^{\top}\mathbf{x})\leq\mathbb{E}_{\mathbf{x},\mathbf{\varepsilon}}\mathbf{\varepsilon}\tanh(\mathbf{\varepsilon}^{\ast}\sigma\theta_{0}^{\top}\mathbf{z})+|\theta_{0}^{\top}\mathbf{v}|=|\theta_{0}^{\top}\mathbf{v}|\,,
$$

where we have used that $\mathbb{E}[\tanh(\mathbf{\varepsilon}^{\ast}\sigma\theta_{0}^{\top}\mathbf{z})|\mathbf{\varepsilon}]=0$. By Lemma 7, $|\theta_{0}^{\top}\mathbf{v}|=o_{P}(1)$. Hence

$$
\frac{1}{|C|}\sum_{i\in C}(\tanh(\theta_{0}^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]})^{2}\geq 1-o_{P}(1)\,.
$$

At iteration $T$, we have that $\theta_{T}^{\top}\mathbf{v}\geq.1$, by assumption, and $\|\theta_{T}-\theta_{0}\|\leq 3$, by the proof of Proposition 3. We can therefore apply Lemma 9 to obtain

$$
\displaystyle\left(\frac{1}{|C|}\sum_{i\in C}(\kappa^{[i]})^{2}\right)^{1/2}
$$
 
$$
\displaystyle\leq\left(\frac{1}{|C|}\sum_{i\in C}((\mathbf{\varepsilon}^{\ast})^{[i]}\tanh(\theta_{T}^{\top}\mathbf{v})-\mathbf{\varepsilon}^{[i]})\right)^{1/2}+\sigma(2+3c_{\Delta})+o_{P}(1)
$$
 
$$
\displaystyle=|\tanh(\theta_{T}^{\top}\mathbf{v})-1|++\sigma(2+3c_{\Delta})+o_{P}(1)
$$
 
$$
\displaystyle\leq|\tanh(.1)-1|+\sigma(2+3c_{\Delta})+o_{P}(1)\,,
$$

where the equality uses the fact that $(\mathbf{\varepsilon}^{\ast})^{[i]}=\mathbf{\varepsilon}^{[i]}$ for all $i\in C$. As long as $\sigma\leq\sigma_{\Delta}<.01/(2+3c_{\Delta})$ this quantity is strictly less than $.95$. We therefore obtain that, for $\sigma\leq\sigma_{\Delta}$, $\frac{1}{|C|}\sum_{i\in C}(\kappa^{[i]}(\theta_{T}))^{2}<\frac{1}{|C|}\sum_{i\in C}(\kappa^{[i]}(\theta_{0}))^{2}-.05$ with high probability. This proves the first claim.

The second claim is established by an analogous argument: for fixed initialization $\theta_{0}$, we have

$$
\mathbb{E}\tanh(\theta_{0}^{\top}\mathbf{x})^{2}\leq\mathbb{E}(\theta_{0}^{\top}\mathbf{x})^{2}=4\sigma^{2}+(\theta_{0}^{\top}\mathbf{v})^{2}\,,
$$

so as above we can conclude that

$$
\frac{1}{|W|}\sum_{i\in W}(\kappa^{[i]}(\theta_{0}))^{2}\leq 1+4\sigma^{2}+o_{P}(1)\,.
$$

We likewise have by another application of Lemma 9

$$
\displaystyle\left(\frac{1}{|W|}\sum_{i\in W}(\tanh(\theta_{T}^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]})^{2}\right)^{1/2}
$$
 
$$
\displaystyle\geq|\tanh(-\theta_{T}^{\top}\mathbf{v})-1|-\sigma(2+3c_{\Delta})-o_{P}(1)
$$
 
$$
\displaystyle\geq 1+\tanh(.1)-\sigma(2+3c_{\Delta})-o_{P}(1)\,.
$$

where we again have used that $(\mathbf{\varepsilon}^{\ast})^{[i]}=-\mathbf{\varepsilon}^{[i]}$ for $i\in W$. If we choose $\sigma_{\Delta}$ small enough that

$$
1.05+4\sigma_{\Delta}^{2}<(1+\tanh(.1)-\sigma_{\Delta}(2+3c_{\Delta}))^{2}\,,
$$

then we will have for all $\sigma\leq\sigma_{\Delta}$ that $\frac{1}{|W|}\sum_{i\in W}(\kappa^{[i]}(\theta_{T}))^{2}>\frac{1}{|W|}\sum_{i\in W}(\kappa^{[i]}(\theta_{0}))^{2}+.05$ with high probability. This proves the claim. ∎

### A.4 Memorization

To show that the labels are memorized asymptotically, it suffices to show that the classes $S_{+}:=\{\mathbf{x}^{[i]}:\mathbf{\varepsilon}^{[i]}=+1\}$ and $S_{-}:=\{\mathbf{x}^{[i]}:\mathbf{\varepsilon}^{[i]}=-1\}$ are linearly separable. Indeed, it is well known that for linearly separable data, gradient descent performed on the logistic loss will yield a classifier which perfectly memorizes the labels [^37]. It is therefore enough to establish the following theorem.

###### Theorem 6.

If $p,n\to\infty$ and $\liminf_{p,n\to\infty}p/n>1-\Delta/2$, then the classes $S_{+}$ and $S_{-}$ are linearly separable with probability tending to $1$.

###### Proof.

Write $X=\{\mathbf{x}^{[1]},\dots,\mathbf{x}^{[n]}\}$. Since the samples $\mathbf{x}^{[1]},\dots,\mathbf{x}^{[n]}$ are drawn from a distribution absolutely continuous with respect to the Lebesgue measure, they are in general position with probability $1$. By a theorem of Schläfli [^8], there exist

$$
C(n,p):=2\sum_{k=0}^{p-1}\binom{n-1}{k}
$$

different subsets $S\subseteq X$ that are linearly separable from their complements. In particular, there are at most

$$
2^{n}-C(n,p)=2\sum_{k=p}^{n-1}\binom{n-1}{k}=2\sum_{k=0}^{n-p-1}\binom{n-1}{k}
$$

partitions of $X$ which are *not* separable.

Write $B$ for the bad set of non-separable subsets $S\subseteq X$. Conditional on $X$, the probability that the classes $S_{+}$ and $S_{-}$ are not separable is just $\mathbb{P}[S_{+}\in B|X]$.

Let us write $T_{+}:=\{\mathbf{x}^{[i]}:(\mathbf{\varepsilon}^{\ast})^{[i]}=+1\}$. For each $i$, the example $\mathbf{x}^{[i]}$ is in $S_{+}$ with probability $1-(\Delta/2)$ if $i\in T_{+}$ or $\Delta/2$ or if $i\notin T_{+}$. We therefore have for any $S\subset X$ that

$$
\mathbb{P}[S_{+}=S|X]=(\Delta/2)^{|T_{+}\triangle S|}(1-(\Delta/2))^{n-|T_{+}\triangle S|}\,.
$$

We obtain that

$$
\displaystyle\mathbb{P}[S_{+}\in B|X]
$$
 
$$
\displaystyle=\sum_{S\in B}(\Delta/2)^{|T_{+}\triangle S|}(1-(\Delta/2))^{n-|T_{+}\triangle S|}
$$
 
$$
\displaystyle=\sum_{k=0}^{n}|\{S\in B:|T_{+}\triangle S|=k\}|\cdot(\Delta/2)^{k}(1-(\Delta/2))^{n-k}\,.
$$

The set $\{S\in B:|T_{+}\triangle S|=k\}$ has cardinality at most $\binom{n}{k}$. Moreover, $\sum_{k=0}^{n}|\{S\in B:|T_{+}\triangle S|=k\}|=|B|$. We can therefore bound the sum (16) by the following optimization problem:

$$
\displaystyle\max_{x_{1},\dots x_{n}}\,\,
$$
$$
\displaystyle\sum_{k=0}^{n}x_{k}\cdot(\Delta/2)^{k}(1-(\Delta/2))^{n-k}
$$
 
$$
\displaystyle\text{s.t.}\,\,\,x_{k}\in\left[0,\binom{n}{k}\right],\sum_{k=0}^{n}x_{k}=|B|\,.
$$

Since $\Delta\leq 1$, the probability $(\Delta/2)^{k}(1-(\Delta/2))^{n-k}$ is a nonincreasing function of $k$. Therefore, because $|B|\leq 2\sum_{k=0}^{n-p-1}\binom{n-1}{k}\leq 2\sum_{k=0}^{n-p}\binom{n}{k}$, the value of (17) is less than

$$
2\sum_{k=0}^{n-p}\binom{n}{k}(\Delta/2)^{k}(1-(\Delta/2))^{n-k}=2\cdot\mathbb{P}[\mathrm{Bin}(n,\Delta/2)\leq(n-p)]\,.
$$

If $\limsup_{n,p\to\infty}1-p/n<\Delta/2$, then this probability approaches $0$ by the law of large numbers. We have shown that if $\liminf_{n,p}p/n>1-\Delta/2$, then

$$
\mathbb{P}[S_{+}\in B|X]\leq 2\cdot\mathbb{P}[\mathrm{Bin}(n,\Delta/2)\leq(n-p)]=o(1)
$$

holds $X$ -almost surely, which proves the claim.

∎

### A.5 Additional lemmas

###### Lemma 7.

Suppose that $\theta_{0}$ is initialized randomly on the sphere of radius $2$.

$$
|\theta_{0}^{\top}\mathbf{v}|=o_{P}(1).
$$

###### Proof.

Without loss of generality, take $\mathbf{v}=\mathbf{e}_{1}$, the first elementary basis vector. Since the coordinates of $\theta_{0}$ each have the same marginal distribution and $\|\theta_{0}\|^{2}=2$ almost surely, we must have $\mathbb{E}|\theta_{0}^{\top}\mathbf{e}_{1}|^{2}=2/p$. The claim follows. ∎

###### Lemma 8.

$$
\sup_{\theta\in\mathbb{R}^{d}}\|\nabla\mathcal{L}_{\text{CE}}(\theta)\|\leq 1+2\sigma+o_{P}(1)\,.
$$

###### Proof.

Denote by $\alpha$ the vector with entries $\alpha_{i}=\frac{1}{2\sqrt{n}}[\tanh(\theta^{\top}\mathbf{x}^{[i]})-\mathbf{\varepsilon}^{[i]}]$. Since $|\tanh(x)|\leq 1$ for all $x\in\mathbb{R}$, we have $\|\alpha\|\leq 1$. Therefore

$$
\nabla\mathcal{L}_{\text{CE}}(\theta)=\frac{1}{\sqrt{n}}\sum_{i=1}^{n}\mathbf{x}^{[i]}\alpha_{i}\leq\left\|\frac{1}{\sqrt{n}}\mathbf{X}\right\|\,,
$$

where $\mathbf{X}\in\mathbb{R}^{p\times n}$ is a matrix whose columns are given by the vectors $\mathbf{x}^{[i]}$. By Lemma 10, we have

$$
\left\|\frac{1}{\sqrt{n}}\mathbf{X}\right\|=\left\|\frac{1}{n}\mathbf{X}\mathbf{X}^{\top}\right\|^{1/2}\leq 1+2\sigma+o_{P}(1)\,.
$$

This yields the claim. ∎

###### Lemma 9.

Fix an initialization $\theta_{0}$ satisfying $\|\theta_{0}\|=2$. For any $\tau>0$ and for $I=C$ or $I=W$, we have

$$
\sup_{\theta:\|\theta-\theta_{0}\|\leq\tau}\left(\frac{1}{|I|}\sum_{i\in I}((\mathbf{\varepsilon}^{\ast})^{[i]}\tanh(\theta^{\top}\mathbf{x}^{[i]})-\tanh(\theta^{\top}\mathbf{v}))^{2}\right)^{1/2}\leq\sigma(2+c_{\Delta}\tau)+o_{P}(1)\,.
$$

The same claim holds with $I=[n]$ with $c_{\Delta}$ replaced by $2$.

###### Proof.

Let us write $\mathbf{x}^{[i]}=(\mathbf{\varepsilon}^{\ast})^{[i]}(\mathbf{v}-\sigma\mathbf{z}^{[i]})$, where $\mathbf{z}^{[i]}$ is a standard Gaussian vector. Since $\tanh$ is odd and 1-Lipschitz, we have

$$
|(\mathbf{\varepsilon}^{\ast})^{[i]}\tanh(\theta^{\top}\mathbf{x}^{[i]})-\tanh(\theta^{\top}\mathbf{v})|=|\tanh(\theta^{\top}\mathbf{v}-\theta^{\top}\sigma\mathbf{z}^{[i]})-\tanh(\theta^{\top}\mathbf{v})|\leq\sigma|\theta^{\top}\mathbf{z}^{[i]}|\,.
$$

We therefore obtain

$$
\displaystyle\Big(\frac{1}{|I|}\sum_{i\in I}((\mathbf{\varepsilon}^{\ast})^{[i]}\tanh(\theta^{\top}\mathbf{x}^{[i]})-\tanh(\theta^{\top}\mathbf{v}))^{2}\Big)^{1/2}
$$
 
$$
\displaystyle\leq\sigma\Big(\frac{1}{|I|}\sum_{i\in I}(\theta^{\top}\mathbf{z}^{[i]})^{2}\Big)^{1/2}
$$
 
$$
\displaystyle\leq\sigma\Big(\frac{1}{|I|}\sum_{i\in I}(\theta_{0}^{\top}\mathbf{z}^{[i]})^{2}\Big)^{1/2}+\sigma\Big(\frac{1}{|I|}\sum_{i\in I}((\theta-\theta_{0})^{\top}\mathbf{z}^{[i]})^{2}\Big)^{1/2}
$$
 
$$
\displaystyle\leq\sigma\Big(\frac{1}{|I|}\sum_{i\in I}(\theta_{0}^{\top}\mathbf{z}^{[i]})^{2}\Big)^{1/2}+\sigma\|\theta-\theta_{0}\|\left\|\frac{1}{|I|}\sum_{i\in I}\mathbf{z}^{[i]}(\mathbf{z}^{[i]})^{\top}\right\|\,.
$$

Taking a supremum over all $\theta$ such that $\|\theta-\theta_{0}\|\leq\tau$ and applying Lemma 10 yields the claim. ∎

###### Lemma 10.

Assume $p\leq n$. There exists a positive constant $c_{\Delta}$ depending on $\Delta$ such that for $I=C$ or $I=W$,

$$
\displaystyle\frac{1}{|I|}\sum_{i\in I}(\theta_{0}^{\top}\mathbf{z}^{[i]})^{2}
$$
 
$$
\displaystyle\leq 2+o_{P}(1)
$$
 
$$
\displaystyle\left\|\frac{1}{|I|}\sum_{i\in I}\mathbf{z}^{[i]}(\mathbf{z}^{[i]})^{\top}\right\|^{1/2}
$$
 
$$
\displaystyle\leq c_{\Delta}+o_{P}(1)
$$
 
$$
\displaystyle\left\|\frac{1}{|I|}\sum_{i\in I}\mathbf{x}^{[i]}(\mathbf{x}^{[i]})^{\top}\right\|^{1/2}
$$
 
$$
\displaystyle\leq 1+\sigma c_{\Delta}+o_{P}(1)\,.
$$

Moreover, the same claims hold with $I=[n]$, when $c_{\Delta}$ can be replaced by $2$.

###### Proof.

The first claim follows immediately from the law of large numbers. For the second two claims, we first consider the case where $I=[n]$. Let us write $\mathbf{Z}$ for the matrix whose columns are given by the vectors $\mathbf{z}^{[i]}$. Then

$$
\left\|\frac{1}{n}\sum_{i\in[n]}\mathbf{z}^{[i]}(\mathbf{z}^{[i]})^{\top}\right\|^{1/2}=\left\|\frac{1}{n}\mathbf{Z}\mathbf{Z}^{\top}\right\|^{1/2}=\left\|\frac{1}{\sqrt{n}}\mathbf{Z}\right\|\leq 1+\sqrt{p/n}+o_{P}(1)\,,
$$

where the last claim is a consequences of standard bounds for the spectral norm of Gaussian random matrices [^42]. Since $p\leq n$ by assumption, the claimed bound follows. When $I=C$ or $W$, the same argument applies, except that we condition on the set of indices in $I$, which yields that, conditioned on $I$,

$$
\left\|\frac{1}{|I|}\sum_{i\in I}\mathbf{z}^{[i]}(\mathbf{z}^{[i]})^{\top}\right\|^{1/2}\leq 2\sqrt{n/|I|}+o_{P}(1)\,.
$$

For any $\Delta$, the random variable $|I|$ concentrates around its expectation, which is $c_{\Delta}n$, for some constant $c_{\Delta}$.

Finally, to bound $\left\|\frac{1}{|I|}\sum_{i\in I}\mathbf{x}^{[i]}(\mathbf{x}^{[i]})^{\top}\right\|^{1/2}$, we again let $\mathbf{X}$ be a matrix whose columns are given by $\mathbf{x}^{[i]}$. Then we can write

$$
\mathbf{X}=\mathbf{v}(\mathbf{\varepsilon}^{\ast})^{\top}+\sigma\mathbf{Z}\,,
$$

where $\mathbf{Z}$ is a Gaussian matrix, as above. Therefore

$$
\left\|\frac{1}{n}\sum_{i\in[n]}\mathbf{x}^{[i]}(\mathbf{x}^{[i]})^{\top}\right\|^{1/2}=\left\|\frac{1}{\sqrt{n}}\mathbf{X}\right\|\leq\frac{1}{\sqrt{n}}\left\|\mathbf{v}(\mathbf{\varepsilon}^{\ast})^{\top}\right\|+\sigma\left\|\frac{1}{\sqrt{n}}\mathbf{Z}\right\|\leq 1+2\sigma+o_{P}(1)\,.
$$

The extension to $I=C$ or $W$ is as above. ∎

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/LR_True_label_behavior_logscale.png)

Refer to caption

## Appendix B Early Learning and Memorization in Linear and Deep-Learning Models

In this section we provide a numerical example to illustrate the theory in Section 3, and the similarities between the behavior of linear and deep-learning models. We train the two-class softmax linear regression model described in Section 3 on data drawn from a mixture of two Gaussians in $\mathbb{R}^{100}$, where 40% of the labels are flipped at random. Figure A.1 shows the training accuracy on the training set for examples with clean and false labels. Analogously to the deep-learning model in Figure 1, the linear model trained with cross entropy begins by learning to predict the true labels, but eventually memorizes the examples with wrong labels as predicted by our theory. The figure also shows the results of applying our proposed early-learning regularization technique with temporal ensembling. ELR prevents memorization, allowing the model to continue learning on the examples with clean labels to attain high accuracy on examples with clean and wrong labels.

As explained in Section 4.2, for both linear and deep-learning models the effect of label noise on the gradient of the cross-entropy loss for each example $i$ is restricted to the term $\mathbf{p}^{[i]}-\mathbf{y}^{[i]}$, where $\mathbf{p}^{[i]}$ is the probability example assigned by the model to the example and $\mathbf{y}^{[i]}$ is the corresponding label. Figure B.1 plots this quantity for the linear model described in the previous paragraph and for the deep-learning model from Figure 1. In both cases, the label noise flips the sign of the term on the wrong labels (left column). The magnitude of this term dominates after early learning (right column), eventually producing memorization of the wrong labels.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/LR_grad_true_logit_logscale.png)

Refer to caption

## Appendix C Regularization Based on Kullback-Leibler Divergence

A natural alternative to our proposed regularization would be to penalize the Kullback-Leibler (KL) divergence between the the model output and the targets. This results in the following loss function

$$
\displaystyle\mathcal{L}_{\text{CE}}(\Theta)-\frac{\lambda}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}\mathbf{t}^{[i]}_{c}\log\mathbf{p}^{[i]}_{c}.
$$

Figure C.1 shows the result of applying this regularization to CIFAR-10 dataset with 40% symmetric noise for different values of the regularization parameter $\lambda$, using targets computed via temporal ensembling. In contrast to ELR, which succeeds in avoiding memorization while allowing the model to learn effectively as demonstrated in the bottom row of Figure 1, regularization based on KL divergence fails to provide robustness. When $\lambda$ is small, memorization of the wrong labels leads to overfitting as in cross-entropy minimization. Increasing $\lambda$ delays memorization, but does not eliminate it. Instead, the model starts overfitting the initial estimates, whether correct or incorrect, and then eventually memorizes the wrong labels (see the bottom right graph in Figure C.1).

Analyzing the gradient of the cost function sheds some light on the reason for the failure of this type of regularization. The gradient with respect to the model parameters $\Theta$ equals

$$
\displaystyle\frac{1}{n}\sum_{i=1}^{n}\nabla\mathcal{N}_{\mathbf{x}^{[i]}}(\Theta)\left(\left(\mathbf{p}^{[i]}-\mathbf{y}^{[i]}\right)+\lambda\left(\mathbf{p}^{[i]}-\mathbf{t}^{[i]}\right)\right).
$$

A key difference between this gradient and the gradient of ELR is the dependence of the sign of the regularization component on the targets. In ELR, the sign of the $c$ th entry for the $i$ th is determined by the difference between $\mathbf{t}^{[i]}_{c}$ and the rest of the entries of $\mathbf{t}^{[i]}$ (see Lemma 2). In contrast, for KL divergence it depends on the difference between $\mathbf{t}^{[i]}_{c}$ and $\mathbf{p}^{[i]}_{c}$. This results in overfitting the target probabilities. To illustrate this, recall that for examples with clean labels, the cross-entropy term $\mathbf{p}^{[i]}-\mathbf{y}^{[i]}$ tends to vanish after the early-learning stage because $\mathbf{p}^{[i]}$ is very close to $\mathbf{y}^{[i]}$, allowing examples with wrong labels to dominate the gradient. Let $c^{\ast}$ denote the true class. When $\mathbf{p}^{[i]}_{c^{\ast}}$ (correctly) approaches one, $\mathbf{t}^{[i]}_{c^{\ast}}$ will generally tend to be smaller, because $\mathbf{t}^{[i]}$ is obtained by a moving average and therefore tends to be smoother than $\mathbf{p}^{[i]}$. Consequently, the regularization term tends to decrease $\mathbf{p}^{[i]}_{c^{\ast}}$. This is exactly the opposite effect than desired. In contrast, ELR tends to keep $\mathbf{p}^{[i]}_{c^{\ast}}$ large, as explained in Section 4.2, which allows the model to continue learning on the clean examples.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/True_label_behavior_KL_lamb1.png)

Refer to caption

## Appendix D The Need for Early Learning Regularization

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/targets_during_training.png)

Figure D.1: Validation accuracy achieved by targets estimated via temporal ensembling using the cross entropy loss and our proposed cost function. The model is a ResNet-34 trained on CIFAR-10 with 40% symmetric noise. The temporal ensembling momentum β \\beta is set to 0.7. Without the regularization term, the targets eventually overfit the noisy labels.

Our proposed framework consists of two components: target estimation and the early-learning regularization term. Figure D.1 shows that the regularization term is critical to avoid memorization. If we just perform target estimation via temporal ensembling while training with a cross-entropy loss, eventually the targets overfit the noisy labels, resulting in decreased accuracy.

## Appendix E Proof of Lemma

To ease notation, we ignore the $i$ superscript, setting $\mathbf{p}:=\mathbf{p}^{[i]}$ and $\mathbf{t}:=\mathbf{t}^{[i]}$. We denote the instance-level ELR by

$$
\displaystyle\mathcal{R}(\Theta):=\log\left(1-\langle\mathbf{p},\mathbf{t}\rangle\right).
$$

The gradient of $\mathcal{R}$ is

$$
\displaystyle\nabla\mathcal{R}(\Theta)
$$
 
$$
\displaystyle=\frac{1}{1-\langle\mathbf{p},\mathbf{t}\rangle}\nabla\left(1-\langle\mathbf{p},\mathbf{t}\rangle\right).
$$

We express the probability estimate in terms of the softmax function and the deep-learning mapping $\mathcal{N}_{\mathbf{x}}{(\Theta)}$, $\mathbf{p}:=\frac{e^{\mathcal{N}_{\mathbf{x}}(\Theta)}}{\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}$, where $e^{\mathcal{N}_{\mathbf{x}}(\Theta)}$ denotes a vector whose entries equal the exponential of the entries of $\mathcal{N}_{\mathbf{x}}{(\Theta)}$. Plugging this into Eq. (21) yields

$$
\displaystyle\nabla\mathcal{R}(\Theta)
$$
 
$$
\displaystyle=\sum_{i=1}^{n}\frac{1}{1-\langle\mathbf{p},\mathbf{t}\rangle}\nabla\left(1-\frac{\langle e^{\mathcal{N}_{\mathbf{x}}(\Theta)},\mathbf{t}\rangle}{\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}\right)
$$
 
$$
\displaystyle=\sum_{i=1}^{n}\frac{-1}{1-\langle\mathbf{p},\mathbf{t}\rangle}\frac{\nabla\langle e^{\mathcal{N}_{\mathbf{x}}(\Theta)},\mathbf{t}\rangle\cdot\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}-\langle e^{\mathcal{N}_{\mathbf{x}}(\Theta)},\mathbf{t}\rangle\cdot\nabla\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}{\left(\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}\right)^{2}}
$$
 
$$
\displaystyle=\sum_{i=1}^{n}\frac{-\nabla\mathcal{N}_{\mathbf{x}}(\Theta)}{1-\langle\mathbf{p},\mathbf{t}\rangle}\frac{e^{\mathcal{N}_{\mathbf{x}}(\Theta)}\odot\mathbf{t}\cdot\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}-\langle e^{\mathcal{N}_{\mathbf{x}}(\Theta)},\mathbf{t}\rangle\cdot e^{\mathcal{N}_{\mathbf{x}}(\Theta)}}{\left(\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}\right)^{2}}
$$
 
$$
\displaystyle=\sum_{i=1}^{n}\frac{-\nabla\mathcal{N}_{\mathbf{x}}(\Theta)}{1-\langle\mathbf{p},\mathbf{t}\rangle}\left(\frac{e^{\mathcal{N}_{\mathbf{x}}(\Theta)}\odot\mathbf{t}}{\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}-\frac{\langle e^{\mathcal{N}_{\mathbf{x}}(\Theta)},\mathbf{t}\rangle}{\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}\cdot\frac{e^{\mathcal{N}_{\mathbf{x}}(\Theta)}}{\sum_{c=1}^{C}e^{\left(\mathcal{N}_{\mathbf{x}}(\Theta)\right)_{c}}}\right).
$$

The formula can be simplified to

$$
\displaystyle\nabla\mathcal{R}(\Theta)
$$
 
$$
\displaystyle=\frac{-\nabla\mathcal{N}_{\mathbf{x}}(\Theta)}{1-\langle\mathbf{p},\mathbf{t}\rangle}\left(\mathbf{p}\odot\mathbf{t}-\langle\mathbf{p},\mathbf{t}\rangle\cdot\mathbf{p}\right)
$$
 
$$
\displaystyle=\frac{\nabla\mathcal{N}_{\mathbf{x}}(\Theta)}{1-\langle\mathbf{p},\mathbf{t}\rangle}\begin{bmatrix}\mathbf{p}_{1}\cdot\left(\langle\mathbf{p},\mathbf{t}\rangle-\mathbf{t}_{1}\right)\\
\vdots\\
\mathbf{p}_{C}\cdot\left(\langle\mathbf{p},\mathbf{t}\rangle-\mathbf{t}_{C}\right)\end{bmatrix}
$$
 
$$
\displaystyle=\frac{\nabla\mathcal{N}_{\mathbf{x}}(\Theta)}{1-\langle\mathbf{p},\mathbf{t}\rangle}\begin{bmatrix}\mathbf{p}_{1}\cdot\sum_{k=1}^{C}\left(\mathbf{t}_{k}-\mathbf{t}_{1}\right)\mathbf{p}_{k}\\
\vdots\\
\mathbf{p}_{C}\cdot\sum_{k=1}^{C}\left(\mathbf{t}_{k}-\mathbf{t}_{C}\right)\mathbf{p}_{k}\end{bmatrix}.
$$

## Appendix F Algorithms

| Require: $\{\mathbf{x}^{[i]},\mathbf{y}^{[i]}\},\;1\leq i\leq n$ | \= training data (with noisy labels) |  |
| --- | --- | --- |
| Require: $\beta$ | \= temporal ensembling momentum, $0\leq\beta<1$ |  |
| Require: $\lambda$ | \= regularization parameter |  |
| Require: $\mathcal{N}_{\mathbf{x}}(\Theta)$ | \= neural network with trainable parameters $\Theta$ |  |
| $\mathbf{t}$ | $\leftarrow\mathbf{0}_{[n\times C]}$ | $\triangleright$ initialize ensemble predictions |
| for $t$ in $[1,\mathit{num\_epochs}]$ do |  |  |
| for each minibatch $B$ do |  |  |
| for $i$ in $B$ do |  |  |
| $\mathbf{p}^{[i]}\leftarrow\mathcal{S}\left(\mathcal{N}_{\mathbf{x}_{i}}(\Theta)\right)$ |  | $\triangleright$ evaluate network outputs |
| $\mathbf{t}^{[i]}\leftarrow\beta\mathbf{t}^{[i]}+(1-\beta)\mathbf{p}^{[i]}$ |  | $\triangleright$ temporal ensembling |
| end for |  |  |
| $\text{loss}{}\leftarrow$ | ${}-\frac{1}{\|B\|}\sum_{i=1}^{\|B\|}\sum_{c=1}^{C}\mathbf{y}^{[i]}_{c}\log\mathcal{S}\left(\mathcal{N}_{\mathbf{x}_{i}}(\Theta)\right)_{c}$ | $\triangleright$ cross entropy loss component |
|  | ${}+\frac{\lambda}{\|B\|}\sum_{i\in B}\log\left(1-\langle\mathcal{S}\left(\mathcal{N}_{\mathbf{x}_{i}}(\Theta)\right),\mathbf{t}^{[i]}\rangle\right)$ | $\triangleright$ proposed regularization component |
|  |  |  |

         update $\Theta$ using stochastic gradient descent $\triangleright$ update network parameters         end for end for return $\Theta$

Algorithm 1 Pseudocode for ELR with temporal ensembling.

| Require: $\{\mathbf{x}^{[i]},\mathbf{y}^{[i]}\},\;1\leq i\leq n$ | \= training data (with noisy labels) |  |
| --- | --- | --- |
| Require: $\beta$ | \= temporal ensembling momentum, $0\leq\beta<1$ |  |
| Require: $\gamma$ | \= weight averaging momentum, $0\leq\gamma<1$ |  |
| Require: $\lambda$ | \= regularization parameter |  |
| Require: $\alpha$ | \= mixup hyperparameter |  |
|  |  |  |

Require: $\mathcal{N}_{\mathbf{x}}(\Theta_{1})$ = neural network 1 with trainable parameters $\Theta_{1}$ Require: $\mathcal{N}_{\mathbf{x}}(\Theta_{2})$ = neural network 2 with trainable parameters $\Theta_{2}$ $\mathbf{t}_{1}$, $\mathbf{t}_{2}$ $\leftarrow\mathbf{0}_{[n\times C]}$, $\mathbf{0}_{[n\times C]}$ $\triangleright$ initialize averaged predictions $\bar{\Theta}_{1}$, $\bar{\Theta}_{2}$ $\leftarrow\mathbf{0}$, $\mathbf{0}$ $\triangleright$ initialize averaged weights (untrainable) for $t$ in $[1,\mathit{num\_epochs}]$ do         for $k$ in $[1,2]$ do $\triangleright$ for each network            for each minibatch $B$ do                $\tilde{B}\leftarrow\text{mixup}(B,\alpha)$ $\triangleright$ mixup augmentation on the mini-batch                $\bar{\Theta}_{k}=\gamma\bar{\Theta}_{k}+(1-\gamma)\Theta_{k}$ $\triangleright$ weight averaging               for $i$ in $B$ do                   $\mathbf{p}^{[i]}\leftarrow\mathcal{S}\left(\mathcal{N}_{\mathbf{x}_{i}}(\bar{\Theta}_{\{1,2\}\setminus k})\right)$ $\triangleright$ network evaluation with weight averaging                   $\mathbf{t}_{k}^{[i]}\leftarrow\beta\mathbf{t}_{k}^{[i]}+(1-\beta)\mathbf{p}^{[i]}$ $\triangleright$ temporal ensembling               end for                $\text{loss}{}\leftarrow$ ${}-\frac{1}{|B|}\sum_{i=1}^{|B|}\sum_{c=1}^{C}\mathbf{y}^{[i]}_{c}\log\mathcal{S}\left(\mathcal{N}_{\tilde{\mathbf{x}}_{i}}(\Theta_{k})\right)_{c}$       $\triangleright$ cross entropy loss component ${}+\frac{\lambda}{|B|}\sum_{i\in B}\log\left(1-\langle\mathcal{S}\left(\mathcal{N}_{\tilde{\mathbf{x}}_{i}}(\Theta_{k}),\tilde{\mathbf{t}}^{[i]}\rangle\right)\right)$       $\triangleright$ proposed regularization component

            update $\Theta_{k}$ using SGD $\triangleright$ update network parameters            end for         end for end for return $\Theta_{1}$, $\Theta_{2}$

Algorithm 2 Pseudocode for ELR+.

Algorithm 1 and Algorithm 2 provide detailed pseudocode for ELR combined with temporal ensembling (denoted simply by ELR) and ELR combined with temporal ensembling, weight averaging, two networks, and mixup data augmentation (denoted by ELR+) respectively. For CIFAR-10 and CIFAR-100, we use the sigmoid shaped function $e^{-5(1-i/40000)^{2}}$ ($i$ is current training step, following [^40]) to ramp-up the weight averaging momentum $\gamma$ to the value we set as a hyper-parameter. For the other datasets, we fixed $\gamma$. For CIFAR-100, we also use previously mentioned sigmoid shaped function to ramp up the coefficient $\lambda$ to the value we set as a hyper-parameter. Moreover, each entry of the labels $y$ will also be updated by the targets $t$ using $\frac{y_{c}t_{c}}{\sum_{c=1}^{C}y_{c}t_{c}}$ in CIFAR-100.

To apply mixup data augmentation, when processing the $i$ th example in a mini-batch $(\mathbf{x}^{[i]},\mathbf{y}^{[i]},\mathbf{t}^{[i]})$, we randomly sample another example $(\mathbf{x}^{[j]},\mathbf{y}^{[j]},\mathbf{t}^{[j]})$, and compute the $i$ th mixed data $(\tilde{\mathbf{x}}^{[i]},\tilde{\mathbf{y}}^{[i]},\tilde{\mathbf{t}}^{[i]})$ as follows:

$$
\displaystyle\ell
$$
 
$$
\displaystyle\sim\text{Beta}(\alpha,\alpha),
$$
$$
\displaystyle\ell^{\prime}
$$
 
$$
\displaystyle=\max(\ell,1-\ell),
$$
$$
\displaystyle\tilde{\mathbf{x}}^{[i]}
$$
 
$$
\displaystyle=\ell^{\prime}\mathbf{x}^{[i]}+(1-\ell^{\prime})\mathbf{x}^{[j]},
$$
$$
\displaystyle\tilde{\mathbf{y}}^{[i]}
$$
 
$$
\displaystyle=\ell^{\prime}{\mathbf{y}}^{[i]}+(1-\ell^{\prime}){\mathbf{y}}^{[j]},
$$
$$
\displaystyle\tilde{\mathbf{t}}^{[i]}
$$
 
$$
\displaystyle=\ell^{\prime}{\mathbf{t}}^{[i]}+(1-\ell^{\prime}){\mathbf{t}}^{[j]},
$$

where $\alpha$ is a fixed hyperparameter used to choose the symmetric beta distribution from which we sample the ratio of the convex combination between data points.

## Appendix G Description of the Computational Experiments

Source code for the experiments is available at [https://github.com/shengliu66/ELR](https://github.com/shengliu66/ELR).

<table><tbody><tr><td>Data set</td><td>Train</td><td>Val</td><td>Test</td><td>Image size</td><td># classes</td></tr><tr><td colspan="6">Datasets with Clean Annotation</td></tr><tr><td>CIFAR-10</td><td>45K</td><td>5k</td><td>10K</td><td><math><semantics><mrow><mn>32</mn> <mo>×</mo> <mn>32</mn></mrow> <annotation>32\times 32</annotation></semantics></math></td><td>10</td></tr><tr><td>CIFAR-100</td><td>45K</td><td>5k</td><td>10K</td><td><math><semantics><mrow><mn>32</mn> <mo>×</mo> <mn>32</mn></mrow> <annotation>32\times 32</annotation></semantics></math></td><td>100</td></tr><tr><td colspan="6">Datasets with Real World Noisy Annotation</td></tr><tr><td>Clothing-1M</td><td>1M</td><td>14K</td><td>10K</td><td><math><semantics><mrow><mn>224</mn> <mo>×</mo> <mn>224</mn></mrow> <annotation>224\times 224</annotation></semantics></math></td><td>14</td></tr><tr><td>Webvision1.0</td><td>66K</td><td>-</td><td>2.5K</td><td><math><semantics><mrow><mn>256</mn> <mo>×</mo> <mn>256</mn></mrow> <annotation>256\times 256</annotation></semantics></math></td><td>50</td></tr></tbody></table>

Table G.1: Description of the datasets used in our computational experiments, including the training, validation and test splits.

### G.1 Dataset Information

In our experiments we apply ELR and ELR+ to perform image classification on four benchmark datasets: CIFAR-10, CIFAR-100, Clothing-1M, and a subset of WebVision. Because CIFAR-10, CIFAR-100 do not have predefined validation sets, we retain 10% of the training sets to perform validation. Table G.1 provides a detailed description of each dataset.

### G.2 Data preprocessing

We apply normalization and simple data augmentation techniques (random crop and horizontal flip) on the training sets of all datasets. The size of the random crop is set to be consistent with previous works [^56] [^17]: $32$ for the CIFAR datasets, $224\times 224$ for Clothing1M (after resizing to $256\times 256$), and $227\times 227$ for WebVision.

### G.3 Training Procedure

Below we describe the training procedure for ELR (i.e. the proposed approach with temporal ensembling) for the different datasets. The information for ELR+ is shown in Table G.2. In ELR+ we ensemble the outputs of two networks during inference, as is customary for methods that train two networks simultaneously [^22] [^14].

CIFAR-10/CIFAR-100: We use a ResNet-34 [^15] and train it using SGD with a momentum of 0.9, a weight decay of $0.001$, and a batch size of $128$. The network is trained for $120$ epochs for CIFAR-10 and $150$ epochs for CIFAR-100. We set the initial learning rate as 0.02, and reduce it by a factor of 100 after 40 and 80 epochs for CIFAR-10 and after 80 and 120 epochs for CIFAR-100. We also experiment with cosine annealing learning rate [^26] where the maximum number of epoch for each period is set to $10$, the maximum and minimum learning rate is set to $0.02$ and $0.001$ respectively, total epoch is set to 150.

Clothing-1M: We use a ResNet-50 pretrained on ImageNet same as Refs. [^45] [^47]. The model is trained with batch size 64 and initial learning rate 0.001, which is reduced by $1/100$ after 5 epochs (10 epochs in total). The optimization is done using SGD with a momentum 0.9, and weight decay $0.001$. For each epoch, we sample 2000 mini-batches from the training data ensuring that the classes of the noisy labels are balanced.

WebVision: Following Refs. [^17] [^22], we use an InceptionResNetV2 as the backbone architecture. All other optimization details are the same as for CIFAR-10, except for the weight decay ($0.0005$) and the batch size ($32$).

### G.4 Hyperparameters selection

We perform hyperparameter tuning on the CIFAR datasets via grid search: the temporal ensembling parameter $\beta$ is chosen from $\{0.5,0.7,0.9,0.99\}$ and the regularization coefficient $\lambda$ is chosen from $\{1,3,5,7,10\}$ using the validation set. The selected values are $\beta=0.7$ and $\lambda=3$ for symmetric noise, $\beta=0.9$ and $\lambda=1$ for assymetric noise on CIFAR-10, and $\beta=0.9$ and $\lambda=7$ CIFAR-100. For Clothing1M and WebVision we use the same values as for CIFAR-10. As shown in Section H, the performance of the proposed method seems to be robust to changes in the hyperparameters. For ELR+, we use the same values for $\lambda$ and $\beta$. The mixup $\alpha$ is set to 1 (chosen from $\{0.1,2,5\}$ via grid search on the validation set) and the value of the weight averaging parameter $\gamma$ is set to $0.997$ (which is the default value in the public code of Ref. [^40]) except Clothing1M, which is set to 0.9999.

|  | CIFAR-10 | CIFAR-100 | Clothing-1M | Webvision |
| --- | --- | --- | --- | --- |
| batch size | 128 | 128 | 64 | 32 |
| architecture | PreActResNet-18 | PreActResNet-18 | ResNet-50 (pretrained) | InceptionResNetV2 |
| training epochs | 200 | 250 | 15 | 100 |
| learning rate (lr) | 0.02 | 0.02 | 0.002 | 0.02 |
| lr scheduler | divide 10 at 150th epoch | divide 10 at 200th epoch | divide 10 at 7th epoch | divide 10 at 50th epoch |
| weight decay | 5e-4 | 5e-4 | 1e-3 | 5e-4 |

Table G.2: Training hyperparameters for ELR+ on CIFAR-10, Clothing-1M and Webvision.

## Appendix H Sensitivity to Hyperparameters

The main hyperparameters of ELR are the temporal ensembling parameter $\beta$ and regularization coefficient $\lambda$. As shown in the left image of Figure H.1, performance is robust to the value of $\beta$, although it is worth noting that this is only as long as the momentum of the moving average is large. The performance degrades to 38% when the model outputs are used to estimate the target without averaging (i.e. $\beta=0$). The regularization parameter $\lambda$ needs to be large enough to neutralize the gradients of the falsely labeled examples but also cannot be too large, to avoid neglecting the cross entropy term in the loss. As shown in the center image of Figure H.1, the sensitivity to $\lambda$ is also quite mild. Finally, the right image of Figure H.1 shows results for ELR combined with mixup data augmentation for different values of the mixup parameter $\alpha$. Performance is again quite robust, unless the parameter becomes very large, resulting in a peaked distribution that produces too much mixing.

![Refer to caption](https://ar5iv.labs.arxiv.org/html/2007.00151/assets/images/ablation_agg.png)

Refer to caption

## Appendix I Training Time Analysis

In Table I.1 we compare the training times of ELR and ELR+ with two state-of-the-art methods, using a single Nvidia v100 GPU. ELR+ is twice as slow as ELR. DivideMix takes more than 2 times longer than ELR+ to train. Co-teaching+ is about twice as slow as ELR+.

| Co-teaching+ [^52] | DivideMix [^22] | ELR | ELR+ |
| --- | --- | --- | --- |
| 4.4h | 5.4h | 1.1h | 2.3h |

Table I.1: Comparison of total training time in hours on CIFAR-10 with 40% symmetric label noise.

[^1]: Yacine Aït-Sahalia, Jianqing Fan, and Dacheng Xiu. High-frequency covariance estimates with noisy and asynchronous financial data. Journal of the American Statistical Association, 105(492):1504–1517, 2010.

[^2]: Eric Arazo, Diego Ortego, Paul Albert, Noel E O’Connor, and Kevin McGuinness. Unsupervised label noise modeling and loss correction. In International Conference on Machine Learning (ICML), June 2019.

[^3]: Devansh Arpit, Stanisław Jastrzębski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 233–242. JMLR. org, 2017.

[^4]: David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver, and Colin A Raffel. Mixmatch: A holistic approach to semi-supervised learning. In Advances in Neural Information Processing Systems, pages 5050–5060, 2019.

[^5]: Avrim Blum, Adam Kalai, and Hal Wasserman. Noise-tolerant learning, the parity problem, and the statistical query model. Journal of the ACM (JACM), 50(4):506–519, 2003.

[^6]: Haw-Shiuan Chang, Erik Learned-Miller, and Andrew McCallum. Active bias: Training more accurate neural networks by emphasizing high variance samples. In Advances in Neural Information Processing Systems, pages 1002–1012, 2017.

[^7]: Pengfei Chen, Benben Liao, Guangyong Chen, and Shengyu Zhang. Understanding and utilizing deep neural networks trained with noisy labels. In ICML, 2019.

[^8]: Thomas M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Trans. Electronic Computers, 14(3):326–334, 1965.

[^9]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[^10]: Aritra Ghosh, Himanshu Kumar, and PS Sastry. Robust loss functions under label noise for deep neural networks. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.

[^11]: Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 580–587, 2014.

[^12]: Jacob Goldberger and Ehud Ben-Reuven. Training deep neural-networks using a noise adaptation layer. In ICLR, 2017.

[^13]: Melody Y Guan, Varun Gulshan, Andrew M Dai, and Geoffrey E Hinton. Who said what: Modeling individual labelers improves classification. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[^14]: Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Wai-Hung Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NeurIPS, 2018.

[^15]: Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[^16]: Dan Hendrycks, Mantas Mazeika, Duncan Wilson, and Kevin Gimpel. Using trusted data to train deep networks on labels corrupted by severe noise. In NeurIPS, 2018.

[^17]: Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. In ICML, 2018.

[^18]: Alex Krizhevsky. Learning multiple layers of features from tiny images. University of Toronto, 05 2012.

[^19]: Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.

[^20]: Samuli Laine and Timo Aila. Temporal ensembling for semi-supervised learning. In ICLR, 2018.

[^21]: Michel Ledoux and Michel Talagrand. Probability in Banach Spaces: isoperimetry and processes. Springer Science & Business Media, 2013.

[^22]: Junnan Li, Richard Socher, and Steven C.H. Hoi. Dividemix: Learning with noisy labels as semi-supervised learning. In International Conference on Learning Representations, 2020.

[^23]: Mingchen Li, Mahdi Soltanolkotabi, and Samet Oymak. Gradient descent with early stopping is provably robust to label noise for overparameterized neural networks. arXiv preprint arXiv:1903.11680, 2019.

[^24]: Wen Li, Limin Wang, Wei Li, Eirikur Agustsson, and Luc Van Gool. Webvision database: Visual learning and understanding from web data. arXiv preprint arXiv:1708.02862, 2017.

[^25]: Sheng Liu, Chhavi Yadav, Carlos Fernandez-Granda, and Narges Razavian. On the design of convolutional neural networks for automatic detection of Alzheimer’s disease. In Proceedings of the Machine Learning for Health NeurIPS Workshop, volume 116 of Proceedings of Machine Learning Research (PMLR), pages 184–201. PMLR, 13 Dec 2020.

[^26]: Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In ICLR, 2017.

[^27]: Xingjun Ma, Yisen Wang, Michael E. Houle, Shuo Zhou, Sarah M. Erfani, Shu-Tao Xia, Sudanthi N. R. Wijewickrema, and James Bailey. Dimensionality-driven learning with noisy labels. In ICML, 2018.

[^28]: Eran Malach and Shai Shalev-Shwartz. Decoupling" when to update" from" how to update". In Advances in Neural Information Processing Systems, pages 960–970, 2017.

[^29]: David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In Proceedings of the main conference on human language technology conference of the North American Chapter of the Association of Computational Linguistics, pages 152–159. Association for Computational Linguistics, 2006.

[^30]: Shahar Mendelson. Learning without concentration. In Conference on Learning Theory, pages 25–39, 2014.

[^31]: Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. Making deep neural networks robust to label noise: A loss correction approach. In Proc. IEEE Conf. Comput. Vis. Pattern Recognit.(CVPR), pages 2233–2241, 2017.

[^32]: Mykola Pechenizkiy, Alexey Tsymbal, Seppo Puuronen, and Oleksandr Pechenizkiy. Class noise and supervised learning in medical domains: The effect of feature extraction. In 19th IEEE symposium on computer-based medical systems (CBMS’06), pages 708–713. IEEE, 2006.

[^33]: Scott E. Reed, Honglak Lee, Dragomir Anguelov, Christian Szegedy, Dumitru Erhan, and Andrew Rabinovich. Training deep neural networks on noisy labels with bootstrapping. CoRR, abs/1412.6596, 2015.

[^34]: Mengye Ren, Wenyuan Zeng, Bin Yang, and Raquel Urtasun. Learning to reweight examples for robust deep learning. In ICML, 2018.

[^35]: Hwanjun Song, Minseok Kim, and Jae-Gil Lee. SELFIE: Refurbishing unclean samples for robust deep learning. In ICML, 2019.

[^36]: Hwanjun Song, Minseok Kim, Dongmin Park, and Jae-Gil Lee. Prestopping: How does early stopping help generalization against label noise? arXiv preprint arXiv:1911.08059, 2019.

[^37]: Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research, 19(1):2822–2878, 2018.

[^38]: Daiki Tanaka, Daiki Ikami, Toshihiko Yamasaki, and Kiyoharu Aizawa. Joint optimization framework for learning with noisy labels. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5552–5560, 2018.

[^39]: Ryutaro Tanno, Ardavan Saeedi, Swami Sankaranarayanan, Daniel C Alexander, and Nathan Silberman. Learning from noisy labels by regularized estimation of annotator confusion. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 11244–11253, 2019.

[^40]: Antti Tarvainen and Harri Valpola. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. In Advances in neural information processing systems, pages 1195–1204, 2017.

[^41]: Andreas Veit, Neil Alldrin, Gal Chechik, Ivan Krasin, Abhinav Gupta, and Serge J Belongie. Learning from noisy large-scale datasets with minimal supervision. In CVPR, pages 6575–6583, 2017.

[^42]: Roman Vershynin. Introduction to the non-asymptotic analysis of random matrices. In Compressed sensing, pages 210–268. Cambridge Univ. Press, Cambridge, 2012.

[^43]: Xinshao Wang, Yang Hua, Elyor Kodirov, and Neil M Robertson. Imae for noise-robust learning: Mean absolute error does not treat examples equally and gradient magnitude’s variance matters. arXiv preprint arXiv:1903.12141, 2019.

[^44]: Yisen Wang, Weiyang Liu, Xingjun Ma, James Bailey, Hongyuan Zha, Le Song, and Shu-Tao Xia. Iterative learning with open-set noisy labels. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8688–8696, 2018.

[^45]: Yisen Wang, Xingjun Ma, Zaiyi Chen, Yuan Luo, Jinfeng Yi, and James Bailey. Symmetric cross entropy for robust learning with noisy labels. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 322–330, 2019.

[^46]: Xiaobo Xia, Tongliang Liu, Nannan Wang, Bo Han, Chen Gong, Gang Niu, and Masashi Sugiyama. Are anchor points really indispensable in label-noise learning? In Advances in Neural Information Processing Systems, pages 6835–6846, 2019.

[^47]: Tong Xiao, Tian Xia, Yi Yang, Chang Huang, and Xiaogang Wang. Learning from massive noisy labeled data for image classification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2691–2699, 2015.

[^48]: Yilun Xu, Peng Cao, Yuqing Kong, and Yizhou Wang. LDMI: A novel information-theoretic loss function for training deep nets robust to label noise. In noisy labels, 2019.

[^49]: Yan Yan, Rómer Rosales, Glenn Fung, Ramanathan Subramanian, and Jennifer Dy. Learning from multiple annotators with varying expertise. Machine learning, 95(3):291–327, 2014.

[^50]: David Yarowsky. Unsupervised word sense disambiguation rivaling supervised methods. In 33rd annual meeting of the association for computational linguistics, pages 189–196, 1995.

[^51]: Kun Yi and Jianxin Wu. Probabilistic end-to-end noise correction for learning with noisy labels. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 7017–7025, 2019.

[^52]: Xingrui Yu, Bo Han, Jiangchao Yao, Gang Niu, Ivor Wai-Hung Tsang, and Masashi Sugiyama. How does disagreement help generalization against label corruption? In ICML, 2019.

[^53]: Xiyu Yu, Tongliang Liu, Mingming Gong, and Dacheng Tao. Learning with biased complementary labels. In Proceedings of the European Conference on Computer Vision (ECCV), pages 68–83, 2018.

[^54]: Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In ICLR, 2017.

[^55]: Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. In ICLR, 2018.

[^56]: Zhilu Zhang and Mert R Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. In NeurIPS, 2018.